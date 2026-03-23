import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvRefine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class EfficientMidMatchHead(nn.Module):
    """
    Efficient LoFTR-inspired middle-layer matching head for VGGT.

    Design goals:
      1) read correspondence-aware intermediate VGGT tokens;
      2) produce coarse dense descriptors/confidence on patch grid;
      3) optionally expose a lightweight CNN fine feature pyramid for local refinement.

    This head is intentionally self-contained so it can be attached from a wrapper
    without modifying the original VGGT backbone implementation.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 14,
        layer_indices: Sequence[int] = (11, 17),
        inner_dim: int = 256,
        coarse_desc_dim: int = 256,
        fine_dim: int = 128,
        fine_stride: int = 4,
        split_frame_global: bool = True,
        use_conf_head: bool = True,
        use_fine_refine: bool = True,
    ) -> None:
        super().__init__()
        if len(layer_indices) == 0:
            raise ValueError("layer_indices must not be empty")
        if fine_stride not in (2, 4, 8):
            raise ValueError("fine_stride must be one of {2,4,8}")

        self.dim_in = dim_in
        self.patch_size = patch_size
        self.layer_indices = tuple(layer_indices)
        self.inner_dim = inner_dim
        self.coarse_desc_dim = coarse_desc_dim
        self.fine_dim = fine_dim
        self.fine_stride = fine_stride
        self.split_frame_global = split_frame_global
        self.use_conf_head = use_conf_head
        self.use_fine_refine = use_fine_refine

        if split_frame_global:
            if dim_in % 2 != 0:
                raise ValueError("dim_in must be even when split_frame_global=True")
            half_dim = dim_in // 2
            self.norm_frame = nn.ModuleList(
                [nn.LayerNorm(half_dim) for _ in self.layer_indices]
            )
            self.norm_global = nn.ModuleList(
                [nn.LayerNorm(half_dim) for _ in self.layer_indices]
            )
            self.proj_frame = nn.ModuleList(
                [nn.Linear(half_dim, inner_dim) for _ in self.layer_indices]
            )
            self.proj_global = nn.ModuleList(
                [nn.Linear(half_dim, inner_dim) for _ in self.layer_indices]
            )
            self.layer_gate_logits = nn.Parameter(
                torch.zeros(len(self.layer_indices), 1, 1, 1, inner_dim)
            )
        else:
            self.norm = nn.ModuleList([nn.LayerNorm(dim_in) for _ in self.layer_indices])
            self.proj = nn.ModuleList(
                [nn.Linear(dim_in, inner_dim) for _ in self.layer_indices]
            )

        self.layer_mix_logits = nn.Parameter(torch.zeros(len(self.layer_indices)))
        self.coarse_refine = _ConvRefine(inner_dim)
        self.coarse_desc_head = nn.Conv2d(inner_dim, coarse_desc_dim, kernel_size=1)
        self.coarse_conf_head = (
            nn.Conv2d(inner_dim, 1, kernel_size=1) if use_conf_head else None
        )

        if use_fine_refine:
            fine_layers = [
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
            ]
            curr_stride = 2
            in_ch = 32
            while curr_stride < fine_stride:
                out_ch = min(128, in_ch * 2)
                fine_layers.extend(
                    [
                        nn.Conv2d(
                            in_ch,
                            out_ch,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(out_ch),
                        nn.GELU(),
                    ]
                )
                in_ch = out_ch
                curr_stride *= 2
            fine_layers.extend(
                [
                    nn.Conv2d(in_ch, fine_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(fine_dim),
                    nn.GELU(),
                    _ConvRefine(fine_dim),
                ]
            )
            self.fine_encoder = nn.Sequential(*fine_layers)
        else:
            self.fine_encoder = None

    def _token_to_map(
        self,
        x: torch.Tensor,
        batch_size: int,
        num_views: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        h = height // self.patch_size
        w = width // self.patch_size
        x = x.view(batch_size, num_views, h, w, -1).permute(0, 1, 4, 2, 3).contiguous()
        return x

    def _fuse_mid_layers(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        patch_start_idx: int,
        batch_size: int,
        num_views: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        layer_feats = []
        for i, layer_idx in enumerate(self.layer_indices):
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            if self.split_frame_global:
                x_frame, x_global = x.chunk(2, dim=-1)
                x_frame = self.proj_frame[i](self.norm_frame[i](x_frame))
                x_global = self.proj_global[i](self.norm_global[i](x_global))
                gate = torch.sigmoid(self.layer_gate_logits[i])
                x = gate * x_global + (1.0 - gate) * x_frame
            else:
                x = self.proj[i](self.norm[i](x))
            x = self._token_to_map(x, batch_size, num_views, height, width)
            layer_feats.append(x)

        layer_mix = torch.softmax(self.layer_mix_logits, dim=0)
        fused = 0.0
        for weight, feat in zip(layer_mix, layer_feats):
            fused = fused + weight * feat
        return fused

    def forward(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
    ) -> dict:
        if images.ndim != 5:
            raise ValueError(f"Expected images [B,S,3,H,W], got {tuple(images.shape)}")
        batch_size, num_views, _, height, width = images.shape
        coarse_h = height // self.patch_size
        coarse_w = width // self.patch_size

        fused = self._fuse_mid_layers(
            aggregated_tokens_list,
            patch_start_idx,
            batch_size,
            num_views,
            height,
            width,
        )
        fused = fused.view(batch_size * num_views, self.inner_dim, coarse_h, coarse_w)
        fused = self.coarse_refine(fused)

        coarse_desc = self.coarse_desc_head(fused)
        coarse_desc = F.normalize(coarse_desc, dim=1)
        coarse_desc = coarse_desc.view(
            batch_size, num_views, self.coarse_desc_dim, coarse_h, coarse_w
        )

        coarse_conf = None
        if self.coarse_conf_head is not None:
            coarse_conf = torch.sigmoid(self.coarse_conf_head(fused))
            coarse_conf = coarse_conf.view(batch_size, num_views, 1, coarse_h, coarse_w)

        fine_feat = None
        if self.fine_encoder is not None:
            fine_images = images.view(batch_size * num_views, 3, height, width)
            fine_feat = self.fine_encoder(fine_images)
            fine_feat = F.normalize(fine_feat, dim=1)
            fine_feat = fine_feat.view(
                batch_size,
                num_views,
                self.fine_dim,
                fine_feat.shape[-2],
                fine_feat.shape[-1],
            )

        return {
            "coarse_desc": coarse_desc,
            "coarse_conf": coarse_conf,
            "fine_feat": fine_feat,
            "patch_size": self.patch_size,
            "fine_stride": self.fine_stride,
            "coarse_hw": (coarse_h, coarse_w),
            "layer_indices": list(self.layer_indices),
        }
