# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# New zero-intrusion wrapper that keeps the original VGGT code untouched.

from __future__ import annotations

from typing import Sequence

import torch

from mapanything.models.external.vggt.models.vggt import VGGT
from mapanything.models.external.vggt.utils.geometry import closed_form_inverse_se3
from mapanything.models.external.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from mapanything.models.external.vggt.utils.rotation import mat_to_quat
from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    convert_z_depth_to_depth_along_ray,
    depthmap_to_camera_frame,
    get_rays_in_camera_frame,
)

from .heads.efficient_midmatch_head import EfficientMidMatchHead


class VGGTMidMatchWrapper(torch.nn.Module):
    """
    Compatible VGGT wrapper with an extra middle-layer matching branch.

    The returned predictions keep all keys used by the stock MapAnything losses and add:
      - match_desc / match_conf / fine_feat
      - match_patch_size / match_fine_stride / match_coarse_hw
    so existing code stays valid.
    """

    def __init__(
        self,
        name: str = "vggt_midmatch",
        torch_hub_force_reload: bool = False,
        load_pretrained_weights: bool = True,
        depth: int = 24,
        num_heads: int = 16,
        intermediate_layer_idx: Sequence[int] = (4, 11, 17, 23),
        pretrained_model_name_or_path: str = "facebook/VGGT-1B",
        load_custom_ckpt: bool = False,
        custom_ckpt_path: str | None = None,
        match_layer_indices: Sequence[int] = (11, 17),
        match_inner_dim: int = 256,
        match_desc_dim: int = 256,
        match_fine_dim: int = 128,
        match_fine_stride: int = 4,
        match_split_frame_global: bool = True,
        match_use_conf_head: bool = True,
        match_use_fine_refine: bool = True,
        match_return_for_inference: bool = True,
    ) -> None:
        super().__init__()
        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload
        self.load_custom_ckpt = load_custom_ckpt
        self.custom_ckpt_path = custom_ckpt_path
        self.match_return_for_inference = match_return_for_inference

        if load_pretrained_weights:
            if not torch_hub_force_reload:
                print(f"Loading backbone from {pretrained_model_name_or_path} ...")
                self.model = VGGT.from_pretrained(pretrained_model_name_or_path)
            else:
                print(f"Re-downloading backbone from {pretrained_model_name_or_path} ...")
                self.model = VGGT.from_pretrained(
                    pretrained_model_name_or_path,
                    force_download=True,
                )
        else:
            self.model = VGGT(
                depth=depth,
                num_heads=num_heads,
                intermediate_layer_idx=list(intermediate_layer_idx),
            )

        embed_dim = getattr(self.model.aggregator, "embed_dim", 1024)
        patch_size = getattr(self.model.aggregator, "patch_size", 14)
        self.midmatch_head = EfficientMidMatchHead(
            dim_in=2 * embed_dim,
            patch_size=patch_size,
            layer_indices=tuple(match_layer_indices),
            inner_dim=match_inner_dim,
            coarse_desc_dim=match_desc_dim,
            fine_dim=match_fine_dim,
            fine_stride=match_fine_stride,
            split_frame_global=match_split_frame_global,
            use_conf_head=match_use_conf_head,
            use_fine_refine=match_use_fine_refine,
        )

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if cap >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        if self.load_custom_ckpt:
            self._load_custom_ckpt(self.custom_ckpt_path)

    def _load_custom_ckpt(self, ckpt_path: str | None) -> None:
        if ckpt_path is None:
            raise ValueError("custom_ckpt_path must be provided when load_custom_ckpt=True")
        print(f"Loading custom checkpoint from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
        del ckpt

    def _attach_match_outputs(self, pred: dict, match_outputs: dict, view_idx: int) -> None:
        pred["match_desc"] = match_outputs["coarse_desc"][:, view_idx]
        if match_outputs["coarse_conf"] is not None:
            pred["match_conf"] = match_outputs["coarse_conf"][:, view_idx].squeeze(1)
        else:
            pred["match_conf"] = None
        pred["fine_feat"] = (
            None
            if match_outputs["fine_feat"] is None
            else match_outputs["fine_feat"][:, view_idx]
        )
        pred["match_patch_size"] = match_outputs["patch_size"]
        pred["match_fine_stride"] = match_outputs["fine_stride"]
        pred["match_coarse_hw"] = match_outputs["coarse_hw"]
        pred["match_layer_indices"] = match_outputs["layer_indices"]

    def forward(self, views):
        batch_size_per_view, _, height, width = views[0]["img"].shape
        num_views = len(views)
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "VGGT expects a normalized image but without the DINOv2 mean/std applied"
        )

        images = torch.stack([view["img"] for view in views], dim=1)

        use_amp = images.is_cuda
        with torch.autocast("cuda", enabled=use_amp, dtype=self.dtype):
            aggregated_tokens_list, ps_idx = self.model.aggregator(images)

        with torch.autocast("cuda", enabled=False):
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth_map, depth_conf = self.model.depth_head(
                aggregated_tokens_list,
                images,
                ps_idx,
            )
            match_outputs = self.midmatch_head(aggregated_tokens_list, images, ps_idx)

        res = []
        for view_idx in range(num_views):
            curr_view_extrinsic = extrinsic[:, view_idx]
            curr_view_extrinsic = closed_form_inverse_se3(curr_view_extrinsic)
            curr_view_intrinsic = intrinsic[:, view_idx]
            curr_view_depth_z = depth_map[:, view_idx].squeeze(-1)
            curr_view_confidence = depth_conf[:, view_idx]
            curr_view_pts3d_cam, _ = depthmap_to_camera_frame(
                curr_view_depth_z, curr_view_intrinsic
            )
            curr_view_cam_translations = curr_view_extrinsic[..., :3, 3]
            curr_view_cam_quats = mat_to_quat(curr_view_extrinsic[..., :3, :3])
            curr_view_depth_along_ray = convert_z_depth_to_depth_along_ray(
                curr_view_depth_z, curr_view_intrinsic
            ).unsqueeze(-1)
            _, curr_view_ray_dirs = get_rays_in_camera_frame(
                curr_view_intrinsic,
                height,
                width,
                normalize_to_unit_sphere=True,
            )
            curr_view_pts3d = convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                curr_view_ray_dirs,
                curr_view_depth_along_ray,
                curr_view_cam_translations,
                curr_view_cam_quats,
            )
            pred = {
                "pts3d": curr_view_pts3d,
                "pts3d_cam": curr_view_pts3d_cam,
                "ray_directions": curr_view_ray_dirs,
                "intrinsics": curr_view_intrinsic,
                "depth_along_ray": curr_view_depth_along_ray,
                "cam_trans": curr_view_cam_translations,
                "cam_quats": curr_view_cam_quats,
                "conf": curr_view_confidence,
            }
            if self.match_return_for_inference:
                self._attach_match_outputs(pred, match_outputs, view_idx)
            res.append(pred)
        return res
