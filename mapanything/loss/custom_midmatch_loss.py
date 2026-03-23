from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapanything.models.external.vggt_midmatch.utils.matching import local_window_refine
from mapanything.utils.geometry import (
    closed_form_pose_inverse,
    recover_pinhole_intrinsics_from_ray_directions,
)


@dataclass
class PairStats:
    coarse_loss: torch.Tensor
    fine_loss: torch.Tensor
    conf_loss: torch.Tensor
    num_matches: int


class VGGTMidMatchCriterion(nn.Module):
    """
    Plug-in criterion:
      total = base_geometry_loss + match_weight * coarse_match_loss
              + fine_weight * fine_refine_loss + conf_weight * match_conf_loss

    Ground-truth correspondences are generated online from GT world points + camera poses,
    so this works with the existing MapAnything training batches and requires no extra labels.
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        match_weight: float = 0.2,
        fine_weight: float = 0.0,
        conf_weight: float = 0.02,
        temperature: float = 0.07,
        num_samples_per_pair: int = 512,
        pair_mode: str = "view0_to_others",
        occlusion_abs_thresh: float = 0.05,
        occlusion_rel_thresh: float = 0.05,
        fine_window_radius: int = 2,
    ) -> None:
        super().__init__()
        self.base_criterion = base_criterion
        self.match_weight = match_weight
        self.fine_weight = fine_weight
        self.conf_weight = conf_weight
        self.temperature = temperature
        self.num_samples_per_pair = num_samples_per_pair
        self.pair_mode = pair_mode
        self.occlusion_abs_thresh = occlusion_abs_thresh
        self.occlusion_rel_thresh = occlusion_rel_thresh
        self.fine_window_radius = fine_window_radius

    def _build_pairs(self, n_views: int) -> List[Tuple[int, int]]:
        if n_views < 2:
            return []
        if self.pair_mode == "view0_to_others":
            return [(0, j) for j in range(1, n_views)]
        if self.pair_mode == "bidirectional_view0":
            pairs = []
            for j in range(1, n_views):
                pairs.extend([(0, j), (j, 0)])
            return pairs
        if self.pair_mode == "all_pairs":
            return [(i, j) for i in range(n_views) for j in range(n_views) if i != j]
        raise ValueError(f"Unsupported pair_mode: {self.pair_mode}")

    @staticmethod
    def _camera_pose(view: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "camera_pose" not in view:
            raise KeyError("Batch views must contain 'camera_pose' for match supervision")
        return view["camera_pose"]

    @staticmethod
    def _center_grid_indices(height: int, width: int, patch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        ys = torch.arange(patch_size // 2, height, patch_size, device=device)
        xs = torch.arange(patch_size // 2, width, patch_size, device=device)
        return torch.meshgrid(ys, xs, indexing="ij")

    def _sample_source_points(
        self,
        view: Dict[str, torch.Tensor],
        patch_size: int,
        batch_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pts3d = view["pts3d"][batch_index]           # [H,W,3]
        valid = view["valid_mask"][batch_index]      # [H,W]
        height, width = valid.shape
        yy, xx = self._center_grid_indices(height, width, patch_size, valid.device)
        yy_flat = yy.flatten().long().clamp(0, height - 1)
        xx_flat = xx.flatten().long().clamp(0, width - 1)
        keep = valid[yy_flat, xx_flat]
        if keep.sum() == 0:
            empty = torch.empty(0, device=valid.device, dtype=torch.long)
            return empty, empty, torch.empty(0, 3, device=valid.device, dtype=pts3d.dtype)

        yy_keep = yy_flat[keep]
        xx_keep = xx_flat[keep]
        if yy_keep.numel() > self.num_samples_per_pair:
            perm = torch.randperm(yy_keep.numel(), device=yy_keep.device)[: self.num_samples_per_pair]
            yy_keep = yy_keep[perm]
            xx_keep = xx_keep[perm]
        world_pts = pts3d[yy_keep, xx_keep]
        return yy_keep, xx_keep, world_pts

    def _project_world_to_view(
        self,
        pts_world: torch.Tensor,
        target_view: Dict[str, torch.Tensor],
        batch_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pts_world: [N,3]
        cam_pose = self._camera_pose(target_view)[batch_index : batch_index + 1]  # [1,4,4]
        world_to_cam = closed_form_pose_inverse(cam_pose)[0]
        rot = world_to_cam[:3, :3]
        trans = world_to_cam[:3, 3]
        pts_cam = pts_world @ rot.T + trans
        intr = recover_pinhole_intrinsics_from_ray_directions(
            target_view["ray_directions_cam"][batch_index]
        )
        uv_h = pts_cam @ intr[:3, :3].T
        uv = uv_h[:, :2] / uv_h[:, 2:3].clamp(min=1e-6)
        return uv, pts_cam[:, 2]

    def _filter_visible_targets(
        self,
        uv: torch.Tensor,
        z_cam: torch.Tensor,
        target_view: Dict[str, torch.Tensor],
        batch_index: int,
        patch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = target_view["valid_mask"].shape[-2:]
        in_img = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] <= width - 1)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] <= height - 1)
            & (z_cam > 1e-6)
        )
        if in_img.sum() == 0:
            empty = torch.empty(0, device=uv.device, dtype=torch.long)
            return empty, empty, torch.empty(0, 2, device=uv.device, dtype=uv.dtype)

        uv = uv[in_img]
        z_cam = z_cam[in_img]
        x_round = uv[:, 0].round().long().clamp(0, width - 1)
        y_round = uv[:, 1].round().long().clamp(0, height - 1)

        tgt_valid = target_view["valid_mask"][batch_index, y_round, x_round]
        tgt_z = target_view["pts3d_cam"][batch_index, y_round, x_round, 2]
        z_tol = torch.maximum(
            torch.full_like(tgt_z, self.occlusion_abs_thresh),
            self.occlusion_rel_thresh * tgt_z.abs(),
        )
        visible = tgt_valid & ((tgt_z - z_cam).abs() <= z_tol)
        if visible.sum() == 0:
            empty = torch.empty(0, device=uv.device, dtype=torch.long)
            return empty, empty, torch.empty(0, 2, device=uv.device, dtype=uv.dtype)

        uv_vis = uv[visible]
        coarse_x = torch.floor(uv_vis[:, 0] / patch_size).long()
        coarse_y = torch.floor(uv_vis[:, 1] / patch_size).long()
        coarse_w = width // patch_size
        coarse_h = height // patch_size
        coarse_ok = (
            (coarse_x >= 0)
            & (coarse_x < coarse_w)
            & (coarse_y >= 0)
            & (coarse_y < coarse_h)
        )
        if coarse_ok.sum() == 0:
            empty = torch.empty(0, device=uv.device, dtype=torch.long)
            return empty, empty, torch.empty(0, 2, device=uv.device, dtype=uv.dtype)

        return coarse_y[coarse_ok], coarse_x[coarse_ok], uv_vis[coarse_ok]

    def _pair_loss(
        self,
        src_view: Dict[str, torch.Tensor],
        tgt_view: Dict[str, torch.Tensor],
        src_pred: Dict[str, torch.Tensor],
        tgt_pred: Dict[str, torch.Tensor],
        batch_index: int,
    ) -> PairStats:
        if src_pred.get("match_desc", None) is None or tgt_pred.get("match_desc", None) is None:
            zero = torch.zeros((), device=src_view["img"].device)
            return PairStats(zero, zero, zero, 0)

        patch_size = int(src_pred["match_patch_size"])
        coarse_desc_src = src_pred["match_desc"][batch_index]     # [C,Hc,Wc]
        coarse_desc_tgt = tgt_pred["match_desc"][batch_index]
        coarse_conf_src = src_pred.get("match_conf", None)
        coarse_conf_tgt = tgt_pred.get("match_conf", None)
        if coarse_conf_src is not None:
            coarse_conf_src = coarse_conf_src[batch_index]
        if coarse_conf_tgt is not None:
            coarse_conf_tgt = coarse_conf_tgt[batch_index]

        src_y, src_x, world_pts = self._sample_source_points(src_view, patch_size, batch_index)
        if world_pts.numel() == 0:
            zero = torch.zeros((), device=src_view["img"].device)
            return PairStats(zero, zero, zero, 0)

        uv_tgt, z_tgt = self._project_world_to_view(world_pts, tgt_view, batch_index)
        tgt_y, tgt_x, uv_tgt = self._filter_visible_targets(
            uv_tgt, z_tgt, tgt_view, batch_index, patch_size
        )
        if tgt_y.numel() == 0:
            zero = torch.zeros((), device=src_view["img"].device)
            return PairStats(zero, zero, zero, 0)

        # keep the same prefix length after filtering
        num = tgt_y.numel()
        src_y = src_y[:num]
        src_x = src_x[:num]

        # source coarse coords from source pixel centers
        src_cy = torch.floor(src_y.float() / patch_size).long()
        src_cx = torch.floor(src_x.float() / patch_size).long()
        src_feat = coarse_desc_src[:, src_cy, src_cx].transpose(0, 1)                # [N,C]
        tgt_feat_map = coarse_desc_tgt.view(coarse_desc_tgt.shape[0], -1)             # [C,HW]
        logits = torch.matmul(src_feat, tgt_feat_map) / self.temperature              # [N,HW]

        target_idx = tgt_y * coarse_desc_tgt.shape[-1] + tgt_x
        coarse_loss = F.cross_entropy(logits, target_idx, reduction="mean")

        conf_loss = torch.zeros_like(coarse_loss)
        if coarse_conf_src is not None and coarse_conf_tgt is not None:
            src_conf_pos = coarse_conf_src[src_cy, src_cx]
            tgt_conf_pos = coarse_conf_tgt[tgt_y, tgt_x]
            conf_loss = 0.5 * (
                F.binary_cross_entropy(src_conf_pos, torch.ones_like(src_conf_pos))
                + F.binary_cross_entropy(tgt_conf_pos, torch.ones_like(tgt_conf_pos))
            )

        fine_loss = torch.zeros_like(coarse_loss)
        fine_feat_src = src_pred.get("fine_feat", None)
        fine_feat_tgt = tgt_pred.get("fine_feat", None)
        if self.fine_weight > 0 and fine_feat_src is not None and fine_feat_tgt is not None:
            fine_feat_src = fine_feat_src[batch_index]   # [C,Hf,Wf]
            fine_feat_tgt = fine_feat_tgt[batch_index]
            fine_stride = int(src_pred["match_fine_stride"])
            src_fx = torch.clamp((src_x.float() / fine_stride).round().long(), 0, fine_feat_src.shape[-1] - 1)
            src_fy = torch.clamp((src_y.float() / fine_stride).round().long(), 0, fine_feat_src.shape[-2] - 1)
            q_feat = fine_feat_src[:, src_fy, src_fx].transpose(0, 1).contiguous()   # [N,C]
            tgt_center = torch.stack(
                [
                    torch.clamp((uv_tgt[:, 0] / fine_stride).round().long(), 0, fine_feat_tgt.shape[-1] - 1),
                    torch.clamp((uv_tgt[:, 1] / fine_stride).round().long(), 0, fine_feat_tgt.shape[-2] - 1),
                ],
                dim=-1,
            )
            refined_xy, _ = local_window_refine(
                q_feat,
                fine_feat_tgt.unsqueeze(0).expand(q_feat.shape[0], -1, -1, -1),
                center_xy=tgt_center,
                window_radius=self.fine_window_radius,
            )
            gt_xy_fine = uv_tgt / fine_stride
            fine_loss = F.l1_loss(refined_xy, gt_xy_fine, reduction="mean")

        return PairStats(coarse_loss, fine_loss, conf_loss, num)

    def forward(self, batch, preds):
        base_loss, base_details = self.base_criterion(batch, preds)

        pairs = self._build_pairs(len(batch))
        if len(pairs) == 0:
            return base_loss, base_details

        coarse_terms = []
        fine_terms = []
        conf_terms = []
        num_matches = 0
        batch_size = batch[0]["img"].shape[0]
        for b in range(batch_size):
            for src_idx, tgt_idx in pairs:
                stats = self._pair_loss(
                    batch[src_idx],
                    batch[tgt_idx],
                    preds[src_idx],
                    preds[tgt_idx],
                    b,
                )
                if stats.num_matches > 0:
                    coarse_terms.append(stats.coarse_loss)
                    fine_terms.append(stats.fine_loss)
                    conf_terms.append(stats.conf_loss)
                    num_matches += stats.num_matches

        if len(coarse_terms) == 0:
            zero = base_loss.new_zeros(())
            details = dict(base_details)
            details.update(
                {
                    "midmatch_pairs_with_supervision": 0.0,
                    "midmatch_num_matches": 0.0,
                }
            )
            return base_loss, details

        coarse_loss = torch.stack(coarse_terms).mean()
        fine_loss = torch.stack(fine_terms).mean() if len(fine_terms) > 0 else base_loss.new_zeros(())
        conf_loss = torch.stack(conf_terms).mean() if len(conf_terms) > 0 else base_loss.new_zeros(())
        total = base_loss + self.match_weight * coarse_loss
        total = total + self.fine_weight * fine_loss + self.conf_weight * conf_loss

        details = dict(base_details)
        details.update(
            {
                "midmatch_coarse": float(coarse_loss.detach()),
                "midmatch_fine": float(fine_loss.detach()),
                "midmatch_conf": float(conf_loss.detach()),
                "midmatch_num_matches": float(num_matches),
                "midmatch_pairs_with_supervision": float(len(coarse_terms)),
                "midmatch_weight": float(self.match_weight),
                "midmatch_fine_weight": float(self.fine_weight),
                "midmatch_conf_weight": float(self.conf_weight),
            }
        )
        return total, details
