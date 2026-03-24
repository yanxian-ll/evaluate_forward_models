from __future__ import annotations

from typing import Optional

import torch

from .losses import MultiLoss


class DA3GlobalCameraFocalLoss(MultiLoss):
    """
    Global focal loss for the DA3-style camera head.

    Priority for target focal retrieval:
      1) use explicit GT intrinsics if they exist in the batch;
      2) otherwise, fit a centered pinhole focal from GT ray directions.
    """

    def __init__(
        self,
        pred_key: str = "camera_focal_px",
        gt_intrinsics_key: str = "camera_intrinsics",
        gt_rays_key: str = "ray_directions_cam",
        valid_mask_key: str = "valid_mask",
        use_log: bool = True,
        isotropic: bool = False,
        eps: float = 1e-6,
        min_abs_ratio: float = 1e-6,
    ):
        super().__init__()
        self.pred_key = pred_key
        self.gt_intrinsics_key = gt_intrinsics_key
        self.gt_rays_key = gt_rays_key
        self.valid_mask_key = valid_mask_key
        self.use_log = use_log
        self.isotropic = isotropic
        self.eps = eps
        self.min_abs_ratio = min_abs_ratio

    def get_name(self):
        return "da3_global_camera_focal_loss"

    def _estimate_focal_from_rays(self, ray_dirs: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        bsz, height, width, _ = ray_dirs.shape
        device = ray_dirs.device
        dtype = ray_dirs.dtype

        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        cx = (width - 1.0) * 0.5
        cy = (height - 1.0) * 0.5
        du = xs - cx
        dv = ys - cy

        gt_focals = []
        for i in range(bsz):
            rays = ray_dirs[i]
            z = rays[..., 2]
            x_over_z = rays[..., 0] / z.clamp(min=self.eps)
            y_over_z = rays[..., 1] / z.clamp(min=self.eps)

            valid_x = torch.isfinite(x_over_z) & torch.isfinite(du)
            valid_y = torch.isfinite(y_over_z) & torch.isfinite(dv)
            valid_x &= (du.abs() > 0) & (x_over_z.abs() > self.min_abs_ratio)
            valid_y &= (dv.abs() > 0) & (y_over_z.abs() > self.min_abs_ratio)
            valid_x &= z > self.eps
            valid_y &= z > self.eps
            if valid_mask is not None:
                valid_x &= valid_mask[i]
                valid_y &= valid_mask[i]

            if valid_x.any():
                fx = torch.median((du[valid_x] / x_over_z[valid_x]).abs())
            else:
                fx = torch.tensor(float("nan"), device=device, dtype=dtype)
            if valid_y.any():
                fy = torch.median((dv[valid_y] / y_over_z[valid_y]).abs())
            else:
                fy = torch.tensor(float("nan"), device=device, dtype=dtype)
            gt_focals.append(torch.stack([fx, fy], dim=0))

        return torch.stack(gt_focals, dim=0)

    def _get_gt_focal_px(self, batch_view: dict):
        if self.gt_intrinsics_key in batch_view:
            intrinsics = batch_view[self.gt_intrinsics_key]
            return torch.stack([intrinsics[:, 0, 0], intrinsics[:, 1, 1]], dim=-1)

        if self.gt_rays_key not in batch_view:
            return None

        valid_mask = batch_view.get(self.valid_mask_key, None)
        return self._estimate_focal_from_rays(batch_view[self.gt_rays_key], valid_mask)

    def compute_loss(self, batch, preds):
        per_view_losses = []

        for view_idx in range(len(batch)):
            if self.pred_key not in preds[view_idx]:
                continue

            pred_focal = preds[view_idx][self.pred_key]
            gt_focal = self._get_gt_focal_px(batch[view_idx])
            if gt_focal is None:
                continue

            valid = torch.isfinite(pred_focal).all(dim=-1)
            valid &= torch.isfinite(gt_focal).all(dim=-1)
            valid &= (pred_focal > self.eps).all(dim=-1)
            valid &= (gt_focal > self.eps).all(dim=-1)
            if not valid.any():
                continue

            pred_focal = pred_focal[valid]
            gt_focal = gt_focal[valid]
            if self.isotropic:
                pred_focal = pred_focal.mean(dim=-1, keepdim=True)
                gt_focal = gt_focal.mean(dim=-1, keepdim=True)

            if self.use_log:
                err = torch.abs(
                    torch.log(pred_focal.clamp_min(self.eps))
                    - torch.log(gt_focal.clamp_min(self.eps))
                )
            else:
                err = torch.abs(pred_focal - gt_focal) / gt_focal.clamp_min(self.eps)
            per_view_losses.append(err.mean())

        if len(per_view_losses) == 0:
            ref = None
            for pred in preds:
                if self.pred_key in pred:
                    ref = pred[self.pred_key]
                    break
            if ref is None:
                return torch.tensor(0.0)
            return ref.sum() * 0.0

        return torch.stack(per_view_losses).mean()
