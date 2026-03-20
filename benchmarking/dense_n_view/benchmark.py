# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Benchmark dense multi-view metric reconstruction.

Key changes compared with the earlier version:
1. Estimate ONE robust global Sim(3) per multi-view set in `get_all_info_for_metric_computation`.
   The Sim(3) is solved from dense point correspondences sampled from all views plus camera-center
   correspondences, and is used to align predicted world geometry / poses to GT.
2. System-level absolute metrics (pose absolute, merged point cloud absolute, global pointmap absolute)
   are computed after this single global alignment.
3. Per-view depth absolute metrics are evaluated with sequence-level scale-only alignment, which better
   reflects single-frame depth quality without introducing an extra per-view affine or per-view Sim(3).
4. Removed the old fragmented logic where point/depth/fused-PC used different alignments.
"""

import json
import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Set the cache directory for torch hub
torch.hub.set_dir("/opt/data/private/code/map-anything/checkpoints/torch_cache/hub")

# load local dino repo
LOCAL_DINO_REPO = "/opt/data/private/code/map-anything/checkpoints/torch_cache/hub/facebookresearch_dinov2_main"
_original_torch_hub_load = torch.hub.load


def offline_torch_hub_load(repo_or_dir, model, *args, **kwargs):
    if repo_or_dir == "facebookresearch/dinov2":
        print("Redirecting DINOv2 torch.hub.load to local repo")
        repo_or_dir = LOCAL_DINO_REPO
        kwargs["source"] = "local"
    return _original_torch_hub_load(repo_or_dir, model, *args, **kwargs)


torch.hub.load = offline_torch_hub_load

from mapanything.datasets import get_test_data_loader
from mapanything.models import init_model
from mapanything.utils.geometry import (
    geotrf,
    inv,
    normalize_multiple_pointclouds,
    quaternion_to_rotation_matrix,
    transform_pose_using_quats_and_trans_2_to_1,
    recover_pinhole_intrinsics_from_ray_directions
)
from mapanything.utils.metrics import (
    compute_set_metrics,
    sim3_from_correspondences_robust,
)
from mapanything.utils.misc import StreamToLogger

log = logging.getLogger(__name__)


# ============================================================
# IO helpers
# ============================================================

def write_ply_xyzrgb(path, points, colors_u8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors_u8 = np.asarray(colors_u8, dtype=np.uint8)

    assert points.shape[0] == colors_u8.shape[0]
    n = points.shape[0]

    vertex = np.empty(
        n,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors_u8[:, 0]
    vertex["green"] = colors_u8[:, 1]
    vertex["blue"] = colors_u8[:, 2]

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header\n",
        ]
    ).encode("ascii")

    with open(path, "wb") as f:
        f.write(header)
        vertex.tofile(f)


# ============================================================
# Serialization helpers
# ============================================================

def _to_python(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _batched_get(x, batch_idx):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x[batch_idx]
    if isinstance(x, (list, tuple)):
        return x[batch_idx]
    return x


def save_repro_bundle_json(
    json_path,
    benchmark_dataset_name,
    scene,
    set_idx,
    batch_views,
    batch_idx,
    gt_info_abs,
    pr_info_abs_aligned,
    scale_factors,
    metrics,
    gt_ply_path,
    pred_ply_path,
):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    n_views = len(batch_views)
    views = []

    for i in range(n_views):
        instance = _batched_get(batch_views[i].get("instance", None), batch_idx)
        instance = str(instance) if instance is not None else None

        gt_intr = _batched_get(gt_info_abs["intrinsics"][i], batch_idx) if "intrinsics" in gt_info_abs else None
        gt_pose = _batched_get(gt_info_abs["poses"][i], batch_idx) if "poses" in gt_info_abs else None

        pred_intr = _batched_get(pr_info_abs_aligned["intrinsics"][i], batch_idx) if "intrinsics" in pr_info_abs_aligned else None
        pred_pose = _batched_get(pr_info_abs_aligned["poses"][i], batch_idx) if "poses" in pr_info_abs_aligned else None

        views.append(
            {
                "view_idx": i,
                "instance": instance,
                "gt_cam": {
                    "intrinsics": _to_python(gt_intr),
                    "c2w": _to_python(gt_pose),
                },
                "pred_cam": {
                    "intrinsics": _to_python(pred_intr),
                    "c2w": _to_python(pred_pose),
                },
            }
        )

    scale_meta = {}
    if scale_factors is not None:
        for key, value in scale_factors.items():
            try:
                scale_meta[key] = _to_python(_batched_get(value, batch_idx))
            except Exception:
                scale_meta[key] = _to_python(value)

    payload = {
        "benchmark_dataset": benchmark_dataset_name,
        "scene": str(scene),
        "set_index": int(set_idx),
        "instances": [v["instance"] for v in views],
        "views": views,
        "metrics": {
            k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else _to_python(v)
            for k, v in metrics.items()
        },
        "scale_factors": scale_meta,
        "fused_outputs": {
            "gt_ply": os.path.abspath(gt_ply_path) if gt_ply_path else None,
            "pred_ply": os.path.abspath(pred_ply_path) if pred_ply_path else None,
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ============================================================
# Sim(3) solving helpers
# ============================================================
@torch.no_grad()
def _robust_high_quantile(values: torch.Tensor, q: float = 0.995):
    if values.numel() == 0:
        return None
    q = min(max(float(q), 0.5), 1.0)
    try:
        return torch.quantile(values, q)
    except Exception:
        k = max(int(round((values.numel() - 1) * q)), 0)
        return values.flatten().kthvalue(k + 1).values

@torch.no_grad()
def _robust_low_quantile(values: torch.Tensor, q: float = 0.005):
    if values.numel() == 0:
        return None
    q = min(max(float(q), 0.0), 0.5)
    try:
        return torch.quantile(values, q)
    except Exception:
        k = max(int(round((values.numel() - 1) * q)), 0)
        return values.flatten().kthvalue(k + 1).values


@torch.no_grad()
def _local_median_mad_mask_2d(
    x2d: torch.Tensor,
    valid2d: torch.Tensor,
    kernel_size: int = 3,
    mad_scale: float = 6.0,
    min_neighbors: int = 4,
):
    """
    对单张 2D 图做局部 median + MAD 过滤。
    x2d: (H,W)
    valid2d: (H,W) bool
    return: refined_valid2d
    """
    assert x2d.ndim == 2 and valid2d.ndim == 2
    H, W = x2d.shape
    pad = kernel_size // 2

    x = x2d.clone()
    v = valid2d.clone().bool()

    # unfold
    x_pad = torch.nn.functional.pad(x[None, None], (pad, pad, pad, pad), mode="replicate")
    v_pad = torch.nn.functional.pad(v.float()[None, None], (pad, pad, pad, pad), mode="replicate")

    x_unf = torch.nn.functional.unfold(x_pad, kernel_size=kernel_size)   # (1, K, H*W)
    v_unf = torch.nn.functional.unfold(v_pad, kernel_size=kernel_size)   # (1, K, H*W)
    K = x_unf.shape[1]

    x_unf = x_unf.view(K, H * W).t()   # (H*W, K)
    v_unf = v_unf.view(K, H * W).t() > 0.5

    center_valid = v.reshape(-1)
    x_center = x.reshape(-1)

    inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    x_masked = torch.where(v_unf, x_unf, inf)

    x_sorted, _ = torch.sort(x_masked, dim=1)
    n_valid = v_unf.sum(dim=1)

    med = torch.empty((H * W,), device=x.device, dtype=x.dtype)
    med.fill_(float("nan"))

    good = n_valid >= min_neighbors
    if good.any():
        mid_idx = torch.clamp((n_valid[good] - 1) // 2, min=0)
        med_good = x_sorted[good, :]
        med[good] = med_good.gather(1, mid_idx[:, None]).squeeze(1)

    # MAD
    abs_dev = torch.abs(x_unf - med[:, None])
    abs_dev = torch.where(v_unf & torch.isfinite(med[:, None]), abs_dev, inf)

    abs_dev_sorted, _ = torch.sort(abs_dev, dim=1)
    mad = torch.empty((H * W,), device=x.device, dtype=x.dtype)
    mad.fill_(float("nan"))
    if good.any():
        mad_good = abs_dev_sorted[good, :]
        mad[good] = mad_good.gather(1, mid_idx[:, None]).squeeze(1)

    thr = mad_scale * 1.4826 * torch.clamp(mad, min=1e-6)
    ok = center_valid & good & torch.isfinite(med) & torch.isfinite(thr) & (torch.abs(x_center - med) <= thr)

    return ok.view(H, W)

@torch.no_grad()
def _mask_outliers_per_batch(
    pts_world: torch.Tensor,
    pts_cam: torch.Tensor,
    base_mask: torch.Tensor,
    world_low_q: float = 0.002,
    world_high_q: float = 0.995,
    depth_low_q: float = 0.01,
    depth_high_q: float = 0.995,
    min_depth: float = 1e-6,
    local_kernel: int = 3,
    local_mad_scale_depth: float = 5.0,
    local_mad_scale_range: float = 5.0,
    global_mad_scale_depth: float = 6.0,
    global_mad_scale_range: float = 6.0,
):
    """
    更稳健的离群点过滤：
    1) 去掉非有限值 / 非正深度
    2) 深度和空间距离做双边分位裁剪（尤其裁掉异常近点）
    3) 对 z-depth 和 point-range 做局部 median+MAD 过滤
    4) 再做一次全局 MAD 过滤
    """
    mask = base_mask.clone().bool()
    B = pts_world.shape[0]

    for b in range(B):
        m = mask[b]
        if m.sum() == 0:
            continue

        pw = pts_world[b]           # (H,W,3)
        pc = pts_cam[b]             # (H,W,3)

        finite = torch.isfinite(pw).all(dim=-1) & torch.isfinite(pc).all(dim=-1)
        z = pc[..., 2]
        r_world = torch.linalg.norm(pw, dim=-1)

        m = m & finite & torch.isfinite(z) & torch.isfinite(r_world) & (z > min_depth)
        if m.sum() < 16:
            mask[b] = m
            continue

        # ---------- 1) 双边分位裁剪 ----------
        z_valid = z[m]
        r_valid = r_world[m]

        z_lo = _robust_low_quantile(z_valid, q=depth_low_q)
        z_hi = _robust_high_quantile(z_valid, q=depth_high_q)
        r_lo = _robust_low_quantile(r_valid, q=world_low_q)
        r_hi = _robust_high_quantile(r_valid, q=world_high_q)

        if z_lo is not None and torch.isfinite(z_lo):
            m = m & (z >= z_lo)
        if z_hi is not None and torch.isfinite(z_hi):
            m = m & (z <= z_hi)
        if r_lo is not None and torch.isfinite(r_lo):
            m = m & (r_world >= r_lo)
        if r_hi is not None and torch.isfinite(r_hi):
            m = m & (r_world <= r_hi)

        if m.sum() < 16:
            mask[b] = m
            continue

        # ---------- 2) 局部一致性过滤 ----------
        # 对 log-depth / log-range 做局部检测更稳
        log_z = torch.zeros_like(z)
        log_r = torch.zeros_like(r_world)
        log_z[m] = torch.log(torch.clamp(z[m], min=1e-6))
        log_r[m] = torch.log(torch.clamp(r_world[m], min=1e-6))

        m_local_z = _local_median_mad_mask_2d(
            log_z, m, kernel_size=local_kernel, mad_scale=local_mad_scale_depth, min_neighbors=4
        )
        m_local_r = _local_median_mad_mask_2d(
            log_r, m, kernel_size=local_kernel, mad_scale=local_mad_scale_range, min_neighbors=4
        )

        m = m & m_local_z & m_local_r
        if m.sum() < 16:
            mask[b] = m
            continue

        # ---------- 3) 全局 MAD 补充过滤 ----------
        z_valid = log_z[m]
        r_valid = log_r[m]

        z_med = torch.median(z_valid)
        z_mad = torch.median(torch.abs(z_valid - z_med))
        z_thr = global_mad_scale_depth * 1.4826 * torch.clamp(z_mad, min=1e-6)

        r_med = torch.median(r_valid)
        r_mad = torch.median(torch.abs(r_valid - r_med))
        r_thr = global_mad_scale_range * 1.4826 * torch.clamp(r_mad, min=1e-6)

        m = m & (torch.abs(log_z - z_med) <= z_thr) & (torch.abs(log_r - r_med) <= r_thr)

        mask[b] = m

    return mask


@torch.no_grad()
def _sample_valid_corr_from_maps(gt_map, pr_map, valid_mask, max_samples=4096):
    """
    Sample correspondences from valid pixels of a single view.

    Args:
        gt_map: (H, W, 3)
        pr_map: (H, W, 3)
        valid_mask: (H, W)
    Returns:
        pr_corr: (N, 3) or None
        gt_corr: (N, 3) or None
    """
    idx = torch.nonzero(valid_mask, as_tuple=False)
    if idx.shape[0] == 0:
        return None, None

    if idx.shape[0] > max_samples:
        sel = torch.randperm(idx.shape[0], device=idx.device)[:max_samples]
        idx = idx[sel]

    u = idx[:, 0]
    v = idx[:, 1]
    gt_corr = gt_map[u, v]
    pr_corr = pr_map[u, v]

    good = torch.isfinite(gt_corr).all(dim=1) & torch.isfinite(pr_corr).all(dim=1)
    if good.sum() == 0:
        return None, None

    return pr_corr[good], gt_corr[good]


@torch.no_grad()
def _estimate_scene_scale_from_corr(gt_corr: torch.Tensor):
    if gt_corr is None or gt_corr.shape[0] == 0:
        return 1.0
    s = torch.median(torch.linalg.norm(gt_corr, dim=1))
    if not torch.isfinite(s) or s <= 1e-8:
        s = torch.tensor(1.0, device=gt_corr.device, dtype=gt_corr.dtype)
    return float(s.item())

@torch.no_grad()
def solve_batch_global_sim3(
    gt_pts_list,
    pr_pts_list,
    valid_masks,
    gt_pose_trans_list,
    pr_pose_trans_list,
    max_samples_per_view=4096,
    cam_center_repeat=64,
    trim_ratio=0.9,
    iters=3,
    min_corr=10,
    min_inlier_ratio=0.2,
    max_median_residual_ratio=0.5,
    max_scale_ratio=2.0,
):
    """
    Solve one robust pred->gt Sim(3) for each sample in the batch.

    Alignment form:
        X_gt = s * R * X_pred + t

    Returns:
        sim3_scale: (B,)
        sim3_rot: (B,3,3)
        sim3_trans: (B,3)
        sim3_valid: (B,) bool
        sim3_num_corr: (B,) long
        sim3_median_residual: (B,)
        sim3_inlier_ratio: (B,)
    """
    device = gt_pts_list[0].device
    dtype = gt_pts_list[0].dtype
    batch_size = gt_pts_list[0].shape[0]
    n_views = len(gt_pts_list)

    sim3_scale = torch.ones(batch_size, device=device, dtype=dtype)
    sim3_rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    sim3_trans = torch.zeros(batch_size, 3, device=device, dtype=dtype)

    sim3_valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
    sim3_num_corr = torch.zeros(batch_size, device=device, dtype=torch.long)
    sim3_median_residual = torch.full((batch_size,), float("nan"), device=device, dtype=dtype)
    sim3_inlier_ratio = torch.full((batch_size,), float("nan"), device=device, dtype=dtype)

    for b in range(batch_size):
        pr_corr_all = []
        gt_corr_all = []

        for v in range(n_views):
            pr_corr, gt_corr = _sample_valid_corr_from_maps(
                gt_pts_list[v][b],
                pr_pts_list[v][b],
                valid_masks[v][b],
                max_samples=max_samples_per_view,
            )
            if pr_corr is not None:
                pr_corr_all.append(pr_corr)
                gt_corr_all.append(gt_corr)

            pr_c = pr_pose_trans_list[v][b].reshape(1, 3)
            gt_c = gt_pose_trans_list[v][b].reshape(1, 3)
            if torch.isfinite(pr_c).all() and torch.isfinite(gt_c).all():
                pr_corr_all.append(pr_c.repeat(cam_center_repeat, 1))
                gt_corr_all.append(gt_c.repeat(cam_center_repeat, 1))

        if len(pr_corr_all) == 0:
            continue

        pr_corr_all = torch.cat(pr_corr_all, dim=0)
        gt_corr_all = torch.cat(gt_corr_all, dim=0)

        sim3_num_corr[b] = pr_corr_all.shape[0]
        if pr_corr_all.shape[0] < min_corr:
            continue

        s, R, t = sim3_from_correspondences_robust(
            pr_corr_all,
            gt_corr_all,
            trim_ratio=trim_ratio,
            iters=iters,
        )

        if not (torch.isfinite(s) and torch.isfinite(R).all() and torch.isfinite(t).all()):
            continue

        pr_al = (s * (R @ pr_corr_all.t())).t() + t[None, :]
        resid = torch.linalg.norm(pr_al - gt_corr_all, dim=1)

        if resid.numel() == 0 or not torch.isfinite(resid).any():
            continue

        med_resid = torch.median(resid)
        scene_scale = _estimate_scene_scale_from_corr(gt_corr_all)
        inlier_thr = max(0.1 * scene_scale, 1e-3)
        inlier_ratio = (resid < inlier_thr).float().mean()

        sim3_median_residual[b] = med_resid
        sim3_inlier_ratio[b] = inlier_ratio

        gt_scale = torch.median(torch.linalg.norm(gt_corr_all, dim=1))
        pr_al_scale = torch.median(torch.linalg.norm(pr_al, dim=1))

        scale_ratio = torch.tensor(float("nan"), device=device, dtype=dtype)
        if torch.isfinite(gt_scale) and torch.isfinite(pr_al_scale) and gt_scale > 1e-8 and pr_al_scale > 1e-8:
            ratio1 = pr_al_scale / gt_scale
            ratio2 = gt_scale / pr_al_scale
            scale_ratio = torch.maximum(ratio1, ratio2)

        valid = (
            torch.isfinite(s)
            and (pr_corr_all.shape[0] >= min_corr)
            and torch.isfinite(med_resid)
            and (float(med_resid.item()) <= max_median_residual_ratio * scene_scale)
            and torch.isfinite(inlier_ratio)
            and (float(inlier_ratio.item()) >= min_inlier_ratio)
            and torch.isfinite(scale_ratio)
            and (float(scale_ratio.item()) <= max_scale_ratio)
        )

        if valid:
            sim3_scale[b] = s
            sim3_rot[b] = R
            sim3_trans[b] = t
            sim3_valid[b] = True

    return (
        sim3_scale,
        sim3_rot,
        sim3_trans,
        sim3_valid,
        sim3_num_corr,
        sim3_median_residual,
        sim3_inlier_ratio,
    )


# ============================================================
# Main info assembly for metric computation
# ============================================================

def get_all_info_for_metric_computation(batch, preds, norm_mode="avg_dis"):
    """
    Returns:
      gt_info,
      pr_info,
      valid_masks,
      gt_info_abs,
      pr_info_abs_aligned,
      scale_factors

    Notes:
      - Relative metrics are computed from normalized pointmaps / normalized poses.
      - System-level absolute metrics are computed from a single global Sim(3)-aligned prediction.
      - Per-view depth absolute metrics are later computed with sequence-level scale-only alignment.
    """
    n_views = len(batch)
    batch_size = batch[0]["camera_pose"].shape[0]
    device = preds[0]["cam_quats"].device
    dtype = preds[0]["cam_quats"].dtype

    # --------------------------------------------------------
    # 1) Collect GT / Pred absolute quantities in view0 world
    # --------------------------------------------------------
    in_camera0 = inv(batch[0]["camera_pose"])

    no_norm_gt_pts = []
    no_norm_gt_pts3d_cam = []
    no_norm_gt_pose_trans = []
    gt_ray_directions = []
    gt_intrinsics = []
    gt_pose_rot = []
    valid_masks = []

    pred_camera0 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    pred_camera0_rot = quaternion_to_rotation_matrix(preds[0]["cam_quats"].clone())
    pred_camera0[..., :3, :3] = pred_camera0_rot
    pred_camera0[..., :3, 3] = preds[0]["cam_trans"].clone()
    pred_in_camera0 = inv(pred_camera0)

    no_norm_pr_pts = []
    no_norm_pr_pts3d_cam = []
    no_norm_pr_pose_trans = []
    pr_ray_directions = []
    pr_intrinsics = []
    pr_pose_rot = []

    for i in range(n_views):
        # ---------------- GT ----------------
        gt_pts_i = geotrf(in_camera0, batch[i]["pts3d"])
        no_norm_gt_pts.append(gt_pts_i)
        no_norm_gt_pts3d_cam.append(batch[i]["pts3d_cam"])
        valid_masks.append(batch[i]["valid_mask"].clone())
        gt_ray_directions.append(batch[i]["ray_directions_cam"])
        gt_intrinsics.append(batch[i]["camera_intrinsics"])

        if i == 0:
            gt_rot_i = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)  # 10,3,3
            gt_trans_i = torch.zeros(batch_size, 3, device=device, dtype=dtype)  # 10,3
        else:
            gt_pose_quats_in_view0, gt_trans_i = transform_pose_using_quats_and_trans_2_to_1(
                batch[0]["camera_pose_quats"],
                batch[0]["camera_pose_trans"],
                batch[i]["camera_pose_quats"],
                batch[i]["camera_pose_trans"],
            )
            gt_rot_i = quaternion_to_rotation_matrix(gt_pose_quats_in_view0)

        gt_pose_rot.append(gt_rot_i)
        no_norm_gt_pose_trans.append(gt_trans_i)

        # ---------------- Pred ----------------
        pr_pose_quats_in_view0, pr_trans_i = transform_pose_using_quats_and_trans_2_to_1(
            preds[0]["cam_quats"], preds[0]["cam_trans"], preds[i]["cam_quats"], preds[i]["cam_trans"]
        )
        pr_rot_i = quaternion_to_rotation_matrix(pr_pose_quats_in_view0)
        pr_pts_i = geotrf(pred_in_camera0, preds[i]["pts3d"])

        if "metric_scaling_factor" in preds[i]:
            sf = preds[i]["metric_scaling_factor"].unsqueeze(-1).unsqueeze(-1)
            pr_pts_i = pr_pts_i / sf
            pr_pts3d_cam_i = preds[i]["pts3d_cam"] / sf
            pr_trans_i = pr_trans_i / preds[i]["metric_scaling_factor"]
        else:
            pr_pts3d_cam_i = preds[i]["pts3d_cam"]
        
        if "intrinsics" in preds[i]:
            pr_intr_i = preds[i]['intrinsics']
        else:
            pr_intr_i = recover_pinhole_intrinsics_from_ray_directions(preds[i]["ray_directions"])

        no_norm_pr_pts.append(pr_pts_i)
        no_norm_pr_pts3d_cam.append(pr_pts3d_cam_i)
        no_norm_pr_pose_trans.append(pr_trans_i)
        pr_ray_directions.append(preds[i]["ray_directions"])
        pr_intrinsics.append(pr_intr_i)
        pr_pose_rot.append(pr_rot_i)

    # --------------------------------------------------------
    # 2) Pre-filter obvious outliers before solving global Sim(3)
    # --------------------------------------------------------
    valid_masks = [
        valid_masks[i]
        & _mask_outliers_per_batch(no_norm_gt_pts[i], no_norm_gt_pts3d_cam[i], valid_masks[i])
        & _mask_outliers_per_batch(no_norm_pr_pts[i], no_norm_pr_pts3d_cam[i], valid_masks[i])
        for i in range(n_views)
    ]

    # --------------------------------------------------------
    # 3) Solve one global pred->gt Sim(3)
    # --------------------------------------------------------
    (
        sim3_scale,
        sim3_rot,
        sim3_trans,
        sim3_valid,
        sim3_num_corr,
        sim3_median_residual,
        sim3_inlier_ratio,
    ) = solve_batch_global_sim3(
        gt_pts_list=no_norm_gt_pts,
        pr_pts_list=no_norm_pr_pts,
        valid_masks=valid_masks,
        gt_pose_trans_list=no_norm_gt_pose_trans,
        pr_pose_trans_list=no_norm_pr_pose_trans,
        max_samples_per_view=4096,
    )

    # --------------------------------------------------------
    # 3) Align pred absolute quantities to GT with this Sim(3)
    # --------------------------------------------------------
    aligned_pr_pts = []
    aligned_pr_pts3d_cam = []
    aligned_pr_pose_trans = []
    aligned_pr_pose_rot = []

    for i in range(n_views):
        pr_pts_i = no_norm_pr_pts[i]  # (B,H,W,3)
        s_map = sim3_scale[:, None, None, None]
        t_map = sim3_trans[:, None, None, :]

        pr_pts_i_al = s_map * torch.einsum("bij,bhwj->bhwi", sim3_rot, pr_pts_i) + t_map
        aligned_pr_pts.append(pr_pts_i_al)

        # camera-frame depth points: only scale is physically meaningful here
        pr_pts3d_cam_i_al = no_norm_pr_pts3d_cam[i] * s_map
        aligned_pr_pts3d_cam.append(pr_pts3d_cam_i_al)

        pr_rot_i_al = torch.einsum("bij,bjk->bik", sim3_rot, pr_pose_rot[i])
        pr_trans_i_al = sim3_scale[:, None] * torch.einsum(
            "bij,bj->bi", sim3_rot, no_norm_pr_pose_trans[i]
        ) + sim3_trans

        aligned_pr_pose_rot.append(pr_rot_i_al)
        aligned_pr_pose_trans.append(pr_trans_i_al)

    # --------------------------------------------------------
    # 5) Normalize GT and aligned pred separately for relative metrics
    # --------------------------------------------------------
    gt_norm_out = normalize_multiple_pointclouds(no_norm_gt_pts, valid_masks, norm_mode, ret_factor=True)
    gt_pts_norm = gt_norm_out[:-1]
    gt_norm_factor = gt_norm_out[-1]

    pr_norm_out = normalize_multiple_pointclouds(aligned_pr_pts, valid_masks, norm_mode, ret_factor=True)
    pr_pts_norm = pr_norm_out[:-1]
    pr_norm_factor = pr_norm_out[-1]

    gt_pts = []
    gt_pts3d_cam = []
    gt_pose_trans = []
    pr_pts = []
    pr_pts3d_cam = []
    pr_pose_trans = []

    for i in range(n_views):
        gt_pts.append(gt_pts_norm[i].cpu())
        gt_pts3d_cam.append((no_norm_gt_pts3d_cam[i] / gt_norm_factor).cpu())
        gt_pose_trans.append((no_norm_gt_pose_trans[i] / gt_norm_factor[:, :, 0, 0]).cpu())

        pr_pts.append(pr_pts_norm[i].cpu())
        pr_pts3d_cam.append((aligned_pr_pts3d_cam[i] / pr_norm_factor).cpu())
        pr_pose_trans.append((aligned_pr_pose_trans[i] / pr_norm_factor[:, :, 0, 0]).cpu())

        valid_masks[i] = valid_masks[i].cpu()

    # --------------------------------------------------------
    # 5) Build pose matrices
    # --------------------------------------------------------
    gt_poses = []
    pr_poses = []
    gt_poses_abs = []
    pr_poses_abs = []

    for i in range(n_views):
        gt_pose_curr = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        gt_pose_curr[..., :3, :3] = gt_pose_rot[i]
        gt_pose_curr[..., :3, 3] = gt_pose_trans[i].to(device)
        gt_poses.append(gt_pose_curr.cpu())

        pr_pose_curr = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        pr_pose_curr[..., :3, :3] = aligned_pr_pose_rot[i]
        pr_pose_curr[..., :3, 3] = pr_pose_trans[i].to(device)
        pr_poses.append(pr_pose_curr.cpu())

        gt_pose_abs_curr = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        gt_pose_abs_curr[..., :3, :3] = gt_pose_rot[i]
        gt_pose_abs_curr[..., :3, 3] = no_norm_gt_pose_trans[i]
        gt_poses_abs.append(gt_pose_abs_curr.cpu())

        pr_pose_abs_curr = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        pr_pose_abs_curr[..., :3, :3] = aligned_pr_pose_rot[i]
        pr_pose_abs_curr[..., :3, 3] = aligned_pr_pose_trans[i]
        pr_poses_abs.append(pr_pose_abs_curr.cpu())

    gt_z_depths = [x[..., 2:].cpu() for x in gt_pts3d_cam]
    pr_z_depths = [x[..., 2:].cpu() for x in pr_pts3d_cam]
    gt_z_depths_abs = [x[..., 2:].cpu() for x in no_norm_gt_pts3d_cam]
    pr_z_depths_abs = [x[..., 2:].cpu() for x in aligned_pr_pts3d_cam]

    gt_info = {
        "ray_directions": gt_ray_directions,
        "intrinsics": gt_intrinsics,
        "z_depths": gt_z_depths,
        "poses": gt_poses,
        "pts3d": gt_pts,
        "metric_scale": gt_norm_factor[:, 0, 0, 0].cpu(),
    }

    pr_info = {
        "ray_directions": pr_ray_directions,
        "intrinsics": pr_intrinsics,
        "z_depths": pr_z_depths,
        "poses": pr_poses,
        "pts3d": pr_pts,
        "metric_scale": pr_norm_factor[:, 0, 0, 0].cpu(),
    }

    gt_info_abs = {
        "ray_directions": gt_ray_directions,
        "intrinsics": gt_intrinsics,
        "poses": gt_poses_abs,
        "pts3d": [x.cpu() for x in no_norm_gt_pts],
        "z_depths": gt_z_depths_abs,
    }

    pr_info_abs_aligned = {
        "ray_directions": pr_ray_directions,
        "intrinsics": pr_intrinsics,
        "poses": pr_poses_abs,
        "pts3d": [x.cpu() for x in aligned_pr_pts],
        "z_depths": pr_z_depths_abs,
    }

    scale_factors = {
        "pr_to_gt_scale": sim3_scale.detach().cpu(),
        "sim3_rot": sim3_rot.detach().cpu(),
        "sim3_trans": sim3_trans.detach().cpu(),
        "sim3_valid": sim3_valid.detach().cpu(),
        "sim3_num_corr": sim3_num_corr.detach().cpu(),
        "sim3_median_residual": sim3_median_residual.detach().cpu(),
        "sim3_inlier_ratio": sim3_inlier_ratio.detach().cpu(),
        "gt_norm_factor": gt_norm_factor.detach().cpu(),
        "pr_norm_factor": pr_norm_factor.detach().cpu(),
    }
    return gt_info, pr_info, valid_masks, gt_info_abs, pr_info_abs_aligned, scale_factors


# ============================================================
# Dataset / benchmark entry
# ============================================================

def build_dataset(dataset, batch_size, num_workers):
    print("Building data loader for dataset: ", dataset)
    loader = get_test_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=False,
        drop_last=False,
    )
    print("Dataset length: ", len(loader))
    return loader


@torch.no_grad()
def benchmark(args):
    print("Output Directory: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    if args.amp:
        if args.amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif args.amp_dtype == "bf16":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    data_loaders = {
        dataset.split("(")[0]: build_dataset(dataset, args.batch_size, args.dataset.num_workers)
        for dataset in args.dataset.test_dataset.split("+")
        if "(" in dataset
    }

    model = init_model(args.model.model_str, args.model.model_config, torch_hub_force_reload=False)
    model.to(device)
    model.eval()

    if args.model.pretrained:
        ckpt = torch.load(args.model.pretrained, map_location=device, weights_only=False)
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt

    per_dataset_results = {}

    for benchmark_dataset_name, data_loader in data_loaders.items():
        print("Benchmarking dataset: ", benchmark_dataset_name)
        data_loader.dataset.set_epoch(0)

        max_sets_per_scene = getattr(args, "save_n_fused_ply", 5)
        per_scene_saved = {scene: 0 for scene in data_loader.dataset.dataset.scenes}

        fused_out_dir = os.path.join(args.output_dir, f"{benchmark_dataset_name}_fused_ply")
        os.makedirs(fused_out_dir, exist_ok=True)

        per_scene_results = {}
        for dataset_scene in data_loader.dataset.dataset.scenes:
            per_scene_results[dataset_scene] = {
                "metric_scale_abs_rel": [],
                "pointmaps_abs_rel": [],
                "pointmaps_inlier_thres_103": [],
                "pose_ate_rmse": [],
                "pose_auc_5": [],
                "z_depth_abs_rel": [],
                "z_depth_inlier_thres_103": [],
                "ray_dirs_err_deg": [],
                # global/system absolute metrics
                "pointmaps_abs_mae_global": [],
                "pointmaps_abs_rmse_global": [],
                "pose_ate_abs": [],
                "merged_pc_abs_chamfer_l1": [],
                "merged_pc_abs_chamfer_rmse": [],
                "merged_pc_abs_inlier_ratio": [],
                # depth absolute metrics with sequence-level scale-only alignment
                "z_depth_abs_mae_seq_scale": [],
                "z_depth_abs_rmse_seq_scale": [],
                "z_depth_abs_rel_seq_scale": [],
                "z_depth_delta1_seq_scale": [],
                "pr_to_gt_scale": [],
                # 
                "sim3_valid": [],
                "sim3_failure": [],
                "sim3_num_corr": [],
                "sim3_median_residual": [],
                "sim3_inlier_ratio": [],
                "total_count": [],
            }

        for batch in tqdm(data_loader):
            n_views = len(batch)

            for view in batch:
                view["idx"] = view["idx"][2:]

            ignore_keys = {
                "depthmap",
                "dataset",
                "label",
                "instance",
                "idx",
                "true_shape",
                "rng",
                "data_norm_type",
            }
            for view in batch:
                for name in view.keys():
                    if name in ignore_keys:
                        continue
                    view[name] = view[name].to(device, non_blocking=True)

            with torch.autocast("cuda", enabled=bool(args.amp), dtype=amp_dtype):
                preds = model(batch)

            gt_info, pr_info, valid_masks, gt_info_abs, pr_info_abs_aligned, scale_factors = (
                get_all_info_for_metric_computation(batch, preds)
            )

            batch_size = batch[0]["img"].shape[0]
            compute_abs = bool(args.get("compute_abs_metrics", False))
            for batch_idx in range(batch_size):
                scene = batch[0]["label"][batch_idx]
                k = per_scene_saved[scene]
                need_save_fused = (max_sets_per_scene > 0) and (k < max_sets_per_scene)

                metrics, fused_debug = compute_set_metrics(
                    batch_views=batch,
                    batch_idx=batch_idx,
                    gt_info=gt_info,
                    pr_info=pr_info,
                    valid_masks=valid_masks,
                    gt_info_abs=gt_info_abs,
                    pr_info_abs=pr_info_abs_aligned,
                    scale_factors=scale_factors,
                    device=device,
                    voxel=0.25,
                    icp_iters=0,
                    trim_ratio=0.8,
                    return_fused_debug=need_save_fused,
                    compute_abs_metrics=compute_abs,
                )

                if fused_debug is not None and k < max_sets_per_scene:
                    safe_scene = str(scene).replace("/", "_")
                    base = f"{safe_scene}_set{str(k).zfill(3)}"
                    score_for_name = fused_debug.chamfer_l1 if np.isfinite(fused_debug.chamfer_l1) else -1.0
                    gt_path = os.path.join(
                        fused_out_dir, safe_scene, base + f"_GT_{score_for_name:.3f}.ply"
                    )
                    pr_path = os.path.join(
                        fused_out_dir, safe_scene, base + f"_Pred_{score_for_name:.3f}.ply"
                    )
                    meta_path = os.path.join(fused_out_dir, safe_scene, base + "_bundle.json")
                    write_ply_xyzrgb(gt_path, fused_debug.gt_ds, fused_debug.gt_colors_ds)
                    write_ply_xyzrgb(pr_path, fused_debug.pr_ds, fused_debug.pr_colors_ds)
                    save_repro_bundle_json(
                        json_path=meta_path,
                        benchmark_dataset_name=benchmark_dataset_name,
                        scene=scene,
                        set_idx=k,
                        batch_views=batch,
                        batch_idx=batch_idx,
                        gt_info_abs=gt_info_abs,
                        pr_info_abs_aligned=pr_info_abs_aligned,
                        scale_factors=scale_factors,
                        metrics=metrics,
                        gt_ply_path=gt_path,
                        pred_ply_path=pr_path,
                    )
                    per_scene_saved[scene] += 1

                for k_metric in per_scene_results[scene].keys():
                    per_scene_results[scene][k_metric].append(float(metrics.get(k_metric, np.nan)))

        with open(os.path.join(args.output_dir, f"{benchmark_dataset_name}_per_scene_results.json"), "w") as f:
            json.dump(per_scene_results, f, indent=4)

        across_dataset_results = {}
        for scene in per_scene_results.keys():
            for metric in per_scene_results[scene].keys():
                across_dataset_results.setdefault(metric, [])
                across_dataset_results[metric].extend(per_scene_results[scene][metric])

        for metric in list(across_dataset_results.keys()):
            if metric == "failure_count":
                continue
            if metric == "failure_rate":
                continue
            if metric == "total_count":
                across_dataset_results[metric] = float(np.nansum(across_dataset_results[metric]))
            elif metric == "sim3_failure":
                across_dataset_results["failure_count"] = float(np.nansum(across_dataset_results[metric]))
                across_dataset_results[metric] = float(np.nanmean(across_dataset_results[metric]))
            else:
                across_dataset_results[metric] = float(np.nanmean(across_dataset_results[metric]))

        total_count = float(across_dataset_results.get("total_count", 0.0))
        failure_count = float(across_dataset_results.get("failure_count", 0.0))
        across_dataset_results["failure_rate"] = failure_count / total_count if total_count > 0 else float("nan")

        with open(os.path.join(args.output_dir, f"{benchmark_dataset_name}_avg_across_all_scenes.json"), "w") as f:
            json.dump(across_dataset_results, f, indent=4)

        print("Average results across all scenes for dataset: ", benchmark_dataset_name)
        for metric in across_dataset_results.keys():
            print(f"{metric}: {across_dataset_results[metric]}")

        per_dataset_results[benchmark_dataset_name] = across_dataset_results

    average_results = {}
    first = next(iter(per_dataset_results))
    for metric in per_dataset_results[first].keys():
        vals = [per_dataset_results[d][metric] for d in per_dataset_results if metric in per_dataset_results[d]]
        if metric in {"total_count", "failure_count"}:
            average_results[metric] = float(np.nansum(vals))
        elif metric == "failure_rate":
            continue
        else:
            average_results[metric] = float(np.nanmean(vals))

    total_count = float(average_results.get("total_count", 0.0))
    failure_count = float(average_results.get("failure_count", 0.0))
    average_results["failure_rate"] = failure_count / total_count if total_count > 0 else float("nan")

    per_dataset_results["Average"] = average_results

    print("Benchmarking Done!")
    for metric in average_results.keys():
        print(f"{metric}: {average_results[metric]}")

    with open(os.path.join(args.output_dir, "per_dataset_results.json"), "w") as f:
        json.dump(per_dataset_results, f, indent=4)


# @hydra.main(version_base=None, config_path="../../configs", config_name="debug_dense_n_view_benchmark")
@hydra.main(version_base=None, config_path="../../configs", config_name="dense_n_view_benchmark")
def execute_benchmarking(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)

    benchmark(cfg)


if __name__ == "__main__":
    execute_benchmarking()  # noqa
