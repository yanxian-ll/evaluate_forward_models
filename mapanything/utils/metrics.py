# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Utils for Metrics

- Pose metrics (ATE, AUC@K)
- Dense metrics (rel abs, inlier ratio, ray-dir angular err)
- Absolute metrics with per-image alignment:
    * pointmap: per-view Sim3 alignment then MAE/RMSE
    * depth (z): per-view affine alignment (a*z+b) then MAE/RMSE
- Fused pointcloud metrics (Sim3 + ICP + Chamfer + inlier ratio)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mapanything.utils.image import rgb


# ============================================================
# Basic helpers (numpy)
# ============================================================

def valid_mean(arr, mask, axis=None, keepdims=np._NoValue):
    mask = mask.astype(arr.dtype) if mask.dtype == bool else mask
    num_valid = np.sum(mask, axis=axis, keepdims=keepdims)
    masked_arr = arr * mask
    masked_arr_sum = np.sum(masked_arr, axis=axis, keepdims=keepdims)

    with np.errstate(divide="ignore", invalid="ignore"):
        vm = masked_arr_sum / num_valid
        is_valid = np.isfinite(vm)
        vm = np.nan_to_num(vm, nan=0, posinf=0, neginf=0)

    return vm, is_valid


def thresh_inliers(gt, pred, thresh=1.03, mask=None, output_scaling_factor=1.0):
    gt_norm = np.linalg.norm(gt, axis=-1)
    pred_norm = np.linalg.norm(pred, axis=-1)

    gt_norm_valid = (gt_norm) > 0
    combined_mask = (mask & gt_norm_valid) if mask is not None else gt_norm_valid

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_1 = np.nan_to_num(gt_norm / pred_norm, nan=thresh + 1, posinf=thresh + 1, neginf=thresh + 1)
        rel_2 = np.nan_to_num(pred_norm / gt_norm, nan=0, posinf=0, neginf=0)

    max_rel = np.maximum(rel_1, rel_2)
    inliers = ((0 < max_rel) & (max_rel < thresh)).astype(np.float32)

    inlier_ratio, valid = valid_mean(inliers, combined_mask)
    inlier_ratio = inlier_ratio * output_scaling_factor
    inlier_ratio = inlier_ratio if valid else np.nan
    return inlier_ratio


def m_rel_ae(gt, pred, mask=None, output_scaling_factor=1.0):
    error_norm = np.linalg.norm(pred - gt, axis=-1)
    gt_norm = np.linalg.norm(gt, axis=-1)

    gt_norm_valid = (gt_norm) > 0
    combined_mask = (mask & gt_norm_valid) if mask is not None else gt_norm_valid

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_ae = np.nan_to_num(error_norm / gt_norm, nan=0, posinf=0, neginf=0)

    out, valid = valid_mean(rel_ae, combined_mask)
    out = out * output_scaling_factor
    out = out if valid else np.nan
    return out


# ============================================================
# Angular error for unit ray directions
# ============================================================

def l2_distance_of_unit_ray_directions_to_angular_error(l2_distance: torch.Tensor) -> torch.Tensor:
    angular_error_radians = 2 * torch.asin(l2_distance / 2)
    angular_error_degrees = angular_error_radians * 180.0 / math.pi
    return angular_error_degrees


# ============================================================
# Pose metrics (ATE / AUC)
# ============================================================

def align(model, data):
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3, -1))
    data_zerocentered = data - data.mean(1).reshape((3, -1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1).reshape((3, -1)) - rot * model.mean(1).reshape((3, -1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]
    return rot, trans, trans_error


def evaluate_ate(gt_traj, est_traj):
    gt_traj_pts = [gt_traj[idx][:3, 3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3, 3] for idx in range(len(est_traj))]

    gt_traj_pts = torch.stack(gt_traj_pts).detach().cpu().numpy().T
    est_traj_pts = torch.stack(est_traj_pts).detach().cpu().numpy().T

    _, _, trans_error = align(gt_traj_pts, est_traj_pts)
    return trans_error.mean()


def build_pair_index(N, B=1):
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    out = out[..., [1, 2, 3, 0]]  # rijk -> ijkr
    out = standardize_quaternion(out)
    return out


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)
    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)
    return rel_rangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))
    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred) * 180.0 / np.pi
    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())
    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)
    return rel_tangle_deg


def calculate_auc_np(r_error, t_error, max_threshold=30):
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def closed_form_inverse_se3(se3, R=None, T=None):
    is_numpy = isinstance(se3, np.ndarray)
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    if is_numpy:
        R_transposed = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)
        top_right = -torch.bmm(R_transposed, T)
        inverted_matrix = torch.eye(4, 4, device=R.device, dtype=R.dtype)[None].repeat(len(R), 1, 1)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right
    return inverted_matrix


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)
    relative_pose_gt = closed_form_inverse_se3(gt_se3[pair_idx_i1]).bmm(gt_se3[pair_idx_i2])
    relative_pose_pred = closed_form_inverse_se3(pred_se3[pair_idx_i1]).bmm(pred_se3[pair_idx_i2])

    rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
    rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3])
    return rel_rangle_deg, rel_tangle_deg


# ============================================================
# Point cloud utils (merge / voxel / nn / ICP / Chamfer)
# ============================================================

def merge_masked_points_list(pts_list_np, rgb_list_u8, masks_list_np):
    all_pts, all_col = [], []
    for pts, col, m in zip(pts_list_np, rgb_list_u8, masks_list_np):
        if pts is None or col is None:
            continue
        valid = m.astype(bool)
        p = pts[valid].reshape(-1, 3)
        c = col[valid].reshape(-1, 3)
        if p.size > 0:
            all_pts.append(p.astype(np.float32))
            all_col.append(c.astype(np.uint8))
    if len(all_pts) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    return np.concatenate(all_pts, axis=0), np.concatenate(all_col, axis=0)


def voxel_downsample_np(points, colors_u8, voxel_size):
    if points.shape[0] == 0 or voxel_size <= 0:
        return points.astype(np.float32), colors_u8.astype(np.uint8)

    grid = np.floor(points / voxel_size).astype(np.int64)
    uniq, inv = np.unique(grid, axis=0, return_inverse=True)

    counts = np.bincount(inv)
    M = uniq.shape[0]

    pts_sum = np.zeros((M, 3), dtype=np.float64)
    col_sum = np.zeros((M, 3), dtype=np.float64)

    np.add.at(pts_sum, inv, points.astype(np.float64))
    np.add.at(col_sum, inv, colors_u8.astype(np.float64))

    pts_ds = (pts_sum / counts[:, None]).astype(np.float32)
    col_ds = np.clip(col_sum / counts[:, None], 0, 255)
    col_ds = (col_ds + 0.5).astype(np.uint8)
    return pts_ds, col_ds


@torch.no_grad()
def _nn_distance_torch(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_chunk: int = 2048,
    dst_chunk: int = 2048,
):
    """
    Memory-safe NN search using block-wise cdist.
    Complexity: O(Ns*Nd) compute, but peak memory O(src_chunk*dst_chunk).
    Returns:
      best_d: (Ns,)
      best_j: (Ns,) indices into dst
    """
    Ns = src.shape[0]
    device = src.device
    dtype = src.dtype

    # use +inf init
    best_d = torch.full((Ns,), float("inf"), device=device, dtype=dtype)
    best_j = torch.zeros((Ns,), device=device, dtype=torch.long)

    Nd = dst.shape[0]
    for i in range(0, Ns, src_chunk):
        s = src[i : i + src_chunk]  # (cs,3)
        cs = s.shape[0]

        bd = torch.full((cs,), float("inf"), device=device, dtype=dtype)
        bj = torch.zeros((cs,), device=device, dtype=torch.long)

        for j0 in range(0, Nd, dst_chunk):
            dblk = dst[j0 : j0 + dst_chunk]  # (cd,3)
            # (cs,cd) but cd is small => low peak memory
            d = torch.cdist(s, dblk, p=2)
            md, mj = torch.min(d, dim=1)  # (cs,)

            better = md < bd
            if better.any():
                bd[better] = md[better]
                bj[better] = mj[better] + j0

        best_d[i : i + cs] = bd
        best_j[i : i + cs] = bj

    return best_d, best_j


@torch.no_grad()
def kabsch_se3_torch(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-9):
    N = A.shape[0]
    if N < 3:
        R = torch.eye(3, device=A.device, dtype=A.dtype)
        t = B.mean(dim=0) - A.mean(dim=0)
        return R, t

    muA = A.mean(dim=0)
    muB = B.mean(dim=0)
    AA = A - muA
    BB = B - muB

    H = AA.t() @ BB
    U, S, Vt = torch.linalg.svd(H, full_matrices=False)
    R = Vt.t() @ U.t()

    if torch.det(R) < 0:
        Vt = Vt.clone()
        Vt[-1, :] *= -1
        R = Vt.t() @ U.t()

    t = muB - (R @ muA)
    return R, t

@torch.no_grad()
def icp_se3_torch(
    src: torch.Tensor,
    dst: torch.Tensor,
    iters: int = 20,
    max_corr_dist: float | None = None,
    nn_src_chunk: int = 2048,
    nn_dst_chunk: int = 2048,
    max_src_corr: int = 30000,
    trimmed_ratio: float | None = 0.8,
):
    device = src.device
    dtype = src.dtype

    R_tot = torch.eye(3, device=device, dtype=dtype)
    t_tot = torch.zeros(3, device=device, dtype=dtype)

    X = src
    for _ in range(iters):
        # subsample src for correspondences
        if X.shape[0] > max_src_corr:
            idx = torch.randperm(X.shape[0], device=device)[:max_src_corr]
            Xc = X[idx]
        else:
            Xc = X

        d, j = _nn_distance_torch(Xc, dst, src_chunk=nn_src_chunk, dst_chunk=nn_dst_chunk)
        Yc = dst[j]

        if max_corr_dist is not None:
            keep = d < max_corr_dist
            if keep.sum() < 50:
                break
            Xk = Xc[keep]
            Yk = Yc[keep]
            dk = d[keep]
        else:
            Xk, Yk, dk = Xc, Yc, d

        if trimmed_ratio is not None and 0 < trimmed_ratio < 1:
            m = Xk.shape[0]
            k_keep = max(int(m * trimmed_ratio), 50)
            _, order = torch.topk(dk, k=k_keep, largest=False)
            A = Xk[order]
            B = Yk[order]
        else:
            A, B = Xk, Yk

        dR, dt = kabsch_se3_torch(A, B)

        R_tot = dR @ R_tot
        t_tot = dR @ t_tot + dt
        X = (dR @ X.t()).t() + dt[None, :]

    return R_tot, t_tot, X

@torch.no_grad()
def chamfer_inlier_torch(
    pred: torch.Tensor,
    gt: torch.Tensor,
    inlier_dist: float,
    nn_src_chunk: int = 2048,
    nn_dst_chunk: int = 2048,
    max_eval: int = 20000,
):
    if pred.shape[0] > max_eval:
        pred = pred[torch.randperm(pred.shape[0], device=pred.device)[:max_eval]]
    if gt.shape[0] > max_eval:
        gt = gt[torch.randperm(gt.shape[0], device=gt.device)[:max_eval]]

    d_pg, _ = _nn_distance_torch(pred, gt, src_chunk=nn_src_chunk, dst_chunk=nn_dst_chunk)
    d_gp, _ = _nn_distance_torch(gt, pred, src_chunk=nn_src_chunk, dst_chunk=nn_dst_chunk)

    chamfer_l1 = (d_pg.mean() + d_gp.mean()).item()
    chamfer_rmse = (torch.sqrt((d_pg * d_pg).mean()) + torch.sqrt((d_gp * d_gp).mean())).item()
    inlier_ratio = (d_pg < inlier_dist).float().mean().item()
    return chamfer_l1, chamfer_rmse, inlier_ratio

# ============================================================
# Per-image alignment helpers (Sim3 for points, affine for depth)
# ============================================================

def apply_sim3_to_points_np(points_hw3: np.ndarray, s: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> np.ndarray:
    P = torch.from_numpy(points_hw3.reshape(-1, 3)).to(device=R.device, dtype=torch.float32)
    P_al = (s * (R @ P.t())).t() + t[None, :]
    return P_al.detach().cpu().numpy().reshape(points_hw3.shape)


def fit_affine_1d_np(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    x = pred[mask].reshape(-1)
    y = gt[mask].reshape(-1)
    if x.size < 10:
        return 1.0, 0.0
    A = np.stack([x, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    if not np.isfinite(a): a = 1.0
    if not np.isfinite(b): b = 0.0
    return a, b


@torch.no_grad()
def gather_correspondences_from_pointmaps(
    pr_pts_list_abs,
    gt_pts_list_abs,
    masks_list,
    device,
    dtype=torch.float32,
    max_samples_total: int = 1000,
):
    pr_all, gt_all = [], []
    valid_counts = [int(m.astype(bool).sum()) for m in masks_list]
    total_valid = sum(valid_counts)
    if total_valid <= 0:
        return None, None

    for pr_map, gt_map, m, cnt in zip(pr_pts_list_abs, gt_pts_list_abs, masks_list, valid_counts):
        if cnt <= 0:
            continue

        if not torch.is_tensor(pr_map):
            pr_map_t = torch.from_numpy(pr_map).to(device=device, dtype=dtype)
        else:
            pr_map_t = pr_map.to(device=device, dtype=dtype)

        if not torch.is_tensor(gt_map):
            gt_map_t = torch.from_numpy(gt_map).to(device=device, dtype=dtype)
        else:
            gt_map_t = gt_map.to(device=device, dtype=dtype)

        m_bool = m.astype(bool)
        ij = np.argwhere(m_bool)
        if ij.shape[0] == 0:
            continue

        k = int(max_samples_total * (cnt / total_valid))
        k = min(k, ij.shape[0])
        sel = np.random.choice(ij.shape[0], k, replace=False)
        uv = ij[sel]

        u = torch.from_numpy(uv[:, 0]).to(device=device, dtype=torch.long)
        v = torch.from_numpy(uv[:, 1]).to(device=device, dtype=torch.long)

        pr = pr_map_t[u, v]
        gt = gt_map_t[u, v]

        good = torch.isfinite(pr).all(dim=1) & torch.isfinite(gt).all(dim=1)
        if good.sum() > 0:
            pr_all.append(pr[good])
            gt_all.append(gt[good])

    if len(pr_all) == 0:
        return None, None
    pr_corr = torch.cat(pr_all, dim=0)
    gt_corr = torch.cat(gt_all, dim=0)
    return pr_corr, gt_corr


@torch.no_grad()
def umeyama_sim3_torch(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-9):
    N = A.shape[0]
    if N < 3:
        s = torch.tensor(1.0, device=A.device, dtype=A.dtype)
        R = torch.eye(3, device=A.device, dtype=A.dtype)
        t = B.mean(dim=0) - A.mean(dim=0)
        return s, R, t

    muA = A.mean(dim=0)
    muB = B.mean(dim=0)
    AA = A - muA
    BB = B - muB

    varA = (AA * AA).sum(dim=1).mean().clamp(min=eps)
    Sigma = (BB.t() @ AA) / N

    U, S, Vt = torch.linalg.svd(Sigma, full_matrices=False)
    det = torch.det(U @ Vt)
    d3 = torch.sign(det)
    D = torch.diag(torch.tensor([1.0, 1.0, d3.item()], device=A.device, dtype=A.dtype))
    R = U @ D @ Vt

    s = (S * torch.diag(D)).sum() / varA
    t = muB - s * (R @ muA)
    return s, R, t


@torch.no_grad()
def sim3_from_correspondences_robust(
    pr_corr: torch.Tensor,
    gt_corr: torch.Tensor,
    trim_ratio: float = 0.8,
    iters: int = 2,
    eps: float = 1e-9,
):
    if pr_corr is None or gt_corr is None or pr_corr.shape[0] < 10:
        device = gt_corr.device if gt_corr is not None else "cpu"
        s = torch.tensor(1.0, device=device)
        R = torch.eye(3, device=device)
        t = torch.zeros(3, device=device)
        return s, R, t

    s, R, t = umeyama_sim3_torch(pr_corr, gt_corr, eps=eps)

    if trim_ratio is None or not (0 < trim_ratio < 1):
        return s, R, t

    X = pr_corr
    Y = gt_corr
    for _ in range(max(iters, 1)):
        X_al = (s * (R @ X.t())).t() + t[None, :]
        err = torch.norm(X_al - Y, dim=1)
        N = err.shape[0]
        k = max(int(N * trim_ratio), 2000) if N >= 2000 else max(int(N * trim_ratio), 10)
        _, idx = torch.topk(err, k=k, largest=False)
        Xk, Yk = X[idx], Y[idx]
        s, R, t = umeyama_sim3_torch(Xk, Yk, eps=eps)

    return s, R, t


# ============================================================
# Global scale helper (used by get_all_info... in benchmark)
# ============================================================

def global_scale_from_pointmaps(
    gt_pts_list,
    pr_pts_list,
    masks_list,
    max_samples_per_view=20000,
    eps=1e-8,
):
    ratios = []
    for gt, pr, m in zip(gt_pts_list, pr_pts_list, masks_list):
        gt = gt.detach().cpu().numpy()
        pr = pr.detach().cpu().numpy()
        m = m.detach().cpu().numpy().astype(bool)

        if gt.ndim == 3:
            gt_v = gt[m].reshape(-1, 3)
            pr_v = pr[m].reshape(-1, 3)
        else:
            gt_v = gt.reshape(-1, 3)
            pr_v = pr.reshape(-1, 3)

        if gt_v.shape[0] == 0:
            continue

        if gt_v.shape[0] > max_samples_per_view:
            idx = np.random.choice(gt_v.shape[0], max_samples_per_view, replace=False)
            gt_v = gt_v[idx]
            pr_v = pr_v[idx]

        gt_n = np.linalg.norm(gt_v, axis=1)
        pr_n = np.linalg.norm(pr_v, axis=1)

        valid = (gt_n > eps) & (pr_n > eps) & np.isfinite(gt_n) & np.isfinite(pr_n)
        if valid.sum() == 0:
            continue

        r = gt_n[valid] / pr_n[valid]
        ratios.append(r)

    if len(ratios) == 0:
        return 1.0
    ratios = np.concatenate(ratios, axis=0)
    if ratios.size == 0:
        return 1.0

    s = float(np.median(ratios))
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    return s


@torch.no_grad()
def random_sample_points_torch(x: torch.Tensor, max_n: int) -> torch.Tensor:
    if max_n is None or max_n <= 0 or x.shape[0] <= max_n:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:max_n]
    return x[idx]

# ============================================================
# High-level: compute all metrics for one multi-view set
# ============================================================

@dataclass
class FusedPCDebug:
    gt_ds: np.ndarray
    gt_colors_ds: np.ndarray
    pr_ds: np.ndarray
    pr_colors_ds: np.ndarray
    chamfer_l1: float


@torch.no_grad()
def compute_set_metrics(
    batch_views: List[Dict[str, Any]],
    batch_idx: int,
    gt_info: Dict[str, Any],
    pr_info: Dict[str, Any],
    valid_masks: List[torch.Tensor],
    gt_info_abs: Dict[str, Any],
    pr_info_abs: Dict[str, Any],  # absolute pred already scaled (from get_all_info...)
    scale_factors: Dict[str, torch.Tensor],
    device: torch.device,
    # fused PC config
    voxel: float = 0.1,
    icp_iters: int = 20,
    trim_ratio: float = 0.8,
    sim3_trim_ratio: float = 0.8,
    sim3_iters: int = 2,
    max_samples_per_view_abs: int = 1000,
    return_fused_debug: bool = False,
    compute_abs_metrics: bool = False,
) -> Tuple[Dict[str, float], Optional[FusedPCDebug]]:
    """
    Returns:
      metrics: dict of scalar metrics for this multi-view set (already averaged across views where applicable)
      fused_debug: optional downsampled pcs to write ply outside
    """

    n_views = len(batch_views)

    # --- per-view accumulators (always computed) ---
    pointmaps_abs_rel_list = []
    pointmaps_inlier_103_list = []
    z_abs_rel_list = []
    z_inlier_103_list = []
    ray_err_deg_list = []
    gt_poses_curr_set = []
    pr_poses_curr_set = []

    # --- accumulators for absolute metrics (only if requested) ---
    if compute_abs_metrics:
        abs_point_mae_list = []
        abs_point_rmse_list = []
        abs_z_mae_list = []
        abs_z_rmse_list = []
        rgb_list_u8 = []
        gt_pts_list_abs = []
        pr_pts_list_abs = []
        masks_list = []
    else:
        # Placeholders to avoid NameError, but never used
        abs_point_mae_list = abs_point_rmse_list = abs_z_mae_list = abs_z_rmse_list = []
        rgb_list_u8 = gt_pts_list_abs = pr_pts_list_abs = masks_list = []

    for view_idx in range(n_views):
        valid_mask = valid_masks[view_idx][batch_idx].cpu().numpy().astype(bool)

        # -------- Relative dense metrics (always) --------
        pm_abs_rel = m_rel_ae(
            gt=gt_info["pts3d"][view_idx][batch_idx].numpy(),
            pred=pr_info["pts3d"][view_idx][batch_idx].numpy(),
            mask=valid_mask,
        )
        pm_inlier = thresh_inliers(
            gt=gt_info["pts3d"][view_idx][batch_idx].numpy(),
            pred=pr_info["pts3d"][view_idx][batch_idx].numpy(),
            mask=valid_mask,
            thresh=1.03,
        )
        z_abs_rel = m_rel_ae(
            gt=gt_info["z_depths"][view_idx][batch_idx].numpy(),
            pred=pr_info["z_depths"][view_idx][batch_idx].numpy(),
            mask=valid_mask,
        )
        z_inlier = thresh_inliers(
            gt=gt_info["z_depths"][view_idx][batch_idx].numpy(),
            pred=pr_info["z_depths"][view_idx][batch_idx].numpy(),
            mask=valid_mask,
            thresh=1.03,
        )

        pointmaps_abs_rel_list.append(float(pm_abs_rel))
        pointmaps_inlier_103_list.append(float(pm_inlier))
        z_abs_rel_list.append(float(z_abs_rel))
        z_inlier_103_list.append(float(z_inlier))

        # -------- Ray direction angular error (always) --------
        ray_dirs_l2 = torch.norm(
            gt_info["ray_directions"][view_idx][batch_idx] - pr_info["ray_directions"][view_idx][batch_idx],
            dim=-1,
        )
        ray_err_deg = l2_distance_of_unit_ray_directions_to_angular_error(ray_dirs_l2).mean().item()
        ray_err_deg_list.append(float(ray_err_deg))

        # -------- Poses (for ATE / AUC) always --------
        gt_poses_curr_set.append(gt_info["poses"][view_idx][batch_idx])
        pr_poses_curr_set.append(pr_info["poses"][view_idx][batch_idx])

        # -------- Absolute metrics (only if requested) --------
        if compute_abs_metrics:
            # points: Sim3 on this view only
            gt_pts_abs_v = gt_info_abs["pts3d"][view_idx][batch_idx].cpu().numpy()   # (H,W,3)
            pr_pts_abs_v = pr_info_abs["pts3d"][view_idx][batch_idx].cpu().numpy()   # (H,W,3)

            pr_corr_t, gt_corr_t = gather_correspondences_from_pointmaps(
                [pr_pts_abs_v], [gt_pts_abs_v], [valid_mask],
                device=device, dtype=torch.float32,
                max_samples_total=max_samples_per_view_abs,
            )
            s_v, R_v, t_v = sim3_from_correspondences_robust(
                pr_corr_t, gt_corr_t, trim_ratio=sim3_trim_ratio, iters=1
            )
            pr_pts_abs_v_al = apply_sim3_to_points_np(pr_pts_abs_v, s_v, R_v, t_v)

            e = np.linalg.norm(pr_pts_abs_v_al - gt_pts_abs_v, axis=-1)
            e_valid = e[valid_mask]
            if e_valid.size == 0:
                abs_point_mae_list.append(np.nan)
                abs_point_rmse_list.append(np.nan)
            else:
                abs_point_mae_list.append(float(np.mean(e_valid)))
                abs_point_rmse_list.append(float(np.sqrt(np.mean(e_valid ** 2))))

            # depth: affine a*z+b on this view only
            gt_z_abs_v = gt_info_abs["z_depths"][view_idx][batch_idx].cpu().numpy()[..., 0]  # (H,W)
            pr_z_abs_v = pr_info_abs["z_depths"][view_idx][batch_idx].cpu().numpy()[..., 0]  # (H,W)
            a, b = fit_affine_1d_np(pr_z_abs_v, gt_z_abs_v, valid_mask)
            pr_z_abs_v_al = a * pr_z_abs_v + b

            ez = np.abs(pr_z_abs_v_al - gt_z_abs_v)
            ez_valid = ez[valid_mask]
            if ez_valid.size == 0:
                abs_z_mae_list.append(np.nan)
                abs_z_rmse_list.append(np.nan)
            else:
                abs_z_mae_list.append(float(np.mean(ez_valid)))
                abs_z_rmse_list.append(float(np.sqrt(np.mean(ez_valid ** 2))))

            # -------- Collect fused PC inputs --------
            masks_list.append(valid_mask)
            gt_pts_list_abs.append(gt_pts_abs_v)
            pr_pts_list_abs.append(pr_pts_abs_v)

            rgb_list_u8.append(
                (rgb(batch_views[view_idx]["img"][batch_idx], batch_views[view_idx]["data_norm_type"][batch_idx]) * 255.0)
                .astype(np.uint8)
            )

    # =========================
    # Aggregate per-view metrics (always)
    # =========================
    metrics: Dict[str, float] = {}
    metrics["pointmaps_abs_rel"] = float(np.nanmean(pointmaps_abs_rel_list))
    metrics["pointmaps_inlier_thres_103"] = float(np.nanmean(pointmaps_inlier_103_list))
    metrics["z_depth_abs_rel"] = float(np.nanmean(z_abs_rel_list))
    metrics["z_depth_inlier_thres_103"] = float(np.nanmean(z_inlier_103_list))
    metrics["ray_dirs_err_deg"] = float(np.nanmean(ray_err_deg_list))

    # =========================
    # Pose metrics (always)
    # =========================
    pose_ate = evaluate_ate(gt_traj=gt_poses_curr_set, est_traj=pr_poses_curr_set)
    metrics["pose_ate_rmse"] = float(pose_ate)

    gt_poses_curr_set_t = torch.stack(gt_poses_curr_set)
    pr_poses_curr_set_t = torch.stack(pr_poses_curr_set)
    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
        pred_se3=pr_poses_curr_set_t,
        gt_se3=gt_poses_curr_set_t,
        num_frames=pr_poses_curr_set_t.shape[0],
    )
    rError = rel_rangle_deg.cpu().numpy()
    tError = rel_tangle_deg.cpu().numpy()
    pose_auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
    metrics["pose_auc_5"] = float(pose_auc_5 * 100.0)

    # =========================
    # Metric scale (always)
    # =========================
    if gt_info.get("metric_scale", None) is not None and pr_info.get("metric_scale", None) is not None:
        gt_metric_scale = gt_info["metric_scale"][batch_idx].detach().cpu().numpy()
        pr_metric_scale = pr_info["metric_scale"][batch_idx].detach().cpu().numpy()

        # robust to shape: (1,1,1,1) / (1,) / scalar
        gt_metric_scale = np.asarray(gt_metric_scale).squeeze()
        pr_metric_scale = np.asarray(pr_metric_scale).squeeze()

        # avoid div-by-zero; and reduce to scalar robustly
        denom = np.maximum(gt_metric_scale, 1e-12)
        val = np.abs(pr_metric_scale - gt_metric_scale) / denom
        metrics["metric_scale_abs_rel"] = float(np.nanmean(val))
    else:
        metrics["metric_scale_abs_rel"] = float("nan")
    
    # Save pr_to_gt_scale (computed in get_all_info...)
    metrics["pr_to_gt_scale"] = float(scale_factors["pr_to_gt_scale"][batch_idx].item())

    # =========================
    # Absolute metrics and fused pointcloud (only if requested)
    # =========================
    fused_debug: Optional[FusedPCDebug] = None
    if compute_abs_metrics:
        # Absolute point/depth errors
        metrics["pointmaps_abs_mae"] = float(np.nanmean(abs_point_mae_list))
        metrics["pointmaps_abs_rmse"] = float(np.nanmean(abs_point_rmse_list))
        metrics["z_depth_abs_mae"] = float(np.nanmean(abs_z_mae_list))
        metrics["z_depth_abs_rmse"] = float(np.nanmean(abs_z_rmse_list))

        # Absolute pose ATE
        gt_poses_abs_curr_set = [gt_info_abs["poses"][v][batch_idx] for v in range(n_views)]
        pr_poses_abs_curr_set = [pr_info_abs["poses"][v][batch_idx] for v in range(n_views)]
        pose_ate_abs = evaluate_ate(gt_traj=gt_poses_abs_curr_set, est_traj=pr_poses_abs_curr_set)
        metrics["pose_ate_abs"] = float(pose_ate_abs)

        # Fused pointcloud metrics (Sim3 + ICP + Chamfer)
        pr_corr_t, gt_corr_t = gather_correspondences_from_pointmaps(
            pr_pts_list_abs, gt_pts_list_abs, masks_list,
            device=device, dtype=torch.float32,
            max_samples_total=60000,
        )
        s0, R0, t0 = sim3_from_correspondences_robust(pr_corr_t, gt_corr_t, trim_ratio=trim_ratio, iters=sim3_iters)

        gt_merged_abs, gt_colors = merge_masked_points_list(gt_pts_list_abs, rgb_list_u8, masks_list)
        pr_merged_abs, pr_colors = merge_masked_points_list(pr_pts_list_abs, rgb_list_u8, masks_list)

        pr_merged_abs_t = torch.from_numpy(pr_merged_abs).to(device=device, dtype=torch.float32)
        pr_merged_abs_aligned_t = (s0 * (R0 @ pr_merged_abs_t.t())).t() + t0[None, :]
        pr_merged_abs_aligned = pr_merged_abs_aligned_t.detach().cpu().numpy()

        gt_ds, gt_colors_ds = voxel_downsample_np(gt_merged_abs, gt_colors, voxel)
        pr_ds, pr_colors_ds = voxel_downsample_np(pr_merged_abs_aligned, pr_colors, voxel)

        gt_t = torch.from_numpy(gt_ds).to(device=device, dtype=torch.float32)
        pr_t = torch.from_numpy(pr_ds).to(device=device, dtype=torch.float32)

        icp_gate = voxel * 3.0
        max_icp_points = 3000
        max_chamfer_points = 2000

        gt_t_icp = random_sample_points_torch(gt_t, max_icp_points)
        pr_t_icp = random_sample_points_torch(pr_t, max_icp_points)

        R1, t1, pr_refined = icp_se3_torch(
            pr_t_icp, gt_t_icp,
            iters=icp_iters,
            max_corr_dist=icp_gate,
            nn_src_chunk=2048,
            nn_dst_chunk=2048,
            max_src_corr=1500,
            trimmed_ratio=trim_ratio,
        )

        inlier_dist = voxel * 2.0
        abs_chamfer_l1, abs_chamfer_rmse, abs_inlier_ratio = chamfer_inlier_torch(
            pr_refined, gt_t_icp,
            inlier_dist,
            nn_src_chunk=2048,
            nn_dst_chunk=2048,
            max_eval=max_chamfer_points,
        )

        metrics["merged_pc_abs_chamfer_l1"] = float(abs_chamfer_l1) if np.isfinite(abs_chamfer_l1) else float("nan")
        metrics["merged_pc_abs_chamfer_rmse"] = float(abs_chamfer_rmse) if np.isfinite(abs_chamfer_rmse) else float("nan")
        metrics["merged_pc_abs_inlier_ratio"] = float(abs_inlier_ratio) if np.isfinite(abs_inlier_ratio) else float("nan")

        if return_fused_debug:
            fused_debug = FusedPCDebug(
                gt_ds=gt_ds,
                gt_colors_ds=gt_colors_ds,
                pr_ds=pr_ds,
                pr_colors_ds=pr_colors_ds,
                chamfer_l1=float(abs_chamfer_l1),
            )

    return metrics, fused_debug

