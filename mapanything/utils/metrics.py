# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Utils for Metrics
Source for Pose AUC Metrics: VGGT
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def l2_distance_of_unit_quats_to_angular_error(l2_distance):
    """
    Converts a given L2 distance (for unit quaternions) to the angular error in degrees.
    For two quaternions differing by an angle θ the relationship is:
    L2 distance = 2 * sin(θ/4)
    Hence, the angular error in degrees is computed as:
    4 * asin(l2_distance / 2) * (180/π)

    Args:
        l2_distance: L2 distance between two unit quaternions (torch.Tensor, shape: (N,))
    Returns:
        angular_error_degrees: Angular error in degrees (torch.Tensor, shape: (N,))
    """
    angular_error_radians = 4 * torch.asin(l2_distance / 2)
    angular_error_degrees = angular_error_radians * 180.0 / math.pi

    return angular_error_degrees


def l2_distance_of_unit_ray_directions_to_angular_error(l2_distance):
    """
    Converts a given L2 distance (for unit ray directions) to the angular error in degrees.
    For two unit ray directions differing by an angle θ the relationship is:
    L2 distance = 2 * sin(θ/2)
    Hence, the angular error in degrees is computed as:
    2 * asin(l2_distance / 2) * (180/π)

    Args:
        l2_distance: L2 distance between two unit ray directions (torch.Tensor, shape: (N,))
    Returns:
        angular_error_degrees: Angular error in degrees (torch.Tensor, shape: (N,))
    """
    angular_error_radians = 2 * torch.asin(l2_distance / 2)
    angular_error_degrees = angular_error_radians * 180.0 / math.pi

    return angular_error_degrees


def valid_mean(arr, mask, axis=None, keepdims=np._NoValue):
    """Compute mean of elements across given dimensions of an array, considering only valid elements.

    Args:
        arr: The array to compute the mean.
        mask: Array with numerical or boolean values for element weights or validity. For bool, False means invalid.
        axis: Dimensions to reduce.
        keepdims: If true, retains reduced dimensions with length 1.

    Returns:
        Mean array/scalar and a valid array/scalar that indicates where the mean could be computed successfully.
    """

    mask = mask.astype(arr.dtype) if mask.dtype == bool else mask
    num_valid = np.sum(mask, axis=axis, keepdims=keepdims)
    masked_arr = arr * mask
    masked_arr_sum = np.sum(masked_arr, axis=axis, keepdims=keepdims)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid_mean = masked_arr_sum / num_valid
        is_valid = np.isfinite(valid_mean)
        valid_mean = np.nan_to_num(valid_mean, nan=0, posinf=0, neginf=0)

    return valid_mean, is_valid


def thresh_inliers(gt, pred, thresh=1.03, mask=None, output_scaling_factor=1.0):
    """Computes the inlier (=error within a threshold) ratio for a predicted and ground truth dense map of size H x W x C.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        thresh: Threshold for the relative difference between the prediction and ground truth. Default: 1.03
        mask: Array of shape HxW with boolean values to indicate validity. For bool, False means invalid. Default: None
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]). Default: 1

    Returns:
        Scalar that indicates the inlier ratio. Scalar is np.nan if the result is invalid.
    """
    # Compute the norms
    gt_norm = np.linalg.norm(gt, axis=-1)
    pred_norm = np.linalg.norm(pred, axis=-1)

    gt_norm_valid = (gt_norm) > 0
    if mask is not None:
        combined_mask = mask & gt_norm_valid
    else:
        combined_mask = gt_norm_valid

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_1 = np.nan_to_num(
            gt_norm / pred_norm, nan=thresh + 1, posinf=thresh + 1, neginf=thresh + 1
        )  # pred=0 should be an outlier
        rel_2 = np.nan_to_num(
            pred_norm / gt_norm, nan=0, posinf=0, neginf=0
        )  # gt=0 is masked out anyways

    max_rel = np.maximum(rel_1, rel_2)
    inliers = ((0 < max_rel) & (max_rel < thresh)).astype(
        np.float32
    )  # 1 for inliers, 0 for outliers

    inlier_ratio, valid = valid_mean(inliers, combined_mask)

    inlier_ratio = inlier_ratio * output_scaling_factor
    inlier_ratio = inlier_ratio if valid else np.nan

    return inlier_ratio


def m_rel_ae(gt, pred, mask=None, output_scaling_factor=1.0):
    """Computes the mean-relative-absolute-error for a predicted and ground truth dense map of size HxWxC.

    Args:
        gt: Ground truth map as numpy array of shape H x W x C.
        pred: Predicted map as numpy array of shape H x W x C.
        mask: Array of shape HxW with boolean values to indicate validity. For bool, False means invalid. Default: None
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]). Default: 1

    Returns:
        Scalar that indicates the mean-relative-absolute-error. Scalar is np.nan if the result is invalid.
    """
    error_norm = np.linalg.norm(pred - gt, axis=-1)
    gt_norm = np.linalg.norm(gt, axis=-1)

    gt_norm_valid = (gt_norm) > 0
    if mask is not None:
        combined_mask = mask & gt_norm_valid
    else:
        combined_mask = gt_norm_valid

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_ae = np.nan_to_num(error_norm / gt_norm, nan=0, posinf=0, neginf=0)

    m_rel_ae, valid = valid_mean(rel_ae, combined_mask)

    m_rel_ae = m_rel_ae * output_scaling_factor
    m_rel_ae = m_rel_ae if valid else np.nan

    return m_rel_ae


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Args:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

    Returns:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

    """
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
    """
    Input :
        gt_traj: list of 4x4 matrices
        est_traj: list of 4x4 matrices
        len(gt_traj) == len(est_traj)
    """
    gt_traj_pts = [gt_traj[idx][:3, 3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3, 3] for idx in range(len(est_traj))]

    gt_traj_pts = torch.stack(gt_traj_pts).detach().cpu().numpy().T
    est_traj_pts = torch.stack(est_traj_pts).detach().cpu().numpy().T

    _, _, trans_error = align(gt_traj_pts, est_traj_pts)

    avg_trans_error = trans_error.mean()

    return avg_trans_error


def build_pair_index(N, B=1):
    """
    Build indices for all possible pairs of frames.

    Args:
        N: Number of frames
        B: Batch size

    Returns:
        i1, i2: Indices for all possible pairs
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

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

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))  # pylint: disable=not-callable

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using NumPy.

    Args:
        r_error: numpy array representing R error values (Degree)
        t_error: numpy array representing T error values (Degree)
        max_threshold: Maximum threshold value for binning the histogram

    Returns:
        AUC value and the normalized histogram
    """
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.

    Args:
        pred_se3: Predicted SE(3) transformations
        gt_se3: Ground truth SE(3) transformations
        num_frames: Number of frames

    Returns:
        Rotation and translation angle errors in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    # Compute relative camera poses between pairs
    # We use closed_form_inverse to avoid potential numerical loss by torch.inverse()
    relative_pose_gt = closed_form_inverse_se3(gt_se3[pair_idx_i1]).bmm(
        gt_se3[pair_idx_i2]
    )
    relative_pose_pred = closed_form_inverse_se3(pred_se3[pair_idx_i1]).bmm(
        pred_se3[pair_idx_i2]
    )

    # Compute the difference in rotation and translation
    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg


# -------------------------------------------------------------- #
##### Point Cloud Utils #####
def merge_masked_points_list(pts_list_np, masks_list_np):
    """
    pts_list_np: list of (H,W,3) or (N,3)
    masks_list_np: list of (H,W) boolean
    return: (M,3) float32
    """
    all_pts = []
    for pts, m in zip(pts_list_np, masks_list_np):
        if pts is None:
            continue
        if pts.ndim == 3:
            # H,W,3 + H,W mask
            valid = m.astype(bool)
            p = pts[valid].reshape(-1, 3)
        else:
            # N,3 + N mask (if any)
            if m is None:
                p = pts.reshape(-1, 3)
            else:
                p = pts[m.astype(bool)].reshape(-1, 3)
        if p.size > 0:
            all_pts.append(p)
    if len(all_pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(all_pts, axis=0).astype(np.float32)

def voxel_downsample_np(points, voxel_size):
    """
    Simple voxel grid downsample: keep one point per voxel (centroid).
    points: (N,3) float
    voxel_size: float > 0
    return: (M,3)
    """
    if points.shape[0] == 0:
        return points
    if voxel_size <= 0:
        return points

    # compute voxel index
    grid = np.floor(points / voxel_size).astype(np.int64)
    # unique voxels
    uniq, inv = np.unique(grid, axis=0, return_inverse=True)

    # accumulate centroids
    counts = np.bincount(inv)
    out = np.zeros((uniq.shape[0], 3), dtype=np.float32)
    np.add.at(out, inv, points)
    out /= counts[:, None].astype(np.float32)
    return out

def estimate_voxel_size(points_np: np.ndarray, max_samples: int = 1500):
    """
    points_np: (N,3)
    Return a robust voxel size in normalized units.
    Use median nearest-neighbor distance on a small random subset.
    """
    N = points_np.shape[0]
    s = min(N, max_samples)
    idx = np.random.choice(N, s, replace=False)
    P = points_np[idx].astype(np.float32)

    # O(s^2) on small s
    diff = P[:, None, :] - P[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return None
    return float(np.median(nn))


@torch.no_grad()
def _nn_distance_torch(src: torch.Tensor, dst: torch.Tensor, chunk: int = 4096):
    Ns = src.shape[0]
    best_d = torch.empty((Ns,), device=src.device, dtype=src.dtype)
    best_j = torch.empty((Ns,), device=src.device, dtype=torch.long)
    for i in range(0, Ns, chunk):
        s = src[i:i+chunk]
        d = torch.cdist(s, dst, p=2)          # (c,Nd) L2
        md, mj = torch.min(d, dim=1)
        best_d[i:i+chunk] = md
        best_j[i:i+chunk] = mj
    return best_d, best_j


@torch.no_grad()
def kabsch_se3_torch(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-9):
    """
    Solve R,t minimizing || R A + t - B ||^2, A,B are corresponding (N,3)
    """
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

    # fix reflection
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
    nn_chunk: int = 4096,
    max_src_corr: int = 30000,
    trimmed_ratio: float | None = 0.8,
):
    """
    SE3 ICP: nearest-neighbor correspondences + Kabsch update.
    Optional trimmed ICP: keep the closest trimmed_ratio correspondences for robustness.
    """
    device = src.device
    dtype = src.dtype

    R_tot = torch.eye(3, device=device, dtype=dtype)
    t_tot = torch.zeros(3, device=device, dtype=dtype)

    X = src
    for _ in range(iters):
        # subsample correspondences for speed
        if X.shape[0] > max_src_corr:
            idx = torch.randperm(X.shape[0], device=device)[:max_src_corr]
            Xc = X[idx]
        else:
            Xc = X

        d, j = _nn_distance_torch(Xc, dst, chunk=nn_chunk)
        Yc = dst[j]

        # correspondence gating
        if max_corr_dist is not None:
            keep = d < max_corr_dist
            if keep.sum() < 50:
                break
            Xk = Xc[keep]
            Yk = Yc[keep]
            dk = d[keep]
        else:
            Xk, Yk, dk = Xc, Yc, d

        # trimmed ICP (robust to outliers / missing areas)
        if trimmed_ratio is not None and 0 < trimmed_ratio < 1:
            m = Xk.shape[0]
            k_keep = max(int(m * trimmed_ratio), 50)
            # keep smallest distances
            _, order = torch.topk(dk, k=k_keep, largest=False)
            A = Xk[order]
            B = Yk[order]
        else:
            A, B = Xk, Yk

        dR, dt = kabsch_se3_torch(A, B)

        # compose
        R_tot = dR @ R_tot
        t_tot = dR @ t_tot + dt

        # update all points
        X = (dR @ X.t()).t() + dt[None, :]

    return R_tot, t_tot, X

@torch.no_grad()
def umeyama_sim3_torch(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-9):
    """
    Solve s,R,t minimizing || s R A + t - B ||^2
    """
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
def sim3_init_by_nn_umeyama(
    src: torch.Tensor,
    dst: torch.Tensor,
    nn_chunk: int = 4096,
    max_src: int = 20000,
    max_corr_dist: float | None = None,
):
    """
    1) subsample src
    2) NN match to dst
    3) Umeyama to get s,R,t
    """
    if src.shape[0] > max_src:
        idx = torch.randperm(src.shape[0], device=src.device)[:max_src]
        A = src[idx]
    else:
        A = src

    d, j = _nn_distance_torch(A, dst, chunk=nn_chunk)
    B = dst[j]

    if max_corr_dist is not None:
        keep = d < max_corr_dist
        if keep.sum() >= 50:
            A = A[keep]
            B = B[keep]

    s, R, t = umeyama_sim3_torch(A, B)
    src_aligned = (s * (R @ src.t())).t() + t[None, :]
    return s, R, t, src_aligned


@torch.no_grad()
def chamfer_inlier_torch(pred: torch.Tensor, gt: torch.Tensor, inlier_dist: float,
                         nn_chunk: int = 4096, max_eval: int = 20000):
    if pred.shape[0] > max_eval:
        pred = pred[torch.randperm(pred.shape[0], device=pred.device)[:max_eval]]
    if gt.shape[0] > max_eval:
        gt = gt[torch.randperm(gt.shape[0], device=gt.device)[:max_eval]]

    d_pg, _ = _nn_distance_torch(pred, gt, chunk=nn_chunk)
    d_gp, _ = _nn_distance_torch(gt, pred, chunk=nn_chunk)

    chamfer_l1 = (d_pg.mean() + d_gp.mean()).item()
    chamfer_rmse = (torch.sqrt((d_pg*d_pg).mean()) + torch.sqrt((d_gp*d_gp).mean())).item()
    inlier_ratio = (d_pg < inlier_dist).float().mean().item()
    return chamfer_l1, chamfer_rmse, inlier_ratio

