from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _flatten_desc(desc: torch.Tensor) -> torch.Tensor:
    # [B, C, H, W] -> [B, HW, C]
    return desc.flatten(2).transpose(1, 2).contiguous()


def dual_softmax_similarity(
    desc0: torch.Tensor,
    desc1: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Args:
        desc0, desc1: [B, C, H, W] normalized descriptor maps.
    Returns:
        prob matrix [B, HW0, HW1]
    """
    f0 = _flatten_desc(desc0)
    f1 = _flatten_desc(desc1)
    sim = torch.matmul(f0, f1.transpose(-1, -2)) / temperature
    return torch.softmax(sim, dim=-1) * torch.softmax(sim, dim=-2)


def mutual_nearest_matches(
    prob: torch.Tensor,
    threshold: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    Args:
        prob: [B, HW0, HW1] dual-softmax confidence matrix
    Returns:
        dict with batch_idx, idx0, idx1, score
    """
    best01_score, best01_idx = prob.max(dim=-1)
    best10_score, best10_idx = prob.max(dim=-2)

    batch_ids = []
    idx0_all = []
    idx1_all = []
    score_all = []
    for b in range(prob.shape[0]):
        idx0 = torch.arange(prob.shape[1], device=prob.device)
        idx1 = best01_idx[b]
        mutual = best10_idx[b, idx1] == idx0
        keep = mutual & (best01_score[b] >= threshold)
        if keep.any():
            batch_ids.append(torch.full((int(keep.sum()),), b, device=prob.device))
            idx0_all.append(idx0[keep])
            idx1_all.append(idx1[keep])
            score_all.append(best01_score[b, keep])

    if len(batch_ids) == 0:
        empty = torch.empty(0, device=prob.device, dtype=torch.long)
        return {"batch_idx": empty, "idx0": empty, "idx1": empty, "score": empty.float()}

    return {
        "batch_idx": torch.cat(batch_ids, dim=0),
        "idx0": torch.cat(idx0_all, dim=0),
        "idx1": torch.cat(idx1_all, dim=0),
        "score": torch.cat(score_all, dim=0),
    }


def idx_to_xy(idx: torch.Tensor, width: int) -> torch.Tensor:
    x = idx % width
    y = idx // width
    return torch.stack([x, y], dim=-1)


def local_window_refine(
    query_feat: torch.Tensor,
    target_feat: torch.Tensor,
    center_xy: torch.Tensor,
    window_radius: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lightweight Efficient LoFTR-style local refinement.

    Args:
        query_feat: [N, C]
        target_feat: [N, C, H, W]
        center_xy: [N, 2] integer centers on target_feat grid
    Returns:
        refined_xy: [N, 2] float refined coordinate on target_feat grid
        local_score: [N]
    """
    n, _, h, w = target_feat.shape
    device = target_feat.device
    offsets = torch.arange(-window_radius, window_radius + 1, device=device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    delta = torch.stack([dx, dy], dim=-1).view(1, -1, 2)

    centers = center_xy.view(n, 1, 2)
    coords = centers + delta
    coords[..., 0] = coords[..., 0].clamp(0, w - 1)
    coords[..., 1] = coords[..., 1].clamp(0, h - 1)

    linear_idx = coords[..., 1] * w + coords[..., 0]
    flat_target = target_feat.flatten(2).transpose(1, 2).contiguous()
    gathered = flat_target.gather(
        1, linear_idx.unsqueeze(-1).expand(-1, -1, flat_target.shape[-1])
    )
    logits = torch.einsum("nc,nkc->nk", query_feat, gathered)
    prob = F.softmax(logits, dim=-1)
    refined_xy = torch.sum(prob.unsqueeze(-1) * coords.float(), dim=1)
    local_score = prob.max(dim=-1).values
    return refined_xy, local_score
