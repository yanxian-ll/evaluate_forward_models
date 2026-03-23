#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Headless Rerun exporter for benchmark bundle json + fused ply point clouds.

Use case:
- Run on a remote server without desktop/GUI.
- Save visualization data to a .rrd file.
- Copy the .rrd file back to your local machine and open it with the Rerun viewer.

Example (remote):
python visualize_benchmark_ply_cams_headless.py \
    --bundle_json /path/to/scene_set000_bundle.json \
    --pc_mode both \
    --cams both \
    --save_rrd /path/to/output/scene_set000.rrd

Example (local):
rerun /path/to/scene_set000.rrd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rerun as rr

try:
    import open3d as o3d
except Exception as e:  # pragma: no cover
    raise ImportError(
        "This script requires open3d to read .ply point clouds. Please install open3d."
    ) from e


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_np(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float64)
    return arr


def validate_c2w(mat: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mat is None:
        return None
    if mat.shape != (4, 4):
        return None
    return mat


def infer_hw_from_intrinsics(
    K: Optional[np.ndarray], default_w: int, default_h: int
) -> Tuple[int, int]:
    if K is None or K.shape != (3, 3):
        return default_h, default_w
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    w = int(round(max(default_w, 2.0 * cx))) if cx > 1e-6 else default_w
    h = int(round(max(default_h, 2.0 * cy))) if cy > 1e-6 else default_h
    return h, w


# -----------------------------------------------------------------------------
# PLY helpers
# -----------------------------------------------------------------------------
def load_ply_xyzrgb(path: str | Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise RuntimeError(f"Failed to read point cloud or empty file: {path}")

    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = None
    if pcd.has_colors():
        cols_f = np.asarray(pcd.colors, dtype=np.float32)
        cols = np.clip(cols_f * 255.0, 0, 255).astype(np.uint8)
    return pts, cols


def collect_point_cloud_paths(
    bundle: dict,
    pc_mode: str,
    custom_paths: Optional[Sequence[str]],
) -> List[Tuple[str, str]]:
    fused = bundle.get("fused_outputs", {})
    items: List[Tuple[str, str]] = []

    if pc_mode == "gt":
        path = fused.get("gt_ply", None)
        if path:
            items.append(("gt", path))
    elif pc_mode == "pred":
        path = fused.get("pred_ply", None)
        if path:
            items.append(("pred", path))
    elif pc_mode == "both":
        gt_path = fused.get("gt_ply", None)
        pred_path = fused.get("pred_ply", None)
        if gt_path:
            items.append(("gt", gt_path))
        if pred_path:
            items.append(("pred", pred_path))
    elif pc_mode == "custom":
        if not custom_paths:
            raise ValueError("--pc_mode custom requires --point_clouds")
        for i, p in enumerate(custom_paths):
            items.append((f"custom_{i}", p))
    elif pc_mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported pc_mode: {pc_mode}")

    return items


# -----------------------------------------------------------------------------
# Camera helpers
# -----------------------------------------------------------------------------
def iter_cameras(
    bundle: dict, which: str
) -> Iterable[Tuple[str, int, str, Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Yields:
      (cam_type, view_idx, instance, c2w, intrinsics)
    cam_type in {"gt", "pred"}
    """
    views = bundle.get("views", [])
    for view in views:
        view_idx = int(view.get("view_idx", -1))
        instance = str(view.get("instance", ""))

        if which in ("gt", "both"):
            gt_cam = view.get("gt_cam", {})
            c2w = validate_c2w(as_np(gt_cam.get("c2w", None)))
            intr = as_np(gt_cam.get("intrinsics", None))
            yield "gt", view_idx, instance, c2w, intr

        if which in ("pred", "both"):
            pred_cam = view.get("pred_cam", {})
            c2w = validate_c2w(as_np(pred_cam.get("c2w", None)))
            intr = as_np(pred_cam.get("intrinsics", None))
            yield "pred", view_idx, instance, c2w, intr


# -----------------------------------------------------------------------------
# Rerun logging
# -----------------------------------------------------------------------------
def log_point_cloud_to_rerun(entity_path: str, pts: np.ndarray, colors: Optional[np.ndarray]) -> None:
    if colors is None:
        rr.log(entity_path, rr.Points3D(positions=pts))
    else:
        rr.log(entity_path, rr.Points3D(positions=pts, colors=colors))


def log_camera_to_rerun(
    entity_base: str,
    c2w: np.ndarray,
    intrinsics: Optional[np.ndarray],
    default_w: int,
    default_h: int,
) -> None:
    rr.log(
        entity_base,
        rr.Transform3D(
            translation=c2w[:3, 3],
            mat3x3=c2w[:3, :3],
        ),
    )

    if intrinsics is not None and intrinsics.shape == (3, 3):
        h, w = infer_hw_from_intrinsics(intrinsics, default_w=default_w, default_h=default_h)
        rr.log(
            f"{entity_base}/pinhole",
            rr.Pinhole(
                image_from_camera=intrinsics,
                height=h,
                width=w,
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export benchmark bundle json visualization into a headless RRD file"
    )
    parser.add_argument("--bundle_json", type=str, help="Path to *_bundle.json",
                        default="experiments/mapanything/benchmarking/dense_24_view/uav_mapa/ 150 @ A3DSynLargeWAI_fused_ply/3ad09ff6c7a7568f0563dce4/3ad09ff6c7a7568f0563dce4_set000_bundle.json" )
    parser.add_argument(
        "--pc_mode",
        type=str,
        default="both",
        choices=["gt", "pred", "both", "custom", "none"],
        help="Which point clouds to visualize",
    )
    parser.add_argument(
        "--point_clouds",
        type=str,
        nargs="*",
        default=None,
        help="Custom point clouds when --pc_mode custom",
    )
    parser.add_argument(
        "--cams",
        type=str,
        default="both",
        choices=["gt", "pred", "both", "none"],
        help="Which camera poses to visualize",
    )
    parser.add_argument(
        "--default_width",
        type=int,
        default=640,
        help="Fallback image width for camera frustums when intrinsics cannot infer image size",
    )
    parser.add_argument(
        "--default_height",
        type=int,
        default=480,
        help="Fallback image height for camera frustums when intrinsics cannot infer image size",
    )
    parser.add_argument(
        "--time_sequence",
        type=int,
        default=0,
        help="Rerun stable_time sequence value",
    )
    parser.add_argument(
        "--save_rrd",
        type=str,
        default="./output/uav_mapa_synl_16.rrd",
        help="Output .rrd file path",
    )
    parser.add_argument(
        "--app_id",
        type=str,
        default="visualize_benchmark_ply_cams",
        help="Rerun application ID",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_json(args.bundle_json)

    save_path = Path(args.save_rrd)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init(args.app_id, spawn=False)
    rr.save(str(save_path))
    rr.set_time("stable_time", sequence=args.time_sequence)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # ------------------------------------------------------------------
    # Point clouds
    # ------------------------------------------------------------------
    pc_items = collect_point_cloud_paths(bundle, args.pc_mode, args.point_clouds)
    if not pc_items and args.cams == "none":
        raise RuntimeError("Nothing to visualize: both point clouds and cameras are disabled.")

    for label, path in pc_items:
        pts, colors = load_ply_xyzrgb(path)
        if label == "gt":
            entity = "world/gt/points"
        elif label == "pred":
            entity = "world/pred/points"
        else:
            entity = f"world/{label}/points"
        log_point_cloud_to_rerun(entity, pts, colors)
        print(f"[Info] Logged point cloud [{label}] from: {path}")

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------
    if args.cams != "none":
        for cam_type, view_idx, instance, c2w, intrinsics in iter_cameras(bundle, args.cams):
            if c2w is None:
                print(f"[Warn] Skip {cam_type} camera for view {view_idx}: invalid c2w")
                continue

            base = f"world/{cam_type}/cameras/view_{view_idx:03d}"
            log_camera_to_rerun(
                entity_base=base,
                c2w=c2w,
                intrinsics=intrinsics,
                default_w=args.default_width,
                default_h=args.default_height,
            )
            print(f"[Info] Logged {cam_type} camera view {view_idx}: {instance}")

    print(f"[Done] Saved RRD to: {save_path}")
    print("[Next] Copy this .rrd file to your local machine and open it with: rerun <file.rrd>")


if __name__ == "__main__":
    main()
