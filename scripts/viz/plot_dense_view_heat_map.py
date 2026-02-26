#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def find_per_dataset_results(search_dir: Path) -> Optional[Path]:
    """Locate <search_dir>/per_dataset_results.json"""
    if not search_dir.is_dir():
        return None
    p = search_dir / "per_dataset_results.json"
    return p if p.is_file() else None


def sanitize_filename(s: str) -> str:
    """Make a safe filename fragment."""
    s = s.strip()
    s = s.replace("/", "_").replace("\\", "_").replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_\-@.]+", "", s)
    return s[:180] if len(s) > 180 else s


def setup_paper_style() -> None:
    """Clean, paper-like matplotlib style."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def pretty_metric_name(metric_key: str) -> str:
    """Metric title with ↑/↓ hint."""
    higher_is_better = {
        "pointmaps_inlier_thres_103": True,
        "pose_auc_5": True,
        "z_depth_inlier_thres_103": True,
        "merged_pc_abs_inlier_ratio": True,
    }
    mapping = {
        "metric_scale_abs_rel": "Scale AbsRel",
        "pointmaps_abs_rel": "PointMaps AbsRel",
        "pointmaps_inlier_thres_103": "PointMaps Inlier@1e-3",
        "pose_ate_rmse": "Pose ATE RMSE",
        "pose_auc_5": "Pose AUC@5°",
        "z_depth_abs_rel": "Z-Depth AbsRel",
        "z_depth_inlier_thres_103": "Z-Depth Inlier@1e-3",
        "ray_dirs_err_deg": "RayDirs Err (deg)",
        "pointmaps_abs_mae": "PointMaps MAE",
        "pointmaps_abs_rmse": "PointMaps RMSE",
        "z_depth_abs_mae": "Z-Depth MAE",
        "z_depth_abs_rmse": "Z-Depth RMSE",
        "pose_ate_abs": "Pose ATE Abs",
        "merged_pc_abs_chamfer_l1": "ChamferL1",
        "merged_pc_abs_chamfer_rmse": "ChamferRMSE",
        "merged_pc_abs_inlier_ratio": "InlierRatio",
        "pr_to_gt_scale": "PR/GT Scale",
    }
    base = mapping.get(metric_key, metric_key.replace("_", " "))
    arrow = "↑" if higher_is_better.get(metric_key, False) else "↓"
    return f"{base} {arrow}"


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class MethodCfg:
    subdir: str


def build_methods_cfg() -> "OrderedDict[str, MethodCfg]":
    """
    Keep the displayed method order stable.
    Modify subdir names to match your filesystem.
    """
    return OrderedDict(
        [
            ("MapAnything", MethodCfg(subdir="mapa_24v")),
            ("VGGT",        MethodCfg(subdir="vggt")),
            ("Pi3",         MethodCfg(subdir="pi3")),
            ("Pi3X",        MethodCfg(subdir="pi3x")),
            ("DA3",         MethodCfg(subdir="da3")),
            ("HunYuan",     MethodCfg(subdir="hunyuan")),
            ("UAVMapa",     MethodCfg(subdir="uav_mapa")),
        ]
    )


# -----------------------------
# Core: heatmap for one dense_n_view
# -----------------------------
def load_metric_matrix_from_dense_view(
    dense_view_dir: Path,
    methods: "OrderedDict[str, MethodCfg]",
    metric_key: str,
    include_average: bool,
) -> tuple[list[str], list[str], "list[list[float]]"]:
    """
    Returns (dataset_keys, method_labels, matrix)
      - rows: dataset_keys
      - cols: method_labels
      - values: float or NaN
    """
    if not dense_view_dir.is_dir():
        raise FileNotFoundError(f"dense_view_dir not found: {dense_view_dir}")

    method_to_dataset_val: dict[str, dict[str, float]] = {}
    all_dataset_keys: set[str] = set()

    # read each method json
    for method_label, cfg in methods.items():
        jf = find_per_dataset_results(dense_view_dir / cfg.subdir)
        if jf is None:
            continue

        try:
            obj = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        dvals: dict[str, float] = {}
        for dkey, dobj in obj.items():
            if not isinstance(dobj, dict):
                continue
            if (not include_average) and (dkey == "Average"):
                continue
            if metric_key not in dobj:
                continue
            try:
                dvals[dkey] = float(dobj[metric_key])
                all_dataset_keys.add(dkey)
            except Exception:
                pass

        if dvals:
            method_to_dataset_val[method_label] = dvals

    if not method_to_dataset_val:
        raise RuntimeError(
            f"No usable per_dataset_results.json with metric='{metric_key}' under: {dense_view_dir}"
        )

    # dataset order: Average last
    dataset_keys = sorted([k for k in all_dataset_keys if k != "Average"])
    if include_average and ("Average" in all_dataset_keys):
        dataset_keys.append("Average")

    # method order: keep config order but only those present
    method_labels = [m for m in methods.keys() if m in method_to_dataset_val]

    import numpy as np

    mat = np.full((len(dataset_keys), len(method_labels)), np.nan, dtype=np.float64)
    for i, dkey in enumerate(dataset_keys):
        for j, m in enumerate(method_labels):
            v = method_to_dataset_val.get(m, {}).get(dkey)
            if v is not None:
                mat[i, j] = v

    return dataset_keys, method_labels, mat.tolist()


def plot_dense_view_metric_heatmap(
    dense_view_dir: Path,
    methods: "OrderedDict[str, MethodCfg]",
    metric_key: str = "metric_scale_abs_rel",
    out_dir: Optional[Path] = None,
    out_name: Optional[str] = None,
    include_average: bool = True,
    annotate: bool = True,
    figsize=(10.5, 6.8),
) -> tuple[Path, Path]:
    """
    Heatmap visualization for one metric across methods and datasets
    within ONE dense_n_view directory.
    """
    if out_dir is None:
        out_dir = dense_view_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_name is None:
        out_name = f"heatmap__{sanitize_filename(dense_view_dir.name)}__{sanitize_filename(metric_key)}"

    dataset_keys, method_labels, mat_list = load_metric_matrix_from_dense_view(
        dense_view_dir=dense_view_dir,
        methods=methods,
        metric_key=metric_key,
        include_average=include_average,
    )

    import numpy as np

    mat = np.array(mat_list, dtype=np.float64)
    mat_masked = np.ma.masked_invalid(mat)

    setup_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(mat_masked, aspect="auto")

    # ticks & labels
    ax.set_xticks(np.arange(len(method_labels)))
    ax.set_xticklabels(method_labels, rotation=35, ha="right")

    ax.set_yticks(np.arange(len(dataset_keys)))
    ax.set_yticklabels(dataset_keys)

    ax.set_title(f"{dense_view_dir.name}: {pretty_metric_name(metric_key)}")

    # subtle grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(method_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(dataset_keys), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.set_ylabel(metric_key, rotation=90)

    # annotate cell values (optional)
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                if abs(v) < 1:
                    txt = f"{v:.4f}"
                elif abs(v) < 10:
                    txt = f"{v:.3f}"
                else:
                    txt = f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    fig.tight_layout()

    out_png = out_dir / f"{out_name}.png"
    out_pdf = out_dir / f"{out_name}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    # fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return out_png, out_pdf


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a single metric across methods and datasets for ONE dense_n_view directory."
    )
    parser.add_argument(
        "--dense_view_dir",
        type=str,
        default="experiments/mapanything/benchmarking/dense_16_view",
        help="Path to dense_n_view directory, e.g. experiments/mapanything/benchmarking/dense_2_view",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="metric_scale_abs_rel",
        help="Metric key in per_dataset_results.json, e.g. metric_scale_abs_rel / pose_auc_5 / z_depth_abs_mae",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/mapanything/benchmarking/figs_per_dataset_metric_scale",
        help="Output directory to save figures. Default: <dense_view_dir>/figs",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="",
        help="Output base name (no extension). Default: auto-generated.",
    )
    parser.add_argument(
        "--no_average",
        action="store_true",
        help="Exclude 'Average' row.",
    )
    parser.add_argument(
        "--no_annotate",
        action="store_true",
        help="Do not annotate numeric values in each cell (cleaner).",
    )
    args = parser.parse_args()

    dense_view_dir = Path(args.dense_view_dir)
    methods = build_methods_cfg()

    out_dir = Path(args.output) if args.output.strip() else None
    out_name = args.out_name.strip() if args.out_name.strip() else None

    out_png, out_pdf = plot_dense_view_metric_heatmap(
        dense_view_dir=dense_view_dir,
        methods=methods,
        metric_key=args.metric,
        out_dir=out_dir,
        out_name=out_name,
        include_average=(not args.no_average),
        annotate=(not args.no_annotate),
    )

    print(f"[OK] Saved heatmap:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    main()