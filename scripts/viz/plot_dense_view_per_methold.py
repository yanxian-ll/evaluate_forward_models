#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ONE dense_n_view → ONE figure (recommended for papers)

Goal:
- Input a dense_n_view directory (e.g. .../dense_16_view)
- Read per_dataset_results.json for each method under it
- Visualize the selected 8 metrics across datasets, with ALL methods in ONE figure

Key design (top-tier, rigorous):
- 8 subplots (2x4), each subplot = ONE metric
- X-axis = datasets (including "Average" optionally)
- Within each subplot: grouped bars (methods) for that metric
  -> This is the cleanest SOTA-style comparison when n is fixed and you compare methods.
- Handle long dataset labels via horizontal layout + rotation and tight layout.

Notes:
- No error bars (your json has single-run numbers).
- If you later have repeated runs: extend to mean±std with error bars.

Usage:
  python plot_dense_view_all_methods_8metrics.py \
    --dense_view_dir experiments/mapanything/benchmarking/dense_16_view \
    --output experiments/mapanything/benchmarking/figs_dense_16_view \
    --include_average

Optional:
  --no_average / --figsize 16 7 / --metrics k1,k2,...
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class MethodCfg:
    subdir: str


def build_methods_cfg() -> "OrderedDict[str, MethodCfg]":
    """Keep your method order stable here."""
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


# 8 metrics: 4 relative + 4 absolute (comprehensive)
METRICS_8 = [
    # -------- Relative (4) --------
    "z_depth_abs_rel",
    "pointmaps_abs_rel",
    "pose_auc_5",
    "ray_dirs_err_deg",
    # -------- Absolute (4) --------
    "z_depth_abs_mae",
    "pointmaps_abs_mae",
    "pose_ate_abs",
    "merged_pc_abs_chamfer_l1",
]


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
    s = s.strip()
    s = s.replace("/", "_").replace("\\", "_").replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_\-@.]+", "", s)
    return s[:180] if len(s) > 180 else s


def setup_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def pretty_metric_name(metric_key: str) -> str:
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


def union_dataset_keys(objs: list[dict], include_average: bool) -> list[str]:
    """
    Union dataset keys across all methods.
    Keep 'Average' last if included.
    """
    keys: set[str] = set()
    for obj in objs:
        for k, v in obj.items():
            if not isinstance(v, dict):
                continue
            if (not include_average) and (k == "Average"):
                continue
            keys.add(k)

    ordered = sorted([k for k in keys if k != "Average"])
    if include_average and ("Average" in keys):
        ordered.append("Average")
    return ordered


# -----------------------------
# Load: build tensor [metric, dataset, method]
# -----------------------------
def load_dense_view_metrics(
    dense_view_dir: Path,
    methods: "OrderedDict[str, MethodCfg]",
    metrics: list[str],
    include_average: bool,
) -> tuple[list[str], list[str], list[str], "list[list[list[float]]]"]:
    """
    Returns:
      dataset_keys, method_labels, metrics, cube

    cube is a 3D list:
      cube[mi][di][mj] = value or NaN
        mi: metric index
        di: dataset index
        mj: method index
    """
    import numpy as np

    if not dense_view_dir.is_dir():
        raise FileNotFoundError(f"dense_view_dir not found: {dense_view_dir}")

    method_labels = list(methods.keys())

    # read json for each method (if missing -> empty dict)
    method_objs: list[dict] = []
    for m, cfg in methods.items():
        jf = find_per_dataset_results(dense_view_dir / cfg.subdir)
        if jf is None:
            method_objs.append({})
            continue
        try:
            method_objs.append(json.loads(jf.read_text(encoding="utf-8")))
        except Exception:
            method_objs.append({})

    dataset_keys = union_dataset_keys(method_objs, include_average=include_average)
    if not dataset_keys:
        raise RuntimeError(f"No dataset keys found under: {dense_view_dir}")

    cube = np.full((len(metrics), len(dataset_keys), len(method_labels)), np.nan, dtype=np.float64)

    for mj, (m_label, _) in enumerate(methods.items()):
        obj = method_objs[mj]
        for di, dkey in enumerate(dataset_keys):
            dobj = obj.get(dkey, {})
            if not isinstance(dobj, dict):
                continue
            for mi, mk in enumerate(metrics):
                if mk not in dobj:
                    continue
                try:
                    cube[mi, di, mj] = float(dobj[mk])
                except Exception:
                    pass

    return dataset_keys, method_labels, metrics, cube.tolist()


# -----------------------------
# Plot: 2x4 grouped bar charts
# -----------------------------
def plot_dense_view_all_methods_8metrics(
    dense_view_dir: Path,
    methods: "OrderedDict[str, MethodCfg]",
    metrics: list[str],
    out_dir: Path,
    out_name: Optional[str],
    include_average: bool,
    figsize: tuple[float, float],
    rotate_xticks: int = 35,
) -> tuple[Path, Path]:
    """
    ONE figure:
      2x4 subplots, each subplot is one metric,
      grouped bars across methods for each dataset.
    """
    import numpy as np

    dataset_keys, method_labels, metrics, cube_list = load_dense_view_metrics(
        dense_view_dir=dense_view_dir,
        methods=methods,
        metrics=metrics,
        include_average=include_average,
    )
    cube = np.array(cube_list, dtype=np.float64)  # [metric, dataset, method]

    out_dir.mkdir(parents=True, exist_ok=True)
    if out_name is None:
        out_name = f"groupbars__{sanitize_filename(dense_view_dir.name)}__{sanitize_filename('8metrics')}"

    setup_paper_style()

    fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=False)
    axes = axes.flatten()

    n_d = len(dataset_keys)
    n_m = len(method_labels)

    x = np.arange(n_d)
    group_width = 0.82
    bar_w = group_width / max(n_m, 1)

    for mi, mk in enumerate(metrics):
        ax = axes[mi]

        for mj, m_label in enumerate(method_labels):
            # center the group around x
            offset = (mj - (n_m - 1) / 2.0) * bar_w
            y = cube[mi, :, mj]  # [dataset]
            ax.bar(x + offset, y, width=bar_w * 0.95, label=m_label)

        ax.set_title(pretty_metric_name(mk))
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_keys, rotation=rotate_xticks, ha="right")

        # Optional: tighten y limits for AUC to improve readability (comment out if not desired)
        # if mk == "pose_auc_5":
        #     ax.set_ylim(bottom=0)

    # hide extra axes if metrics < 8
    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    # suptitle
    fig.suptitle(f"{dense_view_dir.name}: Method comparison on 8 metrics", y=0.995, fontsize=12)

    # unified legend at bottom
    # collect unique handles
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        bbox_transform=fig.transFigure,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.97])

    out_png = out_dir / f"{out_name}.png"
    out_pdf = out_dir / f"{out_name}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return out_png, out_pdf


# -----------------------------
# CLI
# -----------------------------
def parse_figsize(vals: Iterable[str]) -> tuple[float, float]:
    v = list(vals)
    if len(v) != 2:
        raise ValueError("--figsize requires two numbers, e.g. --figsize 16 7")
    return float(v[0]), float(v[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ONE dense_n_view → ONE figure: 2x4 grouped-bar subplots comparing all methods on 8 metrics."
    )
    parser.add_argument(
        "--dense_view_dir",
        type=str,
        default="experiments/mapanything/benchmarking/dense_16_view",
        help="Path to dense_n_view directory, e.g. experiments/mapanything/benchmarking/dense_16_view",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/mapanything/benchmarking/figs_per_method",
        help="Output directory. Default: <dense_view_dir>/figs_all_methods",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="",
        help="Output base name (no extension). Default: auto-generated.",
    )
    parser.add_argument(
        "--include_average",
        action="store_true",
        help="Include 'Average' dataset as the last x-tick (recommended).",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        nargs=2,
        default=["16", "7"],
        help="Figure size (width height), e.g. --figsize 16 7",
    )
    parser.add_argument(
        "--rotate_xticks",
        type=int,
        default=35,
        help="Rotation angle for dataset labels on x-axis.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Optional override: comma-separated metric keys. If empty, uses built-in 8-metric set.",
    )

    args = parser.parse_args()

    dense_view_dir = Path(args.dense_view_dir)
    methods = build_methods_cfg()

    if args.metrics.strip():
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        if not metrics:
            metrics = METRICS_8
    else:
        metrics = METRICS_8

    if args.output.strip():
        out_dir = Path(args.output)
    else:
        out_dir = dense_view_dir / "figs_all_methods"

    out_name = args.out_name.strip() if args.out_name.strip() else None
    figsize = parse_figsize(args.figsize)

    out_png, out_pdf = plot_dense_view_all_methods_8metrics(
        dense_view_dir=dense_view_dir,
        methods=methods,
        metrics=metrics,
        out_dir=out_dir,
        out_name=out_name,
        include_average=bool(args.include_average),
        figsize=figsize,
        rotate_xticks=int(args.rotate_xticks),
    )

    print(f"[OK] Saved:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    main()