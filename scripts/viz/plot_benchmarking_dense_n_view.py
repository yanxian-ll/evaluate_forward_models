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
def parse_num_images(folder_name: str) -> int:
    """
    Parse num_images from a folder name.

    Priority:
      1) folder_name.split('_')[1] if it is an int
      2) first integer found by regex
    """
    parts = folder_name.split("_")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass

    m = re.search(r"(\d+)", folder_name)
    if not m:
        raise ValueError(f"Cannot parse num_images from folder name: {folder_name}")
    return int(m.group(1))


def find_json_file(search_dir: Path, json_name: str) -> Optional[Path]:
    """
    Locate the metrics json under `search_dir`.

    Tries:
      - exact json_name
      - stripped json_name
      - fallback glob "*WAI_avg_across_all_scenes.json" (prefer containing '200')
    """
    if not search_dir.is_dir():
        return None

    cand1 = search_dir / json_name
    if cand1.is_file():
        return cand1

    cand2 = search_dir / json_name.strip()
    if cand2.is_file():
        return cand2

    hits = sorted(search_dir.glob("*WAI_avg_across_all_scenes.json"))
    if not hits:
        return None

    hits_200 = [h for h in hits if re.search(r"\b200\b", h.name)]
    return hits_200[0] if hits_200 else hits[0]


def pretty_metric_name(metric_key: str) -> str:
    """
    Paper-friendly metric title, with an arrow:
    ↑ : higher is better
    ↓ : lower is better
    """
    higher_is_better = {
        # relative metrics
        "metric_scale_abs_rel": False,
        "pointmaps_abs_rel": False,
        "pointmaps_inlier_thres_103": True,
        "pose_ate_rmse": False,
        "pose_auc_5": True,
        "z_depth_abs_rel": False,
        "z_depth_inlier_thres_103": True,
        "ray_dirs_err_deg": False,
        # abs metrics
        "pointmaps_abs_mae": False,
        "pointmaps_abs_rmse": False,
        "z_depth_abs_mae": False,
        "z_depth_abs_rmse": False,
        "pose_ate_abs": False,
        "merged_pc_abs_chamfer_l1": False,
        "merged_pc_abs_chamfer_rmse": False,
        "merged_pc_abs_inlier_ratio": True,  # 这个通常越大越好（若你定义相反，改回 False）
    }

    mapping = {
        # relative metrics
        "metric_scale_abs_rel": "Scale AbsRel",
        "pointmaps_abs_rel": "PointMaps AbsRel",
        "pointmaps_inlier_thres_103": "PointMaps Inlier@1e-3",
        "pose_ate_rmse": "Pose ATE RMSE",
        "pose_auc_5": "Pose AUC@5°",
        "z_depth_abs_rel": "Z-Depth AbsRel",
        "z_depth_inlier_thres_103": "Z-Depth Inlier@1e-3",
        "ray_dirs_err_deg": "RayDirs Err (deg)",
        # abs metrics
        "pointmaps_abs_mae": "PointMaps MAE",
        "pointmaps_abs_rmse": "PointMaps RMSE",
        "z_depth_abs_mae": "Z-Depth MAE",
        "z_depth_abs_rmse": "Z-Depth RMSE",
        "pose_ate_abs": "Pose ATE RMSE",
        "merged_pc_abs_chamfer_l1": "ChamferL1",
        "merged_pc_abs_chamfer_rmse": "ChamferRMSE",
        "merged_pc_abs_inlier_ratio": "InlierRatio",
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
    color: str
    marker: str
    linestyle: str


def build_methods_cfg() -> "OrderedDict[str, MethodCfg]":
    """
    Distinct styles per method (color + marker + linestyle).
    Use tab10 colors: C0..C9 (colorblind-friendly-ish and standard in papers).
    """
    return OrderedDict(
        [
            ("MapAnything", MethodCfg(subdir="mapa_24v", color="C0", marker="o", linestyle="-")),
            ("VGGT",        MethodCfg(subdir="vggt",    color="C1", marker="s", linestyle="-")),
            ("Pi3",         MethodCfg(subdir="pi3",     color="C2", marker="^", linestyle="--")),
            ("Pi3X",        MethodCfg(subdir="pi3x",    color="C3", marker="D", linestyle="--")),
            ("DA3",         MethodCfg(subdir="da3",     color="C4", marker="v", linestyle="-.")),
            ("HunYuan",     MethodCfg(subdir="hunyuan", color="C5", marker="P", linestyle=":")),
        ]
    )


METRICS = [
    # # abs metrics
    # "pose_ate_abs",
    # "pointmaps_abs_mae",
    # "pointmaps_abs_rmse",
    # "ray_dirs_err_deg",
    # "pose_auc_5",
    # "z_depth_abs_mae",
    # "z_depth_abs_rmse",
    # "merged_pc_abs_chamfer_l1",

    # relative metrics
    "metric_scale_abs_rel",
    "pointmaps_abs_rel",
    "pointmaps_inlier_thres_103",
    "pose_ate_rmse",
    "pose_auc_5",
    "z_depth_abs_rel",
    "z_depth_inlier_thres_103",
    "ray_dirs_err_deg",
]


# -----------------------------
# Core logic
# -----------------------------
def collect_metrics(
    root: Path,
    methods: "OrderedDict[str, MethodCfg]",
    metrics: list[str],
    json_name: str,
) -> dict[str, dict[int, dict[str, float]]]:
    """
    data[method_label][num_images][metric] = value
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Root benchmarking directory not found: {root}")

    data: dict[str, dict[int, dict[str, float]]] = {m: {} for m in methods.keys()}

    for cond_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        try:
            num_images = parse_num_images(cond_dir.name)
        except Exception:
            # ignore folders not matching the pattern
            continue

        for method_label, cfg in methods.items():
            search_dir = cond_dir / cfg.subdir
            jf = find_json_file(search_dir, json_name)
            if jf is None:
                continue

            try:
                obj = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue

            bucket = data.setdefault(method_label, {}).setdefault(num_images, {})
            for k in metrics:
                if k in obj:
                    try:
                        bucket[k] = float(obj[k])
                    except Exception:
                        pass

    if not any(len(v) > 0 for v in data.values()):
        raise RuntimeError(
            f"No json metrics found under root={root}. "
            f"Expected structure: <root>/<num_images_folder>/<method_subdir>/<json_file>."
        )

    return data


def setup_paper_style() -> None:
    """
    A clean, paper-like matplotlib style.
    """
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
            # keep it clean
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_metrics(
    data: dict[str, dict[int, dict[str, float]]],
    methods: "OrderedDict[str, MethodCfg]",
    metrics: list[str],
    out_dir: Path,
    out_name: str,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_x = sorted({x for m in data for x in data[m].keys()})
    if not all_x:
        raise RuntimeError("No valid x-axis points (num_images) found.")

    setup_paper_style()

    # 2x4 grid (same as you)
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.2), constrained_layout=False)
    axes = axes.flatten()

    default_kw = dict(linewidth=1.8, markersize=5.0, markeredgewidth=0.8)

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for method_label, cfg in methods.items():
            xs = sorted(data.get(method_label, {}).keys())
            x_valid: list[int] = []
            ys: list[float] = []

            for x in xs:
                v = data[method_label][x].get(metric)
                if v is None:
                    continue
                x_valid.append(x)
                ys.append(v)

            if not x_valid:
                continue

            ax.plot(
                x_valid,
                ys,
                label=method_label,
                color=cfg.color,
                marker=cfg.marker,
                linestyle=cfg.linestyle,
                **default_kw,
            )

        ax.set_title(pretty_metric_name(metric))
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.set_xticks(all_x)
        ax.set_xlabel("")  # keep clean
        ax.set_ylabel("")  # keep clean

    # If fewer than 8 metrics, hide extra axes
    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    # Layout + unified legend (bottom)
    fig.tight_layout(rect=[0.0, 0.14, 1.0, 1.0])

    # Collect unique legend entries across subplots
    uniq_handles: OrderedDict[str, object] = OrderedDict()
    for ax in axes[: len(metrics)]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in uniq_handles:
                uniq_handles[ll] = hh

    fig.legend(
        list(uniq_handles.values()),
        list(uniq_handles.keys()),
        loc="lower center",
        ncol=min(len(uniq_handles), 6),
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        bbox_transform=fig.transFigure,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    out_png = out_dir / f"{out_name}.png"
    out_pdf = out_dir / f"{out_name}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return out_png, out_pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarking",
        type=str,
        default="experiments/mapanything/benchmarking",
        help="Root dir containing different num_images folders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/mapanything/benchmarking",
        help="Output directory to save figures.",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="metrics_vs_num_images_relative_metrics",
        help="Output figure base name (no extension).",
    )
    parser.add_argument(
        "--json_name",
        type=str,
        default=" 20 @ A3DScenesWAI_avg_across_all_scenes.json",
        help="Metrics json filename (kept for compatibility).",
    )
    args = parser.parse_args()

    root = Path(args.benchmarking)
    out_dir = Path(args.output)

    methods = build_methods_cfg()

    data = collect_metrics(
        root=root,
        methods=methods,
        metrics=METRICS,
        json_name=args.json_name,
    )

    out_png, out_pdf = plot_metrics(
        data=data,
        methods=methods,
        metrics=METRICS,
        out_dir=out_dir,
        out_name=args.out_name,
    )

    print(f"[OK] Saved:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    main()
