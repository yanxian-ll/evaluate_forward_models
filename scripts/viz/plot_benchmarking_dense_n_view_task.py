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


def find_per_dataset_results(search_dir: Path) -> Optional[Path]:
    """
    Locate per_dataset_results.json under `search_dir`.
    Searches for: <search_dir>/per_dataset_results.json
    """
    if not search_dir.is_dir():
        return None
    json_path = search_dir / "per_dataset_results.json"
    return json_path if json_path.is_file() else None


def sanitize_filename(s: str) -> str:
    """Make a safe filename fragment."""
    s = s.strip()
    s = s.replace("/", "_").replace("\\", "_").replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_\-@.]+", "", s)
    return s[:180] if len(s) > 180 else s


def pretty_metric_name(metric_key: str) -> str:
    """
    Paper-friendly metric title, with an arrow:
    ↑ : higher is better
    ↓ : lower is better
    """
    higher_is_better = {
        "metric_scale_abs_rel": False,
        "pointmaps_abs_rel": False,
        "pointmaps_inlier_thres_103": True,
        "pose_ate_rmse": False,
        "pose_auc_5": True,
        "z_depth_abs_rel": False,
        "z_depth_inlier_thres_103": True,
        "ray_dirs_err_deg": False,
        "pointmaps_abs_mae": False,
        "pointmaps_abs_rmse": False,
        "z_depth_abs_mae": False,
        "z_depth_abs_rmse": False,
        "pose_ate_abs": False,
        "merged_pc_abs_chamfer_l1": False,
        "merged_pc_abs_chamfer_rmse": False,
        "merged_pc_abs_inlier_ratio": True,
        "pr_to_gt_scale": False,  # 如果你把 scale closer-to-1 当好，可改成 False 并单独解释
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
    color: str
    marker: str
    linestyle: str


def build_methods_cfg() -> "OrderedDict[str, MethodCfg]":
    return OrderedDict(
        [
            ("MapAnything", MethodCfg(subdir="mapa_24v", color="C0", marker="o", linestyle="-")),
            ("Mapa-csfm",   MethodCfg(subdir="mapa_24v_csfm",color="C1", marker="s", linestyle="-")),
            ("Mapa-psfm",   MethodCfg(subdir="mapa_24v_psfm",color="C2", marker="^", linestyle="--")),
            ("Mapa-mvs",    MethodCfg(subdir="mapa_24v_mvs", color="C3", marker="D", linestyle="--")),
        ]
    )

METRICS = [
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
# Core logic (per-dataset)
# -----------------------------
def collect_metrics_per_dataset(
    root: Path,
    methods: "OrderedDict[str, MethodCfg]",
    metrics: list[str],
) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    """
    Collect metrics from per_dataset_results.json files.

    Return:
      data[method_label][num_images][dataset_key][metric] = value

    Directory structure:
      <root>/dense_*_view/<method_subdir>/per_dataset_results.json
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Root benchmarking directory not found: {root}")

    data: dict[str, dict[int, dict[str, dict[str, float]]]] = {m: {} for m in methods.keys()}

    for view_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        if not view_dir.name.startswith("dense_") or not view_dir.name.endswith("_view"):
            continue

        try:
            num_images = parse_num_images(view_dir.name)
        except Exception:
            continue

        for method_label, cfg in methods.items():
            search_dir = view_dir / cfg.subdir
            jf = find_per_dataset_results(search_dir)
            if jf is None:
                continue

            try:
                obj = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue

            num_bucket = data.setdefault(method_label, {}).setdefault(num_images, {})

            # obj: { dataset_key: {metric: value, ...}, "Average": {...} }
            for dataset_key, dataset_obj in obj.items():
                if not isinstance(dataset_obj, dict):
                    continue
                d_bucket = num_bucket.setdefault(dataset_key, {})
                for k in metrics:
                    if k in dataset_obj:
                        try:
                            d_bucket[k] = float(dataset_obj[k])
                        except Exception:
                            pass

    # sanity check
    any_found = False
    for m in data:
        for x in data[m]:
            if any(len(data[m][x][d]) > 0 for d in data[m][x]):
                any_found = True
                break
        if any_found:
            break

    if not any_found:
        raise RuntimeError(
            f"No per_dataset_results.json found under root={root}. "
            f"Expected: <root>/dense_*_view/<method>/per_dataset_results.json."
        )

    return data


def setup_paper_style() -> None:
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


def extract_dataset_slice(
    data_all: dict[str, dict[int, dict[str, dict[str, float]]]],
    dataset_key: str,
) -> dict[str, dict[int, dict[str, float]]]:
    """
    Convert:
      data_all[method][num_images][dataset_key][metric]
    to:
      data_slice[method][num_images][metric]
    (for the plotting code that expects method->x->metric)
    """
    data_slice: dict[str, dict[int, dict[str, float]]] = {}
    for method_label, by_x in data_all.items():
        for x, by_dataset in by_x.items():
            if dataset_key not in by_dataset:
                continue
            metrics_obj = by_dataset[dataset_key]
            if not metrics_obj:
                continue
            data_slice.setdefault(method_label, {}).setdefault(x, {}).update(metrics_obj)
    return data_slice


def plot_metrics_grid(
    data: dict[str, dict[int, dict[str, float]]],
    methods: "OrderedDict[str, MethodCfg]",
    metrics: list[str],
    out_dir: Path,
    out_name: str,
    suptitle: Optional[str] = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_x = sorted({x for m in data for x in data[m].keys()})
    if not all_x:
        raise RuntimeError("No valid x-axis points (num_images) found.")

    setup_paper_style()

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

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    if suptitle:
        fig.suptitle(suptitle, y=0.99, fontsize=12)

    fig.tight_layout(rect=[0.0, 0.14, 1.0, 0.97])

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
    # fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return out_png, out_pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarking",
        type=str,
        default="experiments/mapanything/benchmarking",
        help="Root dir containing dense_*_view folders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/mapanything/benchmarking/figs_per_dataset_task",
        help="Output directory to save figures.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="metrics_vs_num_images",
        help="Output figure prefix (dataset name will be appended).",
    )
    args = parser.parse_args()

    root = Path(args.benchmarking)
    out_dir = Path(args.output)
    methods = build_methods_cfg()

    # 1) Collect all datasets
    data_all = collect_metrics_per_dataset(
        root=root,
        methods=methods,
        metrics=METRICS,
    )

    # 2) Enumerate dataset keys (union across methods/x)
    dataset_keys: set[str] = set()
    for m in data_all:
        for x in data_all[m]:
            dataset_keys.update(data_all[m][x].keys())

    # Ensure Average is included (and preferably last)
    dataset_keys_list = sorted([k for k in dataset_keys if k != "Average"])
    if "Average" in dataset_keys:
        dataset_keys_list.append("Average")

    if not dataset_keys_list:
        raise RuntimeError("No dataset keys found in per_dataset_results.json.")

    # 3) Plot per dataset
    saved = []
    for dkey in dataset_keys_list:
        data_slice = extract_dataset_slice(data_all, dkey)

        # Skip if this dataset has no valid points
        has_any = any(len(v) > 0 for v in data_slice.values())
        if not has_any:
            continue

        name_frag = sanitize_filename(dkey)
        out_name = f"{args.out_prefix}__{name_frag}"

        out_png, out_pdf = plot_metrics_grid(
            data=data_slice,
            methods=methods,
            metrics=METRICS,
            out_dir=out_dir,
            out_name=out_name,
            suptitle=dkey,
        )
        saved.append((out_png, out_pdf))

    print(f"[OK] Saved {len(saved)} dataset figures to: {out_dir}")
    for p1, p2 in saved[:5]:
        print(f"  - {p1.name}\n  - {p2.name}")
    if len(saved) > 5:
        print(f"  ... and {len(saved) - 5} more.")


if __name__ == "__main__":
    main()
