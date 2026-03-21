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

    Searches for:
        /per_dataset_results.json
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
        "ray_dirs_err_deg": False,
        "pose_ate_abs": False,
        "merged_pc_abs_chamfer_l1": False,
        "z_depth_abs_rel_seq_scale": False,
    }

    mapping = {
        "ray_dirs_err_deg": "Ray Error (deg)",
        "pose_ate_abs": "ATE",
        "merged_pc_abs_chamfer_l1": "Chamfer-L1",
        "z_depth_abs_rel_seq_scale": "AbsRel",
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
            # -------------------------
            # Mapa family: same hue, different styles
            # -------------------------
            ("MapAnything", MethodCfg(subdir="mapa_24v",      color="#1f77b4", marker="o", linestyle="-")),
            # ("Mapa-csfm",   MethodCfg(subdir="mapa_24v_csfm", color="#1f77b4", marker="s", linestyle="--")),
            # ("Mapa-psfm",   MethodCfg(subdir="mapa_24v_psfm", color="#1f77b4", marker="^", linestyle="-.")),
            # ("Mapa-mvs",    MethodCfg(subdir="mapa_24v_mvs",  color="#1f77b4", marker="D", linestyle=":")),

            # -------------------------
            # Other baselines
            # -------------------------
            ("VGGT",        MethodCfg(subdir="vggt",    color="#d62728", marker="o", linestyle="-")),
            ("Pi3",         MethodCfg(subdir="pi3",     color="#2ca02c", marker="^", linestyle="--")),
            ("DA3",         MethodCfg(subdir="da3",     color="#9467bd", marker="v", linestyle="-.")),
            ("HunYuan",     MethodCfg(subdir="hunyuan", color="#8c564b", marker="P", linestyle=":")),
        ]
    )


# 论文主图：1x4，排除 failure rate
METRICS = [
    "ray_dirs_err_deg",
    "pose_ate_abs",
    "merged_pc_abs_chamfer_l1",
    "z_depth_abs_rel_seq_scale",
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
        /dense_*_view/<method_subdir>/per_dataset_results.json
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
            f"Expected: /dense_*_view/<method_subdir>/per_dataset_results.json."
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

    # 1 x 4
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8), constrained_layout=False)
    if not isinstance(axes, (list, tuple)):
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
        ax.set_xlabel("# Views")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.set_xticks(all_x)

    if suptitle:
        fig.suptitle(suptitle, y=0.98, fontsize=12)

    fig.tight_layout(rect=[0.0, 0.16, 1.0, 0.95])

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
        ncol=min(len(uniq_handles), 7),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
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
        help="Root dir containing dense_*_view folders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/mapanything/benchmarking/figs_per_dataset",
        help="Output directory to save figures.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="metrics_vs_num_views",
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
        print(f" - {p1.name}\n - {p2.name}")
    if len(saved) > 5:
        print(f" ... and {len(saved) - 5} more.")


if __name__ == "__main__":
    main()
