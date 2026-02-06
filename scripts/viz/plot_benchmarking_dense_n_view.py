#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import argparse
from collections import OrderedDict

import matplotlib.pyplot as plt


def parse_num_images(name: str) -> int:
    """
    Try to parse num_images from a folder name.
    Priority:
      1) name.split('_')[1] if it is an int
      2) first integer found by regex
    """
    parts = name.split("_")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except Exception:
            pass

    m = re.search(r"(\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse num_images from folder name: {name}")
    return int(m.group(1))


def find_json_file(search_dir: str, json_name: str) -> str | None:
    """
    Locate the metrics json under `search_dir`.
    Tries:
      - exact json_name
      - json_name.strip()
      - glob match: *WHUOMVSWAI_avg_across_all_scenes.json (and optionally containing '200')
    """
    if not os.path.isdir(search_dir):
        return None

    cand1 = os.path.join(search_dir, json_name)
    if os.path.isfile(cand1):
        return cand1

    cand2 = os.path.join(search_dir, json_name.strip())
    if os.path.isfile(cand2):
        return cand2

    patt = os.path.join(search_dir, "*WHUOMVSWAI_avg_across_all_scenes.json")
    hits = sorted(glob.glob(patt))
    if not hits:
        return None

    hits_200 = [h for h in hits if re.search(r"\b200\b", os.path.basename(h))]
    return hits_200[0] if hits_200 else hits[0]


def pretty_metric_name(k: str) -> str:
    """
    Return a paper-friendly metric title, with a small arrow indicating
    whether higher (↑) or lower (↓) is better.
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
    }

    base = mapping.get(k, k.replace("_", " "))
    arrow = "↑" if higher_is_better.get(k, False) else "↓"
    return f"{base} {arrow}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarking", type=str, default="experiments/mapanything/benchmarking",
                        help="Root dir containing different num_images folders.")
    parser.add_argument("--output", type=str, default="experiments/mapanything/benchmarking",
                        help="Output directory to save figures.")
    parser.add_argument("--out_name", type=str, default="metrics_vs_num_images",
                        help="Output figure base name (no extension).")
    args = parser.parse_args()

    # =========================
    # ✅ 统一方法配置：dir + plot(line/color/marker/...)
    #   只需要在这里改颜色/线型等
    # =========================
    METHODS_CFG = OrderedDict({
        "MapAnything-img": {
            "dir": "mapa_24v",
            "plot": {"color": "tab:blue", "linestyle": "-", "marker": "o"},
        },
        "MapAnything-psfm": {
            "dir": "mapa_24v_psfm",
            "plot": {"color": "tab:blue", "linestyle": ":", "marker": "^"},
        },
        "UAVMapa-img": {
            "dir": "uav_mapa_16v",
            "plot": {"color": "tab:orange", "linestyle": "-", "marker": "o"},
        },
        "UAVMapa-psfm": {
            "dir": "uav_mapa_16v_psfm",
            "plot": {"color": "tab:orange", "linestyle": ":", "marker": "^"},
        },
        "VGGT": {
            "dir": "vggt",
            "plot": {"color": "tab:purple", "linestyle": "-", "marker": "o"},
        },
        "Pi3": {
            "dir": "pi3",
            "plot": {"color": "tab:green", "linestyle": "-", "marker": "o"},
        }
    })

    # 默认线宽/点大小（各方法可在 plot 里覆盖）
    DEFAULT_PLOT_KW = {"linewidth": 1.8, "markersize": 4.5}

    # NOTE: your string has leading spaces; we keep it but also try .strip() + glob fallback.
    json_name = " 260 @ A3DScenesWAI_avg_across_all_scenes.json"

    metrics = [
        "metric_scale_abs_rel",
        "pointmaps_abs_rel",
        "pointmaps_inlier_thres_103",
        "pose_ate_rmse",
        "pose_auc_5",
        "z_depth_abs_rel",
        "z_depth_inlier_thres_103",
        "ray_dirs_err_deg",
    ]

    root = args.benchmarking
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    # Collect: data[method][num_images][metric] = value
    data: dict[str, dict[int, dict[str, float]]] = {m: {} for m in METHODS_CFG.keys()}

    condition_dirs = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]

    # Parse all num_images folders
    for d in condition_dirs:
        cond_path = os.path.join(root, d)
        try:
            num_images = parse_num_images(d)
        except Exception:
            continue

        for method_label, cfg in METHODS_CFG.items():
            method_dir = cfg["dir"]
            search_dir = os.path.join(cond_path, method_dir)
            jf = find_json_file(search_dir, json_name)
            if jf is None:
                continue

            with open(jf, "r", encoding="utf-8") as f:
                obj = json.load(f)

            data.setdefault(method_label, {})
            data[method_label].setdefault(num_images, {})
            for k in metrics:
                if k in obj:
                    data[method_label][num_images][k] = float(obj[k])

    any_points = any(len(v) > 0 for v in data.values())
    if not any_points:
        raise RuntimeError(
            f"No json metrics found under root={root}. "
            f"Expected structure: <root>/<num_images_folder>/<method_dir>/<json_file>."
        )

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.0,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.2), constrained_layout=False)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for method_label, cfg in METHODS_CFG.items():
            xs = sorted(data.get(method_label, {}).keys())
            ys = []
            x_valid = []
            for x in xs:
                v = data[method_label][x].get(metric, None)
                if v is None:
                    continue
                x_valid.append(x)
                ys.append(v)

            if not x_valid:
                continue

            plot_kw = dict(DEFAULT_PLOT_KW)
            plot_kw.update(cfg.get("plot", {}))  # ✅ 每个方法的线型/颜色/marker
            ax.plot(x_valid, ys, label=method_label, **plot_kw)

        ax.set_title(pretty_metric_name(metric))
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_xlabel("")
        ax.set_ylabel("")

        all_x = sorted({x for m in data for x in data[m].keys()})
        if all_x:
            ax.set_xticks(all_x)

    # 给底部 legend 预留空间
    fig.tight_layout(rect=[0.0, 0.14, 1.0, 1.0])

    # 汇总所有子图的 legend（避免某个子图缺线导致 legend 不全）
    uniq = OrderedDict()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in uniq:
                uniq[ll] = hh

    handles = list(uniq.values())
    labels = list(uniq.keys())

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 4),
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        bbox_transform=fig.transFigure,
        borderaxespad=0.0,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    out_png = os.path.join(out_dir, f"{args.out_name}.png")
    out_pdf = os.path.join(out_dir, f"{args.out_name}.pdf")

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved:\n  {out_png}\n  {out_pdf}")
