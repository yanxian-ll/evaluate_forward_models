# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Script to benchmark the dense multi-view metric reconstruction performance
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from tqdm import tqdm
import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf

from mapanything.datasets import get_test_data_loader
from mapanything.models import init_model
from mapanything.utils.geometry import (
    geotrf,
    inv,
    normalize_multiple_pointclouds,
    quaternion_to_rotation_matrix,
    transform_pose_using_quats_and_trans_2_to_1,
)
from mapanything.utils.metrics import (
    global_scale_from_pointmaps,   # moved here
    compute_set_metrics,           # NEW: all metrics computed in metrics.py
)

from mapanything.utils.misc import StreamToLogger

log = logging.getLogger(__name__)


def write_ply_xyzrgb(path, points, colors_u8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    N = points.shape[0]
    assert colors_u8.shape[0] == N

    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ])

    with open(path, "w") as f:
        f.write(header + "\n")
        for p, c in zip(points, colors_u8):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def get_all_info_for_metric_computation(batch, preds, norm_mode="avg_dis"):
    """
    (基本保持你原来的实现)
    Returns:
      gt_info, pr_info, valid_masks, gt_info_abs, pr_info_abs_aligned, scale_factors
    """
    n_views = len(batch)
    batch_size = batch[0]["camera_pose"].shape[0]

    in_camera0 = inv(batch[0]["camera_pose"])
    no_norm_gt_pts = []
    no_norm_gt_pts3d_cam = []
    no_norm_gt_pose_trans = []
    valid_masks = []
    gt_ray_directions = []
    gt_pose_quats = []

    pred_camera0 = torch.eye(4, device=preds[0]["cam_quats"].device).unsqueeze(0)
    pred_camera0 = pred_camera0.repeat(batch_size, 1, 1)
    pred_camera0_rot = quaternion_to_rotation_matrix(preds[0]["cam_quats"].clone())
    pred_camera0[..., :3, :3] = pred_camera0_rot
    pred_camera0[..., :3, 3] = preds[0]["cam_trans"].clone()
    pred_in_camera0 = inv(pred_camera0)

    no_norm_pr_pts = []
    no_norm_pr_pts3d_cam = []
    no_norm_pr_pose_trans = []
    pr_ray_directions = []
    pr_pose_quats = []
    metric_pr_pts_to_compute_scale = []

    for i in range(n_views):
        no_norm_gt_pts.append(geotrf(in_camera0, batch[i]["pts3d"]))
        valid_masks.append(batch[i]["valid_mask"].clone())
        gt_ray_directions.append(batch[i]["ray_directions_cam"])
        no_norm_gt_pts3d_cam.append(batch[i]["pts3d_cam"])

        if i == 0:
            gt_pose_quats.append(
                torch.tensor([0, 0, 0, 1], dtype=gt_ray_directions[0].dtype, device=gt_ray_directions[0].device)
                .unsqueeze(0).repeat(gt_ray_directions[0].shape[0], 1)
            )
            no_norm_gt_pose_trans.append(
                torch.tensor([0, 0, 0], dtype=gt_ray_directions[0].dtype, device=gt_ray_directions[0].device)
                .unsqueeze(0).repeat(gt_ray_directions[0].shape[0], 1)
            )
        else:
            gt_pose_quats_world = batch[i]["camera_pose_quats"]
            no_norm_gt_pose_trans_world = batch[i]["camera_pose_trans"]
            gt_pose_quats_in_view0, no_norm_gt_pose_trans_in_view0 = transform_pose_using_quats_and_trans_2_to_1(
                batch[0]["camera_pose_quats"],
                batch[0]["camera_pose_trans"],
                gt_pose_quats_world,
                no_norm_gt_pose_trans_world,
            )
            gt_pose_quats.append(gt_pose_quats_in_view0)
            no_norm_gt_pose_trans.append(no_norm_gt_pose_trans_in_view0)

        pr_pose_quats_in_view0, pr_pose_trans_in_view0 = transform_pose_using_quats_and_trans_2_to_1(
            preds[0]["cam_quats"], preds[0]["cam_trans"], preds[i]["cam_quats"], preds[i]["cam_trans"]
        )
        pr_pts3d_in_view0 = geotrf(pred_in_camera0, preds[i]["pts3d"])

        if "metric_scaling_factor" in preds[i].keys():
            curr_view_no_norm_pr_pts = pr_pts3d_in_view0 / preds[i]["metric_scaling_factor"].unsqueeze(-1).unsqueeze(-1)
            curr_view_no_norm_pr_pts3d_cam = preds[i]["pts3d_cam"] / preds[i]["metric_scaling_factor"].unsqueeze(-1).unsqueeze(-1)
            curr_view_no_norm_pr_pose_trans = pr_pose_trans_in_view0 / preds[i]["metric_scaling_factor"]
        else:
            curr_view_no_norm_pr_pts = pr_pts3d_in_view0
            curr_view_no_norm_pr_pts3d_cam = preds[i]["pts3d_cam"]
            curr_view_no_norm_pr_pose_trans = pr_pose_trans_in_view0

        no_norm_pr_pts.append(curr_view_no_norm_pr_pts)
        no_norm_pr_pts3d_cam.append(curr_view_no_norm_pr_pts3d_cam)
        no_norm_pr_pose_trans.append(curr_view_no_norm_pr_pose_trans)
        pr_ray_directions.append(preds[i]["ray_directions"])
        pr_pose_quats.append(pr_pose_quats_in_view0)

        if "metric_scaling_factor" in preds[i].keys():
            curr_view_metric_pr_pts_to_compute_scale = (
                curr_view_no_norm_pr_pts.detach()
                * preds[i]["metric_scaling_factor"].unsqueeze(-1).unsqueeze(-1)
            )
        else:
            curr_view_metric_pr_pts_to_compute_scale = curr_view_no_norm_pr_pts.clone()
        metric_pr_pts_to_compute_scale.append(curr_view_metric_pr_pts_to_compute_scale)

    # normalized tensors
    gt_pts = [torch.zeros_like(pts) for pts in no_norm_gt_pts]
    gt_pts3d_cam = [torch.zeros_like(pts_cam) for pts_cam in no_norm_gt_pts3d_cam]
    gt_pose_trans = [torch.zeros_like(trans) for trans in no_norm_gt_pose_trans]

    pr_pts = [torch.zeros_like(pts) for pts in no_norm_pr_pts]
    pr_pts3d_cam = [torch.zeros_like(pts_cam) for pts_cam in no_norm_pr_pts3d_cam]
    pr_pose_trans = [torch.zeros_like(trans) for trans in no_norm_pr_pose_trans]

    pr_norm_out = normalize_multiple_pointclouds(no_norm_pr_pts, valid_masks, norm_mode, ret_factor=True)
    pr_pts_norm = pr_norm_out[:-1]
    pr_norm_factor = pr_norm_out[-1]

    gt_norm_out = normalize_multiple_pointclouds(no_norm_gt_pts, valid_masks, norm_mode, ret_factor=True)
    gt_pts_norm = gt_norm_out[:-1]
    gt_norm_factor = gt_norm_out[-1]

    for i in range(n_views):
        pr_pts[i] = pr_pts_norm[i]
        pr_pts3d_cam[i] = no_norm_pr_pts3d_cam[i] / pr_norm_factor
        pr_pose_trans[i] = no_norm_pr_pose_trans[i] / pr_norm_factor[:, :, 0, 0]

        gt_pts[i] = gt_pts_norm[i]
        gt_pts3d_cam[i] = no_norm_gt_pts3d_cam[i] / gt_norm_factor
        gt_pose_trans[i] = no_norm_gt_pose_trans[i] / gt_norm_factor[:, :, 0, 0]

    metric_scale_mask = batch[0]["is_metric_scale"]
    valid_gt_norm_factor_mask = gt_norm_factor[:, 0, 0, 0] > 1e-8
    valid_metric_scale_mask = metric_scale_mask & valid_gt_norm_factor_mask

    if valid_metric_scale_mask.any():
        metric_pr_norm_out = normalize_multiple_pointclouds(metric_pr_pts_to_compute_scale, valid_masks, norm_mode, ret_factor=True)
        pr_metric_norm_factor = metric_pr_norm_out[-1]
        gt_metric_norm_factor = gt_norm_factor[valid_metric_scale_mask].cpu()
        pr_metric_norm_factor = pr_metric_norm_factor[valid_metric_scale_mask].cpu()
    else:
        gt_metric_norm_factor = None
        pr_metric_norm_factor = None

    gt_poses, gt_z_depths, pr_poses, pr_z_depths = [], [], [], []
    for i in range(n_views):
        gt_pose_curr_view = torch.eye(4, device=gt_pose_quats[i].device).repeat(gt_pose_quats[i].shape[0], 1, 1)
        gt_pose_curr_view[..., :3, :3] = quaternion_to_rotation_matrix(gt_pose_quats[i])
        gt_pose_curr_view[..., :3, 3] = gt_pose_trans[i]
        gt_poses.append(gt_pose_curr_view)

        pr_pose_curr_view = torch.eye(4, device=pr_pose_quats[i].device).unsqueeze(0).repeat(pr_pose_quats[i].shape[0], 1, 1)
        pr_pose_curr_view[..., :3, :3] = quaternion_to_rotation_matrix(pr_pose_quats[i])
        pr_pose_curr_view[..., :3, 3] = pr_pose_trans[i]
        pr_poses.append(pr_pose_curr_view)

        gt_z_depths.append(gt_pts3d_cam[i][..., 2:].cpu())
        pr_z_depths.append(pr_pts3d_cam[i][..., 2:].cpu())

        gt_pts[i] = gt_pts[i].cpu()
        pr_pts[i] = pr_pts[i].cpu()
        valid_masks[i] = valid_masks[i].cpu()

    gt_info = {"ray_directions": gt_ray_directions, "z_depths": gt_z_depths, "poses": gt_poses, "pts3d": gt_pts, "metric_scale": gt_metric_norm_factor}
    pr_info = {"ray_directions": pr_ray_directions, "z_depths": pr_z_depths, "poses": pr_poses, "pts3d": pr_pts, "metric_scale": pr_metric_norm_factor}

    # absolute (non-normalized) to cpu
    gt_poses_abs, gt_z_abs, pr_poses_abs, pr_z_abs = [], [], [], []
    gt_pts_abs, pr_pts_abs = [], []
    for i in range(n_views):
        gt_pose_abs = torch.eye(4, device=gt_pose_quats[i].device).repeat(gt_pose_quats[i].shape[0], 1, 1)
        gt_pose_abs[..., :3, :3] = quaternion_to_rotation_matrix(gt_pose_quats[i])
        gt_pose_abs[..., :3, 3] = no_norm_gt_pose_trans[i]
        gt_poses_abs.append(gt_pose_abs)

        pr_pose_abs = torch.eye(4, device=pr_pose_quats[i].device).unsqueeze(0).repeat(pr_pose_quats[i].shape[0], 1, 1)
        pr_pose_abs[..., :3, :3] = quaternion_to_rotation_matrix(pr_pose_quats[i])
        pr_pose_abs[..., :3, 3] = no_norm_pr_pose_trans[i]
        pr_poses_abs.append(pr_pose_abs)

        gt_pts_abs.append(no_norm_gt_pts[i].cpu())
        pr_pts_abs.append(no_norm_pr_pts[i].cpu())
        gt_z_abs.append(no_norm_gt_pts3d_cam[i][..., 2:].cpu())
        pr_z_abs.append(no_norm_pr_pts3d_cam[i][..., 2:].cpu())

    gt_info_abs = {"poses": gt_poses_abs, "pts3d": gt_pts_abs, "z_depths": gt_z_abs}
    pr_info_abs = {"poses": pr_poses_abs, "pts3d": pr_pts_abs, "z_depths": pr_z_abs}

    # global scale (per batch element)
    pr_to_gt_scales = torch.ones((batch_size,), device=no_norm_pr_pts[0].device, dtype=no_norm_pr_pts[0].dtype)
    for b in range(batch_size):
        gt_list_b = [no_norm_gt_pts[v][b] for v in range(n_views)]
        pr_list_b = [no_norm_pr_pts[v][b] for v in range(n_views)]
        m_list_b = [valid_masks[v][b] for v in range(n_views)]
        s_b = global_scale_from_pointmaps(gt_list_b, pr_list_b, m_list_b)
        pr_to_gt_scales[b] = pr_to_gt_scales[b] * float(s_b)

    # build scaled absolute pred info
    pr_poses_abs_aligned, pr_pts_abs_aligned, pr_z_abs_aligned = [], [], []
    for i in range(n_views):
        t_scaled = no_norm_pr_pose_trans[i] * pr_to_gt_scales[:, None]
        pr_pose_abs_al = torch.eye(4, device=pr_pose_quats[i].device).unsqueeze(0).repeat(pr_pose_quats[i].shape[0], 1, 1)
        pr_pose_abs_al[..., :3, :3] = quaternion_to_rotation_matrix(pr_pose_quats[i])
        pr_pose_abs_al[..., :3, 3] = t_scaled
        pr_poses_abs_aligned.append(pr_pose_abs_al)

        s_map = pr_to_gt_scales[:, None, None, None]
        pr_pts_abs_aligned.append((no_norm_pr_pts[i] * s_map).cpu())
        pr_z_abs_aligned.append((no_norm_pr_pts3d_cam[i][..., 2:] * s_map).cpu())

    pr_info_abs_aligned = {"poses": pr_poses_abs_aligned, "pts3d": pr_pts_abs_aligned, "z_depths": pr_z_abs_aligned}

    scale_factors = {
        "pr_to_gt_scale": pr_to_gt_scales.detach().cpu(),
        "gt_norm_factor": gt_norm_factor.detach().cpu(),
        "pr_norm_factor": pr_norm_factor.detach().cpu(),
    }
    return gt_info, pr_info, valid_masks, gt_info_abs, pr_info_abs_aligned, scale_factors


def build_dataset(dataset, batch_size, num_workers):
    print("Building data loader for dataset: ", dataset)
    loader = get_test_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=False,
        drop_last=False,
    )
    print("Dataset length: ", len(loader))
    return loader


@torch.no_grad()
def benchmark(args):
    print("Output Directory: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    if args.amp:
        if args.amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif args.amp_dtype == "bf16":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    data_loaders = {
        dataset.split("(")[0]: build_dataset(dataset, args.batch_size, args.dataset.num_workers)
        for dataset in args.dataset.test_dataset.split("+")
        if "(" in dataset
    }

    model = init_model(args.model.model_str, args.model.model_config, torch_hub_force_reload=False)
    model.to(device)
    model.eval()

    if args.model.pretrained:
        ckpt = torch.load(args.model.pretrained, map_location=device, weights_only=False)
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt

    per_dataset_results = {}

    for benchmark_dataset_name, data_loader in data_loaders.items():
        print("Benchmarking dataset: ", benchmark_dataset_name)
        data_loader.dataset.set_epoch(0)

        max_sets_per_scene = getattr(args, "save_n_fused_ply", 5)
        per_scene_saved = {scene: 0 for scene in data_loader.dataset.dataset.scenes}

        fused_out_dir = os.path.join(args.output_dir, f"{benchmark_dataset_name}_fused_ply")
        os.makedirs(fused_out_dir, exist_ok=True)

        per_scene_results = {}
        for dataset_scene in data_loader.dataset.dataset.scenes:
            per_scene_results[dataset_scene] = {
                "metric_scale_abs_rel": [],
                "pointmaps_abs_rel": [],
                "pointmaps_inlier_thres_103": [],
                "pose_ate_rmse": [],
                "pose_auc_5": [],
                "z_depth_abs_rel": [],
                "z_depth_inlier_thres_103": [],
                "ray_dirs_err_deg": [],
                "pointmaps_abs_mae": [],
                "pointmaps_abs_rmse": [],
                "z_depth_abs_mae": [],
                "z_depth_abs_rmse": [],
                "pose_ate_abs": [],
                "merged_pc_abs_chamfer_l1": [],
                "merged_pc_abs_chamfer_rmse": [],
                "merged_pc_abs_inlier_ratio": [],
                "pr_to_gt_scale": [],
            }

        for batch in tqdm(data_loader):
            n_views = len(batch)

            for view in batch:
                view["idx"] = view["idx"][2:]

            ignore_keys = set(["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng", "data_norm_type"])
            for view in batch:
                for name in view.keys():
                    if name in ignore_keys:
                        continue
                    view[name] = view[name].to(device, non_blocking=True)

            with torch.autocast("cuda", enabled=bool(args.amp), dtype=amp_dtype):
                preds = model(batch)

            gt_info, pr_info, valid_masks, gt_info_abs, pr_info_abs_aligned, scale_factors = get_all_info_for_metric_computation(batch, preds)

            batch_size = batch[0]["img"].shape[0]
            for batch_idx in range(batch_size):
                scene = batch[0]["label"][batch_idx]

                metrics, fused_debug = compute_set_metrics(
                    batch_views=batch,
                    batch_idx=batch_idx,
                    gt_info=gt_info,
                    pr_info=pr_info,
                    valid_masks=valid_masks,
                    gt_info_abs=gt_info_abs,
                    pr_info_abs=pr_info_abs_aligned,   # important: scaled pred abs
                    scale_factors=scale_factors,
                    device=device,
                    voxel=0.1,
                    icp_iters=20,
                    trim_ratio=0.8,
                    sim3_trim_ratio=0.8,
                    sim3_iters=1,
                    max_samples_per_view_abs=500,
                    return_fused_debug=True,
                )

                # save ply if needed
                k = per_scene_saved[scene]
                if fused_debug is not None and k < max_sets_per_scene:
                    safe_scene = str(scene).replace("/", "_")
                    base = f"{safe_scene}_set{str(k).zfill(3)}"
                    gt_path = os.path.join(fused_out_dir, safe_scene, base + f"_GT_{fused_debug.chamfer_l1:.3f}.ply")
                    pr_path = os.path.join(fused_out_dir, safe_scene, base + f"_Pred_{fused_debug.chamfer_l1:.3f}.ply")
                    write_ply_xyzrgb(gt_path, fused_debug.gt_ds, fused_debug.gt_colors_ds)
                    write_ply_xyzrgb(pr_path, fused_debug.pr_ds, fused_debug.pr_colors_ds)
                    per_scene_saved[scene] += 1

                # append metrics to per_scene_results
                for k_metric in per_scene_results[scene].keys():
                    per_scene_results[scene][k_metric].append(float(metrics.get(k_metric, np.nan)))

        with open(os.path.join(args.output_dir, f"{benchmark_dataset_name}_per_scene_results.json"), "w") as f:
            json.dump(per_scene_results, f, indent=4)

        across_dataset_results = {}
        for scene in per_scene_results.keys():
            for metric in per_scene_results[scene].keys():
                across_dataset_results.setdefault(metric, [])
                across_dataset_results[metric].extend(per_scene_results[scene][metric])

        for metric in across_dataset_results.keys():
            across_dataset_results[metric] = float(np.nanmean(across_dataset_results[metric]))

        with open(os.path.join(args.output_dir, f"{benchmark_dataset_name}_avg_across_all_scenes.json"), "w") as f:
            json.dump(across_dataset_results, f, indent=4)

        print("Average results across all scenes for dataset: ", benchmark_dataset_name)
        for metric in across_dataset_results.keys():
            print(f"{metric}: {across_dataset_results[metric]}")

        per_dataset_results[benchmark_dataset_name] = across_dataset_results

    # average across datasets
    average_results = {}
    first = next(iter(per_dataset_results))
    for metric in per_dataset_results[first].keys():
        vals = [per_dataset_results[d][metric] for d in per_dataset_results]
        average_results[metric] = float(np.nanmean(vals))
    per_dataset_results["Average"] = average_results

    print("Benchmarking Done!")
    for metric in average_results.keys():
        print(f"{metric}: {average_results[metric]}")

    with open(os.path.join(args.output_dir, "per_dataset_results.json"), "w") as f:
        json.dump(per_dataset_results, f, indent=4)


@hydra.main(version_base=None, config_path="../../configs", config_name="dense_n_view_benchmark")
def execute_benchmarking(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)

    benchmark(cfg)


if __name__ == "__main__":
    execute_benchmarking()  # noqa
