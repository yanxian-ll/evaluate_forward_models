"""
WHUMVS LuoJiaMVS WHUOMVS Dataset using WAI format data.
"""

import os
import json

import torch
import cv2
import numpy as np

from mapanything.datasets.base.base_dataset import BaseDataset
from mapanything.datasets.wai.a3dscenes import A3DScenesWAI

from mapanything.utils.wai.core import load_data, load_frame
from mapanything.datasets.utils.csr_utils import _csr_sampling, _load_covis_graph 


class WHULuoJiaWHUOMVSWAI(A3DScenesWAI):
    """
    WHUMVS LuoJiaMVS WHUOMVS dataset containing object-centric and birds-eye-view scenes.
    """

    def __init__(
        self,
        *args,
        ROOT,
        dataset_metadata_dir,
        split,
        overfit_num_sets=None,
        sample_specific_scene: bool = False,
        specific_scene_name: str = None,
        load_modalities: list = ["image", "depth"],
        covisibility_thres_max: float = 1.0,
        sampling_mode: str = "random_walk",
        walk_restart_prob: float = 0.10,
        walk_temperature: float = 1.0,
        walk_topk_step: int = 50,
        **kwargs,
    ):
        super().__init__(
            *args, 
            ROOT=ROOT,
            dataset_metadata_dir=dataset_metadata_dir,
            split=split,
            overfit_num_sets=overfit_num_sets,
            sample_specific_scene=sample_specific_scene,
            specific_scene_name=specific_scene_name,
            load_modalities=load_modalities,
            covisibility_thres_max=covisibility_thres_max,
            sampling_mode=sampling_mode,
            walk_restart_prob=walk_restart_prob,
            walk_temperature=walk_temperature,
            walk_topk_step=walk_topk_step,
            **kwargs
        )
        # Indicate synthetic dataset
        self.is_synthetic = True
        self.is_metric_scale = True

    def _load_data(self):
        split_metadata_path = os.path.join(
            self.dataset_metadata_dir,
            self.split,
            f"whu_luojia_whuomvs_scene_list_{self.split}.npy",
        )
        split_scene_list = np.load(split_metadata_path, allow_pickle=True)

        if not self.sample_specific_scene:
            self.scenes = list(split_scene_list)
        else:
            self.scenes = [self.specific_scene_name]
        self.num_of_scenes = len(self.scenes)


    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        """
        Get views for a given scene index using specified sampling mode.
        
        Args:
            sampled_idx: Scene index.
            num_views_to_sample: Number of views to sample.
            resolution: Target image resolution.
            sampling_mode: Sampling mode, "random_walk" or "greedy_chain".
            use_bidirectional_covis: Whether to use bidirectional edge weights.
            
        Returns:
            List of view dictionaries.
        """
        scene_index = sampled_idx
        scene_name = self.scenes[scene_index]
        scene_root = os.path.join(self.ROOT, scene_name)

        scene_meta = load_data(os.path.join(scene_root, "scene_meta.json"), "scene_meta")
        scene_file_names = list(scene_meta["frame_names"].keys())
        num_views_in_scene = len(scene_file_names)

        # Load view graph for sampling
        g_view = _load_covis_graph(scene_root, scene_meta)

        # Sample view indices using specified sampling mode
        view_indices = self._sample_view_indices(
            num_views_to_sample=num_views_to_sample,
            num_views_in_scene=num_views_in_scene,
            view_covis_graph=g_view,
        )

        # Load frames for selected indices
        views = []
        for view_index in view_indices:
            view_file_name = scene_file_names[int(view_index)]
            view_data = load_frame(
                scene_root,
                view_file_name,
                # modalities=["image", "depth"],
                modalities=self.load_modalities,
                scene_meta=scene_meta,
            )

            raw_image = view_data["image"].permute(1, 2, 0).numpy()  # (H,W,3)
            raw_image = (raw_image * 255).astype(np.uint8)

            depthmap = view_data["depth"].numpy().astype(np.float32)
            intrinsics = view_data["intrinsics"].numpy().astype(np.float32)
            c2w_pose = view_data["extrinsics"].numpy().astype(np.float32)
            
            depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
            # Generate valid mask from depthmap
            if "mask" not in view_data:
                view_data["mask"] = torch.tensor(depthmap > 0.0, device=view_data["depth"].device)

            non_ambiguous_mask = view_data["mask"].numpy().astype(int)
            non_ambiguous_mask = cv2.resize(
                non_ambiguous_mask,
                (raw_image.shape[1], raw_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            depthmap = np.where(non_ambiguous_mask, depthmap, 0)

            additional_quantities_to_resize = [non_ambiguous_mask]
            image, depthmap, intrinsics, additional_quantities_to_resize = (
                self._crop_resize_if_necessary(
                    image=raw_image,
                    resolution=resolution,
                    depthmap=depthmap,
                    intrinsics=intrinsics,
                    additional_quantities=additional_quantities_to_resize,
                )
            )
            non_ambiguous_mask = additional_quantities_to_resize[0]

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=c2w_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    non_ambiguous_mask=non_ambiguous_mask,
                    dataset="WHULuojiaWHUOMVS",
                    label=scene_name,
                    instance=os.path.join("images", str(view_file_name)),
                )
            )

        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd", "--root_dir", default="../../dataset/data/whu_luojia_whuomvs", type=str
    )
    parser.add_argument(
        "-dmd",
        "--dataset_metadata_dir",
        default="../../dataset/metadata",
        type=str,
    )
    parser.add_argument(
        "-nv",
        "--num_of_views",
        default=16,
        type=int,
    )
    parser.add_argument("--viz", action="store_true", default=False)

    return parser


if __name__ == "__main__":
    import rerun as rr
    from tqdm import tqdm

    from mapanything.datasets.base.base_dataset import view_name
    from mapanything.utils.image import rgb
    from mapanything.utils.viz import script_add_rerun_args

    parser = get_parser()
    script_add_rerun_args(
        parser
    )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    dataset = WHULuoJiaWHUOMVSWAI(
        num_views=args.num_of_views,
        split="train",
        covisibility_thres=0.1,
        covisibility_thres_max=1.0,
        ROOT=args.root_dir,
        dataset_metadata_dir=args.dataset_metadata_dir,
        resolution=(518, 392),
        aug_crop=16,
        transform="colorjitter+grayscale+gaublur",
        data_norm_type="dinov2",
        sampling_mode="greedy_chain",  # "random_walk" or "greedy_chain"
    )
    print(dataset.get_stats())

    if args.viz:
        rr.script_setup(args, "WHULuoJiaWHUOMVS_Dataloader")
        rr.set_time("stable_time", sequence=0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    sampled_indices = np.random.choice(len(dataset), size=10, replace=False)

    for num, idx in enumerate(tqdm(sampled_indices)):
        views = dataset[idx]
        assert len(views) == args.num_of_views
        sample_name = f"{idx}"
        for view_idx in range(args.num_of_views):
            sample_name += f" {view_name(views[view_idx])}"
        print(sample_name)
        for view_idx in range(args.num_of_views):
            image = rgb(
                views[view_idx]["img"], norm_type=views[view_idx]["data_norm_type"]
            )
            depthmap = views[view_idx]["depthmap"]
            pose = views[view_idx]["camera_pose"]
            intrinsics = views[view_idx]["camera_intrinsics"]
            pts3d = views[view_idx]["pts3d"]
            valid_mask = views[view_idx]["valid_mask"]
            if "non_ambiguous_mask" in views[view_idx]:
                non_ambiguous_mask = views[view_idx]["non_ambiguous_mask"]
            else:
                non_ambiguous_mask = None
            if "prior_depth_along_ray" in views[view_idx]:
                prior_depth_along_ray = views[view_idx]["prior_depth_along_ray"]
            else:
                prior_depth_along_ray = None
            if args.viz:
                rr.set_time("stable_time", sequence=num)
                base_name = f"world/view_{view_idx}"
                pts_name = f"world/view_{view_idx}_pointcloud"
                # Log camera info and loaded data
                height, width = image.shape[0], image.shape[1]
                rr.log(
                    base_name,
                    rr.Transform3D(
                        translation=pose[:3, 3],
                        mat3x3=pose[:3, :3],
                    ),
                )
                rr.log(
                    f"{base_name}/pinhole",
                    rr.Pinhole(
                        image_from_camera=intrinsics,
                        height=height,
                        width=width,
                        camera_xyz=rr.ViewCoordinates.RDF,
                    ),
                )
                rr.log(
                    f"{base_name}/pinhole/rgb",
                    rr.Image(image),
                )
                rr.log(
                    f"{base_name}/pinhole/depth",
                    rr.DepthImage(depthmap),
                )
                if prior_depth_along_ray is not None:
                    rr.log(
                        f"prior_depth_along_ray_{view_idx}",
                        rr.DepthImage(prior_depth_along_ray),
                    )
                if non_ambiguous_mask is not None:
                    rr.log(
                        f"{base_name}/pinhole/non_ambiguous_mask",
                        rr.SegmentationImage(non_ambiguous_mask.astype(int)),
                    )
                # Log points in 3D
                filtered_pts = pts3d[valid_mask]
                filtered_pts_col = image[valid_mask]
                rr.log(
                    pts_name,
                    rr.Points3D(
                        positions=filtered_pts.reshape(-1, 3),
                        colors=filtered_pts_col.reshape(-1, 3),
                    ),
                )

     