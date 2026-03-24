import copy
from typing import Dict, Tuple, Union, List

import torch

from mapanything.models.mapanything.model import MapAnything
from uniception.models.prediction_heads.base import PredictionHeadTokenInput
from uniception.models.prediction_heads.mlp_head import MLPHead


class MapAnythingDA3CameraHead(MapAnything):
    """
    MapAnything variant with a lightweight global camera head inspired by the
    DA3 camera decoder path.

    The added head consumes the same global token that is already used by the
    metric scale head and predicts per-sample global focal parameters. The
    dense ray/depth/pose heads remain unchanged.
    """

    def __init__(
        self,
        name: str,
        encoder_config: Dict,
        info_sharing_config: Dict,
        pred_head_config: Dict,
        geometric_input_config: Dict,
        camera_head_config: Dict = None,
        camera_mode: str = "fx_fy",
        principal_point_mode: str = "image_center",
        camera_log_scale_min: float = -6.0,
        camera_log_scale_max: float = 6.0,
        **kwargs,
    ):
        self.camera_head_config = copy.deepcopy(camera_head_config or {"output_dim": 2})
        self.camera_mode = camera_mode
        self.principal_point_mode = principal_point_mode
        self.camera_log_scale_min = camera_log_scale_min
        self.camera_log_scale_max = camera_log_scale_max
        self._last_camera_prediction = None

        super().__init__(
            name=name,
            encoder_config=encoder_config,
            info_sharing_config=info_sharing_config,
            pred_head_config=pred_head_config,
            geometric_input_config=geometric_input_config,
            **kwargs,
        )

        self.class_init_args.update(
            {
                "camera_head_config": self.camera_head_config,
                "camera_mode": self.camera_mode,
                "principal_point_mode": self.principal_point_mode,
                "camera_log_scale_min": self.camera_log_scale_min,
                "camera_log_scale_max": self.camera_log_scale_max,
            }
        )

    def _initialize_prediction_heads(self, pred_head_config):
        super()._initialize_prediction_heads(pred_head_config)
        camera_head_config = copy.deepcopy(self.camera_head_config)
        camera_head_config["input_feature_dim"] = self.info_sharing.dim
        self.camera_head = MLPHead(**camera_head_config)

    def _decode_camera_head(self, scale_head_inputs: torch.Tensor, img_shape: Tuple[int, int]):
        camera_head_output = self.camera_head(
            PredictionHeadTokenInput(last_feature=scale_head_inputs)
        )
        camera_raw = camera_head_output.decoded_channels

        if camera_raw.ndim == 4:
            camera_raw = camera_raw.mean(dim=(-1, -2))
        elif camera_raw.ndim == 3:
            camera_raw = camera_raw.mean(dim=1)
        elif camera_raw.ndim == 2:
            pass
        else:
            camera_raw = camera_raw.reshape(camera_raw.shape[0], -1)

        camera_raw = camera_raw.reshape(camera_raw.shape[0], -1)
        if self.camera_mode == "iso_f":
            camera_log_focal_norm = camera_raw[:, :1].repeat(1, 2)
        elif self.camera_mode == "fx_fy":
            camera_log_focal_norm = camera_raw[:, :2]
        else:
            raise ValueError(
                f"Unsupported camera_mode={self.camera_mode}. Valid options are ['fx_fy', 'iso_f']."
            )

        camera_log_focal_norm = camera_log_focal_norm.clamp(
            min=self.camera_log_scale_min,
            max=self.camera_log_scale_max,
        )

        height, width = img_shape
        size_xy = camera_log_focal_norm.new_tensor([float(width), float(height)])
        camera_focal_px = torch.exp(camera_log_focal_norm) * size_xy.unsqueeze(0)

        intrinsics = torch.zeros(
            camera_focal_px.shape[0], 3, 3, dtype=camera_focal_px.dtype, device=camera_focal_px.device
        )
        intrinsics[:, 0, 0] = camera_focal_px[:, 0]
        intrinsics[:, 1, 1] = camera_focal_px[:, 1]
        intrinsics[:, 2, 2] = 1.0

        if self.principal_point_mode == "image_center":
            intrinsics[:, 0, 2] = (width - 1.0) * 0.5
            intrinsics[:, 1, 2] = (height - 1.0) * 0.5
        else:
            raise ValueError(
                f"Unsupported principal_point_mode={self.principal_point_mode}. Only 'image_center' is currently implemented."
            )

        return {
            "camera_log_focal_norm": camera_log_focal_norm,
            "camera_focal_px": camera_focal_px,
            "camera_intrinsics_cam_head": intrinsics,
        }

    def downstream_head(
        self,
        dense_head_inputs: Union[torch.Tensor, List[torch.Tensor]],
        scale_head_inputs: torch.Tensor,
        img_shape: Tuple[int, int],
        memory_efficient_inference: bool = False,
        minibatch_size: int = None,
    ):
        dense_final_outputs, pose_final_outputs, scale_final_output = super().downstream_head(
            dense_head_inputs=dense_head_inputs,
            scale_head_inputs=scale_head_inputs,
            img_shape=img_shape,
            memory_efficient_inference=memory_efficient_inference,
            minibatch_size=minibatch_size,
        )
        self._last_camera_prediction = self._decode_camera_head(
            scale_head_inputs=scale_head_inputs,
            img_shape=img_shape,
        )
        return dense_final_outputs, pose_final_outputs, scale_final_output

    def forward(self, views, memory_efficient_inference=False, minibatch_size=None):
        self._last_camera_prediction = None
        preds = super().forward(
            views,
            memory_efficient_inference=memory_efficient_inference,
            minibatch_size=minibatch_size,
        )
        if self._last_camera_prediction is None:
            return preds

        for pred in preds:
            pred.update(self._last_camera_prediction)
        return preds
