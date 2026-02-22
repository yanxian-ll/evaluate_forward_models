# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Training Executable for MapAnything

This script serves as the main entry point for training models in the MapAnything project.
It uses Hydra for configuration management and redirects all output to logging.

Usage:
    python train.py [hydra_options]
"""

import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from mapanything.train.training import train
from mapanything.utils.misc import StreamToLogger

# Disable torch hub download
import os
os.environ["TORCH_HUB_DISABLE_DOWNLOAD"] = "1"

# Set the cache directory for torch hub
import torch
torch.hub.set_dir("/opt/data/private/code/map-anything/checkpoints/torch_cache/hub")

# load local dino repo
LOCAL_DINO_REPO = "/opt/data/private/code/map-anything/checkpoints/torch_cache/hub/facebookresearch_dinov2_main"
_original_torch_hub_load = torch.hub.load
def offline_torch_hub_load(repo_or_dir, model, *args, **kwargs):
    if repo_or_dir == "facebookresearch/dinov2":
        print("Redirecting DINOv2 torch.hub.load to local repo")
        repo_or_dir = LOCAL_DINO_REPO
        kwargs["source"] = "local"
    return _original_torch_hub_load(repo_or_dir, model, *args, **kwargs)
torch.hub.load = offline_torch_hub_load

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="debug_train")
def execute_training(cfg: DictConfig):
    """
    Execute the training process with the provided configuration.

    Args:
        cfg (DictConfig): Configuration object loaded by Hydra
    """
    # Allow the config to be editable
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    # Redirect stdout and stderr to the logger
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)

    # Run the training
    train(cfg)


if __name__ == "__main__":
    execute_training()
