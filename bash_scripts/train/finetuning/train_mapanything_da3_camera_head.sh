#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash bash_scripts/train_mapanything_da3_camera_head.sh
#   bash bash_scripts/train_mapanything_da3_camera_head.sh dataset.num_views=6 train_params.max_num_of_imgs_per_gpu=6

python scripts/train.py --config-name debug_train_da3_camera_head "$@"
