#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

export HYDRA_FULL_ERROR=1

# Define the batch sizes and number of views to loop over
batch_sizes_and_views=(
    "2 32 benchmark_518_a3dscenes_whuomvs"
    "20 2 benchmark_518_a3dscenes_whuomvs"
    "20 4 benchmark_518_a3dscenes_whuomvs"
    "10 8 benchmark_518_a3dscenes_whuomvs"
    "5 16 benchmark_518_a3dscenes_whuomvs"
    "3 24 benchmark_518_a3dscenes_whuomvs"
)

# Loop through each combination
for combo in "${batch_sizes_and_views[@]}"; do
    # Split the string into batch_size and num_views
    read -r batch_size num_views dataset <<< "$combo"

    echo "Running $dataset with batch_size=$batch_size and num_views=$num_views"

    python3 \
        benchmarking/dense_n_view/benchmark.py \
        machine=aws \
        dataset=$dataset \
        dataset.num_workers=12 \
        dataset.num_views=$num_views \
        batch_size=$batch_size \
        model=mapanything_v1 \
        model/task=images_only \
        model.encoder.uses_torch_hub=false \
        model.pretrained='${root_experiments_dir}/mapanything/training_ablations/mvs_depth/checkpoint-last.pth' \
        hydra.run.dir='${root_experiments_dir}/mapanything/benchmarking_ablations/dense_'"${num_views}"'_view/mvs_depth'

    echo "Finished running $dataset with batch_size=$batch_size and num_views=$num_views"
done
