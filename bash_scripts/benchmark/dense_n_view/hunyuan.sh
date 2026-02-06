#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

export HYDRA_FULL_ERROR=1

# Define the batch sizes and number of views to loop over
batch_sizes_and_views=(
    "20 2 benchmark_518_a3dscenes_whuomvs"
    "20 4 benchmark_518_a3dscenes_whuomvs"
    "10 8 benchmark_518_a3dscenes_whuomvs"
    "5 16 benchmark_518_a3dscenes_whuomvs"
    "3 24 benchmark_518_a3dscenes_whuomvs"
    "2 32 benchmark_518_a3dscenes_whuomvs"
    "1 50 benchmark_518_a3dscenes_whuomvs"
    "1 100 benchmark_518_a3dscenes_whuomvs"
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
        dataset.num_workers=4 \
        dataset.num_views=$num_views \
        batch_size=$batch_size \
        model=hunyuan \
        model.model_config.hf_model_name='checkpoints/HunyuanWorld-Mirror' \
        model/task=images_only \
        hydra.run.dir='${root_experiments_dir}/mapanything/benchmarking/dense_'"${num_views}"'_view/hunyuan'

    echo "Finished running $dataset with batch_size=$batch_size and num_views=$num_views"
done
