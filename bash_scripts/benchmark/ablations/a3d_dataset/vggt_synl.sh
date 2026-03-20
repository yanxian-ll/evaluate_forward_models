#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

export HYDRA_FULL_ERROR=1


# Define the batch sizes and number of views to loop over
batch_sizes_and_views=(
    "15 2 benchmark_518_synl"
    "10 4 benchmark_518_synl"
    "5 8 benchmark_518_synl"
    "2 16 benchmark_518_synl"
    "1 24 benchmark_518_synl"
    "1 32 benchmark_518_synl"
)

# Loop through each combination
for combo in "${batch_sizes_and_views[@]}"; do
    # Split the string into batch_size and num_views
    read -r batch_size num_views dataset <<< "$combo"

    echo "Running $dataset with batch_size=$batch_size and num_views=$num_views"

    python3 \
        benchmarking/dense_n_view/benchmark.py \
        machine=aws \
        compute_abs_metrics=true \
        save_n_fused_ply=2 \
        dataset=$dataset \
        dataset.num_workers=4 \
        dataset.num_views=$num_views \
        batch_size=$batch_size \
        model=vggt \
        model.model_config.pretrained_model_name_or_path="checkpoints/vggt" \
        hydra.run.dir='${root_experiments_dir}/mapanything/benchmarking_ablations/dense_'"${num_views}"'_view/vggt_synl'

    echo "Finished running $dataset with batch_size=$batch_size and num_views=$num_views"
done
s