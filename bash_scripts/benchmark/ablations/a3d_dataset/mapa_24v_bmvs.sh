#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

export HYDRA_FULL_ERROR=1

# Define the batch sizes and number of views to loop over
batch_sizes_and_views=(
    "25 2 benchmark_518_bmvs"
    "10 4 benchmark_518_bmvs"
    "5 8 benchmark_518_bmvs"
    "2 16 benchmark_518_bmvs"
    "2 24 benchmark_518_bmvs"
    "1 32 benchmark_518_bmvs"
)

# Loop through each combination
for combo in "${batch_sizes_and_views[@]}"; do
    # Split the string into batch_size and num_views
    read -r batch_size num_views dataset <<< "$combo"

    echo "Running $dataset with batch_size=$batch_size and num_views=$num_views"

    python3 \
        benchmarking/dense_n_view/benchmark.py \
        machine=aws \
        compute_abs_metrics=false \
        save_n_fused_ply=1 \
        dataset=$dataset \
        dataset.num_workers=12 \
        dataset.num_views=$num_views \
        batch_size=$batch_size \
        model=mapanything \
        model/task=images_only \
        model.encoder.uses_torch_hub=false \
        model.pretrained='checkpoints/map-anything/map-anything.pth' \
        hydra.run.dir='${root_experiments_dir}/mapanything/benchmarking_ablations/dense_'"${num_views}"'_view/mapa_24v_bmvs'

    echo "Finished running $dataset with batch_size=$batch_size and num_views=$num_views"
done
