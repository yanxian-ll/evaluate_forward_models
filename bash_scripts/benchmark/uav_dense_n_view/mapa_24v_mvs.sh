#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

export HYDRA_FULL_ERROR=1

# batch size, views, dataset, seed
batch_sizes_and_views=(
    "15 2 benchmark_518_synl 2"
    "10 4 benchmark_518_synl 4"
    "5 8 benchmark_518_synl 8"
    "2 16 benchmark_518_synl 16"
    "1 24 benchmark_518_synl 24"
    "1 32 benchmark_518_synl 32"
)


# Loop through each combination
for combo in "${batch_sizes_and_views[@]}"; do
    # Split the string into batch_size and num_views
    read -r batch_size num_views dataset seed <<< "$combo"
    
    echo "Running $dataset with batch_size=$batch_size and num_views=$num_views, seed=$seed"

    python3 \
        benchmarking/dense_n_view/benchmark.py \
        machine=aws \
        seed=$seed \
        compute_abs_metrics=true \
        save_n_fused_ply=3 \
        dataset=$dataset \
        dataset.num_workers=12 \
        dataset.num_views=$num_views \
        batch_size=$batch_size \
        model=mapanything \
        model/task=mvs \
        model.encoder.uses_torch_hub=false \
        model.pretrained='checkpoints/map-anything/map-anything.pth' \
        hydra.run.dir='${root_experiments_dir}/mapanything/benchmarking/dense_'"${num_views}"'_view/mapa_24v_mvs'

    echo "Finished running $dataset with batch_size=$batch_size and num_views=$num_views"
done

