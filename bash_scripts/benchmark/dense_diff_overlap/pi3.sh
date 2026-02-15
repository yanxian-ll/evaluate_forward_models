#!/bin/bash
export HYDRA_FULL_ERROR=1

# 固定设置
dataset="benchmark_518_a3dscenes_whuomvs"
num_views=16
batch_size=4
num_workers=4

# 不同重叠度区间
covis_ranges=(
  "0.05 0.25 cov_005_025"
  "0.25 0.50 cov_025_050"
  "0.50 0.75 cov_050_075"
  "0.75 0.10 cov_075_100"
)

for combo in "${covis_ranges[@]}"; do
  read -r cov_min cov_max tag <<< "$combo"

  echo "Running $dataset with views=$num_views batch_size=$batch_size cov=[${cov_min}, ${cov_max}]"

  python3 benchmarking/dense_n_view/benchmark.py \
    machine=aws \
    dataset=$dataset \
    dataset.num_workers=$num_workers \
    dataset.num_views=$num_views \
    dataset.covisibility_thres_min=$cov_min \
    dataset.covisibility_thres_max=$cov_max \
    batch_size=$batch_size \
    model=pi3 \
    model.model_config.pretrained_model_name_or_path="checkpoints/pi3" \
    hydra.run.dir="\${root_experiments_dir}/mapanything/benchmarking/dense_overlap_${tag}/pi3"

  echo "Finished $dataset cov=[${cov_min}, ${cov_max}]"
done
