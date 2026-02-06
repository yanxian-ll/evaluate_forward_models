#!/bin/bash
export HYDRA_FULL_ERROR=1
export OPENCV_IO_ENABLE_OPENEXR=1

python3 scripts/run_depth_completion.py \
  machine=aws \
  dataset_root='${root_data_dir}/a3dscenes' \
  scene_name='csu2026' \
  resolution='[1024,672]' \
  batch_size=2 \
  align_mode=affine \
  model=mapanything_v1 \
  model/task=depth_completion \
  model.pretrained='${root_experiments_dir}/mapanything/training/mapa_finetuning_depth_completion/checkpoint-last.pth' \
  hydra.run.dir='${root_experiments_dir}/mapanything/calibration_benchmarking/single_scene/csu2026'
