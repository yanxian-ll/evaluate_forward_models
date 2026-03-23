#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

NUM_GPUS=$1

# Logging Configs
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=INFO

module load cuda/12.4 nccl/2.18.3-cuda.12.1 nccl_efa/1.24.1-nccl.2.18.3-cuda.12.0 libfabric-aws/2.1.0amzn5.0 openmpi5/5.0.6

# AWS Multi-Node Configs
export OMP_NUM_THREADS=24
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_NET_CHUNKSIZE=524288

torchrun --nproc_per_node ${NUM_GPUS} \
  scripts/train.py \
  machine=aws \
  dataset=uavtrain_a3dall_518_many_ar_16ipg_2g \
  dataset.num_workers=4 \
  dataset.num_views=16 \
  dataset.principal_point_centered=true \
  loss=vggt_midmatch_loss \
  model=vggt_midmatch \
  model.model_config.pretrained_model_name_or_path="./checkpoints/vggt" \
  model.model_config.match_layer_indices='[11,17]' \
  model.model_config.match_inner_dim=256 \
  model.model_config.match_desc_dim=256 \
  model.model_config.match_fine_dim=128 \
  model.model_config.match_fine_stride=4 \
  model.model_config.match_split_frame_global=true \
  model.model_config.match_use_conf_head=true \
  model.model_config.match_use_fine_refine=true \
  train_params=vggt_finetune \
  train_params.epochs=10 \
  train_params.warmup_epochs=1 \
  train_params.accum_iter=8 \
  train_params.keep_freq=20 \
  train_params.max_num_of_imgs_per_gpu=16 \
  hydra.run.dir='${root_experiments_dir}/mapanything/training/vggt_midmatch_finetuning'
  