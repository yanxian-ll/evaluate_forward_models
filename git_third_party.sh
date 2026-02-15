#!/bin/bash

mkdir -p third_party
cd third_party

# 1) COLMAP extras
git clone https://github.com/cvg/LightGlue.git

# 2) Benchmark models
git clone https://github.com/javrtg/AnyCalib.git

git clone https://github.com/naver/croco.git
cd croco
git checkout croco_module
cd ..

git clone https://github.com/naver/dust3r.git
cd dust3r
git checkout dust3r_setup
cd ..

git clone https://github.com/Nik-V9/mast3r.git
git clone https://github.com/naver/must3r.git
git clone https://github.com/yyfz/Pi3.git
git clone https://github.com/Nik-V9/pow3r.git

# need change src code (included in thirdparty)
# git clone --recurse-submodules https://github.com/Nik-V9/depth-anything-3.git
# cd depth-anything-3
# git submodule update --init --recursive
# cd ..

# need change src code (included in thirdparty)
# git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror

git clone https://github.com/infinity1096/robustmvd.git

# must3r dependency
git clone https://github.com/lojzezust/asmk.git

# wai_processing dependency
git clone https://github.com/NVlabs/nvdiffrast.git
git clone https://github.com/microsoft/MoGe.git
