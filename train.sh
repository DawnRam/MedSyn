#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置Python路径和环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 设置分布式训练相关的环境变量
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 设置NCCL环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno1,ens102f0
export NCCL_P2P_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

# 获取GPU数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# 设置随机端口
MASTER_PORT=$(( ( RANDOM % 50000 )  + 10000 ))

# 运行训练脚本
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    --use_env \
    train.py \
    --config configs/image_diffusion.yaml \
    --model_type image 