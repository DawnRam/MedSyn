#!/bin/bash

# 默认参数
NUM_GPUS=4
MODEL_TYPE="image"  # image/latent/stable
CONFIG="configs/image_diffusion.yaml"
DATA_DIR="/home/eechengyang/Data/ISIC"  # 更新数据路径
BATCH_SIZE=1
NUM_WORKERS=4
MIXED_PRECISION=true
ACCUMULATION_STEPS=1
WARMUP_STEPS=1000
LEARNING_RATE=1e-4
NUM_EPOCHS=100
WANDB_PROJECT="skin_lesion_synthesis"
WANDB_NAME=""
RESUME=""
EXTRA_ARGS=""
GPU_IDS="1,2,3,4" # 新增参数：指定GPU ID列表

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -h, --help                 显示帮助信息"
    echo "  -g, --gpus NUM_GPUS        使用的GPU数量 (默认: 1)"
    echo "  -m, --model MODEL_TYPE     模型类型: image/latent/stable (默认: image)"
    echo "  -c, --config CONFIG        配置文件路径 (默认: configs/image_diffusion.yaml)"
    echo "  -d, --data-dir DIR         数据集路径 (默认: /home/eechengyang/Data/ISIC)"
    echo "  -b, --batch-size SIZE      每个GPU的批次大小 (默认: 32)"
    echo "  -w, --workers NUM          每个GPU的数据加载线程数 (默认: 4)"
    echo "  --no-mixed-precision       禁用混合精度训练"
    echo "  -a, --accumulation STEPS   梯度累积步数 (默认: 1)"
    echo "  --warmup-steps STEPS       学习率预热步数 (默认: 1000)"
    echo "  -l, --lr RATE             学习率 (默认: 1e-4)"
    echo "  -e, --epochs NUM          训练轮数 (默认: 100)"
    echo "  -p, --project NAME        Weights & Biases项目名 (默认: skin_lesion_synthesis)"
    echo "  -n, --name NAME           Weights & Biases运行名称"
    echo "  -r, --resume PATH         从检查点恢复训练"
    echo "  -i, --gpu-ids IDS          指定要使用的GPU ID列表 (例如: '0,1,3')"
    echo "  -- EXTRA_ARGS             传递给训练脚本的额外参数"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -g|--gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -w|--workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --no-mixed-precision)
            MIXED_PRECISION=false
            shift
            ;;
        -a|--accumulation)
            ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --warmup-steps)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        -l|--lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -e|--epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        -p|--project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        -n|--name)
            WANDB_NAME="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME="$2"
            shift 2
            ;;
        -i|--gpu-ids) # 处理新的GPU ID参数
            GPU_IDS="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS="$@"
            break
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 验证参数
if [[ ! "$MODEL_TYPE" =~ ^(image|latent|stable)$ ]]; then
    echo "错误: 无效的模型类型 '$MODEL_TYPE'"
    echo "支持的模型类型: image, latent, stable"
    exit 1
fi

# 验证数据路径
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据路径 '$DATA_DIR' 不存在"
    exit 1
fi

# 设置默认的wandb运行名称
if [ -z "$WANDB_NAME" ]; then
    WANDB_NAME="${MODEL_TYPE}_diffusion_$(date +%Y%m%d_%H%M%S)"
fi

# 构建训练命令
TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS train.py \
    --config $CONFIG \
    --model_type $MODEL_TYPE"

# 添加额外参数
if [ ! -z "$RESUME" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME"
fi

# 构建环境变量
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_NAME="$WANDB_NAME"
export TRAIN_DATA_DIR="$DATA_DIR"  # 添加数据路径环境变量
export TRAIN_BATCH_SIZE="$BATCH_SIZE"
export TRAIN_NUM_WORKERS="$NUM_WORKERS"
export TRAIN_MIXED_PRECISION="$MIXED_PRECISION"
export TRAIN_ACCUMULATION_STEPS="$ACCUMULATION_STEPS"
export TRAIN_WARMUP_STEPS="$WARMUP_STEPS"
export TRAIN_LEARNING_RATE="$LEARNING_RATE"
export TRAIN_NUM_EPOCHS="$NUM_EPOCHS"

# 如果指定了GPU ID，则设置CUDA_VISIBLE_DEVICES环境变量
if [ ! -z "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    # 根据指定的GPU数量更新NUM_GPUS
    NUM_GPUS=$(echo "$GPU_IDS" | tr -cd ',' | wc -c)
    NUM_GPUS=$((NUM_GPUS + 1))
fi

# 打印训练设置
echo "训练设置:"
echo "----------------------------------------"
echo "GPU数量: $NUM_GPUS"
echo "模型类型: $MODEL_TYPE"
echo "配置文件: $CONFIG"
echo "数据路径: $DATA_DIR"
echo "批次大小: $BATCH_SIZE (每个GPU)"
echo "数据加载线程数: $NUM_WORKERS (每个GPU)"
echo "混合精度训练: $MIXED_PRECISION"
echo "梯度累积步数: $ACCUMULATION_STEPS"
echo "学习率预热步数: $WARMUP_STEPS"
echo "学习率: $LEARNING_RATE"
echo "训练轮数: $NUM_EPOCHS"
echo "W&B项目名: $WANDB_PROJECT"
echo "W&B运行名称: $WANDB_NAME"
if [ ! -z "$RESUME" ]; then
    echo "从检查点恢复: $RESUME"
fi
echo "----------------------------------------"

# 打印环境变量值 (用于调试)
echo "[DEBUG] TRAIN_DATA_DIR environment variable: $TRAIN_DATA_DIR"

# 执行训练命令
echo "开始训练..."
$TRAIN_CMD $EXTRA_ARGS 