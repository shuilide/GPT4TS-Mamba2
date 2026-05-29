#!/bin/bash
# 实验二：单个数据集快速测试脚本
# 用法: ./run_single_dataset.sh <dataset_name> [gpu] [epochs]

set -e

# 切换到脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 创建 results 目录（如果不存在）
mkdir -p results

DATASET=${1:-Heartbeat}
GPU=${2:-0}
EPOCHS=${3:-100}

DATA_DIR=${4:-"../../datasets"}
EXP_DIR=$(pwd)

# 创建输出目录（setup() 函数要求 output_dir 必须存在）
mkdir -p "${EXP_DIR}/results/${DATASET}"

echo "=========================================="
echo "实验二：Gate 参数追踪 - 单数据集测试"
echo "=========================================="
echo "工作目录: $(pwd)"
echo "Dataset: ${DATASET}"
echo "GPU: ${GPU}"
echo "Epochs: ${EPOCHS}"
echo "=========================================="

python main_exp2.py \
    --data_dir "${DATA_DIR}/${DATASET}" \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --task classification \
    --gpu ${GPU} \
    --epochs ${EPOCHS} \
    --batch_size 32 \
    --lr 1e-4 \
    --d_model 768 \
    --patch_size 32 \
    --stride 16 \
    --gpt_layers 6 \
    --num_mamba_layers 2 \
    --pos_encoding learnable \
    --optimizer AdamW \
    --normalization standardization \
    --key_metric accuracy \
    --name "exp2_${DATASET}" \
    --no_timestamp \
    --output_dir "${EXP_DIR}/results/${DATASET}" \
    --seed 42

echo ""
echo "生成可视化..."
python visualize_gates.py --data_dir "${EXP_DIR}/results/${DATASET}/exp2_${DATASET}"

echo ""
echo "=========================================="
echo "实验完成！"
echo "结果目录: ${EXP_DIR}/results/${DATASET}/exp2_${DATASET}"
echo "Gate 追踪数据: ${EXP_DIR}/results/${DATASET}/exp2_${DATASET}/gate_tracking/"
echo "=========================================="
