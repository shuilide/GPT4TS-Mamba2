#!/bin/bash
# 实验五：架构消融实验运行脚本
# 对比四种不同架构模式

set -e

# 切换到脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 创建 results 目录
mkdir -p results

# 配置参数
GPU=5
EPOCHS=100
SEED=42
DATASET=${1:-"ERing"}  # 默认使用ERing数据集，可通过参数指定
DATA_DIR="../../datasets/${DATASET}"
OUTPUT_DIR="results"

# 数据集特定参数（可选）
LR_BASE=0.0005
PATCH_SIZE=16
STRIDE=2

echo "================================================================"
echo "EXPERIMENT 5: Architecture Ablation Study"
echo "Dataset: ${DATASET}"
echo "GPU: ${GPU}"
echo "Epochs: ${EPOCHS}"
echo "================================================================"

# 架构模式列表
ARCH_MODES=("sequential")

# 为每种架构模式运行实验
for MODE in "${ARCH_MODES[@]}"; do
    echo ""
    echo "================================================================"
    echo "Running architecture mode: ${MODE}"
    echo "================================================================"
    
    EXPERIMENT_NAME="${DATASET}_${MODE}"
    
    # 根据不同架构模式设置不同的学习率
    if [ "$MODE" = "pure_gpt" ]; then
        LR=0.0001
    elif [ "$MODE" = "pure_mamba" ]; then
        LR=0.0005
    elif [ "$MODE" = "sequential" ]; then
        LR=0.0005
    else
        LR=0.0001  # parallel
    fi
    
    # 运行实验
    python ../../src/main.py \
        --output_dir "${OUTPUT_DIR}" \
        --comment "Exp5: Architecture Ablation - ${MODE}" \
        --name "${EXPERIMENT_NAME}" \
        --records_file exp5_records.xls \
        --data_dir "${DATA_DIR}" \
        --data_class tsra \
        --pattern TRAIN \
        --val_pattern TEST \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --patch_size "${PATCH_SIZE}" \
        --stride "${STRIDE}" \
        --optimizer AdamW \
        --d_model 768 \
        --pos_encoding learnable \
        --task classification \
        --key_metric accuracy \
        --gpu "${GPU}" \
        --seed "${SEED}" \
        --gpt_layers 6 \
        --num_mamba_layers 2 \
        --no_attn_pooling \
        --arch_mode "${MODE}" \
        --no_timestamp
    
    echo "✓ Completed: ${MODE}"
done

echo ""
echo "================================================================"
echo "ALL ARCHITECTURE MODES COMPLETED"
echo "================================================================"
echo ""
echo "Results saved in: ${OUTPUT_DIR}/"
echo ""
echo "Next step: Run visualization"
echo "  python visualize_exp5.py --data_dir ${OUTPUT_DIR}"
echo ""
echo "================================================================"
