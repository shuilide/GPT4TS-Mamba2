#!/bin/bash
# 实验六：GPT-2冻结策略实验运行脚本
# 对比三种不同的冻结策略

set -e

# 配置参数
GPU=4
EPOCHS=100
SEED=42
DATASET=${1:-"Heartbeat"}  # 默认使用Heartbeat数据集，可通过参数指定
DATA_DIR="../../datasets/${DATASET}"
OUTPUT_DIR="results"

echo "================================================================"
echo "EXPERIMENT 6: GPT-2 Freezing Strategy Study"
echo "Dataset: ${DATASET}"
echo "GPU: ${GPU}"
echo "Epochs: ${EPOCHS}"
echo "================================================================"

# 冻结策略列表
FREEZE_STRATEGIES=("all_frozen" "partial_frozen" "all_finetune")

# 为每种冻结策略运行实验
for STRATEGY in "${FREEZE_STRATEGIES[@]}"; do
    echo ""
    echo "================================================================"
    echo "Running freeze strategy: ${STRATEGY}"
    echo "================================================================"
    
    EXPERIMENT_NAME="${DATASET}_${STRATEGY}"
    
    # 根据冻结策略设置不同的学习率
    if [ "$STRATEGY" = "all_frozen" ]; then
        LR=0.0001
    elif [ "$STRATEGY" = "partial_frozen" ]; then
        LR=0.0005
    else
        LR=0.0001  # all_finetune需要更小的学习率
    fi
    
    # 运行实验（使用 main_exp6.py 而不是 src/main.py）
    python main_exp6.py \
        --output_dir "${OUTPUT_DIR}" \
        --name "${EXPERIMENT_NAME}" \
        --data_dir "${DATA_DIR}" \
        --data_class tsra \
        --pattern TRAIN \
        --val_pattern TEST \
        --gpu "${GPU}" \
        --seed "${SEED}" \
        --epochs "${EPOCHS}" \
        --gpt_layers 6 \
        --num_mamba_layers 2 \
        --lr "${LR}" \
        --patch_size 16 \
        --stride 8 \
        --optimizer AdamW \
        --d_model 768 \
        --pos_encoding learnable \
        --task classification \
        --key_metric accuracy \
        --no_timestamp \
        --freeze_strategy "${STRATEGY}"
    
    echo "✓ Completed: ${STRATEGY}"
done

echo ""
echo "================================================================"
echo "ALL FREEZE STRATEGIES COMPLETED"
echo "================================================================"
echo ""
echo "Results saved in: ${OUTPUT_DIR}/"
echo ""
echo "Next step: Run visualization"
echo "  python visualize_exp6.py --data_dir ${OUTPUT_DIR}"
echo ""
echo "================================================================"

