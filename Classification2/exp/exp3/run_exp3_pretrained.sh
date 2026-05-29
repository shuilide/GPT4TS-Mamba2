#!/bin/bash
# 实验3：预训练知识迁移验证 (Pre-trained Knowledge Transfer Study)
# 对比：加载GPT-2预训练权重 vs 随机初始化GPT-2权重
# 所有实验都冻结GPT-2主体，只训练10%参数

set -e

# 切换到脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 创建 results 目录
mkdir -p results

echo "=========================================="
echo "实验3：预训练知识迁移验证"
echo "=========================================="
echo "工作目录: $(pwd)"
echo "GPU: 7"
echo "数据集: AtrialFibrillation (快速测试)"
echo "训练轮数: 100 epochs"
echo "=========================================="

# 定义测试数据集（7个 UCR 数据集）
DATASETS=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket" "Epilepsy" "ERing")

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "========== 处理数据集: $DATASET =========="

    LR=0.0005
    PATCH_SIZE=16
    STRIDE=2

    
    # ========== 实验3-A: Pre-trained（加载预训练权重）==========
    echo ""
    echo ">>> 运行: $DATASET (Pre-trained)"
    python ../../src/main.py \
        --output_dir results \
        --comment "Exp3: Pre-trained GPT-2" \
        --name "${DATASET}_pretrained" \
        --records_file exp3_records.xls \
        --data_dir ../../datasets/$DATASET \
        --data_class tsra \
        --pattern TRAIN \
        --val_pattern TEST \
        --epochs 100 \
        --lr $LR \
        --patch_size $PATCH_SIZE \
        --stride $STRIDE \
        --optimizer AdamW \
        --d_model 768 \
        --pos_encoding learnable \
        --task classification \
        --key_metric accuracy \
        --gpu 7 \
        --seed 42 \
        --gpt_layers 6 \
        --num_mamba_layers 2 \
        --no_timestamp
    # ========== 实验3-B: From Scratch（随机初始化）==========
    echo ""
    echo ">>> 运行: $DATASET (From Scratch)"
    python ../../src/main.py \
        --output_dir results \
        --comment "Exp3: Random Init GPT-2" \
        --name "${DATASET}_scratch" \
        --records_file exp3_records.xls \
        --data_dir ../../datasets/$DATASET \
        --data_class tsra \
        --pattern TRAIN \
        --val_pattern TEST \
        --epochs 100 \
        --lr $LR \
        --patch_size $PATCH_SIZE \
        --stride $STRIDE \
        --optimizer AdamW \
        --d_model 768 \
        --pos_encoding learnable \
        --task classification \
        --key_metric accuracy \
        --gpu 7 \
        --seed 42 \
        --gpt_layers 6 \
        --num_mamba_layers 2 \
        --no_pretrained \
        --no_timestamp
done

echo ""
echo "=========================================="
echo "实验3训练完成！"
echo "结果目录: results/"
echo "=========================================="
echo ""
echo "检查生成的文件..."
ls -lh results/
echo ""
echo "现在生成可视化图表..."
python visualize_convergence.py --data_dir results

echo ""
echo "=========================================="
echo "实验3全部完成！"
echo "=========================================="
