#!/bin/bash
# 实验3：小样本学习能力验证 (Few-Shot Learning Study)
# 使用截断后的训练集，验证预训练模型在小样本情况下的优势

set -e

# 切换到脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 创建目录
mkdir -p results
mkdir -p truncated_data

echo "=========================================="
echo "实验3：小样本学习能力验证"
echo "=========================================="
echo "工作目录: $(pwd)"
echo "GPU: 7"
echo "数据集: RacketSports"
echo "截断比例: 20%, 40%, 60%, 80%, 100%"
echo "=========================================="

# 配置
DATASET="RacketSports"
ORIGINAL_DATA_DIR="../../datasets/$DATASET"
BASE_OUTPUT="results"

# 定义截断比例
RATIOS=(0.2 0.4 0.6 0.8 1.0)
RATIO_NAMES=("20pct" "40pct" "60pct" "80pct" "100pct")
RATIO_PCT=("20%" "40%" "60%" "80%" "100%")

for i in "${!RATIOS[@]}"; do
    RATIO=${RATIOS[$i]}
    RATIO_NAME=${RATIO_NAMES[$i]}
    RATIO_DISPLAY=${RATIO_PCT[$i]}
    
    echo ""
    echo "=========================================="
    echo "处理截断比例: ${RATIO_NAME} (${RATIO_DISPLAY})"
    echo "=========================================="
    
    # 1. 截断数据集
    TRUNCATED_DIR="truncated_data/${DATASET}_${RATIO_NAME}"
    
    if [ "$RATIO" == "1.0" ]; then
        # 100% 使用原始数据
        TRUNCATED_DIR="$ORIGINAL_DATA_DIR"
        echo ">>> 使用完整数据集（100%）"
    else
        echo ">>> 截断数据集到 ${RATIO_DISPLAY}..."
        python truncate_dataset.py \
            --data_dir "$ORIGINAL_DATA_DIR" \
            --output_dir "$TRUNCATED_DIR" \
            --ratio $RATIO \
            --seed 42
    fi
    
    # 2. 运行 Pre-trained 版本
    echo ""
    echo ">>> 运行: ${DATASET}_${RATIO_NAME}_pretrained"
    python ../../src/main.py \
        --output_dir "$BASE_OUTPUT" \
        --comment "Exp3 FewShot: Pre-trained GPT-2 (${RATIO_NAME})" \
        --name "${DATASET}_${RATIO_NAME}_pretrained" \
        --records_file exp3_fewshot_records.xls \
        --data_dir "$TRUNCATED_DIR" \
        --data_class tsra \
        --pattern TRAIN \
        --val_pattern TEST \
        --epochs 100 \
        --lr 0.0005 \
        --patch_size 16 \
        --stride 2 \
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

done

echo ""
echo "=========================================="
echo "实验3小样本实验完成！"
echo "结果目录: results/"
echo "=========================================="
echo ""
echo "检查生成的文件..."
ls -lh results/ | grep RacketSports
echo ""
echo "现在生成可视化图表..."
python visualize_fewshot.py --data_dir results

echo ""
echo "=========================================="
echo "实验3全部完成！"
echo "=========================================="
