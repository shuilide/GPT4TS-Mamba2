#!/bin/bash
# 实验4：模型可解释性可视化 (Interpretability & Visualization)
# 需要提供一个已经训练好的模型目录

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "实验4：模型可解释性可视化"
echo "=========================================="

if [ -z "$1" ]; then
    echo "用法: bash run_exp4.sh <训练好的模型目录路径>"
    echo "例如: bash run_exp4.sh ../../results/BasicMotions_pretrained"
    exit 1
fi

MODEL_DIR=$1
CONFIG_FILE="$MODEL_DIR/configuration.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 在 $MODEL_DIR 中未找到 configuration.json"
    exit 1
fi

echo "加载模型配置: $CONFIG_FILE"

# 从 configuration.json 读取参数 (使用 python 简单解析)
read_data_dir() { python -c "import json; print(json.load(open('$CONFIG_FILE'))['data_dir'])"; }
read_data_class() { python -c "import json; print(json.load(open('$CONFIG_FILE'))['data_class'])"; }
read_pattern() { python -c "import json; print(json.load(open('$CONFIG_FILE'))['pattern'])"; }
read_val_pattern() { python -c "import json; print(json.load(open('$CONFIG_FILE'))['val_pattern'])"; }
read_patch_size() { python -c "import json; print(json.load(open('$CONFIG_FILE'))['patch_size'])"; }
read_stride() { python -c "import json; print(json.load(open('$CONFIG_FILE'))['stride'])"; }
read_lr() { python -c "import json; print(json.load(open('$CONFIG_FILE'))['lr'])"; }
read_d_model() { python -c "import json; print(json.load(open('$CONFIG_FILE'))['d_model'])"; }
read_gpt_layers() { python -c "import json; print(json.load(open('$CONFIG_FILE'))['gpt_layers'])"; }
read_num_mamba_layers() { python -c "import json; print(json.load(open('$CONFIG_FILE'))['num_mamba_layers'])"; }
read_no_stat_prompt() { python -c "import json; print(json.load(open('$CONFIG_FILE')).get('no_stat_prompt', False))"; }
read_no_attn_pooling() { python -c "import json; print(json.load(open('$CONFIG_FILE')).get('no_attn_pooling', False))"; }
read_task() { python -c "import json; print(json.load(open('$CONFIG_FILE'))['task'])"; }
read_seed() { python -c "import json; print(json.load(open('$CONFIG_FILE'))['seed'])"; }
read_gpu() { python -c "import json; print(json.load(open('$CONFIG_FILE'))['gpu'])"; }

DATA_DIR=$(read_data_dir)
DATA_CLASS=$(read_data_class)
PATTERN=$(read_pattern)
VAL_PATTERN=$(read_val_pattern)
PATCH_SIZE=$(read_patch_size)
STRIDE=$(read_stride)
LR=$(read_lr)
D_MODEL=$(read_d_model)
GPT_LAYERS=$(read_gpt_layers)
NUM_MAMBA=$(read_num_mamba_layers)
NO_STAT=$(read_no_stat_prompt)
NO_ATTN=$(read_no_attn_pooling)
TASK=$(read_task)
SEED=$(read_seed)
GPU=$(read_gpu)

# 获取最佳模型路径
BEST_MODEL=$(ls -t $MODEL_DIR/best_val_*.pth 2>/dev/null | head -n 1)
if [ -z "$BEST_MODEL" ]; then
    BEST_MODEL=$(ls -t $MODEL_DIR/*.pth 2>/dev/null | head -n 1)
fi

# ✅ 修复：如果根目录没找到，尝试在 checkpoints/ 子目录中寻找
if [ -z "$BEST_MODEL" ]; then
    BEST_MODEL=$(ls -t $MODEL_DIR/checkpoints/*.pth 2>/dev/null | head -n 1)
fi

if [ -z "$BEST_MODEL" ]; then
    echo "错误: 在 $MODEL_DIR 中未找到模型权重文件 (.pth)"
    exit 1
fi

echo "使用模型: $BEST_MODEL"
echo "数据目录: $DATA_DIR"
echo "=========================================="

# 创建输出目录
mkdir -p data_attn
mkdir -p data_flat
mkdir -p plots

# 1. 提取 Attention Pooling 模式的数据
echo ""
echo ">>> 提取 Attention Pooling 特征与权重..."
python extract_data.py \
    --output_dir data_attn \
    --load_model "$BEST_MODEL" \
    --data_dir "$DATA_DIR" \
    --data_class "$DATA_CLASS" \
    --pattern "$PATTERN" \
    --val_pattern "$VAL_PATTERN" \
    --patch_size "$PATCH_SIZE" \
    --stride "$STRIDE" \
    --lr "$LR" \
    --d_model "$D_MODEL" \
    --gpt_layers "$GPT_LAYERS" \
    --num_mamba_layers "$NUM_MAMBA" \
    --task "$TASK" \
    --seed "$SEED" \
    --gpu "$GPU" \
    $([ "$NO_STAT" == "True" ] && echo "--no_stat_prompt")
    # 注意：这里不加 --no_attn_pooling，保持默认开启 Attention

# 2. 提取 Flatten (Adaptive Pooling) 模式的数据
echo ""
echo ">>> 提取 Flatten (Adaptive Pooling) 特征..."
python extract_data.py \
    --output_dir data_flat \
    --load_model "$BEST_MODEL" \
    --data_dir "$DATA_DIR" \
    --data_class "$DATA_CLASS" \
    --pattern "$PATTERN" \
    --val_pattern "$VAL_PATTERN" \
    --patch_size "$PATCH_SIZE" \
    --stride "$STRIDE" \
    --lr "$LR" \
    --d_model "$D_MODEL" \
    --gpt_layers "$GPT_LAYERS" \
    --num_mamba_layers "$NUM_MAMBA" \
    --task "$TASK" \
    --seed "$SEED" \
    --gpu "$GPU" \
    --no_attn_pooling \
    $([ "$NO_STAT" == "True" ] && echo "--no_stat_prompt")

# 3. 可视化
echo ""
echo ">>> 生成可视化图表..."
python visualize_ex4.py \
    --data_attn_dir data_attn \
    --data_flat_dir data_flat \
    --model_dir "$MODEL_DIR" \
    --output_dir plots

echo ""
echo "=========================================="
echo "实验4完成！图表保存在 plots/ 目录下"
echo "=========================================="
