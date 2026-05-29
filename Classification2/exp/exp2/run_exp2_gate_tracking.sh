#!/bin/bash
# 实验二：Gate 参数追踪实验脚本
# 用于测试不同长度序列数据集上 gate 参数的变化

set -e

# 切换到脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 创建 results 目录（如果不存在）
mkdir -p results

# ==========================================
# 配置参数
# ==========================================
GPU=${1:-0}              # GPU 卡号，默认 0
EPOCHS=${2:-100}          # 训练轮数，默认 100
DATA_DIR=${3:-"../../datasets"}  # 数据集路径

EXP_DIR=$(pwd)

# 创建输出目录（setup() 函数要求 output_dir 必须存在）
mkdir -p "${EXP_DIR}/results/heartbeat"
mkdir -p "${EXP_DIR}/results/ethanol"
mkdir -p "${EXP_DIR}/results/japanese"
mkdir -p "${EXP_DIR}/results/pems"

echo "=========================================="
echo "实验二：Gate 参数训练轨迹追踪"
echo "=========================================="
echo "工作目录: $(pwd)"
echo "GPU: ${GPU}"
echo "Epochs: ${EPOCHS}"
echo "Data Dir: ${DATA_DIR}"
echo "=========================================="

# ==========================================
# 短序列数据集（预期：gate 值变化较小）
# ==========================================
echo ""
echo "=========================================="
echo "[1/4] 运行短序列数据集: Heartbeat"
echo "=========================================="
python main_exp2.py \
    --data_dir "${DATA_DIR}/Heartbeat" \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --task classification \
    --gpu ${GPU} \
    --epochs ${EPOCHS} \
    --batch_size 64 \
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
    --name "exp2_heartbeat_short_seq" \
    --no_timestamp \
    --output_dir "${EXP_DIR}/results/heartbeat" \
    --seed 42

# 可视化结果
echo "生成 Heartbeat 可视化..."
python visualize_gates.py --data_dir "${EXP_DIR}/results/heartbeat/exp2_heartbeat_short_seq"

# ==========================================
# 长序列数据集（预期：gate 值显著增大）
# ==========================================
echo ""
echo "=========================================="
echo "[2/4] 运行长序列数据集: EthanolConcentration"
echo "=========================================="
python main_exp2.py \
    --data_dir "${DATA_DIR}/EthanolConcentration" \
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
    --name "exp2_ethanol_long_seq" \
    --no_timestamp \
    --output_dir "${EXP_DIR}/results/ethanol" \
    --seed 42

# 可视化结果
echo "生成 EthanolConcentration 可视化..."
python visualize_gates.py --data_dir "${EXP_DIR}/results/ethanol/exp2_ethanol_long_seq"

# ==========================================
# 中等长度序列数据集
# ==========================================
echo ""
echo "=========================================="
echo "[3/4] 运行中等序列数据集: JapaneseVowels"
echo "=========================================="
python main_exp2.py \
    --data_dir "${DATA_DIR}/JapaneseVowels" \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --task classification \
    --gpu ${GPU} \
    --epochs ${EPOCHS} \
    --batch_size 64 \
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
    --name "exp2_japanese_medium_seq" \
    --no_timestamp \
    --output_dir "${EXP_DIR}/results/japanese" \
    --seed 42

# 可视化结果
echo "生成 JapaneseVowels 可视化..."
python visualize_gates.py --data_dir "${EXP_DIR}/results/japanese/exp2_japanese_medium_seq"

# ==========================================
# 另一个长序列数据集
# ==========================================
echo ""
echo "=========================================="
echo "[4/4] 运行长序列数据集: PEMS-SF"
echo "=========================================="
python main_exp2.py \
    --data_dir "${DATA_DIR}/PEMS-SF" \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --task classification \
    --gpu ${GPU} \
    --epochs ${EPOCHS} \
    --batch_size 16 \
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
    --name "exp2_pems_long_seq" \
    --no_timestamp \
    --output_dir "${EXP_DIR}/results/pems" \
    --seed 42

# 可视化结果
echo "生成 PEMS-SF 可视化..."
python visualize_gates.py --data_dir "${EXP_DIR}/results/pems/exp2_pems_long_seq"

# ==========================================
# 总结
# ==========================================
echo ""
echo "=========================================="
echo "实验二完成！"
echo "=========================================="
echo "结果目录："
echo "  - Heartbeat (短序列): ${EXP_DIR}/results/heartbeat/exp2_heartbeat_short_seq"
echo "  - EthanolConcentration (长序列): ${EXP_DIR}/results/ethanol/exp2_ethanol_long_seq"
echo "  - JapaneseVowels (中等序列): ${EXP_DIR}/results/japanese/exp2_japanese_medium_seq"
echo "  - PEMS-SF (长序列): ${EXP_DIR}/results/pems/exp2_pems_long_seq"
echo ""
echo "每个实验目录下的 gate_tracking/ 文件夹包含："
echo "  - gate_history.json: 完整追踪数据"
echo "  - gate_history.csv: CSV 格式（可导入 Excel）"
echo "  - gate_tracking_final.json: 包含统计信息"
echo "  - plots/: 可视化图表"
echo "=========================================="
echo ""
echo "分析建议："
echo "1. 对比短序列和长序列数据集的 gate 值变化"
echo "2. 长序列数据集的 gate 值应该显著增大（模型学会依赖 Mamba）"
echo "3. 短序列数据集的 gate 值可能较小（GPT attention 已足够）"
echo "=========================================="
