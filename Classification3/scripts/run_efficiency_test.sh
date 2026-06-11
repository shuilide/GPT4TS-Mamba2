#!/bin/bash

# 效率与复杂度分析实验脚本
# 用于测试长序列数据集（EigenWorms, MotorImagery, StandWalkJump）

echo "=============================================="
echo "效率与复杂度分析实验"
echo "=============================================="

# 设置GPU
GPU_ID=${1:-0}

# 选择数据集
DATASET=${2:-EigenWorms}

echo "GPU: ${GPU_ID}"
echo "数据集: ${DATASET}"
echo "=============================================="

# 运行效率分析实验
python run_efficiency_analysis.py \
    --dataset ${DATASET} \
    --gpu ${GPU_ID} \
    --batch_size 1 \
    --num_runs 3 \
    --patch_size 8 \
    --stride 8 \
    --gpt_layers 6 \
    --num_mamba_layers 2 \
    --output_dir experiments/efficiency_analysis \
    --test_variants

# 生成可视化
echo ""
echo "开始生成可视化图表..."
python visualize_efficiency.py \
    --input_dir experiments/efficiency_analysis/${DATASET} \
    --output_dir experiments/efficiency_analysis/${DATASET}/figures \
    --dataset ${DATASET}

echo ""
echo "=============================================="
echo "实验完成！"
echo "=============================================="
echo "结果位置: experiments/efficiency_analysis/${DATASET}/"
echo "图表位置: experiments/efficiency_analysis/${DATASET}/figures/"
echo ""
echo "CSV文件:"
echo "  - mamba_gpt_results.csv (MambaGPT)"
echo "  - pure_gpt2_results.csv (Pure GPT-2)"
echo "  - comparison_results.csv (对比结果)"
echo ""
echo "图片文件:"
echo "  - ${DATASET}_memory_comparison.png (显存对比)"
echo "  - ${DATASET}_time_comparison.png (推理时间对比)"
echo "  - ${DATASET}_throughput_comparison.png (吞吐量对比)"
echo "  - ${DATASET}_flops_params_table.png (FLOPs/参数量对比)"
echo "  - ${DATASET}_combined_comparison.png (组合对比)"
echo "  - ${DATASET}_scaling_analysis.png (复杂度缩放分析)"
echo "=============================================="
