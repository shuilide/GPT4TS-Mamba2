"""
效率与复杂度分析实验脚本
用于对比 MambaGPT、纯 GPT-2 和纯 Transformer 在不同序列长度下的性能
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path

# 添加src目录到Python路径，使models模块可被正确导入
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.gpt4ts import gpt4ts
from efficiency_profiler import EfficiencyProfiler

def create_model(config_dict, data_mock):
    """创建模型（Mock数据类）"""
    class MockData:
        def __init__(self, feat_dim, seq_len, num_classes):
            self.feature_df = type('obj', (object,), {'shape': [None, feat_dim]})()
            self.max_seq_len = seq_len
            self.class_names = list(range(num_classes))
    
    config = type('obj', (object,), config_dict)()
    return gpt4ts(config_dict, data_mock)


def run_efficiency_experiment(args):
    """运行效率分析实验"""
    
    print("\n" + "="*80)
    print("效率与复杂度分析实验")
    print("="*80 + "\n")
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != '-1' else 'cpu')
    print(f"使用设备: {device}")
    
    # 定义测试的序列长度（针对长序列数据集）
    if args.dataset == 'EigenWorms':
        seq_lengths = [500, 1000, 2000, 3000, 4000, 5000,]
        feat_dim = 6
    elif args.dataset == 'MotorImagery':
        seq_lengths = [500, 1000, 1500, 2000, 2500, 3000]
        feat_dim = 1124
    elif args.dataset == 'StandWalkJump':
        seq_lengths = [500, 1000, 1500, 2000, 2500]
        feat_dim = 6
    else:  # 自定义
        seq_lengths = [int(x) for x in args.seq_lengths.split(',')]
        feat_dim = args.feat_dim
    
    print(f"测试数据集: {args.dataset}")
    print(f"特征维度: {feat_dim}")
    print(f"序列长度列表: {seq_lengths}")
    print(f"Patch大小: {args.patch_size}, 步长: {args.stride}")
    print(f"批次大小: {args.batch_size}")
    print(f"测试次数: {args.num_runs}\n")
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # Mock数据
    class MockData:
        def __init__(self):
            self.feature_df = type('obj', (object,), {'shape': [None, feat_dim]})()
            self.max_seq_len = max(seq_lengths)
            self.class_names = list(range(args.num_classes))
    
    data_mock = MockData()
    
    # 配置字典
    base_config = {
        'patch_size': args.patch_size,
        'stride': args.stride,
        'd_model': 768,
        'dropout': 0.1,
        'gpt_layers': args.gpt_layers,
    }
    
    results_dict = {}
    
    # ========================================
    # 测试1: MambaGPT（默认配置）
    # ========================================
    print("\n" + "="*80)
    print("测试1: MambaGPT (with Mamba adapters)")
    print("="*80)
    
    config_mamba = base_config.copy()
    config_mamba['num_mamba_layers'] = args.num_mamba_layers
    config_mamba['no_stat_prompt'] = False
    config_mamba['no_attn_pooling'] = False
    
    model_mamba = gpt4ts(config_mamba, data_mock).to(device)
    profiler_mamba = EfficiencyProfiler(model_mamba, device)
    
    save_path_mamba = os.path.join(output_dir, 'mamba_gpt_results.csv')
    results_mamba = profiler_mamba.profile_sequence_lengths(
        seq_lengths, feat_dim, args.batch_size, 
        args.patch_size, args.stride, save_path_mamba
    )
    results_dict['MambaGPT'] = results_mamba
    
    # 清理显存
    del model_mamba
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========================================
    # 测试2: 纯 GPT-2（无Mamba）
    # ========================================
    print("\n" + "="*80)
    print("测试2: Pure GPT-2 (no Mamba adapters)")
    print("="*80)
    
    config_gpt = base_config.copy()
    config_gpt['num_mamba_layers'] = 0  # 不使用Mamba
    config_gpt['no_stat_prompt'] = False
    config_gpt['no_attn_pooling'] = False
    
    model_gpt = gpt4ts(config_gpt, data_mock).to(device)
    profiler_gpt = EfficiencyProfiler(model_gpt, device)
    
    save_path_gpt = os.path.join(output_dir, 'pure_gpt2_results.csv')
    results_gpt = profiler_gpt.profile_sequence_lengths(
        seq_lengths, feat_dim, args.batch_size,
        args.patch_size, args.stride, save_path_gpt
    )
    results_dict['PureGPT2'] = results_gpt
    
    # 清理显存
    del model_gpt
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========================================
    # 测试3: 消融变体（可选）
    # ========================================
    if args.test_variants:
        print("\n" + "="*80)
        print("测试3: MambaGPT (no stat prompt, no attn pooling)")
        print("="*80)
        
        config_variant = base_config.copy()
        config_variant['num_mamba_layers'] = args.num_mamba_layers
        config_variant['no_stat_prompt'] = True
        config_variant['no_attn_pooling'] = True
        
        model_variant = gpt4ts(config_variant, data_mock).to(device)
        profiler_variant = EfficiencyProfiler(model_variant, device)
        
        save_path_variant = os.path.join(output_dir, 'mamba_gpt_variant_results.csv')
        results_variant = profiler_variant.profile_sequence_lengths(
            seq_lengths, feat_dim, args.batch_size,
            args.patch_size, args.stride, save_path_variant
        )
        results_dict['MambaGPT_Variant'] = results_variant
        
        del model_variant
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========================================
    # 对比分析
    # ========================================
    print("\n" + "="*80)
    print("生成对比报告")
    print("="*80)
    
    compare_path = os.path.join(output_dir, 'comparison_results.csv')
    EfficiencyProfiler.compare_models(results_dict, compare_path)
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    print(f"\n结果保存在: {output_dir}")
    print("  - MambaGPT: mamba_gpt_results.csv")
    print("  - PureGPT2: pure_gpt2_results.csv")
    if args.test_variants:
        print("  - Variant: mamba_gpt_variant_results.csv")
    print("  - 对比结果: comparison_results.csv")
    print("\n使用 visualization/plot_efficiency.py 生成可视化图表")


def main():
    parser = argparse.ArgumentParser(description='效率与复杂度分析实验')
    
    # 数据集配置
    parser.add_argument('--dataset', type=str, default='EigenWorms',
                       choices=['EigenWorms', 'MotorImagery', 'StandWalkJump', 'custom'],
                       help='测试数据集')
    parser.add_argument('--seq_lengths', type=str, default='500,1000,2000,5000,10000',
                       help='自定义序列长度列表（逗号分隔）')
    parser.add_argument('--feat_dim', type=int, default=6,
                       help='特征维度（用于custom数据集）')
    
    # 模型配置
    parser.add_argument('--patch_size', type=int, default=8, help='Patch大小')
    parser.add_argument('--stride', type=int, default=8, help='步长')
    parser.add_argument('--gpt_layers', type=int, default=6, help='GPT层数')
    parser.add_argument('--num_mamba_layers', type=int, default=2, help='Mamba适配器层数')
    parser.add_argument('--num_classes', type=int, default=5, help='类别数量')
    
    # 实验配置
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--num_runs', type=int, default=3, help='每个配置的运行次数')
    parser.add_argument('--gpu', type=str, default='0', help='GPU索引')
    parser.add_argument('--output_dir', type=str, default='experiments/efficiency_analysis',
                       help='输出目录')
    parser.add_argument('--test_variants', action='store_true',
                       help='是否测试消融变体')
    
    args = parser.parse_args()
    run_efficiency_experiment(args)


if __name__ == '__main__':
    main()
