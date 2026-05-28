"""
可视化脚本：绘制效率与复杂度分析结果
生成显存消耗和推理时间的对比曲线图
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_csv(filepath):
    """加载CSV文件"""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def plot_memory_comparison(results_dict, output_path, dataset_name):
    """
    绘制显存对比图
    Args:
        results_dict: {模型名称: 数据列表}
        output_path: 输出路径
        dataset_name: 数据集名称
    """
    plt.figure(figsize=(12, 8))
    
    colors = {'MambaGPT': 'blue', 'PureGPT2': 'red', 'MambaGPT_Variant': 'green'}
    markers = {'MambaGPT': 'o', 'PureGPT2': 's', 'MambaGPT_Variant': '^'}
    
    for model_name, data in results_dict.items():
        seq_lengths = [int(d['seq_length']) for d in data]
        memories = [float(d['peak_memory_mb']) for d in data]
        
        # 过滤掉失败的数据
        valid_data = [(s, m) for s, m in zip(seq_lengths, memories) if m > 0]
        if not valid_data:
            continue
        
        valid_seq, valid_mem = zip(*valid_data)
        
        color = colors.get(model_name, 'gray')
        marker = markers.get(model_name, 'o')
        
        plt.plot(valid_seq, valid_mem, 
                marker=marker, linewidth=2, markersize=8,
                label=model_name, color=color)
    
    plt.xlabel('Sequence Length', fontsize=14, fontweight='bold')
    plt.ylabel('Peak GPU Memory (MB)', fontsize=14, fontweight='bold')
    plt.title(f'Memory Complexity Comparison - {dataset_name}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 显存对比图已保存: {output_path}")
    plt.close()


def plot_time_comparison(results_dict, output_path, dataset_name):
    """
    绘制推理时间对比图
    """
    plt.figure(figsize=(12, 8))
    
    colors = {'MambaGPT': 'blue', 'PureGPT2': 'red', 'MambaGPT_Variant': 'green'}
    markers = {'MambaGPT': 'o', 'PureGPT2': 's', 'MambaGPT_Variant': '^'}
    
    for model_name, data in results_dict.items():
        seq_lengths = [int(d['seq_length']) for d in data]
        times = [float(d['inference_time_s']) * 1000 for d in data]  # 转换为ms
        
        # 过滤掉失败的数据
        valid_data = [(s, t) for s, t in zip(seq_lengths, times) if t > 0]
        if not valid_data:
            continue
        
        valid_seq, valid_time = zip(*valid_data)
        
        color = colors.get(model_name, 'gray')
        marker = markers.get(model_name, 'o')
        
        plt.plot(valid_seq, valid_time,
                marker=marker, linewidth=2, markersize=8,
                label=model_name, color=color)
    
    plt.xlabel('Sequence Length', fontsize=14, fontweight='bold')
    plt.ylabel('Inference Time (ms)', fontsize=14, fontweight='bold')
    plt.title(f'Inference Time Comparison - {dataset_name}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 推理时间对比图已保存: {output_path}")
    plt.close()


def plot_combined_comparison(results_dict, output_path, dataset_name):
    """
    绘制组合对比图（双Y轴）
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    colors = {'MambaGPT': 'blue', 'PureGPT2': 'red', 'MambaGPT_Variant': 'green'}
    markers = {'MambaGPT': 'o', 'PureGPT2': 's', 'MambaGPT_Variant': '^'}
    
    lines1 = []
    lines2 = []
    labels1 = []
    labels2 = []
    
    for model_name, data in results_dict.items():
        seq_lengths = [int(d['seq_length']) for d in data]
        memories = [float(d['peak_memory_mb']) for d in data]
        times = [float(d['inference_time_s']) * 1000 for d in data]
        
        valid_data = [(s, m, t) for s, m, t in zip(seq_lengths, memories, times) if m > 0 and t > 0]
        if not valid_data:
            continue
        
        valid_seq, valid_mem, valid_time = zip(*valid_data)
        
        color = colors.get(model_name, 'gray')
        marker = markers.get(model_name, 'o')
        
        # 显存（左Y轴）
        line1, = ax1.plot(valid_seq, valid_mem,
                         marker=marker, linewidth=2, markersize=8,
                         label=f'{model_name} (Memory)', color=color)
        lines1.append(line1)
        labels1.append(f'{model_name} (Memory)')
    
    ax1.set_xlabel('Sequence Length', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Peak GPU Memory (MB)', fontsize=14, fontweight='bold', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 推理时间（右Y轴）
    ax2 = ax1.twinx()
    
    for model_name, data in results_dict.items():
        seq_lengths = [int(d['seq_length']) for d in data]
        times = [float(d['inference_time_s']) * 1000 for d in data]
        
        valid_data = [(s, t) for s, t in zip(seq_lengths, times) if t > 0]
        if not valid_data:
            continue
        
        valid_seq, valid_time = zip(*valid_data)
        
        color = colors.get(model_name, 'gray')
        marker = markers.get(model_name, 'o')
        
        line2, = ax2.plot(valid_seq, valid_time,
                         marker='x', linewidth=2, markersize=8,
                         linestyle='--', label=f'{model_name} (Time)', color=color)
        lines2.append(line2)
        labels2.append(f'{model_name} (Time)')
    
    ax2.set_ylabel('Inference Time (ms)', fontsize=14, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 合并图例
    lines = lines1 + lines2
    labels = labels1 + labels2
    ax1.legend(lines, labels, fontsize=11, loc='upper left')
    
    plt.title(f'Memory & Time Complexity - {dataset_name}', fontsize=16, fontweight='bold')
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 组合对比图已保存: {output_path}")
    plt.close()


def plot_scaling_analysis(results_dict, output_path, dataset_name):
    """
    绘制复杂度缩放分析（对数坐标）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'MambaGPT': 'blue', 'PureGPT2': 'red', 'MambaGPT_Variant': 'green'}
    
    for model_name, data in results_dict.items():
        seq_lengths = np.array([int(d['seq_length']) for d in data])
        memories = np.array([float(d['peak_memory_mb']) for d in data])
        times = np.array([float(d['inference_time_s']) * 1000 for d in data])
        
        valid_mask = (memories > 0) & (times > 0)
        if not np.any(valid_mask):
            continue
        
        valid_seq = seq_lengths[valid_mask]
        valid_mem = memories[valid_mask]
        valid_time = times[valid_mask]
        
        color = colors.get(model_name, 'gray')
        
        # 显存缩放（对数坐标）
        ax1.loglog(valid_seq, valid_mem, 'o-', linewidth=2, markersize=8,
                   label=model_name, color=color)
        
        # 时间缩放（对数坐标）
        ax2.loglog(valid_seq, valid_time, 's-', linewidth=2, markersize=8,
                   label=model_name, color=color)
    
    # 添加参考线（线性 vs 二次）
    if len(results_dict) > 0:
        ref_seq = np.array([500, 10000])
        ax1.loglog(ref_seq, ref_seq / 500 * 100, 'k--', alpha=0.5, label='O(n) Reference')
        ax1.loglog(ref_seq, (ref_seq / 500) ** 2 * 100, 'k:', alpha=0.5, label='O(n²) Reference')
        ax2.loglog(ref_seq, ref_seq / 500 * 10, 'k--', alpha=0.5)
        ax2.loglog(ref_seq, (ref_seq / 500) ** 2 * 10, 'k:', alpha=0.5)
    
    ax1.set_xlabel('Sequence Length (log scale)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Peak GPU Memory (MB) (log scale)', fontsize=13, fontweight='bold')
    ax1.set_title('Memory Scaling', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    
    ax2.set_xlabel('Sequence Length (log scale)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Inference Time (ms) (log scale)', fontsize=13, fontweight='bold')
    ax2.set_title('Time Scaling', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.suptitle(f'Complexity Scaling Analysis - {dataset_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 缩放分析图已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='效率分析结果可视化')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='包含CSV结果的目录')
    parser.add_argument('--output_dir', type=str, default='experiments/efficiency_analysis/figures',
                       help='输出图片目录')
    parser.add_argument('--dataset', type=str, default='EigenWorms',
                       help='数据集名称')
    
    args = parser.parse_args()
    
    # 加载数据
    results_dict = {}
    
    # 加载MambaGPT结果
    mamba_path = os.path.join(args.input_dir, 'mamba_gpt_results.csv')
    if os.path.exists(mamba_path):
        results_dict['MambaGPT'] = load_csv(mamba_path)
        print(f"✓ 加载 MambaGPT 结果: {len(results_dict['MambaGPT'])} 条记录")
    
    # 加载PureGPT2结果
    gpt_path = os.path.join(args.input_dir, 'pure_gpt2_results.csv')
    if os.path.exists(gpt_path):
        results_dict['PureGPT2'] = load_csv(gpt_path)
        print(f"✓ 加载 PureGPT2 结果: {len(results_dict['PureGPT2'])} 条记录")
    
    # 加载变体结果
    variant_path = os.path.join(args.input_dir, 'mamba_gpt_variant_results.csv')
    if os.path.exists(variant_path):
        results_dict['MambaGPT_Variant'] = load_csv(variant_path)
        print(f"✓ 加载 MambaGPT_Variant 结果: {len(results_dict['MambaGPT_Variant'])} 条记录")
    
    if not results_dict:
        print("❌ 未找到任何结果文件！")
        return
    
    # 生成图表
    print("\n开始生成可视化图表...")
    
    # 1. 显存对比
    plot_memory_comparison(
        results_dict,
        os.path.join(args.output_dir, f'{args.dataset}_memory_comparison.png'),
        args.dataset
    )
    
    # 2. 推理时间对比
    plot_time_comparison(
        results_dict,
        os.path.join(args.output_dir, f'{args.dataset}_time_comparison.png'),
        args.dataset
    )
    
    # 3. 组合对比
    plot_combined_comparison(
        results_dict,
        os.path.join(args.output_dir, f'{args.dataset}_combined_comparison.png'),
        args.dataset
    )
    
    # 4. 缩放分析
    plot_scaling_analysis(
        results_dict,
        os.path.join(args.output_dir, f'{args.dataset}_scaling_analysis.png'),
        args.dataset
    )
    
    print(f"\n✓ 所有图表已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
