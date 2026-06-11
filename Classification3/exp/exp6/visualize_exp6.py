"""
实验六：GPT-2冻结策略实验可视化工具
生成不同冻结策略的性能对比图表
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse

# 设置中文字体（Linux 服务器兼容）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果 SimHei 不可用，使用默认字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False


class Exp6Visualizer:
    """实验六可视化工具"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.strategies = ['all_frozen', 'partial_frozen', 'all_finetune']
        self.strategy_labels = {
            'all_frozen': 'All Frozen',
            'partial_frozen': 'Partial Frozen',
            'all_finetune': 'All Fine-tuned'
        }
    
    def collect_results(self):
        """收集所有冻结策略的结果"""
        results = {}
        
        print(f"\nSearching for experiment results in: {self.results_dir}")
        print(f"Looking for strategies: {self.strategies}\n")
        
        # 列出所有子目录
        all_subdirs = [d for d in os.listdir(self.results_dir) 
                      if os.path.isdir(os.path.join(self.results_dir, d))]
        
        if not all_subdirs:
            print(f"⚠ No subdirectories found in {self.results_dir}")
            return results
        
        print(f"Found {len(all_subdirs)} directories:")
        for d in all_subdirs:
            print(f"  - {d}")
        print()
        
        # 为每种策略查找对应的目录
        for strategy in self.strategies:
            found = False
            for subdir in all_subdirs:
                # 检查目录名是否包含策略名
                if strategy in subdir:
                    subdir_path = os.path.join(self.results_dir, subdir)
                    summary_file = 'experiment6_summary.json'
                    
                    # 检查 summary 文件是否存在
                    if summary_file in os.listdir(subdir_path):
                        summary_path = os.path.join(subdir_path, summary_file)
                        try:
                            with open(summary_path, 'r') as f:
                                data = json.load(f)
                            results[strategy] = data
                            print(f"✓ Found results for {strategy}: {subdir}")
                            found = True
                            break
                        except Exception as e:
                            print(f"⚠ Error reading {summary_path}: {e}")
                    else:
                        print(f"⚠ Directory found but no summary file: {subdir}")
            
            if not found:
                print(f"✗ No results found for strategy: {strategy}")
        
        print(f"\nSuccessfully loaded {len(results)}/{len(self.strategies)} strategies\n")
        return results
    
    def plot_accuracy_comparison(self, results, save_path):
        """绘制准确率对比柱状图"""
        if not results:
            print("No results found!")
            return
        
        # 提取最佳准确率
        accuracies = []
        labels = []
        
        for strategy in self.strategies:
            if strategy in results:
                metrics = results[strategy].get('best_validation_metrics', {})
                acc = metrics.get('accuracy', 0.0)
                accuracies.append(acc)
                labels.append(self.strategy_labels[strategy])
        
        # 绘制柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        
        bars = ax.bar(range(len(labels)), accuracies, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
        ax.set_ylabel('Validation Accuracy', fontsize=12)
        ax.set_title('GPT-2 Freezing Strategy Comparison - Accuracy', fontsize=14, fontweight='bold')
        
        # 在柱子上标注数值
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{acc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 添加网格线
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Accuracy comparison plot saved to {save_path}")
        plt.close()
    
    def plot_trainable_params_comparison(self, results, save_path):
        """绘制可训练参数比例对比"""
        if not results:
            print("No results found!")
            return
        
        # 提取可训练参数比例
        trainable_ratios = []
        labels = []
        
        for strategy in self.strategies:
            if strategy in results:
                ratio = results[strategy].get('trainable_ratio_percent', 0.0)
                trainable_ratios.append(ratio)
                labels.append(self.strategy_labels[strategy])
        
        # 绘制柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#FFCC99', '#CC99FF', '#99FFFF']
        
        bars = ax.bar(range(len(labels)), trainable_ratios, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
        ax.set_ylabel('Trainable Parameters (%)', fontsize=12)
        ax.set_title('GPT-2 Freezing Strategy - Trainable Parameters Ratio', fontsize=14, fontweight='bold')
        
        # 在柱子上标注数值
        for bar, ratio in zip(bars, trainable_ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{ratio:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 添加网格线
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Trainable parameters comparison plot saved to {save_path}")
        plt.close()
    
    def plot_training_time_comparison(self, results, save_path):
        """绘制训练时间对比柱状图"""
        if not results:
            print("No results found!")
            return
        
        # 提取训练时间（秒）
        training_times = []
        labels = []
        
        for strategy in self.strategies:
            if strategy in results:
                # 直接使用秒为单位
                time_seconds = results[strategy].get('total_training_time_seconds', 0.0)
                training_times.append(time_seconds)
                labels.append(self.strategy_labels[strategy])
        
        # 绘制柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#FFB366', '#66CCFF', '#66FF99']
        
        bars = ax.bar(range(len(labels)), training_times, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
        ax.set_ylabel('Total Training Time (seconds)', fontsize=12)
        ax.set_title('GPT-2 Freezing Strategy - Training Time Comparison', fontsize=14, fontweight='bold')
        
        # 在柱子上标注数值
        for bar, time_val in zip(bars, training_times):
            height = bar.get_height()
            # 直接显示秒数
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{time_val:.2f} s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 添加网格线
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training time comparison plot saved to {save_path}")
        plt.close()
    
    def plot_efficiency_scatter(self, results, save_path):
        """绘制效率散点图：准确率 vs 训练时间"""
        if not results:
            print("No results found!")
            return
        
        # 使用字典存储每个策略的数据，避免索引错位
        strategy_data = {}
        
        for strategy in self.strategies:
            if strategy in results:
                metrics = results[strategy].get('best_validation_metrics', {})
                acc = metrics.get('accuracy', 0.0)
                time_seconds = results[strategy].get('total_training_time_seconds', 0.0)
                
                strategy_data[strategy] = {
                    'accuracy': acc,
                    'time': time_seconds,
                    'label': self.strategy_labels[strategy]
                }
        
        # 绘制散点图
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = {'all_frozen': '#FF6666', 'partial_frozen': '#6699FF', 'all_finetune': '#66CC66'}
        
        # 按策略顺序绘制，确保图例顺序正确
        for strategy in self.strategies:
            if strategy in strategy_data:
                data = strategy_data[strategy]
                color = colors.get(strategy, 'gray')
                ax.scatter(data['time'], data['accuracy'], s=200, c=color, 
                          edgecolors='black', linewidth=2, alpha=0.8, zorder=5, 
                          label=data['label'])
        
        ax.set_xlabel('Total Training Time (seconds)', fontsize=12)
        ax.set_ylabel('Validation Accuracy', fontsize=12)
        ax.set_title('Efficiency Analysis: Accuracy vs Training Time', fontsize=14, fontweight='bold')
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 添加图例到图外右上角
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Efficiency scatter plot saved to {save_path}")
        plt.close()
    
    def plot_training_curves(self, results, save_path):
        """绘制训练曲线对比（如果有TensorBoard数据）"""
        # 尝试从metrics文件中读取训练曲线
        metrics_files = []
        
        for strategy in self.strategies:
            for subdir in os.listdir(self.results_dir):
                if strategy in subdir:
                    for file in os.listdir(os.path.join(self.results_dir, subdir)):
                        if file.startswith('metrics_') and file.endswith('.xls'):
                            metrics_files.append((strategy, os.path.join(self.results_dir, subdir, file)))
        
        if not metrics_files:
            print("No training curve data found (metrics files)")
            return
        
        # 绘制曲线
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for strategy, metrics_file in metrics_files:
            try:
                df = pd.read_excel(metrics_file)
                epochs = range(len(df))
                
                # 查找loss和accuracy列
                loss_col = [c for c in df.columns if 'loss' in c.lower()][0]
                acc_col = [c for c in df.columns if 'accuracy' in c.lower() or 'acc' in c.lower()][0]
                
                color = {'all_frozen': 'red', 'partial_frozen': 'blue', 
                        'all_finetune': 'green'}.get(strategy, 'gray')
                label = self.strategy_labels[strategy]
                
                axes[0].plot(epochs, df[loss_col], label=label, color=color, linewidth=2)
                axes[1].plot(epochs, df[acc_col], label=label, color=color, linewidth=2)
                
            except Exception as e:
                print(f"Warning: Could not load metrics for {strategy}: {e}")
        
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Validation Loss', fontsize=11)
        axes[0].set_title('Validation Loss Comparison', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Validation Accuracy', fontsize=11)
        axes[1].set_title('Validation Accuracy Comparison', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves plot saved to {save_path}")
        plt.close()
    
    def generate_summary_table(self, results, save_path):
        """生成实验结果汇总表"""
        if not results:
            return
        
        data = []
        for strategy in self.strategies:
            if strategy in results:
                metrics = results[strategy].get('best_validation_metrics', {})
                # 直接使用秒为单位
                time_seconds = results[strategy].get('total_training_time_seconds', 0.0)
                avg_epoch_time = results[strategy].get('avg_epoch_time_seconds', 0.0)
                
                data.append({
                    'Freeze Strategy': self.strategy_labels[strategy],
                    'Accuracy': metrics.get('accuracy', 0.0),
                    'Loss': metrics.get('loss', 0.0),
                    'Total Params': results[strategy].get('total_parameters', 0),
                    'Trainable Params': results[strategy].get('trainable_parameters', 0),
                    'Trainable Ratio (%)': results[strategy].get('trainable_ratio_percent', 0.0),
                    'Total Time (seconds)': round(time_seconds, 2),
                    'Avg Time/Epoch (s)': round(avg_epoch_time, 2)
                })
        
        df = pd.DataFrame(data)
        df.to_excel(save_path, index=False)
        print(f"✓ Summary table saved to {save_path}")
        
        # 打印表格
        print("\n" + "="*120)
        print("EXPERIMENT 6 RESULTS SUMMARY")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120 + "\n")
    
    def visualize_all(self):
        """生成所有可视化图表"""
        results = self.collect_results()
        
        if not results:
            print("No experiment results found! Please run the experiments first.")
            return
        
        # 创建输出目录
        plots_dir = os.path.join(self.results_dir, 'exp6_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 生成图表
        self.plot_accuracy_comparison(results, os.path.join(plots_dir, 'accuracy_comparison.png'))
        self.plot_trainable_params_comparison(results, os.path.join(plots_dir, 'trainable_params_comparison.png'))
        self.plot_training_time_comparison(results, os.path.join(plots_dir, 'training_time_comparison.png'))
        self.plot_efficiency_scatter(results, os.path.join(plots_dir, 'efficiency_scatter.png'))
        self.plot_training_curves(results, os.path.join(plots_dir, 'training_curves.png'))
        self.generate_summary_table(results, os.path.join(plots_dir, 'summary_table.xlsx'))
        
        print(f"\n✓ All visualizations saved to: {plots_dir}")
        print(f"\nGenerated plots:")
        print(f"  1. accuracy_comparison.png         - 准确率对比")
        print(f"  2. trainable_params_comparison.png - 可训练参数比例")
        print(f"  3. training_time_comparison.png    - 训练时间对比")
        print(f"  4. efficiency_scatter.png          - 效率散点图 (准确率 vs 时间)")
        print(f"  5. training_curves.png             - 训练曲线对比")
        print(f"  6. summary_table.xlsx              - 汇总表格")


def main():
    parser = argparse.ArgumentParser(description='Experiment 6 Visualization')
    parser.add_argument('--data_dir', type=str, default='results',
                       help='Directory containing experiment results')
    args = parser.parse_args()
    
    visualizer = Exp6Visualizer(args.data_dir)
    visualizer.visualize_all()


if __name__ == '__main__':
    main()
