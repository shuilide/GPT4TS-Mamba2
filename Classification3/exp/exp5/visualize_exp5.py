"""
实验五：架构消融实验可视化工具
生成不同架构模式的性能对比图表
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse

# 设置中文字体（兼容 Windows 和 Linux）
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 使用黑体
else:
    # Linux 系统使用 DejaVu Sans
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Exp5Visualizer:
    """实验五可视化工具"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.arch_modes = ['pure_gpt', 'pure_mamba', 'sequential', 'parallel']
        self.arch_labels = {
            'pure_gpt': 'Pure GPT-2',
            'pure_mamba': 'Pure Mamba',
            'sequential': 'Sequential (GPT→Mamba)',
            'parallel': 'Parallel Fusion (Ours)'
        }
    
    def collect_results(self):
        """收集所有架构模式的结果"""
        results = {}
        
        for mode in self.arch_modes:
            # 查找对应的实验目录
            for subdir in os.listdir(self.results_dir):
                if mode in subdir:
                    # 从 metrics xls 文件中读取最后一条记录
                    for file in os.listdir(os.path.join(self.results_dir, subdir)):
                        if file.startswith('metrics_') and file.endswith('.xls'):
                            metrics_path = os.path.join(self.results_dir, subdir, file)
                            try:
                                df = pd.read_excel(metrics_path)
                                if len(df) > 0:
                                    # 获取最后一条记录（最佳结果）
                                    last_row = df.iloc[-1]
                                    # 查找accuracy和loss列
                                    acc_col = [c for c in df.columns if 'accuracy' in c.lower() or 'acc' in c.lower()]
                                    loss_col = [c for c in df.columns if 'loss' in c.lower()]
                                    
                                    if acc_col and loss_col:
                                        results[mode] = {
                                            'best_validation_metrics': {
                                                'accuracy': float(last_row[acc_col[0]]),
                                                'loss': float(last_row[loss_col[0]])
                                            },
                                            'epochs': len(df)
                                        }
                                        print(f"✓ Found results for {mode}: {subdir} (Accuracy: {results[mode]['best_validation_metrics']['accuracy']:.4f})")
                            except Exception as e:
                                print(f"Warning: Could not load metrics for {subdir}: {e}")
                            break
        
        return results
    
    def plot_accuracy_comparison(self, results, save_path):
        """绘制准确率对比柱状图"""
        if not results:
            print("No results found!")
            return
        
        # 提取最佳准确率
        accuracies = []
        labels = []
        
        for mode in self.arch_modes:
            if mode in results:
                metrics = results[mode].get('best_validation_metrics', {})
                acc = metrics.get('accuracy', 0.0)
                accuracies.append(acc)
                labels.append(self.arch_labels[mode])
        
        # 绘制柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        
        bars = ax.bar(range(len(labels)), accuracies, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel('Validation Accuracy', fontsize=12)
        ax.set_title('Architecture Ablation Study - Accuracy Comparison', fontsize=14, fontweight='bold')
        
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
    
    def plot_training_curves(self, results, save_path):
        """绘制训练曲线对比（如果有TensorBoard数据）"""
        # 尝试从metrics文件中读取训练曲线
        metrics_files = []
        
        for mode in self.arch_modes:
            for subdir in os.listdir(self.results_dir):
                if mode in subdir:
                    for file in os.listdir(os.path.join(self.results_dir, subdir)):
                        if file.startswith('metrics_') and file.endswith('.xls'):
                            metrics_files.append((mode, os.path.join(self.results_dir, subdir, file)))
        
        if not metrics_files:
            print("No training curve data found (metrics files)")
            return
        
        # 绘制曲线
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for mode, metrics_file in metrics_files:
            try:
                df = pd.read_excel(metrics_file)
                epochs = range(len(df))
                
                # 查找loss和accuracy列
                loss_col = [c for c in df.columns if 'loss' in c.lower()][0]
                acc_col = [c for c in df.columns if 'accuracy' in c.lower() or 'acc' in c.lower()][0]
                
                color = {'pure_gpt': 'red', 'pure_mamba': 'blue', 
                        'sequential': 'green', 'parallel': 'orange'}.get(mode, 'gray')
                label = self.arch_labels[mode]
                
                axes[0].plot(epochs, df[loss_col], label=label, color=color, linewidth=2)
                axes[1].plot(epochs, df[acc_col], label=label, color=color, linewidth=2)
                
            except Exception as e:
                print(f"Warning: Could not load metrics for {mode}: {e}")
        
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
        for mode in self.arch_modes:
            if mode in results:
                metrics = results[mode].get('best_validation_metrics', {})
                data.append({
                    'Architecture': self.arch_labels[mode],
                    'Accuracy': metrics.get('accuracy', 0.0),
                    'Loss': metrics.get('loss', 0.0)
                })
        
        df = pd.DataFrame(data)
        df.to_excel(save_path, index=False)
        print(f"✓ Summary table saved to {save_path}")
        
        # 打印表格
        print("\n" + "="*80)
        print("EXPERIMENT 5 RESULTS SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
    
    def visualize_all(self):
        """生成所有可视化图表"""
        results = self.collect_results()
        
        if not results:
            print("No experiment results found! Please run the experiments first.")
            return
        
        # 创建输出目录
        plots_dir = os.path.join(self.results_dir, 'exp5_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 生成图表
        self.plot_accuracy_comparison(results, os.path.join(plots_dir, 'accuracy_comparison.png'))
        self.plot_training_curves(results, os.path.join(plots_dir, 'training_curves.png'))
        self.generate_summary_table(results, os.path.join(plots_dir, 'summary_table.xlsx'))
        
        print(f"\n✓ All visualizations saved to: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 5 Visualization')
    parser.add_argument('--data_dir', type=str, default='results',
                       help='Directory containing experiment results')
    args = parser.parse_args()
    
    visualizer = Exp5Visualizer(args.data_dir)
    visualizer.visualize_all()


if __name__ == '__main__':
    main()
