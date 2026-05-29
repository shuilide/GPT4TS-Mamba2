#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验3：预训练知识迁移验证 - 收敛曲线可视化
对比 Pre-trained vs From Scratch 的 Loss 和 Accuracy 收敛曲线
"""

import os
import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class ConvergenceVisualizer:
    """实验3收敛曲线可视化工具"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.datasets = [
            'ArticularyWordRecognition',
            'AtrialFibrillation',
            'BasicMotions',
            'CharacterTrajectories',
        ]
    
    def load_training_history(self, dataset_name, variant):
        """
        加载训练历史记录
        
        Args:
            dataset_name: 数据集名称
            variant: 'pretrained' 或 'scratch'
        
        Returns:
            dict: {'epochs': [], 'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        """
        exp_name = f"{dataset_name}_{variant}"
        exp_dir = os.path.join(self.data_dir, exp_name)
        
        if not os.path.exists(exp_dir):
            logger.warning(f"Experiment directory not found: {exp_dir}")
            return None
        
        history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # 方法1：尝试读取实验目录下的 metrics 文件
        metrics_file = os.path.join(exp_dir, f'metrics_{exp_name}.xls')
        if os.path.exists(metrics_file):
            try:
                df = pd.read_excel(metrics_file)
                if len(df) > 0:
                    history['epochs'] = df['epoch'].tolist()
                    history['train_loss'] = df['loss'].tolist()
                    if 'val_loss' in df.columns:
                        history['val_loss'] = df['val_loss'].tolist()
                    if 'accuracy' in df.columns:
                        history['val_accuracy'] = df['accuracy'].tolist()
                    
                    logger.info(f"Loaded {len(history['epochs'])} epochs from metrics file for {exp_name}")
                    return history
            except Exception as e:
                logger.warning(f"Failed to load metrics file: {e}")
        
        # 方法2：尝试读取总记录文件
        for records_name in ['exp3_records.xls', 'Classification_records.xls']:
            records_file = os.path.join(self.data_dir, records_name)
            if os.path.exists(records_file):
                try:
                    df = pd.read_excel(records_file)
                    # 过滤当前实验的数据
                    mask = df['experiment'].str.contains(exp_name, na=False)
                    exp_data = df[mask]
                    
                    if len(exp_data) > 0:
                        history['epochs'] = exp_data['epoch'].tolist()
                        history['train_loss'] = exp_data['loss'].tolist()
                        if 'val_loss' in exp_data.columns:
                            history['val_loss'] = exp_data['val_loss'].tolist()
                        if 'accuracy' in exp_data.columns:
                            history['val_accuracy'] = exp_data['accuracy'].tolist()
                        
                        logger.info(f"Loaded {len(history['epochs'])} epochs from {records_name} for {exp_name}")
                        return history
                except Exception as e:
                    logger.warning(f"Failed to load from {records_name}: {e}")
        
        # 方法3：尝试读取 JSON 日志
        log_file = os.path.join(exp_dir, 'training_log.json')
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                history = log_data.get('history', history)
                logger.info(f"Loaded training log from JSON for {exp_name}")
                return history
            except Exception as e:
                logger.warning(f"Failed to load JSON log: {e}")
        
        logger.warning(f"No valid data found for {exp_name}")
        return None
    
    def plot_convergence_curves(self, dataset_name, save_dir=None):
        """
        绘制单个数据集的收敛曲线对比
        
        Args:
            dataset_name: 数据集名称
            save_dir: 保存目录
        """
        pretrained_data = self.load_training_history(dataset_name, 'pretrained')
        scratch_data = self.load_training_history(dataset_name, 'scratch')
        
        if pretrained_data is None or scratch_data is None:
            logger.warning(f"Skipping {dataset_name}: missing data")
            return None
        
        # 创建子图：Loss 和 Accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        epochs_pre = pretrained_data['epochs']
        epochs_scr = scratch_data['epochs']
        
        # === 左图：Loss 曲线 ===
        if pretrained_data['train_loss']:
            ax1.plot(epochs_pre, pretrained_data['train_loss'], 
                    'b-o', linewidth=2, markersize=4, label='Pre-trained (Train Loss)')
        if pretrained_data['val_loss']:
            ax1.plot(epochs_pre, pretrained_data['val_loss'], 
                    'b-s', linewidth=2, markersize=4, label='Pre-trained (Val Loss)')
        
        if scratch_data['train_loss']:
            ax1.plot(epochs_scr, scratch_data['train_loss'], 
                    'r-o', linewidth=2, markersize=4, label='From Scratch (Train Loss)')
        if scratch_data['val_loss']:
            ax1.plot(epochs_scr, scratch_data['val_loss'], 
                    'r-s', linewidth=2, markersize=4, label='From Scratch (Val Loss)')
        
        ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
        ax1.set_title(f'{dataset_name} - Loss Convergence', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(labelsize=12)
        
        # === 右图：Accuracy 曲线 ===
        if pretrained_data['val_accuracy']:
            ax2.plot(epochs_pre, pretrained_data['val_accuracy'], 
                    'b-o', linewidth=2, markersize=4, label='Pre-trained')
        if scratch_data['val_accuracy']:
            ax2.plot(epochs_scr, scratch_data['val_accuracy'], 
                    'r-o', linewidth=2, markersize=4, label='From Scratch')
        
        ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_title(f'{dataset_name} - Accuracy Convergence', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f'{dataset_name}_convergence.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved convergence plot: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_all_datasets_comparison(self, save_dir=None):
        """
        绘制所有数据集的最终 Accuracy 对比柱状图
        
        Args:
            save_dir: 保存目录
        """
        pretrained_acc = []
        scratch_acc = []
        datasets_found = []
        
        for dataset in self.datasets:
            pre_data = self.load_training_history(dataset, 'pretrained')
            scr_data = self.load_training_history(dataset, 'scratch')
            
            if pre_data and scr_data and pre_data['val_accuracy'] and scr_data['val_accuracy']:
                pretrained_acc.append(max(pre_data['val_accuracy']))
                scratch_acc.append(max(scr_data['val_accuracy']))
                datasets_found.append(dataset)
        
        if not datasets_found:
            logger.warning("No data available for comparison plot")
            return None
        
        # 创建柱状图
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(datasets_found))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pretrained_acc, width, 
                      label='Pre-trained GPT-2', color='#2196F3', alpha=0.8)
        bars2 = ax.bar(x + width/2, scratch_acc, width, 
                      label='From Scratch', color='#FF5722', alpha=0.8)
        
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Best Validation Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Pre-trained vs From Scratch: Final Accuracy Comparison', 
                    fontsize=16, fontweight='bold', pad=15)
        
        # 简化数据集名称
        short_names = [d[:15] + '..' if len(d) > 15 else d for d in datasets_found]
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=11, rotation=15, ha='right')
        
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=12)
        
        # 在柱子上添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 5),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'all_datasets_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def generate_all_plots(self, output_dir=None):
        """
        生成所有可视化图表
        
        Args:
            output_dir: 输出目录
        """
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, 'exp3_plots')
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Generating convergence visualization plots...")
        
        # 1. 每个数据集的收敛曲线
        for dataset in self.datasets:
            self.plot_convergence_curves(dataset, save_dir=output_dir)
        
        # 2. 所有数据集的最终对比
        self.plot_all_datasets_comparison(save_dir=output_dir)
        
        logger.info(f"All visualizations saved to {output_dir}")
        return output_dir


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='Visualize experiment 3 convergence curves')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset to visualize (default: all)')
    
    args = parser.parse_args()
    
    visualizer = ConvergenceVisualizer(args.data_dir)
    
    if args.dataset:
        # 只绘制指定数据集
        output_dir = args.output_dir or os.path.join(args.data_dir, 'exp3_plots')
        os.makedirs(output_dir, exist_ok=True)
        visualizer.plot_convergence_curves(args.dataset, save_dir=output_dir)
    else:
        # 绘制所有数据集
        output_dir = visualizer.generate_all_plots(args.output_dir)
    
    logger.info(f"Visualization complete! Check: {output_dir}")


if __name__ == '__main__':
    main()
