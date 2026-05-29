#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验3：小样本学习可视化
绘制不同数据比例下预训练模型的性能变化曲线
"""

import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class FewShotVisualizer:
    """小样本学习可视化工具（仅预训练模型）"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dataset = 'RacketSports'
        self.ratios = ['20pct', '40pct', '60pct', '80pct', '100pct']
        self.ratio_labels = ['20%', '40%', '60%', '80%', '100%']
    
    def load_training_history(self, dataset_name, variant='pretrained'):
        """
        加载训练历史记录
        
        Args:
            dataset_name: 数据集名称（含截断比例，如 RacketSports_10pct）
            variant: 固定为 'pretrained'
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
        
        # 尝试读取实验目录下的 metrics 文件
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
        
        logger.warning(f"No valid data found for {exp_name}")
        return None
    
    def plot_fewshot_comparison(self, save_dir=None):
        """
        绘制小样本学习对比柱状图
        X轴：数据比例 (20%, 40%, 60%, 80%, 100%)
        Y轴：最佳验证集 Accuracy（仅预训练模型）
        """
        pretrained_acc = []
        ratios_found = []
        
        for ratio in self.ratios:
            dataset_name = f"{self.dataset}_{ratio}"
            pre_data = self.load_training_history(dataset_name, 'pretrained')
            
            if pre_data and pre_data['val_accuracy']:
                pretrained_acc.append(max(pre_data['val_accuracy']))
                ratios_found.append(ratio)
        
        if not ratios_found:
            logger.warning("No data available for few-shot comparison plot")
            return None
        
        # 创建柱状图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(ratios_found))
        width = 0.6
        
        bars1 = ax.bar(x, pretrained_acc, width, 
                      label='Pre-trained GPT-2', color='#2196F3', alpha=0.8)
        
        ax.set_xlabel('Training Data Ratio', fontsize=14, fontweight='bold')
        ax.set_ylabel('Best Validation Accuracy', fontsize=14, fontweight='bold')
        ax.set_title(f'{self.dataset} - Few-Shot Scaling Law (Pre-trained)', 
                    fontsize=16, fontweight='bold', pad=15)
        
        # 设置 X 轴标签
        ratio_display_labels = [self.ratio_labels[self.ratios.index(r)] for r in ratios_found]
        ax.set_xticks(x)
        ax.set_xticklabels(ratio_display_labels, fontsize=13, fontweight='bold')
        
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=12)
        ax.set_ylim([0, 1.0])
        
        # 在柱子上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f'{self.dataset}_fewshot_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved few-shot comparison plot: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_convergence_curves(self, ratio='10pct', save_dir=None):
        """
        绘制特定数据比例下的收敛曲线（仅预训练模型）
        
        Args:
            ratio: 数据比例 ('20pct', '40pct', '60pct', '80pct', '100pct')
            save_dir: 保存目录
        """
        dataset_name = f"{self.dataset}_{ratio}"
        pretrained_data = self.load_training_history(dataset_name, 'pretrained')
        
        if pretrained_data is None:
            logger.warning(f"Skipping {dataset_name}: missing data")
            return None
        
        # 创建子图：Loss 和 Accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        epochs_pre = pretrained_data['epochs']
        
        # === 左图：Loss 曲线 ===
        if pretrained_data['train_loss']:
            ax1.plot(epochs_pre, pretrained_data['train_loss'], 
                    'b-o', linewidth=2, markersize=4, label='Train Loss')
        if pretrained_data['val_loss']:
            ax1.plot(epochs_pre, pretrained_data['val_loss'], 
                    'b-s', linewidth=2, markersize=4, label='Val Loss')
        
        ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
        ax1.set_title(f'{self.dataset} ({ratio}) - Loss Convergence', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(labelsize=12)
        
        # === 右图：Accuracy 曲线 ===
        if pretrained_data['val_accuracy']:
            ax2.plot(epochs_pre, pretrained_data['val_accuracy'], 
                    'b-o', linewidth=2, markersize=4, label='Val Accuracy')
        
        ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_title(f'{self.dataset} ({ratio}) - Accuracy Convergence', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f'{self.dataset}_{ratio}_convergence.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved convergence plot: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def generate_all_plots(self, output_dir=None):
        """生成所有可视化图表"""
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, 'exp3_fewshot_plots')
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Generating few-shot learning visualization plots...")
        
        # 1. 小样本对比柱状图（所有比例）
        self.plot_fewshot_comparison(save_dir=output_dir)
        
        # 2. 每个比例的收敛曲线
        for ratio in self.ratios:
            self.plot_convergence_curves(ratio, save_dir=output_dir)
        
        logger.info(f"All visualizations saved to {output_dir}")
        return output_dir


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='Visualize few-shot learning experiment results')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    visualizer = FewShotVisualizer(args.data_dir)
    visualizer.generate_all_plots(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("Visualization Complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
