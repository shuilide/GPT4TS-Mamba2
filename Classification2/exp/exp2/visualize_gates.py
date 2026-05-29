"""
实验二：Gate 参数变化可视化
从训练结果中读取 gate 历史数据并生成可视化图表
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体（如果系统支持）
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Times New Roman']
rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class GateVisualizer:
    """Gate 参数可视化器"""
    
    def __init__(self, data_dir):
        """
        Args:
            data_dir: 包含 gate_tracking 目录的路径
        """
        self.tracking_dir = os.path.join(data_dir, 'gate_tracking')
        self.json_path = os.path.join(self.tracking_dir, 'gate_history.json')
        self.csv_path = os.path.join(self.tracking_dir, 'gate_history.csv')
        self.final_path = os.path.join(self.tracking_dir, 'gate_tracking_final.json')
        
    def load_data(self):
        """加载追踪数据"""
        if not os.path.exists(self.final_path):
            if os.path.exists(self.json_path):
                with open(self.json_path, 'r') as f:
                    return json.load(f)
            else:
                raise FileNotFoundError(f"No gate tracking data found in {self.tracking_dir}")
        
        with open(self.final_path, 'r') as f:
            return json.load(f)
    
    def plot_gate_evolution(self, save_path=None, figsize=(12, 8)):
        """
        绘制 gate 参数随 epoch 的变化曲线
        
        Args:
            save_path: 保存路径（可选）
            figsize: 图像尺寸
        """
        data = self.load_data()
        
        epochs = data['epochs']
        gate_values = data['gate_values']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制每条 Mamba 层的变化曲线
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, (layer_name, values) in enumerate(gate_values.items()):
            color = colors[i % len(colors)]
            ax.plot(epochs, values, marker='o', linewidth=2, markersize=4,
                   label=layer_name, color=color, alpha=0.8)
        
        # 添加参考线
        ax.axhline(y=0.0, color='red', linestyle='--', linewidth=1.5, alpha=0.6,
                  label='Initial Value (0.0)')
        
        # 设置标签和标题
        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Gate Parameter Value (g)', fontsize=14, fontweight='bold')
        ax.set_title('Evolution of Mamba Adapter Gate Parameters During Training', 
                    fontsize=16, fontweight='bold', pad=15)
        
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置刻度
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gate evolution plot saved to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_bar_comparison(self, save_path=None, figsize=(10, 6)):
        """
        绘制初始值和最终值的对比柱状图
        
        Args:
            save_path: 保存路径（可选）
        """
        data = self.load_data()
        stats = data.get('statistics', {})
        
        if not stats:
            logger.warning("No statistics found in tracking data")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        layer_names = list(stats.keys())
        initial_values = [stats[name]['initial'] for name in layer_names]
        final_values = [stats[name]['final'] for name in layer_names]
        
        x = np.arange(len(layer_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, initial_values, width, label='Initial (Epoch 0)',
                      color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x + width/2, final_values, width, label='Final (Last Epoch)',
                      color='#ff7f0e', alpha=0.8)
        
        ax.set_xlabel('Mamba Adapter Layer', fontsize=14, fontweight='bold')
        ax.set_ylabel('Gate Parameter Value (g)', fontsize=14, fontweight='bold')
        ax.set_title('Gate Parameter: Initial vs Final Comparison', 
                    fontsize=16, fontweight='bold', pad=15)
        
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        ax.tick_params(labelsize=12)
        
        # 在柱子上添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Bar comparison plot saved to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def generate_all_plots(self, output_dir=None):
        """
        生成所有可视化图表
        
        Args:
            output_dir: 输出目录（默认在 tracking_dir 下创建 plots 文件夹）
        """
        if output_dir is None:
            output_dir = os.path.join(self.tracking_dir, 'plots')
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Generating gate parameter visualizations...")
        
        # 1. 演化曲线
        evolution_path = os.path.join(output_dir, 'gate_evolution.png')
        self.plot_gate_evolution(save_path=evolution_path)
        
        # 2. 对比柱状图
        bar_path = os.path.join(output_dir, 'gate_comparison.png')
        self.plot_bar_comparison(save_path=bar_path)
        
        logger.info(f"All visualizations saved to {output_dir}")
        
        return output_dir


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='Visualize gate parameter tracking results')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing gate_tracking folder')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: <data_dir>/gate_tracking/plots)')
    parser.add_argument('--show', action='store_true',
                       help='Show plots instead of saving')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                       level=logging.INFO)
    
    visualizer = GateVisualizer(args.data_dir)
    
    if args.show:
        # 显示图表
        visualizer.plot_gate_evolution()
        visualizer.plot_bar_comparison()
    else:
        # 保存图表
        output_dir = visualizer.generate_all_plots(args.output_dir)
        logger.info(f"\nVisualization complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
