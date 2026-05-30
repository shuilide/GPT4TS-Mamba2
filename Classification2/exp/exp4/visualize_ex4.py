#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验4：模型可解释性可视化
1. Attention Heatmap (波形 + 权重)
2. t-SNE Feature Distribution (Flatten vs Attention)
"""

import os
import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(model_dir):
    """加载模型配置以获取 patch_size 和 stride"""
    config_path = os.path.join(model_dir, 'configuration.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def plot_attention_heatmap(raw_data, attention_weights, config, save_path):
    """
    绘制波形与注意力权重叠加图
    raw_data: [L, M] (原始序列)
    attention_weights: [N] (每个 patch 的权重)
    """
    L, M = raw_data.shape
    N = len(attention_weights)
    
    # 获取 patch 参数
    patch_size = config.get('patch_size', 16)
    stride = config.get('stride', 2)
    
    # 创建图像
    fig, axes = plt.subplots(M, 1, figsize=(15, 3 * M), sharex=True)
    if M == 1:
        axes = [axes]
        
    # 将 attention weights 映射回时间轴 (简单的最近邻插值或重复)
    # 每个 patch 对应的时间步范围
    attn_expanded = np.zeros(L)
    for i in range(N):
        start = i * stride
        end = min(start + patch_size, L)
        if start < L:
            attn_expanded[start:end] = attention_weights[i]
            
    # 归一化 attn_expanded 以便绘图 (0-1)
    attn_expanded = (attn_expanded - attn_expanded.min()) / (attn_expanded.max() - attn_expanded.min() + 1e-8)

    for dim in range(M):
        ax = axes[dim]
        time_steps = np.arange(L)
        
        # 绘制波形
        ax.plot(time_steps, raw_data[:, dim], color='black', linewidth=1, label='Signal')
        
        # 绘制注意力权重 (作为填充区域)
        ax.fill_between(time_steps, 0, attn_expanded, color='red', alpha=0.3, label='Attention Weight')
        
        ax.set_ylabel(f'Dim {dim}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel('Time Steps')
    plt.suptitle('Attention Heatmap Overlay on Raw Signal', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved attention heatmap: {save_path}")
    plt.close()


def plot_tsne_comparison(features_attn, labels_attn, features_flat, labels_flat, save_path):
    """
    绘制 t-SNE 对比图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # ✅ 修复：确保 labels 是 1D 数组，避免索引错误
    if labels_attn.ndim > 1:
        labels_attn = labels_attn.flatten()
    if labels_flat is not None and labels_flat.ndim > 1:
        labels_flat = labels_flat.flatten()
    
    # 确定类别数
    classes = np.unique(labels_attn)
    # ✅ 修复：使用新版 matplotlib API
    cmap = plt.colormaps['tab10'].resampled(len(classes))
    
    # Plot 1: Flatten (Adaptive Pooling)
    if features_flat is not None:
        logger.info("Computing t-SNE for Flatten features...")
        
        # ✅ 修复：根据样本数动态调整 perplexity，避免样本数过少时报错
        n_samples_flat = len(features_flat)
        perplexity_flat = min(30, n_samples_flat - 1)
        if perplexity_flat < 1:
            logger.warning(f"Flatten: Too few samples ({n_samples_flat}) for t-SNE, skipping.")
            ax1.text(0.5, 0.5, 'Too few samples for t-SNE', ha='center', va='center', fontsize=20)
        else:
            tsne_flat = TSNE(n_components=2, random_state=42, perplexity=perplexity_flat)
            embed_flat = tsne_flat.fit_transform(features_flat)
            
            for i, cls in enumerate(classes):
                mask = labels_flat == cls
                ax1.scatter(embed_flat[mask, 0], embed_flat[mask, 1], 
                           c=[cmap(i)], label=f'Class {cls}', alpha=0.7, s=50)
            ax1.set_title(f'Feature Distribution: Flatten (Adaptive Pooling)\n(perplexity={perplexity_flat})', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No Flatten Data', ha='center', va='center', fontsize=20)
        
    # Plot 2: Attention Pooling
    logger.info("Computing t-SNE for Attention features...")
    
    # ✅ 修复：根据样本数动态调整 perplexity
    n_samples_attn = len(features_attn)
    perplexity_attn = min(30, n_samples_attn - 1)
    if perplexity_attn < 1:
        logger.warning(f"Attention: Too few samples ({n_samples_attn}) for t-SNE, skipping.")
        ax2.text(0.5, 0.5, 'Too few samples for t-SNE', ha='center', va='center', fontsize=20)
    else:
        tsne_attn = TSNE(n_components=2, random_state=42, perplexity=perplexity_attn)
        embed_attn = tsne_attn.fit_transform(features_attn)
        
        for i, cls in enumerate(classes):
            mask = labels_attn == cls
            ax2.scatter(embed_attn[mask, 0], embed_attn[mask, 1], 
                       c=[cmap(i)], label=f'Class {cls}', alpha=0.7, s=50)
        ax2.set_title(f'Feature Distribution: Attention Pooling\n(perplexity={perplexity_attn})', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('t-SNE Visualization: Flatten vs Attention Pooling', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved t-SNE plot: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Experiment 4 Results')
    parser.add_argument('--data_attn_dir', type=str, required=True, help='Directory with Attention Pooling data')
    parser.add_argument('--data_flat_dir', type=str, default=None, help='Directory with Flatten data (optional)')
    parser.add_argument('--model_dir', type=str, required=True, help='Model directory (for config)')
    parser.add_argument('--output_dir', type=str, default='plots', help='Output directory')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = load_config(args.model_dir)
    
    # 1. Attention Heatmap
    attn_file = os.path.join(args.data_attn_dir, 'sample_attention.npy')
    raw_file = os.path.join(args.data_attn_dir, 'sample_raw_data.npy')
    
    if os.path.exists(attn_file) and os.path.exists(raw_file):
        attention = np.load(attn_file)
        raw_data = np.load(raw_file)
        # raw_data shape might be [L, M] or [M, L] depending on how it was saved.
        # In extract_data.py: x_enc[0] is [L, M] usually.
        # But in gpt4ts forward: x_enc is [B, L, M].
        # Let's assume [L, M].
        if raw_data.ndim == 2:
             pass # OK
        plot_attention_heatmap(raw_data, attention, config, 
                              os.path.join(args.output_dir, 'attention_heatmap.png'))
    else:
        logger.warning("Sample data not found for heatmap.")
        
    # 2. t-SNE
    feat_attn_file = os.path.join(args.data_attn_dir, 'features.npy')
    label_attn_file = os.path.join(args.data_attn_dir, 'labels.npy')
    
    if os.path.exists(feat_attn_file):
        features_attn = np.load(feat_attn_file)
        labels_attn = np.load(label_attn_file)
        
        features_flat = None
        labels_flat = None
        
        if args.data_flat_dir:
            feat_flat_file = os.path.join(args.data_flat_dir, 'features.npy')
            label_flat_file = os.path.join(args.data_flat_dir, 'labels.npy')
            if os.path.exists(feat_flat_file):
                features_flat = np.load(feat_flat_file)
                labels_flat = np.load(label_flat_file)
        
        plot_tsne_comparison(features_attn, labels_attn, features_flat, labels_flat,
                            os.path.join(args.output_dir, 'tsne_comparison.png'))
    else:
        logger.warning("Features data not found for t-SNE.")
        
    logger.info("All visualizations saved to %s", args.output_dir)


if __name__ == '__main__':
    main()
