#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验4：模型可解释性分析 - 特征与注意力提取工具
从训练好的模型中提取 Attention Weights 和 Pooled Features
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

# 添加 src 路径
# 使用 insert(0) 确保本地 modules 优先于 site-packages (避免与 HuggingFace datasets 库冲突)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from options import Options
from datasets.data import data_factory
from models.gpt4ts import gpt4ts
from utils import utils
from running import pipeline_factory

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main(config):
    # 修复路径问题：区分 data_dir 和 load_model 的相对基准目录
    # data_dir 是相对于项目根目录 (Classification2) 的
    # load_model 是相对于脚本运行目录 (exp/exp4) 的
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    if not os.path.isabs(config['data_dir']):
        # 移除开头的 ./ 防止拼接成 .././datasets
        clean_dir = config['data_dir'].replace('./', '', 1)
        config['data_dir'] = os.path.join(base_dir, clean_dir)
        logger.info(f"Resolved data_dir to: {config['data_dir']}")
    
    if not os.path.isabs(config['load_model']):
        # 直接转为绝对路径，基准为当前工作目录
        config['load_model'] = os.path.abspath(config['load_model'])
        logger.info(f"Resolved load_model to: {config['load_model']}")
        
    logger.info(f"Loading model from: {config['load_model']}")
    
    # 1. 加载数据
    data_class = data_factory[config['data_class']]
    my_data = data_class(config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    feat_dim = my_data.feature_df.shape[1]
    
    # 数据集划分 (与训练一致)
    test_data = my_data
    test_indices = None
    val_data = my_data
    val_indices = []
    
    if config['test_pattern']:
        test_data = data_class(config['data_dir'], pattern=config['test_pattern'], n_proc=-1, config=config)
        test_indices = test_data.all_IDs
    if config['val_pattern']:
        val_data = data_class(config['data_dir'], pattern=config['val_pattern'], n_proc=-1, config=config)
        val_indices = val_data.all_IDs
        
    if config['val_ratio'] > 0:
        from datasets.datasplit import split_dataset
        if config['task'] == 'classification':
            validation_method = 'StratifiedShuffleSplit'
            labels = my_data.labels_df.values.flatten()
        else:
            validation_method = 'ShuffleSplit'
            labels = None
            
        train_indices, val_indices, test_indices = split_dataset(
            data_indices=my_data.all_IDs, validation_method=validation_method,
            n_splits=1, validation_ratio=config['val_ratio'], test_set_ratio=config['test_ratio'],
            test_indices=test_indices, random_seed=1337, labels=labels)
        train_indices = train_indices[0]
        val_indices = val_indices[0]
    else:
        train_indices = my_data.all_IDs
        if test_indices is None: test_indices = []
        
    # 归一化
    normalizer = None
    if config['normalization'] is not None:
        from datasets.data import Normalizer
        normalizer = Normalizer(config['normalization'])
        my_data.feature_df.loc[train_indices] = normalizer.normalize(my_data.feature_df.loc[train_indices])
        if len(val_indices):
            val_data.feature_df.loc[val_indices] = normalizer.normalize(val_data.feature_df.loc[val_indices])
        if len(test_indices):
            test_data.feature_df.loc[test_indices] = normalizer.normalize(test_data.feature_df.loc[test_indices])

    # 2. 创建模型
    model = gpt4ts(config, my_data)
    checkpoint = torch.load(config['load_model'], map_location='cpu')
    # 兼容不同的保存格式：优先查找 'state_dict'，其次 'model_state_dict'
    state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict')
    if state_dict is None:
        # 如果 checkpoint 本身就是 state_dict
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 3. 开启中间结果捕获模式
    # 确保属性正确设置（兼容 DataParallel 包装）
    target_model = model.module if hasattr(model, 'module') else model
    target_model.return_attn = True
    logger.info(f"Set return_attn to True on {type(target_model).__name__}")
    logger.info(f"Verified: target_model.return_attn = {target_model.return_attn}")
    
    device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() and config['gpu'] != '-1' else 'cpu')
    model.to(device)
    
    # 4. 构建 DataLoader
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    
    # 假设我们分析验证集（通常代表模型性能最好的状态）
    # 注意：pipeline_factory 可能会根据 config 自动处理数据划分
    # 这里我们手动构建 dataset 以获取所有样本
    # 为了简单，我们直接使用 config 中的 val_pattern 或 val_ratio 逻辑
    # 但 pipeline_factory 返回的类通常需要 (data, indices)
    
    # 重新获取 indices (与 main.py 逻辑一致)
    # ... (copy logic from main.py to get val_indices)
    # 为了简化，我们假设用户想分析测试集或验证集
    # 这里直接使用 val_indices 如果存在
    
    if len(val_indices) > 0:
        target_indices = val_indices
        target_data = val_data
    else:
        target_indices = test_indices
        target_data = test_data
        
    dataset = dataset_class(target_data, target_indices)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # 5. 提取数据
    all_features = []
    all_labels = []
    
    # 仅提取第一个样本的 Attention
    sample_attention = None
    sample_raw_data = None
    sample_label = None
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # 安全解包：兼容不同数量的返回值
            x_enc = batch[0].to(device)
            
            # 提取标签 (通常在第1个索引，兼容元组包装的情况)
            y_raw = batch[1] if len(batch) > 1 else batch[0]
            if isinstance(y_raw, tuple):
                y_np = y_raw[0].cpu().numpy()
            else:
                y_np = y_raw.cpu().numpy()
                
            x_mark_enc = None  # 分类任务通常不依赖 mark 数据
            
            # Forward
            outputs = model(x_enc, x_mark_enc)
            
            # ✅ 关键修复：从 target_model（解包后的模型）获取中间结果
            # 因为 DataParallel 包装后，属性保存在 module 中
            if hasattr(target_model, 'last_pooled_feature') and target_model.last_pooled_feature is not None:
                feats = target_model.last_pooled_feature.cpu().numpy()
                all_features.append(feats)
            
            all_labels.append(y_np)
            
            # ✅ 收集第一个样本的 Attention（仅在 Attention Pooling 模式下）
            if i == 0 and target_model.last_attention_weights is not None:
                # attn_weights shape: [B, 1, N]
                # 取第一个样本 [0, 0, :]
                sample_attention = target_model.last_attention_weights[0, 0, :].cpu().numpy()
                sample_raw_data = x_enc[0].cpu().numpy() # [L, M]
                sample_label = y_np[0].item()
                
    # 6. 保存结果
    os.makedirs(config['output_dir'], exist_ok=True)
    
    if len(all_features) > 0:
        features_arr = np.concatenate(all_features, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)
        
        np.save(os.path.join(config['output_dir'], 'features.npy'), features_arr)
        np.save(os.path.join(config['output_dir'], 'labels.npy'), labels_arr)
        logger.info(f"Saved features: {features_arr.shape}")
    
    if sample_attention is not None:
        np.save(os.path.join(config['output_dir'], 'sample_attention.npy'), sample_attention)
        np.save(os.path.join(config['output_dir'], 'sample_raw_data.npy'), sample_raw_data)
        np.save(os.path.join(config['output_dir'], 'sample_label.npy'), np.array([sample_label]))
        logger.info(f"Saved sample attention: {sample_attention.shape}")
        
    logger.info("Extraction complete!")


if __name__ == '__main__':
    options = Options()
    args = options.parse()
    config = vars(args)
    main(config)
