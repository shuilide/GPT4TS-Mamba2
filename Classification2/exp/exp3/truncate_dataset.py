#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集截断脚本 - 用于 Few-Shot Learning 实验
按比例截断训练集，保留原始测试集和验证集不变
"""

import os
import argparse
import logging
import shutil
import pandas as pd
from pathlib import Path

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def truncate_dataset(data_dir, output_dir, ratio=0.1, random_seed=42):
    """
    按比例截断数据集的训练集
    
    Args:
        data_dir: 原始数据集目录
        output_dir: 输出目录（截断后的数据集）
        ratio: 保留比例（0.1 = 10%, 0.5 = 50%）
        random_seed: 随机种子
    """
    logger.info(f"Loading dataset from: {data_dir}")
    logger.info(f"Target ratio: {ratio*100}%")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找 TRAIN 文件
    train_files = list(Path(data_dir).glob('*TRAIN*'))
    if not train_files:
        raise FileNotFoundError(f"No TRAIN files found in {data_dir}")
    
    for train_file in train_files:
        logger.info(f"Processing: {train_file.name}")
        
        # 读取数据
        if train_file.suffix == '.ts':
            # 解析 .ts 格式文件
            df = parse_ts_file(train_file)
            if df is None:
                logger.warning(f"Failed to parse {train_file.name}, skipping...")
                continue
            
            # 按样本数截断
            n_samples = len(df)
            n_keep = max(1, int(n_samples * ratio))
            
            # 随机采样
            df_truncated = df.sample(n=n_keep, random_state=random_seed).sort_index()
            
            logger.info(f"Original: {n_samples} samples -> Truncated: {n_keep} samples")
            
            # 保存截断后的数据
            output_file = os.path.join(output_dir, train_file.name)
            save_ts_file(df_truncated, output_file)
            
        elif train_file.suffix == '.csv':
            df = pd.read_csv(train_file)
            n_samples = len(df)
            n_keep = max(1, int(n_samples * ratio))
            df_truncated = df.sample(n=n_keep, random_state=random_seed).sort_index()
            
            logger.info(f"Original: {n_samples} samples -> Truncated: {n_keep} samples")
            df_truncated.to_csv(os.path.join(output_dir, train_file.name), index=False)
        else:
            logger.warning(f"Unsupported file format: {train_file.suffix}")
    
    # 复制 TEST 文件（保持不变）
    test_files = list(Path(data_dir).glob('*TEST*'))
    for test_file in test_files:
        shutil.copy(test_file, os.path.join(output_dir, test_file.name))
        logger.info(f"Copied test file: {test_file.name}")
    
    # 复制其他配置文件（如 .ts 格式的描述文件）
    for file in Path(data_dir).iterdir():
        if file.is_file() and file.suffix not in ['.TRAIN', '.TEST', '.csv', '.ts']:
            shutil.copy(file, os.path.join(output_dir, file.name))
    
    logger.info(f"Truncated dataset saved to: {output_dir}")
    return output_dir


def parse_ts_file(filepath):
    """解析 .ts 格式的时间序列文件"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # 跳过元数据部分
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('@data'):
                data_start = i + 1
                break
        
        # 提取数据
        data_lines = lines[data_start:]
        return pd.DataFrame(data_lines)
    except Exception as e:
        logger.error(f"Failed to parse .ts file: {e}")
        return None


def save_ts_file(df, filepath):
    """保存为 .ts 格式"""
    try:
        # 重建 .ts 文件头
        header = """@problemName timeSeries
@timeStamps false
@missing false
@univariate false
@equalLength true
@seriesLength unknown
@classLabel true
@data
"""
        with open(filepath, 'w') as f:
            f.write(header)
            for _, row in df.iterrows():
                f.write(row[0])
        return True
    except Exception as e:
        logger.error(f"Failed to save .ts file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Truncate dataset for few-shot learning experiments')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Original dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for truncated dataset')
    parser.add_argument('--ratio', type=float, default=0.1,
                       help='Truncation ratio (e.g., 0.1 for 10%%, 0.5 for 50%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Dataset Truncation Tool for Few-Shot Learning")
    logger.info("=" * 60)
    
    truncate_dataset(args.data_dir, args.output_dir, args.ratio, args.seed)
    
    logger.info("=" * 60)
    logger.info("Truncation Complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
