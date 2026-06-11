"""
实验六：GPT-2冻结策略实验 (GPT-2 Freezing Strategy Study)
对比三种不同的冻结策略：
1. 全冻结 (Freeze All) - 冻结所有GPT-2层，只训练Mamba和分类头
2. 部分冻结 (Freeze First 3 Layers) - 冻结前3层，微调最后1层
3. 全微调 (Fine-tune All) - 所有GPT-2层均可训练

Usage:
    python main_exp6.py --dataset Heartbeat --gpu 0 --epochs 50 --freeze_strategy all_frozen
    python main_exp6.py --dataset Heartbeat --gpu 0 --epochs 50 --freeze_strategy partial_frozen
    python main_exp6.py --dataset Heartbeat --gpu 0 --epochs 50 --freeze_strategy all_finetune
"""

import sys
import os
import logging
import time

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from options import Options
from running import setup, pipeline_factory, validate, NEG_METRICS
from utils import utils
from datasets.data import data_factory
from datasets.datasplit import split_dataset
from models.gpt4ts import gpt4ts
from models.loss import get_loss_module
from optimizers import get_optimizer

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


def count_trainable_params(model):
    """统计可训练参数数量"""
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return total_params, trainable_params


def main_exp6(config):
    """
    实验六主函数：GPT-2冻结策略实验
    """
    total_epoch_time = 0
    total_start_time = time.time()

    # Add file logging
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    freeze_strategy = config.get('freeze_strategy', 'all_frozen')
    logger.info(f'Running Experiment 6: GPT-2 Freezing Strategy Study - Strategy: {freeze_strategy}')
    logger.info('Command: {}'.format(' '.join(sys.argv)))

    # 设置随机种子
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])

    # 设置设备
    if config['gpu'] != '-1' and torch.cuda.is_available():
        gpu_id = int(config['gpu'])
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
        logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    # 构建数据
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config['data_class']]
    my_data = data_class(config['data_dir'], pattern=config['pattern'], 
                        n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    
    validation_method = 'StratifiedShuffleSplit'
    labels = my_data.labels_df.values.flatten()

    # Split dataset
    test_data = my_data
    test_indices = None
    val_data = my_data
    val_indices = []
    if config['test_pattern']:
        test_data = data_class(config['data_dir'], pattern=config['test_pattern'], n_proc=-1, config=config)
        test_indices = test_data.all_IDs
    if config['test_from']:
        test_indices = list(set([line.rstrip() for line in open(config['test_from']).readlines()]))
        try:
            test_indices = [int(ind) for ind in test_indices]
        except ValueError:
            pass
        logger.info("Loaded {} test IDs from file: '{}'".format(len(test_indices), config['test_from']))
    if config['val_pattern']:
        val_data = data_class(config['data_dir'], pattern=config['val_pattern'], n_proc=-1, config=config)
        val_indices = val_data.all_IDs

    if config['val_ratio'] > 0:
        train_indices, val_indices, test_indices = split_dataset(
            data_indices=my_data.all_IDs,
            validation_method=validation_method,
            n_splits=1,
            validation_ratio=config['val_ratio'],
            test_set_ratio=config['test_ratio'],
            test_indices=test_indices,
            random_seed=1337,
            labels=labels)
        train_indices = train_indices[0]
        val_indices = val_indices[0]
    else:
        train_indices = my_data.all_IDs
        if test_indices is None:
            test_indices = []

    logger.info("{} samples for training".format(len(train_indices)))
    logger.info("{} samples for validation".format(len(val_indices)))
    logger.info("{} samples for testing".format(len(test_indices)))

    # 数据预处理
    from datasets.data import Normalizer
    import pickle
    
    normalizer = None
    if config['norm_from']:
        with open(config['norm_from'], 'rb') as f:
            norm_dict = pickle.load(f)
        normalizer = Normalizer(**norm_dict)
    elif config['normalization'] is not None:
        normalizer = Normalizer(config['normalization'])
        my_data.feature_df.loc[train_indices] = normalizer.normalize(my_data.feature_df.loc[train_indices])
        if not config['normalization'].startswith('per_sample'):
            norm_dict = normalizer.__dict__
            with open(os.path.join(config['output_dir'], 'normalization.pickle'), 'wb') as f:
                pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
    
    if normalizer is not None:
        if len(val_indices):
            if config.get('val_pattern'):
                val_data.feature_df.loc[val_indices] = normalizer.normalize(val_data.feature_df.loc[val_indices])
            else:
                val_data_reload = data_class(config['data_dir'], pattern=config['pattern'], n_proc=-1, config=config)
                val_data_reload.feature_df.loc[val_indices] = normalizer.normalize(val_data_reload.feature_df.loc[val_indices])
                val_data = val_data_reload
        if len(test_indices):
            test_data.feature_df.loc[test_indices] = normalizer.normalize(test_data.feature_df.loc[test_indices])

    # 创建模型
    logger.info("Creating model ...")
    # gpt4ts.py 内部已经根据 config['freeze_strategy'] 自动处理冻结逻辑
    # 与主实验 main.py 的逻辑完全一致
    model = gpt4ts(config, my_data)
    
    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))

    # 初始化优化器
    weight_decay = config['l2_reg'] if config['global_reg'] else 0
    output_reg = None if config['global_reg'] else config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    
    # 全微调时使用更小的学习率
    if freeze_strategy == 'all_finetune':
        lr = config.get('lr', 1e-5)  # 更小的学习率防止灾难性遗忘
    else:
        lr = config.get('lr', 1e-4)
    
    optimizer = optim_class(model.parameters(), lr=lr, weight_decay=weight_decay)
    logger.info(f"Learning rate: {lr}")

    start_epoch = 0
    lr_step = 0

    # 加载预训练模型（如果有）
    if config.get('load_model'):
        logger.info(f"Loading model from {config['load_model']}")
        checkpoint = torch.load(config['load_model'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if config['resume']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']

    model.to(device)
    loss_module = get_loss_module(config)

    # 创建数据加载器
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    
    val_dataset = dataset_class(val_data, val_indices)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=config['num_workers'],
                           pin_memory=True, collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    train_dataset = dataset_class(my_data, train_indices)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'],
                             shuffle=True, num_workers=config['num_workers'],
                             pin_memory=True, collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    trainer = runner_class(model, train_loader, device, loss_module, optimizer, 
                          l2_reg=output_reg, print_interval=config['print_interval'], 
                          console=config['console'])
    val_evaluator = runner_class(model, val_loader, device, loss_module,
                                print_interval=config['print_interval'], 
                                console=config['console'])

    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

    best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16
    metrics = []
    best_metrics = {}

    # 验证初始状态
    if len(val_indices) > 0:
        aggr_metrics_val, best_metrics, best_value = validate(
            val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch=0)
        metrics_names, metrics_values = zip(*aggr_metrics_val.items())
        metrics.append(list(metrics_values))
    else:
        logger.warning("No validation samples found.")
        aggr_metrics_val = {'loss': 0.0}
        best_metrics = aggr_metrics_val.copy()
        best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16

    logger.info(f'Starting training with freeze strategy: {freeze_strategy}...')
    
    for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc='Training Epoch', leave=False):
        mark = epoch if config['save_all'] else 'last'
        epoch_start_time = time.time()
        
        # 训练一个 epoch
        aggr_metrics_train = trainer.train_epoch(epoch)
        epoch_runtime = time.time() - epoch_start_time
        
        print()
        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(
            *utils.readable_time(epoch_runtime)))
        total_epoch_time += epoch_runtime

        # 验证
        if len(val_indices) > 0 and ((epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0)):
            aggr_metrics_val, best_metrics, best_value = validate(
                val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch)
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))

        # 保存模型
        utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(mark)), epoch, model, optimizer)

        # 学习率调度
        if epoch in config['lr_step']:
            utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
            lr = lr * config['lr_factor'][lr_step]
            if lr_step < len(config['lr_step']) - 1:
                lr_step += 1
            logger.info('Learning rate updated to: {}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    # 统计最终参数
    total_params, trainable_params = count_trainable_params(model)
    trainable_ratio = trainable_params / total_params * 100
    
    # 计算总运行时间
    total_training_time = time.time() - total_start_time
    avg_epoch_time = total_epoch_time / config["epochs"] if config["epochs"] > 0 else 0

    # 保存最终结果
    exp6_summary = {
        'experiment': 'Experiment 6 - GPT-2 Freezing Strategy Study',
        'dataset': config['data_class'],
        'freeze_strategy': freeze_strategy,
        'total_epochs': config['epochs'],
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'trainable_ratio_percent': trainable_ratio,
        'total_training_time_seconds': total_training_time,  # 保存总时间（秒）
        'avg_epoch_time_seconds': avg_epoch_time,           # 保存平均每轮时间（秒）
        'best_validation_metrics': best_metrics,
        'conclusion': {
            'all_frozen': 'Freeze all GPT-2 layers: Fast training, good for small datasets, prevents overfitting',
            'partial_frozen': 'Partial freezing: Balance between adaptation and overfitting prevention',
            'all_finetune': 'Fine-tune all: Best performance on large datasets, risk of overfitting on small datasets'
        }
    }
    
    summary_path = os.path.join(config['output_dir'], 'experiment6_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(exp6_summary, f, indent=4)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"EXPERIMENT 6 COMPLETE - Freeze Strategy: {freeze_strategy}")
    logger.info(f"{'='*80}")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,} ({trainable_ratio:.2f}%)")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"{'='*80}\n")

    logger.info('Best {} was {}. Other metrics: {}'.format(
        config['key_metric'], best_value, best_metrics))
    logger.info('All Done!')

    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(
        *utils.readable_time(total_runtime)))

    return best_value


if __name__ == '__main__':
    args = Options().parse()
    config = setup(args)

    # 实验六默认配置
    config.setdefault('freeze_strategy', 'all_frozen')
    config.setdefault('lr', 1e-4)  # 默认学习率

    main_exp6(config)
