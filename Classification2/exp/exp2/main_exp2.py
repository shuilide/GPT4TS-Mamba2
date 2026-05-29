"""
实验二：参数 g 训练轨迹追踪
记录 Mamba Adapter 门控参数在训练过程中的变化

Usage:
    python main_exp2.py --dataset Heartbeat --gpu 0 --epochs 50
    python main_exp2.py --dataset EthanolConcentration --gpu 0 --epochs 100
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

# 导入实验二的 gate 追踪工具
from gate_tracker import create_gate_tracker

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


def main_exp2(config):
    """
    实验二主函数：训练并追踪 gate 参数变化
    """
    total_epoch_time = 0
    total_start_time = time.time()

    # Add file logging
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running Experiment 2: Gate Parameter Tracking')
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
    test_indices = None  # will be converted to empty list in `split_dataset`, if also test_set_ratio == 0
    val_data = my_data
    val_indices = []
    if config['test_pattern']:  # used if test data come from different files / file patterns
        test_data = data_class(config['data_dir'], pattern=config['test_pattern'], n_proc=-1, config=config)
        test_indices = test_data.all_IDs
    if config['test_from']:  # load test IDs directly from file, if available, otherwise use `test_set_ratio`. Can work together with `test_pattern`
        test_indices = list(set([line.rstrip() for line in open(config['test_from']).readlines()]))
        try:
            test_indices = [int(ind) for ind in test_indices]  # integer indices
        except ValueError:
            pass  # in case indices are non-integers
        logger.info("Loaded {} test IDs from file: '{}'".format(len(test_indices), config['test_from']))
    if config['val_pattern']:  # used if val data come from different files / file patterns
        val_data = data_class(config['data_dir'], pattern=config['val_pattern'], n_proc=-1, config=config)
        val_indices = val_data.all_IDs

    # Note: currently a validation set must exist, either with `val_pattern` or `val_ratio`
    # Using a `val_pattern` means that `val_ratio` == 0 and `test_ratio` == 0
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
            # 如果验证集来自独立文件（val_pattern），直接使用 val_data
            # 否则需要重新加载并应用归一化
            if config.get('val_pattern'):
                # 验证集来自独立文件，直接归一化
                val_data.feature_df.loc[val_indices] = normalizer.normalize(val_data.feature_df.loc[val_indices])
            else:
                # 验证集从训练集分割出来，需要重新加载
                val_data_reload = data_class(config['data_dir'], pattern=config['pattern'], n_proc=-1, config=config)
                val_data_reload.feature_df.loc[val_indices] = normalizer.normalize(val_data_reload.feature_df.loc[val_indices])
                val_data = val_data_reload
        if len(test_indices):
            test_data.feature_df.loc[test_indices] = normalizer.normalize(test_data.feature_df.loc[test_indices])

    # 创建模型
    logger.info("Creating model ...")
    model = gpt4ts(config, my_data)
    
    # 初始化 gate 参数为 0（实验要求）
    num_mamba_layers = config.get('num_mamba_layers', 2)
    for name, module in model.gpt2.named_modules():
        if hasattr(module, 'set_gate_initial'):
            module.set_gate_initial(0.0)
            logger.info(f"✓ Initialized gate parameter to 0.0 for {name}")

    logger.info("Model:\n{}".format(model))
    logger.info("Total parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))

    # 创建 Gate Tracker
    gate_tracker = create_gate_tracker(config, num_mamba_layers=num_mamba_layers)
    if gate_tracker is not None:
        logger.info("✓ Gate tracking enabled for Experiment 2")
    else:
        logger.warning("⚠ Gate tracking is disabled. Set --track_gate to enable.")

    # 初始化优化器（策略 2：Gate 参数独立配置，加速自适应）
    weight_decay = config['l2_reg'] if config['global_reg'] else 0
    output_reg = None if config['global_reg'] else config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    
    # 将参数分组：Gate 参数独立设置（更高学习率、无权重衰减）
    gate_params = [p for n, p in model.named_parameters() if 'gate' in n.lower()]
    base_params = [p for n, p in model.named_parameters() if 'gate' not in n.lower()]
    
    param_groups = [
        {'params': base_params, 'lr': config['lr'], 'weight_decay': weight_decay},
        {'params': gate_params, 'lr': 5e-3, 'weight_decay': 0.0}  # Gate 参数：固定学习率 5e-3，零衰减
    ]
    
    optimizer = optim_class(param_groups, lr=config['lr'])
    
    logger.info(f"✓ Optimizer configured: Gate params use fixed LR 5e-3 and 0 weight decay")

    start_epoch = 0
    lr_step = 0
    lr = config['lr']

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
        logger.warning("No validation samples found. Skipping validation and initializing best metrics.")
        aggr_metrics_val = {'loss': 0.0}
        best_metrics = aggr_metrics_val.copy()
        best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16

    # 记录初始 gate 值和 accuracy
    if gate_tracker is not None:
        initial_accuracy = aggr_metrics_val.get('accuracy', None) if len(val_indices) > 0 else None
        gate_tracker.record_epoch_gates(0, model, accuracy=initial_accuracy)

    logger.info('Starting training with gate parameter tracking...')
    
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
        current_accuracy = None
        if len(val_indices) > 0 and ((epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0)):
            aggr_metrics_val, best_metrics, best_value = validate(
                val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch)
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))
            # 获取当前 epoch 的 validation accuracy
            current_accuracy = aggr_metrics_val.get('accuracy', None)

        # 记录 gate 值和 accuracy
        if gate_tracker is not None:
            gate_tracker.record_epoch_gates(epoch, model, accuracy=current_accuracy)

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

    # 保存最终结果
    if gate_tracker is not None:
        gate_stats = gate_tracker.save_final_results()
        
        # 保存实验二专用总结文件
        exp2_summary = {
            'experiment': 'Experiment 2 - Gate Parameter Tracking',
            'dataset': config['data_class'],
            'num_mamba_layers': num_mamba_layers,
            'total_epochs': config['epochs'],
            'initial_gate_value': 0.1,
            'final_gate_statistics': gate_stats,
            'best_validation_metrics': best_metrics,
            'conclusion': {
                'long_sequence': 'Gate value should increase for long sequences (model learns to rely on Mamba)',
                'short_sequence': 'Gate value may remain small for short sequences (GPT attention is sufficient)'
            }
        }
        
        summary_path = os.path.join(config['output_dir'], 'experiment2_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(exp2_summary, f, indent=4)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"EXPERIMENT 2 COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Summary saved to: {summary_path}")
        logger.info(f"Gate tracking data: {gate_tracker.tracking_dir}")
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

    # 实验二默认配置
    config.setdefault('lr', 1e-4)
    config.setdefault('freeze', False)
    config.setdefault('track_gate', True)  # 默认启用 gate 追踪

    main_exp2(config)
