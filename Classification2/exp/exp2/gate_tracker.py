"""
实验二：参数 g 追踪工具
用于记录训练过程中 Mamba Adapter 门控参数的变化轨迹
"""

import os
import json
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class GateTracker:
    """追踪 ParallelMambaAdapter 中 gate 参数的变化"""
    
    def __init__(self, output_dir, num_layers=2):
        """
        Args:
            output_dir: 输出目录
            num_layers: Mamba adapter 层数
        """
        self.output_dir = output_dir
        self.num_layers = num_layers
        
        # 存储每个 epoch 的 gate 值和 accuracy
        self.gate_history = {
            'epochs': [],
            'gate_values': {},  # {layer_name: [value_per_epoch]}
            'accuracy': [],  # [accuracy_per_epoch]
            'metadata': {
                'initial_value': 0.1,
                'num_layers': num_layers,
                'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # 创建保存目录
        self.tracking_dir = os.path.join(output_dir, 'gate_tracking')
        os.makedirs(self.tracking_dir, exist_ok=True)
        
        logger.info(f"GateTracker initialized: tracking {num_layers} Mamba adapter layers")
    
    def record_epoch_gates(self, epoch, model, accuracy=None):
        """
        记录当前 epoch 所有 Mamba 层的 gate 值和 accuracy
        
        Args:
            epoch: 当前 epoch 编号
            model: gpt4ts 模型实例
            accuracy: 验证集上的 accuracy（可选）
        """
        # 获取所有 gate 值
        gate_values = model.get_all_gate_values()
        
        # 记录到历史
        self.gate_history['epochs'].append(epoch)
        
        for layer_name, gate_value in gate_values.items():
            if layer_name not in self.gate_history['gate_values']:
                self.gate_history['gate_values'][layer_name] = []
            self.gate_history['gate_values'][layer_name].append(gate_value)
        
        # 记录 accuracy
        if accuracy is not None:
            self.gate_history['accuracy'].append(accuracy)
        
        # 打印当前 gate 值和 accuracy
        gate_str = f"Epoch {epoch} - Gate values: "
        for layer_name, value in gate_values.items():
            gate_str += f"{layer_name}={value:.4f}, "
        if accuracy is not None:
            gate_str += f"Accuracy={accuracy:.4f}"
        logger.info(gate_str.rstrip(", "))
        
        # 每 10 个 epoch 保存一次
        if epoch % 10 == 0 or epoch == 1:
            self.save_checkpoint()
    
    def save_checkpoint(self):
        """保存当前追踪数据"""
        checkpoint_path = os.path.join(self.tracking_dir, 'gate_history.json')
        with open(checkpoint_path, 'w') as f:
            json.dump(self.gate_history, f, indent=4)
        logger.info(f"Gate history saved to {checkpoint_path}")
    
    def save_final_results(self):
        """保存最终结果并生成统计信息"""
        # 更新结束时间
        self.gate_history['metadata']['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.gate_history['metadata']['total_epochs'] = len(self.gate_history['epochs'])
        
        # 计算统计信息
        stats = {}
        for layer_name, values in self.gate_history['gate_values'].items():
            if len(values) > 0:
                stats[layer_name] = {
                    'initial': values[0],
                    'final': values[-1],
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'change': values[-1] - values[0],
                    'change_percent': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                }
        
        self.gate_history['statistics'] = stats
        
        # 保存最终结果
        final_path = os.path.join(self.tracking_dir, 'gate_tracking_final.json')
        with open(final_path, 'w') as f:
            json.dump(self.gate_history, f, indent=4)
        
        # 生成 CSV 格式（方便导入 Excel）
        self._save_csv()
        
        # 打印统计信息
        logger.info("\n" + "="*80)
        logger.info("Gate Parameter Tracking Summary")
        logger.info("="*80)
        for layer_name, layer_stats in stats.items():
            logger.info(f"\nLayer: {layer_name}")
            logger.info(f"  Initial: {layer_stats['initial']:.4f}")
            logger.info(f"  Final:   {layer_stats['final']:.4f}")
            logger.info(f"  Change:  {layer_stats['change']:+.4f} ({layer_stats['change_percent']:+.2f}%)")
            logger.info(f"  Min:     {layer_stats['min']:.4f}")
            logger.info(f"  Max:     {layer_stats['max']:.4f}")
        logger.info("="*80 + "\n")
        
        logger.info(f"Final gate tracking results saved to {self.tracking_dir}")
        
        return stats
    
    def _save_csv(self):
        """保存为 CSV 格式"""
        csv_path = os.path.join(self.tracking_dir, 'gate_history.csv')
        
        # 获取所有层名
        layer_names = list(self.gate_history['gate_values'].keys())
        
        # 写入 CSV
        with open(csv_path, 'w') as f:
            # 表头
            header = "epoch," + ",".join(layer_names) + ",accuracy\n"
            f.write(header)
            
            # 数据行
            for i, epoch in enumerate(self.gate_history['epochs']):
                values = [str(self.gate_history['gate_values'][layer][i]) for layer in layer_names]
                acc = str(self.gate_history['accuracy'][i]) if i < len(self.gate_history['accuracy']) else ""
                line = f"{epoch}," + ",".join(values) + f",{acc}\n"
                f.write(line)
        
        logger.info(f"CSV format saved to {csv_path}")


def create_gate_tracker(config, num_mamba_layers=2):
    """
    创建 GateTracker 实例
    
    Args:
        config: 配置字典
        num_mamba_layers: Mamba adapter 层数
    
    Returns:
        GateTracker 实例，如果不启用追踪则返回 None
    """
    if not config.get('track_gate', False):
        return None
    
    output_dir = config['output_dir']
    tracker = GateTracker(output_dir, num_layers=num_mamba_layers)
    
    return tracker
