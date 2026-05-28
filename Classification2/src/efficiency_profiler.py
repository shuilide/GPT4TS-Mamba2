"""
效率与复杂度分析工具
用于测量模型在不同序列长度下的显存占用和计算时间
"""

import torch
import time
import csv
import os
from tqdm import tqdm


class EfficiencyProfiler:
    """效率分析器：测量模型的显存和计算复杂度"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def clear_gpu_cache(self):
        """清空GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_peak_memory_mb(self):
        """获取峰值显存（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        return 0.0
    
    def profile_single_run(self, input_tensor, mark_tensor=None, num_runs=3):
        """
        单次性能分析
        Args:
            input_tensor: 输入张量
            mark_tensor: 标记张量（可选）
            num_runs: 运行次数取平均
        Returns:
            dict: 包含显存和时间信息
        """
        self.model.eval()
        self.clear_gpu_cache()
        
        # 预热
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device), mark_tensor.to(self.device) if mark_tensor is not None else None)
        
        self.clear_gpu_cache()
        torch.cuda.synchronize(self.device) if torch.cuda.is_available() else None
        
        # 正式测量
        times = []
        peak_memory = 0.0
        
        for _ in range(num_runs):
            self.clear_gpu_cache()
            start_time = time.time()
            
            with torch.no_grad():
                output = self.model(input_tensor.to(self.device), 
                                   mark_tensor.to(self.device) if mark_tensor is not None else None)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
                current_memory = self.get_peak_memory_mb()
                peak_memory = max(peak_memory, current_memory)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        
        return {
            'peak_memory_mb': peak_memory,
            'inference_time_s': avg_time,
            'output_shape': list(output.shape)
        }
    
    def profile_sequence_lengths(self, seq_lengths, feat_dim, batch_size=1, 
                                  patch_size=8, stride=8, save_path=None):
        """
        测试不同序列长度下的性能
        Args:
            seq_lengths: 序列长度列表
            feat_dim: 特征维度
            batch_size: 批次大小
            patch_size: patch大小
            stride: 步长
            save_path: 保存路径
        Returns:
            list: 性能数据列表
        """
        results = []
        
        print(f"\n{'='*60}")
        print(f"开始效率分析实验")
        print(f"{'='*60}")
        print(f"特征维度: {feat_dim}")
        print(f"批次大小: {batch_size}")
        print(f"Patch大小: {patch_size}, 步长: {stride}")
        print(f"{'='*60}\n")
        
        for seq_len in tqdm(seq_lengths, desc="测试序列长度"):
            # 生成随机输入数据
            input_tensor = torch.randn(batch_size, seq_len, feat_dim)
            mark_tensor = None  # 可以根据需要添加时间标记
            
            try:
                result = self.profile_single_run(input_tensor, mark_tensor)
                result['seq_length'] = seq_len
                result['num_patches'] = (seq_len - patch_size) // stride + 2  # +1 padding + 1 可能的prompt
                
                print(f"\n序列长度: {seq_len:6d} | "
                      f"Patch数量: {result['num_patches']:4d} | "
                      f"显存: {result['peak_memory_mb']:8.2f} MB | "
                      f"推理时间: {result['inference_time_s']*1000:8.2f} ms")
                
                results.append(result)
                
            except RuntimeError as e:
                print(f"\n❌ 序列长度 {seq_len} 失败: {str(e)}")
                results.append({
                    'seq_length': seq_len,
                    'peak_memory_mb': -1,
                    'inference_time_s': -1,
                    'error': str(e)
                })
        
        # 保存结果
        if save_path:
            self.save_results(results, save_path)
        
        return results
    
    @staticmethod
    def save_results(results, filepath):
        """保存结果到CSV文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'seq_length', 'num_patches', 'peak_memory_mb', 
                'inference_time_s', 'output_shape', 'error'
            ])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✓ 结果已保存到: {filepath}")
    
    @staticmethod
    def compare_models(model_results_dict, save_path=None):
        """
        对比多个模型的性能
        Args:
            model_results_dict: {模型名称: 结果列表}
            save_path: 保存路径
        """
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 收集所有序列长度
        all_seq_lengths = set()
        for results in model_results_dict.values():
            for r in results:
                if 'seq_length' in r:
                    all_seq_lengths.add(r['seq_length'])
        all_seq_lengths = sorted(list(all_seq_lengths))
        
        # 写入CSV
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 表头
            header = ['seq_length']
            for model_name in model_results_dict.keys():
                header.extend([f'{model_name}_memory_mb', f'{model_name}_time_s'])
            writer.writerow(header)
            
            # 数据行
            for seq_len in all_seq_lengths:
                row = [seq_len]
                for model_name, results in model_results_dict.items():
                    # 查找对应序列长度的结果
                    result = next((r for r in results if r.get('seq_length') == seq_len), None)
                    if result:
                        row.append(result.get('peak_memory_mb', -1))
                        row.append(result.get('inference_time_s', -1))
                    else:
                        row.extend([-1, -1])
                writer.writerow(row)
        
        print(f"\n✓ 对比结果已保存到: {save_path}")
