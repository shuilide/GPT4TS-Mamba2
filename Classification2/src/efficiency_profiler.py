"""
效率与复杂度分析工具
用于测量模型在不同序列长度下的显存占用、计算时间、吞吐量和 FLOPs
"""

import torch
import time
import csv
import os
from tqdm import tqdm

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️  thop 未安装，FLOPs 计算将不可用。请运行: pip install thop")


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
    
    def get_model_complexity(self, input_tensor, mark_tensor=None):
        """
        计算模型的 FLOPs 和参数量
        Args:
            input_tensor: 输入张量
            mark_tensor: 标记张量（可选）
        Returns:
            dict: 包含 FLOPs 和参数量信息
        """
        if not THOP_AVAILABLE:
            return {
                'flops': -1,
                'params': -1,
                'flops_str': 'N/A',
                'params_str': 'N/A'
            }
        
        self.model.eval()
        
        # 准备输入 (gpt4ts 需要 x_mark_enc 参数，即使为 None)
        if mark_tensor is None:
            input_args = (input_tensor.to(self.device), None)
        else:
            input_args = (input_tensor.to(self.device), mark_tensor.to(self.device))
        
        # 使用 thop 计算 FLOPs 和参数量
        with torch.no_grad():
            flops, params = profile(self.model, inputs=input_args, verbose=False)
        
        flops_str = clever_format(flops)
        params_str = clever_format(params)
        
        return {
            'flops': flops,
            'params': params,
            'flops_str': flops_str,
            'params_str': params_str
        }
    
    def profile_throughput(self, input_tensor, mark_tensor=None, batch_size=1, duration=5.0):
        """
        测量模型吞吐量（每秒处理的样本数）
        Args:
            input_tensor: 输入张量
            mark_tensor: 标记张量（可选）
            batch_size: 批次大小
            duration: 测试持续时间（秒）
        Returns:
            float: 吞吐量（samples/sec）
        """
        self.model.eval()
        
        # 预热
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device), 
                          mark_tensor.to(self.device) if mark_tensor is not None else None)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        
        # 正式测量
        start_time = time.time()
        num_batches = 0
        
        with torch.no_grad():
            while time.time() - start_time < duration:
                _ = self.model(input_tensor.to(self.device),
                              mark_tensor.to(self.device) if mark_tensor is not None else None)
                num_batches += 1
        
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        
        elapsed_time = time.time() - start_time
        throughput = (num_batches * batch_size) / elapsed_time
        
        return throughput
    
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
                                  patch_size=8, stride=8, save_path=None,
                                  measure_throughput=True, measure_flops=True):
        """
        测试不同序列长度下的性能
        Args:
            seq_lengths: 序列长度列表
            feat_dim: 特征维度
            batch_size: 批次大小
            patch_size: patch大小
            stride: 步长
            save_path: 保存路径
            measure_throughput: 是否测量吞吐量
            measure_flops: 是否测量 FLOPs
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
        if measure_throughput:
            print(f"吞吐量测试: 启用 (duration=5s)")
        if measure_flops:
            print(f"FLOPs测试: {'启用' if THOP_AVAILABLE else '禁用 (thop未安装)'}")
        print(f"{'='*60}\n")
        
        for seq_len in tqdm(seq_lengths, desc="测试序列长度"):
            # 生成随机输入数据
            input_tensor = torch.randn(batch_size, seq_len, feat_dim)
            mark_tensor = None  # 可以根据需要添加时间标记
            
            try:
                result = self.profile_single_run(input_tensor, mark_tensor)
                result['seq_length'] = seq_len
                result['num_patches'] = (seq_len - patch_size) // stride + 2  # +1 padding + 1 可能的prompt
                
                # 测量吞吐量
                if measure_throughput:
                    throughput = self.profile_throughput(input_tensor, mark_tensor, batch_size, duration=5.0)
                    result['throughput'] = throughput
                
                # 测量 FLOPs 和参数量（只在第一个序列长度测量一次）
                if measure_flops and seq_len == seq_lengths[0]:
                    complexity = self.get_model_complexity(input_tensor, mark_tensor)
                    result['flops'] = complexity['flops']
                    result['params'] = complexity['params']
                    result['flops_str'] = complexity['flops_str']
                    result['params_str'] = complexity['params_str']
                    print(f"\n 模型复杂度: FLOPs={result['flops_str']}, Params={result['params_str']}")
                elif measure_flops and seq_len != seq_lengths[0]:
                    # 复用第一个序列长度的 FLOPs 结果（模型结构不变）
                    result['flops'] = results[0].get('flops', -1)
                    result['params'] = results[0].get('params', -1)
                    result['flops_str'] = results[0].get('flops_str', 'N/A')
                    result['params_str'] = results[0].get('params_str', 'N/A')
                
                # 打印结果
                time_ms = result['inference_time_s'] * 1000
                throughput_str = f"{result['throughput']:.2f}" if measure_throughput and 'throughput' in result else "N/A"
                print(f"\n序列长度: {seq_len:6d} | "
                      f"Patch数量: {result['num_patches']:4d} | "
                      f"显存: {result['peak_memory_mb']:8.2f} MB | "
                      f"推理时间: {time_ms:8.2f} ms | "
                      f"吞吐量: {throughput_str:8s} samples/s")
                
                results.append(result)
                
            except RuntimeError as e:
                print(f"\n❌ 序列长度 {seq_len} 失败: {str(e)}")
                results.append({
                    'seq_length': seq_len,
                    'peak_memory_mb': -1,
                    'inference_time_s': -1,
                    'throughput': -1,
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
        
        # 检查是否有新字段
        has_throughput = any('throughput' in r for r in results)
        has_flops = any('flops' in r for r in results)
        
        fieldnames = ['seq_length', 'num_patches', 'peak_memory_mb', 
                     'inference_time_s', 'output_shape', 'error']
        
        if has_throughput:
            fieldnames.insert(4, 'throughput')
        if has_flops:
            fieldnames.extend(['flops', 'params', 'flops_str', 'params_str'])
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
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
        
        # 检查是否有新字段
        has_throughput = any(
            any('throughput' in r for r in results) 
            for results in model_results_dict.values()
        )
        has_flops = any(
            any('flops' in r for r in results) 
            for results in model_results_dict.values()
        )
        
        # 写入CSV
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 表头
            header = ['seq_length']
            for model_name in model_results_dict.keys():
                header.extend([
                    f'{model_name}_memory_mb', 
                    f'{model_name}_time_s',
                    f'{model_name}_throughput' if has_throughput else None
                ])
            if has_flops:
                # FLOPs 和 Params 只在表头添加一次
                header.extend(['flops', 'params', 'flops_str', 'params_str'])
            
            # 移除 None
            header = [h for h in header if h is not None]
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
                        if has_throughput:
                            row.append(result.get('throughput', -1))
                    else:
                        row.extend([-1, -1] + ([-1] if has_throughput else []))
                
                # FLOPs 和 Params（只添加一次）
                if has_flops and seq_len == all_seq_lengths[0]:
                    # 从第一个模型获取 FLOPs 信息
                    first_model = list(model_results_dict.values())[0]
                    first_result = first_model[0]
                    row.extend([
                        first_result.get('flops', -1),
                        first_result.get('params', -1),
                        first_result.get('flops_str', 'N/A'),
                        first_result.get('params_str', 'N/A')
                    ])
                
                writer.writerow(row)
        
        print(f"\n✓ 对比结果已保存到: {save_path}")
