#!/usr/bin/env python3
"""
Layer Distribution Analysis Tool
专门分析某个层的tensor分布，支持attention和linear层的q,k,v,output和input,weight,output分析
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm
import signal
import time
warnings.filterwarnings('ignore')

# 设置matplotlib后端
plt.switch_backend('Agg')

class LayerDistributionAnalyzer:
    def __init__(self, tensor_dir: str, output_dir: str):
        self.tensor_dir = Path(tensor_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的量化类型
        self.quant_types = ['bf16', 'mxfp8', 'mxfp4', 'hifp8']
        
        # 支持的tensor类型
        self.attention_tensors = ['query', 'key', 'value', 'output', 'attention_weights', 'weights']
        self.linear_tensors = ['input', 'weight', 'output', 'bias', 'hidden']
        
    def parse_filename(self, filename: str) -> Optional[Dict]:
        """解析文件名，提取层信息"""
        try:
            # 文件名格式: YYYYMMDD_HHMMSS_XXXX_iterXXX_layer_type_LX_operation_phase_component_quant_type_rankXX_sampleXXX_groupXXX_tensor_name.pt
            # 或者: YYYYMMDD_HHMMSS_XXXX_iterXXX_layer_type_operation_quant_type_rankXX_sampleXXX_groupXXX_tensor_name.pt
            
            parts = filename.split('_')
            if len(parts) < 8:
                return None
            
            # 查找量化类型
            quant_type = None
            for qtype in self.quant_types:
                if qtype in parts:
                    quant_type = qtype
                    break
            
            if not quant_type:
                return None
            
            # 查找层号
            layer_match = re.search(r'L(\d+)', filename)
            layer = int(layer_match.group(1)) if layer_match else None
            
            # 查找sample
            sample_match = re.search(r'sample(\d+)', filename)
            sample = int(sample_match.group(1)) if sample_match else None
            
            # 查找层类型
            layer_type = None
            if 'attention' in filename:
                layer_type = 'attention'
            elif 'linear' in filename:
                layer_type = 'linear'
            
            # 查找操作类型
            operation = None
            if 'forward' in filename:
                operation = 'forward'
            elif 'backward' in filename:
                operation = 'backward'
            
            # 查找tensor名称
            tensor_name = parts[-1].replace('.pt', '')
            
            return {
                'quant_type': quant_type,
                'layer': layer,
                'sample': sample,
                'layer_type': layer_type,
                'operation': operation,
                'tensor_name': tensor_name,
                'filename': filename
            }
        except Exception as e:
            print(f"Failed to parse filename: {filename}, error: {e}")
            return None
    
    def load_tensor_data_with_timeout(self, file_info: Dict, timeout: int = 60) -> Optional[np.ndarray]:
        """带超时的tensor数据加载"""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Loading {file_info['filename']} timed out after {timeout} seconds")
        
        # 检查文件大小，大文件使用更长的超时时间
        file_path = self.tensor_dir / file_info['quant_type'] / file_info['filename']
        if file_path.exists():
            file_size = file_path.stat().st_size
            if file_size > 500 * 1024 * 1024:  # 500MB
                timeout = 120  # 大文件使用2分钟超时
                print(f"    Using extended timeout ({timeout}s) for large file")
        
        # 设置超时信号
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            result = self.load_tensor_data(file_info)
            signal.alarm(0)  # 取消超时
            return result
        except TimeoutError as e:
            print(f"    ⚠ Timeout: {e}")
            signal.alarm(0)
            return None
        except Exception as e:
            print(f"    ✗ Error: {e}")
            signal.alarm(0)
            return None
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def load_large_tensor_data(self, file_path: Path, filename: str) -> Optional[np.ndarray]:
        """内存高效地加载大tensor文件"""
        try:
            print(f"    Loading large tensor {filename} with memory-efficient method...")
            
            # 使用mmap模式加载，减少内存使用
            try:
                # 尝试使用weights_only=True来减少内存使用
                data = torch.load(file_path, map_location='cpu', weights_only=True)
            except Exception:
                # 如果失败，尝试常规加载
                data = torch.load(file_path, map_location='cpu', weights_only=False)
            
            if isinstance(data, torch.Tensor):
                # 对于大tensor，直接进行采样
                if data.numel() > 1_000_000:  # 1M elements
                    print(f"    Sampling large tensor ({data.numel()} elements)...")
                    # 使用更激进的采样策略
                    flat_data = data.flatten()
                    sample_size = min(500_000, len(flat_data))  # 最多采样50万个元素
                    indices = np.random.choice(len(flat_data), sample_size, replace=False)
                    sampled_data = flat_data[indices].numpy()
                    print(f"    ✓ Sampled to {sampled_data.shape} tensor")
                    return sampled_data
                else:
                    return data.numpy()
            elif isinstance(data, dict) and 'tensor' in data:
                tensor = data['tensor']
                if isinstance(tensor, torch.Tensor):
                    if tensor.numel() > 1_000_000:
                        print(f"    Sampling large tensor ({tensor.numel()} elements)...")
                        flat_data = tensor.flatten()
                        sample_size = min(500_000, len(flat_data))
                        indices = np.random.choice(len(flat_data), sample_size, replace=False)
                        sampled_data = flat_data[indices].numpy()
                        print(f"    ✓ Sampled to {sampled_data.shape} tensor")
                        return sampled_data
                    else:
                        return tensor.numpy()
            
            return None
        except Exception as e:
            print(f"    ✗ Failed to load large file {filename}: {e}")
            return None
    
    def load_tensor_data(self, file_info: Dict) -> Optional[np.ndarray]:
        """加载tensor数据"""
        try:
            file_path = self.tensor_dir / file_info['quant_type'] / file_info['filename']
            
            if not file_path.exists():
                return None
            
            if file_path.stat().st_size == 0:
                return None
            
            # 检查文件大小，如果太大则使用特殊处理
            file_size = file_path.stat().st_size
            if file_size > 500 * 1024 * 1024:  # 500MB
                print(f"Warning: Large file {file_info['filename']} ({file_size / 1024 / 1024:.1f}MB), using memory-efficient loading...")
                return self.load_large_tensor_data(file_path, file_info['filename'])
            
            # 尝试加载tensor
            try:
                data = torch.load(file_path, map_location='cpu', weights_only=False)
            except Exception as e1:
                try:
                    data = torch.load(file_path, map_location='cpu', weights_only=True)
                except Exception as e2:
                    print(f"Warning: Failed to load {file_info['filename']}: {e2}")
                    return None
            
            # 处理不同的数据格式
            if isinstance(data, torch.Tensor):
                # 检查tensor大小，如果太大则采样
                if data.numel() > 10_000_000:  # 10M elements
                    print(f"Warning: Large tensor {file_info['filename']} ({data.numel()} elements), sampling...")
                    # 随机采样
                    flat_data = data.flatten()
                    sample_size = min(1_000_000, len(flat_data))  # 最多采样1M个元素
                    indices = np.random.choice(len(flat_data), sample_size, replace=False)
                    return flat_data[indices].numpy()
                return data.numpy()
            elif isinstance(data, dict):
                if 'tensor' in data:
                    if isinstance(data['tensor'], torch.Tensor):
                        tensor = data['tensor']
                        # 检查tensor大小
                        if tensor.numel() > 10_000_000:
                            print(f"Warning: Large tensor {file_info['filename']} ({tensor.numel()} elements), sampling...")
                            flat_data = tensor.flatten()
                            sample_size = min(1_000_000, len(flat_data))
                            indices = np.random.choice(len(flat_data), sample_size, replace=False)
                            return flat_data[indices].numpy()
                        return tensor.numpy()
                elif 'tensor_info' in data:
                    # 这是旧的格式，尝试从tensor_info中获取数据
                    return None
            
            return None
        except Exception as e:
            print(f"Warning: Error loading {file_info['filename']}: {e}")
            return None
    
    def find_tensor_files(self, layer: int, sample: int, layer_type: str) -> Dict[str, List[Dict]]:
        """查找指定层和样本的tensor文件"""
        found_files = {
            'attention': {tensor: [] for tensor in self.attention_tensors},
            'linear': {tensor: [] for tensor in self.linear_tensors}
        }
        
        # 收集所有需要检查的文件
        all_files = []
        for quant_type in self.quant_types:
            quant_dir = self.tensor_dir / quant_type
            if quant_dir.exists():
                all_files.extend(list(quant_dir.glob('*.pt')))
        
        # 使用tqdm显示文件扫描进度
        print(f"Scanning {len(all_files)} tensor files for Layer {layer}, Sample {sample}, {layer_type}...")
        pbar = tqdm(total=len(all_files), desc="Scanning files", unit="files")
        
        for file_path in all_files:
            file_info = self.parse_filename(file_path.name)
            if file_info:
                # 检查是否匹配指定的层、样本和层类型
                if (file_info['layer'] == layer and 
                    file_info['sample'] == sample and 
                    file_info['layer_type'] == layer_type):
                    
                    tensor_name = file_info['tensor_name']
                    if tensor_name in found_files[layer_type]:
                        found_files[layer_type][tensor_name].append(file_info)
            
            pbar.update(1)
        
        pbar.close()
        
        # 显示找到的文件统计
        total_found = sum(len(files) for files in found_files[layer_type].values())
        print(f"Found {total_found} matching files:")
        for tensor_type, files in found_files[layer_type].items():
            if files:
                print(f"  - {tensor_type}: {len(files)} files")
        
        return found_files
    
    def plot_tensor_distribution(self, tensor_data: np.ndarray, title: str, ax, color: str = 'blue'):
        """绘制tensor分布图"""
        if tensor_data is None or len(tensor_data) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold')
            return
        
        # 展平数据
        flat_data = tensor_data.flatten()
        
        # 移除NaN和Inf值
        flat_data = flat_data[~np.isnan(flat_data)]
        flat_data = flat_data[~np.isinf(flat_data)]
        
        if len(flat_data) == 0:
            ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold')
            return
        
        # 如果数据点太多，进行采样
        if len(flat_data) > 10000:
            flat_data = np.random.choice(flat_data, 10000, replace=False)
        
        # 绘制直方图
        ax.hist(flat_data, bins=50, alpha=0.7, color=color, density=True, edgecolor='black', linewidth=0.5)
        
        # 添加统计信息
        mean_val = np.mean(flat_data)
        std_val = np.std(flat_data)
        min_val = np.min(flat_data)
        max_val = np.max(flat_data)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1, label=f'Mean+Std: {mean_val + std_val:.3f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1, label=f'Mean-Std: {mean_val - std_val:.3f}')
        
        ax.set_title(f'{title}\nMean: {mean_val:.3f}, Std: {std_val:.3f}', fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def plot_layer_analysis(self, layer: int, sample: int, layer_type: str):
        """绘制指定层的分析图"""
        print(f"Analyzing Layer {layer}, Sample {sample}, Type: {layer_type}")
        
        # 查找tensor文件
        found_files = self.find_tensor_files(layer, sample, layer_type)
        
        # 确定要绘制的tensor类型
        if layer_type == 'attention':
            tensor_types = self.attention_tensors
        else:
            tensor_types = self.linear_tensors
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Layer {layer} - Sample {sample} - {layer_type.title()} Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 颜色映射
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # 使用tqdm显示绘图进度
        print("Loading tensor data and generating plots...")
        pbar = tqdm(total=min(len(tensor_types), 6), desc="Processing tensor types", unit="types")
        
        plot_idx = 0
        for i, tensor_type in enumerate(tensor_types):
            if plot_idx >= 6:  # 最多6个子图
                break
                
            row = plot_idx // 3
            col = plot_idx % 3
            ax = axes[row, col]
            
            # 获取该tensor类型的所有文件
            tensor_files = found_files[layer_type][tensor_type]
            
            if not tensor_files:
                ax.text(0.5, 0.5, f'No {tensor_type} data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{tensor_type.title()}', fontweight='bold')
                plot_idx += 1
                pbar.update(1)
                continue
            
            # 合并所有相同tensor类型的数据
            all_data = []
            print(f"  Loading {len(tensor_files)} {tensor_type} files...")
            for j, file_info in enumerate(tensor_files):
                print(f"    Loading {file_info['filename']} ({j+1}/{len(tensor_files)})...")
                data = self.load_tensor_data_with_timeout(file_info, timeout=30)
                if data is not None:
                    all_data.append(data)
                    print(f"    ✓ Loaded {data.shape} tensor")
                else:
                    print(f"    ✗ Failed to load")
            
            if not all_data:
                ax.text(0.5, 0.5, f'No valid {tensor_type} data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{tensor_type.title()}', fontweight='bold')
                plot_idx += 1
                pbar.update(1)
                continue
            
            # 合并数据
            combined_data = np.concatenate([data.flatten() for data in all_data])
            
            # 绘制分布
            color = colors[i % len(colors)]
            self.plot_tensor_distribution(combined_data, f'{tensor_type.title()}', ax, color)
            
            plot_idx += 1
            pbar.update(1)
        
        pbar.close()
        
        # 隐藏未使用的子图
        for i in range(plot_idx, 6):
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)
        
        print("Saving plot...")
        plt.tight_layout()
        
        # 保存图片
        output_filename = f'layer_{layer}_sample_{sample}_{layer_type}_analysis.png'
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Layer analysis saved: {output_path}")
        return output_path
    
    def plot_quantization_comparison(self, layer: int, sample: int, layer_type: str, tensor_type: str):
        """绘制不同量化类型的对比图"""
        print(f"Comparing quantization types for Layer {layer}, Sample {sample}, {layer_type} {tensor_type}")
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Quantization Comparison - Layer {layer} Sample {sample} {layer_type.title()} {tensor_type.title()}', 
                    fontsize=16, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # 使用tqdm显示量化对比进度
        print("Loading data for quantization comparison...")
        pbar = tqdm(total=len(self.quant_types), desc="Processing quantization types", unit="types")
        
        for i, quant_type in enumerate(self.quant_types):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # 查找该量化类型的文件
            quant_dir = self.tensor_dir / quant_type
            if not quant_dir.exists():
                ax.text(0.5, 0.5, f'No {quant_type} data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{quant_type.upper()}', fontweight='bold')
                pbar.update(1)
                continue
            
            # 查找匹配的文件
            matching_files = []
            for file_path in quant_dir.glob('*.pt'):
                file_info = self.parse_filename(file_path.name)
                if (file_info and 
                    file_info['layer'] == layer and 
                    file_info['sample'] == sample and 
                    file_info['layer_type'] == layer_type and
                    file_info['tensor_name'] == tensor_type):
                    matching_files.append(file_info)
            
            if not matching_files:
                ax.text(0.5, 0.5, f'No {quant_type} {tensor_type} data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{quant_type.upper()}', fontweight='bold')
                pbar.update(1)
                continue
            
            # 加载并合并数据
            all_data = []
            print(f"  Loading {len(matching_files)} {quant_type} {tensor_type} files...")
            for j, file_info in enumerate(matching_files):
                print(f"    Loading {file_info['filename']} ({j+1}/{len(matching_files)})...")
                data = self.load_tensor_data_with_timeout(file_info, timeout=30)
                if data is not None:
                    all_data.append(data)
                    print(f"    ✓ Loaded {data.shape} tensor")
                else:
                    print(f"    ✗ Failed to load")
            
            if not all_data:
                ax.text(0.5, 0.5, f'No valid {quant_type} data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{quant_type.upper()}', fontweight='bold')
                pbar.update(1)
                continue
            
            combined_data = np.concatenate([data.flatten() for data in all_data])
            color = colors[i % len(colors)]
            self.plot_tensor_distribution(combined_data, f'{quant_type.upper()}', ax, color)
            
            pbar.update(1)
        
        pbar.close()
        
        print("Saving quantization comparison plot...")
        plt.tight_layout()
        
        # 保存图片
        output_filename = f'quantization_comparison_layer_{layer}_sample_{sample}_{layer_type}_{tensor_type}.png'
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Quantization comparison saved: {output_path}")
        return output_path
    
    def generate_statistics_report(self, layer: int, sample: int, layer_type: str):
        """生成统计报告"""
        print(f"Generating statistics report for Layer {layer}, Sample {sample}, {layer_type}")
        
        found_files = self.find_tensor_files(layer, sample, layer_type)
        
        # 确定tensor类型
        if layer_type == 'attention':
            tensor_types = self.attention_tensors
        else:
            tensor_types = self.linear_tensors
        
        report_path = self.output_dir / f'statistics_layer_{layer}_sample_{sample}_{layer_type}.txt'
        
        # 使用tqdm显示统计报告生成进度
        print("Processing tensor data for statistics...")
        pbar = tqdm(total=len(tensor_types), desc="Processing tensor types", unit="types")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Layer {layer} Sample {sample} {layer_type.title()} Statistics Report\n")
            f.write("=" * 80 + "\n\n")
            
            for tensor_type in tensor_types:
                f.write(f"{tensor_type.upper()} Statistics:\n")
                f.write("-" * 40 + "\n")
                
                tensor_files = found_files[layer_type][tensor_type]
                if not tensor_files:
                    f.write("No data found\n\n")
                    pbar.update(1)
                    continue
                
                # 收集所有数据
                all_data = []
                print(f"  Loading {len(tensor_files)} {tensor_type} files for statistics...")
                for j, file_info in enumerate(tensor_files):
                    print(f"    Loading {file_info['filename']} ({j+1}/{len(tensor_files)})...")
                    data = self.load_tensor_data_with_timeout(file_info, timeout=30)
                    if data is not None:
                        all_data.append(data)
                        print(f"    ✓ Loaded {data.shape} tensor")
                    else:
                        print(f"    ✗ Failed to load")
                
                if not all_data:
                    f.write("No valid data found\n\n")
                    pbar.update(1)
                    continue
                
                # 合并数据
                combined_data = np.concatenate([data.flatten() for data in all_data])
                
                # 移除无效值
                valid_data = combined_data[~np.isnan(combined_data)]
                valid_data = valid_data[~np.isinf(valid_data)]
                
                if len(valid_data) == 0:
                    f.write("No valid data after filtering\n\n")
                    pbar.update(1)
                    continue
                
                # 计算统计信息
                f.write(f"File count: {len(tensor_files)}\n")
                f.write(f"Total values: {len(valid_data):,}\n")
                f.write(f"Mean: {np.mean(valid_data):.6f}\n")
                f.write(f"Std Dev: {np.std(valid_data):.6f}\n")
                f.write(f"Min: {np.min(valid_data):.6f}\n")
                f.write(f"Max: {np.max(valid_data):.6f}\n")
                f.write(f"Median: {np.median(valid_data):.6f}\n")
                f.write(f"Q25: {np.percentile(valid_data, 25):.6f}\n")
                f.write(f"Q75: {np.percentile(valid_data, 75):.6f}\n")
                f.write(f"Q95: {np.percentile(valid_data, 95):.6f}\n")
                f.write(f"Q99: {np.percentile(valid_data, 99):.6f}\n\n")
                
                pbar.update(1)
        
        pbar.close()
        
        print(f"Statistics report saved: {report_path}")
        return report_path

def main():
    parser = argparse.ArgumentParser(description='Layer Distribution Analysis Tool')
    parser.add_argument('--tensor_dir', type=str, default='./enhanced_tensor_logs',
                       help='Tensor file directory')
    parser.add_argument('--output_dir', type=str, default='./layer_analysis_output',
                       help='Output directory')
    parser.add_argument('--layer', type=int, required=True,
                       help='Layer number (e.g., 1, 2, 3, ...)')
    parser.add_argument('--sample', type=int, required=True,
                       help='Sample number (e.g., 0, 1, 2)')
    parser.add_argument('--layer_type', type=str, choices=['attention', 'linear'], required=True,
                       help='Layer type (attention or linear)')
    parser.add_argument('--tensor_type', type=str, default=None,
                       help='Specific tensor type for quantization comparison (e.g., query, key, value, input, weight, output)')
    parser.add_argument('--quantization_comparison', action='store_true',
                       help='Generate quantization comparison plot')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = LayerDistributionAnalyzer(args.tensor_dir, args.output_dir)
    
    print("=" * 60)
    print("Layer Distribution Analysis Tool")
    print("=" * 60)
    print(f"Tensor directory: {args.tensor_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Layer: {args.layer}")
    print(f"Sample: {args.sample}")
    print(f"Layer type: {args.layer_type}")
    print("=" * 60)
    
    try:
        # 确定需要执行的任务
        tasks = ["Layer analysis"]
        if args.quantization_comparison and args.tensor_type:
            tasks.append("Quantization comparison")
        tasks.append("Statistics report")
        
        # 使用tqdm显示总体进度
        print("Starting analysis tasks...")
        overall_pbar = tqdm(total=len(tasks), desc="Overall progress", unit="tasks")
        
        # 生成层分析图
        layer_analysis_path = analyzer.plot_layer_analysis(args.layer, args.sample, args.layer_type)
        overall_pbar.update(1)
        
        # 生成量化对比图（如果指定了tensor类型）
        quant_comparison_path = None
        if args.quantization_comparison and args.tensor_type:
            quant_comparison_path = analyzer.plot_quantization_comparison(
                args.layer, args.sample, args.layer_type, args.tensor_type)
            overall_pbar.update(1)
        
        # 生成统计报告
        stats_report_path = analyzer.generate_statistics_report(args.layer, args.sample, args.layer_type)
        overall_pbar.update(1)
        
        overall_pbar.close()
        
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print("=" * 60)
        print(f"Layer analysis: {layer_analysis_path}")
        if quant_comparison_path:
            print(f"Quantization comparison: {quant_comparison_path}")
        print(f"Statistics report: {stats_report_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
