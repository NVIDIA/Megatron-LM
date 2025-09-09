#!/usr/bin/env python3
"""
Tensor Visualization Script
Supports one-click visualization of saved tensor data, generating various analysis charts
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import pandas as pd
from datetime import datetime

# 添加项目路径
sys.path.append('/data/charles/Megatron-LM')

# Set matplotlib font
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TensorVisualizer:
    """Tensor Visualizer"""
    
    def __init__(self, tensor_dir: str = "./enhanced_tensor_logs", output_dir: str = "./draw"):
        """
        Initialize visualizer
        
        Args:
            tensor_dir: Tensor file directory
            output_dir: Output image directory
        """
        self.tensor_dir = Path(tensor_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.subdirs = {
            'heatmaps': self.output_dir / 'heatmaps',
            'distributions': self.output_dir / 'distributions', 
            'comparisons': self.output_dir / 'comparisons',
            'statistics': self.output_dir / 'statistics',
            'attention_maps': self.output_dir / 'attention_maps'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"[TensorVisualizer] Initialization complete")
        print(f"  Tensor directory: {self.tensor_dir}")
        print(f"  Output directory: {self.output_dir}")
    
    def load_tensor_files(self) -> List[Dict]:
        """Load all tensor files"""
        tensor_files = glob.glob(str(self.tensor_dir / "*.pt"))
        print(f"[TensorVisualizer] Found {len(tensor_files)} tensor files")
        
        loaded_tensors = []
        for file_path in tensor_files:
            try:
                data = torch.load(file_path, map_location='cpu')
                tensor_info = {
                    'file_path': file_path,
                    'filename': Path(file_path).name,
                    'tensor': data['tensor'],
                    'tensor_info': data['tensor_info'],
                    'metadata': data['metadata']
                }
                loaded_tensors.append(tensor_info)
            except Exception as e:
                print(f"[TensorVisualizer] Failed to load file {file_path}: {e}")
        
        return loaded_tensors
    
    def group_tensors_by_type(self, tensors: List[Dict]) -> Dict[str, List[Dict]]:
        """Group tensors by type"""
        groups = {
            'attention_forward': [],
            'attention_backward': [],
            'linear_forward': [],
            'linear_backward': []
        }
        
        for tensor_info in tensors:
            metadata = tensor_info['metadata']
            layer_type = metadata['layer_type']
            operation = metadata['operation']
            key = f"{layer_type}_{operation}"
            
            if key in groups:
                groups[key].append(tensor_info)
        
        return groups
    
    def plot_tensor_distribution(self, tensor_info: Dict, save_path: str):
        """Plot tensor distribution chart"""
        tensor = tensor_info['tensor'].float().numpy().flatten()
        metadata = tensor_info['metadata']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Tensor Distribution Analysis - {tensor_info['filename']}", fontsize=16)
        
        # Histogram
        axes[0, 0].hist(tensor, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Value Distribution Histogram')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(tensor, patch_artist=True, 
                          boxprops=dict(facecolor='lightgreen', alpha=0.7))
        axes[0, 1].set_title('Value Distribution Box Plot')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(tensor, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistical information
        stats_text = f"""
        Statistical Information:
        Shape: {tensor_info['tensor_info']['shape']}
        Data Type: {tensor_info['tensor_info']['dtype']}
        Min Value: {tensor_info['tensor_info']['min']:.4f}
        Max Value: {tensor_info['tensor_info']['max']:.4f}
        Mean: {tensor_info['tensor_info']['mean']:.4f}
        Std Dev: {tensor_info['tensor_info']['std']:.4f}
        
        Layer Information:
        Layer Type: {metadata['layer_type']}
        Operation: {metadata['operation']}
        Quantization Type: {metadata['quant_type']}
        Layer Index: {metadata.get('layer_idx', 'N/A')}
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_tensor_heatmap(self, tensor_info: Dict, save_path: str, max_size: int = 100):
        """绘制tensor热力图"""
        tensor = tensor_info['tensor'].float().numpy()
        metadata = tensor_info['metadata']
        
        # 如果tensor太大，进行采样
        if tensor.size > max_size * max_size:
            # 对于4D tensor (attention)，取第一个batch和head
            if len(tensor.shape) == 4:
                tensor = tensor[0, 0, :max_size, :max_size]
            # 对于2D tensor (linear)，直接采样
            elif len(tensor.shape) == 2:
                tensor = tensor[:max_size, :max_size]
            else:
                # 其他情况，reshape到2D
                tensor = tensor.flatten()[:max_size*max_size].reshape(max_size, max_size)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Tensor热力图 - {tensor_info['filename']}", fontsize=16)
        
        # 原始热力图
        im1 = axes[0].imshow(tensor, cmap='viridis', aspect='auto')
        axes[0].set_title('原始数值热力图')
        axes[0].set_xlabel('列索引')
        axes[0].set_ylabel('行索引')
        plt.colorbar(im1, ax=axes[0])
        
        # 归一化热力图
        tensor_norm = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        im2 = axes[1].imshow(tensor_norm, cmap='plasma', aspect='auto')
        axes[1].set_title('归一化热力图')
        axes[1].set_xlabel('列索引')
        axes[1].set_ylabel('行索引')
        plt.colorbar(im2, ax=axes[1])
        
        # 添加信息文本
        info_text = f"""
        层类型: {metadata['layer_type']}
        操作: {metadata['operation']}
        量化类型: {metadata['quant_type']}
        显示大小: {tensor.shape}
        """
        
        fig.text(0.02, 0.02, info_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_quantization_comparison(self, tensors: List[Dict], save_path: str):
        """绘制不同量化类型的对比图"""
        if len(tensors) < 2:
            print("[TensorVisualizer] 量化类型对比需要至少2个tensor")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('不同量化类型对比分析', fontsize=16)
        
        # 按量化类型分组
        quant_groups = {}
        for tensor_info in tensors:
            quant_type = tensor_info['metadata']['quant_type']
            if quant_type not in quant_groups:
                quant_groups[quant_type] = []
            quant_groups[quant_type].append(tensor_info)
        
        # 分布对比
        for i, (quant_type, group) in enumerate(quant_groups.items()):
            if i >= 4:  # 最多显示4种量化类型
                break
            
            ax = axes[i//2, i%2]
            for tensor_info in group:
                tensor = tensor_info['tensor'].float().numpy().flatten()
                ax.hist(tensor, bins=30, alpha=0.6, label=tensor_info['filename'][:20])
            
            ax.set_title(f'{quant_type} 分布对比')
            ax.set_xlabel('数值')
            ax.set_ylabel('频次')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_statistics_summary(self, tensors: List[Dict], save_path: str):
        """绘制统计信息汇总图"""
        # 收集统计信息
        stats_data = []
        for tensor_info in tensors:
            metadata = tensor_info['metadata']
            tensor_stats = tensor_info['tensor_info']
            
            stats_data.append({
                'filename': tensor_info['filename'][:30],
                'layer_type': metadata['layer_type'],
                'operation': metadata['operation'],
                'quant_type': metadata['quant_type'],
                'layer_idx': metadata.get('layer_idx', 0),
                'min': tensor_stats['min'],
                'max': tensor_stats['max'],
                'mean': tensor_stats['mean'],
                'std': tensor_stats['std'],
                'shape_0': tensor_stats['shape'][0] if len(tensor_stats['shape']) > 0 else 0,
                'shape_1': tensor_stats['shape'][1] if len(tensor_stats['shape']) > 1 else 0,
            })
        
        df = pd.DataFrame(stats_data)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Tensor统计信息汇总', fontsize=16)
        
        # 数值范围对比
        sns.boxplot(data=df, x='quant_type', y='min', ax=axes[0, 0])
        axes[0, 0].set_title('最小值分布')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        sns.boxplot(data=df, x='quant_type', y='max', ax=axes[0, 1])
        axes[0, 1].set_title('最大值分布')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        sns.boxplot(data=df, x='quant_type', y='std', ax=axes[0, 2])
        axes[0, 2].set_title('标准差分布')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 层类型分布
        layer_counts = df['layer_type'].value_counts()
        axes[1, 0].pie(layer_counts.values, labels=layer_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('层类型分布')
        
        # 操作类型分布
        op_counts = df['operation'].value_counts()
        axes[1, 1].pie(op_counts.values, labels=op_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('操作类型分布')
        
        # 量化类型分布
        quant_counts = df['quant_type'].value_counts()
        axes[1, 2].pie(quant_counts.values, labels=quant_counts.index, autopct='%1.1f%%')
        axes[1, 2].set_title('量化类型分布')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attention_analysis(self, tensors: List[Dict], save_path: str):
        """绘制attention分析图"""
        attention_tensors = [t for t in tensors if t['metadata']['layer_type'] == 'attention']
        
        if not attention_tensors:
            print("[TensorVisualizer] 没有找到attention tensor")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Attention Tensor分析', fontsize=16)
        
        # 按tensor名称分组
        query_tensors = [t for t in attention_tensors if 'query' in t['filename']]
        key_tensors = [t for t in attention_tensors if 'key' in t['filename']]
        value_tensors = [t for t in attention_tensors if 'value' in t['filename']]
        output_tensors = [t for t in attention_tensors if 'output' in t['filename']]
        
        # Query分布
        if query_tensors:
            for tensor_info in query_tensors[:3]:  # 最多显示3个
                tensor = tensor_info['tensor'].float().numpy().flatten()
                axes[0, 0].hist(tensor, bins=30, alpha=0.6, 
                               label=tensor_info['metadata']['quant_type'])
            axes[0, 0].set_title('Query Tensor分布')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Key分布
        if key_tensors:
            for tensor_info in key_tensors[:3]:
                tensor = tensor_info['tensor'].float().numpy().flatten()
                axes[0, 1].hist(tensor, bins=30, alpha=0.6,
                               label=tensor_info['metadata']['quant_type'])
            axes[0, 1].set_title('Key Tensor分布')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Value分布
        if value_tensors:
            for tensor_info in value_tensors[:3]:
                tensor = tensor_info['tensor'].float().numpy().flatten()
                axes[1, 0].hist(tensor, bins=30, alpha=0.6,
                               label=tensor_info['metadata']['quant_type'])
            axes[1, 0].set_title('Value Tensor分布')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Output分布
        if output_tensors:
            for tensor_info in output_tensors[:3]:
                tensor = tensor_info['tensor'].float().numpy().flatten()
                axes[1, 1].hist(tensor, bins=30, alpha=0.6,
                               label=tensor_info['metadata']['quant_type'])
            axes[1, 1].set_title('Output Tensor分布')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualization charts"""
        print("[TensorVisualizer] Starting to generate visualization charts...")
        
        # Load tensor files
        tensors = self.load_tensor_files()
        if not tensors:
            print("[TensorVisualizer] No tensor files found")
            return
        
        # Group by type
        groups = self.group_tensors_by_type(tensors)
        
        # Generate distribution plots
        print("[TensorVisualizer] Generating distribution plots...")
        for i, tensor_info in enumerate(tensors[:10]):  # Limit number to avoid too many files
            save_path = self.subdirs['distributions'] / f"distribution_{i:03d}_{tensor_info['filename'][:20]}.png"
            self.plot_tensor_distribution(tensor_info, str(save_path))
        
        # Generate heatmaps
        print("[TensorVisualizer] Generating heatmaps...")
        for i, tensor_info in enumerate(tensors[:10]):
            save_path = self.subdirs['heatmaps'] / f"heatmap_{i:03d}_{tensor_info['filename'][:20]}.png"
            self.plot_tensor_heatmap(tensor_info, str(save_path))
        
        # Generate quantization type comparison charts
        print("[TensorVisualizer] Generating quantization type comparison charts...")
        for group_name, group_tensors in groups.items():
            if group_tensors:
                save_path = self.subdirs['comparisons'] / f"quant_comparison_{group_name}.png"
                self.plot_quantization_comparison(group_tensors, str(save_path))
        
        # Generate statistical summary charts
        print("[TensorVisualizer] Generating statistical summary charts...")
        save_path = self.subdirs['statistics'] / "statistics_summary.png"
        self.plot_statistics_summary(tensors, str(save_path))
        
        # Generate attention analysis charts
        print("[TensorVisualizer] Generating attention analysis charts...")
        save_path = self.subdirs['attention_maps'] / "attention_analysis.png"
        self.plot_attention_analysis(tensors, str(save_path))
        
        print(f"[TensorVisualizer] Visualization complete! Charts saved to: {self.output_dir}")
        self.print_summary()
    
    def print_summary(self):
        """Print summary of generated files"""
        print("\n=== Generated Visualization Files Summary ===")
        
        for subdir_name, subdir_path in self.subdirs.items():
            files = list(subdir_path.glob("*.png"))
            print(f"{subdir_name}: {len(files)} files")
            for file in files[:3]:  # Show first 3 files
                print(f"  - {file.name}")
            if len(files) > 3:
                print(f"  ... and {len(files) - 3} more files")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Tensor Visualization Tool')
    parser.add_argument('--tensor_dir', type=str, default='./enhanced_tensor_logs',
                       help='Tensor file directory (default: ./enhanced_tensor_logs)')
    parser.add_argument('--output_dir', type=str, default='./draw',
                       help='Output image directory (default: ./draw)')
    parser.add_argument('--max_files', type=int, default=50,
                       help='Maximum number of files to process (default: 50)')
    
    args = parser.parse_args()
    
    print("=== Tensor Visualization Tool ===")
    print(f"Tensor directory: {args.tensor_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Maximum files: {args.max_files}")
    
    # Create visualizer
    visualizer = TensorVisualizer(
        tensor_dir=args.tensor_dir,
        output_dir=args.output_dir
    )
    
    # Generate all visualization charts
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
