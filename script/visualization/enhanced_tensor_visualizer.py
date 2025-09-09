#!/usr/bin/env python3
"""
Enhanced Tensor Visualization Script
Specialized for high-quality visualization of tensor files in enhanced_tensor_logs
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
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

class EnhancedTensorVisualizer:
    """Enhanced Tensor Visualizer"""
    
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
            'distributions': self.output_dir / 'distributions',
            'heatmaps': self.output_dir / 'heatmaps', 
            'comparisons': self.output_dir / 'comparisons',
            'statistics': self.output_dir / 'statistics',
            'attention_analysis': self.output_dir / 'attention_analysis',
            'quantization_analysis': self.output_dir / 'quantization_analysis',
            'layer_analysis': self.output_dir / 'layer_analysis'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"[EnhancedTensorVisualizer] Initialization complete")
        print(f"  Tensor directory: {self.tensor_dir}")
        print(f"  Output directory: {self.output_dir}")
    
    def load_tensor_files(self) -> List[Dict]:
        """Load all tensor files"""
        tensor_files = glob.glob(str(self.tensor_dir / "*.pt"))
        print(f"[EnhancedTensorVisualizer] Found {len(tensor_files)} tensor files")
        
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
                print(f"[EnhancedTensorVisualizer] Failed to load file {file_path}: {e}")
        
        return loaded_tensors
    
    def parse_filename(self, filename: str) -> Dict:
        """Parse filename to extract information"""
        # Format: timestamp_counter_layer_type_L{idx}_operation_phase_component_quant_type_tensor_name.pt
        parts = filename.replace('.pt', '').split('_')
        
        info = {
            'timestamp': parts[0] if len(parts) > 0 else 'unknown',
            'counter': parts[1] if len(parts) > 1 else 'unknown',
            'layer_type': parts[2] if len(parts) > 2 else 'unknown',
            'layer_idx': parts[3] if len(parts) > 3 else 'unknown',
            'operation': parts[4] if len(parts) > 4 else 'unknown',
            'phase': parts[5] if len(parts) > 5 else 'unknown',
            'component': parts[6] if len(parts) > 6 else 'unknown',
            'quant_type': parts[7] if len(parts) > 7 else 'unknown',
            'tensor_name': parts[8] if len(parts) > 8 else 'unknown'
        }
        
        return info
    
    def group_tensors_by_quant_type(self, tensors: List[Dict]) -> Dict[str, List[Dict]]:
        """Group tensors by quantization type"""
        groups = defaultdict(list)
        
        for tensor_info in tensors:
            quant_type = tensor_info['metadata']['quant_type']
            groups[quant_type].append(tensor_info)
        
        return dict(groups)
    
    def group_tensors_by_layer_type(self, tensors: List[Dict]) -> Dict[str, List[Dict]]:
        """Group tensors by layer type"""
        groups = defaultdict(list)
        
        for tensor_info in tensors:
            layer_type = tensor_info['metadata']['layer_type']
            groups[layer_type].append(tensor_info)
        
        return dict(groups)
    
    def plot_quantization_comparison(self, tensors: List[Dict]):
        """Plot quantization type comparison chart"""
        quant_groups = self.group_tensors_by_quant_type(tensors)
        
        if len(quant_groups) < 2:
            print("Insufficient quantization types, skipping comparison chart")
            return
        
        # Select files with same tensor_name for comparison
        tensor_name_groups = defaultdict(list)
        for quant_type, group in quant_groups.items():
            for tensor_info in group:
                tensor_name = tensor_info['metadata']['tensor_name']
                tensor_name_groups[tensor_name].append((quant_type, tensor_info))
        
        # Create comparison chart for each tensor_name
        for tensor_name, quant_tensors in tensor_name_groups.items():
            if len(quant_tensors) < 2:
                continue
                
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{tensor_name} - Quantization Type Comparison Analysis', fontsize=16, fontweight='bold')
            
            # Collect data
            data_by_quant = {}
            for quant_type, tensor_info in quant_tensors:
                tensor_data = tensor_info['tensor'].float().numpy().flatten()
                data_by_quant[quant_type] = tensor_data
            
            # Distribution comparison
            ax1 = axes[0, 0]
            for quant_type, data in data_by_quant.items():
                ax1.hist(data, bins=50, alpha=0.6, label=quant_type, density=True)
            ax1.set_title('Value Distribution Comparison')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot comparison
            ax2 = axes[0, 1]
            box_data = [data_by_quant[qt] for qt in data_by_quant.keys()]
            box_labels = list(data_by_quant.keys())
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            colors = plt.cm.Set3(np.linspace(0, 1, len(box_labels)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax2.set_title('Value Distribution Box Plot Comparison')
            ax2.set_ylabel('Value')
            ax2.grid(True, alpha=0.3)
            
            # 统计信息对比
            ax3 = axes[1, 0]
            stats_data = []
            for quant_type, data in data_by_quant.items():
                stats = {
                    'quant_type': quant_type,
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data)
                }
                stats_data.append(stats)
            
            stats_df = pd.DataFrame(stats_data)
            x = np.arange(len(stats_df))
            width = 0.2
            
            ax3.bar(x - width, stats_df['mean'], width, label='均值', alpha=0.8)
            ax3.bar(x, stats_df['std'], width, label='标准差', alpha=0.8)
            ax3.bar(x + width, stats_df['max'] - stats_df['min'], width, label='范围', alpha=0.8)
            
            ax3.set_xlabel('量化类型')
            ax3.set_ylabel('数值')
            ax3.set_title('统计信息对比')
            ax3.set_xticks(x)
            ax3.set_xticklabels(stats_df['quant_type'])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 相关性分析
            ax4 = axes[1, 1]
            if len(data_by_quant) == 2:
                quant_types = list(data_by_quant.keys())
                data1 = data_by_quant[quant_types[0]]
                data2 = data_by_quant[quant_types[1]]
                
                # 采样以减少计算量
                if len(data1) > 10000:
                    indices = np.random.choice(len(data1), 10000, replace=False)
                    data1 = data1[indices]
                    data2 = data2[indices]
                
                ax4.scatter(data1, data2, alpha=0.5, s=1)
                ax4.set_xlabel(quant_types[0])
                ax4.set_ylabel(quant_types[1])
                ax4.set_title('相关性分析')
                
                # 计算相关系数
                corr = np.corrcoef(data1, data2)[0, 1]
                ax4.text(0.05, 0.95, f'相关系数: {corr:.4f}', 
                        transform=ax4.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                ax4.text(0.5, 0.5, '需要2个量化类型\n进行相关性分析', 
                        ha='center', va='center', transform=ax4.transAxes,
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 1)
            
            plt.tight_layout()
            
            # Save image
            output_path = self.subdirs['quantization_analysis'] / f'{tensor_name}_quantization_comparison.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Quantization comparison chart saved: {output_path}")
    
    def plot_attention_analysis(self, tensors: List[Dict]):
        """Plot attention layer analysis chart"""
        attention_tensors = [t for t in tensors if t['metadata']['layer_type'] == 'attention']
        
        if not attention_tensors:
            print("No attention layer tensors found, skipping attention analysis")
            return
        
        # 按tensor_name分组
        attention_groups = defaultdict(list)
        for tensor_info in attention_tensors:
            tensor_name = tensor_info['metadata']['tensor_name']
            attention_groups[tensor_name].append(tensor_info)
        
        # 为每个attention tensor创建分析图
        for tensor_name, group in attention_groups.items():
            if len(group) < 2:  # 需要多个样本进行分析
                continue
                
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Attention {tensor_name} 分析', fontsize=16, fontweight='bold')
            
            # 收集数据
            all_data = []
            quant_types = []
            for tensor_info in group:
                data = tensor_info['tensor'].float().numpy().flatten()
                all_data.append(data)
                quant_types.append(tensor_info['metadata']['quant_type'])
            
            # 分布分析
            ax1 = axes[0, 0]
            for i, (data, quant_type) in enumerate(zip(all_data, quant_types)):
                ax1.hist(data, bins=50, alpha=0.6, label=quant_type, density=True)
            ax1.set_title(f'{tensor_name} 数值分布')
            ax1.set_xlabel('数值')
            ax1.set_ylabel('密度')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 时间序列分析（如果有多个样本）
            ax2 = axes[0, 1]
            if len(all_data) > 1:
                means = [np.mean(data) for data in all_data]
                stds = [np.std(data) for data in all_data]
                
                x = range(len(means))
                ax2.errorbar(x, means, yerr=stds, marker='o', capsize=5, capthick=2)
                ax2.set_title(f'{tensor_name} 统计信息变化')
                ax2.set_xlabel('样本索引')
                ax2.set_ylabel('均值 ± 标准差')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, '需要多个样本\n进行时间序列分析', 
                        ha='center', va='center', transform=ax2.transAxes,
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
            
            # 热力图（如果是2D tensor）
            ax3 = axes[1, 0]
            if len(group) > 0:
                sample_tensor = group[0]['tensor'].float().numpy()
                if len(sample_tensor.shape) == 2:
                    im = ax3.imshow(sample_tensor, cmap='viridis', aspect='auto')
                    ax3.set_title(f'{tensor_name} 热力图 (样本1)')
                    ax3.set_xlabel('列')
                    ax3.set_ylabel('行')
                    plt.colorbar(im, ax=ax3)
                else:
                    # 如果是高维tensor，取平均值
                    if len(sample_tensor.shape) > 2:
                        sample_tensor = np.mean(sample_tensor, axis=tuple(range(2, len(sample_tensor.shape))))
                    im = ax3.imshow(sample_tensor, cmap='viridis', aspect='auto')
                    ax3.set_title(f'{tensor_name} 热力图 (平均)')
                    ax3.set_xlabel('列')
                    ax3.set_ylabel('行')
                    plt.colorbar(im, ax=ax3)
            
            # 统计信息
            ax4 = axes[1, 1]
            stats_text = f"{tensor_name} 统计信息:\n\n"
            for i, (tensor_info, quant_type) in enumerate(zip(group, quant_types)):
                stats = tensor_info['tensor_info']
                stats_text += f"样本 {i+1} ({quant_type}):\n"
                stats_text += f"  形状: {stats['shape']}\n"
                stats_text += f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]\n"
                stats_text += f"  均值: {stats['mean']:.4f}\n"
                stats_text += f"  标准差: {stats['std']:.4f}\n\n"
            
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            
            plt.tight_layout()
            
            # 保存图片
            output_path = self.subdirs['attention_analysis'] / f'{tensor_name}_analysis.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已保存attention分析图: {output_path}")
    
    def plot_layer_comparison(self, tensors: List[Dict]):
        """绘制层类型对比图"""
        layer_groups = self.group_tensors_by_layer_type(tensors)
        
        if len(layer_groups) < 2:
            print("层类型数量不足，跳过层对比图")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('层类型对比分析', fontsize=16, fontweight='bold')
        
        # 收集数据
        layer_stats = {}
        for layer_type, group in layer_groups.items():
            all_values = []
            for tensor_info in group:
                values = tensor_info['tensor'].float().numpy().flatten()
                all_values.extend(values)
            layer_stats[layer_type] = all_values
        
        # 分布对比
        ax1 = axes[0, 0]
        for layer_type, values in layer_stats.items():
            ax1.hist(values, bins=50, alpha=0.6, label=layer_type, density=True)
        ax1.set_title('层类型数值分布对比')
        ax1.set_xlabel('数值')
        ax1.set_ylabel('密度')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 统计信息对比
        ax2 = axes[0, 1]
        stats_data = []
        for layer_type, values in layer_stats.items():
            stats = {
                'layer_type': layer_type,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            stats_data.append(stats)
        
        stats_df = pd.DataFrame(stats_data)
        x = np.arange(len(stats_df))
        width = 0.2
        
        ax2.bar(x - width, stats_df['mean'], width, label='均值', alpha=0.8)
        ax2.bar(x, stats_df['std'], width, label='标准差', alpha=0.8)
        ax2.bar(x + width, stats_df['max'] - stats_df['min'], width, label='范围', alpha=0.8)
        
        ax2.set_xlabel('层类型')
        ax2.set_ylabel('数值')
        ax2.set_title('层类型统计信息对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(stats_df['layer_type'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 箱线图对比
        ax3 = axes[1, 0]
        box_data = [layer_stats[lt] for lt in layer_stats.keys()]
        box_labels = list(layer_stats.keys())
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(box_labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.set_title('层类型数值分布箱线图对比')
        ax3.set_ylabel('数值')
        ax3.grid(True, alpha=0.3)
        
        # 数量统计
        ax4 = axes[1, 1]
        counts = [len(layer_stats[lt]) for lt in layer_stats.keys()]
        bars = ax4.bar(layer_stats.keys(), counts, alpha=0.8, color=colors)
        ax4.set_title('各层类型tensor数量')
        ax4.set_xlabel('层类型')
        ax4.set_ylabel('tensor数量')
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = self.subdirs['layer_analysis'] / 'layer_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存层对比图: {output_path}")
    
    def generate_summary_report(self, tensors: List[Dict]):
        """生成汇总报告"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Tensor数据汇总报告', fontsize=16, fontweight='bold')
        
        # 统计信息
        total_tensors = len(tensors)
        quant_types = set(t['metadata']['quant_type'] for t in tensors)
        layer_types = set(t['metadata']['layer_type'] for t in tensors)
        operations = set(t['metadata']['operation'] for t in tensors)
        
        # 量化类型分布
        ax1 = axes[0, 0]
        quant_counts = {}
        for tensor_info in tensors:
            quant_type = tensor_info['metadata']['quant_type']
            quant_counts[quant_type] = quant_counts.get(quant_type, 0) + 1
        
        ax1.pie(quant_counts.values(), labels=quant_counts.keys(), autopct='%1.1f%%', startangle=90)
        ax1.set_title('量化类型分布')
        
        # 层类型分布
        ax2 = axes[0, 1]
        layer_counts = {}
        for tensor_info in tensors:
            layer_type = tensor_info['metadata']['layer_type']
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        
        ax2.pie(layer_counts.values(), labels=layer_counts.keys(), autopct='%1.1f%%', startangle=90)
        ax2.set_title('层类型分布')
        
        # 操作类型分布
        ax3 = axes[1, 0]
        op_counts = {}
        for tensor_info in tensors:
            operation = tensor_info['metadata']['operation']
            op_counts[operation] = op_counts.get(operation, 0) + 1
        
        ax3.pie(op_counts.values(), labels=op_counts.keys(), autopct='%1.1f%%', startangle=90)
        ax3.set_title('操作类型分布')
        
        # 汇总信息
        ax4 = axes[1, 1]
        summary_text = f"""
        数据汇总信息:
        
        总tensor数量: {total_tensors}
        
        量化类型: {', '.join(quant_types)}
        
        层类型: {', '.join(layer_types)}
        
        操作类型: {', '.join(operations)}
        
        数据目录: {self.tensor_dir}
        
        生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = self.subdirs['statistics'] / 'summary_report.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存汇总报告: {output_path}")
    
    def run_visualization(self):
        """Run complete visualization workflow"""
        print("[EnhancedTensorVisualizer] Starting visualization workflow...")
        
        # Load tensor files
        tensors = self.load_tensor_files()
        
        if not tensors:
            print("No tensor files found, exiting visualization")
            return
        
        print(f"Successfully loaded {len(tensors)} tensor files")
        
        # Generate various analysis charts
        print("\n1. Generating summary report...")
        self.generate_summary_report(tensors)
        
        print("\n2. Generating quantization type comparison charts...")
        self.plot_quantization_comparison(tensors)
        
        print("\n3. Generating attention analysis charts...")
        self.plot_attention_analysis(tensors)
        
        print("\n4. Generating layer type comparison charts...")
        self.plot_layer_comparison(tensors)
        
        print(f"\n[EnhancedTensorVisualizer] Visualization complete!")
        print(f"All images saved to: {self.output_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced Tensor Visualization Tool')
    parser.add_argument('--tensor_dir', type=str, default='./enhanced_tensor_logs',
                       help='Tensor file directory (default: ./enhanced_tensor_logs)')
    parser.add_argument('--output_dir', type=str, default='./draw',
                       help='Output image directory (default: ./draw)')
    
    args = parser.parse_args()
    
    # Create visualizer and run
    visualizer = EnhancedTensorVisualizer(
        tensor_dir=args.tensor_dir,
        output_dir=args.output_dir
    )
    
    visualizer.run_visualization()


if __name__ == "__main__":
    main()
