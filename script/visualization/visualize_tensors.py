#!/usr/bin/env python3
"""
Tensor可视化脚本
支持一键可视化保存的tensor数据，生成各种分析图表
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

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TensorVisualizer:
    """Tensor可视化器"""
    
    def __init__(self, tensor_dir: str = "./tensor_logs", output_dir: str = "./draw"):
        """
        初始化可视化器
        
        Args:
            tensor_dir: tensor文件目录
            output_dir: 输出图片目录
        """
        self.tensor_dir = Path(tensor_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.subdirs = {
            'heatmaps': self.output_dir / 'heatmaps',
            'distributions': self.output_dir / 'distributions', 
            'comparisons': self.output_dir / 'comparisons',
            'statistics': self.output_dir / 'statistics',
            'attention_maps': self.output_dir / 'attention_maps'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"[TensorVisualizer] 初始化完成")
        print(f"  Tensor目录: {self.tensor_dir}")
        print(f"  输出目录: {self.output_dir}")
    
    def load_tensor_files(self) -> List[Dict]:
        """加载所有tensor文件"""
        tensor_files = glob.glob(str(self.tensor_dir / "*.pt"))
        print(f"[TensorVisualizer] 找到 {len(tensor_files)} 个tensor文件")
        
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
                print(f"[TensorVisualizer] 加载文件失败 {file_path}: {e}")
        
        return loaded_tensors
    
    def group_tensors_by_type(self, tensors: List[Dict]) -> Dict[str, List[Dict]]:
        """按类型分组tensor"""
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
        """绘制tensor分布图"""
        tensor = tensor_info['tensor'].float().numpy().flatten()
        metadata = tensor_info['metadata']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Tensor分布分析 - {tensor_info['filename']}", fontsize=16)
        
        # 直方图
        axes[0, 0].hist(tensor, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('数值分布直方图')
        axes[0, 0].set_xlabel('数值')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 箱线图
        axes[0, 1].boxplot(tensor, patch_artist=True, 
                          boxprops=dict(facecolor='lightgreen', alpha=0.7))
        axes[0, 1].set_title('数值分布箱线图')
        axes[0, 1].set_ylabel('数值')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q图
        from scipy import stats
        stats.probplot(tensor, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q图 (正态分布)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 统计信息
        stats_text = f"""
        统计信息:
        形状: {tensor_info['tensor_info']['shape']}
        数据类型: {tensor_info['tensor_info']['dtype']}
        最小值: {tensor_info['tensor_info']['min']:.4f}
        最大值: {tensor_info['tensor_info']['max']:.4f}
        均值: {tensor_info['tensor_info']['mean']:.4f}
        标准差: {tensor_info['tensor_info']['std']:.4f}
        
        层信息:
        层类型: {metadata['layer_type']}
        操作: {metadata['operation']}
        量化类型: {metadata['quant_type']}
        层索引: {metadata.get('layer_idx', 'N/A')}
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
        """生成所有可视化图表"""
        print("[TensorVisualizer] 开始生成可视化图表...")
        
        # 加载tensor文件
        tensors = self.load_tensor_files()
        if not tensors:
            print("[TensorVisualizer] 没有找到tensor文件")
            return
        
        # 按类型分组
        groups = self.group_tensors_by_type(tensors)
        
        # 生成分布图
        print("[TensorVisualizer] 生成分布图...")
        for i, tensor_info in enumerate(tensors[:10]):  # 限制数量避免过多文件
            save_path = self.subdirs['distributions'] / f"distribution_{i:03d}_{tensor_info['filename'][:20]}.png"
            self.plot_tensor_distribution(tensor_info, str(save_path))
        
        # 生成热力图
        print("[TensorVisualizer] 生成热力图...")
        for i, tensor_info in enumerate(tensors[:10]):
            save_path = self.subdirs['heatmaps'] / f"heatmap_{i:03d}_{tensor_info['filename'][:20]}.png"
            self.plot_tensor_heatmap(tensor_info, str(save_path))
        
        # 生成量化类型对比图
        print("[TensorVisualizer] 生成量化类型对比图...")
        for group_name, group_tensors in groups.items():
            if group_tensors:
                save_path = self.subdirs['comparisons'] / f"quant_comparison_{group_name}.png"
                self.plot_quantization_comparison(group_tensors, str(save_path))
        
        # 生成统计汇总图
        print("[TensorVisualizer] 生成统计汇总图...")
        save_path = self.subdirs['statistics'] / "statistics_summary.png"
        self.plot_statistics_summary(tensors, str(save_path))
        
        # 生成attention分析图
        print("[TensorVisualizer] 生成attention分析图...")
        save_path = self.subdirs['attention_maps'] / "attention_analysis.png"
        self.plot_attention_analysis(tensors, str(save_path))
        
        print(f"[TensorVisualizer] 可视化完成！图表保存在: {self.output_dir}")
        self.print_summary()
    
    def print_summary(self):
        """打印生成的文件摘要"""
        print("\n=== 生成的可视化文件摘要 ===")
        
        for subdir_name, subdir_path in self.subdirs.items():
            files = list(subdir_path.glob("*.png"))
            print(f"{subdir_name}: {len(files)} 个文件")
            for file in files[:3]:  # 显示前3个文件
                print(f"  - {file.name}")
            if len(files) > 3:
                print(f"  ... 还有 {len(files) - 3} 个文件")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Tensor可视化工具')
    parser.add_argument('--tensor_dir', type=str, default='./tensor_logs',
                       help='tensor文件目录 (默认: ./tensor_logs)')
    parser.add_argument('--output_dir', type=str, default='./draw',
                       help='输出图片目录 (默认: ./draw)')
    parser.add_argument('--max_files', type=int, default=50,
                       help='最大处理文件数 (默认: 50)')
    
    args = parser.parse_args()
    
    print("=== Tensor可视化工具 ===")
    print(f"Tensor目录: {args.tensor_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"最大文件数: {args.max_files}")
    
    # 创建可视化器
    visualizer = TensorVisualizer(
        tensor_dir=args.tensor_dir,
        output_dir=args.output_dir
    )
    
    # 生成所有可视化图表
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
