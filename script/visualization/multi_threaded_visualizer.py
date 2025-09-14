#!/usr/bin/env python3
"""
Multi-threaded Tensor Visualization Tool
支持新的数据结构：bf16, mxfp8, mxfp4, hifp8四个量化类型
支持Sample (0,1,2) 和 Layer (1-16) 的多维度比较
使用多线程加速画图过程
"""

import os
import sys
import argparse
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib后端
plt.switch_backend('Agg')

class MultiThreadedTensorVisualizer:
    def __init__(self, tensor_dir: str, output_dir: str, max_workers: int = 4):
        self.tensor_dir = Path(tensor_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        
        # 支持的量化类型
        self.quant_types = ['bf16', 'mxfp8', 'mxfp4', 'hifp8']
        
        # 支持的样本和层
        self.samples = [0, 1, 2]
        self.layers = list(range(1, 17))  # 1-16层
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.subdirs = {
            'quantization_analysis': self.output_dir / 'quantization_analysis',
            'sample_analysis': self.output_dir / 'sample_analysis',
            'layer_analysis': self.output_dir / 'layer_analysis',
            'comparison_analysis': self.output_dir / 'comparison_analysis',
            'statistics': self.output_dir / 'statistics'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
    
    def parse_filename(self, filename: str) -> Optional[Dict]:
        """解析文件名，提取量化类型、层数、样本等信息"""
        try:
            # 文件名格式: YYYYMMDD_HHMMSS_XXXX_iterXXX_layer_type_operation_quant_type_rankXX_sampleXXX_groupXXX_tensor_name.pt
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
            
            # 查找层数
            layer_match = re.search(r'L(\d+)', filename)
            layer = int(layer_match.group(1)) if layer_match else None
            
            # 查找样本
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
            print(f"解析文件名失败: {filename}, 错误: {e}")
            return None
    
    def load_tensor_data(self) -> Dict:
        """加载所有tensor数据"""
        print("正在扫描tensor文件...")
        
        data = {
            'files': [],
            'by_quant_type': {qtype: [] for qtype in self.quant_types},
            'by_sample': {sample: [] for sample in self.samples},
            'by_layer': {layer: [] for layer in self.layers},
            'by_layer_type': {'attention': [], 'linear': []},
            'statistics': {}
        }
        
        # 扫描所有tensor文件
        for quant_type in self.quant_types:
            quant_dir = self.tensor_dir / quant_type
            if not quant_dir.exists():
                print(f"警告: 量化类型目录不存在: {quant_dir}")
                continue
            
            pt_files = list(quant_dir.glob('*.pt'))
            print(f"找到 {len(pt_files)} 个 {quant_type} 文件")
            
            for file_path in pt_files:
                file_info = self.parse_filename(file_path.name)
                if file_info:
                    file_info['file_path'] = file_path
                    data['files'].append(file_info)
                    data['by_quant_type'][quant_type].append(file_info)
                    
                    if file_info['sample'] is not None:
                        data['by_sample'][file_info['sample']].append(file_info)
                    
                    if file_info['layer'] is not None:
                        data['by_layer'][file_info['layer']].append(file_info)
                    
                    if file_info['layer_type']:
                        data['by_layer_type'][file_info['layer_type']].append(file_info)
        
        print(f"总共加载了 {len(data['files'])} 个tensor文件")
        return data
    
    def load_tensor_values(self, file_info: Dict) -> Optional[np.ndarray]:
        """加载tensor数值"""
        try:
            tensor = torch.load(file_info['file_path'], map_location='cpu')
            if isinstance(tensor, torch.Tensor):
                return tensor.numpy()
            return None
        except Exception as e:
            print(f"加载tensor失败: {file_info['filename']}, 错误: {e}")
            return None
    
    def calculate_statistics(self, data: Dict) -> Dict:
        """计算统计信息"""
        print("计算统计信息...")
        
        stats = {
            'quant_type_stats': {},
            'sample_stats': {},
            'layer_stats': {},
            'layer_type_stats': {},
            'overall_stats': {}
        }
        
        # 按量化类型统计
        for quant_type in self.quant_types:
            files = data['by_quant_type'][quant_type]
            if files:
                stats['quant_type_stats'][quant_type] = {
                    'count': len(files),
                    'layers': len(set(f['layer'] for f in files if f['layer'] is not None)),
                    'samples': len(set(f['sample'] for f in files if f['sample'] is not None)),
                    'layer_types': list(set(f['layer_type'] for f in files if f['layer_type']))
                }
        
        # 按样本统计
        for sample in self.samples:
            files = data['by_sample'][sample]
            if files:
                stats['sample_stats'][sample] = {
                    'count': len(files),
                    'quant_types': list(set(f['quant_type'] for f in files)),
                    'layers': len(set(f['layer'] for f in files if f['layer'] is not None))
                }
        
        # 按层统计
        for layer in self.layers:
            files = data['by_layer'][layer]
            if files:
                stats['layer_stats'][layer] = {
                    'count': len(files),
                    'quant_types': list(set(f['quant_type'] for f in files)),
                    'samples': len(set(f['sample'] for f in files if f['sample'] is not None)),
                    'layer_types': list(set(f['layer_type'] for f in files if f['layer_type']))
                }
        
        # 按层类型统计
        for layer_type in ['attention', 'linear']:
            files = data['by_layer_type'][layer_type]
            if files:
                stats['layer_type_stats'][layer_type] = {
                    'count': len(files),
                    'quant_types': list(set(f['quant_type'] for f in files)),
                    'layers': len(set(f['layer'] for f in files if f['layer'] is not None))
                }
        
        # 总体统计
        stats['overall_stats'] = {
            'total_files': len(data['files']),
            'quant_types': len([q for q in self.quant_types if data['by_quant_type'][q]]),
            'samples': len([s for s in self.samples if data['by_sample'][s]]),
            'layers': len([l for l in self.layers if data['by_layer'][l]]),
            'layer_types': len([lt for lt in ['attention', 'linear'] if data['by_layer_type'][lt]])
        }
        
        return stats
    
    def plot_quantization_comparison(self, data: Dict, stats: Dict):
        """绘制量化类型比较图"""
        print("绘制量化类型比较图...")
        
        # 准备数据
        quant_data = []
        for quant_type in self.quant_types:
            files = data['by_quant_type'][quant_type]
            if files:
                quant_data.append({
                    'quant_type': quant_type,
                    'count': len(files),
                    'layers': len(set(f['layer'] for f in files if f['layer'] is not None)),
                    'samples': len(set(f['sample'] for f in files if f['sample'] is not None))
                })
        
        if not quant_data:
            print("没有量化类型数据可绘制")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('量化类型比较分析', fontsize=16, fontweight='bold')
        
        # 1. 文件数量比较
        ax1 = axes[0, 0]
        quant_types = [d['quant_type'] for d in quant_data]
        counts = [d['count'] for d in quant_data]
        bars1 = ax1.bar(quant_types, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('各量化类型文件数量', fontweight='bold')
        ax1.set_ylabel('文件数量')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, count in zip(bars1, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 层数分布
        ax2 = axes[0, 1]
        layers = [d['layers'] for d in quant_data]
        bars2 = ax2.bar(quant_types, layers, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_title('各量化类型层数分布', fontweight='bold')
        ax2.set_ylabel('层数')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, layer in zip(bars2, layers):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(layers)*0.01,
                    f'{layer}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 样本数分布
        ax3 = axes[1, 0]
        samples = [d['samples'] for d in quant_data]
        bars3 = ax3.bar(quant_types, samples, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax3.set_title('各量化类型样本数分布', fontweight='bold')
        ax3.set_ylabel('样本数')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, sample in zip(bars3, samples):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(samples)*0.01,
                    f'{sample}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 综合比较
        ax4 = axes[1, 1]
        x = np.arange(len(quant_types))
        width = 0.25
        
        bars4_1 = ax4.bar(x - width, counts, width, label='文件数量', alpha=0.8)
        bars4_2 = ax4.bar(x, layers, width, label='层数', alpha=0.8)
        bars4_3 = ax4.bar(x + width, samples, width, label='样本数', alpha=0.8)
        
        ax4.set_title('综合比较', fontweight='bold')
        ax4.set_ylabel('数量')
        ax4.set_xticks(x)
        ax4.set_xticklabels(quant_types, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.subdirs['quantization_analysis'] / 'quantization_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"量化类型比较图已保存: {self.subdirs['quantization_analysis'] / 'quantization_comparison.png'}")
    
    def plot_sample_analysis(self, data: Dict, stats: Dict):
        """绘制样本分析图"""
        print("绘制样本分析图...")
        
        # 准备数据
        sample_data = []
        for sample in self.samples:
            files = data['by_sample'][sample]
            if files:
                sample_data.append({
                    'sample': sample,
                    'count': len(files),
                    'quant_types': len(set(f['quant_type'] for f in files)),
                    'layers': len(set(f['layer'] for f in files if f['layer'] is not None))
                })
        
        if not sample_data:
            print("没有样本数据可绘制")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('样本分析', fontsize=16, fontweight='bold')
        
        # 1. 样本文件数量
        ax1 = axes[0, 0]
        samples = [d['sample'] for d in sample_data]
        counts = [d['count'] for d in sample_data]
        bars1 = ax1.bar(samples, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('各样本文件数量', fontweight='bold')
        ax1.set_xlabel('样本编号')
        ax1.set_ylabel('文件数量')
        
        for bar, count in zip(bars1, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 样本量化类型分布
        ax2 = axes[0, 1]
        quant_types = [d['quant_types'] for d in sample_data]
        bars2 = ax2.bar(samples, quant_types, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title('各样本量化类型数量', fontweight='bold')
        ax2.set_xlabel('样本编号')
        ax2.set_ylabel('量化类型数量')
        
        for bar, qtype in zip(bars2, quant_types):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(quant_types)*0.01,
                    f'{qtype}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 样本层数分布
        ax3 = axes[1, 0]
        layers = [d['layers'] for d in sample_data]
        bars3 = ax3.bar(samples, layers, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax3.set_title('各样本层数分布', fontweight='bold')
        ax3.set_xlabel('样本编号')
        ax3.set_ylabel('层数')
        
        for bar, layer in zip(bars3, layers):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(layers)*0.01,
                    f'{layer}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 样本分布饼图
        ax4 = axes[1, 1]
        ax4.pie(counts, labels=[f'Sample {s}' for s in samples], autopct='%1.1f%%', 
                colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax4.set_title('样本分布比例', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.subdirs['sample_analysis'] / 'sample_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"样本分析图已保存: {self.subdirs['sample_analysis'] / 'sample_analysis.png'}")
    
    def plot_layer_analysis(self, data: Dict, stats: Dict):
        """绘制层分析图"""
        print("绘制层分析图...")
        
        # 准备数据
        layer_data = []
        for layer in self.layers:
            files = data['by_layer'][layer]
            if files:
                layer_data.append({
                    'layer': layer,
                    'count': len(files),
                    'quant_types': len(set(f['quant_type'] for f in files)),
                    'samples': len(set(f['sample'] for f in files if f['sample'] is not None)),
                    'layer_types': len(set(f['layer_type'] for f in files if f['layer_type']))
                })
        
        if not layer_data:
            print("没有层数据可绘制")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('层分析', fontsize=16, fontweight='bold')
        
        # 1. 各层文件数量
        ax1 = axes[0, 0]
        layers = [d['layer'] for d in layer_data]
        counts = [d['count'] for d in layer_data]
        bars1 = ax1.bar(layers, counts, color='skyblue', alpha=0.7)
        ax1.set_title('各层文件数量', fontweight='bold')
        ax1.set_xlabel('层编号')
        ax1.set_ylabel('文件数量')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 各层量化类型数量
        ax2 = axes[0, 1]
        quant_types = [d['quant_types'] for d in layer_data]
        bars2 = ax2.bar(layers, quant_types, color='lightcoral', alpha=0.7)
        ax2.set_title('各层量化类型数量', fontweight='bold')
        ax2.set_xlabel('层编号')
        ax2.set_ylabel('量化类型数量')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 各层样本数量
        ax3 = axes[1, 0]
        samples = [d['samples'] for d in layer_data]
        bars3 = ax3.bar(layers, samples, color='lightgreen', alpha=0.7)
        ax3.set_title('各层样本数量', fontweight='bold')
        ax3.set_xlabel('层编号')
        ax3.set_ylabel('样本数量')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 层分布热力图
        ax4 = axes[1, 1]
        
        # 创建层-量化类型矩阵
        layer_quant_matrix = np.zeros((len(self.layers), len(self.quant_types)))
        for i, layer in enumerate(self.layers):
            for j, quant_type in enumerate(self.quant_types):
                layer_files = [f for f in data['by_layer'][layer] if f['quant_type'] == quant_type]
                layer_quant_matrix[i, j] = len(layer_files)
        
        im = ax4.imshow(layer_quant_matrix, cmap='YlOrRd', aspect='auto')
        ax4.set_title('层-量化类型分布热力图', fontweight='bold')
        ax4.set_xlabel('量化类型')
        ax4.set_ylabel('层编号')
        ax4.set_xticks(range(len(self.quant_types)))
        ax4.set_xticklabels(self.quant_types, rotation=45)
        ax4.set_yticks(range(len(self.layers)))
        ax4.set_yticklabels(self.layers)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax4, label='文件数量')
        
        plt.tight_layout()
        plt.savefig(self.subdirs['layer_analysis'] / 'layer_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"层分析图已保存: {self.subdirs['layer_analysis'] / 'layer_analysis.png'}")
    
    def plot_comprehensive_comparison(self, data: Dict, stats: Dict):
        """绘制综合比较图"""
        print("绘制综合比较图...")
        
        # 创建大图
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 量化类型-样本矩阵
        ax1 = fig.add_subplot(gs[0, 0])
        quant_sample_matrix = np.zeros((len(self.quant_types), len(self.samples)))
        for i, quant_type in enumerate(self.quant_types):
            for j, sample in enumerate(self.samples):
                files = [f for f in data['by_quant_type'][quant_type] if f['sample'] == sample]
                quant_sample_matrix[i, j] = len(files)
        
        im1 = ax1.imshow(quant_sample_matrix, cmap='Blues', aspect='auto')
        ax1.set_title('量化类型-样本分布', fontweight='bold')
        ax1.set_xlabel('样本编号')
        ax1.set_ylabel('量化类型')
        ax1.set_xticks(range(len(self.samples)))
        ax1.set_xticklabels(self.samples)
        ax1.set_yticks(range(len(self.quant_types)))
        ax1.set_yticklabels(self.quant_types)
        
        # 添加数值标签
        for i in range(len(self.quant_types)):
            for j in range(len(self.samples)):
                text = ax1.text(j, i, f'{int(quant_sample_matrix[i, j])}', 
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, label='文件数量')
        
        # 2. 量化类型-层矩阵
        ax2 = fig.add_subplot(gs[0, 1])
        quant_layer_matrix = np.zeros((len(self.quant_types), len(self.layers)))
        for i, quant_type in enumerate(self.quant_types):
            for j, layer in enumerate(self.layers):
                files = [f for f in data['by_quant_type'][quant_type] if f['layer'] == layer]
                quant_layer_matrix[i, j] = len(files)
        
        im2 = ax2.imshow(quant_layer_matrix, cmap='Greens', aspect='auto')
        ax2.set_title('量化类型-层分布', fontweight='bold')
        ax2.set_xlabel('层编号')
        ax2.set_ylabel('量化类型')
        ax2.set_xticks(range(0, len(self.layers), 2))
        ax2.set_xticklabels(self.layers[::2])
        ax2.set_yticks(range(len(self.quant_types)))
        ax2.set_yticklabels(self.quant_types)
        
        plt.colorbar(im2, ax=ax2, label='文件数量')
        
        # 3. 样本-层矩阵
        ax3 = fig.add_subplot(gs[0, 2])
        sample_layer_matrix = np.zeros((len(self.samples), len(self.layers)))
        for i, sample in enumerate(self.samples):
            for j, layer in enumerate(self.layers):
                files = [f for f in data['by_sample'][sample] if f['layer'] == layer]
                sample_layer_matrix[i, j] = len(files)
        
        im3 = ax3.imshow(sample_layer_matrix, cmap='Reds', aspect='auto')
        ax3.set_title('样本-层分布', fontweight='bold')
        ax3.set_xlabel('层编号')
        ax3.set_ylabel('样本编号')
        ax3.set_xticks(range(0, len(self.layers), 2))
        ax3.set_xticklabels(self.layers[::2])
        ax3.set_yticks(range(len(self.samples)))
        ax3.set_yticklabels(self.samples)
        
        plt.colorbar(im3, ax=ax3, label='文件数量')
        
        # 4. 层类型分布
        ax4 = fig.add_subplot(gs[0, 3])
        layer_types = ['attention', 'linear']
        layer_type_counts = [len(data['by_layer_type'][lt]) for lt in layer_types]
        bars4 = ax4.bar(layer_types, layer_type_counts, color=['#ff9999', '#66b3ff'])
        ax4.set_title('层类型分布', fontweight='bold')
        ax4.set_ylabel('文件数量')
        
        for bar, count in zip(bars4, layer_type_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(layer_type_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 5. 量化类型文件数量趋势
        ax5 = fig.add_subplot(gs[1, :2])
        quant_counts = [len(data['by_quant_type'][qt]) for qt in self.quant_types]
        bars5 = ax5.bar(self.quant_types, quant_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax5.set_title('量化类型文件数量分布', fontweight='bold')
        ax5.set_ylabel('文件数量')
        ax5.tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars5, quant_counts):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(quant_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 6. 样本文件数量趋势
        ax6 = fig.add_subplot(gs[1, 2:])
        sample_counts = [len(data['by_sample'][s]) for s in self.samples]
        bars6 = ax6.bar(self.samples, sample_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax6.set_title('样本文件数量分布', fontweight='bold')
        ax6.set_xlabel('样本编号')
        ax6.set_ylabel('文件数量')
        
        for bar, count in zip(bars6, sample_counts):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 7. 层文件数量趋势
        ax7 = fig.add_subplot(gs[2, :])
        layer_counts = [len(data['by_layer'][l]) for l in self.layers]
        ax7.plot(self.layers, layer_counts, marker='o', linewidth=2, markersize=6, color='#2ca02c')
        ax7.set_title('各层文件数量趋势', fontweight='bold')
        ax7.set_xlabel('层编号')
        ax7.set_ylabel('文件数量')
        ax7.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, count in enumerate(layer_counts):
            if i % 2 == 0:  # 只显示偶数层的标签，避免重叠
                ax7.annotate(f'{count}', (self.layers[i], count), 
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.suptitle('综合比较分析', fontsize=20, fontweight='bold', y=0.98)
        plt.savefig(self.subdirs['comparison_analysis'] / 'comprehensive_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"综合比较图已保存: {self.subdirs['comparison_analysis'] / 'comprehensive_comparison.png'}")
    
    def save_statistics_report(self, stats: Dict):
        """保存统计报告"""
        print("保存统计报告...")
        
        report_path = self.subdirs['statistics'] / 'detailed_statistics_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("多线程Tensor可视化统计报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 总体统计
            f.write("总体统计:\n")
            f.write("-" * 40 + "\n")
            f.write(f"总文件数: {stats['overall_stats']['total_files']:,}\n")
            f.write(f"量化类型数: {stats['overall_stats']['quant_types']}\n")
            f.write(f"样本数: {stats['overall_stats']['samples']}\n")
            f.write(f"层数: {stats['overall_stats']['layers']}\n")
            f.write(f"层类型数: {stats['overall_stats']['layer_types']}\n\n")
            
            # 量化类型统计
            f.write("量化类型统计:\n")
            f.write("-" * 40 + "\n")
            for quant_type, stat in stats['quant_type_stats'].items():
                f.write(f"{quant_type}:\n")
                f.write(f"  文件数: {stat['count']:,}\n")
                f.write(f"  层数: {stat['layers']}\n")
                f.write(f"  样本数: {stat['samples']}\n")
                f.write(f"  层类型: {', '.join(stat['layer_types'])}\n\n")
            
            # 样本统计
            f.write("样本统计:\n")
            f.write("-" * 40 + "\n")
            for sample, stat in stats['sample_stats'].items():
                f.write(f"Sample {sample}:\n")
                f.write(f"  文件数: {stat['count']:,}\n")
                f.write(f"  量化类型: {', '.join(stat['quant_types'])}\n")
                f.write(f"  层数: {stat['layers']}\n\n")
            
            # 层统计
            f.write("层统计:\n")
            f.write("-" * 40 + "\n")
            for layer, stat in stats['layer_stats'].items():
                f.write(f"Layer {layer}:\n")
                f.write(f"  文件数: {stat['count']:,}\n")
                f.write(f"  量化类型: {', '.join(stat['quant_types'])}\n")
                f.write(f"  样本数: {stat['samples']}\n")
                f.write(f"  层类型: {', '.join(stat['layer_types'])}\n\n")
            
            # 层类型统计
            f.write("层类型统计:\n")
            f.write("-" * 40 + "\n")
            for layer_type, stat in stats['layer_type_stats'].items():
                f.write(f"{layer_type}:\n")
                f.write(f"  文件数: {stat['count']:,}\n")
                f.write(f"  量化类型: {', '.join(stat['quant_types'])}\n")
                f.write(f"  层数: {stat['layers']}\n\n")
        
        print(f"统计报告已保存: {report_path}")
    
    def run_visualization(self):
        """运行可视化"""
        print("开始多线程Tensor可视化...")
        print(f"Tensor目录: {self.tensor_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"最大线程数: {self.max_workers}")
        
        # 加载数据
        data = self.load_tensor_data()
        if not data['files']:
            print("错误: 没有找到tensor文件")
            return
        
        # 计算统计信息
        stats = self.calculate_statistics(data)
        
        # 使用多线程绘制图表
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            futures = [
                executor.submit(self.plot_quantization_comparison, data, stats),
                executor.submit(self.plot_sample_analysis, data, stats),
                executor.submit(self.plot_layer_analysis, data, stats),
                executor.submit(self.plot_comprehensive_comparison, data, stats),
                executor.submit(self.save_statistics_report, stats)
            ]
            
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"任务执行失败: {e}")
        
        print("\n" + "=" * 60)
        print("多线程Tensor可视化完成!")
        print("=" * 60)
        print(f"输出目录: {self.output_dir}")
        print("生成的图表:")
        print(f"  - 量化类型比较: {self.subdirs['quantization_analysis'] / 'quantization_comparison.png'}")
        print(f"  - 样本分析: {self.subdirs['sample_analysis'] / 'sample_analysis.png'}")
        print(f"  - 层分析: {self.subdirs['layer_analysis'] / 'layer_analysis.png'}")
        print(f"  - 综合比较: {self.subdirs['comparison_analysis'] / 'comprehensive_comparison.png'}")
        print(f"  - 统计报告: {self.subdirs['statistics'] / 'detailed_statistics_report.txt'}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='多线程Tensor可视化工具')
    parser.add_argument('--tensor_dir', type=str, default='./enhanced_tensor_logs',
                       help='Tensor文件目录')
    parser.add_argument('--output_dir', type=str, default='./draw',
                       help='输出目录')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='最大线程数')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = MultiThreadedTensorVisualizer(
        tensor_dir=args.tensor_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    # 运行可视化
    visualizer.run_visualization()

if __name__ == "__main__":
    main()
