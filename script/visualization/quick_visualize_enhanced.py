#!/usr/bin/env python3
"""
增强版快速Tensor可视化脚本
专门针对enhanced_tensor_logs中的tensor文件进行快速高质量分析
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import pandas as pd
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

def quick_visualize_enhanced(tensor_dir="./enhanced_tensor_logs", output_dir="./draw"):
    """增强版快速可视化tensor数据"""
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找tensor文件
    tensor_files = glob.glob(f"{tensor_dir}/*.pt")
    print(f"找到 {len(tensor_files)} 个tensor文件")
    
    if not tensor_files:
        print("没有找到tensor文件，请检查目录路径")
        return
    
    # 加载和分析数据
    tensor_data = []
    for file_path in tensor_files:
        try:
            data = torch.load(file_path, map_location='cpu')
            tensor_info = {
                'file_path': file_path,
                'filename': Path(file_path).name,
                'tensor': data['tensor'].float().numpy(),
                'metadata': data['metadata'],
                'tensor_info': data['tensor_info']
            }
            tensor_data.append(tensor_info)
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
    
    if not tensor_data:
        print("没有成功加载任何tensor数据")
        return
    
    print(f"成功加载 {len(tensor_data)} 个tensor文件")
    
    # 1. 创建汇总分析图
    create_summary_analysis(tensor_data, output_path)
    
    # 2. 创建量化类型对比图
    create_quantization_comparison(tensor_data, output_path)
    
    # 3. 创建attention分析图
    create_attention_analysis(tensor_data, output_path)
    
    # 4. 生成详细统计报告
    generate_detailed_stats(tensor_data, output_path)
    
    print(f"所有分析图已保存到: {output_path}")


def create_summary_analysis(tensor_data, output_path):
    """创建汇总分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tensor数据汇总分析', fontsize=16, fontweight='bold')
    
    # 收集统计数据
    quant_types = [t['metadata']['quant_type'] for t in tensor_data]
    layer_types = [t['metadata']['layer_type'] for t in tensor_data]
    operations = [t['metadata']['operation'] for t in tensor_data]
    tensor_names = [t['metadata']['tensor_name'] for t in tensor_data]
    
    # 1. 量化类型分布
    ax1 = axes[0, 0]
    quant_counts = pd.Series(quant_types).value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(quant_counts)))
    wedges, texts, autotexts = ax1.pie(quant_counts.values, labels=quant_counts.index, 
                                       autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('量化类型分布', fontweight='bold')
    
    # 2. 层类型分布
    ax2 = axes[0, 1]
    layer_counts = pd.Series(layer_types).value_counts()
    bars = ax2.bar(layer_counts.index, layer_counts.values, 
                   color=plt.cm.Set2(np.linspace(0, 1, len(layer_counts))), alpha=0.8)
    ax2.set_title('层类型分布', fontweight='bold')
    ax2.set_xlabel('层类型')
    ax2.set_ylabel('数量')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, count in zip(bars, layer_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    # 3. 操作类型分布
    ax3 = axes[1, 0]
    op_counts = pd.Series(operations).value_counts()
    bars = ax3.bar(op_counts.index, op_counts.values,
                   color=plt.cm.Set1(np.linspace(0, 1, len(op_counts))), alpha=0.8)
    ax3.set_title('操作类型分布', fontweight='bold')
    ax3.set_xlabel('操作类型')
    ax3.set_ylabel('数量')
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, count in zip(bars, op_counts.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    # 4. tensor名称分布
    ax4 = axes[1, 1]
    name_counts = pd.Series(tensor_names).value_counts()
    if len(name_counts) > 10:  # 如果tensor名称太多，只显示前10个
        name_counts = name_counts.head(10)
    
    bars = ax4.bar(range(len(name_counts)), name_counts.values,
                   color=plt.cm.tab10(np.linspace(0, 1, len(name_counts))), alpha=0.8)
    ax4.set_title('Tensor名称分布 (Top 10)', fontweight='bold')
    ax4.set_xlabel('Tensor名称')
    ax4.set_ylabel('数量')
    ax4.set_xticks(range(len(name_counts)))
    ax4.set_xticklabels(name_counts.index, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, count in zip(bars, name_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图片
    output_file = output_path / 'summary_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"汇总分析图已保存: {output_file}")


def create_quantization_comparison(tensor_data, output_path):
    """创建量化类型对比图"""
    # 按量化类型分组
    quant_groups = defaultdict(list)
    for tensor_info in tensor_data:
        quant_type = tensor_info['metadata']['quant_type']
        quant_groups[quant_type].append(tensor_info)
    
    if len(quant_groups) < 2:
        print("量化类型数量不足，跳过量化对比图")
        return
    
    # 选择相同tensor_name的文件进行对比
    tensor_name_groups = defaultdict(list)
    for quant_type, group in quant_groups.items():
        for tensor_info in group:
            tensor_name = tensor_info['metadata']['tensor_name']
            tensor_name_groups[tensor_name].append((quant_type, tensor_info))
    
    # 为每个tensor_name创建对比图
    for tensor_name, quant_tensors in tensor_name_groups.items():
        if len(quant_tensors) < 2:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{tensor_name} - 量化类型对比', fontsize=16, fontweight='bold')
        
        # 收集数据
        data_by_quant = {}
        for quant_type, tensor_info in quant_tensors:
            tensor_data = tensor_info['tensor'].flatten()
            data_by_quant[quant_type] = tensor_data
        
        # 分布对比
        ax1 = axes[0, 0]
        for quant_type, data in data_by_quant.items():
            ax1.hist(data, bins=50, alpha=0.6, label=quant_type, density=True)
        ax1.set_title('数值分布对比')
        ax1.set_xlabel('数值')
        ax1.set_ylabel('密度')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 箱线图对比
        ax2 = axes[0, 1]
        box_data = [data_by_quant[qt] for qt in data_by_quant.keys()]
        box_labels = list(data_by_quant.keys())
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_title('数值分布箱线图对比')
        ax2.set_ylabel('数值')
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
        
        # 保存图片
        output_file = output_path / f'{tensor_name}_quantization_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"量化对比图已保存: {output_file}")


def create_attention_analysis(tensor_data, output_path):
    """创建attention分析图"""
    attention_tensors = [t for t in tensor_data if t['metadata']['layer_type'] == 'attention']
    
    if not attention_tensors:
        print("没有找到attention层tensor，跳过attention分析")
        return
    
    # 按tensor_name分组
    attention_groups = defaultdict(list)
    for tensor_info in attention_tensors:
        tensor_name = tensor_info['metadata']['tensor_name']
        attention_groups[tensor_name].append(tensor_info)
    
    # 为每个attention tensor创建分析图
    for tensor_name, group in attention_groups.items():
        if len(group) < 1:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Attention {tensor_name} 分析', fontsize=16, fontweight='bold')
        
        # 收集数据
        all_data = []
        quant_types = []
        for tensor_info in group:
            data = tensor_info['tensor'].flatten()
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
        
        # 热力图（如果是2D tensor）
        ax2 = axes[0, 1]
        if len(group) > 0:
            sample_tensor = group[0]['tensor']
            if len(sample_tensor.shape) == 2:
                im = ax2.imshow(sample_tensor, cmap='viridis', aspect='auto')
                ax2.set_title(f'{tensor_name} 热力图')
                ax2.set_xlabel('列')
                ax2.set_ylabel('行')
                plt.colorbar(im, ax=ax2)
            else:
                # 如果是高维tensor，取平均值
                if len(sample_tensor.shape) > 2:
                    sample_tensor = np.mean(sample_tensor, axis=tuple(range(2, len(sample_tensor.shape))))
                im = ax2.imshow(sample_tensor, cmap='viridis', aspect='auto')
                ax2.set_title(f'{tensor_name} 热力图 (平均)')
                ax2.set_xlabel('列')
                ax2.set_ylabel('行')
                plt.colorbar(im, ax=ax2)
        
        # 统计信息
        ax3 = axes[1, 0]
        stats_text = f"{tensor_name} 统计信息:\n\n"
        for i, (tensor_info, quant_type) in enumerate(zip(group, quant_types)):
            stats = tensor_info['tensor_info']
            stats_text += f"样本 {i+1} ({quant_type}):\n"
            stats_text += f"  形状: {stats['shape']}\n"
            stats_text += f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]\n"
            stats_text += f"  均值: {stats['mean']:.4f}\n"
            stats_text += f"  标准差: {stats['std']:.4f}\n\n"
        
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # 时间序列分析（如果有多个样本）
        ax4 = axes[1, 1]
        if len(all_data) > 1:
            means = [np.mean(data) for data in all_data]
            stds = [np.std(data) for data in all_data]
            
            x = range(len(means))
            ax4.errorbar(x, means, yerr=stds, marker='o', capsize=5, capthick=2)
            ax4.set_title(f'{tensor_name} 统计信息变化')
            ax4.set_xlabel('样本索引')
            ax4.set_ylabel('均值 ± 标准差')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '需要多个样本\n进行时间序列分析', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # 保存图片
        output_file = output_path / f'{tensor_name}_attention_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Attention分析图已保存: {output_file}")


def generate_detailed_stats(tensor_data, output_path):
    """生成详细统计报告"""
    stats_file = output_path / 'detailed_tensor_stats.txt'
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("Tensor数据详细统计报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 基本信息
        f.write(f"总文件数: {len(tensor_data)}\n")
        f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 按量化类型统计
        quant_stats = defaultdict(list)
        for tensor_info in tensor_data:
            quant_type = tensor_info['metadata']['quant_type']
            quant_stats[quant_type].append(tensor_info)
        
        f.write("量化类型统计:\n")
        f.write("-" * 30 + "\n")
        for quant_type, group in quant_stats.items():
            f.write(f"\n{quant_type}:\n")
            f.write(f"  文件数量: {len(group)}\n")
            
            # 计算统计信息
            all_values = []
            for tensor_info in group:
                values = tensor_info['tensor'].flatten()
                all_values.extend(values)
            
            if all_values:
                f.write(f"  总数值数: {len(all_values)}\n")
                f.write(f"  最小值: {np.min(all_values):.6f}\n")
                f.write(f"  最大值: {np.max(all_values):.6f}\n")
                f.write(f"  均值: {np.mean(all_values):.6f}\n")
                f.write(f"  标准差: {np.std(all_values):.6f}\n")
                f.write(f"  中位数: {np.median(all_values):.6f}\n")
        
        # 按层类型统计
        layer_stats = defaultdict(list)
        for tensor_info in tensor_data:
            layer_type = tensor_info['metadata']['layer_type']
            layer_stats[layer_type].append(tensor_info)
        
        f.write("\n\n层类型统计:\n")
        f.write("-" * 30 + "\n")
        for layer_type, group in layer_stats.items():
            f.write(f"\n{layer_type}:\n")
            f.write(f"  文件数量: {len(group)}\n")
            
            # 按tensor_name分组
            name_groups = defaultdict(list)
            for tensor_info in group:
                tensor_name = tensor_info['metadata']['tensor_name']
                name_groups[tensor_name].append(tensor_info)
            
            f.write(f"  Tensor名称: {', '.join(name_groups.keys())}\n")
        
        # 按操作类型统计
        op_stats = defaultdict(list)
        for tensor_info in tensor_data:
            operation = tensor_info['metadata']['operation']
            op_stats[operation].append(tensor_info)
        
        f.write("\n\n操作类型统计:\n")
        f.write("-" * 30 + "\n")
        for operation, group in op_stats.items():
            f.write(f"\n{operation}:\n")
            f.write(f"  文件数量: {len(group)}\n")
        
        # 文件详细信息
        f.write("\n\n文件详细信息:\n")
        f.write("-" * 30 + "\n")
        for i, tensor_info in enumerate(tensor_data, 1):
            metadata = tensor_info['metadata']
            tensor_info_data = tensor_info['tensor_info']
            
            f.write(f"\n文件 {i}: {tensor_info['filename']}\n")
            f.write(f"  层类型: {metadata['layer_type']}\n")
            f.write(f"  操作: {metadata['operation']}\n")
            f.write(f"  量化类型: {metadata['quant_type']}\n")
            f.write(f"  Tensor名称: {metadata['tensor_name']}\n")
            f.write(f"  层索引: {metadata.get('layer_idx', 'N/A')}\n")
            f.write(f"  形状: {tensor_info_data['shape']}\n")
            f.write(f"  数据类型: {tensor_info_data['dtype']}\n")
            f.write(f"  数值范围: [{tensor_info_data['min']:.6f}, {tensor_info_data['max']:.6f}]\n")
            f.write(f"  均值: {tensor_info_data['mean']:.6f}\n")
            f.write(f"  标准差: {tensor_info_data['std']:.6f}\n")
    
    print(f"详细统计报告已保存: {stats_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增强版快速Tensor可视化工具')
    parser.add_argument('--tensor_dir', type=str, default='./enhanced_tensor_logs',
                       help='tensor文件目录 (默认: ./enhanced_tensor_logs)')
    parser.add_argument('--output_dir', type=str, default='./draw',
                       help='输出目录 (默认: ./draw)')
    
    args = parser.parse_args()
    
    # 运行可视化
    quick_visualize_enhanced(
        tensor_dir=args.tensor_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
