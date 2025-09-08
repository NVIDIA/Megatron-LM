#!/usr/bin/env python3
"""
快速Tensor可视化脚本
简化版本，用于快速生成基本图表
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# 添加项目路径
sys.path.append('/data/charles/Megatron-LM')

def quick_visualize(tensor_dir="./tensor_logs", output_dir="./draw"):
    """快速可视化tensor数据"""
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找tensor文件
    tensor_files = glob.glob(f"{tensor_dir}/*.pt")
    print(f"找到 {len(tensor_files)} 个tensor文件")
    
    if not tensor_files:
        print("没有找到tensor文件，请检查目录路径")
        return
    
    # 创建汇总图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Tensor数据快速分析', fontsize=16)
    
    # 收集数据
    all_values = []
    quant_types = []
    layer_types = []
    operations = []
    
    for file_path in tensor_files[:20]:  # 限制处理文件数
        try:
            data = torch.load(file_path, map_location='cpu')
            # 转换BFloat16为Float32以支持numpy
            tensor = data['tensor'].float().numpy().flatten()
            metadata = data['metadata']
            
            all_values.extend(tensor)
            quant_types.append(metadata['quant_type'])
            layer_types.append(metadata['layer_type'])
            operations.append(metadata['operation'])
            
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
    
    # 绘制分布图
    if len(all_values) > 0:
        axes[0, 0].hist(all_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('All Tensor Value Distribution')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('All Tensor Value Distribution')
    
    # 量化类型分布
    from collections import Counter
    quant_counts = Counter(quant_types)
    if quant_counts:
        axes[0, 1].pie(quant_counts.values(), labels=quant_counts.keys(), autopct='%1.1f%%')
        axes[0, 1].set_title('Quantization Type Distribution')
    else:
        axes[0, 1].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Quantization Type Distribution')
    
    # 层类型分布
    layer_counts = Counter(layer_types)
    if layer_counts:
        axes[1, 0].pie(layer_counts.values(), labels=layer_counts.keys(), autopct='%1.1f%%')
        axes[1, 0].set_title('Layer Type Distribution')
    else:
        axes[1, 0].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Layer Type Distribution')
    
    # 操作类型分布
    op_counts = Counter(operations)
    if op_counts:
        axes[1, 1].pie(op_counts.values(), labels=op_counts.keys(), autopct='%1.1f%%')
        axes[1, 1].set_title('Operation Type Distribution')
    else:
        axes[1, 1].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Operation Type Distribution')
    
    plt.tight_layout()
    
    # 保存图片
    output_file = output_path / "quick_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"快速分析图已保存到: {output_file}")
    
    # 生成统计信息
    stats_file = output_path / "tensor_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=== Tensor统计信息 ===\n")
        f.write(f"总文件数: {len(tensor_files)}\n")
        f.write(f"处理文件数: {min(20, len(tensor_files))}\n")
        f.write(f"总数值数量: {len(all_values)}\n")
        if len(all_values) > 0:
            f.write(f"数值范围: [{np.min(all_values):.4f}, {np.max(all_values):.4f}]\n")
            f.write(f"均值: {np.mean(all_values):.4f}\n")
            f.write(f"标准差: {np.std(all_values):.4f}\n\n")
        else:
            f.write("数值范围: 无数据\n")
            f.write("均值: 无数据\n")
            f.write("标准差: 无数据\n\n")
        
        f.write("量化类型分布:\n")
        for quant_type, count in quant_counts.items():
            f.write(f"  {quant_type}: {count} 个文件\n")
        
        f.write("\n层类型分布:\n")
        for layer_type, count in layer_counts.items():
            f.write(f"  {layer_type}: {count} 个文件\n")
        
        f.write("\n操作类型分布:\n")
        for operation, count in op_counts.items():
            f.write(f"  {operation}: {count} 个文件\n")
    
    print(f"统计信息已保存到: {stats_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='快速Tensor可视化')
    parser.add_argument('--tensor_dir', type=str, default='./tensor_logs',
                       help='tensor文件目录')
    parser.add_argument('--output_dir', type=str, default='./draw',
                       help='输出目录')
    
    args = parser.parse_args()
    
    quick_visualize(args.tensor_dir, args.output_dir)
