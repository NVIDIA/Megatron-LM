#!/usr/bin/env python3
"""
Quick Tensor Visualization Script
Enhanced version for quickly generating high-quality analysis charts
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

def quick_visualize(tensor_dir="./enhanced_tensor_logs", output_dir="./draw"):
    """Quick visualization of tensor data"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find tensor files
    tensor_files = glob.glob(f"{tensor_dir}/*.pt")
    print(f"Found {len(tensor_files)} tensor files")
    
    if not tensor_files:
        print("No tensor files found, please check directory path")
        return
    
    # Create summary chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Tensor Data Quick Analysis', fontsize=16)
    
    # Collect data
    all_values = []
    quant_types = []
    layer_types = []
    operations = []
    
    for file_path in tensor_files[:20]:  # Limit number of files to process
        try:
            data = torch.load(file_path, map_location='cpu')
            # Convert BFloat16 to Float32 to support numpy
            tensor = data['tensor'].float().numpy().flatten()
            metadata = data['metadata']
            
            all_values.extend(tensor)
            quant_types.append(metadata['quant_type'])
            layer_types.append(metadata['layer_type'])
            operations.append(metadata['operation'])
            
        except Exception as e:
            print(f"Failed to load file {file_path}: {e}")
    
    # Plot distribution
    if len(all_values) > 0:
        axes[0, 0].hist(all_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('All Tensor Value Distribution')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('All Tensor Value Distribution')
    
    # Quantization type distribution
    from collections import Counter
    quant_counts = Counter(quant_types)
    if quant_counts:
        axes[0, 1].pie(quant_counts.values(), labels=quant_counts.keys(), autopct='%1.1f%%')
        axes[0, 1].set_title('Quantization Type Distribution')
    else:
        axes[0, 1].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Quantization Type Distribution')
    
    # Layer type distribution
    layer_counts = Counter(layer_types)
    if layer_counts:
        axes[1, 0].pie(layer_counts.values(), labels=layer_counts.keys(), autopct='%1.1f%%')
        axes[1, 0].set_title('Layer Type Distribution')
    else:
        axes[1, 0].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Layer Type Distribution')
    
    # Operation type distribution
    op_counts = Counter(operations)
    if op_counts:
        axes[1, 1].pie(op_counts.values(), labels=op_counts.keys(), autopct='%1.1f%%')
        axes[1, 1].set_title('Operation Type Distribution')
    else:
        axes[1, 1].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Operation Type Distribution')
    
    plt.tight_layout()
    
    # Save image
    output_file = output_path / "quick_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Quick analysis chart saved to: {output_file}")
    
    # Generate statistical information
    stats_file = output_path / "tensor_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=== Tensor Statistical Information ===\n")
        f.write(f"Total files: {len(tensor_files)}\n")
        f.write(f"Processed files: {min(20, len(tensor_files))}\n")
        f.write(f"Total value count: {len(all_values)}\n")
        if len(all_values) > 0:
            f.write(f"Value range: [{np.min(all_values):.4f}, {np.max(all_values):.4f}]\n")
            f.write(f"Mean: {np.mean(all_values):.4f}\n")
            f.write(f"Std dev: {np.std(all_values):.4f}\n\n")
        else:
            f.write("Value range: No data\n")
            f.write("Mean: No data\n")
            f.write("Std dev: No data\n\n")
        
        f.write("Quantization type distribution:\n")
        for quant_type, count in quant_counts.items():
            f.write(f"  {quant_type}: {count} files\n")
        
        f.write("\nLayer type distribution:\n")
        for layer_type, count in layer_counts.items():
            f.write(f"  {layer_type}: {count} files\n")
        
        f.write("\nOperation type distribution:\n")
        for operation, count in op_counts.items():
            f.write(f"  {operation}: {count} files\n")
    
    print(f"Statistical information saved to: {stats_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick Tensor Visualization')
    parser.add_argument('--tensor_dir', type=str, default='./enhanced_tensor_logs',
                       help='Tensor file directory')
    parser.add_argument('--output_dir', type=str, default='./draw',
                       help='Output directory')
    
    args = parser.parse_args()
    
    quick_visualize(args.tensor_dir, args.output_dir)
