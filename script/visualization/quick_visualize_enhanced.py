#!/usr/bin/env python3
"""
Enhanced Quick Tensor Visualization Script
Specialized for quick high-quality analysis of tensor files in enhanced_tensor_logs
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
    """Enhanced quick visualization of tensor data"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find tensor files
    tensor_files = glob.glob(f"{tensor_dir}/*.pt")
    print(f"Found {len(tensor_files)} tensor files")
    
    if not tensor_files:
        print("No tensor files found, please check directory path")
        return
    
    # Load and analyze data
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
            print(f"Failed to load file {file_path}: {e}")
    
    if not tensor_data:
        print("Failed to load any tensor data")
        return
    
    print(f"Successfully loaded {len(tensor_data)} tensor files")
    
    # 1. Create summary analysis chart
    create_summary_analysis(tensor_data, output_path)
    
    # 2. Create quantization type comparison chart
    create_quantization_comparison(tensor_data, output_path)
    
    # 3. Create attention analysis chart
    create_attention_analysis(tensor_data, output_path)
    
    # 4. Generate detailed statistical report
    generate_detailed_stats(tensor_data, output_path)
    
    print(f"All analysis charts saved to: {output_path}")


def create_summary_analysis(tensor_data, output_path):
    """Create summary analysis chart"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tensor Data Summary Analysis', fontsize=16, fontweight='bold')
    
    # Collect statistical data
    quant_types = [t['metadata']['quant_type'] for t in tensor_data]
    layer_types = [t['metadata']['layer_type'] for t in tensor_data]
    operations = [t['metadata']['operation'] for t in tensor_data]
    tensor_names = [t['metadata']['tensor_name'] for t in tensor_data]
    
    # 1. Quantization type distribution
    ax1 = axes[0, 0]
    quant_counts = pd.Series(quant_types).value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(quant_counts)))
    wedges, texts, autotexts = ax1.pie(quant_counts.values, labels=quant_counts.index, 
                                       autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('Quantization Type Distribution', fontweight='bold')
    
    # 2. Layer type distribution
    ax2 = axes[0, 1]
    layer_counts = pd.Series(layer_types).value_counts()
    bars = ax2.bar(layer_counts.index, layer_counts.values, 
                   color=plt.cm.Set2(np.linspace(0, 1, len(layer_counts))), alpha=0.8)
    ax2.set_title('Layer Type Distribution', fontweight='bold')
    ax2.set_xlabel('Layer Type')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, layer_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    # 3. Operation type distribution
    ax3 = axes[1, 0]
    op_counts = pd.Series(operations).value_counts()
    bars = ax3.bar(op_counts.index, op_counts.values,
                   color=plt.cm.Set1(np.linspace(0, 1, len(op_counts))), alpha=0.8)
    ax3.set_title('Operation Type Distribution', fontweight='bold')
    ax3.set_xlabel('Operation Type')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, op_counts.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    # 4. Tensor name distribution
    ax4 = axes[1, 1]
    name_counts = pd.Series(tensor_names).value_counts()
    if len(name_counts) > 10:  # If too many tensor names, only show top 10
        name_counts = name_counts.head(10)
    
    bars = ax4.bar(range(len(name_counts)), name_counts.values,
                   color=plt.cm.tab10(np.linspace(0, 1, len(name_counts))), alpha=0.8)
    ax4.set_title('Tensor Name Distribution (Top 10)', fontweight='bold')
    ax4.set_xlabel('Tensor Name')
    ax4.set_ylabel('Count')
    ax4.set_xticks(range(len(name_counts)))
    ax4.set_xticklabels(name_counts.index, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, name_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save image
    output_file = output_path / 'summary_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary analysis chart saved: {output_file}")


def create_quantization_comparison(tensor_data, output_path):
    """Create quantization type comparison chart"""
    # Group by quantization type
    quant_groups = defaultdict(list)
    for tensor_info in tensor_data:
        quant_type = tensor_info['metadata']['quant_type']
        quant_groups[quant_type].append(tensor_info)
    
    if len(quant_groups) < 2:
        print("Insufficient quantization types, skipping quantization comparison chart")
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
        fig.suptitle(f'{tensor_name} - Quantization Type Comparison', fontsize=16, fontweight='bold')
        
        # Collect data
        data_by_quant = {}
        for quant_type, tensor_info in quant_tensors:
            tensor_data = tensor_info['tensor'].flatten()
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
        
        # Statistical information comparison
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
        
        ax3.bar(x - width, stats_df['mean'], width, label='Mean', alpha=0.8)
        ax3.bar(x, stats_df['std'], width, label='Std Dev', alpha=0.8)
        ax3.bar(x + width, stats_df['max'] - stats_df['min'], width, label='Range', alpha=0.8)
        
        ax3.set_xlabel('Quantization Type')
        ax3.set_ylabel('Value')
        ax3.set_title('Statistical Information Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(stats_df['quant_type'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Correlation analysis
        ax4 = axes[1, 1]
        if len(data_by_quant) == 2:
            quant_types = list(data_by_quant.keys())
            data1 = data_by_quant[quant_types[0]]
            data2 = data_by_quant[quant_types[1]]
            
            # Sample to reduce computation
            if len(data1) > 10000:
                indices = np.random.choice(len(data1), 10000, replace=False)
                data1 = data1[indices]
                data2 = data2[indices]
            
            ax4.scatter(data1, data2, alpha=0.5, s=1)
            ax4.set_xlabel(quant_types[0])
            ax4.set_ylabel(quant_types[1])
            ax4.set_title('Correlation Analysis')
            
            # Calculate correlation coefficient
            corr = np.corrcoef(data1, data2)[0, 1]
            ax4.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
                    transform=ax4.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'Need 2 quantization types\nfor correlation analysis', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save image
        output_file = output_path / f'{tensor_name}_quantization_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Quantization comparison chart saved: {output_file}")


def create_attention_analysis(tensor_data, output_path):
    """Create attention analysis chart"""
    attention_tensors = [t for t in tensor_data if t['metadata']['layer_type'] == 'attention']
    
    if not attention_tensors:
        print("No attention layer tensors found, skipping attention analysis")
        return
    
    # Group by tensor_name
    attention_groups = defaultdict(list)
    for tensor_info in attention_tensors:
        tensor_name = tensor_info['metadata']['tensor_name']
        attention_groups[tensor_name].append(tensor_info)
    
    # Create analysis chart for each attention tensor
    for tensor_name, group in attention_groups.items():
        if len(group) < 1:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Attention {tensor_name} Analysis', fontsize=16, fontweight='bold')
        
        # Collect data
        all_data = []
        quant_types = []
        for tensor_info in group:
            data = tensor_info['tensor'].flatten()
            all_data.append(data)
            quant_types.append(tensor_info['metadata']['quant_type'])
        
        # Distribution analysis
        ax1 = axes[0, 0]
        for i, (data, quant_type) in enumerate(zip(all_data, quant_types)):
            ax1.hist(data, bins=50, alpha=0.6, label=quant_type, density=True)
        ax1.set_title(f'{tensor_name} Value Distribution')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Heatmap (if 2D tensor)
        ax2 = axes[0, 1]
        if len(group) > 0:
            sample_tensor = group[0]['tensor']
            if len(sample_tensor.shape) == 2:
                im = ax2.imshow(sample_tensor, cmap='viridis', aspect='auto')
                ax2.set_title(f'{tensor_name} Heatmap')
                ax2.set_xlabel('Column')
                ax2.set_ylabel('Row')
                plt.colorbar(im, ax=ax2)
            else:
                # If high-dimensional tensor, take average
                if len(sample_tensor.shape) > 2:
                    sample_tensor = np.mean(sample_tensor, axis=tuple(range(2, len(sample_tensor.shape))))
                im = ax2.imshow(sample_tensor, cmap='viridis', aspect='auto')
                ax2.set_title(f'{tensor_name} Heatmap (Average)')
                ax2.set_xlabel('Column')
                ax2.set_ylabel('Row')
                plt.colorbar(im, ax=ax2)
        
        # Statistical information
        ax3 = axes[1, 0]
        stats_text = f"{tensor_name} Statistical Information:\n\n"
        for i, (tensor_info, quant_type) in enumerate(zip(group, quant_types)):
            stats = tensor_info['tensor_info']
            stats_text += f"Sample {i+1} ({quant_type}):\n"
            stats_text += f"  Shape: {stats['shape']}\n"
            stats_text += f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n"
            stats_text += f"  Mean: {stats['mean']:.4f}\n"
            stats_text += f"  Std Dev: {stats['std']:.4f}\n\n"
        
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # Time series analysis (if multiple samples)
        ax4 = axes[1, 1]
        if len(all_data) > 1:
            means = [np.mean(data) for data in all_data]
            stds = [np.std(data) for data in all_data]
            
            x = range(len(means))
            ax4.errorbar(x, means, yerr=stds, marker='o', capsize=5, capthick=2)
            ax4.set_title(f'{tensor_name} Statistical Information Changes')
            ax4.set_xlabel('Sample Index')
            ax4.set_ylabel('Mean ± Std Dev')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Need multiple samples\nfor time series analysis', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save image
        output_file = output_path / f'{tensor_name}_attention_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Attention analysis chart saved: {output_file}")


def generate_detailed_stats(tensor_data, output_path):
    """Generate detailed statistical report"""
    stats_file = output_path / 'detailed_tensor_stats.txt'
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("Tensor Data Detailed Statistical Report\n")
        f.write("=" * 60 + "\n\n")
        
        # Basic information
        f.write(f"Total files: {len(tensor_data)}\n")
        f.write(f"Analysis time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Statistics by quantization type
        quant_stats = defaultdict(list)
        for tensor_info in tensor_data:
            quant_type = tensor_info['metadata']['quant_type']
            quant_stats[quant_type].append(tensor_info)
        
        f.write("Quantization Type Statistics:\n")
        f.write("-" * 30 + "\n")
        for quant_type, group in quant_stats.items():
            f.write(f"\n{quant_type}:\n")
            f.write(f"  File count: {len(group)}\n")
            
            # Calculate statistical information
            all_values = []
            for tensor_info in group:
                values = tensor_info['tensor'].flatten()
                all_values.extend(values)
            
            if all_values:
                f.write(f"  Total values: {len(all_values)}\n")
                f.write(f"  Min value: {np.min(all_values):.6f}\n")
                f.write(f"  Max value: {np.max(all_values):.6f}\n")
                f.write(f"  Mean: {np.mean(all_values):.6f}\n")
                f.write(f"  Std dev: {np.std(all_values):.6f}\n")
                f.write(f"  Median: {np.median(all_values):.6f}\n")
        
        # Statistics by layer type
        layer_stats = defaultdict(list)
        for tensor_info in tensor_data:
            layer_type = tensor_info['metadata']['layer_type']
            layer_stats[layer_type].append(tensor_info)
        
        f.write("\n\nLayer Type Statistics:\n")
        f.write("-" * 30 + "\n")
        for layer_type, group in layer_stats.items():
            f.write(f"\n{layer_type}:\n")
            f.write(f"  File count: {len(group)}\n")
            
            # Group by tensor_name
            name_groups = defaultdict(list)
            for tensor_info in group:
                tensor_name = tensor_info['metadata']['tensor_name']
                name_groups[tensor_name].append(tensor_info)
            
            f.write(f"  Tensor names: {', '.join(name_groups.keys())}\n")
        
        # Statistics by operation type
        op_stats = defaultdict(list)
        for tensor_info in tensor_data:
            operation = tensor_info['metadata']['operation']
            op_stats[operation].append(tensor_info)
        
        f.write("\n\nOperation Type Statistics:\n")
        f.write("-" * 30 + "\n")
        for operation, group in op_stats.items():
            f.write(f"\n{operation}:\n")
            f.write(f"  File count: {len(group)}\n")
        
        # File detailed information
        f.write("\n\nFile Detailed Information:\n")
        f.write("-" * 30 + "\n")
        for i, tensor_info in enumerate(tensor_data, 1):
            metadata = tensor_info['metadata']
            tensor_info_data = tensor_info['tensor_info']
            
            f.write(f"\nFile {i}: {tensor_info['filename']}\n")
            f.write(f"  Layer type: {metadata['layer_type']}\n")
            f.write(f"  Operation: {metadata['operation']}\n")
            f.write(f"  Quantization type: {metadata['quant_type']}\n")
            f.write(f"  Tensor name: {metadata['tensor_name']}\n")
            f.write(f"  Layer index: {metadata.get('layer_idx', 'N/A')}\n")
            f.write(f"  Shape: {tensor_info_data['shape']}\n")
            f.write(f"  Data type: {tensor_info_data['dtype']}\n")
            f.write(f"  Value range: [{tensor_info_data['min']:.6f}, {tensor_info_data['max']:.6f}]\n")
            f.write(f"  Mean: {tensor_info_data['mean']:.6f}\n")
            f.write(f"  Std dev: {tensor_info_data['std']:.6f}\n")
    
    print(f"Detailed statistical report saved: {stats_file}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Quick Tensor Visualization Tool')
    parser.add_argument('--tensor_dir', type=str, default='./enhanced_tensor_logs',
                       help='Tensor file directory (default: ./enhanced_tensor_logs)')
    parser.add_argument('--output_dir', type=str, default='./draw',
                       help='Output directory (default: ./draw)')
    
    args = parser.parse_args()
    
    # Run visualization
    quick_visualize_enhanced(
        tensor_dir=args.tensor_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
