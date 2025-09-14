#!/usr/bin/env python3
"""
Multi-threaded Tensor Visualization Tool
Supports new data structure: bf16, mxfp8, mxfp4, hifp8 quantization types
Supports multi-dimensional comparison of Sample (0,1,2) and Layer (1-16)
Uses multi-threading to accelerate plotting process
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
        
        # Supported quantization types
        self.quant_types = ['bf16', 'mxfp8', 'mxfp4', 'hifp8']
        
        # Supported samples and layers
        self.samples = [0, 1, 2]
        self.layers = list(range(1, 17))  # 1-16 layers
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
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
        """Parse filename to extract quantization type, layer, sample and other information"""
        try:
            # Filename format: YYYYMMDD_HHMMSS_XXXX_iterXXX_layer_type_operation_quant_type_rankXX_sampleXXX_groupXXX_tensor_name.pt
            parts = filename.split('_')
            
            if len(parts) < 8:
                return None
            
            # Find quantization type
            quant_type = None
            for qtype in self.quant_types:
                if qtype in parts:
                    quant_type = qtype
                    break
            
            if not quant_type:
                return None
            
            # Find layer number
            layer_match = re.search(r'L(\d+)', filename)
            layer = int(layer_match.group(1)) if layer_match else None
            
            # Find sample
            sample_match = re.search(r'sample(\d+)', filename)
            sample = int(sample_match.group(1)) if sample_match else None
            
            # Find layer type
            layer_type = None
            if 'attention' in filename:
                layer_type = 'attention'
            elif 'linear' in filename:
                layer_type = 'linear'
            
            # Find operation type
            operation = None
            if 'forward' in filename:
                operation = 'forward'
            elif 'backward' in filename:
                operation = 'backward'
            
            # Find tensor name
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
    
    def load_tensor_data(self) -> Dict:
        """Load all tensor data"""
        print("Scanning tensor files...")
        
        data = {
            'files': [],
            'by_quant_type': {qtype: [] for qtype in self.quant_types},
            'by_sample': {sample: [] for sample in self.samples},
            'by_layer': {layer: [] for layer in self.layers},
            'by_layer_type': {'attention': [], 'linear': []},
            'statistics': {}
        }
        
        # Scan all tensor files
        for quant_type in self.quant_types:
            quant_dir = self.tensor_dir / quant_type
            if not quant_dir.exists():
                print(f"Warning: Quantization type directory does not exist: {quant_dir}")
                continue
            
            pt_files = list(quant_dir.glob('*.pt'))
            print(f"Found {len(pt_files)} {quant_type} files")
            
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
        
        print(f"Total loaded {len(data['files'])} tensor files")
        return data
    
    def load_tensor_values(self, file_info: Dict) -> Optional[np.ndarray]:
        """Load tensor values"""
        try:
            tensor = torch.load(file_info['file_path'], map_location='cpu')
            if isinstance(tensor, torch.Tensor):
                return tensor.numpy()
            return None
        except Exception as e:
            print(f"Failed to load tensor: {file_info['filename']}, error: {e}")
            return None
    
    def calculate_statistics(self, data: Dict) -> Dict:
        """Calculate statistics"""
        print("Calculating statistics...")
        
        stats = {
            'quant_type_stats': {},
            'sample_stats': {},
            'layer_stats': {},
            'layer_type_stats': {},
            'overall_stats': {}
        }
        
        # Statistics by quantization type
        for quant_type in self.quant_types:
            files = data['by_quant_type'][quant_type]
            if files:
                stats['quant_type_stats'][quant_type] = {
                    'count': len(files),
                    'layers': len(set(f['layer'] for f in files if f['layer'] is not None)),
                    'samples': len(set(f['sample'] for f in files if f['sample'] is not None)),
                    'layer_types': list(set(f['layer_type'] for f in files if f['layer_type']))
                }
        
        # Statistics by sample
        for sample in self.samples:
            files = data['by_sample'][sample]
            if files:
                stats['sample_stats'][sample] = {
                    'count': len(files),
                    'quant_types': list(set(f['quant_type'] for f in files)),
                    'layers': len(set(f['layer'] for f in files if f['layer'] is not None))
                }
        
        # Statistics by layer
        for layer in self.layers:
            files = data['by_layer'][layer]
            if files:
                stats['layer_stats'][layer] = {
                    'count': len(files),
                    'quant_types': list(set(f['quant_type'] for f in files)),
                    'samples': len(set(f['sample'] for f in files if f['sample'] is not None)),
                    'layer_types': list(set(f['layer_type'] for f in files if f['layer_type']))
                }
        
        # Statistics by layer type
        for layer_type in ['attention', 'linear']:
            files = data['by_layer_type'][layer_type]
            if files:
                stats['layer_type_stats'][layer_type] = {
                    'count': len(files),
                    'quant_types': list(set(f['quant_type'] for f in files)),
                    'layers': len(set(f['layer'] for f in files if f['layer'] is not None))
                }
        
        # Overall statistics
        stats['overall_stats'] = {
            'total_files': len(data['files']),
            'quant_types': len([q for q in self.quant_types if data['by_quant_type'][q]]),
            'samples': len([s for s in self.samples if data['by_sample'][s]]),
            'layers': len([l for l in self.layers if data['by_layer'][l]]),
            'layer_types': len([lt for lt in ['attention', 'linear'] if data['by_layer_type'][lt]])
        }
        
        return stats
    
    def plot_quantization_comparison(self, data: Dict, stats: Dict):
        """Plot quantization type comparison chart"""
        print("Plotting quantization type comparison chart...")
        
        # Prepare data
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
            print("No quantization type data to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Quantization Type Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. File count comparison
        ax1 = axes[0, 0]
        quant_types = [d['quant_type'] for d in quant_data]
        counts = [d['count'] for d in quant_data]
        bars1 = ax1.bar(quant_types, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('File Count by Quantization Type', fontweight='bold')
        ax1.set_ylabel('File Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars1, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Layer distribution
        ax2 = axes[0, 1]
        layers = [d['layers'] for d in quant_data]
        bars2 = ax2.bar(quant_types, layers, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_title('Layer Distribution by Quantization Type', fontweight='bold')
        ax2.set_ylabel('Layer Count')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, layer in zip(bars2, layers):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(layers)*0.01,
                    f'{layer}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Sample distribution
        ax3 = axes[1, 0]
        samples = [d['samples'] for d in quant_data]
        bars3 = ax3.bar(quant_types, samples, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax3.set_title('Sample Distribution by Quantization Type', fontweight='bold')
        ax3.set_ylabel('Sample Count')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, sample in zip(bars3, samples):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(samples)*0.01,
                    f'{sample}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Comprehensive comparison
        ax4 = axes[1, 1]
        x = np.arange(len(quant_types))
        width = 0.25
        
        bars4_1 = ax4.bar(x - width, counts, width, label='File Count', alpha=0.8)
        bars4_2 = ax4.bar(x, layers, width, label='Layer Count', alpha=0.8)
        bars4_3 = ax4.bar(x + width, samples, width, label='Sample Count', alpha=0.8)
        
        ax4.set_title('Comprehensive Comparison', fontweight='bold')
        ax4.set_ylabel('Count')
        ax4.set_xticks(x)
        ax4.set_xticklabels(quant_types, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.subdirs['quantization_analysis'] / 'quantization_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Quantization comparison chart saved: {self.subdirs['quantization_analysis'] / 'quantization_comparison.png'}")
    
    def plot_sample_analysis(self, data: Dict, stats: Dict):
        """Plot sample analysis chart"""
        print("Plotting sample analysis chart...")
        
        # Prepare data
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
            print("No sample data to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sample Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sample file count
        ax1 = axes[0, 0]
        samples = [d['sample'] for d in sample_data]
        counts = [d['count'] for d in sample_data]
        bars1 = ax1.bar(samples, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('File Count by Sample', fontweight='bold')
        ax1.set_xlabel('Sample Number')
        ax1.set_ylabel('File Count')
        
        for bar, count in zip(bars1, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Sample quantization type distribution
        ax2 = axes[0, 1]
        quant_types = [d['quant_types'] for d in sample_data]
        bars2 = ax2.bar(samples, quant_types, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title('Quantization Type Count by Sample', fontweight='bold')
        ax2.set_xlabel('Sample Number')
        ax2.set_ylabel('Quantization Type Count')
        
        for bar, qtype in zip(bars2, quant_types):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(quant_types)*0.01,
                    f'{qtype}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Sample layer distribution
        ax3 = axes[1, 0]
        layers = [d['layers'] for d in sample_data]
        bars3 = ax3.bar(samples, layers, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax3.set_title('Layer Distribution by Sample', fontweight='bold')
        ax3.set_xlabel('Sample Number')
        ax3.set_ylabel('Layer Count')
        
        for bar, layer in zip(bars3, layers):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(layers)*0.01,
                    f'{layer}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Sample distribution pie chart
        ax4 = axes[1, 1]
        ax4.pie(counts, labels=[f'Sample {s}' for s in samples], autopct='%1.1f%%', 
                colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax4.set_title('Sample Distribution Ratio', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.subdirs['sample_analysis'] / 'sample_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Sample analysis chart saved: {self.subdirs['sample_analysis'] / 'sample_analysis.png'}")
    
    def plot_layer_analysis(self, data: Dict, stats: Dict):
        """Plot layer analysis chart"""
        print("Plotting layer analysis chart...")
        
        # Prepare data
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
            print("No layer data to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Layer Analysis', fontsize=16, fontweight='bold')
        
        # 1. File count by layer
        ax1 = axes[0, 0]
        layers = [d['layer'] for d in layer_data]
        counts = [d['count'] for d in layer_data]
        bars1 = ax1.bar(layers, counts, color='skyblue', alpha=0.7)
        ax1.set_title('File Count by Layer', fontweight='bold')
        ax1.set_xlabel('Layer Number')
        ax1.set_ylabel('File Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Quantization type count by layer
        ax2 = axes[0, 1]
        quant_types = [d['quant_types'] for d in layer_data]
        bars2 = ax2.bar(layers, quant_types, color='lightcoral', alpha=0.7)
        ax2.set_title('Quantization Type Count by Layer', fontweight='bold')
        ax2.set_xlabel('Layer Number')
        ax2.set_ylabel('Quantization Type Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Sample count by layer
        ax3 = axes[1, 0]
        samples = [d['samples'] for d in layer_data]
        bars3 = ax3.bar(layers, samples, color='lightgreen', alpha=0.7)
        ax3.set_title('Sample Count by Layer', fontweight='bold')
        ax3.set_xlabel('Layer Number')
        ax3.set_ylabel('Sample Count')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Layer distribution heatmap
        ax4 = axes[1, 1]
        
        # Create layer-quantization type matrix
        layer_quant_matrix = np.zeros((len(self.layers), len(self.quant_types)))
        for i, layer in enumerate(self.layers):
            for j, quant_type in enumerate(self.quant_types):
                layer_files = [f for f in data['by_layer'][layer] if f['quant_type'] == quant_type]
                layer_quant_matrix[i, j] = len(layer_files)
        
        im = ax4.imshow(layer_quant_matrix, cmap='YlOrRd', aspect='auto')
        ax4.set_title('Layer-Quantization Type Distribution Heatmap', fontweight='bold')
        ax4.set_xlabel('Quantization Type')
        ax4.set_ylabel('Layer Number')
        ax4.set_xticks(range(len(self.quant_types)))
        ax4.set_xticklabels(self.quant_types, rotation=45)
        ax4.set_yticks(range(len(self.layers)))
        ax4.set_yticklabels(self.layers)
        
        # Add colorbar
        plt.colorbar(im, ax=ax4, label='File Count')
        
        plt.tight_layout()
        plt.savefig(self.subdirs['layer_analysis'] / 'layer_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Layer analysis chart saved: {self.subdirs['layer_analysis'] / 'layer_analysis.png'}")
    
    def plot_comprehensive_comparison(self, data: Dict, stats: Dict):
        """Plot comprehensive comparison chart"""
        print("Plotting comprehensive comparison chart...")
        
        # Create large figure
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Quantization type-sample matrix
        ax1 = fig.add_subplot(gs[0, 0])
        quant_sample_matrix = np.zeros((len(self.quant_types), len(self.samples)))
        for i, quant_type in enumerate(self.quant_types):
            for j, sample in enumerate(self.samples):
                files = [f for f in data['by_quant_type'][quant_type] if f['sample'] == sample]
                quant_sample_matrix[i, j] = len(files)
        
        im1 = ax1.imshow(quant_sample_matrix, cmap='Blues', aspect='auto')
        ax1.set_title('Quantization Type-Sample Distribution', fontweight='bold')
        ax1.set_xlabel('Sample Number')
        ax1.set_ylabel('Quantization Type')
        ax1.set_xticks(range(len(self.samples)))
        ax1.set_xticklabels(self.samples)
        ax1.set_yticks(range(len(self.quant_types)))
        ax1.set_yticklabels(self.quant_types)
        
        # Add value labels
        for i in range(len(self.quant_types)):
            for j in range(len(self.samples)):
                text = ax1.text(j, i, f'{int(quant_sample_matrix[i, j])}', 
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, label='File Count')
        
        # 2. Quantization type-layer matrix
        ax2 = fig.add_subplot(gs[0, 1])
        quant_layer_matrix = np.zeros((len(self.quant_types), len(self.layers)))
        for i, quant_type in enumerate(self.quant_types):
            for j, layer in enumerate(self.layers):
                files = [f for f in data['by_quant_type'][quant_type] if f['layer'] == layer]
                quant_layer_matrix[i, j] = len(files)
        
        im2 = ax2.imshow(quant_layer_matrix, cmap='Greens', aspect='auto')
        ax2.set_title('Quantization Type-Layer Distribution', fontweight='bold')
        ax2.set_xlabel('Layer Number')
        ax2.set_ylabel('Quantization Type')
        ax2.set_xticks(range(0, len(self.layers), 2))
        ax2.set_xticklabels(self.layers[::2])
        ax2.set_yticks(range(len(self.quant_types)))
        ax2.set_yticklabels(self.quant_types)
        
        plt.colorbar(im2, ax=ax2, label='File Count')
        
        # 3. Sample-layer matrix
        ax3 = fig.add_subplot(gs[0, 2])
        sample_layer_matrix = np.zeros((len(self.samples), len(self.layers)))
        for i, sample in enumerate(self.samples):
            for j, layer in enumerate(self.layers):
                files = [f for f in data['by_sample'][sample] if f['layer'] == layer]
                sample_layer_matrix[i, j] = len(files)
        
        im3 = ax3.imshow(sample_layer_matrix, cmap='Reds', aspect='auto')
        ax3.set_title('Sample-Layer Distribution', fontweight='bold')
        ax3.set_xlabel('Layer Number')
        ax3.set_ylabel('Sample Number')
        ax3.set_xticks(range(0, len(self.layers), 2))
        ax3.set_xticklabels(self.layers[::2])
        ax3.set_yticks(range(len(self.samples)))
        ax3.set_yticklabels(self.samples)
        
        plt.colorbar(im3, ax=ax3, label='File Count')
        
        # 4. Layer type distribution
        ax4 = fig.add_subplot(gs[0, 3])
        layer_types = ['attention', 'linear']
        layer_type_counts = [len(data['by_layer_type'][lt]) for lt in layer_types]
        bars4 = ax4.bar(layer_types, layer_type_counts, color=['#ff9999', '#66b3ff'])
        ax4.set_title('Layer Type Distribution', fontweight='bold')
        ax4.set_ylabel('File Count')
        
        for bar, count in zip(bars4, layer_type_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(layer_type_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Quantization type file count trend
        ax5 = fig.add_subplot(gs[1, :2])
        quant_counts = [len(data['by_quant_type'][qt]) for qt in self.quant_types]
        bars5 = ax5.bar(self.quant_types, quant_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax5.set_title('Quantization Type File Count Distribution', fontweight='bold')
        ax5.set_ylabel('File Count')
        ax5.tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars5, quant_counts):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(quant_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Sample file count trend
        ax6 = fig.add_subplot(gs[1, 2:])
        sample_counts = [len(data['by_sample'][s]) for s in self.samples]
        bars6 = ax6.bar(self.samples, sample_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax6.set_title('Sample File Count Distribution', fontweight='bold')
        ax6.set_xlabel('Sample Number')
        ax6.set_ylabel('File Count')
        
        for bar, count in zip(bars6, sample_counts):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 7. Layer file count trend
        ax7 = fig.add_subplot(gs[2, :])
        layer_counts = [len(data['by_layer'][l]) for l in self.layers]
        ax7.plot(self.layers, layer_counts, marker='o', linewidth=2, markersize=6, color='#2ca02c')
        ax7.set_title('File Count Trend by Layer', fontweight='bold')
        ax7.set_xlabel('Layer Number')
        ax7.set_ylabel('File Count')
        ax7.grid(True, alpha=0.3)
        
        # Add value labels
        for i, count in enumerate(layer_counts):
            if i % 2 == 0:  # Only show even layer labels to avoid overlap
                ax7.annotate(f'{count}', (self.layers[i], count), 
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.suptitle('Comprehensive Comparison Analysis', fontsize=20, fontweight='bold', y=0.98)
        plt.savefig(self.subdirs['comparison_analysis'] / 'comprehensive_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive comparison chart saved: {self.subdirs['comparison_analysis'] / 'comprehensive_comparison.png'}")
    
    def save_statistics_report(self, stats: Dict):
        """Save statistics report"""
        print("Saving statistics report...")
        
        report_path = self.subdirs['statistics'] / 'detailed_statistics_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Multi-threaded Tensor Visualization Statistics Report\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Files: {stats['overall_stats']['total_files']:,}\n")
            f.write(f"Quantization Types: {stats['overall_stats']['quant_types']}\n")
            f.write(f"Samples: {stats['overall_stats']['samples']}\n")
            f.write(f"Layers: {stats['overall_stats']['layers']}\n")
            f.write(f"Layer Types: {stats['overall_stats']['layer_types']}\n\n")
            
            # Quantization type statistics
            f.write("Quantization Type Statistics:\n")
            f.write("-" * 40 + "\n")
            for quant_type, stat in stats['quant_type_stats'].items():
                f.write(f"{quant_type}:\n")
                f.write(f"  File Count: {stat['count']:,}\n")
                f.write(f"  Layers: {stat['layers']}\n")
                f.write(f"  Samples: {stat['samples']}\n")
                f.write(f"  Layer Types: {', '.join(stat['layer_types'])}\n\n")
            
            # Sample statistics
            f.write("Sample Statistics:\n")
            f.write("-" * 40 + "\n")
            for sample, stat in stats['sample_stats'].items():
                f.write(f"Sample {sample}:\n")
                f.write(f"  File Count: {stat['count']:,}\n")
                f.write(f"  Quantization Types: {', '.join(stat['quant_types'])}\n")
                f.write(f"  Layers: {stat['layers']}\n\n")
            
            # Layer statistics
            f.write("Layer Statistics:\n")
            f.write("-" * 40 + "\n")
            for layer, stat in stats['layer_stats'].items():
                f.write(f"Layer {layer}:\n")
                f.write(f"  File Count: {stat['count']:,}\n")
                f.write(f"  Quantization Types: {', '.join(stat['quant_types'])}\n")
                f.write(f"  Samples: {stat['samples']}\n")
                f.write(f"  Layer Types: {', '.join(stat['layer_types'])}\n\n")
            
            # Layer type statistics
            f.write("Layer Type Statistics:\n")
            f.write("-" * 40 + "\n")
            for layer_type, stat in stats['layer_type_stats'].items():
                f.write(f"{layer_type}:\n")
                f.write(f"  File Count: {stat['count']:,}\n")
                f.write(f"  Quantization Types: {', '.join(stat['quant_types'])}\n")
                f.write(f"  Layers: {stat['layers']}\n\n")
        
        print(f"Statistics report saved: {report_path}")
    
    def run_visualization(self):
        """Run visualization"""
        print("Starting multi-threaded Tensor visualization...")
        print(f"Tensor directory: {self.tensor_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Max workers: {self.max_workers}")
        
        # Load data
        data = self.load_tensor_data()
        if not data['files']:
            print("Error: No tensor files found")
            return
        
        # Calculate statistics
        stats = self.calculate_statistics(data)
        
        # Use multi-threading to plot charts
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            futures = [
                executor.submit(self.plot_quantization_comparison, data, stats),
                executor.submit(self.plot_sample_analysis, data, stats),
                executor.submit(self.plot_layer_analysis, data, stats),
                executor.submit(self.plot_comprehensive_comparison, data, stats),
                executor.submit(self.save_statistics_report, stats)
            ]
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Task execution failed: {e}")
        
        print("\n" + "=" * 60)
        print("Multi-threaded Tensor visualization completed!")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print("Generated charts:")
        print(f"  - Quantization comparison: {self.subdirs['quantization_analysis'] / 'quantization_comparison.png'}")
        print(f"  - Sample analysis: {self.subdirs['sample_analysis'] / 'sample_analysis.png'}")
        print(f"  - Layer analysis: {self.subdirs['layer_analysis'] / 'layer_analysis.png'}")
        print(f"  - Comprehensive comparison: {self.subdirs['comparison_analysis'] / 'comprehensive_comparison.png'}")
        print(f"  - Statistics report: {self.subdirs['statistics'] / 'detailed_statistics_report.txt'}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Multi-threaded Tensor Visualization Tool')
    parser.add_argument('--tensor_dir', type=str, default='./enhanced_tensor_logs',
                       help='Tensor file directory')
    parser.add_argument('--output_dir', type=str, default='./draw',
                       help='Output directory')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum number of workers')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = MultiThreadedTensorVisualizer(
        tensor_dir=args.tensor_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    # Run visualization
    visualizer.run_visualization()

if __name__ == "__main__":
    main()
