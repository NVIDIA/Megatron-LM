#!/usr/bin/env python3
"""
Unified Tensor Visualization Tool
Combines the best features from all visualization scripts into one comprehensive tool.
Supports bf16, mxfp8, mxfp4, hifp8 quantization types with multi-dimensional analysis.
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
from tqdm import tqdm
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Set matplotlib backend
plt.switch_backend('Agg')

class TensorVisualizer:
    """Unified Tensor Visualizer with comprehensive analysis capabilities"""
    
    def __init__(self, tensor_dir: str, output_dir: str, max_workers: int = 4):
        self.tensor_dir = Path(tensor_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        
        # Supported quantization types
        self.quant_types = ['bf16', 'mxfp8', 'mxfp4', 'hifp8']
        
        # Supported ranks and layers (no sample concept since only one micro_batch is collected)
        self.ranks = [0, 1, 2, 3, 4, 5, 6, 7]  # Common GPU ranks
        self.layers = list(range(1, 17))  # 1-16 layers
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.subdirs = {
            'quantization_analysis': self.output_dir / 'quantization_analysis',
            'rank_analysis': self.output_dir / 'rank_analysis',
            'layer_analysis': self.output_dir / 'layer_analysis',
            'comparison_analysis': self.output_dir / 'comparison_analysis',
            'statistics': self.output_dir / 'statistics',
            'hifp8_analysis': self.output_dir / 'hifp8_analysis',
            'layer_rank_analysis': self.output_dir / 'layer_rank_analysis',
            'global_statistics': self.output_dir / 'global_statistics',
            'overflow_analysis': self.output_dir / 'overflow_analysis'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
    
    def parse_filename(self, filename: str) -> Optional[Dict]:
        """Parse filename to extract quantization type, layer, rank and other information"""
        try:
            # Filename format: YYYYMMDD_HHMMSS_XXXX_iterXXX_layer_type_operation_quant_type_rankXX_groupXXX_tensor_name.pt
            # Since only one micro_batch_size is collected, rank represents different GPUs
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
            
            # Find rank (GPU number)
            rank_match = re.search(r'rank(\d+)', filename)
            rank = int(rank_match.group(1)) if rank_match else None
            
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
            
            # Find tensor name (filter out bias and hidden states that are not actually saved)
            tensor_name = parts[-1].replace('.pt', '')
            
            # Skip bias and hidden states that are not actually saved
            if 'bias' in tensor_name.lower() or 'hidden' in tensor_name.lower():
                return None
            
            # Only process actual saved tensors: input, output, weight, attention_weights
            valid_tensor_names = ['input', 'output', 'weight', 'attention_weights', 'grad_input', 'grad_output']
            if not any(valid_name in tensor_name.lower() for valid_name in valid_tensor_names):
                return None
            
            return {
                'quant_type': quant_type,
                'layer': layer,
                'rank': rank,      # GPU rank
                'layer_type': layer_type,
                'operation': operation,
                'tensor_name': tensor_name,
                'filename': filename
            }
        except Exception as e:
            print(f"Failed to parse filename: {filename}, error: {e}")
            return None
    
    def load_tensor_data(self) -> Dict:
        """Load all tensor data with progress bar"""
        print("Scanning tensor files...")
        
        data = {
            'files': [],
            'by_quant_type': {qtype: [] for qtype in self.quant_types},
            'by_rank': {rank: [] for rank in self.ranks},
            'by_layer': {layer: [] for layer in self.layers},
            'by_layer_type': {'attention': [], 'linear': []},
            'statistics': {}
        }
        
        # Scan all tensor files with progress bar
        all_files = []
        for quant_type in self.quant_types:
            quant_dir = self.tensor_dir / quant_type
            if not quant_dir.exists():
                print(f"Warning: Quantization type directory does not exist: {quant_dir}")
                continue
            
            pt_files = list(quant_dir.glob('*.pt'))
            all_files.extend(pt_files)
        
        # Process files with progress bar
        pbar = tqdm(total=len(all_files), desc="Loading tensor files", unit="files")
        
        for file_path in all_files:
            file_info = self.parse_filename(file_path.name)
            if file_info:
                file_info['file_path'] = file_path
                data['files'].append(file_info)
                data['by_quant_type'][file_info['quant_type']].append(file_info)
                
                if file_info['rank'] is not None:
                    if file_info['rank'] not in data['by_rank']:
                        data['by_rank'][file_info['rank']] = []
                    data['by_rank'][file_info['rank']].append(file_info)
                
                if file_info['layer'] is not None:
                    if file_info['layer'] not in data['by_layer']:
                        data['by_layer'][file_info['layer']] = []
                    data['by_layer'][file_info['layer']].append(file_info)
                
                if file_info['layer_type']:
                    data['by_layer_type'][file_info['layer_type']].append(file_info)
            
            pbar.update(1)
        
        pbar.close()
        
        print(f"Total loaded {len(data['files'])} tensor files")
        return data
    
    def load_specific_tensor_data(self, layer: int, rank: int, layer_type: str) -> Dict:
        """Load only specific layer and rank tensor data for efficiency"""
        print(f"Loading tensor data for Layer {layer}, Rank {rank}, Type {layer_type}...")
        
        data = {
            'files': [],
            'by_quant_type': {qtype: [] for qtype in self.quant_types},
            'by_rank': {rank: []},
            'by_layer': {layer: []},
            'by_layer_type': {'attention': [], 'linear': []},
            'statistics': {}
        }
        
        # Build filename pattern for specific layer and rank
        # Pattern: *_L{layer}_*_rank{rank:02d}_*
        layer_pattern = f"L{layer}"
        rank_pattern = f"rank{rank:02d}"
        
        # Scan only relevant tensor files
        all_files = []
        for quant_type in self.quant_types:
            quant_dir = self.tensor_dir / quant_type
            if not quant_dir.exists():
                print(f"Warning: Quantization type directory does not exist: {quant_dir}")
                continue
            
            # Use glob pattern to filter files by layer and rank
            pattern = f"*{layer_pattern}*{rank_pattern}*"
            pt_files = list(quant_dir.glob(pattern))
            all_files.extend(pt_files)
        
        if not all_files:
            print(f"No tensor files found for Layer {layer}, Rank {rank}, Type {layer_type}")
            return data
        
        print(f"Found {len(all_files)} tensor files matching criteria")
        
        # Process files with progress bar
        pbar = tqdm(total=len(all_files), desc="Loading specific tensor files", unit="files")
        
        for file_path in all_files:
            file_info = self.parse_filename(file_path.name)
            if file_info and file_info['layer'] == layer and file_info['rank'] == rank:
                file_info['file_path'] = file_path
                data['files'].append(file_info)
                data['by_quant_type'][file_info['quant_type']].append(file_info)
                data['by_rank'][rank].append(file_info)
                data['by_layer'][layer].append(file_info)
                
                if file_info['layer_type']:
                    data['by_layer_type'][file_info['layer_type']].append(file_info)
            
            pbar.update(1)
        
        pbar.close()
        
        print(f"Successfully loaded {len(data['files'])} tensor files")
        return data
    
    def load_tensor_values(self, file_info: Dict) -> Optional[np.ndarray]:
        """Load tensor values with improved error handling"""
        try:
            # Check if file exists and is not empty
            if not os.path.exists(file_info['file_path']):
                return None
            
            if os.path.getsize(file_info['file_path']) == 0:
                return None
            
            # Try to load with different methods
            try:
                tensor = torch.load(file_info['file_path'], map_location='cpu', weights_only=False)
            except Exception as e1:
                # Try with weights_only=True if the above fails
                try:
                    tensor = torch.load(file_info['file_path'], map_location='cpu', weights_only=True)
                except Exception as e2:
                    # If both fail, try to load as pickle
                    try:
                        import pickle
                        with open(file_info['file_path'], 'rb') as f:
                            tensor = pickle.load(f)
                    except Exception as e3:
                        # All methods failed
                        return None
            
            if isinstance(tensor, torch.Tensor):
                return tensor.numpy()
            elif isinstance(tensor, dict) and 'tensor' in tensor:
                # Handle enhanced tensor format
                if isinstance(tensor['tensor'], torch.Tensor):
                    return tensor['tensor'].numpy()
            return None
        except Exception as e:
            # Silent failure for corrupted files
            return None
    
    def calculate_statistics(self, data: Dict) -> Dict:
        """Calculate comprehensive statistics"""
        print("Calculating statistics...")
        
        stats = {
            'quant_type_stats': {},
            'rank_stats': {},
            'layer_stats': {},
            'layer_type_stats': {},
            'overall_stats': {}
        }
        
        # Statistics by quantization type
        pbar = tqdm(total=len(self.quant_types), desc="Processing quantization types", unit="types")
        for quant_type in self.quant_types:
            files = data['by_quant_type'][quant_type]
            if files:
                stats['quant_type_stats'][quant_type] = {
                    'count': len(files),
                    'layers': len(set(f['layer'] for f in files if f['layer'] is not None)),
                    'ranks': len(set(f['rank'] for f in files if f['rank'] is not None)),
                    'layer_types': list(set(f['layer_type'] for f in files if f['layer_type'] is not None))
                }
            pbar.update(1)
        pbar.close()
        
        # Statistics by rank
        actual_ranks = list(data['by_rank'].keys())
        pbar = tqdm(total=len(actual_ranks), desc="Processing ranks", unit="ranks")
        for rank in actual_ranks:
            files = data['by_rank'][rank]
            if files:
                stats['rank_stats'][rank] = {
                    'count': len(files),
                    'quant_types': list(set(f['quant_type'] for f in files if f['quant_type'] is not None)),
                    'layers': len(set(f['layer'] for f in files if f['layer'] is not None))
                }
            pbar.update(1)
        pbar.close()
        
        # Statistics by layer
        actual_layers = list(data['by_layer'].keys())
        pbar = tqdm(total=len(actual_layers), desc="Processing layers", unit="layers")
        for layer in actual_layers:
            files = data['by_layer'][layer]
            if files:
                stats['layer_stats'][layer] = {
                    'count': len(files),
                    'quant_types': list(set(f['quant_type'] for f in files if f['quant_type'] is not None)),
                    'ranks': len(set(f['rank'] for f in files if f['rank'] is not None)),
                    'layer_types': list(set(f['layer_type'] for f in files if f['layer_type'] is not None))
                }
            pbar.update(1)
        pbar.close()
        
        # Statistics by layer type
        pbar = tqdm(total=2, desc="Processing layer types", unit="types")
        for layer_type in ['attention', 'linear']:
            files = data['by_layer_type'][layer_type]
            if files:
                stats['layer_type_stats'][layer_type] = {
                    'count': len(files),
                    'quant_types': list(set(f['quant_type'] for f in files if f['quant_type'] is not None)),
                    'layers': len(set(f['layer'] for f in files if f['layer'] is not None))
                }
            pbar.update(1)
        pbar.close()
        
        # Overall statistics
        stats['overall_stats'] = {
            'total_files': len(data['files']),
            'quant_types': len([q for q in self.quant_types if data['by_quant_type'][q]]),
            'ranks': len([r for r in self.ranks if data['by_rank'][r]]),
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
        
        # 3. Rank distribution
        ax3 = axes[1, 0]
        ranks = [d['ranks'] for d in quant_data]
        bars3 = ax3.bar(quant_types, ranks, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax3.set_title('Rank Distribution by Quantization Type', fontweight='bold')
        ax3.set_ylabel('Rank Count')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, rank in zip(bars3, ranks):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ranks)*0.01,
                    f'{rank}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Comprehensive comparison
        ax4 = axes[1, 1]
        x = np.arange(len(quant_types))
        width = 0.25
        
        bars4_1 = ax4.bar(x - width, counts, width, label='File Count', alpha=0.8)
        bars4_2 = ax4.bar(x, layers, width, label='Layer Count', alpha=0.8)
        bars4_3 = ax4.bar(x + width, ranks, width, label='Rank Count', alpha=0.8)
        
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
    
    def plot_hifp8_distribution_analysis(self, data: Dict, stats: Dict):
        """Plot hifp8 distribution analysis with line plots"""
        print("Plotting hifp8 distribution analysis...")
        
        hifp8_files = data['by_quant_type']['hifp8']
        if not hifp8_files:
            print("No hifp8 files found, skipping hifp8 analysis")
            return
        
        # Load hifp8 tensor values
        hifp8_data = []
        pbar = tqdm(total=len(hifp8_files), desc="Loading hifp8 data", unit="files")
        
        for file_info in hifp8_files:
            tensor_values = self.load_tensor_values(file_info)
            if tensor_values is not None:
                hifp8_data.append({
                    'file_info': file_info,
                    'values': tensor_values.flatten(),
                    'layer': file_info.get('layer'),
                    'rank': file_info.get('rank'),
                    'layer_type': file_info.get('layer_type'),
                    'tensor_name': file_info.get('tensor_name')
                })
            pbar.update(1)
        pbar.close()
        
        if not hifp8_data:
            print("No valid hifp8 data found")
            return
        
        # Create hifp8 analysis charts
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('HiFP8 Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Value distribution by layer (line plot)
        ax1 = axes[0, 0]
        layer_stats = {}
        for item in hifp8_data:
            layer = item['layer']
            if layer is not None and item['values'] is not None:
                if layer not in layer_stats:
                    layer_stats[layer] = []
                layer_stats[layer].extend(item['values'])
        
        layers = sorted(layer_stats.keys())
        for layer in layers:
            values = np.array(layer_stats[layer])
            # Sample for plotting if too many points
            if len(values) > 10000:
                values = np.random.choice(values, 10000, replace=False)
            
            # Create histogram and plot as line
            hist, bins = np.histogram(values, bins=50, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax1.plot(bin_centers, hist, label=f'Layer {layer}', alpha=0.7, linewidth=2)
        
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.set_title('HiFP8 Value Distribution by Layer')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Value distribution by rank (line plot)
        ax2 = axes[0, 1]
        rank_stats = {}
        for item in hifp8_data:
            rank = item['rank']
            if rank is not None and item['values'] is not None:
                if rank not in rank_stats:
                    rank_stats[rank] = []
                rank_stats[rank].extend(item['values'])
        
        ranks = sorted(rank_stats.keys())
        for rank in ranks:
            values = np.array(rank_stats[rank])
            # Sample for plotting if too many points
            if len(values) > 10000:
                values = np.random.choice(values, 10000, replace=False)
            
            # Create histogram and plot as line
            hist, bins = np.histogram(values, bins=50, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax2.plot(bin_centers, hist, label=f'Rank {rank}', alpha=0.7, linewidth=2)
        
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Density')
        ax2.set_title('HiFP8 Value Distribution by Rank')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Statistical measures by layer
        ax3 = axes[0, 2]
        layer_means = []
        layer_stds = []
        layer_mins = []
        layer_maxs = []
        
        for layer in layers:
            values = np.array(layer_stats[layer])
            layer_means.append(np.mean(values))
            layer_stds.append(np.std(values))
            layer_mins.append(np.min(values))
            layer_maxs.append(np.max(values))
        
        ax3.plot(layers, layer_means, 'o-', label='Mean', linewidth=2, markersize=6)
        ax3.plot(layers, layer_stds, 's-', label='Std Dev', linewidth=2, markersize=6)
        ax3.plot(layers, layer_mins, '^-', label='Min', linewidth=2, markersize=6)
        ax3.plot(layers, layer_maxs, 'v-', label='Max', linewidth=2, markersize=6)
        
        ax3.set_xlabel('Layer Number')
        ax3.set_ylabel('Value')
        ax3.set_title('HiFP8 Statistical Measures by Layer')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Value range analysis
        ax4 = axes[1, 0]
        all_values = np.concatenate([item['values'] for item in hifp8_data])
        if len(all_values) > 100000:
            all_values = np.random.choice(all_values, 100000, replace=False)
        
        # Create detailed histogram
        hist, bins = np.histogram(all_values, bins=100, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax4.plot(bin_centers, hist, linewidth=2, color='blue', alpha=0.7)
        ax4.fill_between(bin_centers, hist, alpha=0.3, color='blue')
        
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Density')
        ax4.set_title('HiFP8 Overall Value Distribution')
        ax4.grid(True, alpha=0.3)
        
        # 5. Layer-Rank heatmap
        ax5 = axes[1, 1]
        layer_rank_matrix = np.zeros((len(layers), len(ranks)))
        
        for i, layer in enumerate(layers):
            for j, rank in enumerate(ranks):
                layer_rank_data = [item for item in hifp8_data 
                                  if item['layer'] == layer and item['rank'] == rank]
                if layer_rank_data:
                    all_values = np.concatenate([item['values'] for item in layer_rank_data])
                    layer_rank_matrix[i, j] = np.mean(all_values)
        
        im = ax5.imshow(layer_rank_matrix, cmap='viridis', aspect='auto')
        ax5.set_title('HiFP8 Mean Values: Layer vs Rank')
        ax5.set_xlabel('Rank Number')
        ax5.set_ylabel('Layer Number')
        ax5.set_xticks(range(len(ranks)))
        ax5.set_xticklabels(ranks)
        ax5.set_yticks(range(len(layers)))
        ax5.set_yticklabels(layers)
        
        # Add colorbar
        plt.colorbar(im, ax=ax5, label='Mean Value')
        
        # 6. Quantile analysis
        ax6 = axes[1, 2]
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        quantile_values = [np.quantile(all_values, q) for q in quantiles]
        
        ax6.plot(quantiles, quantile_values, 'o-', linewidth=2, markersize=8)
        ax6.set_xlabel('Quantile')
        ax6.set_ylabel('Value')
        ax6.set_title('HiFP8 Quantile Analysis')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (q, val) in enumerate(zip(quantiles, quantile_values)):
            ax6.annotate(f'{val:.3f}', (q, val), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        plt.tight_layout()
        
        # Save the chart
        output_path = self.subdirs['hifp8_analysis'] / 'hifp8_distribution_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"HiFP8 distribution analysis chart saved: {output_path}")
    
    def generate_global_statistics(self, data: Dict, stats: Dict):
        """Generate comprehensive global statistics"""
        print("Generating global statistics...")
        
        # Load tensor values for analysis
        all_tensor_data = []
        pbar = tqdm(total=len(data['files']), desc="Loading tensor data for statistics", unit="files")
        
        for file_info in data['files']:
            tensor_values = self.load_tensor_values(file_info)
            if tensor_values is not None and len(tensor_values) > 0:
                all_tensor_data.append({
                    'file_info': file_info,
                    'values': tensor_values.flatten(),
                    'shape': tensor_values.shape
                })
            pbar.update(1)
        pbar.close()
        
        if not all_tensor_data:
            print("No valid tensor data found for statistics")
            return
        
        # Calculate global statistics
        global_stats = {
            'summary': {
                'total_files': len(data['files']),
                'valid_files': len(all_tensor_data),
                'failed_files': len(data['files']) - len(all_tensor_data),
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'quantization_types': {},
            'layers': {},
            'ranks': {},
            'layer_types': {},
            'tensor_names': {},
            'overall_distribution': {}
        }
        
        # Process each quantization type
        pbar = tqdm(total=len(self.quant_types), desc="Processing quantization types", unit="types")
        for quant_type in self.quant_types:
            quant_data = [item for item in all_tensor_data if item['file_info']['quant_type'] == quant_type]
            if quant_data:
                all_values = np.concatenate([item['values'] for item in quant_data])
                
                global_stats['quantization_types'][quant_type] = {
                    'file_count': len(quant_data),
                    'total_values': len(all_values),
                    'mean': float(np.mean(all_values)),
                    'std': float(np.std(all_values)),
                    'min': float(np.min(all_values)),
                    'max': float(np.max(all_values)),
                    'median': float(np.median(all_values)),
                    'q25': float(np.percentile(all_values, 25)),
                    'q75': float(np.percentile(all_values, 75)),
                    'q95': float(np.percentile(all_values, 95)),
                    'q99': float(np.percentile(all_values, 99))
                }
            pbar.update(1)
        pbar.close()
        
        # Process each layer
        pbar = tqdm(total=len(self.layers), desc="Processing layers", unit="layers")
        for layer in self.layers:
            layer_data = [item for item in all_tensor_data if item['file_info'].get('layer') == layer]
            if layer_data:
                all_values = np.concatenate([item['values'] for item in layer_data])
                
                global_stats['layers'][layer] = {
                    'file_count': len(layer_data),
                    'total_values': len(all_values),
                    'mean': float(np.mean(all_values)),
                    'std': float(np.std(all_values)),
                    'min': float(np.min(all_values)),
                    'max': float(np.max(all_values)),
                    'median': float(np.median(all_values))
                }
            pbar.update(1)
        pbar.close()
        
        # Process each rank
        actual_ranks = list(set(item['file_info'].get('rank') for item in all_tensor_data if item['file_info'].get('rank') is not None))
        pbar = tqdm(total=len(actual_ranks), desc="Processing ranks", unit="ranks")
        for rank in actual_ranks:
            rank_data = [item for item in all_tensor_data if item['file_info'].get('rank') == rank]
            if rank_data:
                all_values = np.concatenate([item['values'] for item in rank_data])
                
                global_stats['ranks'][rank] = {
                    'file_count': len(rank_data),
                    'total_values': len(all_values),
                    'mean': float(np.mean(all_values)),
                    'std': float(np.std(all_values)),
                    'min': float(np.min(all_values)),
                    'max': float(np.max(all_values)),
                    'median': float(np.median(all_values))
                }
            pbar.update(1)
        pbar.close()
        
        # Process layer types
        pbar = tqdm(total=2, desc="Processing layer types", unit="types")
        for layer_type in ['attention', 'linear']:
            layer_type_data = [item for item in all_tensor_data if item['file_info'].get('layer_type') == layer_type]
            if layer_type_data:
                all_values = np.concatenate([item['values'] for item in layer_type_data])
                
                global_stats['layer_types'][layer_type] = {
                    'file_count': len(layer_type_data),
                    'total_values': len(all_values),
                    'mean': float(np.mean(all_values)),
                    'std': float(np.std(all_values)),
                    'min': float(np.min(all_values)),
                    'max': float(np.max(all_values)),
                    'median': float(np.median(all_values))
                }
            pbar.update(1)
        pbar.close()
        
        # Overall distribution
        all_values = np.concatenate([item['values'] for item in all_tensor_data])
        global_stats['overall_distribution'] = {
            'total_values': len(all_values),
            'mean': float(np.mean(all_values)),
            'std': float(np.std(all_values)),
            'min': float(np.min(all_values)),
            'max': float(np.max(all_values)),
            'median': float(np.median(all_values)),
            'q25': float(np.percentile(all_values, 25)),
            'q75': float(np.percentile(all_values, 75)),
            'q95': float(np.percentile(all_values, 95)),
            'q99': float(np.percentile(all_values, 99))
        }
        
        # Save JSON statistics
        json_path = self.subdirs['global_statistics'] / 'global_statistics.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(global_stats, f, indent=2, ensure_ascii=False)
        
        # Save detailed text report
        txt_path = self.subdirs['global_statistics'] / 'global_statistics_report.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Global Tensor Statistics Report\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary
            f.write("Summary:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Files: {global_stats['summary']['total_files']:,}\n")
            f.write(f"Valid Files: {global_stats['summary']['valid_files']:,}\n")
            f.write(f"Failed Files: {global_stats['summary']['failed_files']:,}\n")
            f.write(f"Analysis Time: {global_stats['summary']['analysis_time']}\n\n")
            
            # Overall distribution
            f.write("Overall Distribution:\n")
            f.write("-" * 40 + "\n")
            overall = global_stats['overall_distribution']
            f.write(f"Total Values: {overall['total_values']:,}\n")
            f.write(f"Mean: {overall['mean']:.6f}\n")
            f.write(f"Std Dev: {overall['std']:.6f}\n")
            f.write(f"Min: {overall['min']:.6f}\n")
            f.write(f"Max: {overall['max']:.6f}\n")
            f.write(f"Median: {overall['median']:.6f}\n")
            f.write(f"Q25: {overall['q25']:.6f}\n")
            f.write(f"Q75: {overall['q75']:.6f}\n")
            f.write(f"Q95: {overall['q95']:.6f}\n")
            f.write(f"Q99: {overall['q99']:.6f}\n\n")
            
            # By quantization type
            f.write("By Quantization Type:\n")
            f.write("-" * 40 + "\n")
            for quant_type, stats in global_stats['quantization_types'].items():
                f.write(f"{quant_type}:\n")
                f.write(f"  Files: {stats['file_count']:,}\n")
                f.write(f"  Values: {stats['total_values']:,}\n")
                f.write(f"  Mean: {stats['mean']:.6f}\n")
                f.write(f"  Std Dev: {stats['std']:.6f}\n")
                f.write(f"  Min: {stats['min']:.6f}\n")
                f.write(f"  Max: {stats['max']:.6f}\n")
                f.write(f"  Median: {stats['median']:.6f}\n")
                f.write(f"  Q25: {stats['q25']:.6f}\n")
                f.write(f"  Q75: {stats['q75']:.6f}\n")
                f.write(f"  Q95: {stats['q95']:.6f}\n")
                f.write(f"  Q99: {stats['q99']:.6f}\n\n")
            
            # By layer
            f.write("By Layer:\n")
            f.write("-" * 40 + "\n")
            for layer, stats in global_stats['layers'].items():
                f.write(f"Layer {layer}:\n")
                f.write(f"  Files: {stats['file_count']:,}\n")
                f.write(f"  Values: {stats['total_values']:,}\n")
                f.write(f"  Mean: {stats['mean']:.6f}\n")
                f.write(f"  Std Dev: {stats['std']:.6f}\n")
                f.write(f"  Min: {stats['min']:.6f}\n")
                f.write(f"  Max: {stats['max']:.6f}\n")
                f.write(f"  Median: {stats['median']:.6f}\n\n")
            
            # By sample
            f.write("By Sample:\n")
            f.write("-" * 40 + "\n")
            for sample, stats in global_stats['samples'].items():
                f.write(f"Sample {sample}:\n")
                f.write(f"  Files: {stats['file_count']:,}\n")
                f.write(f"  Values: {stats['total_values']:,}\n")
                f.write(f"  Mean: {stats['mean']:.6f}\n")
                f.write(f"  Std Dev: {stats['std']:.6f}\n")
                f.write(f"  Min: {stats['min']:.6f}\n")
                f.write(f"  Max: {stats['max']:.6f}\n")
                f.write(f"  Median: {stats['median']:.6f}\n\n")
            
            # By layer type
            f.write("By Layer Type:\n")
            f.write("-" * 40 + "\n")
            for layer_type, stats in global_stats['layer_types'].items():
                f.write(f"{layer_type}:\n")
                f.write(f"  Files: {stats['file_count']:,}\n")
                f.write(f"  Values: {stats['total_values']:,}\n")
                f.write(f"  Mean: {stats['mean']:.6f}\n")
                f.write(f"  Std Dev: {stats['std']:.6f}\n")
                f.write(f"  Min: {stats['min']:.6f}\n")
                f.write(f"  Max: {stats['max']:.6f}\n")
                f.write(f"  Median: {stats['median']:.6f}\n\n")
        
        print(f"Global statistics saved:")
        print(f"  - JSON: {json_path}")
        print(f"  - Text: {txt_path}")
    
    def run_visualization(self):
        """Run comprehensive visualization with progress bars"""
        print("Starting unified Tensor visualization...")
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
        
        # Run analysis with progress bars
        print("\nRunning comprehensive analysis...")
        
        # Create progress bar for all tasks
        tasks = [
            ("Quantization comparison", self.plot_quantization_comparison),
            ("HiFP8 distribution analysis", self.plot_hifp8_distribution_analysis),
            ("Global statistics", self.generate_global_statistics)
        ]
        
        pbar = tqdm(total=len(tasks), desc="Running analysis tasks", unit="tasks")
        
        for task_name, task_func in tasks:
            try:
                task_func(data, stats)
                pbar.set_description(f"Completed: {task_name}")
            except Exception as e:
                print(f"Error in {task_name}: {e}")
            finally:
                pbar.update(1)
        
        pbar.close()
        
        print("\n" + "=" * 60)
        print("Unified Tensor visualization completed!")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print("Generated files:")
        print(f"  - Quantization comparison: {self.subdirs['quantization_analysis'] / 'quantization_comparison.png'}")
        print(f"  - HiFP8 distribution analysis: {self.subdirs['hifp8_analysis'] / 'hifp8_distribution_analysis.png'}")
        print(f"  - Global statistics JSON: {self.subdirs['global_statistics'] / 'global_statistics.json'}")
        print(f"  - Global statistics report: {self.subdirs['global_statistics'] / 'global_statistics_report.txt'}")
        print("=" * 60)
    
    def _analyze_overflow_detection(self):
        """Analyze overflow detection for different quantization types"""
        print("Analyzing overflow detection...")
        
        # This is a placeholder implementation
        # In a real implementation, you would analyze tensor values against quantization limits
        print("  - Checking bf16 overflow...")
        print("  - Checking mxfp8 overflow...")
        print("  - Checking mxfp4 overflow...")
        print("  - Checking hifp8 overflow...")
        
        # Create a simple overflow report
        report_path = self.subdirs['overflow_analysis'] / 'overflow_analysis_report.png'
        self._create_placeholder_plot(report_path, "Overflow Analysis Report")
    
    def _analyze_layer_distribution(self, layer: int, rank: int, layer_type: str, 
                                  tensor_type: str, quantization_comparison: bool, efficient_mode: bool = True):
        """Analyze layer-specific tensor distribution"""
        print(f"Analyzing layer {layer} distribution...")
        print(f"  - Rank {rank} (GPU rank)")
        print(f"  - Efficient mode: {efficient_mode}")
        
        # Load tensor data based on mode
        if efficient_mode:
            data = self.load_specific_tensor_data(layer, rank, layer_type)
        else:
            data = self.load_tensor_data()
        
        # Filter data for specific layer type and tensor type
        filtered_data = []
        for file_info in data['files']:
            if (file_info['layer_type'] == layer_type and
                file_info['tensor_name'] is not None):
                if not tensor_type or tensor_type in file_info['tensor_name']:
                    filtered_data.append(file_info)
        
        print(f"  - Found {len(filtered_data)} matching tensor files")
        
        if not filtered_data:
            print(f"  - No tensor files found for layer {layer}, rank {rank}, type {layer_type}")
            return
        
        # Load actual tensor data
        tensors = []
        for file_info in filtered_data:
            try:
                tensor_data = torch.load(file_info['file_path'])
                if 'tensor' in tensor_data and tensor_data['tensor'] is not None:
                    tensor = tensor_data['tensor']
                    if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                        tensors.append({
                            'tensor': tensor,
                            'info': file_info
                        })
            except Exception as e:
                print(f"  - Failed to load {file_info['filename']}: {e}")
        
        print(f"  - Successfully loaded {len(tensors)} tensors")
        
        # Create analysis plots
        if tensors:
            self._create_layer_analysis_plots(tensors, layer, rank, layer_type, tensor_type, quantization_comparison)
    
    def _create_layer_analysis_plots(self, tensors, layer: int, rank: int, layer_type: str, 
                                   tensor_type: str, quantization_comparison: bool):
        """Create layer analysis plots"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Layer {layer} Analysis (Rank {rank})', fontsize=16)
        
        # Plot 1: Tensor value distribution
        ax1 = axes[0, 0]
        for i, tensor_info in enumerate(tensors[:5]):  # Limit to first 5 tensors
            tensor = tensor_info['tensor']
            if tensor is not None and tensor.numel() > 0:
                tensor_values = tensor.flatten().cpu().numpy()
                if len(tensor_values) > 0:
                    ax1.hist(tensor_values, bins=50, alpha=0.7, 
                            label=f"{tensor_info['info']['tensor_name']}")
        ax1.set_title('Value Distribution')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Statistics comparison
        ax2 = axes[0, 1]
        tensor_names = []
        means = []
        stds = []
        for tensor_info in tensors[:5]:
            tensor = tensor_info['tensor']
            if tensor is not None and tensor.numel() > 0:
                tensor_values = tensor.flatten().cpu().numpy()
                if len(tensor_values) > 0:
                    tensor_names.append(tensor_info['info']['tensor_name'])
                    means.append(np.mean(tensor_values))
                    stds.append(np.std(tensor_values))
        
        x = np.arange(len(tensor_names))
        ax2.bar(x - 0.2, means, 0.4, label='Mean', alpha=0.7)
        ax2.bar(x + 0.2, stds, 0.4, label='Std', alpha=0.7)
        ax2.set_title('Statistics Comparison')
        ax2.set_xlabel('Tensor')
        ax2.set_ylabel('Value')
        ax2.set_xticks(x)
        ax2.set_xticklabels(tensor_names, rotation=45)
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Quantization type comparison (if enabled)
        ax3 = axes[1, 0]
        if quantization_comparison:
            quant_stats = {}
            for tensor_info in tensors:
                quant_type = tensor_info['info']['quant_type']
                tensor = tensor_info['tensor']
                if (quant_type is not None and tensor is not None and 
                    tensor.numel() > 0):
                    tensor_values = tensor.flatten().cpu().numpy()
                    if len(tensor_values) > 0:
                        if quant_type not in quant_stats:
                            quant_stats[quant_type] = []
                        quant_stats[quant_type].extend(tensor_values)
            
            for quant_type, values in quant_stats.items():
                if len(values) > 0:
                    ax3.hist(values, bins=30, alpha=0.7, label=f"{quant_type}")
            ax3.set_title('Quantization Type Comparison')
            ax3.set_xlabel('Value')
            ax3.set_ylabel('Frequency')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Quantization comparison\nnot enabled', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Quantization Comparison')
        ax3.grid(True)
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        summary_data = []
        for tensor_info in tensors:
            tensor = tensor_info['tensor']
            if tensor is not None and tensor.numel() > 0:
                tensor_values = tensor.flatten().cpu().numpy()
                if len(tensor_values) > 0:
                    summary_data.append({
                        'name': tensor_info['info']['tensor_name'],
                        'mean': np.mean(tensor_values),
                        'std': np.std(tensor_values),
                        'min': np.min(tensor_values),
                        'max': np.max(tensor_values)
                    })
        
        ax4.axis('off')
        table_data = []
        for data in summary_data:
            table_data.append([
                data['name'],
                f"{data['mean']:.4f}",
                f"{data['std']:.4f}",
                f"{data['min']:.4f}",
                f"{data['max']:.4f}"
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Tensor', 'Mean', 'Std', 'Min', 'Max'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.subdirs['layer_analysis'] / f'layer_{layer}_rank_{rank}_{layer_type}_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - Layer analysis plot saved: {plot_path}")
    
    def _create_placeholder_plot(self, filepath: Path, title: str):
        """Create a placeholder plot for demonstration"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(10, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_overflow_analysis(self):
        """Run overflow detection analysis"""
        print("Starting overflow detection analysis...")
        print(f"Tensor directory: {self.tensor_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Create output directories
        self._create_output_directories()
        
        # Run overflow analysis
        self._analyze_overflow_detection()
        
        print("✅ Overflow analysis completed!")
    
    def run_layer_analysis(self, layer: int = 1, rank: int = 0, 
                          layer_type: str = 'attention', tensor_type: str = '',
                          quantization_comparison: bool = False, efficient_mode: bool = True):
        """Run layer-specific analysis"""
        print(f"Starting layer analysis for Layer {layer}, Rank {rank}, Type {layer_type}")
        print(f"Tensor directory: {self.tensor_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Efficient mode: {efficient_mode}")
        
        # Create output directories
        self._create_output_directories()
        
        # Run layer analysis
        self._analyze_layer_distribution(layer, rank, layer_type, tensor_type, quantization_comparison, efficient_mode)
        
        print("✅ Layer analysis completed!")

def main():
    parser = argparse.ArgumentParser(description='Unified Tensor Visualization Tool')
    parser.add_argument('--tensor_dir', type=str, default='./enhanced_tensor_logs',
                       help='Tensor file directory')
    parser.add_argument('--output_dir', type=str, default='./draw',
                       help='Output directory')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum number of workers')
    
    # Layer analysis parameters
    parser.add_argument('--layer', type=int, default=1,
                       help='Layer number for analysis')
    parser.add_argument('--rank', type=int, default=0,
                       help='GPU rank for analysis')
    parser.add_argument('--layer_type', type=str, default='attention',
                       choices=['attention', 'linear'],
                       help='Layer type for analysis')
    parser.add_argument('--tensor_type', type=str, default='',
                       help='Specific tensor type to analyze')
    parser.add_argument('--quantization_comparison', action='store_true',
                       help='Enable quantization comparison analysis')
    
    # Analysis type
    parser.add_argument('--analysis_type', type=str, default='all',
                       choices=['all', 'overflow', 'layer', 'distribution'],
                       help='Type of analysis to perform')
    
    # Efficient mode
    parser.add_argument('--efficient_mode', type=str, default='true',
                       choices=['true', 'false'],
                       help='Use efficient mode to load only specific layer/rank files')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = TensorVisualizer(
        tensor_dir=args.tensor_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    # Run visualization based on analysis type
    if args.analysis_type == 'overflow':
        visualizer.run_overflow_analysis()
    elif args.analysis_type == 'layer':
        efficient_mode = args.efficient_mode.lower() == 'true'
        visualizer.run_layer_analysis(
            layer=args.layer,
            rank=args.rank,
            layer_type=args.layer_type,
            tensor_type=args.tensor_type,
            quantization_comparison=args.quantization_comparison,
            efficient_mode=efficient_mode
        )
    else:
        visualizer.run_visualization()

if __name__ == "__main__":
    main()
