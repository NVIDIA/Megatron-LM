#!/usr/bin/env python3
"""
Enhanced Multi-threaded Tensor Visualization Tool
Supports new data structure: bf16, mxfp8, mxfp4, hifp8 quantization types
Supports multi-dimensional comparison of Sample (0,1,2) and Layer (1-16)
Uses multi-threading to accelerate plotting process
Includes tqdm progress bars and advanced analysis features
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

class EnhancedMultiThreadedTensorVisualizer:
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
            'statistics': self.output_dir / 'statistics',
            'hifp8_analysis': self.output_dir / 'hifp8_analysis',
            'layer_sample_analysis': self.output_dir / 'layer_sample_analysis',
            'global_statistics': self.output_dir / 'global_statistics'
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
        """Load all tensor data with progress bar"""
        print("Scanning tensor files...")
        
        data = {
            'files': [],
            'by_quant_type': {qtype: [] for qtype in self.quant_types},
            'by_sample': {sample: [] for sample in self.samples},
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
                
                if file_info['sample'] is not None:
                    data['by_sample'][file_info['sample']].append(file_info)
                
                if file_info['layer'] is not None:
                    data['by_layer'][file_info['layer']].append(file_info)
                
                if file_info['layer_type']:
                    data['by_layer_type'][file_info['layer_type']].append(file_info)
            
            pbar.update(1)
        
        pbar.close()
        
        print(f"Total loaded {len(data['files'])} tensor files")
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
        """Calculate statistics with progress bar"""
        print("Calculating statistics...")
        
        stats = {
            'quant_type_stats': {},
            'sample_stats': {},
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
                    'samples': len(set(f['sample'] for f in files if f['sample'] is not None)),
                    'layer_types': list(set(f['layer_type'] for f in files if f['layer_type']))
                }
            pbar.update(1)
        pbar.close()
        
        # Statistics by sample
        pbar = tqdm(total=len(self.samples), desc="Processing samples", unit="samples")
        for sample in self.samples:
            files = data['by_sample'][sample]
            if files:
                stats['sample_stats'][sample] = {
                    'count': len(files),
                    'quant_types': list(set(f['quant_type'] for f in files)),
                    'layers': len(set(f['layer'] for f in files if f['layer'] is not None))
                }
            pbar.update(1)
        pbar.close()
        
        # Statistics by layer
        pbar = tqdm(total=len(self.layers), desc="Processing layers", unit="layers")
        for layer in self.layers:
            files = data['by_layer'][layer]
            if files:
                stats['layer_stats'][layer] = {
                    'count': len(files),
                    'quant_types': list(set(f['quant_type'] for f in files)),
                    'samples': len(set(f['sample'] for f in files if f['sample'] is not None)),
                    'layer_types': list(set(f['layer_type'] for f in files if f['layer_type']))
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
                    'quant_types': list(set(f['quant_type'] for f in files)),
                    'layers': len(set(f['layer'] for f in files if f['layer'] is not None))
                }
            pbar.update(1)
        pbar.close()
        
        # Overall statistics
        stats['overall_stats'] = {
            'total_files': len(data['files']),
            'quant_types': len([q for q in self.quant_types if data['by_quant_type'][q]]),
            'samples': len([s for s in self.samples if data['by_sample'][s]]),
            'layers': len([l for l in self.layers if data['by_layer'][l]]),
            'layer_types': len([lt for lt in ['attention', 'linear'] if data['by_layer_type'][lt]])
        }
        
        return stats
    
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
                    'sample': file_info.get('sample'),
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
            if layer is not None:
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
        
        # 2. Value distribution by sample (line plot)
        ax2 = axes[0, 1]
        sample_stats = {}
        for item in hifp8_data:
            sample = item['sample']
            if sample is not None:
                if sample not in sample_stats:
                    sample_stats[sample] = []
                sample_stats[sample].extend(item['values'])
        
        samples = sorted(sample_stats.keys())
        for sample in samples:
            values = np.array(sample_stats[sample])
            # Sample for plotting if too many points
            if len(values) > 10000:
                values = np.random.choice(values, 10000, replace=False)
            
            # Create histogram and plot as line
            hist, bins = np.histogram(values, bins=50, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax2.plot(bin_centers, hist, label=f'Sample {sample}', alpha=0.7, linewidth=2)
        
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Density')
        ax2.set_title('HiFP8 Value Distribution by Sample')
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
        
        # 5. Layer-Sample heatmap
        ax5 = axes[1, 1]
        layer_sample_matrix = np.zeros((len(layers), len(samples)))
        
        for i, layer in enumerate(layers):
            for j, sample in enumerate(samples):
                layer_sample_data = [item for item in hifp8_data 
                                   if item['layer'] == layer and item['sample'] == sample]
                if layer_sample_data:
                    all_values = np.concatenate([item['values'] for item in layer_sample_data])
                    layer_sample_matrix[i, j] = np.mean(all_values)
        
        im = ax5.imshow(layer_sample_matrix, cmap='viridis', aspect='auto')
        ax5.set_title('HiFP8 Mean Values: Layer vs Sample')
        ax5.set_xlabel('Sample Number')
        ax5.set_ylabel('Layer Number')
        ax5.set_xticks(range(len(samples)))
        ax5.set_xticklabels(samples)
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
    
    def plot_layer_sample_analysis(self, data: Dict, stats: Dict):
        """Plot layer-sample analysis for selected layers"""
        print("Plotting layer-sample analysis...")
        
        # Select key layers for analysis (e.g., 1, 4, 8, 12, 16)
        selected_layers = [1, 4, 8, 12, 16]
        available_layers = [l for l in selected_layers if l in data['by_layer'] and data['by_layer'][l]]
        
        if not available_layers:
            print("No selected layers available for analysis")
            return
        
        # Load tensor values for selected layers
        layer_sample_data = {}
        pbar = tqdm(total=len(available_layers) * len(self.samples), 
                   desc="Loading layer-sample data", unit="combinations")
        
        for layer in available_layers:
            layer_sample_data[layer] = {}
            for sample in self.samples:
                # Find files for this layer-sample combination
                files = [f for f in data['by_layer'][layer] if f.get('sample') == sample]
                
                values_list = []
                for file_info in files:
                    tensor_values = self.load_tensor_values(file_info)
                    if tensor_values is not None:
                        values_list.append(tensor_values.flatten())
                
                if values_list:
                    layer_sample_data[layer][sample] = np.concatenate(values_list)
                else:
                    layer_sample_data[layer][sample] = np.array([])
                
                pbar.update(1)
        pbar.close()
        
        # Create layer-sample analysis charts
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Layer-Sample Analysis for Selected Layers', fontsize=16, fontweight='bold')
        
        # 1. Mean values heatmap
        ax1 = axes[0, 0]
        mean_matrix = np.zeros((len(available_layers), len(self.samples)))
        
        for i, layer in enumerate(available_layers):
            for j, sample in enumerate(self.samples):
                if sample in layer_sample_data[layer] and len(layer_sample_data[layer][sample]) > 0:
                    mean_matrix[i, j] = np.mean(layer_sample_data[layer][sample])
                else:
                    mean_matrix[i, j] = np.nan
        
        im1 = ax1.imshow(mean_matrix, cmap='viridis', aspect='auto')
        ax1.set_title('Mean Values: Layer vs Sample')
        ax1.set_xlabel('Sample Number')
        ax1.set_ylabel('Layer Number')
        ax1.set_xticks(range(len(self.samples)))
        ax1.set_xticklabels(self.samples)
        ax1.set_yticks(range(len(available_layers)))
        ax1.set_yticklabels(available_layers)
        
        # Add colorbar
        plt.colorbar(im1, ax=ax1, label='Mean Value')
        
        # 2. Standard deviation heatmap
        ax2 = axes[0, 1]
        std_matrix = np.zeros((len(available_layers), len(self.samples)))
        
        for i, layer in enumerate(available_layers):
            for j, sample in enumerate(self.samples):
                if sample in layer_sample_data[layer] and len(layer_sample_data[layer][sample]) > 0:
                    std_matrix[i, j] = np.std(layer_sample_data[layer][sample])
                else:
                    std_matrix[i, j] = np.nan
        
        im2 = ax2.imshow(std_matrix, cmap='plasma', aspect='auto')
        ax2.set_title('Standard Deviation: Layer vs Sample')
        ax2.set_xlabel('Sample Number')
        ax2.set_ylabel('Layer Number')
        ax2.set_xticks(range(len(self.samples)))
        ax2.set_xticklabels(self.samples)
        ax2.set_yticks(range(len(available_layers)))
        ax2.set_yticklabels(available_layers)
        
        # Add colorbar
        plt.colorbar(im2, ax=ax2, label='Standard Deviation')
        
        # 3. Value range analysis by layer
        ax3 = axes[1, 0]
        for layer in available_layers:
            all_values = []
            for sample in self.samples:
                if sample in layer_sample_data[layer] and len(layer_sample_data[layer][sample]) > 0:
                    all_values.extend(layer_sample_data[layer][sample])
            
            if all_values:
                all_values = np.array(all_values)
                # Sample for plotting if too many points
                if len(all_values) > 10000:
                    all_values = np.random.choice(all_values, 10000, replace=False)
                
                # Create histogram and plot as line
                hist, bins = np.histogram(all_values, bins=50, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                ax3.plot(bin_centers, hist, label=f'Layer {layer}', alpha=0.7, linewidth=2)
        
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Density')
        ax3.set_title('Value Distribution by Layer')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistical measures by layer
        ax4 = axes[1, 1]
        layer_means = []
        layer_stds = []
        layer_counts = []
        
        for layer in available_layers:
            all_values = []
            for sample in self.samples:
                if sample in layer_sample_data[layer] and len(layer_sample_data[layer][sample]) > 0:
                    all_values.extend(layer_sample_data[layer][sample])
            
            if all_values:
                all_values = np.array(all_values)
                layer_means.append(np.mean(all_values))
                layer_stds.append(np.std(all_values))
                layer_counts.append(len(all_values))
            else:
                layer_means.append(0)
                layer_stds.append(0)
                layer_counts.append(0)
        
        x = np.arange(len(available_layers))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, layer_means, width, label='Mean', alpha=0.8)
        bars2 = ax4.bar(x + width/2, layer_stds, width, label='Std Dev', alpha=0.8)
        
        ax4.set_xlabel('Layer Number')
        ax4.set_ylabel('Value')
        ax4.set_title('Statistical Measures by Layer')
        ax4.set_xticks(x)
        ax4.set_xticklabels(available_layers)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean_val in zip(bars1, layer_means):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(layer_means)*0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save the chart
        output_path = self.subdirs['layer_sample_analysis'] / 'layer_sample_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Layer-sample analysis chart saved: {output_path}")
    
    def generate_global_statistics(self, data: Dict, stats: Dict):
        """Generate comprehensive global statistics"""
        print("Generating global statistics...")
        
        # Load tensor values for analysis
        all_tensor_data = []
        pbar = tqdm(total=len(data['files']), desc="Loading tensor data for statistics", unit="files")
        
        for file_info in data['files']:
            tensor_values = self.load_tensor_values(file_info)
            if tensor_values is not None:
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
            'samples': {},
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
        
        # Process each sample
        pbar = tqdm(total=len(self.samples), desc="Processing samples", unit="samples")
        for sample in self.samples:
            sample_data = [item for item in all_tensor_data if item['file_info'].get('sample') == sample]
            if sample_data:
                all_values = np.concatenate([item['values'] for item in sample_data])
                
                global_stats['samples'][sample] = {
                    'file_count': len(sample_data),
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
        """Run enhanced visualization with progress bars"""
        print("Starting enhanced multi-threaded Tensor visualization...")
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
        
        # Run enhanced analysis with progress bars
        print("\nRunning enhanced analysis...")
        
        # Create progress bar for all tasks
        tasks = [
            ("Quantization comparison", self.plot_quantization_comparison),
            ("Sample analysis", self.plot_sample_analysis),
            ("Layer analysis", self.plot_layer_analysis),
            ("Comprehensive comparison", self.plot_comprehensive_comparison),
            ("HiFP8 distribution analysis", self.plot_hifp8_distribution_analysis),
            ("Layer-sample analysis", self.plot_layer_sample_analysis),
            ("Global statistics", self.generate_global_statistics),
            ("Statistics report", self.save_statistics_report)
        ]
        
        pbar = tqdm(total=len(tasks), desc="Running analysis tasks", unit="tasks")
        
        for task_name, task_func in tasks:
            try:
                if task_name in ["Quantization comparison", "Sample analysis", "Layer analysis", "Comprehensive comparison"]:
                    task_func(data, stats)
                elif task_name == "Statistics report":
                    task_func(stats)
                else:
                    task_func(data, stats)
                pbar.set_description(f"Completed: {task_name}")
            except Exception as e:
                print(f"Error in {task_name}: {e}")
            finally:
                pbar.update(1)
        
        pbar.close()
        
        print("\n" + "=" * 60)
        print("Enhanced multi-threaded Tensor visualization completed!")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print("Generated files:")
        print(f"  - Quantization comparison: {self.subdirs['quantization_analysis'] / 'quantization_comparison.png'}")
        print(f"  - Sample analysis: {self.subdirs['sample_analysis'] / 'sample_analysis.png'}")
        print(f"  - Layer analysis: {self.subdirs['layer_analysis'] / 'layer_analysis.png'}")
        print(f"  - Comprehensive comparison: {self.subdirs['comparison_analysis'] / 'comprehensive_comparison.png'}")
        print(f"  - HiFP8 distribution analysis: {self.subdirs['hifp8_analysis'] / 'hifp8_distribution_analysis.png'}")
        print(f"  - Layer-sample analysis: {self.subdirs['layer_sample_analysis'] / 'layer_sample_analysis.png'}")
        print(f"  - Global statistics JSON: {self.subdirs['global_statistics'] / 'global_statistics.json'}")
        print(f"  - Global statistics report: {self.subdirs['global_statistics'] / 'global_statistics_report.txt'}")
        print(f"  - Statistics report: {self.subdirs['statistics'] / 'detailed_statistics_report.txt'}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Enhanced Multi-threaded Tensor Visualization Tool')
    parser.add_argument('--tensor_dir', type=str, default='./enhanced_tensor_logs',
                       help='Tensor file directory')
    parser.add_argument('--output_dir', type=str, default='./draw',
                       help='Output directory')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum number of workers')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = EnhancedMultiThreadedTensorVisualizer(
        tensor_dir=args.tensor_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    # Run visualization
    visualizer.run_visualization()

if __name__ == "__main__":
    main()
