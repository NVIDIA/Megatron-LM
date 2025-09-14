#!/usr/bin/env python3
"""
Overflow Detection Analyzer - Improved Version
Detect tensor overflow based on quantization type characteristics
Supports bf16, mxfp8, mxfp4, hifp8 quantization types
Includes tqdm progress bar and better error handling
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, NamedTuple
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set matplotlib backend
plt.switch_backend('Agg')

class QuantizationLimits(NamedTuple):
    """Quantization type limits"""
    max_positive_normal: float
    min_positive_normal: float
    max_positive_denormal: float
    min_positive_denormal: float
    exponent_range: Tuple[int, int]
    exponent_range_with_denormal: Tuple[int, int]
    supports_infinity: bool
    supports_nan: bool
    supports_zero: bool

class OverflowDetectionAnalyzer:
    def __init__(self, tensor_dir: str, output_dir: str, max_workers: int = 4):
        self.tensor_dir = Path(tensor_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        
        # Supported quantization types
        self.quant_types = ['bf16', 'mxfp8', 'mxfp4', 'hifp8']
        
        # Supported samples and layers
        self.samples = [0, 1, 2]
        self.layers = list(range(1, 17))  # 1-16 layers
        
        # Define limits for each quantization type (based on chart data)
        self.quantization_limits = {
            'bf16': QuantizationLimits(
                max_positive_normal=2**15 * (2 - 2**-10),  # 65504
                min_positive_normal=2**-14,  # 6.103515625e-05
                max_positive_denormal=2**-14 * (1 - 2**-10),  # 6.095551e-05
                min_positive_denormal=2**-24,  # 5.960464477539063e-08
                exponent_range=(-14, 15),
                exponent_range_with_denormal=(-24, 15),
                supports_infinity=True,
                supports_nan=True,
                supports_zero=True
            ),
            'hifp8': QuantizationLimits(
                max_positive_normal=2**15,  # 32768
                min_positive_normal=2**-15,  # 3.0517578125e-05
                max_positive_denormal=2**-16,  # 1.52587890625e-05
                min_positive_denormal=2**-22,  # 2.384185791015625e-07
                exponent_range=(-15, 15),
                exponent_range_with_denormal=(-22, 15),
                supports_infinity=True,
                supports_nan=True,
                supports_zero=True
            ),
            'mxfp8': QuantizationLimits(  # FP8-E4M3
                max_positive_normal=1.75 * 2**8,  # 448
                min_positive_normal=2**-6,  # 0.015625
                max_positive_denormal=1.75 * 2**-7,  # 0.0013671875
                min_positive_denormal=2**-9,  # 0.001953125
                exponent_range=(-6, 8),
                exponent_range_with_denormal=(-9, 8),
                supports_infinity=False,
                supports_nan=True,
                supports_zero=True
            ),
            'mxfp4': QuantizationLimits(  # FP4-E2M1: 1 sign bit + 2 exponent bits + 1 mantissa bit
                max_positive_normal=1.5 * 2**3,  # 12 (max exponent=3, mantissa=1.5)
                min_positive_normal=2**-2,  # 0.25 (min exponent=-2)
                max_positive_denormal=1.5 * 2**-3,  # 0.1875 (max denormal value)
                min_positive_denormal=2**-4,  # 0.0625 (min denormal value)
                exponent_range=(-2, 3),
                exponent_range_with_denormal=(-4, 3),
                supports_infinity=False,
                supports_nan=False,
                supports_zero=True
            )
        }
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.subdirs = {
            'overflow_analysis': self.output_dir / 'overflow_analysis',
            'detailed_reports': self.output_dir / 'detailed_reports'
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
    
    def detect_overflow(self, tensor_values: np.ndarray, quant_type: str) -> Dict:
        """Detect tensor overflow conditions"""
        if quant_type not in self.quantization_limits:
            return {'error': f'Unsupported quantization type: {quant_type}'}
        
        limits = self.quantization_limits[quant_type]
        
        # Remove NaN and Inf values for statistics
        finite_values = tensor_values[np.isfinite(tensor_values)]
        
        if len(finite_values) == 0:
            return {
                'quant_type': quant_type,
                'total_values': len(tensor_values),
                'finite_values': 0,
                'nan_count': np.sum(np.isnan(tensor_values)),
                'inf_count': np.sum(np.isinf(tensor_values)),
                'min_value': np.nan,
                'max_value': np.nan,
                'mean_value': np.nan,
                'std_value': np.nan,
                'overflow_upper_count': 0,
                'overflow_lower_count': 0,
                'overflow_upper_percentage': 0.0,
                'overflow_lower_percentage': 0.0,
                'overflow_percentage': 0.0,
                'error': 'All values are NaN or Inf'
            }
        
        # Calculate basic statistics
        total_elements = len(finite_values)
        
        # Detect upper overflow (exceeds max normal value)
        upper_overflow = finite_values > limits.max_positive_normal
        
        # Detect lower overflow (less than min normal value)
        lower_overflow = (finite_values > 0) & (finite_values < limits.min_positive_normal)
        
        # Detect extreme overflow (exceeds max denormal value)
        extreme_upper_overflow = finite_values > limits.max_positive_denormal
        
        # Detect extreme lower overflow (less than min denormal value)
        extreme_lower_overflow = (finite_values > 0) & (finite_values < limits.min_positive_denormal)
        
        # Calculate overflow percentages
        upper_overflow_count = np.sum(upper_overflow)
        lower_overflow_count = np.sum(lower_overflow)
        extreme_upper_overflow_count = np.sum(extreme_upper_overflow)
        extreme_lower_overflow_count = np.sum(extreme_lower_overflow)
        
        upper_overflow_percentage = (upper_overflow_count / total_elements) * 100
        lower_overflow_percentage = (lower_overflow_count / total_elements) * 100
        total_overflow_percentage = upper_overflow_percentage + lower_overflow_percentage
        
        return {
            'quant_type': quant_type,
            'total_values': len(tensor_values),
            'finite_values': len(finite_values),
            'nan_count': np.sum(np.isnan(tensor_values)),
            'inf_count': np.sum(np.isinf(tensor_values)),
            'min_value': np.min(finite_values),
            'max_value': np.max(finite_values),
            'mean_value': np.mean(finite_values),
            'std_value': np.std(finite_values),
            'overflow_upper_count': upper_overflow_count,
            'overflow_lower_count': lower_overflow_count,
            'extreme_upper_overflow_count': extreme_upper_overflow_count,
            'extreme_lower_overflow_count': extreme_lower_overflow_count,
            'overflow_upper_percentage': upper_overflow_percentage,
            'overflow_lower_percentage': lower_overflow_percentage,
            'overflow_percentage': total_overflow_percentage,
            'limits': {
                'max_positive_normal': limits.max_positive_normal,
                'min_positive_normal': limits.min_positive_normal,
                'max_positive_denormal': limits.max_positive_denormal,
                'min_positive_denormal': limits.min_positive_denormal
            }
        }
    
    def analyze_tensor_file(self, file_info: Dict) -> Dict:
        """Analyze single tensor file with improved error handling"""
        try:
            tensor_values = self.load_tensor_values(file_info)
            if tensor_values is None:
                # Return error info instead of None
                return {
                    'filename': file_info['filename'],
                    'file_path': str(file_info['file_path']),
                    'quant_type': file_info['quant_type'],
                    'layer': file_info.get('layer'),
                    'sample': file_info.get('sample'),
                    'layer_type': file_info.get('layer_type'),
                    'operation': file_info.get('operation'),
                    'tensor_name': file_info.get('tensor_name'),
                    'error': 'Failed to load tensor',
                    'total_values': 0,
                    'finite_values': 0,
                    'overflow_percentage': 0.0
                }
            
            # Detect overflow
            overflow_result = self.detect_overflow(tensor_values, file_info['quant_type'])
            
            # Add file information
            overflow_result.update({
                'filename': file_info['filename'],
                'file_path': str(file_info['file_path']),
                'quant_type': file_info['quant_type'],
                'layer': file_info.get('layer'),
                'sample': file_info.get('sample'),
                'layer_type': file_info.get('layer_type'),
                'operation': file_info.get('operation'),
                'tensor_name': file_info.get('tensor_name')
            })
            
            return overflow_result
        except Exception as e:
            # Return error info instead of None
            return {
                'filename': file_info['filename'],
                'file_path': str(file_info['file_path']),
                'quant_type': file_info['quant_type'],
                'layer': file_info.get('layer'),
                'sample': file_info.get('sample'),
                'layer_type': file_info.get('layer_type'),
                'operation': file_info.get('operation'),
                'tensor_name': file_info.get('tensor_name'),
                'error': str(e),
                'total_values': 0,
                'finite_values': 0,
                'overflow_percentage': 0.0
            }
    
    def load_tensor_data(self) -> List[Dict]:
        """Load all tensor data"""
        print("Scanning tensor files...")
        
        all_files = []
        
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
                    all_files.append(file_info)
        
        print(f"Total found {len(all_files)} tensor files")
        return all_files
    
    def analyze_all_tensors(self, file_infos: List[Dict]) -> List[Dict]:
        """Analyze all tensor files with progress bar"""
        print("Starting tensor overflow analysis...")
        
        results = []
        failed_files = []
        
        # Create progress bar
        pbar = tqdm(total=len(file_infos), desc="Analyzing tensors", unit="files")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.analyze_tensor_file, file_info): file_info 
                for file_info in file_infos
            }
            
            # Collect results with progress bar
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        
                        # Track failed files
                        if 'error' in result:
                            failed_files.append({
                                'filename': result['filename'],
                                'error': result['error']
                            })
                except Exception as e:
                    failed_files.append({
                        'filename': file_info['filename'],
                        'error': str(e)
                    })
                
                # Update progress bar
                pbar.update(1)
        
        pbar.close()
        
        # Print summary
        print(f"\nAnalysis completed:")
        print(f"  - Successfully processed: {len(results)} files")
        print(f"  - Failed files: {len(failed_files)} files")
        
        # Show failed files if any
        if failed_files:
            print(f"\nFailed files (first 10):")
            for i, failed in enumerate(failed_files[:10]):
                print(f"  {i+1}. {failed['filename']}: {failed['error']}")
            if len(failed_files) > 10:
                print(f"  ... and {len(failed_files) - 10} more")
        
        return results
    
    def generate_overflow_summary(self, results: List[Dict]) -> Dict:
        """Generate overflow summary"""
        print("Generating overflow summary...")
        
        summary = {
            'total_files': len(results),
            'by_quant_type': {},
            'by_sample': {},
            'by_layer': {},
            'by_layer_type': {},
            'overall_stats': {
                'total_overflow_upper': 0,
                'total_overflow_lower': 0,
                'total_underflow_upper': 0,
                'total_underflow_lower': 0,
                'total_values': 0,
                'total_finite_values': 0,
                'total_nan_count': 0,
                'total_inf_count': 0
            }
        }
        
        # Summarize by quantization type
        for quant_type in self.quant_types:
            quant_results = [r for r in results if r['quant_type'] == quant_type]
            if quant_results:
                summary['by_quant_type'][quant_type] = self._summarize_results(quant_results)
        
        # Summarize by sample
        for sample in self.samples:
            sample_results = [r for r in results if r.get('sample') == sample]
            if sample_results:
                summary['by_sample'][sample] = self._summarize_results(sample_results)
        
        # Summarize by layer
        for layer in self.layers:
            layer_results = [r for r in results if r.get('layer') == layer]
            if layer_results:
                summary['by_layer'][layer] = self._summarize_results(layer_results)
        
        # Summarize by layer type
        for layer_type in ['attention', 'linear']:
            layer_type_results = [r for r in results if r.get('layer_type') == layer_type]
            if layer_type_results:
                summary['by_layer_type'][layer_type] = self._summarize_results(layer_type_results)
        
        # Overall statistics
        for result in results:
            if 'error' not in result:
                summary['overall_stats']['total_overflow_upper'] += result.get('overflow_upper_count', 0)
                summary['overall_stats']['total_overflow_lower'] += result.get('overflow_lower_count', 0)
                summary['overall_stats']['total_values'] += result.get('total_values', 0)
                summary['overall_stats']['total_finite_values'] += result.get('finite_values', 0)
                summary['overall_stats']['total_nan_count'] += result.get('nan_count', 0)
                summary['overall_stats']['total_inf_count'] += result.get('inf_count', 0)
        
        return summary
    
    def _summarize_results(self, results: List[Dict]) -> Dict:
        """Summarize a list of results"""
        if not results:
            return {}
        
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {
                'count': len(results),
                'valid_count': 0,
                'error_count': len(results),
                'avg_overflow_percentage': 0.0,
                'max_overflow_percentage': 0.0,
                'total_values': 0,
                'total_finite_values': 0
            }
        
        overflow_percentages = [r.get('overflow_percentage', 0) for r in valid_results]
        
        return {
            'count': len(results),
            'valid_count': len(valid_results),
            'error_count': len(results) - len(valid_results),
            'avg_overflow_percentage': np.mean(overflow_percentages),
            'max_overflow_percentage': np.max(overflow_percentages),
            'total_values': sum(r.get('total_values', 0) for r in valid_results),
            'total_finite_values': sum(r.get('finite_values', 0) for r in valid_results)
        }
    
    def plot_overflow_analysis(self, results: List[Dict], summary: Dict):
        """Plot overflow analysis charts"""
        print("Plotting overflow analysis charts...")
        
        # Filter out error results for plotting
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            print("No valid results to plot")
            return
        
        # Create comprehensive analysis chart
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Tensor Overflow Analysis Report', fontsize=16, fontweight='bold')
        
        # 1. Overflow percentage by quantization type
        ax1 = axes[0, 0]
        quant_overflow = {}
        for quant_type in self.quant_types:
            quant_results = [r for r in valid_results if r['quant_type'] == quant_type]
            if quant_results:
                overflow_percentages = [r.get('overflow_percentage', 0) for r in quant_results]
                quant_overflow[quant_type] = {
                    'mean': np.mean(overflow_percentages),
                    'std': np.std(overflow_percentages),
                    'max': np.max(overflow_percentages)
                }
        
        if quant_overflow:
            quant_types = list(quant_overflow.keys())
            means = [quant_overflow[qt]['mean'] for qt in quant_types]
            stds = [quant_overflow[qt]['std'] for qt in quant_types]
            maxs = [quant_overflow[qt]['max'] for qt in quant_types]
            
            x = np.arange(len(quant_types))
            width = 0.25
            
            ax1.bar(x - width, means, width, label='Mean', alpha=0.8)
            ax1.bar(x, maxs, width, label='Max', alpha=0.8)
            ax1.bar(x + width, stds, width, label='Std Dev', alpha=0.8)
            
            ax1.set_xlabel('Quantization Type')
            ax1.set_ylabel('Overflow Percentage (%)')
            ax1.set_title('Overflow Percentage by Quantization Type')
            ax1.set_xticks(x)
            ax1.set_xticklabels(quant_types)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Overflow distribution histogram
        ax2 = axes[0, 1]
        overflow_percentages = [r.get('overflow_percentage', 0) for r in valid_results]
        ax2.hist(overflow_percentages, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Overflow Percentage (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Overflow Percentage Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Upper vs Lower overflow scatter plot
        ax3 = axes[0, 2]
        upper_overflow = [r.get('overflow_upper_percentage', 0) for r in valid_results]
        lower_overflow = [r.get('overflow_lower_percentage', 0) for r in valid_results]
        
        # Color by quantization type
        colors = {'bf16': 'blue', 'hifp8': 'green', 'mxfp8': 'orange', 'mxfp4': 'red'}
        for quant_type in self.quant_types:
            quant_results = [r for r in valid_results if r['quant_type'] == quant_type]
            if quant_results:
                upper = [r.get('overflow_upper_percentage', 0) for r in quant_results]
                lower = [r.get('overflow_lower_percentage', 0) for r in quant_results]
                ax3.scatter(upper, lower, label=quant_type, alpha=0.6, s=20, color=colors.get(quant_type, 'gray'))
        
        ax3.set_xlabel('Upper Overflow Percentage (%)')
        ax3.set_ylabel('Lower Overflow Percentage (%)')
        ax3.set_title('Upper vs Lower Overflow')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Overflow by layer
        ax4 = axes[1, 0]
        layer_overflow = {}
        for layer in self.layers:
            layer_results = [r for r in valid_results if r.get('layer') == layer]
            if layer_results:
                overflow_percentages = [r.get('overflow_percentage', 0) for r in layer_results]
                layer_overflow[layer] = np.mean(overflow_percentages)
        
        if layer_overflow:
            layers = sorted(layer_overflow.keys())
            overflow_values = [layer_overflow[l] for l in layers]
            ax4.plot(layers, overflow_values, marker='o', linewidth=2, markersize=6)
            ax4.set_xlabel('Layer Number')
            ax4.set_ylabel('Average Overflow Percentage (%)')
            ax4.set_title('Overflow Percentage by Layer')
            ax4.grid(True, alpha=0.3)
        
        # 5. Overflow by sample
        ax5 = axes[1, 1]
        sample_overflow = {}
        for sample in self.samples:
            sample_results = [r for r in valid_results if r.get('sample') == sample]
            if sample_results:
                overflow_percentages = [r.get('overflow_percentage', 0) for r in sample_results]
                sample_overflow[sample] = np.mean(overflow_percentages)
        
        if sample_overflow:
            samples = sorted(sample_overflow.keys())
            overflow_values = [sample_overflow[s] for s in samples]
            ax5.bar(samples, overflow_values, alpha=0.7)
            ax5.set_xlabel('Sample Number')
            ax5.set_ylabel('Average Overflow Percentage (%)')
            ax5.set_title('Overflow Percentage by Sample')
            ax5.grid(True, alpha=0.3)
        
        # 6. Quantization type-sample overflow heatmap
        ax6 = axes[1, 2]
        overflow_matrix = np.zeros((len(self.quant_types), len(self.samples)))
        for i, quant_type in enumerate(self.quant_types):
            for j, sample in enumerate(self.samples):
                if quant_type in summary['by_quant_type'] and sample in summary['by_sample']:
                    # Calculate overflow rate for this quant type and sample combination
                    quant_sample_results = [r for r in results 
                                          if r['quant_type'] == quant_type and r.get('sample') == sample]
                    if quant_sample_results:
                        total_overflow = sum(r.get('overflow_percentage', 0) for r in quant_sample_results)
                        overflow_matrix[i, j] = total_overflow / len(quant_sample_results)
        
        im = ax6.imshow(overflow_matrix, cmap='Reds', aspect='auto')
        ax6.set_title('Quantization Type-Sample Overflow Heatmap')
        ax6.set_xlabel('Sample Number')
        ax6.set_ylabel('Quantization Type')
        ax6.set_xticks(range(len(self.samples)))
        ax6.set_xticklabels(self.samples)
        ax6.set_yticks(range(len(self.quant_types)))
        ax6.set_yticklabels(self.quant_types)
        
        # Add colorbar
        plt.colorbar(im, ax=ax6, label='Overflow Percentage (%)')
        
        plt.tight_layout()
        
        # Save the chart
        output_path = self.subdirs['overflow_analysis'] / 'overflow_analysis_report.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Overflow analysis chart saved: {output_path}")
    
    def save_detailed_report(self, results: List[Dict], summary: Dict):
        """Save detailed overflow report"""
        print("Saving detailed report...")
        
        report_path = self.subdirs['detailed_reports'] / 'overflow_detection_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Tensor Overflow Detection Detailed Report\n")
            f.write("=" * 80 + "\n\n")
            
            # Quantization type limits
            f.write("Quantization Type Limits:\n")
            f.write("-" * 50 + "\n")
            for quant_type, limits in self.quantization_limits.items():
                f.write(f"{quant_type}:\n")
                f.write(f"  Max Normal Value: {limits.max_positive_normal:.6e}\n")
                f.write(f"  Min Normal Value: {limits.min_positive_normal:.6e}\n")
                f.write(f"  Max Denormal Value: {limits.max_positive_denormal:.6e}\n")
                f.write(f"  Min Denormal Value: {limits.min_positive_denormal:.6e}\n")
                f.write(f"  Exponent Range: {limits.exponent_range}\n")
                f.write(f"  Exponent Range (with denormal): {limits.exponent_range_with_denormal}\n")
                f.write(f"  Supports Infinity: {limits.supports_infinity}\n")
                f.write(f"  Supports NaN: {limits.supports_nan}\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Files: {summary['total_files']:,}\n")
            f.write(f"Total Values: {summary['overall_stats']['total_values']:,}\n")
            f.write(f"Total Finite Values: {summary['overall_stats']['total_finite_values']:,}\n")
            f.write(f"Total NaN Count: {summary['overall_stats']['total_nan_count']:,}\n")
            f.write(f"Total Inf Count: {summary['overall_stats']['total_inf_count']:,}\n")
            f.write(f"Total Upper Overflow: {summary['overall_stats']['total_overflow_upper']:,}\n")
            f.write(f"Total Lower Overflow: {summary['overall_stats']['total_overflow_lower']:,}\n\n")
            
            # By quantization type
            f.write("By Quantization Type:\n")
            f.write("-" * 50 + "\n")
            for quant_type, stats in summary['by_quant_type'].items():
                f.write(f"{quant_type}:\n")
                f.write(f"  Files: {stats['count']:,}\n")
                f.write(f"  Valid Files: {stats['valid_count']:,}\n")
                f.write(f"  Error Files: {stats['error_count']:,}\n")
                f.write(f"  Avg Overflow %: {stats['avg_overflow_percentage']:.4f}%\n")
                f.write(f"  Max Overflow %: {stats['max_overflow_percentage']:.4f}%\n")
                f.write(f"  Total Values: {stats['total_values']:,}\n")
                f.write(f"  Total Finite Values: {stats['total_finite_values']:,}\n\n")
            
            # By sample
            f.write("By Sample:\n")
            f.write("-" * 50 + "\n")
            for sample, stats in summary['by_sample'].items():
                f.write(f"Sample {sample}:\n")
                f.write(f"  Files: {stats['count']:,}\n")
                f.write(f"  Valid Files: {stats['valid_count']:,}\n")
                f.write(f"  Error Files: {stats['error_count']:,}\n")
                f.write(f"  Avg Overflow %: {stats['avg_overflow_percentage']:.4f}%\n")
                f.write(f"  Max Overflow %: {stats['max_overflow_percentage']:.4f}%\n\n")
            
            # By layer
            f.write("By Layer:\n")
            f.write("-" * 50 + "\n")
            for layer, stats in summary['by_layer'].items():
                f.write(f"Layer {layer}:\n")
                f.write(f"  Files: {stats['count']:,}\n")
                f.write(f"  Valid Files: {stats['valid_count']:,}\n")
                f.write(f"  Error Files: {stats['error_count']:,}\n")
                f.write(f"  Avg Overflow %: {stats['avg_overflow_percentage']:.4f}%\n")
                f.write(f"  Max Overflow %: {stats['max_overflow_percentage']:.4f}%\n\n")
            
            # By layer type
            f.write("By Layer Type:\n")
            f.write("-" * 50 + "\n")
            for layer_type, stats in summary['by_layer_type'].items():
                f.write(f"{layer_type}:\n")
                f.write(f"  Files: {stats['count']:,}\n")
                f.write(f"  Valid Files: {stats['valid_count']:,}\n")
                f.write(f"  Error Files: {stats['error_count']:,}\n")
                f.write(f"  Avg Overflow %: {stats['avg_overflow_percentage']:.4f}%\n")
                f.write(f"  Max Overflow %: {stats['max_overflow_percentage']:.4f}%\n\n")
        
        print(f"Detailed report saved: {report_path}")
    
    def run_analysis(self):
        """Run complete overflow analysis"""
        print("Starting Tensor overflow detection analysis...")
        print(f"Tensor directory: {self.tensor_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Max workers: {self.max_workers}")
        
        # Load tensor data
        file_infos = self.load_tensor_data()
        if not file_infos:
            print("Error: No tensor files found")
            return
        
        # Analyze all tensors
        results = self.analyze_all_tensors(file_infos)
        
        # Generate summary
        summary = self.generate_overflow_summary(results)
        
        # Plot analysis
        self.plot_overflow_analysis(results, summary)
        
        # Save detailed report
        self.save_detailed_report(results, summary)
        
        print("\n" + "=" * 60)
        print("Tensor overflow detection analysis completed!")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print("Generated files:")
        print(f"  - Overflow analysis chart: {self.subdirs['overflow_analysis'] / 'overflow_analysis_report.png'}")
        print(f"  - Detailed report: {self.subdirs['detailed_reports'] / 'overflow_detection_report.txt'}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Overflow Detection Analyzer - Improved Version')
    parser.add_argument('--tensor_dir', type=str, default='./enhanced_tensor_logs',
                       help='Tensor file directory')
    parser.add_argument('--output_dir', type=str, default='./draw',
                       help='Output directory')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum number of workers')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = OverflowDetectionAnalyzer(
        tensor_dir=args.tensor_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    # Run analysis
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
