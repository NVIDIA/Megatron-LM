#!/usr/bin/env python3
"""
Enhanced Overflow Analysis Tool

This program analyzes overflow/underflow data from the enhanced tensor structure
with forward/backward passes and detailed tensor type classification.

Author: AI Assistant
Created: 2025-09-23
"""

import os
import re
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class EnhancedQuantizationMetrics:
    """Enhanced data class for quantization metrics"""
    tensor_name: str
    layer: int
    pass_type: str  # forward, backward
    operation_type: str  # linear, attention
    tensor_type: str  # input_A, input_B, output, weight, query, key, value, etc.
    rank: int
    group: int
    total_elements: int
    underflow_percentage: float
    flush_to_zero_percentage: float
    overflow_percentage: float
    underflow_significant: bool
    overflow_significant: bool


class EnhancedOverflowAnalyzer:
    """Enhanced analyzer for overflow/underflow analysis"""
    
    def __init__(self, base_directory: str = "/Users/charles/Downloads/draw"):
        self.base_directory = Path(base_directory)
        self.scaling_dir = self.base_directory / "scaling_analysis"
        self.results: List[EnhancedQuantizationMetrics] = []
        self.fp8_max_norm = 448.0
        
        # Enhanced color scheme for different categories
        self.colors = {
            'forward': '#2E86AB',    # Blue
            'backward': '#A23B72',   # Purple
            'linear': '#F18F01',     # Orange
            'attention': '#C73E1D',  # Red
            'input_A': '#2E8B57',    # Sea Green
            'input_B': '#4682B4',    # Steel Blue
            'output': '#DC143C',     # Crimson
            'weight': '#8B4513',     # Saddle Brown
            'query': '#9370DB',      # Medium Purple
            'key': '#20B2AA',        # Light Sea Green
            'value': '#FF6347',      # Tomato
            'probs': '#FFD700',      # Gold
            'buffer': '#32CD32',     # Lime Green
            'grad_output': '#FF1493', # Deep Pink
            'grad_attention_probs': '#00CED1', # Dark Turquoise
            'grad_value': '#FF8C00', # Dark Orange
            'grad_query': '#8A2BE2', # Blue Violet
            'grad_key': '#00FF7F',   # Spring Green
            'mm_output': '#FF69B4',  # Hot Pink
        }
        
    def find_result_files(self) -> List[Path]:
        """Find all result files in the scaling analysis directory"""
        pattern = str(self.scaling_dir / "**" / "*_results_fp8_e4m3.txt")
        result_files = glob.glob(pattern, recursive=True)
        return [Path(f) for f in result_files]
    
    def extract_enhanced_metadata_from_path(self, result_file: Path) -> Optional[Dict]:
        """Extract enhanced metadata from file path"""
        try:
            # Extract from the directory name
            parent_dir = result_file.parent.name
            
            # Parse the enhanced naming format:
            # 20250923_100142_0001_iter000_linear_L1_forward_pre_linear_bf16_rank00_group000_input_A
            
            # Extract layer
            layer_match = re.search(r'_L(\d+)_', parent_dir)
            if not layer_match:
                return None
            layer = int(layer_match.group(1))
            
            # Extract pass type (forward/backward)
            if '_forward_' in parent_dir:
                pass_type = 'forward'
            elif '_backward_' in parent_dir:
                pass_type = 'backward'
            else:
                pass_type = 'unknown'
            
            # Extract operation type
            if '_linear_' in parent_dir:
                operation_type = 'linear'
            elif '_attention_' in parent_dir:
                operation_type = 'attention'
            else:
                operation_type = 'unknown'
            
            # Extract rank
            rank_match = re.search(r'_rank(\d+)_', parent_dir)
            rank = int(rank_match.group(1)) if rank_match else 0
            
            # Extract group
            group_match = re.search(r'_group(\d+)_', parent_dir)
            group = int(group_match.group(1)) if group_match else 0
            
            # Extract tensor type (the last part after the last underscore)
            parts = parent_dir.split('_')
            tensor_type = parts[-1]  # input_A, input_B, output, weight, query, key, value, etc.
            
            # Create a more readable tensor name
            tensor_name = f"L{layer}_{pass_type}_{operation_type}_{tensor_type}"
            
            return {
                'tensor_name': tensor_name,
                'layer': layer,
                'pass_type': pass_type,
                'operation_type': operation_type,
                'tensor_type': tensor_type,
                'rank': rank,
                'group': group
            }
            
        except Exception as e:
            print(f"Error extracting metadata from {result_file}: {e}")
            return None
    
    def parse_recommended_metrics(self, result_file: Path) -> Optional[EnhancedQuantizationMetrics]:
        """Parse metrics at recommended scaling factor"""
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = self.extract_enhanced_metadata_from_path(result_file)
            if not metadata:
                return None
            
            # First, get the recommended scale exponent from the corresponding log file
            log_file = result_file.parent / f"mxfp_scaling_test_{result_file.parent.name}_fp8_e4m3.log"
            recommended_scale_exp = None
            
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    scale_match = re.search(r'⭐ RECOMMENDED Scaling Factor: [\d\.e\-\+]+.*?Scale Exponent: (-?\d+\.?\d*)', log_content, re.DOTALL)
                    if scale_match:
                        recommended_scale_exp = float(scale_match.group(1))
                except:
                    pass
            
            if recommended_scale_exp is None:
                return None
            
            # Now find the specific section for the recommended scale exponent
            scale_exp_str = f"{recommended_scale_exp:.2f}"
            recommended_section_pattern = rf'Scale Exponent {re.escape(scale_exp_str)} \(Factor: [\d\.e\-\+]+\):\s*\n.*?Overflow/Underflow Analysis:\s*\n\s*Total Elements: ([\d,]+)\s*\n\s*Underflow Count: [\d,]+ \(([\d\.]+)%\)\s*\n\s*Flush to Zero Count: [\d,]+ \(([\d\.]+)%\)\s*\n\s*Overflow Count: [\d,]+ \(([\d\.]+)%\)'
            
            analysis_match = re.search(recommended_section_pattern, content, re.DOTALL)
            
            if not analysis_match:
                return None
            
            total_elements = int(analysis_match.group(1).replace(',', ''))
            underflow_percentage = float(analysis_match.group(2))
            flush_to_zero_percentage = float(analysis_match.group(3))
            overflow_percentage = float(analysis_match.group(4))
            
            # Extract significance flags from the same section
            section_content = analysis_match.group(0)
            underflow_significant = 'Has Significant Underflow: Yes' in section_content
            overflow_significant = 'Has Significant Overflow: Yes' in section_content
            
            return EnhancedQuantizationMetrics(
                tensor_name=metadata['tensor_name'],
                layer=metadata['layer'],
                pass_type=metadata['pass_type'],
                operation_type=metadata['operation_type'],
                tensor_type=metadata['tensor_type'],
                rank=metadata['rank'],
                group=metadata['group'],
                total_elements=total_elements,
                underflow_percentage=underflow_percentage,
                flush_to_zero_percentage=flush_to_zero_percentage,
                overflow_percentage=overflow_percentage,
                underflow_significant=underflow_significant,
                overflow_significant=overflow_significant
            )
        except Exception as e:
            print(f"Error parsing {result_file}: {e}")
            return None
    
    def analyze_all_files(self) -> None:
        """Analyze all result files and store metrics"""
        result_files = self.find_result_files()
        print(f"Found {len(result_files)} result files to analyze...")
        
        successful_parses = 0
        for result_file in result_files:
            metrics = self.parse_recommended_metrics(result_file)
            if metrics:
                self.results.append(metrics)
                successful_parses += 1
        
        print(f"Successfully parsed {successful_parses} out of {len(result_files)} result files")
    
    def create_enhanced_overflow_analysis_plot(self) -> None:
        """Create enhanced overflow analysis plot by layer and pass type"""
        if not self.results:
            print("No data to plot!")
            return
        
        # Group data by layer and pass type
        layer_pass_data = defaultdict(lambda: {'overflow': [], 'underflow': [], 'flush_to_zero': []})
        
        for result in self.results:
            key = f"L{result.layer}_{result.pass_type}"
            layer_pass_data[key]['overflow'].append(result.overflow_percentage)
            layer_pass_data[key]['underflow'].append(result.underflow_percentage)
            layer_pass_data[key]['flush_to_zero'].append(result.flush_to_zero_percentage)
        
        # Create the plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16))
        
        # Prepare data for plotting
        layers = sorted(set(result.layer for result in self.results))
        pass_types = ['forward', 'backward']
        
        x_pos = np.arange(len(layers))
        width = 0.35
        
        # Overflow plot
        forward_overflow = []
        backward_overflow = []
        for layer in layers:
            forward_key = f"L{layer}_forward"
            backward_key = f"L{layer}_backward"
            
            forward_avg = np.mean(layer_pass_data[forward_key]['overflow']) if forward_key in layer_pass_data else 0
            backward_avg = np.mean(layer_pass_data[backward_key]['overflow']) if backward_key in layer_pass_data else 0
            
            forward_overflow.append(forward_avg)
            backward_overflow.append(backward_avg)
        
        bars1 = ax1.bar(x_pos - width/2, forward_overflow, width, label='Forward Pass', 
                       color=self.colors['forward'], alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, backward_overflow, width, label='Backward Pass', 
                       color=self.colors['backward'], alpha=0.8)
        
        ax1.set_title('Overflow Analysis by Layer and Pass Type', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Overflow Percentage (%)', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'Layer {l}' for l in layers])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Underflow plot
        forward_underflow = []
        backward_underflow = []
        for layer in layers:
            forward_key = f"L{layer}_forward"
            backward_key = f"L{layer}_backward"
            
            forward_avg = np.mean(layer_pass_data[forward_key]['underflow']) if forward_key in layer_pass_data else 0
            backward_avg = np.mean(layer_pass_data[backward_key]['underflow']) if backward_key in layer_pass_data else 0
            
            forward_underflow.append(forward_avg)
            backward_underflow.append(backward_avg)
        
        bars3 = ax2.bar(x_pos - width/2, forward_underflow, width, label='Forward Pass', 
                       color=self.colors['forward'], alpha=0.8)
        bars4 = ax2.bar(x_pos + width/2, backward_underflow, width, label='Backward Pass', 
                       color=self.colors['backward'], alpha=0.8)
        
        ax2.set_title('Underflow Analysis by Layer and Pass Type', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Underflow Percentage (%)', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'Layer {l}' for l in layers])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Flush-to-Zero plot
        forward_flush = []
        backward_flush = []
        for layer in layers:
            forward_key = f"L{layer}_forward"
            backward_key = f"L{layer}_backward"
            
            forward_avg = np.mean(layer_pass_data[forward_key]['flush_to_zero']) if forward_key in layer_pass_data else 0
            backward_avg = np.mean(layer_pass_data[backward_key]['flush_to_zero']) if backward_key in layer_pass_data else 0
            
            forward_flush.append(forward_avg)
            backward_flush.append(backward_avg)
        
        bars5 = ax3.bar(x_pos - width/2, forward_flush, width, label='Forward Pass', 
                       color=self.colors['forward'], alpha=0.8)
        bars6 = ax3.bar(x_pos + width/2, backward_flush, width, label='Backward Pass', 
                       color=self.colors['backward'], alpha=0.8)
        
        ax3.set_title('Flush-to-Zero Analysis by Layer and Pass Type', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Layer', fontsize=12)
        ax3.set_ylabel('Flush-to-Zero Percentage (%)', fontsize=12)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'Layer {l}' for l in layers])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = self.base_directory / "enhanced_overflow_plots"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "enhanced_layer_pass_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Enhanced layer-pass analysis plot saved to: {output_file}")
        
        plt.show()
    
    def create_tensor_type_analysis_plot(self) -> None:
        """Create tensor type analysis plot"""
        if not self.results:
            print("No data to plot!")
            return
        
        # Group data by tensor type
        tensor_type_data = defaultdict(lambda: {'overflow': [], 'underflow': [], 'flush_to_zero': []})
        
        for result in self.results:
            tensor_type_data[result.tensor_type]['overflow'].append(result.overflow_percentage)
            tensor_type_data[result.tensor_type]['underflow'].append(result.underflow_percentage)
            tensor_type_data[result.tensor_type]['flush_to_zero'].append(result.flush_to_zero_percentage)
        
        # Create the plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 18))
        
        # Prepare data for plotting
        tensor_types = sorted(tensor_type_data.keys())
        x_pos = np.arange(len(tensor_types))
        
        # Overflow plot
        overflow_means = [np.mean(tensor_type_data[t]['overflow']) for t in tensor_types]
        overflow_stds = [np.std(tensor_type_data[t]['overflow']) for t in tensor_types]
        
        bars1 = ax1.bar(x_pos, overflow_means, yerr=overflow_stds, capsize=5,
                       color=[self.colors.get(t, '#808080') for t in tensor_types], alpha=0.8)
        
        ax1.set_title('Overflow Analysis by Tensor Type', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Tensor Type', fontsize=12)
        ax1.set_ylabel('Overflow Percentage (%)', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(tensor_types, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Underflow plot
        underflow_means = [np.mean(tensor_type_data[t]['underflow']) for t in tensor_types]
        underflow_stds = [np.std(tensor_type_data[t]['underflow']) for t in tensor_types]
        
        bars2 = ax2.bar(x_pos, underflow_means, yerr=underflow_stds, capsize=5,
                       color=[self.colors.get(t, '#808080') for t in tensor_types], alpha=0.8)
        
        ax2.set_title('Underflow Analysis by Tensor Type', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Tensor Type', fontsize=12)
        ax2.set_ylabel('Underflow Percentage (%)', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(tensor_types, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Flush-to-Zero plot
        flush_means = [np.mean(tensor_type_data[t]['flush_to_zero']) for t in tensor_types]
        flush_stds = [np.std(tensor_type_data[t]['flush_to_zero']) for t in tensor_types]
        
        bars3 = ax3.bar(x_pos, flush_means, yerr=flush_stds, capsize=5,
                       color=[self.colors.get(t, '#808080') for t in tensor_types], alpha=0.8)
        
        ax3.set_title('Flush-to-Zero Analysis by Tensor Type', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Tensor Type', fontsize=12)
        ax3.set_ylabel('Flush-to-Zero Percentage (%)', fontsize=12)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(tensor_types, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = self.base_directory / "enhanced_overflow_plots"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "enhanced_tensor_type_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Enhanced tensor type analysis plot saved to: {output_file}")
        
        plt.show()
    
    def create_operation_type_analysis_plot(self) -> None:
        """Create operation type analysis plot"""
        if not self.results:
            print("No data to plot!")
            return
        
        # Group data by operation type and pass type
        operation_data = defaultdict(lambda: {'overflow': [], 'underflow': [], 'flush_to_zero': []})
        
        for result in self.results:
            key = f"{result.operation_type}_{result.pass_type}"
            operation_data[key]['overflow'].append(result.overflow_percentage)
            operation_data[key]['underflow'].append(result.underflow_percentage)
            operation_data[key]['flush_to_zero'].append(result.flush_to_zero_percentage)
        
        # Create the plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))
        
        # Prepare data for plotting
        operation_types = sorted(operation_data.keys())
        x_pos = np.arange(len(operation_types))
        
        # Overflow plot
        overflow_means = [np.mean(operation_data[t]['overflow']) for t in operation_types]
        overflow_stds = [np.std(operation_data[t]['overflow']) for t in operation_types]
        
        colors = []
        for op_type in operation_types:
            if 'linear' in op_type:
                colors.append(self.colors['linear'])
            elif 'attention' in op_type:
                colors.append(self.colors['attention'])
            else:
                colors.append('#808080')
        
        bars1 = ax1.bar(x_pos, overflow_means, yerr=overflow_stds, capsize=5,
                       color=colors, alpha=0.8)
        
        ax1.set_title('Overflow Analysis by Operation Type and Pass', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Operation Type', fontsize=12)
        ax1.set_ylabel('Overflow Percentage (%)', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(operation_types, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Underflow plot
        underflow_means = [np.mean(operation_data[t]['underflow']) for t in operation_types]
        underflow_stds = [np.std(operation_data[t]['underflow']) for t in operation_types]
        
        bars2 = ax2.bar(x_pos, underflow_means, yerr=underflow_stds, capsize=5,
                       color=colors, alpha=0.8)
        
        ax2.set_title('Underflow Analysis by Operation Type and Pass', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Operation Type', fontsize=12)
        ax2.set_ylabel('Underflow Percentage (%)', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(operation_types, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Flush-to-Zero plot
        flush_means = [np.mean(operation_data[t]['flush_to_zero']) for t in operation_types]
        flush_stds = [np.std(operation_data[t]['flush_to_zero']) for t in operation_types]
        
        bars3 = ax3.bar(x_pos, flush_means, yerr=flush_stds, capsize=5,
                       color=colors, alpha=0.8)
        
        ax3.set_title('Flush-to-Zero Analysis by Operation Type and Pass', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Operation Type', fontsize=12)
        ax3.set_ylabel('Flush-to-Zero Percentage (%)', fontsize=12)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(operation_types, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = self.base_directory / "enhanced_overflow_plots"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "enhanced_operation_type_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Enhanced operation type analysis plot saved to: {output_file}")
        
        plt.show()
    
    def generate_enhanced_summary(self) -> Dict:
        """Generate enhanced summary statistics"""
        if not self.results:
            return {}
        
        # Group by various dimensions
        layer_stats = defaultdict(lambda: {'total': 0, 'overflow': [], 'underflow': [], 'flush_to_zero': []})
        pass_type_stats = defaultdict(lambda: {'total': 0, 'overflow': [], 'underflow': [], 'flush_to_zero': []})
        operation_type_stats = defaultdict(lambda: {'total': 0, 'overflow': [], 'underflow': [], 'flush_to_zero': []})
        tensor_type_stats = defaultdict(lambda: {'total': 0, 'overflow': [], 'underflow': [], 'flush_to_zero': []})
        
        for result in self.results:
            # Layer statistics
            layer_key = f"Layer_{result.layer}"
            layer_stats[layer_key]['total'] += 1
            layer_stats[layer_key]['overflow'].append(result.overflow_percentage)
            layer_stats[layer_key]['underflow'].append(result.underflow_percentage)
            layer_stats[layer_key]['flush_to_zero'].append(result.flush_to_zero_percentage)
            
            # Pass type statistics
            pass_type_stats[result.pass_type]['total'] += 1
            pass_type_stats[result.pass_type]['overflow'].append(result.overflow_percentage)
            pass_type_stats[result.pass_type]['underflow'].append(result.underflow_percentage)
            pass_type_stats[result.pass_type]['flush_to_zero'].append(result.flush_to_zero_percentage)
            
            # Operation type statistics
            operation_type_stats[result.operation_type]['total'] += 1
            operation_type_stats[result.operation_type]['overflow'].append(result.overflow_percentage)
            operation_type_stats[result.operation_type]['underflow'].append(result.underflow_percentage)
            operation_type_stats[result.operation_type]['flush_to_zero'].append(result.flush_to_zero_percentage)
            
            # Tensor type statistics
            tensor_type_stats[result.tensor_type]['total'] += 1
            tensor_type_stats[result.tensor_type]['overflow'].append(result.overflow_percentage)
            tensor_type_stats[result.tensor_type]['underflow'].append(result.underflow_percentage)
            tensor_type_stats[result.tensor_type]['flush_to_zero'].append(result.flush_to_zero_percentage)
        
        # Calculate averages
        def calc_averages(stats_dict):
            result = {}
            for key, data in stats_dict.items():
                result[key] = {
                    'total': data['total'],
                    'avg_overflow': np.mean(data['overflow']) if data['overflow'] else 0,
                    'avg_underflow': np.mean(data['underflow']) if data['underflow'] else 0,
                    'avg_flush_to_zero': np.mean(data['flush_to_zero']) if data['flush_to_zero'] else 0,
                    'max_overflow': np.max(data['overflow']) if data['overflow'] else 0,
                    'max_underflow': np.max(data['underflow']) if data['underflow'] else 0,
                    'max_flush_to_zero': np.max(data['flush_to_zero']) if data['flush_to_zero'] else 0
                }
            return result
        
        return {
            'total_tensors': len(self.results),
            'layer_stats': calc_averages(layer_stats),
            'pass_type_stats': calc_averages(pass_type_stats),
            'operation_type_stats': calc_averages(operation_type_stats),
            'tensor_type_stats': calc_averages(tensor_type_stats)
        }
    
    def print_enhanced_summary(self) -> None:
        """Print enhanced summary report"""
        if not self.results:
            print("No results to report!")
            return
        
        summary = self.generate_enhanced_summary()
        
        print("\n" + "="*100)
        print("ENHANCED OVERFLOW/UNDERFLOW ANALYSIS SUMMARY")
        print("="*100)
        
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"   Total Tensors Analyzed: {summary['total_tensors']}")
        
        # Pass type breakdown
        print(f"\n🔄 BREAKDOWN BY PASS TYPE:")
        for pass_type, stats in summary['pass_type_stats'].items():
            print(f"   {pass_type.upper():10} | Total: {stats['total']:3d} | "
                  f"Avg Overflow: {stats['avg_overflow']:6.3f}% | "
                  f"Avg Underflow: {stats['avg_underflow']:6.3f}% | "
                  f"Avg Flush-to-Zero: {stats['avg_flush_to_zero']:6.3f}%")
        
        # Operation type breakdown
        print(f"\n⚙️  BREAKDOWN BY OPERATION TYPE:")
        for op_type, stats in summary['operation_type_stats'].items():
            print(f"   {op_type.upper():10} | Total: {stats['total']:3d} | "
                  f"Avg Overflow: {stats['avg_overflow']:6.3f}% | "
                  f"Avg Underflow: {stats['avg_underflow']:6.3f}% | "
                  f"Avg Flush-to-Zero: {stats['avg_flush_to_zero']:6.3f}%")
        
        # Layer breakdown
        print(f"\n🏗️  BREAKDOWN BY LAYER:")
        for layer, stats in sorted(summary['layer_stats'].items()):
            print(f"   {layer:10} | Total: {stats['total']:3d} | "
                  f"Avg Overflow: {stats['avg_overflow']:6.3f}% | "
                  f"Avg Underflow: {stats['avg_underflow']:6.3f}% | "
                  f"Avg Flush-to-Zero: {stats['avg_flush_to_zero']:6.3f}%")
        
        # Tensor type breakdown
        print(f"\n📋 BREAKDOWN BY TENSOR TYPE:")
        for tensor_type, stats in summary['tensor_type_stats'].items():
            print(f"   {tensor_type.upper():15} | Total: {stats['total']:3d} | "
                  f"Avg Overflow: {stats['avg_overflow']:6.3f}% | "
                  f"Avg Underflow: {stats['avg_underflow']:6.3f}% | "
                  f"Avg Flush-to-Zero: {stats['avg_flush_to_zero']:6.3f}%")
        
        print("\n" + "="*100)


def main():
    """Main function"""
    print("🔍 Starting Enhanced Overflow Analysis...")
    
    # Initialize analyzer
    analyzer = EnhancedOverflowAnalyzer()
    
    # Check if scaling directory exists
    if not analyzer.scaling_dir.exists():
        print(f"❌ Error: Scaling analysis directory not found: {analyzer.scaling_dir}")
        return
    
    # Analyze all files
    analyzer.analyze_all_files()
    
    if not analyzer.results:
        print("❌ No valid results found!")
        return
    
    # Print enhanced summary
    analyzer.print_enhanced_summary()
    
    # Create enhanced plots
    print("\n📊 Creating enhanced analysis plots...")
    analyzer.create_enhanced_overflow_analysis_plot()
    analyzer.create_tensor_type_analysis_plot()
    analyzer.create_operation_type_analysis_plot()
    
    print("\n✅ Enhanced overflow analysis completed successfully!")


if __name__ == "__main__":
    main()
