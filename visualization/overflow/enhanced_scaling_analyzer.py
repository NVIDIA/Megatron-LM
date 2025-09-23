#!/usr/bin/env python3
"""
Enhanced Scaling Factor Analysis Tool

This program analyzes all log files in the scaling_analysis directory with enhanced
tensor type classification for forward/backward passes and detailed tensor naming.

Author: AI Assistant
Created: 2025-09-23
"""

import os
import re
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class EnhancedTensorAnalysis:
    """Enhanced data class to store tensor analysis results"""
    tensor_name: str
    file_path: str
    max_align: float
    min_align: float
    recommended_scale_exp: float
    recommended_scale_factor: float
    composite_score: float
    is_at_max: bool
    mse: float
    cosine_similarity: float
    psnr: float
    mae: float
    relative_error: float
    
    # Enhanced classification
    layer: int
    pass_type: str  # forward, backward
    operation_type: str  # linear, attention
    tensor_type: str  # input_A, input_B, output, weight, query, key, value, etc.
    rank: int
    group: int


class EnhancedScalingAnalyzer:
    """Enhanced analyzer class for scaling factor analysis"""
    
    def __init__(self, base_directory: str = "/Users/charles/Downloads/draw"):
        self.base_directory = Path(base_directory)
        self.scaling_dir = self.base_directory / "scaling_analysis"
        self.results: List[EnhancedTensorAnalysis] = []
        
    def find_log_files(self) -> List[Path]:
        """Find all log files in the scaling analysis directory"""
        pattern = str(self.scaling_dir / "**" / "*.log")
        log_files = glob.glob(pattern, recursive=True)
        return [Path(f) for f in log_files]
    
    def parse_log_file(self, log_file: Path) -> Optional[EnhancedTensorAnalysis]:
        """Parse a single log file and extract scaling information"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract enhanced tensor information from file path
            tensor_info = self._extract_enhanced_tensor_info(log_file)
            if not tensor_info:
                return None
            
            # Extract alignment information
            alignment_match = re.search(r'Calculated alignment \(reference\): max_align=(-?\d+\.?\d*), min_align=(-?\d+\.?\d*)', content)
            if not alignment_match:
                print(f"Warning: Could not find alignment info in {log_file}")
                return None
            
            max_align = float(alignment_match.group(1))
            min_align = float(alignment_match.group(2))
            
            # Extract recommended scaling information
            scale_exp_match = re.search(r'⭐ RECOMMENDED Scaling Factor: ([\d\.e\-\+]+)\s+.*?Scale Exponent: (-?\d+\.?\d*)', content, re.DOTALL)
            if not scale_exp_match:
                print(f"Warning: Could not find recommended scaling info in {log_file}")
                return None
                
            recommended_scale_factor = float(scale_exp_match.group(1))
            recommended_scale_exp = float(scale_exp_match.group(2))
            
            # Extract composite score
            composite_match = re.search(r'Composite Score: ([\d\.e\-\+]+)', content)
            composite_score = float(composite_match.group(1)) if composite_match else 0.0
            
            # Extract performance metrics
            mse_match = re.search(r'- MSE: ([\d\.e\-\+]+)', content)
            cosine_match = re.search(r'- Cosine Similarity: ([\d\.e\-\+]+)', content)
            psnr_match = re.search(r'- PSNR: ([\d\.e\-\+]+) dB', content)
            mae_match = re.search(r'- MAE: ([\d\.e\-\+]+)', content)
            rel_error_match = re.search(r'- Relative Error: ([\d\.]+)%', content)
            
            mse = float(mse_match.group(1)) if mse_match else 0.0
            cosine_similarity = float(cosine_match.group(1)) if cosine_match else 0.0
            psnr = float(psnr_match.group(1)) if psnr_match else 0.0
            mae = float(mae_match.group(1)) if mae_match else 0.0
            relative_error = float(rel_error_match.group(1)) if rel_error_match else 0.0
            
            # Check if recommended scaling is at maximum value
            is_at_max = abs(recommended_scale_exp - max_align) < 1e-6
            
            return EnhancedTensorAnalysis(
                tensor_name=tensor_info['tensor_name'],
                file_path=str(log_file),
                max_align=max_align,
                min_align=min_align,
                recommended_scale_exp=recommended_scale_exp,
                recommended_scale_factor=recommended_scale_factor,
                composite_score=composite_score,
                is_at_max=is_at_max,
                mse=mse,
                cosine_similarity=cosine_similarity,
                psnr=psnr,
                mae=mae,
                relative_error=relative_error,
                layer=tensor_info['layer'],
                pass_type=tensor_info['pass_type'],
                operation_type=tensor_info['operation_type'],
                tensor_type=tensor_info['tensor_type'],
                rank=tensor_info['rank'],
                group=tensor_info['group']
            )
            
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
            return None
    
    def _extract_enhanced_tensor_info(self, log_file: Path) -> Optional[Dict]:
        """Extract enhanced tensor information from file path"""
        try:
            # Extract from the directory name
            parent_dir = log_file.parent.name
            
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
            print(f"Error extracting tensor info from {log_file}: {e}")
            return None
    
    def analyze_all_files(self) -> None:
        """Analyze all log files and store results"""
        log_files = self.find_log_files()
        print(f"Found {len(log_files)} log files to analyze...")
        
        successful_parses = 0
        for log_file in log_files:
            result = self.parse_log_file(log_file)
            if result:
                self.results.append(result)
                successful_parses += 1
            
        print(f"Successfully parsed {successful_parses} out of {len(log_files)} log files")
    
    def generate_enhanced_summary(self) -> Dict:
        """Generate enhanced summary statistics"""
        if not self.results:
            return {}
            
        total_tensors = len(self.results)
        at_max_count = sum(1 for r in self.results if r.is_at_max)
        not_at_max_count = total_tensors - at_max_count
        
        # Group by various dimensions
        layer_stats = defaultdict(lambda: {'total': 0, 'at_max': 0})
        pass_type_stats = defaultdict(lambda: {'total': 0, 'at_max': 0})
        operation_type_stats = defaultdict(lambda: {'total': 0, 'at_max': 0})
        tensor_type_stats = defaultdict(lambda: {'total': 0, 'at_max': 0})
        
        # Combined statistics
        layer_pass_stats = defaultdict(lambda: {'total': 0, 'at_max': 0})
        layer_operation_stats = defaultdict(lambda: {'total': 0, 'at_max': 0})
        
        for result in self.results:
            # Layer statistics
            layer_key = f"Layer_{result.layer}"
            layer_stats[layer_key]['total'] += 1
            if result.is_at_max:
                layer_stats[layer_key]['at_max'] += 1
            
            # Pass type statistics
            pass_type_stats[result.pass_type]['total'] += 1
            if result.is_at_max:
                pass_type_stats[result.pass_type]['at_max'] += 1
            
            # Operation type statistics
            operation_type_stats[result.operation_type]['total'] += 1
            if result.is_at_max:
                operation_type_stats[result.operation_type]['at_max'] += 1
            
            # Tensor type statistics
            tensor_type_stats[result.tensor_type]['total'] += 1
            if result.is_at_max:
                tensor_type_stats[result.tensor_type]['at_max'] += 1
            
            # Combined statistics
            layer_pass_key = f"L{result.layer}_{result.pass_type}"
            layer_pass_stats[layer_pass_key]['total'] += 1
            if result.is_at_max:
                layer_pass_stats[layer_pass_key]['at_max'] += 1
            
            layer_operation_key = f"L{result.layer}_{result.operation_type}"
            layer_operation_stats[layer_operation_key]['total'] += 1
            if result.is_at_max:
                layer_operation_stats[layer_operation_key]['at_max'] += 1
        
        return {
            'total_tensors': total_tensors,
            'at_max_count': at_max_count,
            'not_at_max_count': not_at_max_count,
            'at_max_percentage': (at_max_count / total_tensors) * 100 if total_tensors > 0 else 0,
            'layer_stats': dict(layer_stats),
            'pass_type_stats': dict(pass_type_stats),
            'operation_type_stats': dict(operation_type_stats),
            'tensor_type_stats': dict(tensor_type_stats),
            'layer_pass_stats': dict(layer_pass_stats),
            'layer_operation_stats': dict(layer_operation_stats)
        }
    
    def print_enhanced_report(self) -> None:
        """Print enhanced analysis report"""
        if not self.results:
            print("No results to report!")
            return
            
        summary = self.generate_enhanced_summary()
        
        print("\n" + "="*100)
        print("ENHANCED SCALING FACTOR ANALYSIS REPORT")
        print("="*100)
        
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"   Total Tensors Analyzed: {summary['total_tensors']}")
        print(f"   Tensors at Maximum Scaling: {summary['at_max_count']} ({summary['at_max_percentage']:.1f}%)")
        print(f"   Tensors NOT at Maximum: {summary['not_at_max_count']} ({100-summary['at_max_percentage']:.1f}%)")
        
        # Pass type breakdown
        print(f"\n🔄 BREAKDOWN BY PASS TYPE:")
        for pass_type, stats in summary['pass_type_stats'].items():
            percentage = (stats['at_max'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"   {pass_type.upper():10} | Total: {stats['total']:3d} | At Max: {stats['at_max']:3d} ({percentage:5.1f}%)")
        
        # Operation type breakdown
        print(f"\n⚙️  BREAKDOWN BY OPERATION TYPE:")
        for op_type, stats in summary['operation_type_stats'].items():
            percentage = (stats['at_max'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"   {op_type.upper():10} | Total: {stats['total']:3d} | At Max: {stats['at_max']:3d} ({percentage:5.1f}%)")
        
        # Tensor type breakdown
        print(f"\n📋 BREAKDOWN BY TENSOR TYPE:")
        for tensor_type, stats in summary['tensor_type_stats'].items():
            percentage = (stats['at_max'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"   {tensor_type.upper():15} | Total: {stats['total']:3d} | At Max: {stats['at_max']:3d} ({percentage:5.1f}%)")
        
        # Layer breakdown
        if summary['layer_stats']:
            print(f"\n🏗️  BREAKDOWN BY LAYER:")
            for layer, stats in sorted(summary['layer_stats'].items()):
                percentage = (stats['at_max'] / stats['total']) * 100 if stats['total'] > 0 else 0
                print(f"   {layer:10} | Total: {stats['total']:3d} | At Max: {stats['at_max']:3d} ({percentage:5.1f}%)")
        
        # Layer-Pass combination
        print(f"\n🔄🏗️  BREAKDOWN BY LAYER-PASS COMBINATION:")
        for layer_pass, stats in sorted(summary['layer_pass_stats'].items()):
            percentage = (stats['at_max'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"   {layer_pass:15} | Total: {stats['total']:3d} | At Max: {stats['at_max']:3d} ({percentage:5.1f}%)")
        
        # Layer-Operation combination
        print(f"\n⚙️🏗️  BREAKDOWN BY LAYER-OPERATION COMBINATION:")
        for layer_op, stats in sorted(summary['layer_operation_stats'].items()):
            percentage = (stats['at_max'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"   {layer_op:15} | Total: {stats['total']:3d} | At Max: {stats['at_max']:3d} ({percentage:5.1f}%)")
        
        print("\n" + "="*100)
    
    def save_enhanced_results_to_json(self, output_file: str = "enhanced_scaling_analysis_results.json") -> None:
        """Save enhanced results to JSON file"""
        output_path = self.base_directory / output_file
        
        # Convert dataclass objects to dictionaries
        results_dict = []
        for result in self.results:
            results_dict.append({
                'tensor_name': result.tensor_name,
                'file_path': result.file_path,
                'max_align': result.max_align,
                'min_align': result.min_align,
                'recommended_scale_exp': result.recommended_scale_exp,
                'recommended_scale_factor': result.recommended_scale_factor,
                'composite_score': result.composite_score,
                'is_at_max': result.is_at_max,
                'mse': result.mse,
                'cosine_similarity': result.cosine_similarity,
                'psnr': result.psnr,
                'mae': result.mae,
                'relative_error': result.relative_error,
                'layer': result.layer,
                'pass_type': result.pass_type,
                'operation_type': result.operation_type,
                'tensor_type': result.tensor_type,
                'rank': result.rank,
                'group': result.group
            })
        
        summary = self.generate_enhanced_summary()
        
        output_data = {
            'analysis_summary': summary,
            'detailed_results': results_dict,
            'metadata': {
                'total_files_analyzed': len(self.results),
                'analysis_date': '2025-09-23',
                'base_directory': str(self.base_directory),
                'enhanced_analysis': True
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Enhanced results saved to: {output_path}")


def main():
    """Main function"""
    print("🔍 Starting Enhanced Scaling Factor Analysis...")
    
    # Initialize analyzer
    analyzer = EnhancedScalingAnalyzer()
    
    # Check if scaling directory exists
    if not analyzer.scaling_dir.exists():
        print(f"❌ Error: Scaling analysis directory not found: {analyzer.scaling_dir}")
        return
    
    # Analyze all files
    analyzer.analyze_all_files()
    
    if not analyzer.results:
        print("❌ No valid results found!")
        return
    
    # Print enhanced report
    analyzer.print_enhanced_report()
    
    # Save results to JSON
    analyzer.save_enhanced_results_to_json()
    
    print("\n✅ Enhanced analysis completed successfully!")


if __name__ == "__main__":
    main()
