#!/usr/bin/env python3
"""
Overflow Analysis Tool by Tensor Type

Analyzes overflow when simulating quantization on BF16 tensors.
Groups tensors by type (input, weight, query, key, value) and analyzes overflow
in forward and backward passes, generating line plots.
Usage: python fake_quant_type_underflow_analysis.py --elem-format hifp8
"""

import re
import json
import torch
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import sys,os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define numerical format ranges based on research and specifications
DATA_TYPE_RANGES = {
    'bf16': {
        'min_normal': 6.103516e-05,     # BFloat16 minimum normal value
        'max_normal': 6.550400e+04,     # BFloat16 maximum normal value
        'min_denormal': 5.960464e-08,   # BFloat16 minimum denormal value
        'max_denormal': 6.097555e-05,   # BFloat16 maximum denormal value
        'min': -6.550400e+04,           # Effective minimum (negative max normal)
        'max': 6.550400e+04,            # Effective maximum (positive max normal)
        'supports_infinity': True,
        'supports_nan': True,
        'description': 'Brain Float 16-bit'
    },
    'hifp8': {
        'min_normal': 3.051758e-05,     # HiFP8 minimum normal value (2^-15)
        'max_normal': 3.276800e+04,     # HiFP8 maximum normal value (2^15)
        'min_denormal': 2.384186e-07,   # HiFP8 minimum denormal value (2^-22)
        'max_denormal': 1.525879e-05,   # HiFP8 maximum denormal value (approx 2^-16)
        'min': -3.276800e+04,           # Effective minimum (negative max normal)
        'max': 3.276800e+04,            # Effective maximum (positive max normal)
        'supports_infinity': True,
        'supports_nan': True,
        'description': 'Huawei HiFP8 E4M3 format'
    },
    'mxfp8e4m3': {
        'min_normal': 1.562500e-02,     # MX FP8 minimum normal value (0.015625)
        'max_normal': 4.480000e+02,     # MX FP8 maximum normal value (448.0)
        'min_denormal': 1.953125e-03,   # MX FP8 minimum denormal value (2^-9)
        'max_denormal': 1.367188e-02,   # MX FP8 maximum denormal value (7*2^-9)
        'min': -4.480000e+02,           # Effective minimum (negative max normal)
        'max': 4.480000e+02,            # Effective maximum (positive max normal)
        'supports_infinity': False,
        'supports_nan': True,
        'description': 'Microsoft MX FP8 E4M3 format'
    },
    'mxfp8e5m2': {
        'min_normal': 6.103516e-05,     # MX FP8 E5M2 minimum normal value (2^-14)
        'max_normal': 5.734400e+04,     # MX FP8 E5M2 maximum normal value (57344.0)
        'min_denormal': 1.525879e-05,   # MX FP8 E5M2 minimum denormal value (2^-16)
        'max_denormal': 4.577637e-05,   # MX FP8 E5M2 maximum denormal value (3*2^-16)
        'min': -5.734400e+04,           # Effective minimum (negative max normal)
        'max': 5.734400e+04,            # Effective maximum (positive max normal)
        'supports_infinity': True,
        'supports_nan': True,
        'description': 'Microsoft MX FP8 E5M2 format'
    },
    'mxfp4': {
        'min_normal': 1.000000e+00,     # MX FP4 minimum normal value
        'max_normal': 6.000000e+00,     # MX FP4 maximum normal value
        'min_denormal': 5.000000e-01,   # MX FP4 minimum denormal value (2^-1)
        'max_denormal': 5.000000e-01,   # MX FP4 maximum denormal value (only one denormal: 0.5)
        'min': -6.000000e+00,           # Effective minimum (negative max normal)
        'max': 6.000000e+00,            # Effective maximum (positive max normal)
        'supports_infinity': False,
        'supports_nan': False,
        'description': 'Microsoft MX FP4 E2M1 format'
    }
}

# Supported tensor types
TENSOR_TYPES = ['input', 'weight', 'query', 'key', 'value']


def get_tensor_type_and_pass(filename: str) -> tuple:
    """Extract tensor type and pass type (forward/backward) from filename."""
    filename_lower = filename.lower()
    
    # Extract pass type
    if '_forward_' in filename_lower:
        pass_type = 'forward'
    elif '_backward_' in filename_lower:
        pass_type = 'backward'
    else:
        pass_type = None
    
    # Extract tensor type from suffix
    tensor_type = None
    for ttype in TENSOR_TYPES:
        # Check if filename ends with the tensor type (with optional separators)
        if filename_lower.endswith(f'_{ttype}.pt') or filename_lower.endswith(f'-{ttype}.pt'):
            tensor_type = ttype
            break
        # Also check for patterns like ..._input_... or ..._weight_...
        if f'_{ttype}_' in filename_lower or f'-{ttype}-' in filename_lower:
            tensor_type = ttype
            break
    
    return tensor_type, pass_type


def remove_scaling(tensor: torch.Tensor, elem_format: str) -> torch.Tensor:
    """Remove scaling from tensor based on element format."""
    from quant.mxfp_scaling_test import remove_scaling
    if elem_format == 'hifp8':
        return tensor
    elif elem_format == 'mxfp8e4m3':
        return remove_scaling(tensor, "fp8_e4m3")
    elif elem_format == 'mxfp8e5m2':
        return remove_scaling(tensor, "fp8_e5m2")
    elif elem_format == 'mxfp4':
        return remove_scaling(tensor, "fp4_e2m1")
    else:
        raise ValueError(f"Unsupported element format: {elem_format}")


def analyze_tensor(file_path: Path, elem_format: str) -> dict:
    """Analyze tensor for overflow."""
    # Load tensor
    tensor = torch.load(file_path, map_location='cpu', weights_only=False)
    
    # Handle different formats
    if not isinstance(tensor, torch.Tensor):
        if isinstance(tensor, dict) and 'tensor' in tensor:
            tensor = tensor['tensor']
        elif isinstance(tensor, (list, tuple)) and len(tensor) > 0:
            tensor = tensor[0]
        else:
            return None
    
    # Convert to float32
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    else:
        tensor = tensor.float()

    tensor = remove_scaling(tensor, elem_format)
    
    # Flatten and analyze
    flat = tensor.flatten()
    total = flat.numel()
    if total == 0:
        return None
    
    # Calculate overflow: values exceeding maximum representable value
    abs_vals = torch.abs(flat)
    if elem_format in DATA_TYPE_RANGES:
        max_normal = DATA_TYPE_RANGES[elem_format]['max_normal']
    else:
        raise ValueError(f"Unsupported element format: {elem_format}")
    
    overflow = abs_vals > max_normal
    overflow_count = torch.sum(overflow).item()
    overflow_pct = (overflow_count / total) * 100.0
    
    # Get metadata
    tensor_type, pass_type = get_tensor_type_and_pass(file_path.name)
    if tensor_type is None or pass_type is None:
        return None
    
    return {
        'tensor_type': tensor_type,
        'pass_type': pass_type,
        'total_elements': int(total),
        'overflow_count': int(overflow_count),
        'overflow_percentage': float(overflow_pct),
        'filename': file_path.name,
    }


def plot_overflow_analysis(results: dict, elem_format: str, output_path: Path):
    """Generate line plots for overflow analysis."""
    # Prepare data for plotting
    # Structure: {tensor_type: {pass_type: overflow_percentage}}
    plot_data = defaultdict(lambda: {'forward': [], 'backward': []})
    tensor_types = sorted([t for t in TENSOR_TYPES if t in results])
    
    for tensor_type in tensor_types:
        for pass_type in ['forward', 'backward']:
            if pass_type in results[tensor_type]:
                plot_data[tensor_type][pass_type] = results[tensor_type][pass_type]['overflow_percentage']
            else:
                plot_data[tensor_type][pass_type] = 0.0
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Forward pass
    forward_data = {ttype: plot_data[ttype]['forward'] for ttype in tensor_types}
    backward_data = {ttype: plot_data[ttype]['backward'] for ttype in tensor_types}
    
    x_pos = np.arange(len(tensor_types))
    width = 0.35
    
    # Forward pass bars
    forward_values = [forward_data[ttype] for ttype in tensor_types]
    ax1.bar(x_pos - width/2, forward_values, width, label='Forward', color='#2ecc71', alpha=0.8)
    
    # Add value labels on bars
    for i, (ttype, val) in enumerate(zip(tensor_types, forward_values)):
        ax1.text(i - width/2, val, f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Tensor Type', fontsize=12)
    ax1.set_ylabel('Overflow Percentage (%)', fontsize=12)
    ax1.set_title(f'{elem_format.upper()} Overflow Analysis - Forward Pass', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(tensor_types, fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Backward pass
    backward_values = [backward_data[ttype] for ttype in tensor_types]
    ax2.bar(x_pos - width/2, backward_values, width, label='Backward', color='#e74c3c', alpha=0.8)
    
    # Add value labels on bars
    for i, (ttype, val) in enumerate(zip(tensor_types, backward_values)):
        ax2.text(i - width/2, val, f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Tensor Type', fontsize=12)
    ax2.set_ylabel('Overflow Percentage (%)', fontsize=12)
    ax2.set_title(f'{elem_format.upper()} Overflow Analysis - Backward Pass', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tensor_types, fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a combined line plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(tensor_types))
    forward_line = ax.plot(x, forward_values, marker='o', linewidth=2, markersize=8, 
                          label='Forward', color='#2ecc71')
    backward_line = ax.plot(x, backward_values, marker='s', linewidth=2, markersize=8, 
                            label='Backward', color='#e74c3c')
    
    # Add value annotations
    for i, (ttype, fval, bval) in enumerate(zip(tensor_types, forward_values, backward_values)):
        ax.annotate(f'{fval:.2f}%', (i, fval), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9, color='#2ecc71')
        ax.annotate(f'{bval:.2f}%', (i, bval), textcoords="offset points", 
                   xytext=(0,-15), ha='center', fontsize=9, color='#e74c3c')
    
    ax.set_xlabel('Tensor Type', fontsize=12)
    ax.set_ylabel('Overflow Percentage (%)', fontsize=12)
    ax.set_title(f'{elem_format.upper()} Overflow Analysis - Forward vs Backward', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tensor_types, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Save combined plot
    combined_path = output_path.parent / f"{output_path.stem}_combined{output_path.suffix}"
    plt.tight_layout()
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {output_path} and {combined_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze overflow by tensor type for BF16 tensors')
    parser.add_argument('--base-dir', default='enhanced_tensor_logs', help='Base directory')
    parser.add_argument('--elem-format', default='hifp8', 
                       choices=['hifp8', 'mxfp8e4m3', 'mxfp8e5m2', 'mxfp4'], 
                       help='Element format: hifp8, mxfp8e4m3, mxfp8e5m2, mxfp4')
    parser.add_argument('--output-dir', default=None, help='Output directory for plots (default: base-dir)')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    bf16_dir = base_dir / 'bf16'
    
    if not bf16_dir.exists():
        print(f"Error: Directory not found: {bf16_dir}")
        return 1
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else base_dir
    
    # Find and analyze files
    # Structure: {tensor_type: {pass_type: aggregated_stats}}
    results = defaultdict(lambda: defaultdict(lambda: {
        'total_elements': 0,
        'total_overflow': 0,
        'overflow_percentage': 0.0,
        'num_tensors': 0
    }))
    
    # Collect all matching files
    all_files = list(bf16_dir.glob("*.pt"))
    matching_files = []
    for file_path in all_files:
        tensor_type, pass_type = get_tensor_type_and_pass(file_path.name)
        if tensor_type in TENSOR_TYPES and pass_type:
            matching_files.append(file_path)
    
    print(f"Found {len(matching_files)} matching tensor files to analyze")
    print(f"Tensor types: {TENSOR_TYPES}")
    print(f"Element format: {args.elem_format}")
    print()
    
    # Analyze all matching files
    for file_path in tqdm(matching_files, desc="Processing tensors", unit="file"):
        result = analyze_tensor(file_path, args.elem_format)
        if result:
            tensor_type = result['tensor_type']
            pass_type = result['pass_type']
            
            # Aggregate results
            results[tensor_type][pass_type]['total_elements'] += result['total_elements']
            results[tensor_type][pass_type]['total_overflow'] += result['overflow_count']
            results[tensor_type][pass_type]['num_tensors'] += 1
    
    # Calculate percentages
    for tensor_type in results:
        for pass_type in results[tensor_type]:
            total_elements = results[tensor_type][pass_type]['total_elements']
            total_overflow = results[tensor_type][pass_type]['total_overflow']
            if total_elements > 0:
                results[tensor_type][pass_type]['overflow_percentage'] = \
                    (total_overflow / total_elements) * 100.0
    
    # Print report
    print("\n" + "=" * 80)
    print(f"{args.elem_format.upper()} OVERFLOW ANALYSIS BY TENSOR TYPE")
    print("=" * 80)
    
    # Prepare data for JSON output
    json_data = {
        'elem_format': args.elem_format,
        'tensor_types': sorted([t for t in TENSOR_TYPES if t in results]),
        'results': []
    }
    
    for tensor_type in sorted(results.keys()):
        print(f"\n{tensor_type.upper()}:")
        for pass_type in ['forward', 'backward']:
            if pass_type not in results[tensor_type]:
                continue
            
            stats = results[tensor_type][pass_type]
            print(f"  {pass_type.upper()}: {stats['overflow_percentage']:.4f}% overflow "
                  f"({stats['total_overflow']:,}/{stats['total_elements']:,} elements, "
                  f"{stats['num_tensors']} tensors)")
            
            # Add to JSON data
            json_data['results'].append({
                'tensor_type': tensor_type,
                'pass_type': pass_type,
                'total_elements': int(stats['total_elements']),
                'total_overflow': int(stats['total_overflow']),
                'overflow_percentage': float(stats['overflow_percentage']),
                'num_tensors': int(stats['num_tensors'])
            })
    
    # Save to JSON file
    json_filename = f"{args.elem_format}_overflow_by_type.json"
    json_path = output_dir / json_filename
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Generate plots
    plot_filename = f"{args.elem_format}_overflow_by_type.png"
    plot_path = output_dir / plot_filename
    plot_overflow_analysis(results, args.elem_format, plot_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
