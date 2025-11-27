#!/usr/bin/env python3
"""
Underflow Analysis Tool by Layer and Tensor Type

Analyzes underflow when simulating quantization on BF16 tensors.
Groups tensors by type (input, weight, query, key, value) and layer,
analyzes underflow in forward pass only, generating line plots by layer.
Usage: python fake_quant_type_underflow_analysis.py --layer 1,8,15,16 --elem-format hifp8
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


def parse_layers(layer_str: str) -> list:
    """Parse layer argument: '1,8,15,16' -> [1, 8, 15, 16]"""
    return [int(l.strip()) for l in layer_str.split(',')]


def get_tensor_type_layer_and_pass(filename: str) -> tuple:
    """Extract tensor type, layer number and pass type (forward/backward) from filename."""
    filename_lower = filename.lower()
    
    # Extract layer number
    layer_match = re.search(r'_L(\d+)_', filename)
    if not layer_match:
        return None, None, None
    
    layer = int(layer_match.group(1))
    
    # Extract pass type (only forward)
    if '_forward_' in filename_lower:
        pass_type = 'forward'
    else:
        pass_type = None  # Only process forward pass
    
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
    
    return tensor_type, layer, pass_type


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
    """Analyze tensor for underflow."""
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
    
    # Calculate underflow: non-zero values < min_normal
    abs_vals = torch.abs(flat)
    non_zero = flat != 0.0
    if elem_format in DATA_TYPE_RANGES:
        min_normal = DATA_TYPE_RANGES[elem_format]['min_normal']
    else:
        raise ValueError(f"Unsupported element format: {elem_format}")
    
    underflow = non_zero & (abs_vals < min_normal)
    underflow_count = torch.sum(underflow).item()
    underflow_pct = (underflow_count / total) * 100.0
    
    # Get metadata
    tensor_type, layer, pass_type = get_tensor_type_layer_and_pass(file_path.name)
    if tensor_type is None or layer is None or pass_type is None:
        return None
    
    return {
        'tensor_type': tensor_type,
        'layer': layer,
        'pass_type': pass_type,
        'total_elements': int(total),
        'underflow_count': int(underflow_count),
        'underflow_percentage': float(underflow_pct),
        'filename': file_path.name,
    }


def plot_underflow_analysis(results: dict, elem_format: str, output_path: Path):
    """Generate line plots for underflow analysis by layer."""
    # Structure: {tensor_type: {layer: underflow_percentage}}
    # Prepare data: collect all layers and tensor types
    all_layers = set()
    tensor_types_in_data = set()
    
    for tensor_type in results:
        if 'forward' in results[tensor_type]:
            for layer in results[tensor_type]['forward']:
                all_layers.add(layer)
                tensor_types_in_data.add(tensor_type)
    
    all_layers = sorted(all_layers)
    tensor_types_in_data = sorted(tensor_types_in_data)
    
    if not all_layers or not tensor_types_in_data:
        print("Warning: No data to plot")
        return
    
    # Prepare plot data: {tensor_type: [underflow_pct for each layer]}
    plot_data = {}
    for tensor_type in tensor_types_in_data:
        plot_data[tensor_type] = []
        for layer in all_layers:
            if layer in results[tensor_type]['forward']:
                plot_data[tensor_type].append(results[tensor_type]['forward'][layer]['underflow_percentage'])
            else:
                plot_data[tensor_type].append(0.0)
    
    # Create line plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color palette for different tensor types
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, tensor_type in enumerate(tensor_types_in_data):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(all_layers, plot_data[tensor_type], 
               marker=marker, linewidth=2.5, markersize=8, 
               label=tensor_type.capitalize(), color=color, alpha=0.8)
        
        # Add value annotations for better readability
        for i, (layer, val) in enumerate(zip(all_layers, plot_data[tensor_type])):
            if val > 0:  # Only annotate non-zero values
                ax.annotate(f'{val:.2f}%', (layer, val), 
                           textcoords="offset points", 
                           xytext=(0, 8 if idx % 2 == 0 else -15), 
                           ha='center', fontsize=8, color=color, alpha=0.7)
    
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Underflow Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'{elem_format.upper()} Underflow Analysis by Layer (Forward Pass)', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(all_layers)
    ax.set_xticklabels(all_layers, fontsize=10)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)  # Start y-axis from 0
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze underflow by tensor type and layer for BF16 tensors')
    parser.add_argument('--layer', required=True, help='Layer numbers: 1,8,15,16')
    parser.add_argument('--base-dir', default='enhanced_tensor_logs', help='Base directory')
    parser.add_argument('--elem-format', default='hifp8', 
                       choices=['hifp8', 'mxfp8e4m3', 'mxfp8e5m2', 'mxfp4'], 
                       help='Element format: hifp8, mxfp8e4m3, mxfp8e5m2, mxfp4')
    parser.add_argument('--output-dir', default=None, help='Output directory for plots (default: base-dir)')
    args = parser.parse_args()
    
    # Parse layers
    layers = parse_layers(args.layer)
    
    base_dir = Path(args.base_dir)
    bf16_dir = base_dir / 'bf16'
    
    if not bf16_dir.exists():
        print(f"Error: Directory not found: {bf16_dir}")
        return 1
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else base_dir
    
    # Find and analyze files
    # Structure: {tensor_type: {'forward': {layer: aggregated_stats}}}
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'total_elements': 0,
        'total_underflow': 0,
        'underflow_percentage': 0.0,
        'num_tensors': 0
    })))
    
    # Collect all matching files
    all_files = list(bf16_dir.glob("*.pt"))
    matching_files = []
    for file_path in all_files:
        tensor_type, layer, pass_type = get_tensor_type_layer_and_pass(file_path.name)
        if tensor_type in TENSOR_TYPES and layer in layers and pass_type == 'forward':
            matching_files.append(file_path)
    
    print(f"Analyzing layers: {layers}")
    print(f"Found {len(matching_files)} matching tensor files to analyze")
    print(f"Tensor types: {TENSOR_TYPES}")
    print(f"Element format: {args.elem_format}")
    print(f"Pass type: forward only")
    print()
    
    # Analyze all matching files
    for file_path in tqdm(matching_files, desc="Processing tensors", unit="file"):
        result = analyze_tensor(file_path, args.elem_format)
        if result:
            tensor_type = result['tensor_type']
            layer = result['layer']
            pass_type = result['pass_type']
            
            # Only process forward pass
            if pass_type == 'forward' and layer in layers:
                # Aggregate results by layer
                results[tensor_type][pass_type][layer]['total_elements'] += result['total_elements']
                results[tensor_type][pass_type][layer]['total_underflow'] += result['underflow_count']
                results[tensor_type][pass_type][layer]['num_tensors'] += 1
    
    # Calculate percentages for each layer
    for tensor_type in results:
        for pass_type in results[tensor_type]:
            for layer in results[tensor_type][pass_type]:
                total_elements = results[tensor_type][pass_type][layer]['total_elements']
                total_underflow = results[tensor_type][pass_type][layer]['total_underflow']
                if total_elements > 0:
                    results[tensor_type][pass_type][layer]['underflow_percentage'] = \
                        (total_underflow / total_elements) * 100.0
    
    # Print report
    print("\n" + "=" * 80)
    print(f"{args.elem_format.upper()} UNDERFLOW ANALYSIS BY LAYER AND TENSOR TYPE (FORWARD)")
    print("=" * 80)
    
    # Prepare data for JSON output
    json_data = {
        'elem_format': args.elem_format,
        'layers': layers,
        'tensor_types': sorted([t for t in TENSOR_TYPES if t in results]),
        'results': []
    }
    
    for tensor_type in sorted(results.keys()):
        print(f"\n{tensor_type.upper()}:")
        if 'forward' in results[tensor_type]:
            for layer in sorted(results[tensor_type]['forward'].keys()):
                stats = results[tensor_type]['forward'][layer]
                print(f"  Layer {layer}: {stats['underflow_percentage']:.4f}% underflow "
                      f"({stats['total_underflow']:,}/{stats['total_elements']:,} elements, "
                      f"{stats['num_tensors']} tensors)")
                
                # Add to JSON data
                json_data['results'].append({
                    'tensor_type': tensor_type,
                    'layer': layer,
                    'pass_type': 'forward',
                    'total_elements': int(stats['total_elements']),
                    'total_underflow': int(stats['total_underflow']),
                    'underflow_percentage': float(stats['underflow_percentage']),
                    'num_tensors': int(stats['num_tensors'])
                })
    
    # Save to JSON file
    layers_str = '_'.join(map(str, layers))
    json_filename = f"{args.elem_format}_underflow_by_layer_{layers_str}.json"
    json_path = output_dir / json_filename
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Generate plots
    plot_filename = f"{args.elem_format}_underflow_by_layer_{layers_str}.png"
    plot_path = output_dir / plot_filename
    plot_underflow_analysis(results, args.elem_format, plot_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
