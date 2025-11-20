#!/usr/bin/env python3
"""
HiFP8 Underflow Analysis Tool

Analyzes underflow when simulating HiFP8 quantization on BF16 tensors.
Usage: python hifp8_underflow_analysis.py --layer 1,8,15,16
"""

import re
import json
import torch
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
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

def parse_layers(layer_str: str) -> list:
    """Parse layer argument: '1,8,15,16' -> [1, 8, 15, 16]"""
    return [int(l.strip()) for l in layer_str.split(',')]


def get_layer_and_pass(filename: str) -> tuple:
    """Extract layer number and pass type (forward/backward) from filename."""
    layer_match = re.search(r'_L(\d+)_', filename)
    if not layer_match:
        return None, None
    
    layer = int(layer_match.group(1))
    
    # Extract pass type
    if '_forward_' in filename:
        pass_type = 'forward'
    elif '_backward_' in filename:
        pass_type = 'backward'
    else:
        pass_type = None
    
    return layer, pass_type

def remove_scaling(tensor: torch.Tensor,elem_format: str) -> torch.Tensor:
    from quant.mxfp_scaling_test import remove_scaling
    if elem_format == 'hifp8':
        return tensor
    elif elem_format == 'mxfp8e4m3':
        return remove_scaling(tensor,"fp8_e4m3")
    elif elem_format == 'mxfp8e5m2':
        return remove_scaling(tensor,"fp8_e5m2")
    elif elem_format == 'mxfp4':
        return remove_scaling(tensor,"fp4_e2m1")
    else:
        raise ValueError(f"Unsupported element format: {elem_format}")

def analyze_tensor(file_path: Path,elem_format: str) -> dict:
    """Analyze tensor for HiFP8 underflow."""
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

    tensor = remove_scaling(tensor,elem_format)
    
    # Flatten and analyze
    flat = tensor.flatten()
    total = flat.numel()
    if total == 0:
        return None
    
    # Calculate underflow: non-zero values < min_denormal
    abs_vals = torch.abs(flat)
    non_zero = flat != 0.0
    if elem_format in DATA_TYPE_RANGES:
        min_denormal = DATA_TYPE_RANGES[elem_format]['min_denormal']
    else:
        raise ValueError(f"Unsupported element format: {elem_format}")
    
    underflow = non_zero & (abs_vals < min_denormal)
    underflow_count = torch.sum(underflow).item()
    underflow_pct = (underflow_count / total) * 100.0
    
    # Get metadata
    layer, pass_type = get_layer_and_pass(file_path.name)
    if layer is None or pass_type is None:
        return None
    
    return {
        'layer': layer,
        'pass_type': pass_type,
        'total_elements': int(total),
        'underflow_count': int(underflow_count),
        'underflow_percentage': float(underflow_pct),
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze HiFP8 underflow for BF16 tensors')
    parser.add_argument('--layer', required=True, help='Layer numbers: 1,8,15,16')
    parser.add_argument('--base-dir', default='enhanced_tensor_logs', help='Base directory')
    parser.add_argument('--elem-format', default='hifp8', choices=['hifp8', 'mxfp8e4m3', 'mxfp8e5m2', 'mxfp4'], help='Element format: hifp8, mxfp8e4m3, mxfp8e5m2, mxfp4')
    args = parser.parse_args()
    
    # Parse layers
    layers = parse_layers(args.layer)
    base_dir = Path(args.base_dir)
    bf16_dir = base_dir / 'bf16'
    
    if not bf16_dir.exists():
        print(f"Error: Directory not found: {bf16_dir}")
        return 1
    
    # Find and analyze files
    results = defaultdict(lambda: defaultdict(list))
    
    # Collect all matching files first
    all_files = list(bf16_dir.glob("*.pt"))
    matching_files = []
    for file_path in all_files:
        layer, pass_type = get_layer_and_pass(file_path.name)
        if layer in layers and pass_type:
            matching_files.append(file_path)
    
    print(f"Analyzing layers: {layers}")
    print(f"Found {len(matching_files)} matching files to analyze")
    
    for file_path in tqdm(matching_files, desc="Processing tensors", unit="file"):
        result = analyze_tensor(file_path,args.elem_format)
        if result:
            results[result['layer']][result['pass_type']].append(result)
    
    # Prepare data for JSON output
    json_data = {
        'elem_format': args.elem_format,
        'layers': sorted(results.keys()),
        'results': []
    }
    
    # Print report and collect data
    print("\n" + "=" * 60)
    print(f"{args.elem_format.upper()} UNDERFLOW ANALYSIS")
    print("=" * 60)
    
    for layer in sorted(results.keys()):
        print(f"\nLayer {layer}:")
        for pass_type in ['forward', 'backward']:
            if pass_type not in results[layer]:
                continue
            
            pass_results = results[layer][pass_type]
            total_elements = sum(r['total_elements'] for r in pass_results)
            total_underflow = sum(r['underflow_count'] for r in pass_results)
            underflow_pct = (total_underflow / total_elements * 100.0) if total_elements > 0 else 0.0
            
            print(f"  {pass_type.upper()}: {underflow_pct:.4f}% underflow "
                  f"({total_underflow:,}/{total_elements:,} elements, {len(pass_results)} tensors)")
            
            # Add to JSON data
            json_data['results'].append({
                'layer': layer,
                'pass_type': pass_type,
                'total_elements': int(total_elements),
                'total_underflow': int(total_underflow),
                'underflow_percentage': float(underflow_pct),
                'num_tensors': len(pass_results)
            })
    
    # Save to JSON file
    json_filename = f"{args.elem_format}_underflow.json"
    json_path = base_dir / json_filename
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
