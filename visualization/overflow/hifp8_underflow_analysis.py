#!/usr/bin/env python3
"""
HiFP8 Underflow Analysis Tool

Analyzes underflow when simulating HiFP8 quantization on BF16 tensors.
Usage: python hifp8_underflow_analysis.py --layer 1,8,15,16
"""

import re
import torch
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

# HiFP8 range
HIFP8_MIN_DENORMAL = 2.384186e-07  # HiFP8 minimum denormal value (2^-22)


def parse_layers(layer_str: str) -> list:
    """Parse layer argument: '1,8,15,16' -> [1, 8, 15, 16]"""
    return [int(l.strip()) for l in layer_str.split(',')]


def get_layer_and_pass(filename: str) -> tuple:
    """Extract layer number and pass type (forward/backward) from filename."""
    # Extract layer: _L14_ -> 14
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


def analyze_tensor(file_path: Path) -> dict:
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
    
    # Flatten and analyze
    flat = tensor.flatten()
    total = flat.numel()
    if total == 0:
        return None
    
    # Calculate underflow: non-zero values < min_denormal
    abs_vals = torch.abs(flat)
    non_zero = flat != 0.0
    underflow = non_zero & (abs_vals < HIFP8_MIN_DENORMAL)
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
    
    print(f"Analyzing layers: {layers}")
    for file_path in bf16_dir.glob("*.pt"):
        layer, pass_type = get_layer_and_pass(file_path.name)
        if layer in layers and pass_type:
            result = analyze_tensor(file_path)
            if result:
                results[layer][pass_type].append(result)
    
    # Print report
    print("\n" + "=" * 60)
    print("HiFP8 UNDERFLOW ANALYSIS")
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
    
    return 0


if __name__ == "__main__":
    exit(main())
