#!/usr/bin/env python3
"""
Test script for quantization error analysis

Tests quant_hif8 and _quantize_mx quantization functions on BF16 tensors
and calculates error metrics (MSE, Max Error, Relative Error).

Usage:
    python test_dtype.py <tensor_file> [--format hifp8|fp8_e4m3|fp8_e5m2|fp4_e2m1]
"""

import torch
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import quantization functions
from quant.hifp import quant_hif8
from quant.mxfp import _quantize_mx


def analyze_quantization_error(original: torch.Tensor, quantized: torch.Tensor, method_name: str):
    """Calculate and print quantization error metrics."""
    # Convert to float32 for accurate error calculation
    if original.dtype != torch.float32:
        original = original.float()
    if quantized.dtype != torch.float32:
        quantized = quantized.float()
    
    # Calculate errors
    mse = torch.mean((original - quantized) ** 2)
    max_err = torch.max(torch.abs(original - quantized))
    
    # Calculate relative error
    original_squared_mean = torch.mean(original ** 2)
    relative_error = mse / original_squared_mean if original_squared_mean > 0 else 0.0
    
    # Calculate statistics
    original_max = torch.max(original)
    original_min = torch.min(original)
    quantized_max = torch.max(quantized)
    quantized_min = torch.min(quantized)
    
    print(f"\n{'='*60}")
    print(f"{method_name} Quantization Results")
    print(f"{'='*60}")
    print(f"Original tensor:")
    print(f"  Shape: {original.shape}")
    print(f"  Max: {original_max:.6f}, Min: {original_min:.6f}")
    print(f"  Mean: {torch.mean(original):.6f}, Std: {torch.std(original):.6f}")
    print(f"\nQuantized tensor:")
    print(f"  Shape: {quantized.shape}")
    print(f"  Max: {quantized_max:.6f}, Min: {quantized_min:.6f}")
    print(f"  Mean: {torch.mean(quantized):.6f}, Std: {torch.std(quantized):.6f}")
    print(f"\nError Metrics:")
    print(f"  MSE: {mse:.20f}")
    print(f"  Max Error: {max_err:.20f}")
    print(f"  Relative Error: {relative_error:.20f}")
    print(f"  Has NaN: {torch.isnan(quantized).any().item()}")
    print(f"  Has Inf: {torch.isinf(quantized).any().item()}")
    print(f"{'='*60}\n")
    
    return {
        'mse': mse.item(),
        'max_error': max_err.item(),
        'relative_error': relative_error.item(),
        'has_nan': torch.isnan(quantized).any().item(),
        'has_inf': torch.isinf(quantized).any().item(),
    }


def test_hifp8_quantization(tensor: torch.Tensor):
    """Test HiFP8 quantization."""
    print("Testing HiFP8 quantization...")
    
    # Convert to float32 if needed
    if tensor.dtype == torch.bfloat16:
        tensor_fp32 = tensor.float()
    else:
        tensor_fp32 = tensor.float()
    
    # Quantize
    quantized = quant_hif8(tensor_fp32)
    
    # Analyze error
    return analyze_quantization_error(tensor_fp32, quantized, "HiFP8")


def test_mxfp_quantization(tensor: torch.Tensor, elem_format: str, block_size: int = 0):
    """Test MXFP quantization."""
    print(f"Testing MXFP quantization ({elem_format})...")
    
    # Convert to float32 if needed
    if tensor.dtype == torch.bfloat16:
        tensor_fp32 = tensor.float()
    else:
        tensor_fp32 = tensor.float()
    
    # Determine axes based on tensor shape
    # For 2D tensors, use last dimension for shared exponent
    if tensor_fp32.ndim >= 2:
        axes = -1
    else:
        axes = None
    
    # Quantize
    quantized = _quantize_mx(
        tensor_fp32,
        scale_bits=8,
        elem_format=elem_format,
        shared_exp_method="max",
        axes=axes,
        block_size=block_size,
        round="nearest",
        flush_fp32_subnorms=False,
        scaling_control="max"
    )
    
    # Analyze error
    return analyze_quantization_error(tensor_fp32, quantized, f"MXFP ({elem_format})")


def main():
    parser = argparse.ArgumentParser(
        description='Test quantization functions on BF16 tensors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_dtype.py tensor.pt
  python test_dtype.py tensor.pt --format hifp8
  python test_dtype.py tensor.pt --format fp8_e4m3
  python test_dtype.py tensor.pt --format fp8_e5m2
  python test_dtype.py tensor.pt --format fp4_e2m1
  python test_dtype.py tensor.pt --all-formats
        """
    )
    parser.add_argument('tensor_file', type=str, help='Path to the tensor file (.pt)')
    parser.add_argument('--format', type=str, 
                       choices=['hifp8', 'fp8_e4m3', 'fp8_e5m2', 'fp4_e2m1'],
                       default='hifp8',
                       help='Quantization format to test (default: hifp8)')
    parser.add_argument('--all-formats', action='store_true',
                       help='Test all quantization formats')
    parser.add_argument('--block-size', type=int, default=0,
                       help='Block size for MXFP quantization (default: 0, no blocking)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run on (default: cpu)')
    
    args = parser.parse_args()
    
    # Load tensor
    tensor_path = Path(args.tensor_file)
    if not tensor_path.exists():
        print(f"Error: Tensor file not found: {tensor_path}")
        return 1
    
    print(f"Loading tensor from: {tensor_path}")
    try:
        tensor = torch.load(tensor_path, map_location=args.device, weights_only=False)
        
        # Handle different tensor formats
        if not isinstance(tensor, torch.Tensor):
            if isinstance(tensor, dict) and 'tensor' in tensor:
                tensor = tensor['tensor']
            elif isinstance(tensor, (list, tuple)) and len(tensor) > 0:
                tensor = tensor[0]
            else:
                print(f"Error: Unsupported tensor format in file")
                return 1
        
        # Convert to bfloat16 if needed
        if tensor.dtype != torch.bfloat16:
            if tensor.dtype in [torch.float32, torch.float]:
                tensor = tensor.to(torch.bfloat16)
            else:
                print(f"Warning: Tensor dtype is {tensor.dtype}, converting to bfloat16")
                tensor = tensor.float().to(torch.bfloat16)
        
        print(f"Loaded tensor:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Max: {torch.max(tensor.float()):.6f}, Min: {torch.min(tensor.float()):.6f}")
        print(f"  Device: {tensor.device}")
        
    except Exception as e:
        print(f"Error loading tensor: {e}")
        return 1
    
    # Test quantization
    results = {}
    
    if args.all_formats:
        # Test all formats
        formats_to_test = [
            ('hifp8', None),
            ('fp8_e4m3', 'fp8_e4m3'),
            ('fp8_e5m2', 'fp8_e5m2'),
            ('fp4_e2m1', 'fp4_e2m1'),
        ]
        
        for format_name, format_param in formats_to_test:
            try:
                if format_name == 'hifp8':
                    result = test_hifp8_quantization(tensor)
                else:
                    result = test_mxfp_quantization(tensor, format_param, args.block_size)
                results[format_name] = result
            except Exception as e:
                print(f"Error testing {format_name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Test single format
        try:
            if args.format == 'hifp8':
                result = test_hifp8_quantization(tensor)
            else:
                result = test_mxfp_quantization(tensor, args.format, args.block_size)
            results[args.format] = result
        except Exception as e:
            print(f"Error testing {args.format}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("Summary Comparison")
        print(f"{'='*60}")
        print(f"{'Format':<15} {'MSE':<20} {'Max Error':<15} {'Relative Error':<20}")
        print(f"{'-'*60}")
        for format_name, result in results.items():
            print(f"{format_name:<15} {result['mse']:<20.10e} {result['max_error']:<15.10e} {result['relative_error']:<20.10e}")
        print(f"{'='*60}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

