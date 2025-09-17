#!/usr/bin/env python3
"""
Overflow detection script for tensor files based on numerical format.
Analyzes tensors for overflow and underflow conditions based on their data format.
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path

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
    'mxfp8': {
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

def detect_data_format(filename):
    """
    Extract data format from filename.
    
    Args:
        filename (str): Tensor file name
        
    Returns:
        str: Data format (bf16, hifp8, mxfp8, mxfp4) or None if not found
    """
    for fmt in DATA_TYPE_RANGES.keys():
        if fmt in filename:
            return fmt
    return None

def analyze_tensor_overflow(tensor, data_format):
    """
    Analyze tensor for overflow and underflow conditions.
    
    Args:
        tensor (torch.Tensor): Input tensor
        data_format (str): Data format identifier
        
    Returns:
        dict: Analysis results containing overflow/underflow statistics
    """
    if data_format not in DATA_TYPE_RANGES:
        raise ValueError(f"Unknown data format: {data_format}")
    
    format_info = DATA_TYPE_RANGES[data_format]
    max_val = format_info['max']
    min_denormal = format_info['min_denormal']  # Smallest representable non-zero value
    
    # Convert tensor to numpy for analysis
    if tensor.is_cuda:
        tensor_np = tensor.cpu().numpy()
    else:
        tensor_np = tensor.numpy()
    
    # Handle empty tensors
    if tensor_np.size == 0:
        return {
            'filename': 'empty_tensor',
            'data_format': data_format,
            'total_elements': 0,
            'overflow_count': 0,
            'underflow_count': 0,
            'overflow_percent': 0.0,
            'underflow_percent': 0.0,
            'has_overflow': False,
            'has_underflow': False,
            'has_issues': False,
            'tensor_min': 0.0,
            'tensor_max': 0.0,
            'tensor_mean': 0.0,
            'tensor_std': 0.0,
            'format_min_denormal': min_denormal,
            'format_max': max_val,
            'shape': list(tensor.shape)
        }
    
    # Handle different tensor types
    if tensor_np.dtype == np.complex64 or tensor_np.dtype == np.complex128:
        # For complex tensors, analyze magnitude
        tensor_np = np.abs(tensor_np)
    
    # Count overflow and underflow
    total_elements = tensor_np.size
    
    # Overflow: values exceeding maximum representable value
    overflow_count = np.sum(np.abs(tensor_np) > max_val)
    
    # Underflow: non-zero values closer to zero than smallest representable non-zero value
    # This means |value| > 0 and |value| < min_denormal
    non_zero_mask = tensor_np != 0.0
    abs_tensor = np.abs(tensor_np)
    underflow_count = np.sum(non_zero_mask & (abs_tensor < min_denormal))
    
    # Calculate percentages
    overflow_percent = (overflow_count / total_elements) * 100
    underflow_percent = (underflow_count / total_elements) * 100
    
    # Additional statistics
    tensor_min = np.min(tensor_np)
    tensor_max = np.max(tensor_np)
    tensor_mean = np.mean(tensor_np)
    tensor_std = np.std(tensor_np)
    
    return {
        'filename': os.path.basename(tensor.filename) if hasattr(tensor, 'filename') else 'unknown',
        'data_format': data_format,
        'total_elements': total_elements,
        'overflow_count': overflow_count,
        'underflow_count': underflow_count,
        'overflow_percent': overflow_percent,
        'underflow_percent': underflow_percent,
        'has_overflow': overflow_count > 0,
        'has_underflow': underflow_count > 0,
        'has_issues': (overflow_count > 0) or (underflow_count > 0),
        'tensor_min': tensor_min,
        'tensor_max': tensor_max,
        'tensor_mean': tensor_mean,
        'tensor_std': tensor_std,
        'format_min_denormal': min_denormal,
        'format_max': max_val,
        'shape': list(tensor.shape)
    }

def analyze_file(filepath):
    """
    Analyze a single tensor file for overflow/underflow.
    
    Args:
        filepath (str): Path to tensor file
        
    Returns:
        dict or None: Analysis results or None if file cannot be processed
    """
    try:
        # Extract data format from filename
        filename = os.path.basename(filepath)
        data_format = detect_data_format(filename)
        
        if data_format is None:
            print(f"Warning: Could not detect data format from filename: {filename}")
            return None
        
        # Load tensor
        tensor = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # Handle case where loaded object is not a tensor
        if not isinstance(tensor, torch.Tensor):
            if isinstance(tensor, dict) and 'tensor' in tensor:
                tensor = tensor['tensor']
            elif isinstance(tensor, (list, tuple)) and len(tensor) > 0:
                tensor = tensor[0]
            else:
                print(f"Warning: Loaded object is not a tensor: {filename}")
                return None
        
        # Convert BFloat16 and other unsupported types to Float32 for CPU processing
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        elif tensor.dtype in [torch.float16, torch.half]:
            tensor = tensor.float()
        elif tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            tensor = tensor.float()
        elif tensor.dtype in [torch.uint8]:
            tensor = tensor.float()
        
        # Analyze overflow
        result = analyze_tensor_overflow(tensor, data_format)
        result['filename'] = filename
        result['filepath'] = filepath
        
        return result
        
    except Exception as e:
        print(f"Error processing file {filepath}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Analyze tensor files for overflow/underflow conditions')
    parser.add_argument('input_path', help='Path to tensor file or directory')
    parser.add_argument('--output', '-o', default='./draw/tensor_overflow/', help='Output file for results (default: ./draw/tensor_overflow/)')
    parser.add_argument('--format', '-f', choices=['txt', 'csv', 'json'], default='txt',
                        help='Output format (default: txt)')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Recursively search directories for tensor files')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    results = []
    
    if input_path.is_file():
        # Single file analysis
        result = analyze_file(str(input_path))
        if result:
            results.append(result)
    elif input_path.is_dir():
        # Directory analysis
        pattern = "**/*.pt" if args.recursive else "*.pt"
        for filepath in input_path.glob(pattern):
            result = analyze_file(str(filepath))
            if result:
                results.append(result)
    else:
        print(f"Error: Path does not exist: {input_path}")
        return 1
    
    if not results:
        print("No valid tensor files found or processed.")
        return 1
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        
        # If output is a directory (like the default), generate filename based on input
        if str(output_path).endswith('/') or output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename: same as input but with .log extension
            input_path_obj = Path(args.input_path)
            if input_path_obj.is_file():
                # For single file: change extension to .log
                filename = input_path_obj.stem + '.log'
            else:
                # For directory: use directory name + .log
                filename = input_path_obj.name + '.log'
            
            output_path = output_path / filename
        else:
            # If specific filename provided, create parent directories
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif args.format == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
        else:  # txt format
            with open(output_path, 'w') as f:
                write_text_report(f, results)
        
        print(f"Results saved to: {output_path}")
    else:
        write_text_report(None, results)
    
    return 0

def write_text_report(file_handle, results):
    """Write analysis results in text format."""
    def write_line(line=""):
        if file_handle:
            file_handle.write(line + "\n")
        else:
            print(line)
    
    write_line("=" * 80)
    write_line("TENSOR VALUE OVERFLOW/UNDERFLOW ANALYSIS REPORT")
    write_line("=" * 80)
    write_line("This report shows the PERCENTAGE of tensor values that overflow/underflow")
    write_line("Overflow: Values with |value| > max_representable")
    write_line("Underflow: Non-zero values with 0 < |value| < min_denormal")
    write_line("=" * 80)
    write_line()
    
    # Group results by data format
    format_groups = {}
    for result in results:
        fmt = result['data_format']
        if fmt not in format_groups:
            format_groups[fmt] = []
        format_groups[fmt].append(result)
    
    # Summary statistics
    write_line("SUMMARY STATISTICS")
    write_line("-" * 40)
    for fmt, fmt_results in format_groups.items():
        total_files = len(fmt_results)
        total_elements = sum(r['total_elements'] for r in fmt_results)
        total_overflow = sum(r['overflow_count'] for r in fmt_results)
        total_underflow = sum(r['underflow_count'] for r in fmt_results)
        
        overflow_percent = (total_overflow / total_elements) * 100 if total_elements > 0 else 0
        underflow_percent = (total_underflow / total_elements) * 100 if total_elements > 0 else 0
        
        format_info = DATA_TYPE_RANGES[fmt]
        write_line(f"{fmt.upper()} ({format_info['description']}):")
        write_line(f"  Files analyzed: {total_files}")
        write_line(f"  Total elements: {total_elements:,}")
        write_line(f"  Overflow: {total_overflow:,} ({overflow_percent:.4f}%)")
        write_line(f"  Underflow: {total_underflow:,} ({underflow_percent:.4f}%)")
        write_line(f"  Max Normal: ±{format_info['max_normal']:.2e}")
        write_line(f"  Min Denormal: {format_info['min_denormal']:.2e}")
        write_line(f"  Supports Inf/NaN: {format_info['supports_infinity']}/{format_info['supports_nan']}")
        write_line()
    
    # Detailed results
    write_line("DETAILED RESULTS")
    write_line("-" * 40)
    for fmt, fmt_results in format_groups.items():
        write_line(f"\n{fmt.upper()} FILES:")
        write_line("-" * 20)
        
        for result in fmt_results:
            write_line(f"File: {result['filename']}")
            write_line(f"  Shape: {result['shape']}")
            write_line(f"  Elements: {result['total_elements']:,}")
            write_line(f"  Value range: [{result['tensor_min']:.6f}, {result['tensor_max']:.6f}]")
            write_line(f"  Mean ± Std: {result['tensor_mean']:.6f} ± {result['tensor_std']:.6f}")
            write_line(f"  Overflow: {result['overflow_count']:,} ({result['overflow_percent']:.4f}%)")
            write_line(f"  Underflow: {result['underflow_count']:,} ({result['underflow_percent']:.4f}%)")
            
            if result['overflow_count'] > 0 or result['underflow_count'] > 0:
                write_line("  ⚠️  OVERFLOW/UNDERFLOW DETECTED!")
            
            write_line()

if __name__ == "__main__":
    exit(main())
