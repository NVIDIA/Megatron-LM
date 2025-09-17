#!/usr/bin/env python3
"""
Layer-specific tensor analysis and visualization.
Generates distribution plots and analysis reports for specific layers and ranks.
"""

import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import argparse
from pathlib import Path
from datetime import datetime

# Define numerical format ranges and representable values
DATA_TYPE_INFO = {
    'bf16': {
        'min_normal': 6.103516e-05,     # BFloat16 minimum normal value
        'max_normal': 6.550400e+04,     # BFloat16 maximum normal value
        'min_denormal': 5.960464e-08,   # BFloat16 minimum denormal value
        'max_denormal': 6.097555e-05,   # BFloat16 maximum denormal value
        'min': -6.550400e+04,           # Effective minimum (negative max normal)
        'max': 6.550400e+04,            # Effective maximum (positive max normal)
        'supports_infinity': True,
        'supports_nan': True,
        'description': 'Brain Float 16-bit',
        'color': '#1f77b4',  # Blue
        'representable_values': None  # Too many to enumerate
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
        'description': 'Huawei HiFP8 E4M3',
        'color': '#ff7f0e',  # Orange
        'representable_values': None  # Generate programmatically
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
        'description': 'Microsoft MX FP8 E4M3',
        'color': '#2ca02c',  # Green
        'representable_values': None  # Generate programmatically
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
        'description': 'Microsoft MX FP4 E2M1',
        'color': '#d62728',  # Red
        'representable_values': [
            # Negative values (symmetric to positive)
            -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.500000e+00, -3.000000e+00, -2.500000e+00,
            -2.000000e+00, -1.750000e+00, -1.500000e+00, -1.250000e+00, -1.000000e+00, -5.000000e-01,
            # Zero
            0.0,
            # Positive denormal value (only one: 0.5)
            5.000000e-01,
            # Positive normal values (1.0 and above)
            1.000000e+00, 1.250000e+00, 1.500000e+00, 1.750000e+00, 2.000000e+00,
            2.500000e+00, 3.000000e+00, 3.500000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00
        ]
    }
}

# Define tensor component types for different operations
LINEAR_COMPONENTS = ['input', 'weight', 'output', 'bias', 'hidden']
ATTENTION_COMPONENTS = ['query', 'key', 'value', 'attention_weights', 'output']

def generate_fp8_representable_values(format_name):
    """Generate representable values for FP8 formats based on mxfp.py implementation."""
    values = set([0.0])  # Include zero
    
    if format_name == 'hifp8':
        # HiFP8 format with dynamic mantissa bits based on hifp.py implementation
        # Mantissa bits depend on |exponent|:
        # |e| <= 3: 3 mantissa bits
        # |e| <= 7: 2 mantissa bits  
        # |e| <= 15: 1 mantissa bit
        # Range: exp ∈ [-22, 15], but normal starts from exp=-15
        
        # Generate denormal values (exp < -15)
        for exp in range(-22, -15):  # -22 to -16
            # For very small exponents, use minimal mantissa representation
            value = 2**exp
            values.add(value)
            values.add(-value)
        
        # Generate normal values (exp >= -15)
        for exp in range(-15, 16):  # -15 to 15
            # Determine mantissa bits based on |exp|
            abs_exp = abs(exp)
            if abs_exp <= 3:
                mant_bits = 3
            elif abs_exp <= 7:
                mant_bits = 2
            else:  # abs_exp <= 15
                mant_bits = 1
            
            # Generate all possible mantissa values for this exponent
            for mantissa in range(2**mant_bits):
                value = (1 + mantissa / (2**mant_bits)) * (2**exp)
                
                # Apply the max clamp from hifp.py: max = 2^15 = 32768
                if abs(value) <= 32768.0:
                    values.add(value)
                    values.add(-value)
                
    elif format_name == 'mxfp8':
        # MX FP8 E4M3 format: 4 exponent bits, 3 mantissa bits
        # Based on mxfp.py: ebits=4, mbits=5 (total), emax=8, bias=7
        # Special handling: max_norm = 2^emax * 1.75 = 448.0
        ebits, mbits_frac = 4, 3  # 3 fractional mantissa bits
        bias = 2**(ebits-1) - 1  # 7
        emin = 1 - bias  # -6
        emax = 8  # From mxfp.py code
        
        # Generate denormal values (exponent = emin, mantissa != 0)
        for mantissa in range(1, 2**mbits_frac):  # 1 to 7
            value = mantissa * (2**(emin - mbits_frac))  # mantissa * 2^(-9)
            values.add(value)
            values.add(-value)
        
        # Generate normal values
        for exp in range(emin, emax + 1):  # -6 to 8
            for mantissa in range(2**mbits_frac):  # 0 to 7
                value = (1 + mantissa / (2**mbits_frac)) * (2**exp)
                
                # Special handling for MXFP8: max representable is 448.0
                # This corresponds to exp=8, mantissa=6: (1 + 6/8) * 2^8 = 448.0
                # Skip exp=8, mantissa=7 which would give 480.0
                if exp == emax and mantissa >= 7:  # Skip only mantissa=7 at exp=8
                    continue
                    
                values.add(value)
                values.add(-value)
                
    return sorted(list(values))

# Generate representable values for FP8 formats
DATA_TYPE_INFO['hifp8']['representable_values'] = generate_fp8_representable_values('hifp8')
DATA_TYPE_INFO['mxfp8']['representable_values'] = generate_fp8_representable_values('mxfp8')

def find_matching_files(base_dir, layer, rank, tensor_type, data_format=None):
    """
    Find tensor files matching the specified criteria.
    
    Args:
        base_dir (str): Base directory containing tensor files
        layer (int): Layer number
        rank (int): Rank number
        tensor_type (str): Type of tensor ('linear' or 'attention')
        data_format (str, optional): Specific data format to search
        
    Returns:
        dict: Dictionary mapping data formats to lists of matching files
    """
    base_path = Path(base_dir)
    matching_files = {}
    
    # Define search patterns
    layer_pattern = f"L{layer}_"
    rank_pattern = f"rank{rank:02d}_"
    type_pattern = f"{tensor_type}_"
    
    # Search in each data format directory
    search_formats = [data_format] if data_format else DATA_TYPE_INFO.keys()
    
    for fmt in search_formats:
        format_dir = base_path / fmt
        if not format_dir.exists():
            continue
        
        format_files = []
        for file_path in format_dir.glob("*.pt"):
            filename = file_path.name
            
            # Check if file matches criteria
            if (layer_pattern in filename and 
                rank_pattern in filename and 
                type_pattern in filename and
                fmt in filename):
                format_files.append(file_path)
        
        if format_files:
            matching_files[fmt] = format_files
    
    return matching_files

def extract_component_type(filename):
    """Extract the component type from filename."""
    filename_lower = filename.lower()
    
    # Check for linear components
    for component in LINEAR_COMPONENTS:
        if component in filename_lower:
            return component
    
    # Check for attention components
    for component in ATTENTION_COMPONENTS:
        if component in filename_lower:
            return component
    
    return 'unknown'

def analyze_tensor(tensor_path, data_format):
    """
    Analyze a single tensor file.
    
    Args:
        tensor_path (Path): Path to tensor file
        data_format (str): Data format identifier
        
    Returns:
        dict: Analysis results
    """
    try:
        tensor = torch.load(tensor_path, map_location='cpu', weights_only=False)
        
        if not isinstance(tensor, torch.Tensor):
            if isinstance(tensor, dict) and 'tensor' in tensor:
                tensor = tensor['tensor']
            elif isinstance(tensor, (list, tuple)) and len(tensor) > 0:
                tensor = tensor[0]
            else:
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
        
        # Convert to numpy for analysis
        if tensor.is_cuda:
            tensor_np = tensor.cpu().numpy()
        else:
            tensor_np = tensor.numpy()
        
        # Handle empty tensors
        if tensor_np.size == 0:
            return None
        
        # Handle complex tensors
        if tensor_np.dtype in [np.complex64, np.complex128]:
            tensor_np = np.abs(tensor_np)
        
        # Flatten tensor for distribution analysis
        flat_tensor = tensor_np.flatten()
        
        # Calculate statistics
        stats = {
            'filename': tensor_path.name,
            'data_format': data_format,
            'shape': list(tensor.shape),
            'total_elements': tensor_np.size,
            'min_val': float(np.min(flat_tensor)),
            'max_val': float(np.max(flat_tensor)),
            'mean_val': float(np.mean(flat_tensor)),
            'std_val': float(np.std(flat_tensor)),
            'median_val': float(np.median(flat_tensor)),
            'component_type': extract_component_type(tensor_path.name),
            'tensor_data': flat_tensor,
            'format_info': DATA_TYPE_INFO[data_format]
        }
        
        # Calculate overflow/underflow
        format_max = DATA_TYPE_INFO[data_format]['max']
        min_denormal = DATA_TYPE_INFO[data_format]['min_denormal']  # Smallest representable non-zero value
        
        # Overflow: values exceeding maximum representable value
        overflow_count = np.sum(np.abs(flat_tensor) > format_max)
        
        # Underflow: non-zero values closer to zero than smallest representable non-zero value
        non_zero_mask = flat_tensor != 0.0
        abs_tensor = np.abs(flat_tensor)
        underflow_count = np.sum(non_zero_mask & (abs_tensor < min_denormal))
        
        stats.update({
            'overflow_count': int(overflow_count),
            'underflow_count': int(underflow_count),
            'overflow_percent': float(overflow_count / tensor_np.size * 100),
            'underflow_percent': float(underflow_count / tensor_np.size * 100)
        })
        
        return stats
        
    except Exception as e:
        print(f"Error analyzing {tensor_path}: {str(e)}")
        return None

def create_distribution_plot(tensor_stats_list, layer, rank, tensor_type, output_dir):
    """
    Create comprehensive distribution plots for all data formats.
    
    Args:
        tensor_stats_list (list): List of tensor analysis results
        layer (int): Layer number
        rank (int): Rank number  
        tensor_type (str): Type of tensor
        output_dir (Path): Output directory
    """
    if not tensor_stats_list:
        print("No tensor data to plot")
        return
    
    # Group by component type
    components = {}
    for stats in tensor_stats_list:
        comp_type = stats['component_type']
        if comp_type not in components:
            components[comp_type] = {}
        components[comp_type][stats['data_format']] = stats
    
    # Determine subplot layout
    n_components = len(components)
    n_formats = len(DATA_TYPE_INFO)
    
    # Create figure with subplots
    fig_height = max(12, 4 * n_components)
    fig_width = max(16, 4 * n_formats)
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(n_components, n_formats, figure=fig, hspace=0.3, wspace=0.3)
    
    # Set overall title
    fig.suptitle(f'Layer {layer} Rank {rank} - {tensor_type.capitalize()} Analysis\n'
                 f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                 fontsize=16, fontweight='bold')
    
    # Create subplots for each component and format
    row_idx = 0
    for comp_type, comp_data in components.items():
        col_idx = 0
        
        for fmt, fmt_info in DATA_TYPE_INFO.items():
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            if fmt in comp_data:
                stats = comp_data[fmt]
                create_single_distribution_subplot(ax, stats, fmt_info)
            else:
                # No data for this format
                ax.text(0.5, 0.5, f'No {fmt.upper()}\ndata available', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, style='italic', color='gray')
                ax.set_title(f'{comp_type.capitalize()} - {fmt.upper()}')
                ax.set_xticks([])
                ax.set_yticks([])
            
            col_idx += 1
        
        row_idx += 1
    
    # Save the plot
    output_file = output_dir / f'layer_{layer}_rank_{rank}_{tensor_type}_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Distribution plot saved to: {output_file}")

def create_single_distribution_subplot(ax, stats, fmt_info):
    """Create a single distribution subplot."""
    tensor_data = stats['tensor_data']
    data_format = stats['data_format']
    
    # Create histogram
    n_bins = min(100, max(20, int(np.sqrt(len(tensor_data)))))
    counts, bins, patches = ax.hist(tensor_data, bins=n_bins, alpha=0.7, 
                                   color=fmt_info['color'], density=True,
                                   label='Tensor Values')
    
    # Add representable values overlay for formats that have them
    if fmt_info['representable_values'] and len(fmt_info['representable_values']) < 200:
        rep_values = np.array(fmt_info['representable_values'])
        # Filter to visible range
        data_min, data_max = np.min(tensor_data), np.max(tensor_data)
        margin = (data_max - data_min) * 0.1
        visible_range = (data_min - margin, data_max + margin)
        
        rep_in_range = rep_values[(rep_values >= visible_range[0]) & 
                                 (rep_values <= visible_range[1])]
        
        if len(rep_in_range) > 0:
            # Add vertical lines for representable values
            for val in rep_in_range[::max(1, len(rep_in_range)//50)]:  # Limit to 50 lines
                ax.axvline(val, color='red', alpha=0.3, linewidth=0.5)
    
    # Add format range indicators
    ax.axvline(fmt_info['max'], color='red', linestyle='--', linewidth=2, 
               alpha=0.8, label=f'Max Normal ({fmt_info["max"]:.2e})')
    ax.axvline(-fmt_info['max'], color='red', linestyle='--', linewidth=2, 
               alpha=0.8, label=f'Min Normal ({-fmt_info["max"]:.2e})')
    
    # Add underflow threshold indicators (smallest representable non-zero values)
    min_denormal = fmt_info['min_denormal']
    ax.axvline(min_denormal, color='orange', linestyle=':', linewidth=1.5, 
               alpha=0.8, label=f'Min Denormal (+{min_denormal:.2e})')
    ax.axvline(-min_denormal, color='orange', linestyle=':', linewidth=1.5, 
               alpha=0.8, label=f'Min Denormal (-{min_denormal:.2e})')
    
    # Add statistics text
    stats_text = (f'Min: {stats["min_val"]:.4f}\n'
                  f'Max: {stats["max_val"]:.4f}\n'
                  f'Mean: {stats["mean_val"]:.4f}\n'
                  f'Std: {stats["std_val"]:.4f}\n'
                  f'Shape: {stats["shape"]}')
    
    # Add overflow/underflow warning with clear explanation
    if stats['overflow_count'] > 0 or stats['underflow_count'] > 0:
        warning_text = f'\n⚠️ Overflow: {stats["overflow_count"]} (|val| > max)\n⚠️ Underflow: {stats["underflow_count"]} (0 < |val| < min_denormal)'
        stats_text += warning_text
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set title and labels
    ax.set_title(f'{stats["component_type"].capitalize()} - {data_format.upper()}\n'
                f'{fmt_info["description"]}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    
    # Add legend
    if fmt_info['representable_values'] and len(fmt_info['representable_values']) < 200:
        rep_patch = mpatches.Patch(color='red', alpha=0.3, label='Representable Values')
        ax.legend(handles=[rep_patch], loc='upper right', fontsize=8)

def generate_analysis_report(tensor_stats_list, layer, rank, tensor_type, output_dir):
    """Generate detailed analysis report."""
    if not tensor_stats_list:
        return
    
    report_file = output_dir / f'layer_{layer}_rank_{rank}_{tensor_type}_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"LAYER {layer} RANK {rank} - {tensor_type.upper()} ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Layer: {layer}\n")
        f.write(f"Rank: {rank}\n")
        f.write(f"Tensor Type: {tensor_type}\n")
        f.write(f"Files Analyzed: {len(tensor_stats_list)}\n\n")
        
        # Group by component type
        components = {}
        for stats in tensor_stats_list:
            comp_type = stats['component_type']
            if comp_type not in components:
                components[comp_type] = []
            components[comp_type].append(stats)
        
        # Component analysis
        f.write("COMPONENT ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        for comp_type, comp_stats in components.items():
            f.write(f"\n{comp_type.upper()} Component:\n")
            f.write("─" * 30 + "\n")
            
            for stats in comp_stats:
                f.write(f"\nData Format: {stats['data_format'].upper()}\n")
                f.write(f"File: {stats['filename']}\n")
                f.write(f"Shape: {stats['shape']}\n")
                f.write(f"Total Elements: {stats['total_elements']:,}\n")
                f.write(f"Value Range: [{stats['min_val']:.6f}, {stats['max_val']:.6f}]\n")
                f.write(f"Mean ± Std: {stats['mean_val']:.6f} ± {stats['std_val']:.6f}\n")
                f.write(f"Median: {stats['median_val']:.6f}\n")
                f.write(f"Format Range: [{stats['format_info']['min']}, {stats['format_info']['max']}]\n")
                
                if stats['overflow_count'] > 0:
                    f.write(f"⚠️  OVERFLOW: {stats['overflow_count']:,} elements ({stats['overflow_percent']:.4f}%)\n")
                
                if stats['underflow_count'] > 0:
                    f.write(f"⚠️  UNDERFLOW: {stats['underflow_count']:,} elements ({stats['underflow_percent']:.4f}%)\n")
                
                if stats['overflow_count'] == 0 and stats['underflow_count'] == 0:
                    f.write("✅ No overflow/underflow detected\n")
        
        # Recommendations
        f.write("\n\nRECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        has_issues = any(s['overflow_count'] > 0 or s['underflow_count'] > 0 for s in tensor_stats_list)
        
        if has_issues:
            f.write("Issues detected in this layer/rank combination:\n")
            f.write("• Consider gradient clipping or scaling\n")
            f.write("• Review initialization schemes\n")
            f.write("• Monitor numerical stability during training\n")
            f.write("• Consider mixed precision training strategies\n")
        else:
            f.write("No numerical issues detected in this layer/rank combination.\n")
            f.write("Values are within representable ranges for all formats.\n")
    
    print(f"Analysis report saved to: {report_file}")

def main():
    """Main function for layer analysis."""
    parser = argparse.ArgumentParser(description='Analyze and visualize layer-specific tensor distributions')
    parser.add_argument('--layer', type=int, required=True, help='Layer number to analyze')
    parser.add_argument('--rank', type=int, required=True, help='Rank number to analyze')
    parser.add_argument('--type', choices=['linear', 'attention'], required=True, 
                        help='Type of tensor operation to analyze')
    parser.add_argument('--base-dir', default='enhanced_tensor_logs',
                        help='Base directory containing tensor files (default: enhanced_tensor_logs)')
    parser.add_argument('--output-dir', default='./draw/layer_analysis/',
                        help='Output directory for plots and reports (default: ./draw/layer_analysis/)')
    parser.add_argument('--format', choices=['bf16', 'hifp8', 'mxfp8', 'mxfp4'],
                        help='Specific data format to analyze (default: all formats)')
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return 1
    
    print(f"Analyzing Layer {args.layer}, Rank {args.rank}, Type: {args.type}")
    print("=" * 60)
    
    # Find matching files
    matching_files = find_matching_files(base_dir, args.layer, args.rank, args.type, args.format)
    
    if not matching_files:
        print(f"No matching files found for Layer {args.layer}, Rank {args.rank}, Type: {args.type}")
        return 1
    
    # Analyze tensors
    tensor_stats_list = []
    
    for data_format, file_list in matching_files.items():
        print(f"\nAnalyzing {data_format.upper()} files:")
        
        for file_path in file_list:
            print(f"  Processing: {file_path.name}")
            stats = analyze_tensor(file_path, data_format)
            if stats:
                tensor_stats_list.append(stats)
    
    if not tensor_stats_list:
        print("No valid tensor data found for analysis")
        return 1
    
    print(f"\nSuccessfully analyzed {len(tensor_stats_list)} tensor files")
    
    # Generate visualizations and reports
    print("\nGenerating distribution plots...")
    create_distribution_plot(tensor_stats_list, args.layer, args.rank, args.type, output_dir)
    
    print("Generating analysis report...")
    generate_analysis_report(tensor_stats_list, args.layer, args.rank, args.type, output_dir)
    
    print("\nAnalysis complete!")
    print(f"Output files saved in: {output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())
