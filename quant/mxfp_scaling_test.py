#!/usr/bin/env python3
"""
MXFP Scaling Test Tool
Tests different scaling strategies for MXFP quantization and evaluates their impact on accuracy.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys
import os

# Add the parent directory to path to import mxfp module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant.mxfp import _quantize_mx, _get_format_params, ElemFormat

def calculate_metrics(original_tensor, quantized_tensor):
    """
    Calculate various metrics between original and quantized tensors.
    
    Args:
        original_tensor (torch.Tensor): Original BF16 tensor
        quantized_tensor (torch.Tensor): Quantized tensor
        
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    # Convert to float32 for calculation
    orig_f32 = original_tensor.float()
    quant_f32 = quantized_tensor.float()
    
    # MSE (Mean Squared Error)
    mse = torch.mean((orig_f32 - quant_f32) ** 2).item()
    
    # RMSE (Root Mean Squared Error)
    rmse = torch.sqrt(torch.mean((orig_f32 - quant_f32) ** 2)).item()
    
    # Cosine Similarity
    orig_flat = orig_f32.flatten()
    quant_flat = quant_f32.flatten()
    
    # Avoid division by zero
    orig_norm = torch.norm(orig_flat)
    quant_norm = torch.norm(quant_flat)
    
    if orig_norm > 0 and quant_norm > 0:
        cosine_sim = torch.dot(orig_flat, quant_flat) / (orig_norm * quant_norm)
        cosine_sim = cosine_sim.item()
    else:
        cosine_sim = 1.0 if orig_norm == 0 and quant_norm == 0 else 0.0
    
    # PSNR (Peak Signal-to-Noise Ratio)
    if mse > 0:
        # Use the maximum value in original tensor as peak signal
        max_val = torch.max(torch.abs(orig_f32)).item()
        psnr = 20 * np.log10(max_val / np.sqrt(mse)) if max_val > 0 else float('inf')
    else:
        psnr = float('inf')
    
    # MAE (Mean Absolute Error)
    mae = torch.mean(torch.abs(orig_f32 - quant_f32)).item()
    
    # Maximum Absolute Error
    max_abs_error = torch.max(torch.abs(orig_f32 - quant_f32)).item()
    
    # Relative Error (percentage)
    orig_mean_abs = torch.mean(torch.abs(orig_f32)).item()
    relative_error = (mae / orig_mean_abs * 100) if orig_mean_abs > 0 else 0.0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'cosine_similarity': cosine_sim,
        'psnr': psnr,
        'mae': mae,
        'max_abs_error': max_abs_error,
        'relative_error': relative_error
    }

def test_scaling_levels(input_tensor, elem_format='fp8_e4m3', scale_bits=8, 
                       max_scale_exp=10, min_scale_exp=-10, num_levels=21):
    """
    Test different scaling levels for MXFP quantization.
    
    Args:
        input_tensor (torch.Tensor): Input BF16 tensor
        elem_format (str): Element format for quantization
        scale_bits (int): Number of scale bits
        max_scale_exp (int): Maximum scale exponent (aligned with max value)
        min_scale_exp (int): Minimum scale exponent (aligned with min value)
        num_levels (int): Number of scaling levels to test
        
    Returns:
        dict: Results for each scaling level
    """
    # Get format parameters
    ebits, mbits, emax, max_norm, min_norm = _get_format_params(elem_format)
    
    # Generate scale exponents from max to min
    scale_exponents = np.linspace(max_scale_exp, min_scale_exp, num_levels)
    
    results = {
        'scale_exponents': scale_exponents.tolist(),
        'metrics': {},
        'elem_format': elem_format,
        'scale_bits': scale_bits,
        'format_params': {
            'ebits': ebits,
            'mbits': mbits,
            'emax': emax,
            'max_norm': max_norm,
            'min_norm': min_norm
        }
    }
    
    print(f"Testing {num_levels} scaling levels from {max_scale_exp} to {min_scale_exp}")
    print(f"Element format: {elem_format} (e{ebits}m{mbits})")
    print(f"Scale bits: {scale_bits}")
    print("-" * 60)
    
    for i, scale_exp in enumerate(scale_exponents):
        print(f"Testing scale exponent {scale_exp:.2f} ({i+1}/{num_levels})...")
        
        # Create a custom quantize function with fixed scale exponent
        quantized_tensor = quantize_with_fixed_scale(
            input_tensor, elem_format, scale_bits, scale_exp, 
            ebits, mbits, max_norm
        )
        
        # Calculate metrics
        metrics = calculate_metrics(input_tensor, quantized_tensor)
        
        # Store results
        results['metrics'][f'scale_{i}'] = {
            'scale_exponent': float(scale_exp),
            'metrics': metrics
        }
        
        # Print current metrics
        print(f"  MSE: {metrics['mse']:.6e}, "
              f"Cosine Sim: {metrics['cosine_similarity']:.6f}, "
              f"PSNR: {metrics['psnr']:.2f} dB")
    
    return results

def quantize_with_fixed_scale(input_tensor, elem_format, scale_bits, scale_exp,
                             ebits, mbits, max_norm, axes=None, block_size=0):
    """
    Custom quantization function with fixed scale exponent.
    
    Args:
        input_tensor (torch.Tensor): Input tensor
        elem_format (str): Element format
        scale_bits (int): Number of scale bits
        scale_exp (float): Fixed scale exponent
        ebits (int): Exponent bits
        mbits (int): Mantissa bits
        max_norm (float): Maximum normal value
        axes (list): Axes for shared exponent calculation
        block_size (int): Block size for tiling
        
    Returns:
        torch.Tensor: Quantized tensor
    """
    A = input_tensor.clone()
    
    # Use fixed scale exponent instead of calculating from max value
    shared_exp = torch.full_like(A, scale_exp)
    
    # Offset by the largest representable exponent in element format
    shared_exp = shared_exp - ebits + 1  # Adjust for format-specific bias
    
    # Clamp scale exponent to valid range
    scale_emax = 2**(scale_bits-1) - 1
    shared_exp = torch.clamp(shared_exp, -scale_emax, scale_emax)
    
    # Apply scaling
    A = A / (2**shared_exp)
    
    # Quantize element-wise
    from mxfp import _quantize_elemwise_core
    A = _quantize_elemwise_core(
        A, mbits, ebits, max_norm, round='nearest',
        allow_denorm=True, saturate_normals=True
    )
    
    # Undo scaling
    A = A * (2**shared_exp)
    
    return A

def plot_scaling_results(results, output_path):
    """
    Create comprehensive plots showing scaling test results.
    
    Args:
        results (dict): Results from test_scaling_levels
        output_path (Path): Output directory for plots
    """
    scale_exponents = results['scale_exponents']
    elem_format = results['elem_format']
    
    # Extract metrics for plotting
    metrics_data = {}
    for metric_name in ['mse', 'rmse', 'cosine_similarity', 'psnr', 'mae', 'max_abs_error', 'relative_error']:
        metrics_data[metric_name] = []
        for i in range(len(scale_exponents)):
            scale_key = f'scale_{i}'
            if scale_key in results['metrics']:
                metrics_data[metric_name].append(results['metrics'][scale_key]['metrics'][metric_name])
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f'MXFP Scaling Test Results - {elem_format.upper()}', fontsize=16, fontweight='bold')
    
    # Plot 1: MSE
    axes[0, 0].semilogy(scale_exponents, metrics_data['mse'], 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Scale Exponent')
    axes[0, 0].set_ylabel('MSE (log scale)')
    axes[0, 0].set_title('Mean Squared Error vs Scale Exponent')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Cosine Similarity
    axes[0, 1].plot(scale_exponents, metrics_data['cosine_similarity'], 'g-o', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Scale Exponent')
    axes[0, 1].set_ylabel('Cosine Similarity')
    axes[0, 1].set_title('Cosine Similarity vs Scale Exponent')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Plot 3: PSNR
    # Handle infinite PSNR values
    psnr_values = metrics_data['psnr']
    psnr_finite = [p if p != float('inf') else 1000 for p in psnr_values]  # Cap at 1000 for plotting
    
    axes[1, 0].plot(scale_exponents, psnr_finite, 'r-o', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Scale Exponent')
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].set_title('Peak Signal-to-Noise Ratio vs Scale Exponent')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: MAE
    axes[1, 1].semilogy(scale_exponents, metrics_data['mae'], 'm-o', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('Scale Exponent')
    axes[1, 1].set_ylabel('MAE (log scale)')
    axes[1, 1].set_title('Mean Absolute Error vs Scale Exponent')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Maximum Absolute Error
    axes[2, 0].semilogy(scale_exponents, metrics_data['max_abs_error'], 'c-o', linewidth=2, markersize=4)
    axes[2, 0].set_xlabel('Scale Exponent')
    axes[2, 0].set_ylabel('Max Absolute Error (log scale)')
    axes[2, 0].set_title('Maximum Absolute Error vs Scale Exponent')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Relative Error
    axes[2, 1].plot(scale_exponents, metrics_data['relative_error'], 'orange', marker='o', linewidth=2, markersize=4)
    axes[2, 1].set_xlabel('Scale Exponent')
    axes[2, 1].set_ylabel('Relative Error (%)')
    axes[2, 1].set_title('Relative Error vs Scale Exponent')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Add format information
    format_params = results['format_params']
    info_text = f"Format: {elem_format}\nE-bits: {format_params['ebits']}, M-bits: {format_params['mbits']}\n"
    info_text += f"Max Normal: ±{format_params['max_norm']:.1e}\nMin Normal: {format_params['min_norm']:.1e}"
    
    fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.15)
    
    # Save plot
    plot_path = output_path / f'mxfp_scaling_test_{elem_format}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Scaling test plots saved to: {plot_path}")
    
    # Create summary plot with key metrics
    create_summary_plot(results, output_path)

def create_summary_plot(results, output_path):
    """Create a summary plot with the most important metrics."""
    scale_exponents = results['scale_exponents']
    elem_format = results['elem_format']
    
    # Extract key metrics
    mse_values = []
    cosine_sim_values = []
    psnr_values = []
    
    for i in range(len(scale_exponents)):
        scale_key = f'scale_{i}'
        if scale_key in results['metrics']:
            metrics = results['metrics'][scale_key]['metrics']
            mse_values.append(metrics['mse'])
            cosine_sim_values.append(metrics['cosine_similarity'])
            psnr_values.append(metrics['psnr'])
    
    # Handle infinite PSNR values
    psnr_finite = [p if p != float('inf') else 1000 for p in psnr_values]
    
    # Create summary plot
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot MSE and PSNR on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Scale Exponent', fontsize=12)
    ax1.set_ylabel('MSE (log scale)', color=color1, fontsize=12)
    line1 = ax1.semilogy(scale_exponents, mse_values, 'o-', color=color1, linewidth=2, markersize=6, label='MSE')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for cosine similarity
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Cosine Similarity', color=color2, fontsize=12)
    line2 = ax2.plot(scale_exponents, cosine_sim_values, 's-', color=color2, linewidth=2, markersize=6, label='Cosine Similarity')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 1])
    
    # Add PSNR as dashed line on ax1
    ax1_2 = ax1.twinx()
    ax1_2.spines['right'].set_position(('outward', 60))
    color3 = 'tab:red'
    ax1_2.set_ylabel('PSNR (dB)', color=color3, fontsize=12)
    line3 = ax1_2.plot(scale_exponents, psnr_finite, '^-', color=color3, linewidth=2, markersize=6, linestyle='--', label='PSNR')
    ax1_2.tick_params(axis='y', labelcolor=color3)
    
    # Add title and legend
    plt.title(f'MXFP Scaling Test Summary - {elem_format.upper()}\nKey Metrics vs Scale Exponent', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save summary plot
    summary_path = output_path / f'mxfp_scaling_summary_{elem_format}.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Summary plot saved to: {summary_path}")

def save_results_to_file(results, output_path):
    """Save detailed results to a text file."""
    results_path = output_path / f'mxfp_scaling_results_{results["elem_format"]}.txt'
    
    with open(results_path, 'w') as f:
        f.write("MXFP Scaling Test Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Element Format: {results['elem_format']}\n")
        f.write(f"Scale Bits: {results['scale_bits']}\n")
        f.write(f"Format Parameters: {results['format_params']}\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 30 + "\n")
        
        for i, scale_exp in enumerate(results['scale_exponents']):
            scale_key = f'scale_{i}'
            if scale_key in results['metrics']:
                metrics = results['metrics'][scale_key]['metrics']
                f.write(f"Scale Exponent {scale_exp:.2f}:\n")
                f.write(f"  MSE: {metrics['mse']:.6e}\n")
                f.write(f"  RMSE: {metrics['rmse']:.6e}\n")
                f.write(f"  Cosine Similarity: {metrics['cosine_similarity']:.6f}\n")
                f.write(f"  PSNR: {metrics['psnr']:.2f} dB\n")
                f.write(f"  MAE: {metrics['mae']:.6e}\n")
                f.write(f"  Max Absolute Error: {metrics['max_abs_error']:.6e}\n")
                f.write(f"  Relative Error: {metrics['relative_error']:.2f}%\n\n")
    
    print(f"Detailed results saved to: {results_path}")

def main():
    """Main function for MXFP scaling test."""
    parser = argparse.ArgumentParser(description='Test different scaling strategies for MXFP quantization')
    parser.add_argument('input_tensor', help='Path to input BF16 tensor file (.pt)')
    parser.add_argument('--output-dir', default=None, 
                        help='Output directory for results (default: ./draw/scaling_analysis/{tensor_name}/)')
    parser.add_argument('--elem-format', default='fp8_e4m3', 
                        choices=['fp8_e4m3', 'fp8_e5m2', 'fp4_e2m1', 'fp6_e3m2', 'fp6_e2m3'],
                        help='Element format for quantization (default: fp8_e4m3)')
    parser.add_argument('--scale-bits', type=int, default=8,
                        help='Number of scale bits (default: 8)')
    parser.add_argument('--max-scale-exp', type=int, default=10,
                        help='Maximum scale exponent (default: 10)')
    parser.add_argument('--min-scale-exp', type=int, default=-10,
                        help='Minimum scale exponent (default: -10)')
    parser.add_argument('--num-levels', type=int, default=21,
                        help='Number of scaling levels to test (default: 21)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_tensor)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        return 1
    
    if not input_path.is_file():
        print(f"Error: Input path is not a file: {input_path}")
        return 1
    
    # Setup output directory
    if args.output_dir is None:
        # Generate output directory based on tensor name
        tensor_name = input_path.stem  # Get filename without extension
        output_dir = Path(f"./draw/scaling_analysis/{tensor_name}")
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading input tensor: {input_path.name}")
    print("=" * 60)
    
    # Load input tensor
    try:
        input_tensor = torch.load(str(input_path), map_location='cpu', weights_only=False)
        
        # Handle case where loaded object is not a tensor
        if not isinstance(input_tensor, torch.Tensor):
            if isinstance(input_tensor, dict) and 'tensor' in input_tensor:
                input_tensor = input_tensor['tensor']
            elif isinstance(input_tensor, (list, tuple)) and len(input_tensor) > 0:
                input_tensor = input_tensor[0]
            else:
                print(f"Error: Loaded object is not a tensor: {input_path.name}")
                return 1
        
        # Convert to BF16 if needed
        if input_tensor.dtype != torch.bfloat16:
            print(f"Converting tensor from {input_tensor.dtype} to bfloat16")
            input_tensor = input_tensor.bfloat16()
        
        print(f"Tensor shape: {input_tensor.shape}")
        print(f"Tensor dtype: {input_tensor.dtype}")
        print(f"Value range: [{torch.min(input_tensor):.6f}, {torch.max(input_tensor):.6f}]")
        print(f"Mean ± Std: {torch.mean(input_tensor.float()):.6f} ± {torch.std(input_tensor.float()):.6f}")
        
    except Exception as e:
        print(f"Error loading tensor: {str(e)}")
        return 1
    
    print(f"\nStarting scaling test...")
    print(f"Element format: {args.elem_format}")
    print(f"Scale bits: {args.scale_bits}")
    print(f"Scale exponent range: [{args.max_scale_exp}, {args.min_scale_exp}]")
    print(f"Number of levels: {args.num_levels}")
    
    # Run scaling test
    results = test_scaling_levels(
        input_tensor, 
        elem_format=args.elem_format,
        scale_bits=args.scale_bits,
        max_scale_exp=args.max_scale_exp,
        min_scale_exp=args.min_scale_exp,
        num_levels=args.num_levels
    )
    
    # Save results to file
    save_results_to_file(results, output_dir)
    
    # Generate plots unless disabled
    if not args.no_plots:
        plot_scaling_results(results, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SCALING TEST SUMMARY")
    print("=" * 60)
    
    # Find best and worst cases
    best_cosine_idx = 0
    worst_cosine_idx = 0
    best_mse_idx = 0
    worst_mse_idx = 0
    
    for i in range(len(results['scale_exponents'])):
        scale_key = f'scale_{i}'
        if scale_key in results['metrics']:
            metrics = results['metrics'][scale_key]['metrics']
            
            if metrics['cosine_similarity'] > results['metrics'][f'scale_{best_cosine_idx}']['metrics']['cosine_similarity']:
                best_cosine_idx = i
            if metrics['cosine_similarity'] < results['metrics'][f'scale_{worst_cosine_idx}']['metrics']['cosine_similarity']:
                worst_cosine_idx = i
                
            if metrics['mse'] < results['metrics'][f'scale_{best_mse_idx}']['metrics']['mse']:
                best_mse_idx = i
            if metrics['mse'] > results['metrics'][f'scale_{worst_mse_idx}']['metrics']['mse']:
                worst_mse_idx = i
    
    best_cosine_scale = results['scale_exponents'][best_cosine_idx]
    best_cosine_metrics = results['metrics'][f'scale_{best_cosine_idx}']['metrics']
    
    best_mse_scale = results['scale_exponents'][best_mse_idx]
    best_mse_metrics = results['metrics'][f'scale_{best_mse_idx}']['metrics']
    
    print(f"Best Cosine Similarity: {best_cosine_metrics['cosine_similarity']:.6f} at scale {best_cosine_scale:.2f}")
    print(f"Best MSE: {best_mse_metrics['mse']:.6e} at scale {best_mse_scale:.2f}")
    print(f"Best PSNR: {best_mse_metrics['psnr']:.2f} dB at scale {best_mse_scale:.2f}")
    
    print(f"\nResults saved to: {output_dir}")
    print("Test completed successfully!")
    
    return 0

if __name__ == "__main__":
    exit(main())
