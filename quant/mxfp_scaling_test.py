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
import logging
from datetime import datetime

# Add the parent directory to path to import mxfp module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant.mxfp import _quantize_mx, _get_format_params, ElemFormat

def setup_logging(output_dir, tensor_name, elem_format):
    """
    Setup logging to both console and file.
    
    Args:
        output_dir (Path): Output directory for log file
        tensor_name (str): Name of the input tensor
        elem_format (str): Element format being tested
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger('mxfp_scaling_test')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_filename = f"mxfp_scaling_test_{tensor_name}_{elem_format}.log"
    log_path = output_dir / log_filename
    
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Log initial information
    logger.info("=" * 80)
    logger.info("MXFP SCALING TEST LOG")
    logger.info("=" * 80)
    logger.info(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input tensor: {tensor_name}")
    logger.info(f"Element format: {elem_format}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    
    return logger

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
                       max_scale_exp=10, min_scale_exp=-10, num_levels=21, logger=None):
    """
    Test different scaling levels for MXFP quantization.
    
    Args:
        input_tensor (torch.Tensor): Input BF16 tensor
        elem_format (str): Element format for quantization
        scale_bits (int): Number of scale bits
        max_scale_exp (int): Maximum scale exponent (aligned with max value)
        min_scale_exp (int): Minimum scale exponent (aligned with min value)
        num_levels (int): Number of scaling levels to test
        logger: Logger instance for output
        
    Returns:
        dict: Results for each scaling level
    """
    # Get format parameters
    ebits, mbits, emax, max_norm, min_norm = _get_format_params(elem_format)
    
    # Calculate tensor statistics for alignment
    tensor_abs_max = torch.max(torch.abs(input_tensor)).item()
    tensor_abs_min = torch.min(torch.abs(input_tensor[input_tensor != 0])).item() if torch.any(input_tensor != 0) else tensor_abs_max
    
    # Calculate emax for the format (following mxfp.py logic)
    emax = 2**(ebits - 1) - 1 if ebits > 0 else 0
    
    # Calculate scale exponents following mxfp.py _quantize_mx logic:
    # In mxfp.py:
    # 1. shared_exp = floor(log2(max_abs_value))  (from _shared_exponents with method="max")
    # 2. shared_exp = shared_exp - emax           (offset by emax)
    # 3. A = A / (2^shared_exp)                   (apply scaling)
    #
    # So the actual scaling factor used by mxfp.py is: 2^(floor(log2(max)) - emax)
    #
    # For alignment calculations:
    # - Max alignment: Use the same logic as mxfp.py (global max alignment)
    #   This gives: scale_exp = floor(log2(tensor_abs_max)) - emax
    # - Min alignment: Find scale_exp such that tensor_abs_min / (2^scale_exp) >= min_norm
    #   So scale_exp <= log2(tensor_abs_min / min_norm)
    
    # Calculate the scale exponent that mxfp.py would use (for reference)
    tensor_shared_exp = np.floor(np.log2(tensor_abs_max)) if tensor_abs_max > 0 else 0
    max_align_exp = tensor_shared_exp - emax  # This is what mxfp.py actually uses
    
    # Calculate min alignment: find scale_exp such that scaled min >= min_norm
    min_align_exp = np.floor(np.log2(tensor_abs_min / min_norm)) if tensor_abs_min > 0 and min_norm > 0 else max_align_exp
    
    # Use user-specified parameters directly, with calculated values as fallback for default parameters
    if max_scale_exp == 10:  # Default value, use calculated
        max_scale_exp = max_align_exp
    if min_scale_exp == -10:  # Default value, use calculated
        min_scale_exp = min_align_exp
    
    # Ensure max_scale_exp >= min_scale_exp
    if max_scale_exp < min_scale_exp:
        max_scale_exp, min_scale_exp = min_scale_exp, max_scale_exp
    
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
    
    log_func = logger.info if logger else print
    log_func(f"Tensor absolute value range: [{tensor_abs_min:.6e}, {tensor_abs_max:.6e}]")
    log_func(f"Format range: max_norm={max_norm:.6e}, min_norm={min_norm:.6e}")
    log_func(f"Calculated alignment (reference): max_align={max_align_exp:.2f}, min_align={min_align_exp:.2f}")
    log_func(f"Testing {num_levels} scaling levels from {max_scale_exp:.2f} to {min_scale_exp:.2f}")
    log_func(f"Element format: {elem_format} (e{ebits}m{mbits})")
    log_func(f"Scale bits: {scale_bits}")
    log_func("-" * 60)
    
    for i, scale_exp in enumerate(scale_exponents):
        log_func(f"Testing scale exponent {scale_exp:.2f} ({i+1}/{num_levels})...")
        
        # Create a custom quantize function with fixed scale exponent
        quantized_tensor, overflow_underflow_analysis = quantize_with_fixed_scale(
            input_tensor, elem_format, scale_bits, scale_exp, 
            ebits, mbits, max_norm
        )
        
        # Calculate metrics
        metrics = calculate_metrics(input_tensor, quantized_tensor)
        
        # Store results
        results['metrics'][f'scale_{i}'] = {
            'scale_exponent': float(scale_exp),
            'metrics': metrics,
            'overflow_underflow_analysis': overflow_underflow_analysis
        }
        
        # Print current metrics
        log_func(f"  MSE: {metrics['mse']:.6e}, "
                 f"Cosine Sim: {metrics['cosine_similarity']:.6f}, "
                 f"PSNR: {metrics['psnr']:.2f} dB")
    
    return results

def analyze_scaling_results(results, logger=None):
    """
    Analyze scaling test results and recommend optimal scaling factors.
    
    Args:
        results (dict): Results from test_scaling_levels
        logger: Logger instance for output
        
    Returns:
        dict: Analysis results with recommendations
    """
    log_func = logger.info if logger else print
    
    scale_exponents = results['scale_exponents']
    elem_format = results['elem_format']
    format_params = results['format_params']
    
    # Extract metrics for analysis
    metrics_data = {}
    for metric_name in ['mse', 'cosine_similarity', 'psnr', 'mae', 'relative_error']:
        metrics_data[metric_name] = []
        for i in range(len(scale_exponents)):
            scale_key = f'scale_{i}'
            if scale_key in results['metrics']:
                metrics_data[metric_name].append(results['metrics'][scale_key]['metrics'][metric_name])
    
    # Find best indices for different metrics
    best_mse_idx = np.argmin(metrics_data['mse'])
    best_cosine_idx = np.argmax(metrics_data['cosine_similarity'])
    best_psnr_idx = np.argmax(metrics_data['psnr'])
    best_mae_idx = np.argmin(metrics_data['mae'])
    best_relative_error_idx = np.argmin(metrics_data['relative_error'])
    
    # Calculate composite scores
    # Normalize metrics to [0, 1] range for comparison
    mse_normalized = 1 - (np.array(metrics_data['mse']) - np.min(metrics_data['mse'])) / (np.max(metrics_data['mse']) - np.min(metrics_data['mse']) + 1e-10)
    cosine_normalized = np.array(metrics_data['cosine_similarity'])
    psnr_normalized = (np.array(metrics_data['psnr']) - np.min(metrics_data['psnr'])) / (np.max(metrics_data['psnr']) - np.min(metrics_data['psnr']) + 1e-10)
    mae_normalized = 1 - (np.array(metrics_data['mae']) - np.min(metrics_data['mae'])) / (np.max(metrics_data['mae']) - np.min(metrics_data['mae']) + 1e-10)
    relative_error_normalized = 1 - (np.array(metrics_data['relative_error']) - np.min(metrics_data['relative_error'])) / (np.max(metrics_data['relative_error']) - np.min(metrics_data['relative_error']) + 1e-10)
    
    # Weighted composite score (can be adjusted based on priorities)
    composite_scores = (
        0.3 * mse_normalized +           # Lower MSE is better
        0.3 * cosine_normalized +        # Higher cosine similarity is better
        0.2 * psnr_normalized +          # Higher PSNR is better
        0.1 * mae_normalized +           # Lower MAE is better
        0.1 * relative_error_normalized  # Lower relative error is better
    )
    
    best_composite_idx = np.argmax(composite_scores)
    
    # Calculate scaling factor from scale exponent
    def exp_to_factor(exp):
        return 2 ** exp
    
    # Analysis results
    analysis = {
        'best_mse': {
            'index': best_mse_idx,
            'scale_exp': scale_exponents[best_mse_idx],
            'scale_factor': exp_to_factor(scale_exponents[best_mse_idx]),
            'metrics': {
                'mse': metrics_data['mse'][best_mse_idx],
                'cosine_similarity': metrics_data['cosine_similarity'][best_mse_idx],
                'psnr': metrics_data['psnr'][best_mse_idx],
                'mae': metrics_data['mae'][best_mse_idx],
                'relative_error': metrics_data['relative_error'][best_mse_idx]
            }
        },
        'best_cosine': {
            'index': best_cosine_idx,
            'scale_exp': scale_exponents[best_cosine_idx],
            'scale_factor': exp_to_factor(scale_exponents[best_cosine_idx]),
            'metrics': {
                'mse': metrics_data['mse'][best_cosine_idx],
                'cosine_similarity': metrics_data['cosine_similarity'][best_cosine_idx],
                'psnr': metrics_data['psnr'][best_cosine_idx],
                'mae': metrics_data['mae'][best_cosine_idx],
                'relative_error': metrics_data['relative_error'][best_cosine_idx]
            }
        },
        'best_psnr': {
            'index': best_psnr_idx,
            'scale_exp': scale_exponents[best_psnr_idx],
            'scale_factor': exp_to_factor(scale_exponents[best_psnr_idx]),
            'metrics': {
                'mse': metrics_data['mse'][best_psnr_idx],
                'cosine_similarity': metrics_data['cosine_similarity'][best_psnr_idx],
                'psnr': metrics_data['psnr'][best_psnr_idx],
                'mae': metrics_data['mae'][best_psnr_idx],
                'relative_error': metrics_data['relative_error'][best_psnr_idx]
            }
        },
        'best_composite': {
            'index': best_composite_idx,
            'scale_exp': scale_exponents[best_composite_idx],
            'scale_factor': exp_to_factor(scale_exponents[best_composite_idx]),
            'composite_score': composite_scores[best_composite_idx],
            'metrics': {
                'mse': metrics_data['mse'][best_composite_idx],
                'cosine_similarity': metrics_data['cosine_similarity'][best_composite_idx],
                'psnr': metrics_data['psnr'][best_composite_idx],
                'mae': metrics_data['mae'][best_composite_idx],
                'relative_error': metrics_data['relative_error'][best_composite_idx]
            }
        }
    }
    
    # Log detailed analysis
    log_func("\n" + "=" * 80)
    log_func("SCALING FACTOR ANALYSIS & RECOMMENDATIONS")
    log_func("=" * 80)
    
    log_func(f"Format: {elem_format} (e{format_params['ebits']}m{format_params['mbits']})")
    log_func(f"Tested {len(scale_exponents)} scaling levels from {scale_exponents[0]:.2f} to {scale_exponents[-1]:.2f}")
    log_func("-" * 80)
    
    # Best results for individual metrics
    log_func("INDIVIDUAL METRIC OPTIMA:")
    log_func("-" * 40)
    
    log_func(f"🏆 Best MSE: Scale Exp = {analysis['best_mse']['scale_exp']:.2f}, Factor = {analysis['best_mse']['scale_factor']:.6f}")
    log_func(f"    MSE: {analysis['best_mse']['metrics']['mse']:.6e}, Cosine: {analysis['best_mse']['metrics']['cosine_similarity']:.6f}, PSNR: {analysis['best_mse']['metrics']['psnr']:.2f} dB")
    
    log_func(f"🎯 Best Cosine Similarity: Scale Exp = {analysis['best_cosine']['scale_exp']:.2f}, Factor = {analysis['best_cosine']['scale_factor']:.6f}")
    log_func(f"    MSE: {analysis['best_cosine']['metrics']['mse']:.6e}, Cosine: {analysis['best_cosine']['metrics']['cosine_similarity']:.6f}, PSNR: {analysis['best_cosine']['metrics']['psnr']:.2f} dB")
    
    log_func(f"📊 Best PSNR: Scale Exp = {analysis['best_psnr']['scale_exp']:.2f}, Factor = {analysis['best_psnr']['scale_factor']:.6f}")
    log_func(f"    MSE: {analysis['best_psnr']['metrics']['mse']:.6e}, Cosine: {analysis['best_psnr']['metrics']['cosine_similarity']:.6f}, PSNR: {analysis['best_psnr']['metrics']['psnr']:.2f} dB")
    
    # Composite recommendation
    log_func("-" * 80)
    log_func("COMPOSITE RECOMMENDATION:")
    log_func("-" * 40)
    
    log_func(f"⭐ RECOMMENDED Scaling Factor: {analysis['best_composite']['scale_factor']:.6f}")
    log_func(f"   Scale Exponent: {analysis['best_composite']['scale_exp']:.2f}")
    log_func(f"   Composite Score: {analysis['best_composite']['composite_score']:.4f}")
    log_func(f"   Balanced Performance:")
    log_func(f"     - MSE: {analysis['best_composite']['metrics']['mse']:.6e}")
    log_func(f"     - Cosine Similarity: {analysis['best_composite']['metrics']['cosine_similarity']:.6f}")
    log_func(f"     - PSNR: {analysis['best_composite']['metrics']['psnr']:.2f} dB")
    log_func(f"     - MAE: {analysis['best_composite']['metrics']['mae']:.6e}")
    log_func(f"     - Relative Error: {analysis['best_composite']['metrics']['relative_error']:.2f}%")
    
    # Performance analysis
    log_func("-" * 80)
    log_func("PERFORMANCE ANALYSIS:")
    log_func("-" * 40)
    
    # Calculate performance ranges
    mse_range = np.max(metrics_data['mse']) - np.min(metrics_data['mse'])
    cosine_range = np.max(metrics_data['cosine_similarity']) - np.min(metrics_data['cosine_similarity'])
    psnr_range = np.max(metrics_data['psnr']) - np.min(metrics_data['psnr'])
    
    log_func(f"MSE Range: {np.min(metrics_data['mse']):.6e} to {np.max(metrics_data['mse']):.6e} (Δ: {mse_range:.6e})")
    log_func(f"Cosine Range: {np.min(metrics_data['cosine_similarity']):.6f} to {np.max(metrics_data['cosine_similarity']):.6f} (Δ: {cosine_range:.6f})")
    log_func(f"PSNR Range: {np.min(metrics_data['psnr']):.2f} to {np.max(metrics_data['psnr']):.2f} dB (Δ: {psnr_range:.2f} dB)")
    
    # Stability analysis
    mse_std = np.std(metrics_data['mse'])
    cosine_std = np.std(metrics_data['cosine_similarity'])
    
    log_func(f"MSE Stability (std): {mse_std:.6e}")
    log_func(f"Cosine Stability (std): {cosine_std:.6f}")
    
    # Recommendations based on analysis
    log_func("-" * 80)
    log_func("RECOMMENDATIONS:")
    log_func("-" * 40)
    
    if mse_range / np.min(metrics_data['mse']) < 0.1:
        log_func("✅ MSE is relatively stable across scaling factors - any factor in the tested range should work well")
    else:
        log_func("⚠️  MSE varies significantly with scaling - choose the recommended factor carefully")
    
    if cosine_range < 0.01:
        log_func("✅ Cosine similarity is very stable - scaling factor has minimal impact on direction preservation")
    else:
        log_func("⚠️  Cosine similarity varies with scaling - consider the impact on vector direction")
    
    if psnr_range > 20:
        log_func("📈 Large PSNR range indicates significant quality differences - scaling factor choice is critical")
    elif psnr_range > 10:
        log_func("📊 Moderate PSNR range - scaling factor has noticeable impact on quality")
    else:
        log_func("✅ Small PSNR range - scaling factor has limited impact on quality")
    
    # Final recommendation
    log_func("-" * 80)
    log_func("FINAL RECOMMENDATION:")
    log_func("-" * 40)
    log_func(f"🎯 Use scaling factor: {analysis['best_composite']['scale_factor']:.6f}")
    log_func(f"   This provides the best balance of accuracy and stability for {elem_format} quantization")
    log_func(f"   Scale exponent: {analysis['best_composite']['scale_exp']:.2f}")
    
    if analysis['best_composite']['index'] == 0:
        log_func("   📍 This is at the maximum alignment end (minimal overflow risk)")
    elif analysis['best_composite']['index'] == len(scale_exponents) - 1:
        log_func("   📍 This is at the minimum alignment end (minimal underflow risk)")
    else:
        log_func("   📍 This is a balanced middle ground between overflow and underflow")
    
    log_func("=" * 80)
    
    return analysis

def analyze_overflow_underflow_results(results, logger=None):
    """
    Analyze and display overflow and underflow results from scaling tests.
    
    Args:
        results (dict): Results from test_scaling_levels
        logger: Logger instance for output
    """
    log_func = logger.info if logger else print
    
    scale_exponents = results['scale_exponents']
    elem_format = results['elem_format']
    
    # Collect all overflow/underflow analyses
    overflow_underflow_results = []
    significant_issues = []
    
    for i in range(len(scale_exponents)):
        scale_key = f'scale_{i}'
        if scale_key in results['metrics']:
            analysis = results['metrics'][scale_key]['overflow_underflow_analysis']
            analysis['scale_exp'] = scale_exponents[i]
            analysis['scale_factor'] = 2 ** scale_exponents[i]
            overflow_underflow_results.append(analysis)
            
            if analysis['has_significant_underflow'] or analysis['has_significant_overflow']:
                significant_issues.append(analysis)
    
    # Only display analysis if there are significant issues
    if not significant_issues:
        log_func("\n✅ No significant overflow or underflow issues detected across all scaling levels")
        return
    
    # Display comprehensive overflow/underflow analysis
    log_func("\n" + "=" * 80)
    log_func("OVERFLOW/UNDERFLOW ANALYSIS SUMMARY")
    log_func("=" * 80)
    
    log_func(f"Format: {elem_format}")
    log_func(f"Analyzed {len(scale_exponents)} scaling levels")
    log_func(f"Significant overflow/underflow detected in {len(significant_issues)} levels")
    log_func("-" * 80)
    
    # Group by severity
    high_severity = [u for u in significant_issues if u['severity'] == 'high']
    moderate_severity = [u for u in significant_issues if u['severity'] == 'moderate']
    
    # Separate overflow and underflow issues
    overflow_issues = [u for u in significant_issues if u['has_significant_overflow']]
    underflow_issues = [u for u in significant_issues if u['has_significant_underflow']]
    
    # Display overflow issues
    if overflow_issues:
        log_func("🔴 OVERFLOW ISSUES:")
        log_func("-" * 40)
        for uf in overflow_issues:
            log_func(f"  Scale Exp: {uf['scale_exp']:.2f} (Factor: {uf['scale_factor']:.6f})")
            log_func(f"    Overflow: {uf['overflow_count']:,} ({uf['overflow_percent']:.2f}%)")
            log_func(f"    Max Normal: {uf['max_norm']:.2e}")
            log_func(f"    Tensor Range: [{uf['tensor_range'][0]:.2e}, {uf['tensor_range'][1]:.2e}]")
            log_func(f"    Severity: {uf['severity'].upper()}")
            log_func("")
    
    # Display underflow issues
    if underflow_issues:
        log_func("🟡 UNDERFLOW ISSUES:")
        log_func("-" * 40)
        for uf in underflow_issues:
            log_func(f"  Scale Exp: {uf['scale_exp']:.2f} (Factor: {uf['scale_factor']:.6f})")
            log_func(f"    Underflow: {uf['underflow_count']:,} ({uf['underflow_percent']:.2f}%)")
            log_func(f"    Flush to Zero: {uf['flush_count']:,} ({uf['flush_percent']:.2f}%)")
            log_func(f"    Min Normal: {uf['min_norm']:.2e}")
            log_func(f"    Tensor Range: [{uf['tensor_range'][0]:.2e}, {uf['tensor_range'][1]:.2e}]")
            log_func(f"    Severity: {uf['severity'].upper()}")
            log_func("")
    
    # Find best and worst cases
    if overflow_issues:
        worst_overflow = max(overflow_issues, key=lambda x: x['overflow_percent'])
        log_func("OVERFLOW EXTREMES:")
        log_func("-" * 40)
        log_func(f"Worst Overflow: Scale Exp {worst_overflow['scale_exp']:.2f}")
        log_func(f"  {worst_overflow['overflow_percent']:.2f}% overflow")
    
    if underflow_issues:
        worst_underflow = max(underflow_issues, key=lambda x: x['underflow_percent'])
        best_underflow = min(underflow_issues, key=lambda x: x['underflow_percent'])
        log_func("UNDERFLOW EXTREMES:")
        log_func("-" * 40)
        log_func(f"Worst Underflow: Scale Exp {worst_underflow['scale_exp']:.2f}")
        log_func(f"  {worst_underflow['underflow_percent']:.2f}% underflow, {worst_underflow['flush_percent']:.2f}% flushed to zero")
        log_func(f"Best Underflow: Scale Exp {best_underflow['scale_exp']:.2f}")
        log_func(f"  {best_underflow['underflow_percent']:.2f}% underflow, {best_underflow['flush_percent']:.2f}% flushed to zero")
    
    # Recommendations
    log_func("-" * 80)
    log_func("OVERFLOW/UNDERFLOW RECOMMENDATIONS:")
    log_func("-" * 40)
    
    if high_severity:
        log_func("⚠️  AVOID scaling factors with HIGH overflow/underflow severity")
        log_func("   These factors cause significant precision loss")
    
    if overflow_issues:
        log_func("🔴 OVERFLOW WARNING:")
        log_func("   Avoid scaling factors that cause overflow")
        log_func("   These values will be saturated to max representable value")
    
    if underflow_issues:
        log_func("🟡 UNDERFLOW CONSIDERATIONS:")
        log_func("   Moderate underflow may be acceptable depending on use case")
        log_func("   Balance between underflow and overflow risks")
    
    # Find optimal range
    no_issue_levels = [u for u in overflow_underflow_results if not u['has_significant_underflow'] and not u['has_significant_overflow']]
    if no_issue_levels:
        optimal_range = [min(u['scale_exp'] for u in no_issue_levels),
                        max(u['scale_exp'] for u in no_issue_levels)]
        log_func(f"✅ RECOMMENDED scaling range: {optimal_range[0]:.2f} to {optimal_range[1]:.2f}")
        log_func("   This range minimizes both overflow and underflow issues")
    else:
        log_func("⚠️  All scaling levels have some overflow/underflow - choose least problematic")
        # Find least problematic range
        least_problematic = min(overflow_underflow_results, key=lambda x: max(x['overflow_percent'], x['underflow_percent']))
        log_func(f"💡 Least problematic scaling: {least_problematic['scale_exp']:.2f}")
        log_func(f"   Overflow: {least_problematic['overflow_percent']:.2f}%, Underflow: {least_problematic['underflow_percent']:.2f}%")
    
    log_func("=" * 80)

def quantize_with_fixed_scale(input_tensor, elem_format, scale_bits, scale_exp,
                             ebits, mbits, max_norm, axes=None, block_size=0):
    """
    Custom quantization function with fixed scale exponent.
    This function simulates the exact behavior of mxfp.py _quantize_mx function.
    
    Args:
        input_tensor (torch.Tensor): Input tensor
        elem_format (str): Element format
        scale_bits (int): Number of scale bits
        scale_exp (float): Fixed scale exponent (log2 of scaling factor)
        ebits (int): Exponent bits
        mbits (int): Mantissa bits
        max_norm (float): Maximum normal value
        axes (list): Axes for shared exponent calculation
        block_size (int): Block size for tiling
        
    Returns:
        tuple: (quantized_tensor, overflow_underflow_analysis)
    """
    A = input_tensor.clone()
    
    # Apply scaling directly (this simulates the A = A / (2**shared_exp) step in mxfp.py)
    scale_factor = 2 ** scale_exp
    A = A / scale_factor
    
    # Quantize element-wise
    from quant.mxfp import _quantize_elemwise_core,_analyze_overflow_underflow_before_quantization
    
    # Analyze overflow/underflow without printing (collect results)
    overflow_underflow_analysis = _analyze_overflow_underflow_before_quantization(
        A, elem_format, mbits, ebits, max_norm, verbose=False
    )
    
    A = _quantize_elemwise_core(
        A, mbits, ebits, max_norm, round='nearest',
        allow_denorm=True, saturate_normals=True
    )
    
    # Undo scaling
    A = A * scale_factor
    
    return A, overflow_underflow_analysis

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
    
    # This will be logged by the caller
    pass
    
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
    
    # This will be logged by the caller
    pass

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
                overflow_underflow_analysis = results['metrics'][scale_key]['overflow_underflow_analysis']
                
                f.write(f"Scale Exponent {scale_exp:.2f} (Factor: {2**scale_exp:.6f}):\n")
                f.write("  Performance Metrics:\n")
                f.write(f"    MSE: {metrics['mse']:.6e}\n")
                f.write(f"    RMSE: {metrics['rmse']:.6e}\n")
                f.write(f"    Cosine Similarity: {metrics['cosine_similarity']:.6f}\n")
                f.write(f"    PSNR: {metrics['psnr']:.2f} dB\n")
                f.write(f"    MAE: {metrics['mae']:.6e}\n")
                f.write(f"    Max Absolute Error: {metrics['max_abs_error']:.6e}\n")
                f.write(f"    Relative Error: {metrics['relative_error']:.2f}%\n")
                
                f.write("  Overflow/Underflow Analysis:\n")
                f.write(f"    Total Elements: {overflow_underflow_analysis['total_elements']:,}\n")
                f.write(f"    Underflow Count: {overflow_underflow_analysis['underflow_count']:,} ({overflow_underflow_analysis['underflow_percent']:.2f}%)\n")
                f.write(f"    Flush to Zero Count: {overflow_underflow_analysis['flush_count']:,} ({overflow_underflow_analysis['flush_percent']:.2f}%)\n")
                f.write(f"    Overflow Count: {overflow_underflow_analysis['overflow_count']:,} ({overflow_underflow_analysis['overflow_percent']:.2f}%)\n")
                f.write(f"    Min Denormal: {overflow_underflow_analysis['min_denormal']:.2e}\n")
                f.write(f"    Min Normal: {overflow_underflow_analysis['min_norm']:.2e}\n")
                f.write(f"    Max Normal: {overflow_underflow_analysis['max_norm']:.2e}\n")
                f.write(f"    Tensor Range: [{overflow_underflow_analysis['tensor_range'][0]:.2e}, {overflow_underflow_analysis['tensor_range'][1]:.2e}]\n")
                f.write(f"    Severity: {overflow_underflow_analysis['severity'].upper()}\n")
                f.write(f"    Has Significant Underflow: {'Yes' if overflow_underflow_analysis['has_significant_underflow'] else 'No'}\n")
                f.write(f"    Has Significant Overflow: {'Yes' if overflow_underflow_analysis['has_significant_overflow'] else 'No'}\n")
                if overflow_underflow_analysis['error']:
                    f.write(f"    Analysis Error: {overflow_underflow_analysis['error']}\n")
                f.write("\n")
    
    # This will be logged by the caller
    pass

def process_single_tensor(input_path, args, logger=None):
    """Process a single tensor file."""
    
    # Validate input file
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
    
    # Setup logging for this tensor
    tensor_name = input_path.stem
    tensor_logger = setup_logging(output_dir, tensor_name, args.elem_format)
    
    tensor_logger.info(f"Loading input tensor: {input_path.name}")
    tensor_logger.info("=" * 60)
    
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
                tensor_logger.error(f"Error: Loaded object is not a tensor: {input_path.name}")
                return 1
        
        # Convert to BF16 if needed
        if input_tensor.dtype != torch.bfloat16:
            tensor_logger.info(f"Converting tensor from {input_tensor.dtype} to bfloat16")
            input_tensor = input_tensor.bfloat16()
        
        tensor_logger.info(f"Tensor shape: {input_tensor.shape}")
        tensor_logger.info(f"Tensor dtype: {input_tensor.dtype}")
        tensor_logger.info(f"Value range: [{torch.min(input_tensor):.6f}, {torch.max(input_tensor):.6f}]")
        tensor_logger.info(f"Mean ± Std: {torch.mean(input_tensor):.6f} ± {torch.std(input_tensor):.6f}")
        
    except Exception as e:
        tensor_logger.error(f"Error loading tensor {input_path.name}: {str(e)}")
        return 1
    
    # Run scaling test
    results = test_scaling_levels(
        input_tensor, 
        args.elem_format, 
        args.scale_bits,
        max_scale_exp=args.max_scale_exp,
        min_scale_exp=args.min_scale_exp,
        num_levels=args.num_levels,
        logger=tensor_logger
    )
    
    # Save results to file
    save_results_to_file(results, output_dir)
    tensor_logger.info(f"Detailed results saved to: {output_dir}")
    
    # Generate plots unless disabled
    if not args.no_plots:
        plot_scaling_results(results, output_dir)
        tensor_logger.info(f"Plots saved to: {output_dir}")
    
    # Perform detailed analysis
    analysis_results = analyze_scaling_results(results, tensor_logger)
    
    # Analyze overflow/underflow results
    analyze_overflow_underflow_results(results, tensor_logger)
    
    # Print summary
    tensor_logger.info("\n" + "=" * 60)
    tensor_logger.info("SCALING TEST SUMMARY")
    tensor_logger.info("=" * 60)
    
    # Use analysis results for summary
    best_composite = analysis_results['best_composite']
    best_mse = analysis_results['best_mse']
    best_cosine = analysis_results['best_cosine']
    
    tensor_logger.info(f"Best Cosine Similarity: {best_cosine['metrics']['cosine_similarity']:.6f} at scale {best_cosine['scale_exp']:.2f}")
    tensor_logger.info(f"Best MSE: {best_mse['metrics']['mse']:.6e} at scale {best_mse['scale_exp']:.2f}")
    tensor_logger.info(f"Best PSNR: {best_mse['metrics']['psnr']:.2f} dB at scale {best_mse['scale_exp']:.2f}")
    
    tensor_logger.info(f"\n🎯 RECOMMENDED Scaling Factor: {best_composite['scale_factor']:.6f}")
    tensor_logger.info(f"   Scale Exponent: {best_composite['scale_exp']:.2f}")
    tensor_logger.info(f"   Composite Score: {best_composite['composite_score']:.4f}")
    
    tensor_logger.info(f"\nResults saved to: {output_dir}")
    tensor_logger.info("Test completed successfully!")
    
    # Log completion time
    tensor_logger.info("=" * 80)
    tensor_logger.info(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tensor_logger.info("=" * 80)
    
    return 0

def main():
    """Main function for MXFP scaling test."""
    parser = argparse.ArgumentParser(description='Test different scaling strategies for MXFP quantization')
    parser.add_argument('input_tensors', nargs='+', help='Path(s) to input BF16 tensor file(s) (.pt)')
    parser.add_argument('--output-dir', default=None, 
                        help='Output directory for results (default: ./draw/scaling_analysis/{tensor_name}/)')
    parser.add_argument('--elem-format', default='fp8_e4m3', 
                        choices=['fp8_e4m3', 'fp8_e5m2', 'fp4_e2m1', 'fp6_e3m2', 'fp6_e2m3'],
                        help='Element format for quantization (default: fp8_e4m3)')
    parser.add_argument('--scale-bits', type=int, default=8,
                        help='Number of scale bits (default: 8)')
    parser.add_argument('--max-scale-exp', type=int, default=10,
                        help='Maximum scale exponent (default: auto-calculated from tensor max if using default value)')
    parser.add_argument('--min-scale-exp', type=int, default=-10,
                        help='Minimum scale exponent (default: auto-calculated from tensor min if using default value)')
    parser.add_argument('--num-levels', type=int, default=21,
                        help='Number of scaling levels to test (default: 21)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Process multiple tensors
    total_tensors = len(args.input_tensors)
    successful_tests = 0
    
    print(f"Processing {total_tensors} tensor(s)...")
    print("=" * 80)
    
    for i, tensor_path in enumerate(args.input_tensors, 1):
        print(f"\n[{i}/{total_tensors}] Processing: {tensor_path}")
        print("-" * 60)
        
        input_path = Path(tensor_path)
        result = process_single_tensor(input_path, args)
        
        if result == 0:
            successful_tests += 1
            print(f"✅ Successfully processed: {tensor_path}")
        else:
            print(f"❌ Failed to process: {tensor_path}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total tensors: {total_tensors}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tensors - successful_tests}")
    
    if successful_tests == total_tensors:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("⚠️  Some tests failed. Check individual logs for details.")
        return 1

if __name__ == '__main__':
    exit(main())
