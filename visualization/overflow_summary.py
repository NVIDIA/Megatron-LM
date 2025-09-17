#!/usr/bin/env python3
"""
Comprehensive overflow/underflow analysis for all tensor files.
Generates detailed analysis reports for all data formats.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import subprocess

# Add current directory to path to import overflow module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overflow import analyze_file, DATA_TYPE_RANGES

def analyze_all_tensors(base_dir="enhanced_tensor_logs", output_dir="visualization"):
    """
    Analyze all tensor files in the enhanced_tensor_logs directory structure.
    
    Args:
        base_dir (str): Base directory containing tensor files
        output_dir (str): Output directory for reports
        
    Returns:
        dict: Complete analysis results
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    all_results = {}
    summary_stats = {}
    
    print("Starting comprehensive tensor overflow/underflow analysis...")
    print("=" * 60)
    
    # Analyze each data format directory
    for data_format in DATA_TYPE_RANGES.keys():
        format_dir = base_path / data_format
        
        if not format_dir.exists():
            print(f"Warning: Directory not found for {data_format}: {format_dir}")
            continue
        
        print(f"Analyzing {data_format.upper()} tensors...")
        
        format_results = []
        tensor_files = list(format_dir.glob("*.pt"))
        
        if not tensor_files:
            print(f"  No tensor files found in {format_dir}")
            continue
        
        print(f"  Found {len(tensor_files)} tensor files")
        
        # Analyze each tensor file
        for i, tensor_file in enumerate(tensor_files):
            if i % 50 == 0 and i > 0:
                print(f"  Processed {i}/{len(tensor_files)} files...")
            
            result = analyze_file(str(tensor_file))
            if result:
                format_results.append(result)
        
        all_results[data_format] = format_results
        
        # Calculate summary statistics for this format
        if format_results:
            total_files = len(format_results)
            total_elements = sum(r['total_elements'] for r in format_results)
            total_overflow = sum(r['overflow_count'] for r in format_results)
            total_underflow = sum(r['underflow_count'] for r in format_results)
            
            files_with_overflow = sum(1 for r in format_results if r['overflow_count'] > 0)
            files_with_underflow = sum(1 for r in format_results if r['underflow_count'] > 0)
            
            overflow_percent = (total_overflow / total_elements) * 100 if total_elements > 0 else 0
            underflow_percent = (total_underflow / total_elements) * 100 if total_elements > 0 else 0
            
            # Value statistics
            all_mins = [r['tensor_min'] for r in format_results]
            all_maxs = [r['tensor_max'] for r in format_results]
            all_means = [r['tensor_mean'] for r in format_results]
            
            summary_stats[data_format] = {
                'total_files': total_files,
                'total_elements': total_elements,
                'total_overflow': total_overflow,
                'total_underflow': total_underflow,
                'files_with_overflow': files_with_overflow,
                'files_with_underflow': files_with_underflow,
                'overflow_percent': overflow_percent,
                'underflow_percent': underflow_percent,
                'global_min': min(all_mins),
                'global_max': max(all_maxs),
                'avg_mean': sum(all_means) / len(all_means),
                'format_range': [DATA_TYPE_RANGES[data_format]['min'], DATA_TYPE_RANGES[data_format]['max']],
                'description': DATA_TYPE_RANGES[data_format]['description']
            }
        
        print(f"  Completed {data_format.upper()}: {len(format_results)} files processed")
    
    # Generate comprehensive report
    generate_comprehensive_report(all_results, summary_stats, output_path)
    
    # Generate detailed JSON report
    generate_json_report(all_results, summary_stats, output_path)
    
    # Generate CSV summary
    generate_csv_summary(summary_stats, output_path)
    
    print("\nAnalysis complete! Generated reports:")
    print(f"  - {output_path}/overflow_comprehensive_report.txt")
    print(f"  - {output_path}/overflow_detailed_results.json") 
    print(f"  - {output_path}/overflow_summary.csv")
    
    return all_results, summary_stats

def generate_comprehensive_report(all_results, summary_stats, output_path):
    """Generate a comprehensive text report."""
    report_file = output_path / "overflow_comprehensive_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TENSOR FILE OVERFLOW/UNDERFLOW ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write("This report shows the PERCENTAGE of tensor files that have overflow/underflow issues\n")
        f.write("(vs. the percentage of values within each file)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        total_files_all = sum(stats['total_files'] for stats in summary_stats.values())
        total_files_with_overflow = sum(stats['files_with_overflow'] for stats in summary_stats.values())
        total_files_with_underflow = sum(stats['files_with_underflow'] for stats in summary_stats.values())
        total_files_with_issues = len(set().union(*[
            [r['filename'] for r in results if r['has_issues']] 
            for results in all_results.values()
        ]))
        
        f.write(f"Total tensor files analyzed: {total_files_all:,}\n")
        f.write(f"Files with overflow issues: {total_files_with_overflow:,} ({(total_files_with_overflow/total_files_all)*100:.2f}%)\n")
        f.write(f"Files with underflow issues: {total_files_with_underflow:,} ({(total_files_with_underflow/total_files_all)*100:.2f}%)\n")
        f.write(f"Files with any issues: {total_files_with_issues:,} ({(total_files_with_issues/total_files_all)*100:.2f}%)\n\n")
        
        # Format-specific summaries
        f.write("FORMAT-SPECIFIC ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        for data_format, stats in summary_stats.items():
            f.write(f"\n{data_format.upper()} ({stats['description']})\n")
            f.write("─" * 50 + "\n")
            f.write(f"Representable range: [{stats['format_range'][0]}, {stats['format_range'][1]}]\n")
            f.write(f"Files analyzed: {stats['total_files']:,}\n")
            f.write(f"Value range observed: [{stats['global_min']:.6f}, {stats['global_max']:.6f}]\n")
            f.write(f"Average mean value: {stats['avg_mean']:.6f}\n\n")
            
            f.write("File-Level Analysis (Primary Focus):\n")
            f.write(f"  Files with overflow: {stats['files_with_overflow']}/{stats['total_files']} ({(stats['files_with_overflow']/stats['total_files'])*100:.1f}%)\n")
            f.write(f"  Files with underflow: {stats['files_with_underflow']}/{stats['total_files']} ({(stats['files_with_underflow']/stats['total_files'])*100:.1f}%)\n")
            
            f.write("Value-Level Statistics (Reference):\n")
            f.write(f"  Overflow value percentage: {stats['overflow_percent']:.4f}%\n")
            f.write(f"  Underflow value percentage: {stats['underflow_percent']:.4f}%\n")
            f.write(f"  Total elements: {stats['total_elements']:,}\n")
            
            # Risk assessment based on file percentage
            file_overflow_pct = (stats['files_with_overflow'] / stats['total_files']) * 100
            file_underflow_pct = (stats['files_with_underflow'] / stats['total_files']) * 100
            
            risk_level = "LOW"
            if file_overflow_pct > 10.0 or file_underflow_pct > 10.0:
                risk_level = "HIGH"
            elif file_overflow_pct > 1.0 or file_underflow_pct > 1.0:
                risk_level = "MEDIUM"
            
            f.write(f"Risk Level: {risk_level} (based on file percentage)\n")
            
            if stats['files_with_overflow'] > 0 or stats['files_with_underflow'] > 0:
                f.write("⚠️  ATTENTION: Some files have overflow/underflow issues!\n")
            
            f.write("\n")
        
        # Detailed file listings for problematic cases
        f.write("DETAILED PROBLEMATIC FILES\n")
        f.write("-" * 40 + "\n")
        
        for data_format, results in all_results.items():
            problematic_files = [r for r in results if r['overflow_count'] > 0 or r['underflow_count'] > 0]
            
            if problematic_files:
                f.write(f"\n{data_format.upper()} - Files with overflow/underflow:\n")
                f.write("─" * 30 + "\n")
                
                for result in problematic_files[:20]:  # Limit to first 20 problematic files
                    f.write(f"File: {result['filename']}\n")
                    f.write(f"  Shape: {result['shape']}\n")
                    f.write(f"  Range: [{result['tensor_min']:.6f}, {result['tensor_max']:.6f}]\n")
                    f.write(f"  Overflow: {result['overflow_count']:,} ({result['overflow_percent']:.4f}%)\n")
                    f.write(f"  Underflow: {result['underflow_count']:,} ({result['underflow_percent']:.4f}%)\n")
                    f.write("\n")
                
                if len(problematic_files) > 20:
                    f.write(f"... and {len(problematic_files) - 20} more problematic files\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        for data_format, stats in summary_stats.items():
            if stats['total_overflow'] > 0 or stats['total_underflow'] > 0:
                f.write(f"{data_format.upper()}:\n")
                
                if stats['total_overflow'] > 0:
                    f.write(f"  - Consider clipping values above {stats['format_range'][1]} to prevent overflow\n")
                    f.write(f"  - Review model architecture or training parameters causing extreme values\n")
                
                if stats['total_underflow'] > 0:
                    f.write(f"  - Consider clipping values below {stats['format_range'][0]} to prevent underflow\n")
                    f.write(f"  - Review initialization or gradient scaling settings\n")
                
                f.write(f"  - Monitor numerical stability during training\n")
                f.write(f"  - Consider mixed precision training strategies\n\n")

def generate_json_report(all_results, summary_stats, output_path):
    """Generate detailed JSON report."""
    json_file = output_path / "overflow_detailed_results.json"
    
    report_data = {
        'metadata': {
            'generated_on': datetime.now().isoformat(),
            'total_formats': len(summary_stats),
            'total_files': sum(stats['total_files'] for stats in summary_stats.values())
        },
        'summary_statistics': summary_stats,
        'detailed_results': all_results
    }
    
    with open(json_file, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)

def generate_csv_summary(summary_stats, output_path):
    """Generate CSV summary report."""
    import csv
    
    csv_file = output_path / "overflow_summary.csv"
    
    with open(csv_file, 'w', newline='') as f:
        fieldnames = [
            'data_format', 'description', 'total_files', 'total_elements',
            'total_overflow', 'total_underflow', 'overflow_percent', 'underflow_percent',
            'files_with_overflow', 'files_with_underflow', 'global_min', 'global_max',
            'avg_mean', 'format_min', 'format_max'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for data_format, stats in summary_stats.items():
            row = {
                'data_format': data_format,
                'description': stats['description'],
                'total_files': stats['total_files'],
                'total_elements': stats['total_elements'],
                'total_overflow': stats['total_overflow'],
                'total_underflow': stats['total_underflow'],
                'overflow_percent': stats['overflow_percent'],
                'underflow_percent': stats['underflow_percent'],
                'files_with_overflow': stats['files_with_overflow'],
                'files_with_underflow': stats['files_with_underflow'],
                'global_min': stats['global_min'],
                'global_max': stats['global_max'],
                'avg_mean': stats['avg_mean'],
                'format_min': stats['format_range'][0],
                'format_max': stats['format_range'][1]
            }
            writer.writerow(row)

def main():
    """Main function to run comprehensive analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive tensor overflow/underflow analysis')
    parser.add_argument('--base-dir', default='enhanced_tensor_logs',
                        help='Base directory containing tensor files (default: enhanced_tensor_logs)')
    parser.add_argument('--output-dir', default='./draw/overflow_summary/',
                        help='Output directory for reports (default: ./draw/overflow_summary/)')
    
    args = parser.parse_args()
    
    try:
        all_results, summary_stats = analyze_all_tensors(args.base_dir, args.output_dir)
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        for data_format, stats in summary_stats.items():
            file_overflow_pct = (stats['files_with_overflow'] / stats['total_files']) * 100
            file_underflow_pct = (stats['files_with_underflow'] / stats['total_files']) * 100
            print(f"{data_format.upper()}: {stats['total_files']} files, "
                  f"{file_overflow_pct:.1f}% files with overflow, "
                  f"{file_underflow_pct:.1f}% files with underflow")
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
