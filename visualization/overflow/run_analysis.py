#!/usr/bin/env python3
"""
Simple wrapper script to run the scaling factor analysis.

Usage:
    python3 run_analysis.py [base_directory]

If no base_directory is provided, it will use the current directory.
"""

import sys
from pathlib import Path
from analyze_scaling_factors import ScalingAnalyzer


def main():
    # Get base directory from command line or use current directory
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = str(Path.cwd())
    
    print(f"🎯 Running scaling factor analysis on: {base_dir}")
    
    try:
        analyzer = ScalingAnalyzer(base_dir)
        analyzer.analyze_all_files()
        
        if analyzer.results:
            analyzer.print_detailed_report()
            analyzer.save_results_to_json()
            
            # Quick summary
            summary = analyzer.generate_summary()
            if summary['at_max_percentage'] == 100.0:
                print("\n🎉 RESULT: ALL tensors are using their maximum scaling factors!")
            else:
                print(f"\n⚠️  RESULT: {summary['not_at_max_count']} out of {summary['total_tensors']} tensors are NOT at maximum scaling.")
        else:
            print("❌ No valid log files found or parsed.")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()




