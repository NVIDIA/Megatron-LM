#!/bin/bash

# =============================================================================
# Enhanced Multi-threaded Tensor Drawing Script
# Supports new data structure: bf16, mxfp8, mxfp4, hifp8 quantization types
# Supports multi-dimensional comparison of Sample (0,1,2) and Layer (1-16)
# Uses multi-threading to accelerate plotting process
# Features:
#   - HiFP8 distribution analysis with line plots and statistical measures
#   - Layer-Sample analysis for selected layers (1, 4, 8, 12, 16)
#   - Global statistics with detailed JSON and text reports
#   - Progress bars for all operations using tqdm
#   - Enhanced visualization with comprehensive analysis
# =============================================================================

# Set script metadata
SCRIPT_NAME="$(basename "$0")"
SCRIPT_VERSION="3.0.0"
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "=================================================================================="
echo "Enhanced Multi-threaded Tensor Drawing Script"
echo "Script: $SCRIPT_NAME"
echo "Version: $SCRIPT_VERSION"
echo "Start Time: $START_TIME"
echo "=================================================================================="

# Default parameters
TENSOR_DIR=${1:-"./enhanced_tensor_logs"}
OUTPUT_DIR=${2:-"./draw"}
MAX_WORKERS=${3:-4}

echo "Parameter settings:"
echo "  - Tensor directory: $TENSOR_DIR"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Max workers: $MAX_WORKERS"

# Check if tensor directory exists
if [ ! -d "$TENSOR_DIR" ]; then
    echo "Error: Tensor directory does not exist: $TENSOR_DIR"
    echo "Please ensure training script has been run and tensor files generated"
    exit 1
fi

# Check quantization type directories
QUANT_TYPES=("bf16" "mxfp8" "mxfp4" "hifp8")
echo ""
echo "Checking quantization type directories:"
for quant_type in "${QUANT_TYPES[@]}"; do
    quant_dir="$TENSOR_DIR/$quant_type"
    if [ -d "$quant_dir" ]; then
        file_count=$(find "$quant_dir" -name "*.pt" 2>/dev/null | wc -l)
        echo "  ✅ $quant_type: $file_count files"
    else
        echo "  ❌ $quant_type: directory does not exist"
    fi
done

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# Check required Python packages
echo ""
echo "Checking Python dependencies..."
python -c "import torch, matplotlib, numpy, pandas, seaborn, concurrent.futures" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Missing required Python packages, attempting to install..."
    pip install matplotlib numpy pandas seaborn scipy
fi

# Run enhanced multi-threaded visualization
echo ""
echo "Running enhanced multi-threaded visualization..."
python script/visualization/enhanced_multi_threaded_visualizer.py \
    --tensor_dir "$TENSOR_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_workers "$MAX_WORKERS"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Multi-threaded visualization completed!"
    
    # Show generated files
    echo ""
    echo "Generated chart files:"
    find "$OUTPUT_DIR" -name "*.png" | while read file; do
        echo "  - $(basename "$file")"
    done
    
    echo ""
    echo "Generated report files:"
    find "$OUTPUT_DIR" -name "*.txt" | while read file; do
        echo "  - $(basename "$file")"
    done
    
    # Show main outputs
    echo ""
    echo "Main output files:"
    if [ -f "$OUTPUT_DIR/quantization_analysis/quantization_comparison.png" ]; then
        echo "  🎯 Quantization comparison: $OUTPUT_DIR/quantization_analysis/quantization_comparison.png"
    fi
    if [ -f "$OUTPUT_DIR/sample_analysis/sample_analysis.png" ]; then
        echo "  📊 Sample analysis: $OUTPUT_DIR/sample_analysis/sample_analysis.png"
    fi
    if [ -f "$OUTPUT_DIR/layer_analysis/layer_analysis.png" ]; then
        echo "  🏗️ Layer analysis: $OUTPUT_DIR/layer_analysis/layer_analysis.png"
    fi
    if [ -f "$OUTPUT_DIR/comparison_analysis/comprehensive_comparison.png" ]; then
        echo "  🔍 Comprehensive comparison: $OUTPUT_DIR/comparison_analysis/comprehensive_comparison.png"
    fi
    if [ -f "$OUTPUT_DIR/hifp8_analysis/hifp8_distribution_analysis.png" ]; then
        echo "  🔬 HiFP8 distribution analysis: $OUTPUT_DIR/hifp8_analysis/hifp8_distribution_analysis.png"
    fi
    if [ -f "$OUTPUT_DIR/layer_sample_analysis/layer_sample_analysis.png" ]; then
        echo "  📈 Layer-Sample analysis: $OUTPUT_DIR/layer_sample_analysis/layer_sample_analysis.png"
    fi
    if [ -f "$OUTPUT_DIR/global_statistics/global_statistics.json" ]; then
        echo "  📊 Global statistics (JSON): $OUTPUT_DIR/global_statistics/global_statistics.json"
    fi
    if [ -f "$OUTPUT_DIR/global_statistics/global_statistics_report.txt" ]; then
        echo "  📋 Global statistics report: $OUTPUT_DIR/global_statistics/global_statistics_report.txt"
    fi
    if [ -f "$OUTPUT_DIR/statistics/detailed_statistics_report.txt" ]; then
        echo "  📋 Statistics report: $OUTPUT_DIR/statistics/detailed_statistics_report.txt"
    fi
    
    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo ""
    echo "=================================================================================="
    echo "Visualization completed"
    echo "Start time: $START_TIME"
    echo "End time: $END_TIME"
    echo "=================================================================================="
else
    echo ""
    echo "❌ Multi-threaded visualization failed"
    echo "Attempting to run basic visualization..."
    
    # Fallback to basic visualization
    bash script/visualization/one_click_visualize.sh "$TENSOR_DIR" "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✅ Basic visualization completed"
    else
        echo "❌ Basic visualization also failed"
        exit 1
    fi
fi