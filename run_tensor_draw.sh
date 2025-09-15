#!/bin/bash

# =============================================================================
# Unified Tensor Analysis Script
# Combines tensor collection and visualization in one comprehensive tool
# Supports bf16, mxfp8, mxfp4, hifp8 quantization types
# Features:
#   - Automatic tensor collection with multiple modes
#   - Quantization type comparison analysis
#   - HiFP8 distribution analysis with statistical measures
#   - Global statistics with detailed JSON and text reports
#   - Progress bars for all operations using tqdm
#   - Comprehensive error handling and fallback methods
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
MODE="visualize"
TENSOR_DIR="./enhanced_tensor_logs"
OUTPUT_DIR="./draw"
MAX_WORKERS=4
QUANT_TYPE="mxfp8"
CONTROL_ITER=1  # 控制收集的micro_batch数量
# collect_micro_batches已固定为1，进行一次完整forward后跳出

# 显示使用帮助
show_help() {
    echo "用法: $0 [OPTIONS] [MODE] [QUANT_TYPE]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  --mode MODE             运行模式 (collect|visualize|both) [默认: visualize]"
    echo "  --tensor-dir DIR        Tensor目录 [默认: ./enhanced_tensor_logs]"
    echo "  --output-dir DIR        输出目录 [默认: ./draw]"
    echo "  --max-workers NUM       最大工作线程数 [默认: 4]"
    echo "  --quant-type TYPE       量化类型 (bf16|mxfp8|mxfp4|hifp8) [默认: mxfp8]"
    echo "  --control-iter NUM      控制收集的micro_batch数量 [默认: 1]"
    echo "  (collect_micro_batches已固定为1，进行一次完整forward后跳出)"
    echo ""
    echo "位置参数:"
    echo "  MODE                    运行模式 (collect|visualize|both)"
    echo "  QUANT_TYPE              量化类型 (bf16|mxfp8|mxfp4|hifp8)"
    echo ""
    echo "使用示例:"
    echo "  # 基本用法"
    echo "  $0 visualize"
    echo ""
    echo "  # 使用命令行参数"
    echo "  $0 --mode both --quant-type mxfp8 --control-iter 3"
    echo ""
    echo "  # 收集tensor"
    echo "  $0 collect mxfp8"
    echo ""
    echo "  # 可视化现有数据"
    echo "  $0 visualize"
    echo ""
    echo "  # 收集并可视化"
    echo "  $0 both mxfp8"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --tensor-dir)
            TENSOR_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --quant-type)
            QUANT_TYPE="$2"
            shift 2
            ;;
        --control-iter)
            CONTROL_ITER="$2"
            shift 2
            ;;
        # collect_micro_batches参数已移除
        collect|visualize|both)
            MODE="$1"
            shift
            ;;
        bf16|mxfp8|mxfp4|hifp8)
            QUANT_TYPE="$1"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

echo "Parameter settings:"
echo "  - Mode: $MODE"
echo "  - Tensor directory: $TENSOR_DIR"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Max workers: $MAX_WORKERS"
echo "  - Quantization type: $QUANT_TYPE"
echo "  - Control micro_batches: $CONTROL_ITER"
echo "  - 收集micro_batch数量: 1 (固定，一次forward后跳出)"

# Handle different modes
case "$MODE" in
    "collect")
        echo ""
        echo "Running tensor collection mode..."
        bash run_tensor_collection.sh --mode single --quant-type "$QUANT_TYPE" --tensor-path "$TENSOR_DIR" --control-iter "$CONTROL_ITER"
        if [ $? -eq 0 ]; then
            echo "✅ Tensor collection completed"
        else
            echo "❌ Tensor collection failed"
            exit 1
        fi
        ;;
    "both")
        echo ""
        echo "Running both tensor collection and visualization..."
        bash run_tensor_collection.sh --mode single --quant-type "$QUANT_TYPE" --tensor-path "$TENSOR_DIR" --control-iter "$CONTROL_ITER"
        if [ $? -eq 0 ]; then
            echo "✅ Tensor collection completed, proceeding to visualization..."
        else
            echo "❌ Tensor collection failed, proceeding to visualization with existing data..."
        fi
        ;;
    "visualize")
        echo ""
        echo "Running visualization mode only..."
        ;;
    *)
        echo "Error: Invalid mode '$MODE'"
        echo "Supported modes: collect, visualize, both"
        echo "Use --help to see usage information"
        exit 1
        ;;
esac

# Check if tensor directory exists
if [ ! -d "$TENSOR_DIR" ]; then
    echo "Error: Tensor directory does not exist: $TENSOR_DIR"
    echo "Please run tensor collection first or specify a valid directory"
    echo "Use --help to see usage information"
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

# Run unified visualization
echo ""
echo "Running unified tensor visualization..."
python script/visualization/tensor_visualizer.py \
    --tensor_dir "$TENSOR_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_workers "$MAX_WORKERS"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Unified visualization completed!"
    
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
    if [ -f "$OUTPUT_DIR/hifp8_analysis/hifp8_distribution_analysis.png" ]; then
        echo "  🔬 HiFP8 distribution analysis: $OUTPUT_DIR/hifp8_analysis/hifp8_distribution_analysis.png"
    fi
    if [ -f "$OUTPUT_DIR/global_statistics/global_statistics.json" ]; then
        echo "  📊 Global statistics (JSON): $OUTPUT_DIR/global_statistics/global_statistics.json"
    fi
    if [ -f "$OUTPUT_DIR/global_statistics/global_statistics_report.txt" ]; then
        echo "  📋 Global statistics report: $OUTPUT_DIR/global_statistics/global_statistics_report.txt"
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
    echo "❌ Unified visualization failed"
    echo "Please check the error messages above and ensure all dependencies are installed"
    exit 1
fi