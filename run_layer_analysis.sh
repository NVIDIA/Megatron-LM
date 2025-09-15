#!/bin/bash

# =============================================================================
# Layer Distribution Analysis Script
# 专门分析某个层的tensor分布，支持attention和linear层的详细分析
# =============================================================================

# 设置脚本元数据
SCRIPT_NAME="$(basename "$0")"
SCRIPT_VERSION="1.0.0"
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "=================================================================================="
echo "Layer Distribution Analysis Script"
echo "Script: $SCRIPT_NAME"
echo "Version: $SCRIPT_VERSION"
echo "Start Time: $START_TIME"
echo "=================================================================================="

# 默认参数
TENSOR_DIR=${1:-"./enhanced_tensor_logs"}
OUTPUT_DIR=${2:-"./layer_analysis_output"}
LAYER=${3:-1}
SAMPLE=${4:-0}  # Sample now represents rank (different GPUs = different samples)
LAYER_TYPE=${5:-"attention"}
TENSOR_TYPE=${6:-""}
QUANTIZATION_COMPARISON=${7:-"false"}

echo "Parameter settings:"
echo "  - Tensor directory: $TENSOR_DIR"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Layer: $LAYER"
echo "  - Sample (Rank): $SAMPLE (Note: Sample now represents GPU rank since only one micro_batch is collected)"
echo "  - Layer type: $LAYER_TYPE"
echo "  - Tensor type: $TENSOR_TYPE"
echo "  - Quantization comparison: $QUANTIZATION_COMPARISON"

# 检查tensor目录是否存在
if [ ! -d "$TENSOR_DIR" ]; then
    echo "Error: Tensor directory does not exist: $TENSOR_DIR"
    echo "Please ensure tensor files are available"
    exit 1
fi

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# 检查必需的Python包
echo ""
echo "Checking Python dependencies..."
python -c "import torch, matplotlib, numpy, pandas, seaborn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Missing required Python packages, attempting to install..."
    pip install matplotlib numpy pandas seaborn scipy
fi

# 构建Python命令
PYTHON_CMD="python script/visualization/tensor_visualizer.py --tensor_dir \"$TENSOR_DIR\" --output_dir \"$OUTPUT_DIR\" --layer $LAYER --sample $SAMPLE --layer_type $LAYER_TYPE"

# 如果指定了tensor类型，添加参数
if [ -n "$TENSOR_TYPE" ]; then
    PYTHON_CMD="$PYTHON_CMD --tensor_type $TENSOR_TYPE"
fi

# 如果启用量化对比，添加参数
if [ "$QUANTIZATION_COMPARISON" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --quantization_comparison"
fi

# 运行分析
echo ""
echo "Running layer distribution analysis..."
eval $PYTHON_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Layer distribution analysis completed!"
    
    # 显示生成的文件
    echo ""
    echo "Generated files:"
    find "$OUTPUT_DIR" -name "*.png" | while read file; do
        echo "  - $(basename "$file")"
    done
    
    echo ""
    find "$OUTPUT_DIR" -name "*.txt" | while read file; do
        echo "  - $(basename "$file")"
    done
    
    # 显示主要输出
    echo ""
    echo "Main output files:"
    if [ -f "$OUTPUT_DIR/layer_analysis/layer_${LAYER}_sample_${SAMPLE}_${LAYER_TYPE}_analysis.png" ]; then
        echo "  🎯 Layer analysis: $OUTPUT_DIR/layer_analysis/layer_${LAYER}_sample_${SAMPLE}_${LAYER_TYPE}_analysis.png"
    fi
    if [ -n "$TENSOR_TYPE" ] && [ "$QUANTIZATION_COMPARISON" = "true" ]; then
        if [ -f "$OUTPUT_DIR/quantization_analysis/quantization_comparison_layer_${LAYER}_sample_${SAMPLE}_${LAYER_TYPE}_${TENSOR_TYPE}.png" ]; then
            echo "  🔍 Quantization comparison: $OUTPUT_DIR/quantization_analysis/quantization_comparison_layer_${LAYER}_sample_${SAMPLE}_${LAYER_TYPE}_${TENSOR_TYPE}.png"
        fi
    fi
    if [ -f "$OUTPUT_DIR/global_statistics/statistics_layer_${LAYER}_sample_${SAMPLE}_${LAYER_TYPE}.txt" ]; then
        echo "  📋 Statistics report: $OUTPUT_DIR/global_statistics/statistics_layer_${LAYER}_sample_${SAMPLE}_${LAYER_TYPE}.txt"
    fi
    
    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo ""
    echo "=================================================================================="
    echo "Analysis completed"
    echo "Start time: $START_TIME"
    echo "End time: $END_TIME"
    echo "=================================================================================="
else
    echo ""
    echo "❌ Layer distribution analysis failed"
    exit 1
fi
