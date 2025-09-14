#!/bin/bash

# =============================================================================
# Overflow Detection Analysis Script
# 基于量化类型特征值检测tensor溢出情况
# 支持bf16, mxfp8, mxfp4, hifp8四种量化类型
# 支持Sample (0,1,2) 和 Layer (1-16) 的多维度分析
# =============================================================================

# 设置脚本元数据
SCRIPT_NAME="$(basename "$0")"
SCRIPT_VERSION="1.0.0"
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "=================================================================================="
echo "Overflow Detection Analysis Script"
echo "Script: $SCRIPT_NAME"
echo "Version: $SCRIPT_VERSION"
echo "Start Time: $START_TIME"
echo "=================================================================================="

# 默认参数
TENSOR_DIR=${1:-"./enhanced_tensor_logs"}
OUTPUT_DIR=${2:-"./draw"}
MAX_WORKERS=${3:-32}

echo "参数设置:"
echo "  - Tensor目录: $TENSOR_DIR"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 最大线程数: $MAX_WORKERS"

# 检查tensor目录是否存在
if [ ! -d "$TENSOR_DIR" ]; then
    echo "错误: Tensor目录不存在: $TENSOR_DIR"
    echo "请确保已经运行了训练脚本并生成了tensor文件"
    exit 1
fi

# 检查量化类型目录
QUANT_TYPES=("bf16" "mxfp8" "mxfp4" "hifp8")
echo ""
echo "检查量化类型目录:"
total_files=0
for quant_type in "${QUANT_TYPES[@]}"; do
    quant_dir="$TENSOR_DIR/$quant_type"
    if [ -d "$quant_dir" ]; then
        file_count=$(find "$quant_dir" -name "*.pt" 2>/dev/null | wc -l)
        echo "  ✅ $quant_type: $file_count 个文件"
        total_files=$((total_files + file_count))
    else
        echo "  ❌ $quant_type: 目录不存在"
    fi
done

echo ""
echo "总文件数: $total_files"

if [ $total_files -eq 0 ]; then
    echo "错误: 没有找到任何tensor文件"
    exit 1
fi

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

# 检查必要的Python包
echo ""
echo "检查Python依赖..."
python -c "import torch, matplotlib, numpy, pandas, seaborn, concurrent.futures" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: 缺少必要的Python包，正在尝试安装..."
    pip install matplotlib numpy pandas seaborn scipy
fi

# 显示量化类型限制值信息
echo ""
echo "量化类型限制值信息:"
echo "  bf16:"
echo "    - 最大正常值: 65504.0"
echo "    - 最小正常值: 6.103515625e-05"
echo "    - 指数范围: [-14, 15]"
echo "  hifp8:"
echo "    - 最大正常值: 32768.0"
echo "    - 最小正常值: 3.0517578125e-05"
echo "    - 指数范围: [-15, 15]"
echo "  mxfp8 (FP8-E4M3):"
echo "    - 最大正常值: 448.0"
echo "    - 最小正常值: 0.015625"
echo "    - 指数范围: [-6, 8]"
echo "  mxfp4 (FP4-E2M1):"
echo "    - 最大正常值: 12.0"
echo "    - 最小正常值: 0.25"
echo "    - 指数范围: [-2, 3]"

# 运行溢出检测分析
echo ""
echo "运行溢出检测分析..."
python script/visualization/overflow_detection_analyzer.py \
    --tensor_dir "$TENSOR_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_workers "$MAX_WORKERS"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 溢出检测分析完成!"
    
    # 显示生成的文件
    echo ""
    echo "生成的分析文件:"
    find "$OUTPUT_DIR" -name "*.png" | while read file; do
        echo "  - $(basename "$file")"
    done
    
    echo ""
    find "$OUTPUT_DIR" -name "*.txt" | while read file; do
        echo "  - $(basename "$file")"
    done
    
    # 显示主要输出
    echo ""
    echo "主要输出文件:"
    if [ -f "$OUTPUT_DIR/overflow_analysis/overflow_analysis_report.png" ]; then
        echo "  🎯 溢出分析图: $OUTPUT_DIR/overflow_analysis/overflow_analysis_report.png"
    fi
    if [ -f "$OUTPUT_DIR/detailed_reports/overflow_detection_report.txt" ]; then
        echo "  📋 详细报告: $OUTPUT_DIR/detailed_reports/overflow_detection_report.txt"
    fi
    
    # 显示溢出统计摘要
    echo ""
    echo "溢出统计摘要:"
    if [ -f "$OUTPUT_DIR/detailed_reports/overflow_detection_report.txt" ]; then
        echo "  - 查看详细报告了解各量化类型的溢出情况"
        echo "  - 上溢出: 数值超过最大正常值"
        echo "  - 下溢出: 数值小于最小正常值"
        echo "  - 溢出率: 溢出数值占总数值的百分比"
    fi
    
    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo ""
    echo "=================================================================================="
    echo "溢出检测分析完成"
    echo "开始时间: $START_TIME"
    echo "结束时间: $END_TIME"
    echo "=================================================================================="
else
    echo ""
    echo "❌ 溢出检测分析失败"
    exit 1
fi
