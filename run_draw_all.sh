#!/bin/bash

# =============================================================================
# Comprehensive Tensor Visualization Script
# 实现所有支持的可视化功能，包括：
#   - 量化类型对比分析 (Quantization Comparison)
#   - HiFP8分布分析 (HiFP8 Distribution Analysis)
#   - 全局统计分析 (Global Statistics)
#   - 层分析 (Layer Analysis)
#   - 溢出检测分析 (Overflow Analysis)
#   - 多维度分析 (Multi-dimensional Analysis)
# =============================================================================

# 设置脚本元数据
SCRIPT_NAME="$(basename "$0")"
SCRIPT_VERSION="1.0.0"
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "=================================================================================="
echo "Comprehensive Tensor Visualization Script"
echo "Script: $SCRIPT_NAME"
echo "Version: $SCRIPT_VERSION"
echo "Start Time: $START_TIME"
echo "=================================================================================="

# 默认参数
TENSOR_DIR="./enhanced_tensor_logs"
OUTPUT_DIR="./draw"
MAX_WORKERS=4
LAYER=1
RANK=0
LAYER_TYPE="attention"
TENSOR_TYPE=""
QUANTIZATION_COMPARISON="true"
EFFICIENT_MODE="true"
SKIP_LAYER_ANALYSIS="false"
SKIP_OVERFLOW_ANALYSIS="false"
SKIP_GLOBAL_ANALYSIS="false"

# 显示使用帮助
show_help() {
    echo "用法: $0 [OPTIONS]"
    echo ""
    echo "选项:"
    echo "  -h, --help                    显示此帮助信息"
    echo "  --tensor-dir DIR              Tensor目录 [默认: ./enhanced_tensor_logs]"
    echo "  --output-dir DIR              输出目录 [默认: ./draw]"
    echo "  --max-workers NUM             最大工作线程数 [默认: 4]"
    echo "  --layer NUM                   层号 [默认: 1]"
    echo "  --rank NUM                    GPU rank [默认: 0]"
    echo "  --layer-type TYPE             层类型 (attention|linear) [默认: attention]"
    echo "  --tensor-type TYPE            特定tensor类型 [默认: 空]"
    echo "  --quantization-comparison     启用量化对比分析 [默认: true]"
    echo "  --efficient-mode              使用高效模式 [默认: true]"
    echo "  --skip-layer-analysis         跳过层分析"
    echo "  --skip-overflow-analysis      跳过溢出分析"
    echo "  --skip-global-analysis        跳过全局分析"
    echo ""
    echo "支持的可视化功能:"
    echo "  1. 量化类型对比分析 - 比较bf16, mxfp8, mxfp4, hifp8的分布"
    echo "  2. HiFP8分布分析 - 详细的HiFP8数值分布和统计"
    echo "  3. 全局统计分析 - 全面的统计报告和JSON数据"
    echo "  4. 层分析 - 特定层和rank的详细分析"
    echo "  5. 溢出检测分析 - 检测各量化类型的溢出情况"
    echo "  6. 多维度分析 - 按层、rank、类型等多维度分析"
    echo ""
    echo "使用示例:"
    echo "  # 运行所有可视化"
    echo "  $0"
    echo ""
    echo "  # 指定目录和参数"
    echo "  $0 --tensor-dir ./my_tensors --output-dir ./my_draw --layer 2 --rank 1"
    echo ""
    echo "  # 跳过某些分析"
    echo "  $0 --skip-layer-analysis --skip-overflow-analysis"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
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
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --rank)
            RANK="$2"
            shift 2
            ;;
        --layer-type)
            LAYER_TYPE="$2"
            shift 2
            ;;
        --tensor-type)
            TENSOR_TYPE="$2"
            shift 2
            ;;
        --quantization-comparison)
            QUANTIZATION_COMPARISON="true"
            shift
            ;;
        --no-quantization-comparison)
            QUANTIZATION_COMPARISON="false"
            shift
            ;;
        --efficient-mode)
            EFFICIENT_MODE="true"
            shift
            ;;
        --no-efficient-mode)
            EFFICIENT_MODE="false"
            shift
            ;;
        --skip-layer-analysis)
            SKIP_LAYER_ANALYSIS="true"
            shift
            ;;
        --skip-overflow-analysis)
            SKIP_OVERFLOW_ANALYSIS="true"
            shift
            ;;
        --skip-global-analysis)
            SKIP_GLOBAL_ANALYSIS="true"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

echo "参数设置:"
echo "  - Tensor目录: $TENSOR_DIR"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 最大工作线程: $MAX_WORKERS"
echo "  - 层号: $LAYER"
echo "  - GPU rank: $RANK"
echo "  - 层类型: $LAYER_TYPE"
echo "  - Tensor类型: $TENSOR_TYPE"
echo "  - 量化对比: $QUANTIZATION_COMPARISON"
echo "  - 高效模式: $EFFICIENT_MODE"
echo "  - 跳过层分析: $SKIP_LAYER_ANALYSIS"
echo "  - 跳过溢出分析: $SKIP_OVERFLOW_ANALYSIS"
echo "  - 跳过全局分析: $SKIP_GLOBAL_ANALYSIS"

# 检查tensor目录是否存在
if [ ! -d "$TENSOR_DIR" ]; then
    echo "Error: Tensor目录不存在: $TENSOR_DIR"
    echo "请确保tensor文件可用"
    exit 1
fi

# 检查量化类型目录
QUANT_TYPES=("bf16" "mxfp8" "mxfp4" "hifp8")
echo ""
echo "检查量化类型目录:"
AVAILABLE_TYPES=()
for quant_type in "${QUANT_TYPES[@]}"; do
    quant_dir="$TENSOR_DIR/$quant_type"
    if [ -d "$quant_dir" ]; then
        file_count=$(find "$quant_dir" -name "*.pt" 2>/dev/null | wc -l)
        echo "  ✅ $quant_type: $file_count files"
        if [ $file_count -gt 0 ]; then
            AVAILABLE_TYPES+=("$quant_type")
        fi
    else
        echo "  ❌ $quant_type: 目录不存在"
    fi
done

if [ ${#AVAILABLE_TYPES[@]} -eq 0 ]; then
    echo "Error: 没有找到可用的量化类型数据"
    exit 1
fi

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "Error: Python未找到"
    exit 1
fi

# 检查必需的Python包
echo ""
echo "检查Python依赖..."
python -c "import torch, matplotlib, numpy, pandas, seaborn, concurrent.futures, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: 缺少必需的Python包，尝试安装..."
    pip install matplotlib numpy pandas seaborn scipy tqdm
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行可视化分析
echo ""
echo "开始运行综合可视化分析..."
echo "=================================================================================="

# 1. 运行统一可视化 (包含量化对比、HiFP8分析、全局统计)
if [ "$SKIP_GLOBAL_ANALYSIS" = "false" ]; then
    echo ""
    echo "1️⃣ 运行统一可视化分析 (量化对比 + HiFP8分析 + 全局统计)..."
    echo "--------------------------------------------------------------------------------"
    
    python script/visualization/tensor_visualizer.py \
        --tensor_dir "$TENSOR_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --max_workers "$MAX_WORKERS" \
        --analysis_type "all"
    
    if [ $? -eq 0 ]; then
        echo "✅ 统一可视化分析完成"
    else
        echo "❌ 统一可视化分析失败"
    fi
else
    echo "⏭️  跳过统一可视化分析"
fi

# 2. 运行层分析
if [ "$SKIP_LAYER_ANALYSIS" = "false" ]; then
    echo ""
    echo "2️⃣ 运行层分析 (Layer $LAYER, Rank $RANK, Type $LAYER_TYPE)..."
    echo "--------------------------------------------------------------------------------"
    
    # 构建层分析命令
    LAYER_CMD="python script/visualization/tensor_visualizer.py --tensor_dir \"$TENSOR_DIR\" --output_dir \"$OUTPUT_DIR\" --layer $LAYER --rank $RANK --layer_type $LAYER_TYPE --efficient_mode $EFFICIENT_MODE --analysis_type layer"
    
    # 如果指定了tensor类型，添加参数
    if [ -n "$TENSOR_TYPE" ]; then
        LAYER_CMD="$LAYER_CMD --tensor_type $TENSOR_TYPE"
    fi
    
    # 如果启用量化对比，添加参数
    if [ "$QUANTIZATION_COMPARISON" = "true" ]; then
        LAYER_CMD="$LAYER_CMD --quantization_comparison"
    fi
    
    # 运行层分析
    eval $LAYER_CMD
    
    if [ $? -eq 0 ]; then
        echo "✅ 层分析完成"
    else
        echo "❌ 层分析失败"
    fi
else
    echo "⏭️  跳过层分析"
fi

# 3. 运行溢出分析
if [ "$SKIP_OVERFLOW_ANALYSIS" = "false" ]; then
    echo ""
    echo "3️⃣ 运行溢出检测分析..."
    echo "--------------------------------------------------------------------------------"
    
    python script/visualization/tensor_visualizer.py \
        --tensor_dir "$TENSOR_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --max_workers "$MAX_WORKERS" \
        --analysis_type "overflow"
    
    if [ $? -eq 0 ]; then
        echo "✅ 溢出分析完成"
    else
        echo "❌ 溢出分析失败"
    fi
else
    echo "⏭️  跳过溢出分析"
fi

# 4. 运行多维度分析 (为每个可用的量化类型运行层分析)
if [ "$SKIP_LAYER_ANALYSIS" = "false" ] && [ ${#AVAILABLE_TYPES[@]} -gt 1 ]; then
    echo ""
    echo "4️⃣ 运行多维度分析 (为每个量化类型)..."
    echo "--------------------------------------------------------------------------------"
    
    for quant_type in "${AVAILABLE_TYPES[@]}"; do
        echo "  分析量化类型: $quant_type"
        
        # 为每个量化类型创建专门的输出目录
        quant_output_dir="$OUTPUT_DIR/quant_analysis_$quant_type"
        mkdir -p "$quant_output_dir"
        
        # 运行该量化类型的分析
        python script/visualization/tensor_visualizer.py \
            --tensor_dir "$TENSOR_DIR" \
            --output_dir "$quant_output_dir" \
            --max_workers "$MAX_WORKERS" \
            --analysis_type "all"
        
        if [ $? -eq 0 ]; then
            echo "    ✅ $quant_type 分析完成"
        else
            echo "    ❌ $quant_type 分析失败"
        fi
    done
else
    echo "⏭️  跳过多维度分析"
fi

# 5. 生成综合报告
echo ""
echo "5️⃣ 生成综合报告..."
echo "--------------------------------------------------------------------------------"

# 创建综合报告
REPORT_FILE="$OUTPUT_DIR/comprehensive_analysis_report.txt"
cat > "$REPORT_FILE" << EOF
================================================================================
Comprehensive Tensor Analysis Report
================================================================================
Analysis Time: $START_TIME
Tensor Directory: $TENSOR_DIR
Output Directory: $OUTPUT_DIR
Available Quantization Types: ${AVAILABLE_TYPES[*]}

================================================================================
Analysis Summary
================================================================================

1. Unified Visualization Analysis
   - Quantization Comparison: $([ "$SKIP_GLOBAL_ANALYSIS" = "false" ] && echo "✅ Completed" || echo "⏭️ Skipped")
   - HiFP8 Distribution Analysis: $([ "$SKIP_GLOBAL_ANALYSIS" = "false" ] && echo "✅ Completed" || echo "⏭️ Skipped")
   - Global Statistics: $([ "$SKIP_GLOBAL_ANALYSIS" = "false" ] && echo "✅ Completed" || echo "⏭️ Skipped")

2. Layer Analysis
   - Layer: $LAYER
   - Rank: $RANK
   - Type: $LAYER_TYPE
   - Status: $([ "$SKIP_LAYER_ANALYSIS" = "false" ] && echo "✅ Completed" || echo "⏭️ Skipped")

3. Overflow Analysis
   - Status: $([ "$SKIP_OVERFLOW_ANALYSIS" = "false" ] && echo "✅ Completed" || echo "⏭️ Skipped")

4. Multi-dimensional Analysis
   - Quantization Types: ${#AVAILABLE_TYPES[@]}
   - Status: $([ "$SKIP_LAYER_ANALYSIS" = "false" ] && [ ${#AVAILABLE_TYPES[@]} -gt 1 ] && echo "✅ Completed" || echo "⏭️ Skipped")

================================================================================
Generated Files
================================================================================

EOF

# 添加生成的文件列表
echo "Generated visualization files:" >> "$REPORT_FILE"
find "$OUTPUT_DIR" -name "*.png" | sort | while read file; do
    echo "  - $(basename "$file")" >> "$REPORT_FILE"
done

echo "" >> "$REPORT_FILE"
echo "Generated report files:" >> "$REPORT_FILE"
find "$OUTPUT_DIR" -name "*.txt" | sort | while read file; do
    echo "  - $(basename "$file")" >> "$REPORT_FILE"
done

echo "" >> "$REPORT_FILE"
echo "Generated JSON files:" >> "$REPORT_FILE"
find "$OUTPUT_DIR" -name "*.json" | sort | while read file; do
    echo "  - $(basename "$file")" >> "$REPORT_FILE"
done

echo "✅ 综合报告生成完成: $REPORT_FILE"

# 显示主要输出文件
echo ""
echo "=================================================================================="
echo "主要输出文件:"
echo "=================================================================================="

# 显示主要图表
if [ -f "$OUTPUT_DIR/quantization_analysis/quantization_comparison.png" ]; then
    echo "🎯 量化对比分析: $OUTPUT_DIR/quantization_analysis/quantization_comparison.png"
fi

if [ -f "$OUTPUT_DIR/hifp8_analysis/hifp8_distribution_analysis.png" ]; then
    echo "🔬 HiFP8分布分析: $OUTPUT_DIR/hifp8_analysis/hifp8_distribution_analysis.png"
fi

if [ -f "$OUTPUT_DIR/global_statistics/global_statistics.json" ]; then
    echo "📊 全局统计 (JSON): $OUTPUT_DIR/global_statistics/global_statistics.json"
fi

if [ -f "$OUTPUT_DIR/global_statistics/global_statistics_report.txt" ]; then
    echo "📋 全局统计报告: $OUTPUT_DIR/global_statistics/global_statistics_report.txt"
fi

if [ -f "$OUTPUT_DIR/layer_analysis/layer_${LAYER}_rank_${RANK}_${LAYER_TYPE}_analysis.png" ]; then
    echo "🔍 层分析: $OUTPUT_DIR/layer_analysis/layer_${LAYER}_rank_${RANK}_${LAYER_TYPE}_analysis.png"
fi

if [ -f "$OUTPUT_DIR/overflow_analysis/overflow_analysis_report.png" ]; then
    echo "⚠️  溢出分析: $OUTPUT_DIR/overflow_analysis/overflow_analysis_report.png"
fi

if [ -f "$REPORT_FILE" ]; then
    echo "📄 综合报告: $REPORT_FILE"
fi

# 显示所有生成的文件
echo ""
echo "所有生成的文件:"
echo "--------------------------------------------------------------------------------"
find "$OUTPUT_DIR" -type f \( -name "*.png" -o -name "*.txt" -o -name "*.json" \) | sort | while read file; do
    echo "  - $file"
done

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "=================================================================================="
echo "综合可视化分析完成"
echo "开始时间: $START_TIME"
echo "结束时间: $END_TIME"
echo "输出目录: $OUTPUT_DIR"
echo "=================================================================================="
