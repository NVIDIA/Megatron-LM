#!/bin/bash
"""
一键可视化脚本
自动检测tensor文件并生成可视化图表
"""

# 设置默认参数
TENSOR_DIR=${1:-"./enhanced_tensor_logs"}
OUTPUT_DIR=${2:-"./draw"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== 一键Tensor可视化工具 ==="
echo "Tensor目录: $TENSOR_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "脚本目录: $SCRIPT_DIR"

# 检查tensor目录是否存在
if [ ! -d "$TENSOR_DIR" ]; then
    echo "错误: Tensor目录不存在: $TENSOR_DIR"
    echo "请确保已经运行过训练脚本并生成了tensor文件"
    exit 1
fi

# 检查是否有tensor文件
TENSOR_COUNT=$(find "$TENSOR_DIR" -name "*.pt" | wc -l)
if [ "$TENSOR_COUNT" -eq 0 ]; then
    echo "错误: 在 $TENSOR_DIR 中没有找到tensor文件 (*.pt)"
    echo "请确保已经运行过训练脚本并生成了tensor文件"
    exit 1
fi

echo "找到 $TENSOR_COUNT 个tensor文件"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 没有找到Python"
    exit 1
fi

# 检查必要的Python包
echo "检查Python依赖..."
python -c "import torch, matplotlib, numpy, pandas, seaborn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: 缺少必要的Python包，尝试安装..."
    pip install matplotlib numpy pandas seaborn scipy
fi

# 运行增强版快速可视化
echo "运行增强版快速可视化..."
python "$SCRIPT_DIR/quick_visualize_enhanced.py" --tensor_dir "$TENSOR_DIR" --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "✅ 增强版快速可视化完成"
else
    echo "❌ 增强版快速可视化失败，尝试基础版本..."
    python "$SCRIPT_DIR/quick_visualize.py" --tensor_dir "$TENSOR_DIR" --output_dir "$OUTPUT_DIR"
    if [ $? -eq 0 ]; then
        echo "✅ 基础版快速可视化完成"
    else
        echo "❌ 基础版快速可视化也失败"
    fi
fi

# 运行完整可视化（如果文件数量不太多）
if [ "$TENSOR_COUNT" -le 100 ]; then
    echo "运行完整可视化..."
    python "$SCRIPT_DIR/enhanced_tensor_visualizer.py" --tensor_dir "$TENSOR_DIR" --output_dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✅ 完整可视化完成"
    else
        echo "❌ 完整可视化失败，尝试基础版本..."
        python "$SCRIPT_DIR/visualize_tensors.py" --tensor_dir "$TENSOR_DIR" --output_dir "$OUTPUT_DIR" --max_files 50
        if [ $? -eq 0 ]; then
            echo "✅ 基础版完整可视化完成"
        else
            echo "❌ 基础版完整可视化也失败"
        fi
    fi
else
    echo "⚠️  Tensor文件数量较多 ($TENSOR_COUNT)，跳过完整可视化"
    echo "   如需完整可视化，请手动运行:"
    echo "   python $SCRIPT_DIR/enhanced_tensor_visualizer.py --tensor_dir $TENSOR_DIR --output_dir $OUTPUT_DIR"
fi

# 显示结果
echo ""
echo "=== 可视化完成 ==="
echo "输出目录: $OUTPUT_DIR"
echo "生成的文件:"
find "$OUTPUT_DIR" -name "*.png" -o -name "*.txt" | head -10

# 检查生成的主要文件
if [ -f "$OUTPUT_DIR/summary_analysis.png" ]; then
    echo ""
    echo "🎉 主要分析图: $OUTPUT_DIR/summary_analysis.png"
fi

if [ -f "$OUTPUT_DIR/detailed_tensor_stats.txt" ]; then
    echo "📊 详细统计报告: $OUTPUT_DIR/detailed_tensor_stats.txt"
fi

# 检查子目录
if [ -d "$OUTPUT_DIR/quantization_analysis" ]; then
    echo "🔍 量化分析图: $OUTPUT_DIR/quantization_analysis/"
fi

if [ -d "$OUTPUT_DIR/attention_analysis" ]; then
    echo "🧠 Attention分析图: $OUTPUT_DIR/attention_analysis/"
fi

echo ""
echo "💡 提示:"
echo "   - 查看 summary_analysis.png 了解基本统计信息"
echo "   - 查看 detailed_tensor_stats.txt 了解详细统计报告"
echo "   - 查看 quantization_analysis/ 目录了解量化类型对比"
echo "   - 查看 attention_analysis/ 目录了解attention层分析"
echo "   - 查看 *_quantization_comparison.png 了解量化效果对比"
echo "   - 查看 *_attention_analysis.png 了解attention层详细分析"
