#!/bin/bash

# =============================================================================
# run_draw_all.sh 使用示例
# 展示如何使用综合可视化脚本的各种功能
# =============================================================================

echo "=================================================================================="
echo "run_draw_all.sh 使用示例"
echo "=================================================================================="

# 示例1: 基本用法 - 运行所有可视化
echo ""
echo "示例1: 基本用法 - 运行所有可视化"
echo "命令: ./run_draw_all.sh"
echo "说明: 使用默认参数运行所有可视化功能"
echo ""

# 示例2: 指定目录和参数
echo "示例2: 指定目录和参数"
echo "命令: ./run_draw_all.sh --tensor-dir ./my_tensors --output-dir ./my_draw --layer 2 --rank 1"
echo "说明: 指定tensor目录、输出目录、层号和GPU rank"
echo ""

# 示例3: 跳过某些分析
echo "示例3: 跳过某些分析"
echo "命令: ./run_draw_all.sh --skip-layer-analysis --skip-overflow-analysis"
echo "说明: 只运行统一可视化分析，跳过层分析和溢出分析"
echo ""

# 示例4: 只运行层分析
echo "示例4: 只运行层分析"
echo "命令: ./run_draw_all.sh --skip-global-analysis --skip-overflow-analysis --layer 1 --rank 0 --layer-type attention"
echo "说明: 只运行层分析，分析第1层、rank 0的attention层"
echo ""

# 示例5: 启用量化对比
echo "示例5: 启用量化对比"
echo "命令: ./run_draw_all.sh --quantization-comparison --layer 2 --rank 1 --tensor-type output"
echo "说明: 启用量化对比分析，分析第2层、rank 1的output tensor"
echo ""

# 示例6: 使用高效模式
echo "示例6: 使用高效模式"
echo "命令: ./run_draw_all.sh --efficient-mode --layer 1 --rank 0"
echo "说明: 使用高效模式，只加载特定层和rank的文件"
echo ""

# 示例7: 显示帮助
echo "示例7: 显示帮助"
echo "命令: ./run_draw_all.sh --help"
echo "说明: 显示完整的帮助信息"
echo ""

echo "=================================================================================="
echo "支持的可视化功能:"
echo "=================================================================================="
echo "1. 量化类型对比分析 - 比较bf16, mxfp8, mxfp4, hifp8的分布"
echo "2. HiFP8分布分析 - 详细的HiFP8数值分布和统计"
echo "3. 全局统计分析 - 全面的统计报告和JSON数据"
echo "4. 层分析 - 特定层和rank的详细分析"
echo "5. 溢出检测分析 - 检测各量化类型的溢出情况"
echo "6. 多维度分析 - 按层、rank、类型等多维度分析"
echo ""

echo "=================================================================================="
echo "输出文件结构:"
echo "=================================================================================="
echo "draw/"
echo "├── quantization_analysis/          # 量化对比分析"
echo "│   └── quantization_comparison.png"
echo "├── hifp8_analysis/                 # HiFP8分析"
echo "│   └── hifp8_distribution_analysis.png"
echo "├── global_statistics/              # 全局统计"
echo "│   ├── global_statistics.json"
echo "│   └── global_statistics_report.txt"
echo "├── layer_analysis/                 # 层分析"
echo "│   └── layer_*_rank_*_*_analysis.png"
echo "├── overflow_analysis/              # 溢出分析"
echo "│   └── overflow_analysis_report.png"
echo "├── quant_analysis_*/               # 各量化类型分析"
echo "└── comprehensive_analysis_report.txt  # 综合报告"
echo ""

echo "=================================================================================="
echo "快速开始:"
echo "=================================================================================="
echo "1. 确保enhanced_tensor_logs目录存在且包含tensor文件"
echo "2. 运行: ./run_draw_all.sh"
echo "3. 查看draw目录中的结果"
echo "=================================================================================="
