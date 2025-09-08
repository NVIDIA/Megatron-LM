#!/bin/bash
"""
C4数据处理脚本
用于处理C4 (Colossal Clean Crawled Corpus) 数据集
"""

# 设置默认参数
INPUT_PATH=${1:-"./dataset/c4/**/*.json"}
OUTPUT_PREFIX=${2:-"./dataset/c4_processed"}
WORKERS=${3:-24}
PARTITIONS=${4:-6}
TOKENIZER_MODEL=${5:-"./model/llama3/"}
TOKENIZER_TYPE=${6:-"HuggingFaceTokenizer"}

# 可选参数
APPEND_EOD=${7:-"true"}
SEQUENCE_LENGTH=${8:-2048}
OVERWRITE=${9:-"false"}

echo "=== C4数据处理脚本 ==="
echo "输入路径: $INPUT_PATH"
echo "输出前缀: $OUTPUT_PREFIX"
echo "工作进程数: $WORKERS"
echo "分区数: $PARTITIONS"
echo "分词器模型: $TOKENIZER_MODEL"
echo "分词器类型: $TOKENIZER_TYPE"
echo "追加EOD: $APPEND_EOD"
echo "序列长度: $SEQUENCE_LENGTH"
echo "覆盖输出: $OVERWRITE"

# 检查输入路径
if [ ! -d "$(dirname "$INPUT_PATH")" ]; then
    echo "错误: 输入目录不存在: $(dirname "$INPUT_PATH")"
    exit 1
fi

# 检查分词器模型
if [ ! -d "$TOKENIZER_MODEL" ]; then
    echo "错误: 分词器模型目录不存在: $TOKENIZER_MODEL"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR=$(dirname "$OUTPUT_PREFIX")
mkdir -p "$OUTPUT_DIR"

# 构建命令
CMD="python tools/preprocess_data.py"
CMD="$CMD --input '$INPUT_PATH'"
CMD="$CMD --workers $WORKERS"
CMD="$CMD --partitions $PARTITIONS"
CMD="$CMD --output-prefix $OUTPUT_PREFIX"
CMD="$CMD --tokenizer-type $TOKENIZER_TYPE"
CMD="$CMD --tokenizer-model $TOKENIZER_MODEL"

# 添加可选参数
if [ "$APPEND_EOD" = "true" ]; then
    CMD="$CMD --append-eod"
fi

if [ "$SEQUENCE_LENGTH" != "2048" ]; then
    CMD="$CMD --seq-length $SEQUENCE_LENGTH"
fi

if [ "$OVERWRITE" = "true" ]; then
    CMD="$CMD --overwrite"
fi

echo ""
echo "执行命令:"
echo "$CMD"
echo ""

# 记录开始时间
START_TIME=$(date +%s)
echo "开始处理时间: $(date)"

# 执行命令
eval $CMD

# 检查执行结果
if [ $? -eq 0 ]; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ""
    echo "✅ C4数据处理完成!"
    echo "处理时间: ${DURATION}秒"
    echo "完成时间: $(date)"
    
    # 显示输出文件信息
    echo ""
    echo "输出文件:"
    ls -lh "${OUTPUT_PREFIX}"* 2>/dev/null || echo "未找到输出文件"
    
else
    echo ""
    echo "❌ C4数据处理失败!"
    exit 1
fi

echo ""
echo "=== 处理完成 ==="
