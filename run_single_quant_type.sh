#!/bin/bash

# =============================================================================
# 单次量化类型Tensor收集脚本
# 用法: ./run_single_quant_type.sh <quant_type>
# 例如: ./run_single_quant_type.sh mxfp8
# =============================================================================

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <quant_type>"
    echo "支持的量化类型: mxfp8, mxfp4, hifp8"
    echo "例如: $0 mxfp8"
    exit 1
fi

QUANT_TYPE=$1
VALID_TYPES=("mxfp8" "mxfp4" "hifp8")

# 验证量化类型
if [[ ! " ${VALID_TYPES[@]} " =~ " ${QUANT_TYPE} " ]]; then
    echo "错误: 不支持的量化类型 '$QUANT_TYPE'"
    echo "支持的量化类型: ${VALID_TYPES[*]}"
    exit 1
fi

echo "=================================================================================="
echo "单次量化类型Tensor收集"
echo "量化类型: $QUANT_TYPE"
echo "=================================================================================="

# 配置参数
BASE_TENSOR_PATH="enhanced_tensor_logs"
TOKENIZER_PATH="model/llama3.2-1b"
DATA_PATH="dataset/wikipedia_processed/wikipedia_processed_text_document"
DTYPE="bf16"

# 设置tensor保存环境
export TENSOR_SAVE_ENABLED="true"
export TENSOR_SAVE_DIR="$BASE_TENSOR_PATH/${QUANT_TYPE}"
export HOST_TENSORBOARD_LOGS_PATH="tensorboard_logs/${QUANT_TYPE}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 配置信息:"
echo "  - 量化类型: $QUANT_TYPE"
echo "  - Tensor保存路径: $TENSOR_SAVE_DIR"
echo "  - TensorBoard路径: $HOST_TENSORBOARD_LOGS_PATH"
echo "  - 分词器路径: $TOKENIZER_PATH"
echo "  - 数据路径: $DATA_PATH"
echo "  - 数据类型: $DTYPE"

# 检查必要文件
if [ ! -f "examples/llama/train_llama32_1b_h100_fp8.sh" ]; then
    echo "[ERROR] 训练脚本不存在: examples/llama/train_llama32_1b_h100_fp8.sh"
    exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
    echo "[ERROR] 数据路径不存在: $DATA_PATH"
    exit 1
fi

# 创建目录
mkdir -p "$TENSOR_SAVE_DIR"
mkdir -p "$HOST_TENSORBOARD_LOGS_PATH"
mkdir -p "checkpoints/llama32_1b/${QUANT_TYPE}"

# 修改量化类型
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 修改量化类型为: $QUANT_TYPE"

# 修改linear层量化类型
sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'$QUANT_TYPE'/" \
    megatron/core/tensor_parallel/layers.py

# 修改attention层量化类型
sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'$QUANT_TYPE'/" \
    megatron/core/transformer/dot_product_attention.py

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] 量化类型修改完成"

# 运行训练脚本
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 开始训练并收集tensor..."

CHECKPOINT_PATH="checkpoints/llama32_1b/${QUANT_TYPE}"

# 执行训练脚本
bash examples/llama/train_llama32_1b_h100_fp8.sh \
    "$CHECKPOINT_PATH" \
    "$HOST_TENSORBOARD_LOGS_PATH" \
    "$TOKENIZER_PATH" \
    "$DATA_PATH" \
    "$DTYPE" \
    2>&1 | tee "${HOST_TENSORBOARD_LOGS_PATH}/training_${QUANT_TYPE}_$(date +'%y-%m-%d_%H-%M-%S').log"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

# 分析结果
echo ""
echo "=================================================================================="
echo "Tensor收集结果分析"
echo "=================================================================================="

if [ -d "$TENSOR_SAVE_DIR" ]; then
    tensor_count=$(find "$TENSOR_SAVE_DIR" -name "*.pt" 2>/dev/null | wc -l)
    echo "[INFO] 收集到的tensor文件数量: $tensor_count"
    
    if [ $tensor_count -gt 0 ]; then
        echo ""
        echo "Tensor文件统计:"
        
        # 统计不同类型
        pre_count=$(find "$TENSOR_SAVE_DIR" -name "*_pre_*" 2>/dev/null | wc -l)
        post_count=$(find "$TENSOR_SAVE_DIR" -name "*_post_*" 2>/dev/null | wc -l)
        fa_count=$(find "$TENSOR_SAVE_DIR" -name "*_FA_*" 2>/dev/null | wc -l)
        linear_count=$(find "$TENSOR_SAVE_DIR" -name "*_linear_*" 2>/dev/null | wc -l)
        forward_count=$(find "$TENSOR_SAVE_DIR" -name "*_forward_*" 2>/dev/null | wc -l)
        backward_count=$(find "$TENSOR_SAVE_DIR" -name "*_backward_*" 2>/dev/null | wc -l)
        
        echo "  - Pre阶段: $pre_count 个"
        echo "  - Post阶段: $post_count 个"
        echo "  - FA组件: $fa_count 个"
        echo "  - Linear组件: $linear_count 个"
        echo "  - Forward操作: $forward_count 个"
        echo "  - Backward操作: $backward_count 个"
        
        echo ""
        echo "部分tensor文件示例:"
        find "$TENSOR_SAVE_DIR" -name "*.pt" | head -5 | while read file; do
            echo "  - $(basename "$file")"
        done
        
        if [ $tensor_count -gt 5 ]; then
            echo "  - ... 还有 $((tensor_count - 5)) 个文件"
        fi
    else
        echo "[WARNING] 未收集到任何tensor文件"
    fi
else
    echo "[ERROR] Tensor保存目录不存在: $TENSOR_SAVE_DIR"
fi

# 最终状态
echo ""
echo "=================================================================================="
echo "执行完成"
echo "=================================================================================="

if [[ $TRAINING_EXIT_CODE -eq 0 ]]; then
    echo "[SUCCESS] 训练和tensor收集完成"
    echo "  - 检查点保存到: $CHECKPOINT_PATH"
    echo "  - TensorBoard日志: $HOST_TENSORBOARD_LOGS_PATH"
    echo "  - Tensor文件: $TENSOR_SAVE_DIR"
else
    echo "[ERROR] 训练失败，退出代码: $TRAINING_EXIT_CODE"
fi

echo ""
echo "下一步操作建议:"
echo "1. 查看收集到的tensor文件:"
echo "   ls -la $TENSOR_SAVE_DIR/"
echo ""
echo "2. 使用可视化脚本分析:"
echo "   python script/visualization/quick_visualize.py --tensor_dir $TENSOR_SAVE_DIR"
echo ""
echo "3. 使用一键可视化:"
echo "   bash script/visualization/one_click_visualize.sh $TENSOR_SAVE_DIR"

exit $TRAINING_EXIT_CODE
