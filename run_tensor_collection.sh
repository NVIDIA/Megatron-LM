#!/bin/bash

# =============================================================================
# 统一Tensor收集脚本
# 简化并整合了所有tensor收集功能，支持多种使用模式
# =============================================================================

# 脚本元数据
SCRIPT_NAME="$(basename "$0")"
SCRIPT_VERSION="2.0.0"
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "=================================================================================="
echo "统一Tensor收集脚本"
echo "Script: $SCRIPT_NAME"
echo "Version: $SCRIPT_VERSION"
echo "Start Time: $START_TIME"
echo "=================================================================================="

# 默认参数
MODE="single"
QUANT_TYPE="mxfp8"
BASE_TENSOR_PATH="./enhanced_tensor_logs"
TOKENIZER_PATH="model/llama3.2-1b"
DATA_PATH="dataset/wikipedia_processed/wikipedia_processed_text_document"
DTYPE="bf16"
CONTROL_ITER=1  # 控制收集的micro_batch数量
COLLECT_MICRO_BATCHES=1  # 收集的micro_batch数量

# 显示使用帮助
show_help() {
    echo "用法: $0 [OPTIONS] [MODE] [QUANT_TYPE]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  --mode MODE             收集模式 (single|batch|quick) [默认: single]"
    echo "  --quant-type TYPE       量化类型 (bf16|mxfp8|mxfp4|hifp8) [默认: mxfp8]"
    echo "  --tensor-path PATH      Tensor保存路径 [默认: ./enhanced_tensor_logs]"
    echo "  --tokenizer-path PATH   分词器路径 [默认: model/llama3.2-1b]"
    echo "  --data-path PATH        数据路径 [默认: dataset/wikipedia_processed/wikipedia_processed_text_document]"
    echo "  --dtype TYPE            数据类型 [默认: bf16]"
    echo "  --control-iter NUM      控制收集的micro_batch数量 [默认: 1]"
    echo "  --collect-micro-batches NUM  收集的micro_batch数量 [默认: 1]"
    echo ""
    echo "位置参数:"
    echo "  MODE                    收集模式 (single|batch|quick)"
    echo "  QUANT_TYPE              量化类型 (bf16|mxfp8|mxfp4|hifp8)"
    echo ""
    echo "使用示例:"
    echo "  # 基本用法"
    echo "  $0 single mxfp8"
    echo ""
    echo "  # 使用命令行参数"
    echo "  $0 --mode single --quant-type mxfp8 --control-iter 3"
    echo ""
    echo "  # 批量收集所有类型"
    echo "  $0 batch"
    echo ""
    echo "  # 快速收集（收集少量数据用于测试）"
    echo "  $0 quick hifp8"
    echo ""
    echo "  # 自定义路径和iteration数量"
    echo "  $0 --mode single --quant-type mxfp4 --tensor-path ./my_tensors --control-iter 3"
    echo ""
    echo "  # 收集多个micro_batch的数据"
    echo "  $0 --mode single --quant-type mxfp8 --control-iter 5"
    echo ""
    echo "  # 自定义收集的micro_batch数量"
    echo "  $0 --mode single --quant-type mxfp8 --collect-micro-batches 2 --control-iter 3"
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
        --quant-type)
            QUANT_TYPE="$2"
            shift 2
            ;;
        --tensor-path)
            BASE_TENSOR_PATH="$2"
            shift 2
            ;;
        --tokenizer-path)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --control-iter|--control-micro-batches)
            CONTROL_ITER="$2"
            shift 2
            ;;
        --collect-micro-batches)
            COLLECT_MICRO_BATCHES="$2"
            shift 2
            ;;
        single|batch|quick)
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

# 支持的量化类型
VALID_QUANT_TYPES=("bf16" "mxfp8" "mxfp4" "hifp8")

# 验证模式
if [[ ! "$MODE" =~ ^(single|batch|quick)$ ]]; then
    echo "错误: 不支持的模式 '$MODE'"
    echo "支持的模式: single, batch, quick"
    show_help
    exit 1
fi

# 验证量化类型
if [[ ! " ${VALID_QUANT_TYPES[@]} " =~ " ${QUANT_TYPE} " ]]; then
    echo "错误: 不支持的量化类型 '$QUANT_TYPE'"
    echo "支持的量化类型: ${VALID_QUANT_TYPES[*]}"
    exit 1
fi

# 验证control_iter参数
if ! [[ "$CONTROL_ITER" =~ ^[0-9]+$ ]] || [ "$CONTROL_ITER" -lt 0 ]; then
    echo "错误: control_iter 必须是大于等于0的整数"
    echo "当前值: $CONTROL_ITER"
    exit 1
fi

# 验证collect_micro_batches参数
if ! [[ "$COLLECT_MICRO_BATCHES" =~ ^[0-9]+$ ]] || [ "$COLLECT_MICRO_BATCHES" -lt 1 ]; then
    echo "错误: collect_micro_batches 必须是大于0的整数"
    echo "当前值: $COLLECT_MICRO_BATCHES"
    exit 1
fi

echo "配置信息:"
echo "  - 模式: $MODE"
echo "  - 量化类型: $QUANT_TYPE"
echo "  - Tensor保存路径: $BASE_TENSOR_PATH"
echo "  - 控制micro_batch数量: $CONTROL_ITER"
echo "  - 收集micro_batch数量: $COLLECT_MICRO_BATCHES"
echo "  - 分词器路径: $TOKENIZER_PATH"
echo "  - 数据路径: $DATA_PATH"
echo "  - 数据类型: $DTYPE"

# 检查必要文件
check_requirements() {
    echo ""
    echo "检查必要文件..."
    
    if [ ! -f "examples/llama/train_llama32_1b_h100_fp8.sh" ]; then
        echo "错误: 训练脚本不存在: examples/llama/train_llama32_1b_h100_fp8.sh"
        exit 1
    fi
    
    echo "✅ 必要文件检查完成"
}

# 修改量化类型
modify_quant_type() {
    local quant_type=$1
    echo ""
    echo "修改量化类型为: $quant_type"
    
    # 修改linear层量化类型
    if [ -f "megatron/core/tensor_parallel/layers.py" ]; then
        sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'$quant_type'/" \
            megatron/core/tensor_parallel/layers.py
        echo "  ✅ 已修改 linear 层量化类型"
    fi
    
    # 修改attention层量化类型
    if [ -f "megatron/core/transformer/dot_product_attention.py" ]; then
        sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'$quant_type'/" \
            megatron/core/transformer/dot_product_attention.py
        echo "  ✅ 已修改 attention 层量化类型"
    fi
}

# 收集单个量化类型的tensor
collect_single_quant_type() {
    local quant_type=$1
    local tensor_path="$BASE_TENSOR_PATH/${quant_type}"
    local max_wait=300  # 最大等待时间（秒）
    local control_iter=$CONTROL_ITER  # 控制收集的micro_batch数量
    local collect_micro_batches=$COLLECT_MICRO_BATCHES  # 收集的micro_batch数量
    
    echo ""
    echo "=================================================================================="
    echo "收集 $quant_type 量化类型的tensor"
    echo "=================================================================================="
    
# 设置环境变量
export TENSOR_SAVE_ENABLED="true"
export TENSOR_SAVE_DIR="$tensor_path"
export HOST_TENSORBOARD_LOGS_PATH="tensorboard_logs/${quant_type}"
export CONTROL_ITER="$control_iter"
export COLLECT_MICRO_BATCHES="$collect_micro_batches"
    
    # 创建目录
    mkdir -p "$tensor_path"
    mkdir -p "tensorboard_logs/${quant_type}"
    mkdir -p "checkpoints/llama32_1b/${quant_type}"
    
    # 修改量化类型
    modify_quant_type "$quant_type"
    
    # 设置检查点路径
    local checkpoint_path="checkpoints/llama32_1b/${quant_type}"
    
    echo "开始训练并收集tensor..."
    echo "  - Tensor保存路径: $tensor_path"
    echo "  - 检查点路径: $checkpoint_path"
    echo "  - TensorBoard路径: $HOST_TENSORBOARD_LOGS_PATH"
    echo "  - 控制micro_batch数量: $control_iter"
    
    # 运行训练脚本
    local log_file="training_${quant_type}_$(date +'%y-%m-%d_%H-%M-%S').log"
    bash examples/llama/train_llama32_1b_h100_fp8.sh \
        "$checkpoint_path" \
        "$HOST_TENSORBOARD_LOGS_PATH" \
        "$TOKENIZER_PATH" \
        "$DATA_PATH" \
        "$DTYPE" \
        --control-iter "$control_iter" \
        --collect-micro-batches "$collect_micro_batches" \
        2>&1 | tee "$log_file" 
    
    # 最终统计
    local final_count=$(find "$tensor_path" -name "*.pt" 2>/dev/null | wc -l)
    echo ""
    echo "✅ $quant_type 量化类型tensor收集完成"
    echo "  - 最终收集到: $final_count 个tensor文件"
    
    # 显示统计信息
    if [ $final_count -gt 0 ]; then
        echo "  - 文件类型统计:"
        local pre_count=$(find "$tensor_path" -name "*_pre_*" 2>/dev/null | wc -l)
        local post_count=$(find "$tensor_path" -name "*_post_*" 2>/dev/null | wc -l)
        local fa_count=$(find "$tensor_path" -name "*_FA_*" 2>/dev/null | wc -l)
        local linear_count=$(find "$tensor_path" -name "*_linear_*" 2>/dev/null | wc -l)
        local forward_count=$(find "$tensor_path" -name "*_forward_*" 2>/dev/null | wc -l)
        local backward_count=$(find "$tensor_path" -name "*_backward_*" 2>/dev/null | wc -l)
        
        echo "    * Pre阶段: $pre_count, Post阶段: $post_count"
        echo "    * FA组件: $fa_count, Linear组件: $linear_count"
        echo "    * Forward: $forward_count, Backward: $backward_count"
        
        echo "  - 部分tensor文件:"
        find "$tensor_path" -name "*.pt" | head -3 | while read file; do
            echo "    * $(basename "$file")"
        done
    fi
    
    return $final_count
}

# 批量收集所有量化类型
collect_batch() {
    echo ""
    echo "=================================================================================="
    echo "批量收集所有量化类型的tensor"
    echo "=================================================================================="
    
    local total_tensors=0
    local success_count=0
    
    for quant_type in "${VALID_QUANT_TYPES[@]}"; do
        collect_single_quant_type "$quant_type"
        local result=$?
        
        if [ $result -gt 0 ]; then
            success_count=$((success_count + 1))
            total_tensors=$((total_tensors + result))
        fi
        
        # 在每次运行之间稍作休息
        if [ "$quant_type" != "${VALID_QUANT_TYPES[-1]}" ]; then
            echo ""
            echo "等待5秒后继续下一个量化类型..."
            sleep 5
        fi
    done
    
    echo ""
    echo "=================================================================================="
    echo "批量收集完成"
    echo "=================================================================================="
    echo "成功收集: $success_count/${#VALID_QUANT_TYPES[@]} 个量化类型"
    echo "总计tensor文件: $total_tensors 个"
    echo "保存位置: $BASE_TENSOR_PATH"
}

# 快速收集（用于测试）
collect_quick() {
    echo ""
    echo "=================================================================================="
    echo "快速收集模式（用于测试）"
    echo "=================================================================================="
    
    # 设置较短的等待时间
    local original_max_wait=$max_wait
    max_wait=60  # 快速模式只等待60秒
    
    collect_single_quant_type "$QUANT_TYPE"
    local result=$?
    
    max_wait=$original_max_wait
    
    echo ""
    echo "快速收集完成，收集到 $result 个tensor文件"
}

# 显示结果总结
show_summary() {
    echo ""
    echo "=================================================================================="
    echo "Tensor收集结果总结"
    echo "=================================================================================="
    
    local total_tensors=0
    local quant_types_found=()
    
    for quant_type in "${VALID_QUANT_TYPES[@]}"; do
        local tensor_path="$BASE_TENSOR_PATH/${quant_type}"
        if [ -d "$tensor_path" ]; then
            local count=$(find "$tensor_path" -name "*.pt" 2>/dev/null | wc -l)
            if [ $count -gt 0 ]; then
                total_tensors=$((total_tensors + count))
                quant_types_found+=("$quant_type")
                echo "  - $quant_type: $count 个tensor文件"
            fi
        fi
    done
    
    echo ""
    echo "总计收集到: $total_tensors 个tensor文件"
    echo "成功收集的量化类型: ${quant_types_found[*]}"
    echo "保存位置: $BASE_TENSOR_PATH"
    
    if [ $total_tensors -gt 0 ]; then
        echo ""
        echo "下一步操作建议:"
        echo "1. 查看收集到的tensor文件:"
        echo "   ls -la $BASE_TENSOR_PATH/*/"
        echo ""
        echo "2. 使用统一可视化脚本分析:"
        echo "   ./run_tensor_draw.sh $BASE_TENSOR_PATH"
        echo ""
        echo "3. 分析特定量化类型:"
        for quant_type in "${quant_types_found[@]}"; do
            echo "   ls -la $BASE_TENSOR_PATH/$quant_type/"
        done
    fi
}

# 主执行流程
main() {
    check_requirements
    
    case "$MODE" in
        "single")
            collect_single_quant_type "$QUANT_TYPE"
            ;;
        "batch")
            collect_batch
            ;;
        "quick")
            collect_quick
            ;;
    esac
    
    show_summary
    
    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo ""
    echo "=================================================================================="
    echo "Tensor收集完成"
    echo "Start time: $START_TIME"
    echo "End time: $END_TIME"
    echo "=================================================================================="
}

# 执行主函数 
main
