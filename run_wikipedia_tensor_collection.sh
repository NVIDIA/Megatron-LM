#!/bin/bash

# =============================================================================
# Wikipedia Tensor Collection Script
# 收集wikipedia数据集在不同量化类型下的tensor
# 量化类型: mxfp8, mxfp4, hifp8
# =============================================================================

# 设置脚本元数据
SCRIPT_NAME="$(basename "$0")"
SCRIPT_VERSION="1.0.0"
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
START_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

echo "=================================================================================="
echo "Wikipedia Tensor Collection Script"
echo "Script: $SCRIPT_NAME"
echo "Version: $SCRIPT_VERSION"
echo "Start Time: $START_TIME"
echo "=================================================================================="

# =============================================================================
# 配置参数
# =============================================================================

# 基础路径配置
BASE_CHECKPOINT_PATH="checkpoints/llama32_1b"
BASE_TENSORBOARD_PATH="tensorboard_logs/llama32_1b"
BASE_TENSOR_PATH="enhanced_tensor_logs"
TOKENIZER_PATH="model/llama3.2-1b"
DATA_PATH="dataset/wikipedia_processed/wikipedia_processed_text_document"
DTYPE="bf16"

# 要测试的量化类型
QUANT_TYPES=("bf16" "mxfp8" "mxfp4" "hifp8")

# 训练步数（用于快速收集tensor，不需要完整训练）
TRAINING_STEPS=10

# 控制iteration数量（默认收集1个iteration后停止）
CONTROL_ITER=1

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --control-iter)
            CONTROL_ITER="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--control-iter N]"
            echo "  --control-iter N: 控制收集的iteration数量，达到后停止收集 (默认: 1)"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 基础配置:"
echo "  - 检查点路径: $BASE_CHECKPOINT_PATH"
echo "  - TensorBoard路径: $BASE_TENSORBOARD_PATH"
echo "  - Tensor保存路径: $BASE_TENSOR_PATH"
echo "  - 分词器路径: $TOKENIZER_PATH"
echo "  - 数据路径: $DATA_PATH"
echo "  - 数据类型: $DTYPE"
echo "  - 训练步数: $TRAINING_STEPS"
echo "  - 控制iteration数量: $CONTROL_ITER"
echo "  - 量化类型: ${QUANT_TYPES[*]}"

# =============================================================================
# 环境设置
# =============================================================================

# 设置tensor保存环境变量
export TENSOR_SAVE_ENABLED="true"
export TENSOR_SAVE_DIR="$BASE_TENSOR_PATH"

# 创建必要的目录
mkdir -p "$BASE_CHECKPOINT_PATH"
mkdir -p "$BASE_TENSORBOARD_PATH"
mkdir -p "$BASE_TENSOR_PATH"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 环境设置完成"
echo "  - TENSOR_SAVE_ENABLED: $TENSOR_SAVE_ENABLED"
echo "  - TENSOR_SAVE_DIR: $TENSOR_SAVE_DIR"

# =============================================================================
# 函数定义
# =============================================================================

# 修改量化类型的函数
modify_quant_type() {
    local quant_type=$1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 修改量化类型为: $quant_type"
    
    # 修改linear层量化类型
    sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'$quant_type'/" \
        megatron/core/tensor_parallel/layers.py
    
    # 修改attention层量化类型
    sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'$quant_type'/" \
        megatron/core/transformer/dot_product_attention.py
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] 量化类型修改完成: $quant_type"
}

# 运行训练收集tensor的函数
run_training_for_tensor_collection() {
    local quant_type=$1
    local checkpoint_path="$BASE_CHECKPOINT_PATH/pretrain_llama32-1b_wikipedia_${quant_type}"
    local tensorboard_path="$BASE_TENSORBOARD_PATH/${quant_type}"
    local tensor_path="$BASE_TENSOR_PATH/${quant_type}"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 开始收集 $quant_type 量化类型的tensor"
    echo "  - 检查点路径: $checkpoint_path"
    echo "  - TensorBoard路径: $tensorboard_path"
    echo "  - Tensor保存路径: $tensor_path"
    
    # 设置当前量化类型的tensor保存路径
    export TENSOR_SAVE_DIR="$tensor_path"
    export HOST_TENSORBOARD_LOGS_PATH="$tensorboard_path"
    
    # 创建目录
    mkdir -p "$(dirname "$checkpoint_path")"
    mkdir -p "$tensorboard_path"
    mkdir -p "$tensor_path"
    
    # 修改量化类型
    modify_quant_type "$quant_type"
    
    # 设置tensor保存器的iteration和sample信息
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 设置tensor保存器iteration为0..."
    export TENSOR_SAVER_ITERATION=0
    export CURRENT_SAMPLE_IDX=0
    export LOCAL_RANK=0  # 设置默认rank
    export CONTROL_ITER=$CONTROL_ITER  # 设置控制iteration数量
    
    # 创建初始化脚本
    cat > "${tensor_path}/init_tensor_collection.py" << 'EOF'
#!/usr/bin/env python3
"""初始化tensor收集状态"""
import sys
import os
sys.path.append('/data/charles/codes/Megatron-LM')

# 尝试导入torch.distributed来获取rank
try:
    import torch.distributed as dist
    if dist.is_initialized():
        rank = dist.get_rank()
        print(f"从torch.distributed获取rank: {rank}")
    else:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"从环境变量获取rank: {rank}")
except Exception as e:
    rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"无法从torch.distributed获取rank，使用环境变量: {rank}, 错误: {e}")

sample_idx = int(os.environ.get("CURRENT_SAMPLE_IDX", 0))
iteration = int(os.environ.get("TENSOR_SAVER_ITERATION", 0))

# 导入并初始化tensor收集状态
from megatron.core.tensor_saver import initialize_tensor_collection
initialize_tensor_collection(rank=rank, sample_idx=sample_idx, iteration=iteration)
print(f"Tensor收集状态初始化完成: rank={rank}, sample_idx={sample_idx}, iteration={iteration}")
EOF
    
    chmod +x "${tensor_path}/init_tensor_collection.py"
    
    # 在训练开始前运行初始化脚本
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 初始化tensor收集状态..."
    python "${tensor_path}/init_tensor_collection.py"
    
    # 运行训练脚本（限制步数）
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 执行训练脚本..."
    
    # 使用训练脚本，control_iter会自动控制训练结束
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 执行训练脚本，将收集 $CONTROL_ITER 个iteration的数据..."
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 训练将在完成 $CONTROL_ITER 个iteration后自动结束"
    
    bash examples/llama/train_llama32_1b_h100_fp8.sh \
        "$checkpoint_path" \
        "$tensorboard_path" \
        "$TOKENIZER_PATH" \
        "$DATA_PATH" \
        "$DTYPE" \
        --control-iter "$CONTROL_ITER" \
        --save-tensors \
        --tensor-save-dir "$tensor_path" \
        2>&1 | tee "${tensorboard_path}/training_${quant_type}_$(date +'%y-%m-%d_%H-%M-%S').log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] 训练已完成，control_iter自动控制结束"
    
    # 统计收集到的tensor
    final_tensor_count=$(find "$tensor_path" -name "*.pt" 2>/dev/null | wc -l)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $quant_type 量化类型tensor收集完成"
    echo "  - 最终收集到 $final_tensor_count 个tensor文件"
    echo "  - 保存路径: $tensor_path"
    
    # 显示收集到的tensor文件信息
    if [ $final_tensor_count -gt 0 ]; then
        echo "  - 收集到的tensor文件:"
        find "$tensor_path" -name "*.pt" | head -5 | while read file; do
            echo "    * $(basename "$file")"
        done
        if [ $final_tensor_count -gt 5 ]; then
            echo "    * ... 还有 $((final_tensor_count - 5)) 个文件"
        fi
    fi
    
    echo ""
}

# 分析收集到的tensor的函数
analyze_collected_tensors() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 分析收集到的tensor..."
    
    for quant_type in "${QUANT_TYPES[@]}"; do
        local tensor_path="$BASE_TENSOR_PATH/${quant_type}"
        
        if [ -d "$tensor_path" ]; then
            local tensor_count=$(find "$tensor_path" -name "*.pt" 2>/dev/null | wc -l)
            echo "  - $quant_type: $tensor_count 个tensor文件"
            
            if [ $tensor_count -gt 0 ]; then
                # 统计不同类型的tensor
                local pre_count=$(find "$tensor_path" -name "*_pre_*" 2>/dev/null | wc -l)
                local post_count=$(find "$tensor_path" -name "*_post_*" 2>/dev/null | wc -l)
                local fa_count=$(find "$tensor_path" -name "*_FA_*" 2>/dev/null | wc -l)
                local linear_count=$(find "$tensor_path" -name "*_linear_*" 2>/dev/null | wc -l)
                
                echo "    * Pre阶段: $pre_count 个"
                echo "    * Post阶段: $post_count 个"
                echo "    * FA组件: $fa_count 个"
                echo "    * Linear组件: $linear_count 个"
            fi
        else
            echo "  - $quant_type: 未找到tensor文件"
        fi
    done
}

# =============================================================================
# 主执行流程
# =============================================================================

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 开始Wikipedia tensor收集流程..."

# 检查必要的文件是否存在
if [ ! -f "examples/llama/train_llama32_1b_h100_fp8.sh" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] 训练脚本不存在: examples/llama/train_llama32_1b_h100_fp8.sh"
    exit 1
fi

# 循环运行不同量化类型的训练
for quant_type in "${QUANT_TYPES[@]}"; do
    echo ""
    echo "=================================================================================="
    echo "处理量化类型: $quant_type"
    echo "=================================================================================="
    
    run_training_for_tensor_collection "$quant_type"
    
    # 在每次运行之间稍作休息
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 等待5秒后继续下一个量化类型..."
    sleep 5
done

# 分析收集到的tensor
echo ""
echo "=================================================================================="
echo "Tensor收集结果分析"
echo "=================================================================================="
analyze_collected_tensors

# =============================================================================
# 完成总结
# =============================================================================

echo ""
echo "=================================================================================="
echo "Wikipedia Tensor收集完成"
echo "=================================================================================="

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] 所有量化类型的tensor收集完成"
echo "  - 开始时间: $START_TIME"
echo "  - 结束时间: $END_TIME"
echo "  - 总耗时: $(($(date +%s) - $(date -d "$START_TIME" +%s))) 秒"

echo ""
echo "收集到的tensor文件位置:"
for quant_type in "${QUANT_TYPES[@]}"; do
    local tensor_path="$BASE_TENSOR_PATH/${quant_type}"
    if [ -d "$tensor_path" ]; then
        local tensor_count=$(find "$tensor_path" -name "*.pt" 2>/dev/null | wc -l)
        echo "  - $quant_type: $tensor_path ($tensor_count 个文件)"
    fi
done

