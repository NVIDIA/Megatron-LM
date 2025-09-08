#!/bin/bash

# =============================================================================
# 快速Tensor收集脚本
# 专门用于收集wikipedia数据集在不同量化类型下的tensor
# =============================================================================

echo "=================================================================================="
echo "快速Wikipedia Tensor收集脚本"
echo "量化类型: mxfp8, mxfp4, hifp8"
echo "=================================================================================="

# 配置参数
QUANT_TYPES=("mxfp8" "mxfp4" "hifp8")
BASE_TENSOR_PATH="enhanced_tensor_logs"
TOKENIZER_PATH="model/llama3.2-1b"
DATA_PATH="dataset/wikipedia_processed/wikipedia_processed_text_document"
DTYPE="bf16"

# 设置tensor保存环境
export TENSOR_SAVE_ENABLED="true"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] 开始快速tensor收集..."

# 检查必要文件
if [ ! -f "examples/llama/train_llama32_1b_h100_fp8.sh" ]; then
    echo "[ERROR] 训练脚本不存在: examples/llama/train_llama32_1b_h100_fp8.sh"
    exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
    echo "[ERROR] 数据路径不存在: $DATA_PATH"
    exit 1
fi

# 创建基础目录
mkdir -p "$BASE_TENSOR_PATH"

# 函数：修改量化类型
modify_quant_type() {
    local quant_type=$1
    echo "[INFO] 修改量化类型为: $quant_type"
    
    sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'$quant_type'/" \
        megatron/core/tensor_parallel/layers.py
    
    sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'$quant_type'/" \
        megatron/core/transformer/dot_product_attention.py
}

# 函数：快速收集tensor
quick_collect_tensors() {
    local quant_type=$1
    local tensor_path="$BASE_TENSOR_PATH/${quant_type}"
    
    echo ""
    echo "=================================================================================="
    echo "收集 $quant_type 量化类型的tensor"
    echo "=================================================================================="
    
    # 设置tensor保存路径
    export TENSOR_SAVE_DIR="$tensor_path"
    export HOST_TENSORBOARD_LOGS_PATH="tensorboard_logs/${quant_type}"
    
    # 创建目录
    mkdir -p "$tensor_path"
    mkdir -p "tensorboard_logs/${quant_type}"
    
    # 修改量化类型
    modify_quant_type "$quant_type"
    
    # 设置检查点路径
    local checkpoint_path="checkpoints/llama32_1b/${quant_type}"
    mkdir -p "$(dirname "$checkpoint_path")"
    
    echo "[INFO] 开始训练并收集tensor..."
    echo "  - Tensor保存路径: $tensor_path"
    echo "  - 检查点路径: $checkpoint_path"
    
    # 运行训练脚本（后台运行）
    bash examples/llama/train_llama32_1b_h100_fp8.sh \
        "$checkpoint_path" \
        "tensorboard_logs/${quant_type}" \
        "$TOKENIZER_PATH" \
        "$DATA_PATH" \
        "$DTYPE" \
        > "training_${quant_type}.log" 2>&1 &
    
    local training_pid=$!
    echo "[INFO] 训练进程ID: $training_pid"
    
    # 等待并监控tensor生成
    local max_wait=60  # 最大等待60秒
    local wait_time=0
    local tensor_count=0
    
    while [ $wait_time -lt $max_wait ]; do
        sleep 5
        wait_time=$((wait_time + 5))
        
        # 检查tensor文件数量
        if [ -d "$tensor_path" ]; then
            tensor_count=$(find "$tensor_path" -name "*.pt" 2>/dev/null | wc -l)
        fi
        
        echo "[INFO] 等待时间: ${wait_time}s, 已收集tensor: $tensor_count 个"
        
        # 如果收集到足够的tensor，可以提前停止
        if [ $tensor_count -ge 20 ]; then
            echo "[INFO] 已收集到足够的tensor ($tensor_count 个)，准备停止训练"
            break
        fi
    done
    
    # 停止训练进程
    echo "[INFO] 停止训练进程..."
    kill $training_pid 2>/dev/null
    wait $training_pid 2>/dev/null
    
    # 最终统计
    local final_count=$(find "$tensor_path" -name "*.pt" 2>/dev/null | wc -l)
    echo "[SUCCESS] $quant_type 量化类型tensor收集完成"
    echo "  - 最终收集到: $final_count 个tensor文件"
    
    # 显示部分文件
    if [ $final_count -gt 0 ]; then
        echo "  - 部分tensor文件:"
        find "$tensor_path" -name "*.pt" | head -3 | while read file; do
            echo "    * $(basename "$file")"
        done
    fi
}

# 主执行流程
for quant_type in "${QUANT_TYPES[@]}"; do
    quick_collect_tensors "$quant_type"
    
    # 在每次运行之间稍作休息
    echo "[INFO] 等待3秒后继续下一个量化类型..."
    sleep 3
done

# 最终分析
echo ""
echo "=================================================================================="
echo "Tensor收集结果总结"
echo "=================================================================================="

total_tensors=0
for quant_type in "${QUANT_TYPES[@]}"; do
    local tensor_path="$BASE_TENSOR_PATH/${quant_type}"
    if [ -d "$tensor_path" ]; then
        local count=$(find "$tensor_path" -name "*.pt" 2>/dev/null | wc -l)
        total_tensors=$((total_tensors + count))
        echo "  - $quant_type: $count 个tensor文件"
        
        # 统计不同类型
        local pre_count=$(find "$tensor_path" -name "*_pre_*" 2>/dev/null | wc -l)
        local post_count=$(find "$tensor_path" -name "*_post_*" 2>/dev/null | wc -l)
        local fa_count=$(find "$tensor_path" -name "*_FA_*" 2>/dev/null | wc -l)
        local linear_count=$(find "$tensor_path" -name "*_linear_*" 2>/dev/null | wc -l)
        
        echo "    * Pre阶段: $pre_count, Post阶段: $post_count"
        echo "    * FA组件: $fa_count, Linear组件: $linear_count"
    else
        echo "  - $quant_type: 未收集到tensor文件"
    fi
done

echo ""
echo "总计收集到: $total_tensors 个tensor文件"
echo "保存位置: $BASE_TENSOR_PATH"

echo ""
echo "=================================================================================="
echo "下一步操作建议"
echo "=================================================================================="
echo "1. 查看收集到的tensor文件:"
echo "   ls -la $BASE_TENSOR_PATH/*/"
echo ""
echo "2. 使用可视化脚本分析:"
echo "   python script/visualization/quick_visualize.py --tensor_dir $BASE_TENSOR_PATH"
echo ""
echo "3. 使用一键可视化:"
echo "   bash script/visualization/one_click_visualize.sh $BASE_TENSOR_PATH"
echo ""
echo "4. 分析特定量化类型:"
echo "   ls -la $BASE_TENSOR_PATH/mxfp8/"
echo "   ls -la $BASE_TENSOR_PATH/mxfp4/"
echo "   ls -la $BASE_TENSOR_PATH/hifp8/"

echo ""
echo "=================================================================================="
echo "快速Tensor收集完成"
echo "=================================================================================="
