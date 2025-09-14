#!/bin/bash

# 测试iteration检测逻辑

echo "测试iteration检测逻辑..."

# 创建测试目录
test_dir="./test_tensor_logs"
mkdir -p "$test_dir"

# 清理之前的测试文件
rm -f "$test_dir"/*.pt

echo "1. 创建模拟tensor文件..."

# 创建iteration 0的文件
touch "$test_dir/20250914_100000_0001_iter000_attention_L0_forward_pre_FA_bf16_rank00_sample000_group000_query.pt"
touch "$test_dir/20250914_100000_0002_iter000_attention_L0_forward_pre_FA_bf16_rank00_sample000_group000_key.pt"
touch "$test_dir/20250914_100000_0003_iter000_attention_L0_forward_pre_FA_bf16_rank00_sample000_group000_value.pt"
touch "$test_dir/20250914_100000_0004_iter000_linear_L1_forward_pre_linear_bf16_rank00_sample000_group000_input.pt"

echo "2. 测试iteration 0的检测..."
tensor_count=$(find "$test_dir" -name "*.pt" 2>/dev/null | wc -l)
collected_iterations=$(find "$test_dir" -name "*.pt" 2>/dev/null | grep -o "iter[0-9][0-9][0-9]" | sort -u | wc -l)
echo "   - 总文件数: $tensor_count"
echo "   - 不同iteration数: $collected_iterations"

# 创建iteration 1的文件
touch "$test_dir/20250914_100100_0005_iter001_attention_L0_forward_pre_FA_bf16_rank00_sample000_group000_query.pt"
touch "$test_dir/20250914_100100_0006_iter001_attention_L0_forward_pre_FA_bf16_rank00_sample000_group000_key.pt"
touch "$test_dir/20250914_100100_0007_iter001_linear_L1_forward_pre_linear_bf16_rank00_sample000_group000_input.pt"

echo "3. 测试iteration 0+1的检测..."
tensor_count=$(find "$test_dir" -name "*.pt" 2>/dev/null | wc -l)
collected_iterations=$(find "$test_dir" -name "*.pt" 2>/dev/null | grep -o "iter[0-9][0-9][0-9]" | sort -u | wc -l)
echo "   - 总文件数: $tensor_count"
echo "   - 不同iteration数: $collected_iterations"

# 创建iteration 2的文件
touch "$test_dir/20250914_100200_0008_iter002_attention_L0_forward_pre_FA_bf16_rank00_sample000_group000_query.pt"
touch "$test_dir/20250914_100200_0009_iter002_linear_L1_forward_pre_linear_bf16_rank00_sample000_group000_input.pt"

echo "4. 测试iteration 0+1+2的检测..."
tensor_count=$(find "$test_dir" -name "*.pt" 2>/dev/null | wc -l)
collected_iterations=$(find "$test_dir" -name "*.pt" 2>/dev/null | grep -o "iter[0-9][0-9][0-9]" | sort -u | wc -l)
echo "   - 总文件数: $tensor_count"
echo "   - 不同iteration数: $collected_iterations"

echo "5. 测试控制逻辑..."
CONTROL_ITER=1
if [ $collected_iterations -ge $CONTROL_ITER ]; then
    echo "   - 达到控制iteration数量 $CONTROL_ITER，应该停止"
else
    echo "   - 未达到控制iteration数量 $CONTROL_ITER，继续等待"
fi

CONTROL_ITER=2
if [ $collected_iterations -ge $CONTROL_ITER ]; then
    echo "   - 达到控制iteration数量 $CONTROL_ITER，应该停止"
else
    echo "   - 未达到控制iteration数量 $CONTROL_ITER，继续等待"
fi

echo "6. 显示所有检测到的iteration:"
find "$test_dir" -name "*.pt" 2>/dev/null | grep -o "iter[0-9][0-9][0-9]" | sort -u

# 清理测试文件
rm -rf "$test_dir"

echo "测试完成！"
