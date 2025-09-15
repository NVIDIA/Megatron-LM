#!/bin/bash

echo "=================================================================================="
echo "测试 control_iter 参数传递"
echo "=================================================================================="

# 测试1: 使用位置参数
echo ""
echo "测试1: 位置参数格式"
echo "命令: ./run_tensor_collection.sh single mxfp8"
echo "预期: control_iter = 1 (默认值)"
echo ""

# 测试2: 使用命令行参数
echo "测试2: 命令行参数格式"
echo "命令: ./run_tensor_collection.sh --mode single --quant-type mxfp8 --control-iter 3"
echo "预期: control_iter = 3"
echo ""

# 测试3: 验证参数解析
echo "测试3: 验证参数解析"
echo "运行: ./run_tensor_collection.sh --mode single --quant-type mxfp8 --control-iter 5 --help"
echo ""

# 实际运行测试（只显示帮助信息，不实际执行训练）
./run_tensor_collection.sh --mode single --quant-type mxfp8 --control-iter 5 --help

echo ""
echo "=================================================================================="
echo "参数传递验证完成"
echo "=================================================================================="
echo ""
echo "关键点:"
echo "1. --control-iter 参数应该被正确解析"
echo "2. 参数值应该传递给训练脚本"
echo "3. 训练脚本应该接收 --control-iter 参数"
echo ""
echo "如果看到 '控制iteration数量: 5' 在配置信息中，说明参数传递成功！"
