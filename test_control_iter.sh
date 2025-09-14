#!/bin/bash

# 测试脚本：演示如何使用 --control-iter 参数
echo "=================================================================================="
echo "测试 --control-iter 参数功能"
echo "=================================================================================="

echo ""
echo "1. 测试默认行为（收集1个iteration后停止）:"
echo "   bash run_wikipedia_tensor_collection.sh"
echo ""

echo "2. 测试收集2个iteration后停止:"
echo "   bash run_wikipedia_tensor_collection.sh --control-iter 2"
echo ""

echo "3. 测试收集5个iteration后停止:"
echo "   bash run_wikipedia_tensor_collection.sh --control-iter 5"
echo ""

echo "4. 查看帮助信息:"
echo "   bash run_wikipedia_tensor_collection.sh --help"
echo ""

echo "=================================================================================="
echo "参数说明:"
echo "  --control-iter N: 控制收集的iteration数量，达到后停止收集"
echo "  - 默认值: 1"
echo "  - 作用: 在循环进行N个iteration后会停止tensor的收集"
echo "=================================================================================="

# 实际运行帮助命令来验证
echo ""
echo "实际运行帮助命令:"
bash run_wikipedia_tensor_collection.sh --help
