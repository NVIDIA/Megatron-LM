#!/bin/bash

# =============================================================================
# 测试 control_iter 参数功能
# =============================================================================

echo "=================================================================================="
echo "测试 control_iter 参数功能"
echo "=================================================================================="

# 测试不同的 control_iter 值
echo ""
echo "测试 1: control_iter = 1 (默认值)"
echo "命令: ./run_tensor_collection.sh single mxfp8"
echo "预期: 收集1个iteration的tensor数据"
echo ""

echo "测试 2: control_iter = 3 (使用命令行参数)"
echo "命令: ./run_tensor_collection.sh --mode single --quant-type mxfp8 --control-iter 3"
echo "预期: 收集3个iteration的tensor数据"
echo ""

echo "测试 3: control_iter = 5 (使用命令行参数)"
echo "命令: ./run_tensor_collection.sh --mode single --quant-type mxfp8 --control-iter 5"
echo "预期: 收集5个iteration的tensor数据"
echo ""

echo "测试 4: 通过 run_tensor_draw.sh 测试"
echo "命令: ./run_tensor_draw.sh --mode collect --quant-type mxfp8 --control-iter 2"
echo "预期: 收集2个iteration的tensor数据并进行可视化"
echo ""

echo "=================================================================================="
echo "参数说明"
echo "=================================================================================="
echo "control_iter 参数的作用:"
echo "  - 控制tensor收集的iteration数量"
echo "  - 设置为1表示只收集第1个iteration的数据"
echo "  - 设置为3表示收集前3个iteration的数据"
echo "  - 设置为0表示收集所有iteration的数据（不推荐）"
echo ""

echo "使用示例:"
echo "  # 收集1个iteration的数据（默认）"
echo "  ./run_tensor_collection.sh single mxfp8"
echo ""
echo "  # 收集3个iteration的数据（使用命令行参数）"
echo "  ./run_tensor_collection.sh --mode single --quant-type mxfp8 --control-iter 3"
echo ""
echo "  # 通过主脚本收集2个iteration的数据"
echo "  ./run_tensor_draw.sh --mode collect --quant-type mxfp8 --control-iter 2"
echo ""

echo "=================================================================================="
echo "注意事项"
echo "=================================================================================="
echo "1. control_iter 参数会通过环境变量 CONTROL_ITER 传递给训练脚本"
echo "2. 训练脚本中的 tensor_saver.py 会读取这个参数来控制收集行为"
echo "3. 建议根据实际需要设置合适的iteration数量，避免收集过多数据"
echo "4. 对于测试目的，建议使用较小的值（如1-3）"
echo "5. 对于生产环境，可以根据需要设置更大的值"
echo ""

echo "测试完成！"
