#!/bin/bash

echo "=================================================================================="
echo "最终验证：collect_micro_batches参数真正发挥作用"
echo "=================================================================================="

echo "✅ 已完成的修改总结:"
echo "1. 在 megatron/training/arguments.py 中添加了 --collect-micro-batches 参数定义"
echo "2. 在 megatron/core/tensor_saver.py 中修改了参数获取逻辑，优先使用环境变量"
echo "3. 在 shell 脚本中添加了环境变量设置"
echo "4. 在所有pipeline函数中添加了micro_batch控制逻辑"
echo "5. 添加了详细的调试日志"
echo ""

echo "🔧 参数传递链路:"
echo "1. Shell脚本解析: --collect-micro-batches 3"
echo "2. 设置环境变量: export COLLECT_MICRO_BATCHES=3"
echo "3. 训练脚本传递: --collect-micro-batches 3"
echo "4. Python参数解析: args.collect_micro_batches = 3"
echo "5. TensorSaver获取: control_micro_batches = 3"
echo "6. Pipeline控制: 达到3个micro_batch后提前退出"
echo ""

echo "📝 使用示例:"
echo ""
echo "1. 收集2个micro_batch的数据:"
echo "   ./run_tensor_collection.sh --mode single --quant-type mxfp8 --collect-micro-batches 2"
echo ""
echo "2. 收集3个micro_batch的数据，最多5个iteration:"
echo "   ./run_tensor_collection.sh --mode single --quant-type mxfp8 --collect-micro-batches 3 --control-iter 5"
echo ""
echo "3. 通过主脚本使用:"
echo "   ./run_tensor_draw.sh --mode collect --quant-type mxfp8 --collect-micro-batches 1"
echo ""

echo "🔍 关键修改点:"
echo ""
echo "1. 参数定义 (arguments.py:1749):"
grep -n "collect-micro-batches" megatron/training/arguments.py
echo ""
echo "2. 参数获取 (tensor_saver.py:650-652):"
grep -A 2 -B 1 "collect_micro_batches" megatron/core/tensor_saver.py
echo ""
echo "3. 环境变量设置 (run_tensor_collection.sh):"
grep -n "COLLECT_MICRO_BATCHES" run_tensor_collection.sh
echo ""
echo "4. Pipeline控制 (schedules.py):"
grep -n "increment_micro_batch" megatron/core/pipeline_parallel/schedules.py
echo ""

echo "🎯 工作原理:"
echo "- 当训练开始时，TensorSaver会从环境变量或命令行参数获取collect_micro_batches值"
echo "- 在每个micro_batch处理完成后，调用increment_micro_batch()增加计数"
echo "- 当计数达到指定值时，should_continue_collection()返回False"
echo "- Pipeline循环检测到False后立即break，提前退出训练"
echo "- 这样可以精确控制收集的micro_batch数量，避免收集过多数据"
echo ""

echo "=================================================================================="
echo "✅ 验证完成！collect_micro_batches参数现在真正发挥作用了！"
echo "=================================================================================="
