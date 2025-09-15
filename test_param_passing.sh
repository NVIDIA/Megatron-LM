#!/bin/bash

echo "=================================================================================="
echo "测试参数传递：从shell脚本到Python代码"
echo "=================================================================================="

echo "1. 测试shell脚本参数解析:"
echo "   ./run_tensor_collection.sh --mode single --quant-type mxfp8 --collect-micro-batches 3"
echo ""

# 模拟参数解析
MODE="single"
QUANT_TYPE="mxfp8"
COLLECT_MICRO_BATCHES=3

echo "解析结果:"
echo "  - MODE: $MODE"
echo "  - QUANT_TYPE: $QUANT_TYPE"
echo "  - COLLECT_MICRO_BATCHES: $COLLECT_MICRO_BATCHES"
echo ""

echo "2. 测试环境变量设置:"
export COLLECT_MICRO_BATCHES="$COLLECT_MICRO_BATCHES"
echo "  - COLLECT_MICRO_BATCHES: $COLLECT_MICRO_BATCHES"
echo ""

echo "3. 测试训练脚本参数传递:"
echo "   训练脚本会接收: --collect-micro-batches $COLLECT_MICRO_BATCHES"
echo ""

echo "4. 测试Python代码参数获取:"
echo "   在tensor_saver.py中:"
echo "   - 优先从环境变量获取: os.environ.get('COLLECT_MICRO_BATCHES', '1')"
echo "   - 然后从命令行参数获取: getattr(args, 'collect_micro_batches', 1)"
echo ""

echo "5. 验证参数定义:"
echo "   在arguments.py中:"
grep -n "collect-micro-batches" megatron/training/arguments.py
echo ""

echo "6. 验证参数使用:"
echo "   在tensor_saver.py中:"
grep -n "collect_micro_batches" megatron/core/tensor_saver.py
echo ""

echo "7. 验证pipeline控制:"
echo "   在schedules.py中:"
grep -n "increment_micro_batch" megatron/core/pipeline_parallel/schedules.py | wc -l
echo "   个位置添加了micro_batch控制"
echo ""

echo "=================================================================================="
echo "参数传递链路:"
echo "shell脚本 -> 环境变量 -> 训练脚本 -> pretrain_gpt.py -> arguments.py -> tensor_saver.py -> pipeline控制"
echo "=================================================================================="
