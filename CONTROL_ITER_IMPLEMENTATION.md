# Control Iter Parameter Implementation

## 概述

本次实现为Megatron-LM添加了 `--control-iter` 参数，用于控制在tensor收集过程中收集多少个iteration的数据后停止收集。

## 修改的文件

### 1. `megatron/training/arguments.py`
- 添加了 `--control-iter` 参数定义
- 默认值为1，表示收集1个iteration后停止

### 2. `megatron/core/tensor_saver.py`
- 修改了 `TensorSaver` 类的构造函数，添加了 `control_iter` 参数
- 在 `save_tensor` 方法中添加了检查逻辑，当当前iteration达到 `control_iter` 时停止收集
- 更新了 `get_tensor_saver` 函数以从命令行参数或环境变量获取 `control_iter` 值

### 3. `run_wikipedia_tensor_collection.sh`
- 添加了命令行参数解析功能
- 添加了 `--control-iter` 参数支持
- 更新了配置显示，显示控制iteration数量
- 修改了监控逻辑，根据 `control_iter` 值调整等待的tensor数量
- 在调用训练脚本时传递 `--control-iter` 参数

### 4. `examples/llama/train_llama32_1b_h100_fp8.sh`
- 添加了额外参数解析功能
- 支持 `--control-iter`、`--save-tensors`、`--tensor-save-dir` 等参数
- 将这些参数传递给 `pretrain_gpt.py` 脚本

## 使用方法

### 基本用法
```bash
# 使用默认值（收集1个iteration后停止）
bash run_wikipedia_tensor_collection.sh

# 收集2个iteration后停止
bash run_wikipedia_tensor_collection.sh --control-iter 2

# 收集5个iteration后停止
bash run_wikipedia_tensor_collection.sh --control-iter 5

# 查看帮助信息
bash run_wikipedia_tensor_collection.sh --help
```

### 直接使用训练脚本
```bash
bash examples/llama/train_llama32_1b_h100_fp8.sh \
    checkpoint_path \
    tensorboard_path \
    tokenizer_path \
    data_path \
    dtype \
    --control-iter 3 \
    --save-tensors \
    --tensor-save-dir ./tensor_logs
```

## 工作原理

1. **参数传递**: `--control-iter` 参数通过命令行传递到训练脚本
2. **环境变量**: 参数值也会设置到环境变量 `CONTROL_ITER` 中
3. **Tensor收集控制**: `TensorSaver` 类检查当前iteration是否达到 `control_iter` 值
4. **停止收集**: 当达到指定iteration数量时，`save_tensor` 方法返回 `None`，停止保存tensor
5. **监控逻辑**: 脚本根据 `control_iter` 值计算需要等待的tensor数量

## 技术细节

### 参数检查逻辑
```python
# 检查是否已经达到控制的iteration数量
if self.current_iteration >= self.control_iter:
    return None
```

### 环境变量支持
- `CONTROL_ITER`: 控制iteration数量
- `TENSOR_SAVER_ITERATION`: 当前iteration计数
- `TENSOR_SAVE_ENABLED`: 是否启用tensor保存
- `TENSOR_SAVE_DIR`: tensor保存目录

### 监控逻辑
```bash
# 检查是否收集了足够的iteration数据
required_tensors=$((CONTROL_ITER * 10))
if [ $tensor_count -ge $required_tensors ]; then
    iteration_collected=true
fi
```

## 测试

运行测试脚本验证功能：
```bash
bash test_control_iter.sh
```

## 注意事项

1. 默认值为1，意味着默认只收集1个iteration的数据
2. 参数值必须为正整数
3. 当达到指定iteration数量时，tensor收集会立即停止
4. 监控逻辑假设每个iteration至少生成10个tensor文件
5. 支持通过环境变量 `CONTROL_ITER` 设置参数值

## 兼容性

- 向后兼容：不指定 `--control-iter` 参数时使用默认值1
- 支持所有现有的tensor收集功能
- 不影响其他训练参数和功能
