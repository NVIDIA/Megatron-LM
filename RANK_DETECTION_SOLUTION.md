# Rank检测问题解决方案

## 问题描述
在 `tensor_parallel/layers.py` 中的 `save_tensor` 调用时，`rank=None` 导致无法正确获取GPU卡的信息。

## 解决方案

### 1. 直接获取Rank信息
在每个 `save_tensor` 调用点直接获取rank信息，而不是依赖全局状态管理：

```python
# 尝试获取rank信息
rank = None
try:
    import torch.distributed as dist
    if dist.is_initialized():
        rank = dist.get_rank()
except:
    pass

if rank is None:
    rank = int(os.environ.get("LOCAL_RANK", 0))

# 尝试获取sample信息
sample_idx = int(os.environ.get("CURRENT_SAMPLE_IDX", 0))

save_tensor(
    tensor=output,
    layer_type="linear",
    operation="forward",
    quant_type=custom_quant_type,
    tensor_name="output",
    layer_idx=getattr(ctx, 'layer_idx', None),
    phase="post",
    component="linear",
    rank=rank,  # 直接获取
    sample_idx=sample_idx,  # 直接获取
    metadata={...}
)
```

### 2. 多重检测机制
实现了多层次的rank检测机制：

1. **分布式环境检测**: 优先从 `torch.distributed.get_rank()` 获取
2. **环境变量检测**: 从 `LOCAL_RANK` 或 `RANK` 环境变量获取
3. **Tensor设备检测**: 从tensor的设备信息推断rank
4. **默认值**: 如果都无法获取，使用默认值0

### 3. 环境变量设置
在训练脚本中正确设置环境变量：

```bash
export LOCAL_RANK=0
export CURRENT_SAMPLE_IDX=0
export TENSOR_SAVER_ITERATION=0
```

### 4. 调试信息
添加了详细的调试信息，帮助诊断rank获取问题：

```python
print(f"[TensorSaver] 保存tensor - Rank: {rank}, Sample: {sample_idx}, Layer: {layer_type}, Operation: {operation}")
```

## 修改的文件

### 1. `megatron/core/tensor_parallel/layers.py`
- Forward和backward的 `save_tensor` 调用都添加了直接rank获取逻辑
- 优先从 `torch.distributed` 获取，失败则从环境变量获取

### 2. `megatron/core/transformer/dot_product_attention.py`
- Attention层的 `save_tensor` 调用添加了相同的rank获取逻辑

### 3. `megatron/core/tensor_saver.py`
- 增强了 `get_current_rank_and_sample` 函数
- 添加了从tensor设备信息推断rank的功能
- 改进了调试信息输出

### 4. `run_wikipedia_tensor_collection.sh`
- 添加了初始化脚本，在训练开始前设置rank信息
- 确保环境变量正确设置

## 使用方法

### 方法1: 使用环境变量（推荐）
```bash
# 设置环境变量
export LOCAL_RANK=0
export CURRENT_SAMPLE_IDX=0
export TENSOR_SAVER_ITERATION=0

# 运行训练
bash run_wikipedia_tensor_collection.sh
```

### 方法2: 在代码中直接设置
```python
import os
os.environ["LOCAL_RANK"] = "0"
os.environ["CURRENT_SAMPLE_IDX"] = "0"
os.environ["TENSOR_SAVER_ITERATION"] = "0"
```

### 方法3: 在分布式训练中
```python
import torch.distributed as dist
if dist.is_initialized():
    rank = dist.get_rank()
    # 系统会自动使用这个rank
```

## 验证方法

使用提供的测试脚本验证rank检测功能：

```bash
python test_rank_detection.py
```

## 文件命名结果

使用新的rank检测机制后，保存的文件名格式为：
```
timestamp_counter_iter{iteration}_layer_type_L{layer_idx}_operation_phase_component_quant_type_rank{rank}_sample{sample_idx}_tensor_name.pt
```

例如：
```
20241201_143022_0001_iter000_linear_L0_forward_post_linear_bf16_rank0_sample0_output.pt
```

## 优势

1. **可靠性**: 多重检测机制确保能够获取到rank信息
2. **兼容性**: 支持多种环境（分布式、单机、环境变量）
3. **调试性**: 详细的日志信息帮助诊断问题
4. **向后兼容**: 不影响现有的代码结构
5. **灵活性**: 支持手动指定rank和sample信息

## 注意事项

1. 确保在训练开始前正确设置环境变量
2. 在多GPU环境中，每个进程会自动获取自己的rank
3. 如果无法获取rank信息，系统会使用默认值0并打印警告
4. 建议在生产环境中使用环境变量方式设置rank信息
