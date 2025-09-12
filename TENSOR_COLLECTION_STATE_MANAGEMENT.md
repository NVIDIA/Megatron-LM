# Tensor收集状态管理使用指南

## 概述

为了解决rank和sample信息获取的问题，我们实现了一个全局状态管理系统，可以自动跟踪和管理tensor收集过程中的各种状态信息。

## 核心组件

### 1. TensorCollectionState 类
全局状态管理器，负责跟踪：
- `current_rank`: 当前GPU rank
- `current_sample_idx`: 当前样本索引
- `current_iteration`: 当前迭代次数
- `batch_idx`: 当前batch索引
- `sequence_idx`: 当前sequence索引

### 2. 全局状态函数
```python
# 设置全局状态
set_global_rank(rank: int)
set_global_sample_idx(sample_idx: int)
set_global_iteration(iteration: int)
set_global_batch_idx(batch_idx: int)
set_global_sequence_idx(sequence_idx: int)

# 获取全局状态
get_current_rank_and_sample() -> tuple[Optional[int], Optional[int]]

# 初始化状态
initialize_tensor_collection(rank=None, sample_idx=None, iteration=0, batch_idx=0, sequence_idx=0)
```

## 使用方法

### 方法1: 自动初始化（推荐）
```python
from megatron.core.tensor_saver import get_tensor_saver

# 获取tensor_saver时会自动初始化状态
saver = get_tensor_saver()

# 保存tensor时自动获取rank和sample信息
saver.save_tensor(
    tensor=my_tensor,
    layer_type="linear",
    operation="forward",
    quant_type="bf16",
    tensor_name="output",
    layer_idx=0,
    phase="post",
    component="linear"
    # rank和sample_idx会自动获取
)
```

### 方法2: 手动初始化
```python
from megatron.core.tensor_saver import initialize_tensor_collection, get_tensor_saver

# 手动初始化状态
initialize_tensor_collection(rank=0, sample_idx=0, iteration=0)

# 获取tensor_saver
saver = get_tensor_saver()

# 保存tensor
saver.save_tensor(...)
```

### 方法3: 在训练循环中动态更新
```python
from megatron.core.tensor_saver import set_global_iteration, set_global_sample_idx, get_tensor_saver

saver = get_tensor_saver()

for iteration in range(num_iterations):
    set_global_iteration(iteration)
    
    for batch_idx in range(num_batches):
        set_global_sample_idx(batch_idx)
        
        # 保存tensor时会使用当前的iteration和sample_idx
        saver.save_tensor(...)
```

## 环境变量支持

系统会自动从以下环境变量获取信息：
- `LOCAL_RANK`: GPU rank信息
- `CURRENT_SAMPLE_IDX`: 样本索引
- `TENSOR_SAVER_ITERATION`: 迭代次数

## 自动检测机制

1. **Rank检测优先级**：
   - 手动设置的值
   - 分布式环境中的rank
   - 环境变量LOCAL_RANK
   - 默认值0

2. **Sample索引检测优先级**：
   - 手动设置的值
   - 环境变量CURRENT_SAMPLE_IDX
   - 默认值0

## 文件命名格式

使用全局状态后，保存的文件名格式为：
```
timestamp_counter_iter{iteration}_layer_type_L{layer_idx}_operation_phase_component_quant_type_rank{rank}_sample{sample_idx}_tensor_name.pt
```

例如：
```
20241201_143022_0001_iter000_linear_L0_forward_post_linear_bf16_rank0_sample0_output.pt
```

## 元数据信息

保存的tensor文件包含完整的元数据：
```python
{
    "rank": 0,
    "sample_idx": 0,
    "iteration": 0,
    "layer_type": "linear",
    "operation": "forward",
    "quant_type": "bf16",
    "tensor_name": "output",
    "layer_idx": 0,
    "phase": "post",
    "component": "linear",
    "save_time": "2024-12-01 14:30:22"
}
```

## 训练脚本集成

在训练脚本中，系统会自动：
1. 从环境变量读取配置
2. 初始化全局状态
3. 在保存tensor时自动获取rank和sample信息

```bash
# 设置环境变量
export TENSOR_SAVE_ENABLED="true"
export TENSOR_SAVE_DIR="./enhanced_tensor_logs"
export TENSOR_SAVER_ITERATION=0
export CURRENT_SAMPLE_IDX=0
export LOCAL_RANK=0

# 运行训练
bash run_wikipedia_tensor_collection.sh
```

## 多GPU支持

在多GPU环境中，每个进程会自动获取自己的rank信息：
```python
# 每个GPU进程会自动获取自己的rank
rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
```

## 测试和验证

使用提供的测试脚本验证功能：
```bash
python test_tensor_saver_rank_sample.py
python example_tensor_collection_usage.py
```

## 优势

1. **自动化**: 无需手动传递rank和sample信息
2. **灵活性**: 支持多种获取方式（环境变量、分布式环境、手动设置）
3. **一致性**: 全局状态确保所有tensor使用相同的状态信息
4. **可扩展性**: 易于添加新的状态信息
5. **向后兼容**: 现有的save_tensor调用无需修改

## 注意事项

1. 确保在训练开始前正确设置环境变量
2. 在多GPU环境中，每个进程会自动获取自己的rank
3. 如果需要手动控制状态，可以在训练循环中调用相应的设置函数
4. 系统会提供默认值，确保即使在没有设置的情况下也能正常工作
