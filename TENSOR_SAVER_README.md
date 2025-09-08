# Tensor保存器使用说明

## 概述

本模块提供了在Megatron-LM训练过程中保存attention和linear层的forward/backward输入tensor的功能，支持多种量化类型（hifp8、mxfp8、mxfp4、bf16等）。

## 功能特性

- ✅ 自动保存attention层的query、key、value tensor
- ✅ 自动保存linear层的input、weight tensor
- ✅ 支持forward和backward操作
- ✅ 支持多种量化类型
- ✅ 包含详细的tensor元数据
- ✅ 自动生成带时间戳的文件名
- ✅ 支持环境变量配置

## 文件结构

```
megatron/core/
├── tensor_saver.py          # 主要的tensor保存器模块
└── transformer/
    └── dot_product_attention.py  # 已修改，支持tensor保存
└── tensor_parallel/
    └── layers.py            # 已修改，支持tensor保存
```

## 环境变量配置

### 必需的环境变量

```bash
# 设置量化类型
export CUSTOM_QUANT_TYPE="hifp8"  # 可选: hifp8, mxfp8, mxfp4, bf16, fp16

# 设置tensor保存目录
export TENSOR_SAVE_DIR="./tensor_logs"

# 启用/禁用tensor保存
export TENSOR_SAVE_ENABLED="true"  # true/false
```

### 在训练脚本中使用

```bash
#!/bin/bash

# 设置环境变量
export CUSTOM_QUANT_TYPE="hifp8"
export TENSOR_SAVE_DIR="./tensor_logs/experiment_001"
export TENSOR_SAVE_ENABLED="true"

# 运行训练
bash examples/llama/train_llama32_1b_h100_fp8.sh \
    checkpoints/llama32_1b_hifp8 \
    tensorboard_logs/llama32_1b_hifp8 \
    model/llama3.2-1b \
    dataset/wikipedia_processed/wikipedia_processed_text_document \
    bf16
```

## 保存的文件格式

### 文件命名规则

```
{timestamp}_{counter}_{layer_type}_{operation}_{quant_type}_{tensor_name}.pt
```

示例：
- `20241208_143022_0001_attention_forward_hifp8_query.pt`
- `20241208_143022_0002_linear_forward_mxfp8_input.pt`
- `20241208_143022_0003_attention_backward_mxfp4_key.pt`

### 文件内容结构

```python
{
    "tensor": torch.Tensor,  # 实际的tensor数据 (CPU, detached)
    "tensor_info": {
        "shape": [batch_size, seq_len, hidden_size],
        "dtype": "torch.bfloat16",
        "device": "cuda:0",
        "requires_grad": True,
        "is_leaf": False,
        "min": -2.5,
        "max": 3.1,
        "mean": 0.02,
        "std": 0.8
    },
    "metadata": {
        "layer_type": "attention",  # 或 "linear"
        "operation": "forward",     # 或 "backward"
        "quant_type": "hifp8",
        "tensor_name": "query",     # query, key, value, input, weight
        "layer_idx": 0,
        "save_time": "2024-12-08 14:30:22",
        # 其他自定义元数据...
    }
}
```

## 支持的量化类型

| 量化类型 | 描述 | 适用场景 |
|----------|------|----------|
| `hifp8` | HiFloat8格式 | 高精度训练和推理 |
| `mxfp8` | Micro-scaling FP8 | 平衡精度和效率 |
| `mxfp4` | Micro-scaling FP4 | 极致压缩推理 |
| `bf16` | Brain Float 16 | 标准半精度 |
| `fp16` | Float 16 | 标准半精度 |

## 使用示例

### 1. 基本使用

```python
from megatron.core.tensor_saver import save_attention_tensors, save_linear_tensors
import torch

# 保存attention tensor
query = torch.randn(2, 4, 8, 16, dtype=torch.bfloat16)
key = torch.randn(2, 4, 8, 16, dtype=torch.bfloat16)
value = torch.randn(2, 4, 8, 16, dtype=torch.bfloat16)

results = save_attention_tensors(
    query=query,
    key=key,
    value=value,
    quant_type="hifp8",
    operation="forward",
    layer_idx=0,
    metadata={"experiment": "test_run"}
)

# 保存linear tensor
input_tensor = torch.randn(32, 64, dtype=torch.bfloat16)
weight = torch.randn(128, 64, dtype=torch.bfloat16)

results = save_linear_tensors(
    input_tensor=input_tensor,
    weight=weight,
    quant_type="mxfp8",
    operation="forward",
    layer_idx=1,
    metadata={"experiment": "test_run"}
)
```

### 2. 加载保存的tensor

```python
import torch

# 加载tensor文件
data = torch.load("20241208_143022_0001_attention_forward_hifp8_query.pt", map_location='cpu')

tensor = data['tensor']
tensor_info = data['tensor_info']
metadata = data['metadata']

print(f"Tensor形状: {tensor.shape}")
print(f"量化类型: {metadata['quant_type']}")
print(f"操作类型: {metadata['operation']}")
print(f"数值范围: [{tensor_info['min']:.4f}, {tensor_info['max']:.4f}]")
```

### 3. 批量分析保存的tensor

```python
import glob
import torch
import numpy as np

def analyze_saved_tensors(tensor_dir="./tensor_logs"):
    """分析保存的tensor文件"""
    tensor_files = glob.glob(f"{tensor_dir}/*.pt")
    
    results = {}
    for file_path in tensor_files:
        data = torch.load(file_path, map_location='cpu')
        tensor = data['tensor']
        metadata = data['metadata']
        
        key = f"{metadata['layer_type']}_{metadata['operation']}_{metadata['quant_type']}"
        if key not in results:
            results[key] = []
        
        results[key].append({
            'file': file_path,
            'shape': tensor.shape,
            'mean': float(tensor.mean()),
            'std': float(tensor.std()),
            'min': float(tensor.min()),
            'max': float(tensor.max())
        })
    
    return results

# 使用示例
results = analyze_saved_tensors()
for key, tensors in results.items():
    print(f"\n{key}:")
    for tensor_info in tensors:
        print(f"  {tensor_info['file']}: shape={tensor_info['shape']}, "
              f"mean={tensor_info['mean']:.4f}, std={tensor_info['std']:.4f}")
```

## 测试

运行测试脚本验证功能：

```bash
cd /data/charles/Megatron-LM
python test_tensor_saver.py
```

## 注意事项

1. **存储空间**: tensor保存会占用大量磁盘空间，建议定期清理
2. **性能影响**: 保存tensor会有轻微的性能开销
3. **内存使用**: tensor会被移动到CPU内存，注意内存使用
4. **文件管理**: 建议为不同实验创建不同的保存目录

## 故障排除

### 常见问题

1. **保存失败**: 检查目录权限和磁盘空间
2. **内存不足**: 减少保存的tensor数量或使用更小的batch size
3. **文件过多**: 定期清理旧的tensor文件

### 调试模式

```bash
# 启用详细日志
export TENSOR_SAVE_DEBUG="true"

# 只保存特定层
export TENSOR_SAVE_LAYER_FILTER="0,1,2"  # 只保存前3层
```

## 扩展功能

### 自定义保存逻辑

```python
from megatron.core.tensor_saver import TensorSaver

# 创建自定义保存器
saver = TensorSaver(save_dir="./custom_logs", enabled=True)

# 自定义保存逻辑
def custom_save_hook(tensor, layer_type, operation, quant_type, tensor_name, layer_idx):
    # 只保存特定条件的tensor
    if layer_idx < 5 and operation == "forward":
        return saver.save_tensor(
            tensor, layer_type, operation, quant_type, tensor_name, layer_idx
        )
    return None
```

这个tensor保存器为量化研究提供了强大的数据收集和分析能力，帮助理解不同量化类型对模型行为的影响。
