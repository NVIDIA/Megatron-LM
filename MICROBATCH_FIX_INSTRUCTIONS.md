# Micro Batch Sample映射修复说明

## 问题
在原始的数据收集过程中，所有的tensor都被保存为sample 0，但实际上应该根据micro batch索引来设置sample_idx。

## 解决方案
在micro batch循环中正确更新sample_idx，使每个micro batch的tensor保存为对应的sample。

## 使用方法

### 1. 在训练脚本中集成
在训练脚本的micro batch循环中添加sample更新：

```python
# 导入更新函数
from microbatch_sample_update import update_microbatch_sample

# 在micro batch循环中
for micro_batch_index in range(num_micro_batches):
    # 更新sample_idx
    update_microbatch_sample(micro_batch_index)
    
    # 继续正常的训练流程
    # 所有的tensor保存将自动使用正确的sample_idx
```

### 2. 在tensor保存代码中集成
在tensor保存调用之前添加sample更新：

```python
# 在save_linear_tensors或save_attention_tensors调用之前
from microbatch_sample_update import update_microbatch_sample

# 获取当前micro batch索引（从调用栈或参数传递）
current_microbatch = get_current_microbatch_index()  # 需要实现此函数
update_microbatch_sample(current_microbatch)

# 然后调用tensor保存函数
save_linear_tensors(...)
```

### 3. 修改现有的tensor保存调用
将现有的tensor保存调用包装为支持micro batch的版本：

```python
def save_tensor_with_microbatch(tensor, layer_type, operation, quant_type, tensor_name, 
                               layer_idx=None, phase="unknown", component="unknown", 
                               rank=None, microbatch_index=0, metadata=None):
    """支持micro batch的tensor保存函数"""
    # 更新sample_idx
    from microbatch_sample_update import update_microbatch_sample
    update_microbatch_sample(microbatch_index)
    
    # 调用原始的tensor保存函数
    from megatron.core.tensor_saver import save_tensor
    return save_tensor(tensor, layer_type, operation, quant_type, tensor_name, 
                      layer_idx, phase, component, rank, None, metadata)
```

## 验证修复
运行数据收集后，检查生成的tensor文件：
- sample 0应该包含第一个micro batch的tensor
- sample 1应该包含第二个micro batch的tensor
- 以此类推

## 文件说明
- `microbatch_sample_update.py`: 简单的sample更新函数
- 此文件提供了基本的micro batch sample映射功能
