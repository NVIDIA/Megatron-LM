#!/usr/bin/env python3
"""
简化的micro batch sample映射修复方案
"""

def create_simple_fix():
    """创建简化的修复方案"""
    
    # 创建micro batch sample更新函数
    update_code = '''#!/usr/bin/env python3
"""
简单的micro batch sample更新函数
"""

import os
import sys
sys.path.append('.')

def update_microbatch_sample(microbatch_index: int):
    """更新micro batch对应的sample_idx"""
    try:
        from megatron.core.tensor_saver import set_global_sample_idx
        set_global_sample_idx(microbatch_index)
        print(f"[MicroBatchSample] 更新sample_idx为: {microbatch_index}")
    except Exception as e:
        print(f"[MicroBatchSample] 更新sample_idx失败: {e}")

def get_current_microbatch_sample():
    """获取当前micro batch对应的sample"""
    try:
        from megatron.core.tensor_saver import get_tensor_collection_state
        state = get_tensor_collection_state()
        return state.get_sample_idx()
    except Exception as e:
        print(f"[MicroBatchSample] 获取当前sample失败: {e}")
        return 0

if __name__ == "__main__":
    print("Micro batch sample更新函数已准备就绪")
    print("使用方法:")
    print("1. 在micro batch循环开始时调用 update_microbatch_sample(microbatch_index)")
    print("2. 使用 get_current_microbatch_sample() 获取当前sample")
'''
    
    with open("microbatch_sample_update.py", "w", encoding="utf-8") as f:
        f.write(update_code)
    
    print("已创建: microbatch_sample_update.py")

def create_usage_instructions():
    """创建使用说明"""
    
    instructions = '''# Micro Batch Sample映射修复说明

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
'''
    
    with open("MICROBATCH_FIX_INSTRUCTIONS.md", "w", encoding="utf-8") as f:
        f.write(instructions)
    
    print("已创建: MICROBATCH_FIX_INSTRUCTIONS.md")

def main():
    print("=== 创建简化的Micro Batch Sample映射修复方案 ===")
    
    # 创建micro batch sample更新函数
    create_simple_fix()
    
    # 创建使用说明
    create_usage_instructions()
    
    print("\\n修复方案已创建完成!")
    print("\\n文件列表:")
    print("1. microbatch_sample_update.py - 简单的sample更新函数")
    print("2. MICROBATCH_FIX_INSTRUCTIONS.md - 详细使用说明")
    print("\\n下一步:")
    print("1. 阅读 MICROBATCH_FIX_INSTRUCTIONS.md 了解使用方法")
    print("2. 在训练脚本的micro batch循环中添加 update_microbatch_sample(microbatch_index) 调用")
    print("3. 重新运行数据收集脚本")

if __name__ == "__main__":
    main()


