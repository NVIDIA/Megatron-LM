#!/usr/bin/env python3
"""
Micro batch和sample映射修复方案
直接在tensor保存代码中处理micro batch和sample的正确映射
"""

import os
import sys
import re
import shutil
from pathlib import Path

def create_microbatch_aware_tensor_saver():
    """创建支持micro batch的tensor保存器"""
    
    code = '''#!/usr/bin/env python3
"""
支持micro batch的tensor保存器
正确处理micro batch和sample的映射关系
"""

import os
import sys
import torch
import time
from typing import Optional, Dict, Any
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append('.')

from megatron.core.tensor_saver import (
    TensorCollectionState, 
    get_tensor_collection_state,
    set_global_sample_idx,
    get_current_rank_and_sample
)

class MicroBatchAwareTensorSaver:
    """支持micro batch的tensor保存器"""
    
    def __init__(self, save_dir: str = "./enhanced_tensor_logs", enabled: bool = True):
        self.save_dir = Path(save_dir)
        self.enabled = enabled
        self.tensor_counter = 0
        self.current_iteration = 0
        self.current_microbatch = 0
        
        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"[MicroBatchAwareTensorSaver] 初始化完成，保存目录: {self.save_dir}")
    
    def set_iteration(self, iteration: int):
        """设置当前iteration"""
        self.current_iteration = iteration
        print(f"[MicroBatchAwareTensorSaver] 设置当前iteration: {iteration}")
    
    def set_microbatch(self, microbatch: int):
        """设置当前micro batch并更新sample_idx"""
        self.current_microbatch = microbatch
        # 更新全局sample_idx
        set_global_sample_idx(microbatch)
        print(f"[MicroBatchAwareTensorSaver] 设置当前micro batch: {microbatch}, sample_idx: {microbatch}")
    
    def save_tensor(self, 
                   tensor: torch.Tensor,
                   layer_type: str,
                   operation: str,
                   quant_type: str,
                   tensor_name: str,
                   layer_idx: Optional[int] = None,
                   phase: str = "unknown",
                   component: str = "unknown",
                   rank: Optional[int] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """保存tensor，使用当前micro batch作为sample_idx"""
        if not self.enabled:
            return None
        
        # 获取rank信息
        if rank is None:
            rank, _ = get_current_rank_and_sample()
            if rank is None:
                rank = 0
        
        # 使用当前micro batch作为sample_idx
        sample_idx = self.current_microbatch
        
        # 生成文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.tensor_counter += 1
        
        parts = [
            timestamp,
            f"{self.tensor_counter:04d}",
            f"iter{self.current_iteration:03d}",
            layer_type
        ]
        
        if layer_idx is not None:
            parts.append(f"L{layer_idx}")
        
        parts.extend([operation, phase, component, quant_type])
        
        if rank is not None:
            parts.append(f"rank{rank:02d}")
        
        parts.append(f"sample{sample_idx:03d}")
        parts.append(tensor_name)
        
        filename = "_".join(parts) + ".pt"
        filepath = self.save_dir / quant_type / filename
        
        # 确保目录存在
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 准备保存数据
            save_data = {
                "tensor": tensor.detach().cpu(),
                "metadata": {
                    "layer_type": layer_type,
                    "operation": operation,
                    "quant_type": quant_type,
                    "tensor_name": tensor_name,
                    "layer_idx": layer_idx,
                    "phase": phase,
                    "component": component,
                    "rank": rank,
                    "sample_idx": sample_idx,
                    "microbatch": self.current_microbatch,
                    "iteration": self.current_iteration,
                    "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    **(metadata or {})
                }
            }
            
            # 保存到文件
            torch.save(save_data, filepath)
            
            print(f"[MicroBatchAwareTensorSaver] 已保存: {filename} "
                  f"(microbatch={self.current_microbatch}, sample={sample_idx}, "
                  f"shape={tensor.shape}, dtype={tensor.dtype})")
            
            return str(filepath)
            
        except Exception as e:
            print(f"[MicroBatchAwareTensorSaver] 保存tensor失败: {e}")
            return None

# 全局实例
_microbatch_aware_saver = None

def get_microbatch_aware_saver() -> MicroBatchAwareTensorSaver:
    """获取micro batch感知的tensor保存器实例"""
    global _microbatch_aware_saver
    if _microbatch_aware_saver is None:
        _microbatch_aware_saver = MicroBatchAwareTensorSaver()
    return _microbatch_aware_saver

def update_microbatch_sample(microbatch: int):
    """更新micro batch和对应的sample"""
    saver = get_microbatch_aware_saver()
    saver.set_microbatch(microbatch)

if __name__ == "__main__":
    print("Micro batch感知的tensor保存器已准备就绪")
    print("使用方法:")
    print("1. 在micro batch循环开始时调用 update_microbatch_sample(microbatch_index)")
    print("2. 使用 get_microbatch_aware_saver().save_tensor() 保存tensor")
'''
    
    with open("microbatch_aware_tensor_saver.py", "w", encoding="utf-8") as f:
        f.write(code)
    
    print("已创建: microbatch_aware_tensor_saver.py")

def create_patch_for_tensor_parallel_layers():
    """为tensor_parallel/layers.py创建补丁"""
    
    patch_code = '''#!/usr/bin/env python3
"""
为tensor_parallel/layers.py创建micro batch sample映射补丁
"""

import os
import re
import shutil

def apply_patch():
    """应用补丁到tensor_parallel/layers.py"""
    file_path = "megatron/core/tensor_parallel/layers.py"
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    # 备份原文件
    backup_path = f"{file_path}.backup"
    shutil.copy2(file_path, backup_path)
    print(f"已备份: {file_path} -> {backup_path}")
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 在save_linear_tensors调用之前添加micro batch sample更新
    pattern = r'(save_linear_tensors\(\n.*?\))'
    replacement = '''# 更新micro batch sample映射
try:
    from microbatch_aware_tensor_saver import update_microbatch_sample
    # 尝试从调用栈获取当前micro batch索引
    import inspect
    frame = inspect.currentframe()
    current_microbatch = 0
    while frame:
        frame = frame.f_back
        if frame and 'micro_batch_index' in frame.f_locals:
            current_microbatch = frame.f_locals['micro_batch_index']
            break
        elif frame and 'current_microbatch' in frame.f_locals:
            current_microbatch = frame.f_locals['current_microbatch']
            break
    update_microbatch_sample(current_microbatch)
except Exception as e:
    print(f"[TensorParallel] Warning: Failed to update micro batch sample: {e}")

save_linear_tensors(
    input_tensor=total_input,
    weight=weight,
    quant_type=custom_quant_type,
    operation="forward",
    layer_idx=layer_idx,
    phase="pre",
    component="linear",
    metadata={
        "sequence_parallel": sequence_parallel,
        "use_bias": ctx.use_bias,
        "tp_group_size": tp_group.size() if tp_group else None,
    }
)'''
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"已应用补丁: {file_path}")
        return True
    else:
        print(f"未找到需要补丁的内容: {file_path}")
        return False

if __name__ == "__main__":
    apply_patch()
'''
    
    with open("patch_tensor_parallel_layers.py", "w", encoding="utf-8") as f:
        f.write(patch_code)
    
    print("已创建: patch_tensor_parallel_layers.py")

def create_usage_guide():
    """创建使用指南"""
    
    guide = '''# Micro Batch Sample映射修复使用指南

## 问题描述
在原始的数据收集过程中，所有的tensor都被保存为sample 0，但实际上应该根据micro batch索引来设置sample_idx。

## 解决方案
1. 创建了支持micro batch的tensor保存器
2. 在micro batch循环中正确更新sample_idx
3. 每个micro batch的tensor保存为对应的sample

## 使用方法

### 1. 使用microbatch_aware_tensor_saver.py
```python
from microbatch_aware_tensor_saver import update_microbatch_sample, get_microbatch_aware_saver

# 在micro batch循环中
for micro_batch_index in range(num_micro_batches):
    # 更新sample_idx
    update_microbatch_sample(micro_batch_index)
    
    # 进行forward pass
    # tensor保存将自动使用正确的sample_idx
```

### 2. 修改现有的tensor保存调用
将现有的tensor保存调用替换为使用microbatch_aware_tensor_saver：

```python
# 原来的调用
from megatron.core.tensor_saver import save_linear_tensors
save_linear_tensors(...)

# 修改为
from microbatch_aware_tensor_saver import get_microbatch_aware_saver
saver = get_microbatch_aware_saver()
saver.save_tensor(tensor, layer_type, operation, quant_type, tensor_name, ...)
```

### 3. 在训练脚本中集成
在训练脚本的micro batch循环中添加sample更新：

```python
# 在micro batch循环开始时
for micro_batch_index in range(num_micro_batches):
    # 更新sample_idx
    from microbatch_aware_tensor_saver import update_microbatch_sample
    update_microbatch_sample(micro_batch_index)
    
    # 继续正常的训练流程
    # ...
```

## 验证修复
运行数据收集后，检查生成的tensor文件：
- sample 0应该包含第一个micro batch的tensor
- sample 1应该包含第二个micro batch的tensor
- 以此类推

## 文件说明
- `microbatch_aware_tensor_saver.py`: 支持micro batch的tensor保存器
- `patch_tensor_parallel_layers.py`: 为tensor_parallel/layers.py创建补丁
- `microbatch_sample_update.py`: 简单的sample更新函数
'''
    
    with open("MICROBATCH_SAMPLE_FIX_GUIDE.md", "w", encoding="utf-8") as f:
        f.write(guide)
    
    print("已创建: MICROBATCH_SAMPLE_FIX_GUIDE.md")

def main():
    print("=== 创建Micro Batch Sample映射修复方案 ===")
    
    # 创建支持micro batch的tensor保存器
    create_microbatch_aware_tensor_saver()
    
    # 创建tensor_parallel/layers.py补丁
    create_patch_for_tensor_parallel_layers()
    
    # 创建使用指南
    create_usage_guide()
    
    print("\\n修复方案已创建完成!")
    print("\\n文件列表:")
    print("1. microbatch_aware_tensor_saver.py - 支持micro batch的tensor保存器")
    print("2. patch_tensor_parallel_layers.py - tensor_parallel/layers.py补丁")
    print("3. MICROBATCH_SAMPLE_FIX_GUIDE.md - 详细使用指南")
    print("\\n下一步:")
    print("1. 阅读 MICROBATCH_SAMPLE_FIX_GUIDE.md 了解使用方法")
    print("2. 根据需要应用补丁或修改代码")
    print("3. 重新运行数据收集脚本")

if __name__ == "__main__":
    main()
