#!/usr/bin/env python3
"""
修复micro batch和sample映射问题的工具
在micro batch处理过程中正确更新sample_idx
"""

import os
import sys
import re
from pathlib import Path

def create_microbatch_sample_fix():
    """创建修复micro batch sample映射的补丁"""
    
    # 修复training.py中的sample_idx更新逻辑
    training_py_fix = '''
# 在train_step函数中，在forward_backward_func调用之前添加micro batch sample更新逻辑
# 找到这行：losses_reduced = forward_backward_func(
# 在其前面添加：

# Update sample index for each micro batch if tensor saving is enabled
if getattr(args, 'save_tensors', False):
    try:
        from megatron.core.tensor_saver import get_tensor_collection_state, set_global_sample_idx
        state = get_tensor_collection_state()
        # 在micro batch循环中更新sample_idx的逻辑将在forward_backward_func内部处理
        # 这里我们确保初始状态正确
        state.set_sample_idx(0)
    except Exception as e:
        print(f"[TrainStep] Warning: Failed to initialize sample index: {e}")
'''
    
    # 修复pipeline_parallel/schedules.py中的micro batch处理
    schedules_py_fix = '''
# 在forward_step函数中，在micro batch循环开始时添加sample_idx更新
# 找到micro batch循环，在循环开始时添加：

# Update sample index for tensor saving if enabled
if hasattr(config, 'save_tensors') and config.save_tensors:
    try:
        from megatron.core.tensor_saver import set_global_sample_idx
        # 使用current_microbatch作为sample_idx
        if current_microbatch is not None:
            set_global_sample_idx(current_microbatch)
    except Exception as e:
        print(f"[ForwardStep] Warning: Failed to update sample index: {e}")
'''
    
    # 创建修复脚本
    fix_script = '''#!/usr/bin/env python3
"""
自动修复micro batch sample映射问题
"""

import os
import re
import shutil
from pathlib import Path

def backup_file(file_path):
    """备份原文件"""
    backup_path = f"{file_path}.backup"
    shutil.copy2(file_path, backup_path)
    print(f"已备份: {file_path} -> {backup_path}")
    return backup_path

def fix_training_py():
    """修复training.py中的sample_idx更新逻辑"""
    file_path = "megatron/training/training.py"
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    # 备份原文件
    backup_file(file_path)
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找并替换train_step函数中的sample_idx重置逻辑
    pattern = r'(# Update sample index for tensor saving if enabled\n.*?state\.set_sample_idx\(0\))'
    replacement = '''# Update sample index for tensor saving if enabled
        if getattr(args, 'save_tensors', False):
            try:
                from megatron.core.tensor_saver import get_tensor_collection_state, set_global_sample_idx
                state = get_tensor_collection_state()
                # 在micro batch循环中更新sample_idx的逻辑将在forward_backward_func内部处理
                # 这里我们确保初始状态正确
                state.set_sample_idx(0)
            except Exception as e:
                print(f"[TrainStep] Warning: Failed to initialize sample index: {e}")'''
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"已修复: {file_path}")
        return True
    else:
        print(f"未找到需要修复的内容: {file_path}")
        return False

def fix_schedules_py():
    """修复schedules.py中的micro batch处理"""
    file_path = "megatron/core/pipeline_parallel/schedules.py"
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    # 备份原文件
    backup_file(file_path)
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 在forward_step函数中添加micro batch sample更新逻辑
    # 查找micro batch循环的开始位置
    pattern = r'(if is_first_microbatch and hasattr\(model, \'set_is_first_microbatch\'\):\n.*?set_input_tensor\(input_tensor\))'
    replacement = '''if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)
        
        # Update sample index for tensor saving if enabled
        if hasattr(config, 'save_tensors') and config.save_tensors:
            try:
                from megatron.core.tensor_saver import set_global_sample_idx
                # 使用current_microbatch作为sample_idx
                set_global_sample_idx(current_microbatch)
            except Exception as e:
                print(f"[ForwardStep] Warning: Failed to update sample index: {e}")

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)'''
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"已修复: {file_path}")
        return True
    else:
        print(f"未找到需要修复的内容: {file_path}")
        return False

def main():
    print("开始修复micro batch sample映射问题...")
    
    success_count = 0
    
    # 修复training.py
    if fix_training_py():
        success_count += 1
    
    # 修复schedules.py
    if fix_schedules_py():
        success_count += 1
    
    print(f"\\n修复完成! 成功修复 {success_count} 个文件")
    print("\\n修复说明:")
    print("1. 在micro batch循环中正确更新sample_idx")
    print("2. sample_idx现在对应micro batch索引")
    print("3. 每个micro batch的tensor将保存为对应的sample")
    print("\\n请重新运行数据收集脚本来生成正确的tensor文件")

if __name__ == "__main__":
    main()
'''
    
    # 保存修复脚本
    with open("fix_microbatch_sample_mapping.py", "w", encoding="utf-8") as f:
        f.write(fix_script)
    
    print("已创建修复脚本: fix_microbatch_sample_mapping.py")
    print("运行此脚本来修复micro batch sample映射问题")

def create_improved_tensor_saver():
    """创建改进的tensor保存逻辑"""
    
    improved_saver = '''#!/usr/bin/env python3
"""
改进的tensor保存逻辑，正确处理micro batch和sample映射
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

class ImprovedTensorSaver:
    """改进的tensor保存器，正确处理micro batch和sample映射"""
    
    def __init__(self, save_dir: str = "./enhanced_tensor_logs", enabled: bool = True):
        self.save_dir = Path(save_dir)
        self.enabled = enabled
        self.tensor_counter = 0
        self.current_iteration = 0
        self.current_microbatch = 0
        
        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"[ImprovedTensorSaver] 初始化完成，保存目录: {self.save_dir}")
    
    def set_iteration(self, iteration: int):
        """设置当前iteration"""
        self.current_iteration = iteration
        print(f"[ImprovedTensorSaver] 设置当前iteration: {iteration}")
    
    def set_microbatch(self, microbatch: int):
        """设置当前micro batch"""
        self.current_microbatch = microbatch
        # 更新全局sample_idx
        set_global_sample_idx(microbatch)
        print(f"[ImprovedTensorSaver] 设置当前micro batch: {microbatch}")
    
    def save_tensor_with_microbatch(self, 
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
        """使用micro batch信息保存tensor"""
        if not self.enabled:
            return None
        
        # 获取rank和sample信息
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
            
            print(f"[ImprovedTensorSaver] 已保存: {filename} "
                  f"(microbatch={self.current_microbatch}, sample={sample_idx}, "
                  f"shape={tensor.shape}, dtype={tensor.dtype})")
            
            return str(filepath)
            
        except Exception as e:
            print(f"[ImprovedTensorSaver] 保存tensor失败: {e}")
            return None

# 全局实例
_improved_tensor_saver = None

def get_improved_tensor_saver() -> ImprovedTensorSaver:
    """获取改进的tensor保存器实例"""
    global _improved_tensor_saver
    if _improved_tensor_saver is None:
        _improved_tensor_saver = ImprovedTensorSaver()
    return _improved_tensor_saver

def update_microbatch_sample(microbatch: int):
    """更新micro batch和对应的sample"""
    saver = get_improved_tensor_saver()
    saver.set_microbatch(microbatch)

if __name__ == "__main__":
    print("改进的tensor保存器已准备就绪")
    print("使用方法:")
    print("1. 在micro batch循环开始时调用 update_microbatch_sample(microbatch_index)")
    print("2. 使用 get_improved_tensor_saver().save_tensor_with_microbatch() 保存tensor")
'''
    
    # 保存改进的tensor保存器
    with open("improved_tensor_saver.py", "w", encoding="utf-8") as f:
        f.write(improved_saver)
    
    print("已创建改进的tensor保存器: improved_tensor_saver.py")

def main():
    print("=== 修复Micro Batch Sample映射问题 ===")
    
    # 创建修复脚本
    create_microbatch_sample_fix()
    
    # 创建改进的tensor保存器
    create_improved_tensor_saver()
    
    print("\\n修复方案已准备就绪:")
    print("1. 运行 fix_microbatch_sample_mapping.py 来修复现有代码")
    print("2. 使用 improved_tensor_saver.py 来正确处理micro batch和sample映射")
    print("3. 重新运行数据收集脚本来生成正确的tensor文件")

if __name__ == "__main__":
    main()
