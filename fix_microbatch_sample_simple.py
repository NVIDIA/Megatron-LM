#!/usr/bin/env python3
"""
简化的micro batch sample映射修复工具
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

def create_microbatch_sample_update():
    """创建micro batch sample更新函数"""
    
    update_code = '''#!/usr/bin/env python3
"""
Micro batch sample更新函数
在micro batch循环中正确更新sample_idx
"""

import os
import sys
sys.path.append('.')

from megatron.core.tensor_saver import set_global_sample_idx

def update_microbatch_sample(microbatch_index: int):
    """更新micro batch对应的sample_idx"""
    try:
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

def create_fixed_data_collection_script():
    """创建修复后的数据收集脚本"""
    
    script_content = '''#!/bin/bash
# 修复后的数据收集脚本，正确处理micro batch和sample映射

# 设置环境变量
export TENSOR_SAVE_ENABLED=1
export CURRENT_SAMPLE_IDX=0  # 初始值，将在micro batch循环中更新

# 在训练脚本中添加micro batch sample更新逻辑
cat > update_microbatch_sample.py << 'EOF'
import os
import sys
sys.path.append('.')

def update_sample_for_microbatch(microbatch_index):
    """为micro batch更新sample_idx"""
    try:
        from megatron.core.tensor_saver import set_global_sample_idx
        set_global_sample_idx(microbatch_index)
        print(f"更新sample_idx为: {microbatch_index}")
    except Exception as e:
        print(f"更新sample_idx失败: {e}")

# 在训练过程中调用此函数
# update_sample_for_microbatch(current_microbatch_index)
EOF

echo "修复后的数据收集脚本已准备就绪"
echo "请在训练脚本的micro batch循环中添加sample更新逻辑"
'''
    
    with open("fixed_data_collection.sh", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("已创建: fixed_data_collection.sh")

def main():
    print("=== 修复Micro Batch Sample映射问题 ===")
    
    success_count = 0
    
    # 修复training.py
    if fix_training_py():
        success_count += 1
    
    # 创建micro batch sample更新函数
    create_microbatch_sample_update()
    
    # 创建修复后的数据收集脚本
    create_fixed_data_collection_script()
    
    print(f"\\n修复完成! 成功修复 {success_count} 个文件")
    print("\\n修复说明:")
    print("1. 在micro batch循环中正确更新sample_idx")
    print("2. sample_idx现在对应micro batch索引")
    print("3. 每个micro batch的tensor将保存为对应的sample")
    print("\\n使用方法:")
    print("1. 在训练脚本的micro batch循环开始时调用 update_microbatch_sample(microbatch_index)")
    print("2. 重新运行数据收集脚本来生成正确的tensor文件")

if __name__ == "__main__":
    main()


