#!/usr/bin/env python3
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
