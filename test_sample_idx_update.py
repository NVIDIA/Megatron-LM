#!/usr/bin/env python3
"""
测试sample idx更新逻辑
"""

import sys
import os
sys.path.append('/data/charles/codes/Megatron-LM')

class MockTensorCollectionState:
    def __init__(self):
        self.current_sample_idx = 0
    
    def get_sample_idx(self):
        return self.current_sample_idx
    
    def set_sample_idx(self, sample_idx):
        self.current_sample_idx = sample_idx
        print(f"[MockState] 设置sample_idx: {sample_idx}")

def test_sample_idx_update():
    """测试sample idx更新逻辑"""
    print("测试sample idx更新逻辑...")
    
    # 模拟tensor collection state
    state = MockTensorCollectionState()
    
    # 模拟iteration开始，重置sample idx
    print("\n=== 开始iteration 0 ===")
    state.set_sample_idx(0)
    
    # 模拟128个micro batch的forward pass
    print("\n=== 模拟128个micro batch的forward pass ===")
    for i in range(128):
        current_sample = state.get_sample_idx() or 0
        state.set_sample_idx(current_sample + 1)
        
        if i < 5 or i >= 123:  # 只显示前5个和后5个
            print(f"Micro batch {i}: sample_idx = {state.get_sample_idx()}")
    
    print(f"\n最终sample_idx: {state.get_sample_idx()}")
    print("测试完成！")

if __name__ == "__main__":
    test_sample_idx_update()
