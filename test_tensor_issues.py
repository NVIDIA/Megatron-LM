#!/usr/bin/env python3
"""
测试tensor收集问题
"""

import sys
import os
sys.path.append('/data/charles/codes/Megatron-LM')

def test_layer_idx_detection():
    """测试layer_idx检测逻辑"""
    print("测试layer_idx检测逻辑...")
    
    import inspect
    
    class MockTransformerLayer:
        def __init__(self, layer_number):
            self.layer_number = layer_number
    
    class MockMLP:
        def __init__(self, layer_idx):
            self.layer_idx = layer_idx
    
    def test_inspect_logic():
        """测试inspect逻辑"""
        # 模拟调用栈
        def inner_function():
            frame = inspect.currentframe()
            layer_idx = None
            while frame:
                frame = frame.f_back
                if frame and 'self' in frame.f_locals:
                    self_obj = frame.f_locals['self']
                    if hasattr(self_obj, 'layer_number'):
                        layer_idx = self_obj.layer_number
                        break
                    elif hasattr(self_obj, 'layer_idx'):
                        layer_idx = self_obj.layer_idx
                        break
            return layer_idx
        
        def middle_function():
            return inner_function()
        
        def outer_function_with_layer():
            self = MockTransformerLayer(5)
            return middle_function()
        
        def outer_function_with_mlp():
            self = MockMLP(3)
            return middle_function()
        
        # 测试layer_number
        result1 = outer_function_with_layer()
        print(f"检测到layer_number: {result1}")
        assert result1 == 5, f"期望5，实际{result1}"
        
        # 测试layer_idx
        result2 = outer_function_with_mlp()
        print(f"检测到layer_idx: {result2}")
        assert result2 == 3, f"期望3，实际{result2}"
        
        print("✅ layer_idx检测逻辑测试通过！")
    
    test_inspect_logic()

def test_sample_count_estimation():
    """测试sample数量估算"""
    print("\n测试sample数量估算...")
    
    # 模拟参数
    global_batch_size = 128
    micro_batch_size = 1
    num_microbatches = global_batch_size // micro_batch_size
    num_layers = 32  # 假设32层
    num_ranks = 8    # 假设8个rank
    
    # 估算tensor数量
    # 每个micro batch，每个rank，每个layer可能生成多个tensor
    tensors_per_microbatch_per_rank_per_layer = 4  # attention: query, key, value, weights; linear: input, output
    total_tensors = num_microbatches * num_ranks * num_layers * tensors_per_microbatch_per_rank_per_layer
    
    print(f"参数设置:")
    print(f"  global_batch_size: {global_batch_size}")
    print(f"  micro_batch_size: {micro_batch_size}")
    print(f"  num_microbatches: {num_microbatches}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_ranks: {num_ranks}")
    print(f"  tensors_per_microbatch_per_rank_per_layer: {tensors_per_microbatch_per_rank_per_layer}")
    print(f"  估算总tensor数量: {total_tensors}")
    
    # 实际收集到的数量
    actual_tensors = 56
    print(f"  实际收集到的数量: {actual_tensors}")
    
    if actual_tensors < total_tensors * 0.1:  # 如果少于10%
        print(f"❌ 收集数量过少！期望至少 {total_tensors * 0.1}，实际 {actual_tensors}")
    else:
        print(f"✅ 收集数量合理")
    
    # 分析可能的原因
    print(f"\n可能的原因分析:")
    print(f"1. sample_idx没有正确更新，导致只收集了第一个sample")
    print(f"2. layer_idx没有正确设置，导致linear层没有层数标识")
    print(f"3. 某些tensor保存被跳过")

def test_filename_patterns():
    """测试文件名模式"""
    print("\n测试文件名模式...")
    
    # 模拟文件名
    filenames = [
        "20250914_022153_0003_iter000_linear_forward_post_linear_bf16_rank02_sample000_group000_output.pt",
        "20250914_022153_0004_iter000_attention_L1_forward_pre_FA_bf16_rank00_sample000_group000_query.pt",
    ]
    
    print("文件名分析:")
    for filename in filenames:
        parts = filename.split('_')
        print(f"  {filename}")
        
        # 检查是否有layer标识
        has_layer = any(part.startswith('L') and part[1:].isdigit() for part in parts)
        print(f"    有layer标识: {has_layer}")
        
        # 检查sample_idx
        sample_parts = [part for part in parts if part.startswith('sample')]
        if sample_parts:
            sample_idx = sample_parts[0].replace('sample', '')
            print(f"    sample_idx: {sample_idx}")
        
        # 检查iteration
        iter_parts = [part for part in parts if part.startswith('iter')]
        if iter_parts:
            iteration = iter_parts[0].replace('iter', '')
            print(f"    iteration: {iteration}")
        
        print()

if __name__ == "__main__":
    test_layer_idx_detection()
    test_sample_count_estimation()
    test_filename_patterns()
    
    print("\n🎯 问题总结:")
    print("1. Linear层缺少layer_idx标识")
    print("2. 收集的tensor数量过少（56个 vs 预期数千个）")
    print("3. 可能sample_idx没有正确更新")
    print("\n🔧 修复方案:")
    print("1. 添加layer_idx检测逻辑")
    print("2. 确保sample_idx正确更新")
    print("3. 验证tensor保存逻辑")
