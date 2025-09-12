#!/usr/bin/env python3
"""
测试tensor_saver的rank和sample信息获取功能
"""

import os
import sys
import torch
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.append('/data/charles/codes/Megatron-LM')

def test_rank_sample_detection():
    """测试rank和sample信息检测"""
    print("=== 测试Rank和Sample信息检测 ===")
    
    # 测试环境变量设置
    os.environ["CURRENT_SAMPLE_IDX"] = "42"
    os.environ["LOCAL_RANK"] = "3"
    
    # 导入tensor_saver模块
    from megatron.core.tensor_saver import get_current_rank_and_sample, get_tensor_saver, initialize_tensor_collection
    
    # 手动初始化状态
    print("手动初始化tensor收集状态...")
    initialize_tensor_collection(rank=5, sample_idx=10, iteration=1)
    
    # 测试获取rank和sample信息
    rank, sample_idx = get_current_rank_and_sample()
    print(f"检测到的Rank: {rank}")
    print(f"检测到的Sample IDX: {sample_idx}")
    
    # 测试tensor保存
    print("\n=== 测试Tensor保存 ===")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 设置环境变量
        os.environ["TENSOR_SAVE_ENABLED"] = "true"
        os.environ["TENSOR_SAVE_DIR"] = temp_dir
        os.environ["TENSOR_SAVER_ITERATION"] = "0"
        
        # 获取tensor_saver
        saver = get_tensor_saver()
        
        # 创建测试tensor
        test_tensor = torch.randn(10, 20)
        
        # 保存tensor（不指定rank和sample_idx，应该自动获取）
        result = saver.save_tensor(
            tensor=test_tensor,
            layer_type="linear",
            operation="forward",
            quant_type="bf16",
            tensor_name="test_output",
            layer_idx=0,
            phase="post",
            component="linear"
        )
        
        if result:
            print(f"Tensor保存成功: {result}")
            
            # 加载并检查保存的数据
            saved_data = torch.load(result, map_location='cpu')
            metadata = saved_data['metadata']
            
            print(f"保存的Rank: {metadata.get('rank')}")
            print(f"保存的Sample IDX: {metadata.get('sample_idx')}")
            print(f"保存的Iteration: {metadata.get('iteration')}")
            print(f"文件名包含rank和sample信息: {'rank' in Path(result).name and 'sample' in Path(result).name}")
        else:
            print("Tensor保存失败")

def test_manual_rank_sample():
    """测试手动指定rank和sample"""
    print("\n=== 测试手动指定Rank和Sample ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 设置环境变量
        os.environ["TENSOR_SAVE_ENABLED"] = "true"
        os.environ["TENSOR_SAVE_DIR"] = temp_dir
        os.environ["TENSOR_SAVER_ITERATION"] = "0"
        
        # 获取tensor_saver
        saver = get_tensor_saver()
        
        # 创建测试tensor
        test_tensor = torch.randn(5, 10)
        
        # 手动指定rank和sample_idx
        result = saver.save_tensor(
            tensor=test_tensor,
            layer_type="attention",
            operation="forward",
            quant_type="hifp8",
            tensor_name="test_query",
            layer_idx=1,
            phase="pre",
            component="FA",
            rank=3,  # 手动指定
            sample_idx=7  # 手动指定
        )
        
        if result:
            print(f"Tensor保存成功: {result}")
            
            # 加载并检查保存的数据
            saved_data = torch.load(result, map_location='cpu')
            metadata = saved_data['metadata']
            
            print(f"手动指定的Rank: {metadata.get('rank')}")
            print(f"手动指定的Sample IDX: {metadata.get('sample_idx')}")
            print(f"文件名: {Path(result).name}")
        else:
            print("Tensor保存失败")

if __name__ == "__main__":
    test_rank_sample_detection()
    test_manual_rank_sample()
    print("\n测试完成！")
