#!/usr/bin/env python3
"""
Tensor收集功能使用示例
展示如何在训练代码中正确使用全局状态管理
"""

import os
import sys
import torch
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.append('/data/charles/codes/Megatron-LM')

def example_training_loop():
    """模拟训练循环中的tensor收集"""
    print("=== 模拟训练循环中的Tensor收集 ===")
    
    # 导入tensor_saver模块
    from megatron.core.tensor_saver import (
        initialize_tensor_collection, 
        get_tensor_saver,
        set_global_rank,
        set_global_sample_idx,
        set_global_iteration
    )
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 设置环境变量
        os.environ["TENSOR_SAVE_ENABLED"] = "true"
        os.environ["TENSOR_SAVE_DIR"] = temp_dir
        
        # 模拟多GPU训练环境
        print("1. 初始化tensor收集状态...")
        initialize_tensor_collection(rank=0, sample_idx=0, iteration=0)
        
        # 获取tensor_saver
        saver = get_tensor_saver()
        
        # 模拟训练循环
        for iteration in range(3):
            print(f"\n--- 训练迭代 {iteration} ---")
            
            # 更新全局状态
            set_global_iteration(iteration)
            
            # 模拟处理不同的batch
            for batch_idx in range(2):
                print(f"  处理batch {batch_idx}")
                set_global_sample_idx(batch_idx)
                
                # 模拟linear层forward
                print("    执行linear层forward...")
                linear_output = torch.randn(8, 16)
                
                # 保存tensor（不需要手动指定rank和sample_idx）
                result = saver.save_tensor(
                    tensor=linear_output,
                    layer_type="linear",
                    operation="forward",
                    quant_type="bf16",
                    tensor_name="output",
                    layer_idx=0,
                    phase="post",
                    component="linear"
                )
                
                if result:
                    print(f"    保存成功: {Path(result).name}")
                
                # 模拟attention层forward
                print("    执行attention层forward...")
                attn_output = torch.randn(8, 16)
                
                result = saver.save_tensor(
                    tensor=attn_output,
                    layer_type="attention",
                    operation="forward",
                    quant_type="hifp8",
                    tensor_name="output",
                    layer_idx=0,
                    phase="post",
                    component="FA"
                )
                
                if result:
                    print(f"    保存成功: {Path(result).name}")
        
        # 检查保存的文件
        print(f"\n=== 保存的文件列表 ===")
        saved_files = list(Path(temp_dir).glob("*.pt"))
        for file_path in saved_files:
            print(f"文件: {file_path.name}")
            
            # 加载并检查元数据
            data = torch.load(file_path, map_location='cpu')
            metadata = data['metadata']
            print(f"  Rank: {metadata.get('rank')}")
            print(f"  Sample IDX: {metadata.get('sample_idx')}")
            print(f"  Iteration: {metadata.get('iteration')}")
            print(f"  Layer: {metadata.get('layer_type')}")
            print(f"  Operation: {metadata.get('operation')}")
            print()

def example_multi_gpu_simulation():
    """模拟多GPU环境下的tensor收集"""
    print("\n=== 模拟多GPU环境下的Tensor收集 ===")
    
    from megatron.core.tensor_saver import (
        initialize_tensor_collection, 
        get_tensor_saver,
        set_global_rank
    )
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["TENSOR_SAVE_ENABLED"] = "true"
        os.environ["TENSOR_SAVE_DIR"] = temp_dir
        
        # 模拟4个GPU
        for gpu_rank in range(4):
            print(f"\n--- GPU {gpu_rank} ---")
            
            # 为每个GPU设置不同的rank
            set_global_rank(gpu_rank)
            initialize_tensor_collection(rank=gpu_rank, sample_idx=0, iteration=0)
            
            # 获取tensor_saver
            saver = get_tensor_saver()
            
            # 模拟每个GPU处理不同的数据
            for sample_idx in range(2):
                from megatron.core.tensor_saver import set_global_sample_idx
                set_global_sample_idx(sample_idx)
                
                # 创建tensor
                tensor = torch.randn(4, 8) + gpu_rank  # 每个GPU的数据略有不同
                
                # 保存tensor
                result = saver.save_tensor(
                    tensor=tensor,
                    layer_type="linear",
                    operation="forward",
                    quant_type="mxfp8",
                    tensor_name="output",
                    layer_idx=gpu_rank,
                    phase="post",
                    component="linear"
                )
                
                if result:
                    print(f"  GPU{gpu_rank} Sample{sample_idx}: {Path(result).name}")
        
        # 检查所有保存的文件
        print(f"\n=== 所有保存的文件 ===")
        saved_files = list(Path(temp_dir).glob("*.pt"))
        for file_path in sorted(saved_files):
            data = torch.load(file_path, map_location='cpu')
            metadata = data['metadata']
            print(f"{file_path.name}")
            print(f"  GPU Rank: {metadata.get('rank')}")
            print(f"  Sample: {metadata.get('sample_idx')}")
            print(f"  Layer: {metadata.get('layer_idx')}")
            print()

if __name__ == "__main__":
    example_training_loop()
    example_multi_gpu_simulation()
    print("示例完成！")
