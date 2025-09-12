#!/usr/bin/env python3
"""
测试rank检测功能
"""

import os
import sys
import torch
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.append('/data/charles/codes/Megatron-LM')

def test_rank_detection():
    """测试rank检测功能"""
    print("=== 测试Rank检测功能 ===")
    
    # 设置环境变量
    os.environ["LOCAL_RANK"] = "2"
    os.environ["CURRENT_SAMPLE_IDX"] = "5"
    os.environ["TENSOR_SAVER_ITERATION"] = "1"
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["TENSOR_SAVE_ENABLED"] = "true"
        os.environ["TENSOR_SAVE_DIR"] = temp_dir
        
        # 导入tensor_saver
        from megatron.core.tensor_saver import get_tensor_saver
        
        # 获取tensor_saver
        saver = get_tensor_saver()
        
        # 创建测试tensor
        test_tensor = torch.randn(4, 8)
        
        print("测试1: 使用环境变量中的rank信息")
        result = saver.save_tensor(
            tensor=test_tensor,
            layer_type="linear",
            operation="forward",
            quant_type="bf16",
            tensor_name="output",
            layer_idx=0,
            phase="post",
            component="linear"
        )
        
        if result:
            print(f"保存成功: {Path(result).name}")
            
            # 检查保存的数据
            data = torch.load(result, map_location='cpu')
            metadata = data['metadata']
            print(f"保存的Rank: {metadata.get('rank')}")
            print(f"保存的Sample IDX: {metadata.get('sample_idx')}")
            print(f"保存的Iteration: {metadata.get('iteration')}")
        
        print("\n测试2: 模拟分布式环境")
        # 模拟torch.distributed环境
        try:
            import torch.distributed as dist
            # 这里我们无法真正初始化distributed，但可以测试代码逻辑
            print("尝试从torch.distributed获取rank...")
        except Exception as e:
            print(f"torch.distributed不可用: {e}")
        
        print("\n测试3: 直接指定rank和sample")
        result2 = saver.save_tensor(
            tensor=test_tensor,
            layer_type="attention",
            operation="forward",
            quant_type="hifp8",
            tensor_name="output",
            layer_idx=1,
            phase="post",
            component="FA",
            rank=3,  # 直接指定
            sample_idx=7  # 直接指定
        )
        
        if result2:
            print(f"保存成功: {Path(result2).name}")
            
            # 检查保存的数据
            data = torch.load(result2, map_location='cpu')
            metadata = data['metadata']
            print(f"直接指定的Rank: {metadata.get('rank')}")
            print(f"直接指定的Sample IDX: {metadata.get('sample_idx')}")
        
        # 检查所有保存的文件
        print(f"\n=== 所有保存的文件 ===")
        saved_files = list(Path(temp_dir).glob("*.pt"))
        for file_path in saved_files:
            print(f"文件: {file_path.name}")
            data = torch.load(file_path, map_location='cpu')
            metadata = data['metadata']
            print(f"  Rank: {metadata.get('rank')}")
            print(f"  Sample: {metadata.get('sample_idx')}")
            print(f"  Iteration: {metadata.get('iteration')}")
            print()

def test_tensor_device_rank():
    """测试从tensor设备信息推断rank"""
    print("\n=== 测试从Tensor设备信息推断Rank ===")
    
    from megatron.core.tensor_saver import get_rank_from_tensor_device
    
    # 测试CPU tensor
    cpu_tensor = torch.randn(2, 3)
    rank = get_rank_from_tensor_device(cpu_tensor)
    print(f"CPU tensor rank: {rank}")
    
    # 测试CUDA tensor（如果可用）
    if torch.cuda.is_available():
        cuda_tensor = torch.randn(2, 3).cuda(0)
        rank = get_rank_from_tensor_device(cuda_tensor)
        print(f"CUDA tensor (device 0) rank: {rank}")
        
        cuda_tensor2 = torch.randn(2, 3).cuda(1)
        rank = get_rank_from_tensor_device(cuda_tensor2)
        print(f"CUDA tensor (device 1) rank: {rank}")
    else:
        print("CUDA不可用，跳过CUDA tensor测试")

if __name__ == "__main__":
    test_rank_detection()
    test_tensor_device_rank()
    print("测试完成！")
