#!/usr/bin/env python3
"""
测试tensor索引管理功能
验证同一层的不同tensor使用相同的索引
"""

import os
import sys
import torch
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.append('/data/charles/codes/Megatron-LM')

def test_tensor_index_management():
    """测试tensor索引管理功能"""
    print("=== 测试Tensor索引管理功能 ===")
    
    # 设置环境变量
    os.environ["LOCAL_RANK"] = "0"
    os.environ["CURRENT_SAMPLE_IDX"] = "0"
    os.environ["TENSOR_SAVER_ITERATION"] = "0"
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["TENSOR_SAVER_ENABLED"] = "true"
        os.environ["TENSOR_SAVE_DIR"] = temp_dir
        
        # 导入tensor_saver
        from megatron.core.tensor_saver import get_tensor_saver, get_tensor_index_manager
        
        # 获取tensor_saver和索引管理器
        saver = get_tensor_saver()
        index_manager = get_tensor_index_manager()
        
        # 创建测试tensor
        test_tensor = torch.randn(4, 8)
        
        print("测试1: Linear层 - 同一层的input、weight、output应该使用相同索引")
        
        # 模拟Linear层L0的forward操作
        layer_idx = 0
        operation = "forward"
        
        # 保存input
        result1 = saver.save_tensor(
            tensor=test_tensor,
            layer_type="linear",
            operation=operation,
            quant_type="bf16",
            tensor_name="input",
            layer_idx=layer_idx,
            phase="pre",
            component="linear"
        )
        
        # 保存weight
        result2 = saver.save_tensor(
            tensor=test_tensor,
            layer_type="linear",
            operation=operation,
            quant_type="bf16",
            tensor_name="weight",
            layer_idx=layer_idx,
            phase="pre",
            component="linear"
        )
        
        # 保存output
        result3 = saver.save_tensor(
            tensor=test_tensor,
            layer_type="linear",
            operation=operation,
            quant_type="bf16",
            tensor_name="output",
            layer_idx=layer_idx,
            phase="post",
            component="linear"
        )
        
        print(f"Linear L0 input: {Path(result1).name if result1 else 'None'}")
        print(f"Linear L0 weight: {Path(result2).name if result2 else 'None'}")
        print(f"Linear L0 output: {Path(result3).name if result3 else 'None'}")
        
        print("\n测试2: Attention层 - 同一层的query、key、value应该使用相同索引")
        
        # 模拟Attention层L1的forward操作
        layer_idx = 1
        operation = "forward"
        
        # 保存query
        result4 = saver.save_tensor(
            tensor=test_tensor,
            layer_type="attention",
            operation=operation,
            quant_type="hifp8",
            tensor_name="query",
            layer_idx=layer_idx,
            phase="pre",
            component="FA"
        )
        
        # 保存key
        result5 = saver.save_tensor(
            tensor=test_tensor,
            layer_type="attention",
            operation=operation,
            quant_type="hifp8",
            tensor_name="key",
            layer_idx=layer_idx,
            phase="pre",
            component="FA"
        )
        
        # 保存value
        result6 = saver.save_tensor(
            tensor=test_tensor,
            layer_type="attention",
            operation=operation,
            quant_type="hifp8",
            tensor_name="value",
            layer_idx=layer_idx,
            phase="pre",
            component="FA"
        )
        
        # 保存attention_weights
        result7 = saver.save_tensor(
            tensor=test_tensor,
            layer_type="attention",
            operation=operation,
            quant_type="hifp8",
            tensor_name="attention_weights",
            layer_idx=layer_idx,
            phase="post",
            component="FA"
        )
        
        print(f"Attention L1 query: {Path(result4).name if result4 else 'None'}")
        print(f"Attention L1 key: {Path(result5).name if result5 else 'None'}")
        print(f"Attention L1 value: {Path(result6).name if result6 else 'None'}")
        print(f"Attention L1 attention_weights: {Path(result7).name if result7 else 'None'}")
        
        print("\n测试3: 不同层应该使用不同的索引")
        
        # 模拟Linear层L2的forward操作
        layer_idx = 2
        operation = "forward"
        
        result8 = saver.save_tensor(
            tensor=test_tensor,
            layer_type="linear",
            operation=operation,
            quant_type="mxfp8",
            tensor_name="output",
            layer_idx=layer_idx,
            phase="post",
            component="linear"
        )
        
        print(f"Linear L2 output: {Path(result8).name if result8 else 'None'}")
        
        print("\n测试4: 不同操作类型应该使用不同的索引")
        
        # 模拟Linear层L0的backward操作
        layer_idx = 0
        operation = "backward"
        
        result9 = saver.save_tensor(
            tensor=test_tensor,
            layer_type="linear",
            operation=operation,
            quant_type="bf16",
            tensor_name="grad_input",
            layer_idx=layer_idx,
            phase="post",
            component="linear"
        )
        
        print(f"Linear L0 backward grad_input: {Path(result9).name if result9 else 'None'}")
        
        # 检查所有保存的文件
        print(f"\n=== 所有保存的文件分析 ===")
        saved_files = list(Path(temp_dir).glob("*.pt"))
        for file_path in sorted(saved_files):
            print(f"文件: {file_path.name}")
            data = torch.load(file_path, map_location='cpu')
            metadata = data['metadata']
            print(f"  Layer: {metadata.get('layer_type')} L{metadata.get('layer_idx')}")
            print(f"  Operation: {metadata.get('operation')}")
            print(f"  Tensor: {metadata.get('tensor_name')}")
            print(f"  GroupIdx: {metadata.get('tensor_group_idx', 'N/A')}")
            print()

def test_attention_weights_saving():
    """测试attention权重保存功能"""
    print("\n=== 测试Attention权重保存功能 ===")
    
    # 设置环境变量
    os.environ["LOCAL_RANK"] = "0"
    os.environ["CURRENT_SAMPLE_IDX"] = "0"
    os.environ["TENSOR_SAVER_ITERATION"] = "0"
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["TENSOR_SAVER_ENABLED"] = "true"
        os.environ["TENSOR_SAVE_DIR"] = temp_dir
        
        # 导入tensor_saver
        from megatron.core.tensor_saver import get_tensor_saver
        
        # 获取tensor_saver
        saver = get_tensor_saver()
        
        # 创建模拟的attention权重（P分布）
        batch_size = 2
        num_heads = 8
        seq_len = 128
        attention_weights = torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
        
        print(f"Attention权重形状: {attention_weights.shape}")
        print(f"Attention权重统计: mean={attention_weights.mean():.4f}, std={attention_weights.std():.4f}")
        print(f"Attention权重范围: [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")
        
        # 保存attention权重
        result = saver.save_tensor(
            tensor=attention_weights,
            layer_type="attention",
            operation="forward",
            quant_type="bf16",
            tensor_name="attention_weights",
            layer_idx=0,
            phase="post",
            component="FA",
            metadata={
                "attention_weights_shape": list(attention_weights.shape),
                "softmax_scale": 1.0,
                "description": "Attention probability distribution (P)"
            }
        )
        
        if result:
            print(f"保存成功: {Path(result).name}")
            
            # 检查保存的数据
            data = torch.load(result, map_location='cpu')
            metadata = data['metadata']
            print(f"保存的元数据: {metadata}")
            
            # 验证数据完整性
            saved_weights = data['tensor']
            print(f"保存的权重形状: {saved_weights.shape}")
            print(f"数据是否一致: {torch.allclose(attention_weights, saved_weights)}")

if __name__ == "__main__":
    test_tensor_index_management()
    test_attention_weights_saving()
    print("测试完成！")
