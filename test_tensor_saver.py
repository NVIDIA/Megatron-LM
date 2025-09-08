#!/usr/bin/env python3
"""
测试tensor保存功能的脚本
"""

import os
import torch
import sys
sys.path.append('/data/charles/Megatron-LM')

from megatron.core.tensor_saver import TensorSaver, save_attention_tensors, save_linear_tensors


def test_tensor_saver():
    """测试tensor保存器基本功能"""
    print("=== 测试TensorSaver基本功能 ===")
    
    # 创建测试tensor
    query = torch.randn(2, 4, 8, 16, dtype=torch.bfloat16)
    key = torch.randn(2, 4, 8, 16, dtype=torch.bfloat16)
    value = torch.randn(2, 4, 8, 16, dtype=torch.bfloat16)
    
    input_tensor = torch.randn(32, 64, dtype=torch.bfloat16)
    weight = torch.randn(128, 64, dtype=torch.bfloat16)
    
    # 测试attention tensor保存
    print("\n--- 测试Attention Tensor保存 ---")
    results = save_attention_tensors(
        query=query,
        key=key,
        value=value,
        quant_type="hifp8",
        operation="forward",
        layer_idx=0,
        metadata={"test": "attention_forward"}
    )
    print(f"保存结果: {results}")
    
    # 测试linear tensor保存
    print("\n--- 测试Linear Tensor保存 ---")
    results = save_linear_tensors(
        input_tensor=input_tensor,
        weight=weight,
        quant_type="mxfp8",
        operation="forward",
        layer_idx=1,
        metadata={"test": "linear_forward"}
    )
    print(f"保存结果: {results}")
    
    # 测试backward tensor保存
    print("\n--- 测试Backward Tensor保存 ---")
    grad_output = torch.randn(32, 128, dtype=torch.bfloat16)
    results = save_linear_tensors(
        input_tensor=grad_output,
        weight=weight,
        quant_type="mxfp4",
        operation="backward",
        layer_idx=1,
        metadata={"test": "linear_backward"}
    )
    print(f"保存结果: {results}")


def test_environment_variables():
    """测试环境变量设置"""
    print("\n=== 测试环境变量设置 ===")
    
    # 设置环境变量
    os.environ['TENSOR_SAVE_DIR'] = './test_tensor_logs'
    os.environ['TENSOR_SAVE_ENABLED'] = 'true'
    os.environ['CUSTOM_QUANT_TYPE'] = 'hifp8'
    
    print(f"TENSOR_SAVE_DIR: {os.environ.get('TENSOR_SAVE_DIR')}")
    print(f"TENSOR_SAVE_ENABLED: {os.environ.get('TENSOR_SAVE_ENABLED')}")
    print(f"CUSTOM_QUANT_TYPE: {os.environ.get('CUSTOM_QUANT_TYPE')}")
    
    # 重新导入以使用新的环境变量
    from megatron.core.tensor_saver import get_tensor_saver
    saver = get_tensor_saver()
    print(f"TensorSaver保存目录: {saver.save_dir}")
    print(f"TensorSaver启用状态: {saver.enabled}")


def test_different_quant_types():
    """测试不同量化类型的tensor保存"""
    print("\n=== 测试不同量化类型 ===")
    
    quant_types = ['hifp8', 'mxfp8', 'mxfp4', 'bf16', 'fp16']
    
    for quant_type in quant_types:
        print(f"\n--- 测试量化类型: {quant_type} ---")
        
        # 设置量化类型
        os.environ['CUSTOM_QUANT_TYPE'] = quant_type
        
        # 创建测试tensor
        query = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        key = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        value = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        
        # 保存attention tensor
        results = save_attention_tensors(
            query=query,
            key=key,
            value=value,
            quant_type=quant_type,
            operation="forward",
            layer_idx=0,
            metadata={"quant_type_test": quant_type}
        )
        print(f"量化类型 {quant_type} 保存结果: {len([r for r in results.values() if r is not None])} 个文件")


def load_and_verify_saved_tensors():
    """加载并验证保存的tensor"""
    print("\n=== 验证保存的Tensor ===")
    
    import glob
    from pathlib import Path
    
    # 查找保存的tensor文件
    tensor_files = glob.glob('./tensor_logs/*.pt')
    print(f"找到 {len(tensor_files)} 个保存的tensor文件")
    
    for i, file_path in enumerate(tensor_files[:3]):  # 只验证前3个文件
        print(f"\n--- 验证文件 {i+1}: {Path(file_path).name} ---")
        try:
            data = torch.load(file_path, map_location='cpu')
            print(f"Tensor形状: {data['tensor'].shape}")
            print(f"数据类型: {data['tensor'].dtype}")
            print(f"层类型: {data['metadata']['layer_type']}")
            print(f"操作类型: {data['metadata']['operation']}")
            print(f"量化类型: {data['metadata']['quant_type']}")
            print(f"Tensor名称: {data['metadata']['tensor_name']}")
            print(f"保存时间: {data['metadata']['save_time']}")
            print(f"数值范围: [{data['tensor_info']['min']:.4f}, {data['tensor_info']['max']:.4f}]")
        except Exception as e:
            print(f"加载文件失败: {e}")


if __name__ == "__main__":
    print("开始测试Tensor保存功能...")
    
    # 设置环境变量
    os.environ['TENSOR_SAVE_DIR'] = './tensor_logs'
    os.environ['TENSOR_SAVE_ENABLED'] = 'true'
    
    try:
        # 运行测试
        test_tensor_saver()
        test_environment_variables()
        test_different_quant_types()
        load_and_verify_saved_tensors()
        
        print("\n=== 所有测试完成 ===")
        print("请检查 ./tensor_logs/ 目录中的保存文件")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
