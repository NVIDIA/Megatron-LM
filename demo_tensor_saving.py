#!/usr/bin/env python3
"""
演示如何在训练脚本中使用tensor保存功能
"""

import os
import torch
import sys
sys.path.append('/data/charles/Megatron-LM')

def setup_tensor_saving_environment():
    """设置tensor保存环境"""
    print("=== 设置Tensor保存环境 ===")
    
    # 设置环境变量
    os.environ['CUSTOM_QUANT_TYPE'] = 'hifp8'  # 设置量化类型
    os.environ['TENSOR_SAVE_DIR'] = './demo_tensor_logs'  # 设置保存目录
    os.environ['TENSOR_SAVE_ENABLED'] = 'true'  # 启用保存功能
    
    print(f"量化类型: {os.environ.get('CUSTOM_QUANT_TYPE')}")
    print(f"保存目录: {os.environ.get('TENSOR_SAVE_DIR')}")
    print(f"保存启用: {os.environ.get('TENSOR_SAVE_ENABLED')}")


def simulate_attention_forward():
    """模拟attention层的forward过程"""
    print("\n=== 模拟Attention Forward过程 ===")
    
    # 模拟attention输入tensor
    batch_size, num_heads, seq_len, hidden_size = 2, 8, 128, 64
    
    query = torch.randn(batch_size, num_heads, seq_len, hidden_size, dtype=torch.bfloat16)
    key = torch.randn(batch_size, num_heads, seq_len, hidden_size, dtype=torch.bfloat16)
    value = torch.randn(batch_size, num_heads, seq_len, hidden_size, dtype=torch.bfloat16)
    
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    
    # 导入tensor保存功能
    from megatron.core.tensor_saver import save_attention_tensors
    
    # 保存attention tensor
    results = save_attention_tensors(
        query=query,
        key=key,
        value=value,
        quant_type=os.environ.get('CUSTOM_QUANT_TYPE', 'hifp8'),
        operation="forward",
        layer_idx=0,
        metadata={
            "batch_size": batch_size,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "simulation": "attention_forward"
        }
    )
    
    print(f"保存结果: {len([r for r in results.values() if r is not None])} 个文件")
    return results


def simulate_linear_forward():
    """模拟linear层的forward过程"""
    print("\n=== 模拟Linear Forward过程 ===")
    
    # 模拟linear输入tensor
    batch_size, input_size, output_size = 32, 512, 1024
    
    input_tensor = torch.randn(batch_size, input_size, dtype=torch.bfloat16)
    weight = torch.randn(output_size, input_size, dtype=torch.bfloat16)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Weight shape: {weight.shape}")
    
    # 导入tensor保存功能
    from megatron.core.tensor_saver import save_linear_tensors
    
    # 保存linear tensor
    results = save_linear_tensors(
        input_tensor=input_tensor,
        weight=weight,
        quant_type=os.environ.get('CUSTOM_QUANT_TYPE', 'hifp8'),
        operation="forward",
        layer_idx=1,
        metadata={
            "batch_size": batch_size,
            "input_size": input_size,
            "output_size": output_size,
            "simulation": "linear_forward"
        }
    )
    
    print(f"保存结果: {len([r for r in results.values() if r is not None])} 个文件")
    return results


def simulate_linear_backward():
    """模拟linear层的backward过程"""
    print("\n=== 模拟Linear Backward过程 ===")
    
    # 模拟backward输入tensor
    batch_size, output_size, input_size = 32, 1024, 512
    
    grad_output = torch.randn(batch_size, output_size, dtype=torch.bfloat16)
    weight = torch.randn(output_size, input_size, dtype=torch.bfloat16)
    
    print(f"Grad output shape: {grad_output.shape}")
    print(f"Weight shape: {weight.shape}")
    
    # 导入tensor保存功能
    from megatron.core.tensor_saver import save_linear_tensors
    
    # 保存backward tensor
    results = save_linear_tensors(
        input_tensor=grad_output,
        weight=weight,
        quant_type=os.environ.get('CUSTOM_QUANT_TYPE', 'hifp8'),
        operation="backward",
        layer_idx=1,
        metadata={
            "batch_size": batch_size,
            "output_size": output_size,
            "input_size": input_size,
            "simulation": "linear_backward"
        }
    )
    
    print(f"保存结果: {len([r for r in results.values() if r is not None])} 个文件")
    return results


def test_different_quant_types():
    """测试不同量化类型"""
    print("\n=== 测试不同量化类型 ===")
    
    quant_types = ['hifp8', 'mxfp8', 'mxfp4', 'bf16']
    
    for quant_type in quant_types:
        print(f"\n--- 测试量化类型: {quant_type} ---")
        
        # 设置量化类型
        os.environ['CUSTOM_QUANT_TYPE'] = quant_type
        
        # 创建测试tensor
        query = torch.randn(1, 4, 32, 16, dtype=torch.bfloat16)
        key = torch.randn(1, 4, 32, 16, dtype=torch.bfloat16)
        value = torch.randn(1, 4, 32, 16, dtype=torch.bfloat16)
        
        # 保存tensor
        from megatron.core.tensor_saver import save_attention_tensors
        results = save_attention_tensors(
            query=query,
            key=key,
            value=value,
            quant_type=quant_type,
            operation="forward",
            layer_idx=0,
            metadata={"quant_type_test": quant_type}
        )
        
        print(f"量化类型 {quant_type}: 保存了 {len([r for r in results.values() if r is not None])} 个文件")


def analyze_saved_tensors():
    """分析保存的tensor"""
    print("\n=== 分析保存的Tensor ===")
    
    import glob
    from pathlib import Path
    
    # 查找保存的tensor文件
    tensor_dir = os.environ.get('TENSOR_SAVE_DIR', './demo_tensor_logs')
    tensor_files = glob.glob(f"{tensor_dir}/*.pt")
    
    print(f"找到 {len(tensor_files)} 个保存的tensor文件")
    
    # 按量化类型分组统计
    quant_type_stats = {}
    layer_type_stats = {}
    operation_stats = {}
    
    for file_path in tensor_files:
        try:
            data = torch.load(file_path, map_location='cpu')
            metadata = data['metadata']
            
            quant_type = metadata['quant_type']
            layer_type = metadata['layer_type']
            operation = metadata['operation']
            
            quant_type_stats[quant_type] = quant_type_stats.get(quant_type, 0) + 1
            layer_type_stats[layer_type] = layer_type_stats.get(layer_type, 0) + 1
            operation_stats[operation] = operation_stats.get(operation, 0) + 1
            
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
    
    print(f"\n按量化类型统计:")
    for quant_type, count in quant_type_stats.items():
        print(f"  {quant_type}: {count} 个文件")
    
    print(f"\n按层类型统计:")
    for layer_type, count in layer_type_stats.items():
        print(f"  {layer_type}: {count} 个文件")
    
    print(f"\n按操作类型统计:")
    for operation, count in operation_stats.items():
        print(f"  {operation}: {count} 个文件")


def main():
    """主函数"""
    print("开始演示Tensor保存功能...")
    
    try:
        # 设置环境
        setup_tensor_saving_environment()
        
        # 模拟不同的训练过程
        simulate_attention_forward()
        simulate_linear_forward()
        simulate_linear_backward()
        
        # 测试不同量化类型
        test_different_quant_types()
        
        # 分析保存的tensor
        analyze_saved_tensors()
        
        print("\n=== 演示完成 ===")
        print("请检查保存目录中的tensor文件")
        print(f"保存目录: {os.environ.get('TENSOR_SAVE_DIR')}")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
