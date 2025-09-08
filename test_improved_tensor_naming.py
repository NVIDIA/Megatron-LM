#!/usr/bin/env python3
"""
测试改进的tensor命名功能
验证新的phase和component参数是否正确工作
"""

import os
import torch
import sys
sys.path.append('/data/charles/Megatron-LM')

def test_improved_tensor_naming():
    """测试改进的tensor命名功能"""
    print("=== 测试改进的Tensor命名功能 ===")
    
    # 设置环境变量
    os.environ['CUSTOM_QUANT_TYPE'] = 'hifp8'
    os.environ['TENSOR_SAVE_DIR'] = './improved_tensor_logs'
    os.environ['TENSOR_SAVE_ENABLED'] = 'true'
    
    print(f"量化类型: {os.environ.get('CUSTOM_QUANT_TYPE')}")
    print(f"保存目录: {os.environ.get('TENSOR_SAVE_DIR')}")
    
    # 测试attention层tensor保存
    print("\n--- 测试Attention层Tensor保存 (新命名) ---")
    from megatron.core.tensor_saver import save_attention_tensors, save_tensor
    
    # 模拟attention输入tensor
    batch_size, num_heads, seq_len, hidden_size = 2, 8, 128, 64
    query = torch.randn(batch_size, num_heads, seq_len, hidden_size, dtype=torch.bfloat16)
    key = torch.randn(batch_size, num_heads, seq_len, hidden_size, dtype=torch.bfloat16)
    value = torch.randn(batch_size, num_heads, seq_len, hidden_size, dtype=torch.bfloat16)
    
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    
    # 保存attention输入tensor (pre-FA)
    results = save_attention_tensors(
        query=query,
        key=key,
        value=value,
        quant_type=os.environ.get('CUSTOM_QUANT_TYPE', 'hifp8'),
        operation="forward",
        layer_idx=0,
        phase="pre",
        component="FA",
        metadata={
            "batch_size": batch_size,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "test": "attention_forward_input_pre_FA"
        }
    )
    print(f"保存pre-FA输入tensor: {len([r for r in results.values() if r is not None])} 个文件")
    
    # 模拟attention输出tensor
    output_shape = (seq_len, batch_size, hidden_size)
    attention_output = torch.randn(output_shape, dtype=torch.bfloat16)
    
    # 保存attention输出tensor (post-FA)
    output_result = save_tensor(
        tensor=attention_output,
        layer_type="attention",
        operation="forward",
        quant_type=os.environ.get('CUSTOM_QUANT_TYPE', 'hifp8'),
        tensor_name="output",
        layer_idx=0,
        phase="post",
        component="FA",
        metadata={
            "batch_size": batch_size,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "test": "attention_forward_output_post_FA"
        }
    )
    print(f"保存post-FA输出tensor: {output_result}")
    
    # 测试linear层tensor保存
    print("\n--- 测试Linear层Tensor保存 (新命名) ---")
    from megatron.core.tensor_saver import save_linear_tensors
    
    # 模拟linear输入tensor
    batch_size, input_size, output_size = 32, 512, 1024
    input_tensor = torch.randn(batch_size, input_size, dtype=torch.bfloat16)
    weight = torch.randn(output_size, input_size, dtype=torch.bfloat16)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Weight shape: {weight.shape}")
    
    # 保存linear输入tensor (pre-linear)
    results = save_linear_tensors(
        input_tensor=input_tensor,
        weight=weight,
        quant_type=os.environ.get('CUSTOM_QUANT_TYPE', 'hifp8'),
        operation="forward",
        layer_idx=1,
        phase="pre",
        component="linear",
        metadata={
            "batch_size": batch_size,
            "input_size": input_size,
            "output_size": output_size,
            "test": "linear_forward_input_pre_linear"
        }
    )
    print(f"保存pre-linear输入tensor: {len([r for r in results.values() if r is not None])} 个文件")
    
    # 模拟linear输出tensor
    linear_output = torch.randn(batch_size, output_size, dtype=torch.bfloat16)
    
    # 保存linear输出tensor (post-linear)
    output_result = save_tensor(
        tensor=linear_output,
        layer_type="linear",
        operation="forward",
        quant_type=os.environ.get('CUSTOM_QUANT_TYPE', 'hifp8'),
        tensor_name="output",
        layer_idx=1,
        phase="post",
        component="linear",
        metadata={
            "batch_size": batch_size,
            "input_size": input_size,
            "output_size": output_size,
            "test": "linear_forward_output_post_linear"
        }
    )
    print(f"保存post-linear输出tensor: {output_result}")
    
    # 测试backward tensor保存
    print("\n--- 测试Backward Tensor保存 (新命名) ---")
    
    # 模拟backward输入tensor
    grad_output = torch.randn(batch_size, output_size, dtype=torch.bfloat16)
    
    # 保存backward输入tensor (pre-linear)
    results = save_linear_tensors(
        input_tensor=grad_output,
        weight=weight,
        quant_type=os.environ.get('CUSTOM_QUANT_TYPE', 'hifp8'),
        operation="backward",
        layer_idx=1,
        phase="pre",
        component="linear",
        metadata={
            "batch_size": batch_size,
            "output_size": output_size,
            "input_size": input_size,
            "test": "linear_backward_input_pre_linear"
        }
    )
    print(f"保存pre-linear backward输入tensor: {len([r for r in results.values() if r is not None])} 个文件")
    
    # 模拟backward输出tensor
    grad_input = torch.randn(batch_size, input_size, dtype=torch.bfloat16)
    
    # 保存backward输出tensor (post-linear)
    output_result = save_tensor(
        tensor=grad_input,
        layer_type="linear",
        operation="backward",
        quant_type=os.environ.get('CUSTOM_QUANT_TYPE', 'hifp8'),
        tensor_name="output",
        layer_idx=1,
        phase="post",
        component="linear",
        metadata={
            "batch_size": batch_size,
            "input_size": input_size,
            "output_size": output_size,
            "test": "linear_backward_output_post_linear"
        }
    )
    print(f"保存post-linear backward输出tensor: {output_result}")
    
    # 测试不同量化类型
    print("\n--- 测试不同量化类型 (新命名) ---")
    quant_types = ['hifp8', 'mxfp8', 'mxfp4', 'bf16']
    
    for quant_type in quant_types:
        print(f"\n测试量化类型: {quant_type}")
        os.environ['CUSTOM_QUANT_TYPE'] = quant_type
        
        # 创建测试tensor
        test_tensor = torch.randn(1, 4, 32, 16, dtype=torch.bfloat16)
        
        # 保存tensor (pre-FA)
        result = save_tensor(
            tensor=test_tensor,
            layer_type="attention",
            operation="forward",
            quant_type=quant_type,
            tensor_name="query",
            layer_idx=0,
            phase="pre",
            component="FA",
            metadata={"quant_type_test": quant_type, "phase": "pre", "component": "FA"}
        )
        print(f"  保存pre-FA结果: {result is not None}")
        
        # 保存tensor (post-FA)
        result = save_tensor(
            tensor=test_tensor,
            layer_type="attention",
            operation="forward",
            quant_type=quant_type,
            tensor_name="output",
            layer_idx=0,
            phase="post",
            component="FA",
            metadata={"quant_type_test": quant_type, "phase": "post", "component": "FA"}
        )
        print(f"  保存post-FA结果: {result is not None}")
        
        # 保存tensor (pre-linear)
        result = save_tensor(
            tensor=test_tensor,
            layer_type="linear",
            operation="forward",
            quant_type=quant_type,
            tensor_name="input",
            layer_idx=0,
            phase="pre",
            component="linear",
            metadata={"quant_type_test": quant_type, "phase": "pre", "component": "linear"}
        )
        print(f"  保存pre-linear结果: {result is not None}")
        
        # 保存tensor (post-linear)
        result = save_tensor(
            tensor=test_tensor,
            layer_type="linear",
            operation="forward",
            quant_type=quant_type,
            tensor_name="output",
            layer_idx=0,
            phase="post",
            component="linear",
            metadata={"quant_type_test": quant_type, "phase": "post", "component": "linear"}
        )
        print(f"  保存post-linear结果: {result is not None}")
    
    print("\n=== 测试完成 ===")


def analyze_saved_tensors():
    """分析保存的tensor"""
    print("\n=== 分析保存的Tensor (新命名) ===")
    
    import glob
    from pathlib import Path
    
    # 查找保存的tensor文件
    tensor_dir = os.environ.get('TENSOR_SAVE_DIR', './improved_tensor_logs')
    tensor_files = glob.glob(f"{tensor_dir}/*.pt")
    
    print(f"找到 {len(tensor_files)} 个保存的tensor文件")
    
    # 按类型统计
    pre_count = 0
    post_count = 0
    fa_count = 0
    linear_count = 0
    forward_count = 0
    backward_count = 0
    attention_count = 0
    
    for file_path in tensor_files:
        try:
            data = torch.load(file_path, map_location='cpu')
            metadata = data['metadata']
            
            if metadata.get('phase') == 'pre':
                pre_count += 1
            if metadata.get('phase') == 'post':
                post_count += 1
            if metadata.get('component') == 'FA':
                fa_count += 1
            if metadata.get('component') == 'linear':
                linear_count += 1
            if metadata['operation'] == 'forward':
                forward_count += 1
            if metadata['operation'] == 'backward':
                backward_count += 1
            if metadata['layer_type'] == 'attention':
                attention_count += 1
                
        except Exception as e:
            print(f"分析文件失败 {file_path}: {e}")
    
    print(f"\n统计结果:")
    print(f"  Pre阶段tensor: {pre_count} 个")
    print(f"  Post阶段tensor: {post_count} 个")
    print(f"  FA组件tensor: {fa_count} 个")
    print(f"  Linear组件tensor: {linear_count} 个")
    print(f"  Forward操作: {forward_count} 个")
    print(f"  Backward操作: {backward_count} 个")
    print(f"  Attention层: {attention_count} 个")
    
    # 显示前几个文件的信息
    print(f"\n前10个文件信息:")
    for i, file_path in enumerate(tensor_files[:10]):
        try:
            data = torch.load(file_path, map_location='cpu')
            metadata = data['metadata']
            tensor_info = data['tensor_info']
            
            print(f"  {i+1}. {Path(file_path).name}")
            print(f"     层类型: {metadata['layer_type']}")
            print(f"     操作: {metadata['operation']}")
            print(f"     阶段: {metadata.get('phase', 'unknown')}")
            print(f"     组件: {metadata.get('component', 'unknown')}")
            print(f"     Tensor名称: {metadata['tensor_name']}")
            print(f"     量化类型: {metadata['quant_type']}")
            print(f"     形状: {tensor_info['shape']}")
            print(f"     数值范围: [{tensor_info['min']:.4f}, {tensor_info['max']:.4f}]")
            
        except Exception as e:
            print(f"  {i+1}. 加载失败: {e}")


def main():
    """主函数"""
    print("开始测试改进的Tensor命名功能...")
    
    try:
        # 运行测试
        test_improved_tensor_naming()
        
        # 分析保存的tensor
        analyze_saved_tensors()
        
        print("\n=== 所有测试完成 ===")
        print("请检查保存目录中的tensor文件")
        print(f"保存目录: {os.environ.get('TENSOR_SAVE_DIR')}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
