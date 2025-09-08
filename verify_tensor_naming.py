#!/usr/bin/env python3
"""
验证改进的tensor命名功能
确保所有修改都正确工作
"""

import os
import sys
import torch

def verify_tensor_saver():
    """验证tensor_saver模块"""
    print("=== 验证TensorSaver模块 ===")
    
    try:
        from megatron.core.tensor_saver import TensorSaver, save_tensor, save_attention_tensors, save_linear_tensors
        print("✅ 成功导入所有tensor_saver函数")
        
        # 测试TensorSaver类
        saver = TensorSaver(save_dir="./test_logs", enabled=True)
        print("✅ 成功创建TensorSaver实例")
        
        # 测试文件名生成
        filename = saver._generate_filename(
            layer_type="attention",
            operation="forward",
            quant_type="hifp8",
            tensor_name="query",
            layer_idx=0,
            phase="pre",
            component="FA"
        )
        print(f"✅ 文件名生成测试: {filename}")
        
        # 验证文件名格式
        expected_parts = ["attention", "L0", "forward", "pre", "FA", "hifp8", "query"]
        for part in expected_parts:
            if part in filename:
                print(f"  ✅ 包含 {part}")
            else:
                print(f"  ❌ 缺少 {part}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ TensorSaver验证失败: {e}")
        return False

def verify_attention_layer():
    """验证attention层修改"""
    print("\n=== 验证Attention层修改 ===")
    
    try:
        # 检查文件是否存在
        attention_file = "/data/charles/Megatron-LM/megatron/core/transformer/dot_product_attention.py"
        if not os.path.exists(attention_file):
            print(f"❌ 文件不存在: {attention_file}")
            return False
        
        # 读取文件内容
        with open(attention_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键修改
        checks = [
            ('phase="pre"', 'pre阶段参数'),
            ('component="FA"', 'FA组件参数'),
            ('phase="post"', 'post阶段参数'),
            ('save_attention_tensors', 'attention tensor保存函数'),
            ('save_tensor', 'tensor保存函数')
        ]
        
        for check, description in checks:
            if check in content:
                print(f"  ✅ {description}: 找到 {check}")
            else:
                print(f"  ❌ {description}: 未找到 {check}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Attention层验证失败: {e}")
        return False

def verify_linear_layer():
    """验证linear层修改"""
    print("\n=== 验证Linear层修改 ===")
    
    try:
        # 检查文件是否存在
        linear_file = "/data/charles/Megatron-LM/megatron/core/tensor_parallel/layers.py"
        if not os.path.exists(linear_file):
            print(f"❌ 文件不存在: {linear_file}")
            return False
        
        # 读取文件内容
        with open(linear_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键修改
        checks = [
            ('phase="pre"', 'pre阶段参数'),
            ('component="linear"', 'linear组件参数'),
            ('phase="post"', 'post阶段参数'),
            ('save_linear_tensors', 'linear tensor保存函数'),
            ('save_tensor', 'tensor保存函数')
        ]
        
        for check, description in checks:
            if check in content:
                print(f"  ✅ {description}: 找到 {check}")
            else:
                print(f"  ❌ {description}: 未找到 {check}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Linear层验证失败: {e}")
        return False

def verify_function_signatures():
    """验证函数签名"""
    print("\n=== 验证函数签名 ===")
    
    try:
        from megatron.core.tensor_saver import save_tensor, save_attention_tensors, save_linear_tensors
        import inspect
        
        # 检查save_tensor函数签名
        sig = inspect.signature(save_tensor)
        params = list(sig.parameters.keys())
        expected_params = ['tensor', 'layer_type', 'operation', 'quant_type', 'tensor_name', 'layer_idx', 'phase', 'component', 'metadata']
        
        for param in expected_params:
            if param in params:
                print(f"  ✅ save_tensor包含参数: {param}")
            else:
                print(f"  ❌ save_tensor缺少参数: {param}")
                return False
        
        # 检查save_attention_tensors函数签名
        sig = inspect.signature(save_attention_tensors)
        params = list(sig.parameters.keys())
        expected_params = ['query', 'key', 'value', 'quant_type', 'operation', 'layer_idx', 'phase', 'component', 'metadata']
        
        for param in expected_params:
            if param in params:
                print(f"  ✅ save_attention_tensors包含参数: {param}")
            else:
                print(f"  ❌ save_attention_tensors缺少参数: {param}")
                return False
        
        # 检查save_linear_tensors函数签名
        sig = inspect.signature(save_linear_tensors)
        params = list(sig.parameters.keys())
        expected_params = ['input_tensor', 'weight', 'quant_type', 'operation', 'layer_idx', 'phase', 'component', 'metadata']
        
        for param in expected_params:
            if param in params:
                print(f"  ✅ save_linear_tensors包含参数: {param}")
            else:
                print(f"  ❌ save_linear_tensors缺少参数: {param}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 函数签名验证失败: {e}")
        return False

def main():
    """主验证函数"""
    print("开始验证改进的tensor命名功能...")
    
    # 设置环境变量
    os.environ['CUSTOM_QUANT_TYPE'] = 'hifp8'
    os.environ['TENSOR_SAVE_DIR'] = './test_logs'
    os.environ['TENSOR_SAVE_ENABLED'] = 'true'
    
    # 运行验证
    results = []
    results.append(verify_tensor_saver())
    results.append(verify_attention_layer())
    results.append(verify_linear_layer())
    results.append(verify_function_signatures())
    
    # 总结结果
    print("\n=== 验证结果总结 ===")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 所有验证通过! ({passed}/{total})")
        print("✅ 改进的tensor命名功能已正确实现")
        return True
    else:
        print(f"❌ 验证失败! ({passed}/{total})")
        print("请检查上述失败的验证项")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
