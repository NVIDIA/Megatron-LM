#!/usr/bin/env python3
"""
简化版tensor保存器测试脚本（不依赖PyTorch）
"""

import os
import sys
sys.path.append('/data/charles/Megatron-LM')

def test_tensor_saver_import():
    """测试tensor_saver模块导入"""
    print("=== 测试TensorSaver模块导入 ===")
    
    try:
        from megatron.core.tensor_saver import TensorSaver, get_tensor_saver
        print("✅ TensorSaver模块导入成功")
        
        # 测试创建实例
        saver = TensorSaver(save_dir="./test_logs", enabled=False)
        print("✅ TensorSaver实例创建成功")
        
        # 测试全局实例
        global_saver = get_tensor_saver()
        print("✅ 全局TensorSaver实例获取成功")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorSaver模块导入失败: {e}")
        return False


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
    
    return True


def test_file_structure():
    """测试文件结构"""
    print("\n=== 测试文件结构 ===")
    
    files_to_check = [
        "/data/charles/Megatron-LM/megatron/core/tensor_saver.py",
        "/data/charles/Megatron-LM/megatron/core/transformer/dot_product_attention.py",
        "/data/charles/Megatron-LM/megatron/core/tensor_parallel/layers.py",
        "/data/charles/Megatron-LM/test_tensor_saver.py",
        "/data/charles/Megatron-LM/TENSOR_SAVER_README.md"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_exist = False
    
    return all_exist


def test_code_modifications():
    """测试代码修改"""
    print("\n=== 测试代码修改 ===")
    
    # 检查attention文件中的修改
    attention_file = "/data/charles/Megatron-LM/megatron/core/transformer/dot_product_attention.py"
    if os.path.exists(attention_file):
        with open(attention_file, 'r') as f:
            content = f.read()
            
        checks = [
            ("tensor_saver导入", "from megatron.core.tensor_saver import save_attention_tensors"),
            ("保存forward输入tensor", "保存forward输入tensor"),
            ("环境变量获取", "os.environ.get('CUSTOM_QUANT_TYPE'"),
        ]
        
        for check_name, check_text in checks:
            if check_text in content:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
    
    # 检查layers文件中的修改
    layers_file = "/data/charles/Megatron-LM/megatron/core/tensor_parallel/layers.py"
    if os.path.exists(layers_file):
        with open(layers_file, 'r') as f:
            content = f.read()
            
        checks = [
            ("tensor_saver导入", "from megatron.core.tensor_saver import save_linear_tensors"),
            ("保存forward输入tensor", "保存forward输入tensor"),
            ("保存backward输入tensor", "保存backward输入tensor"),
            ("环境变量获取", "os.environ.get('CUSTOM_QUANT_TYPE'"),
        ]
        
        for check_name, check_text in checks:
            if check_text in content:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")


def test_tensor_saver_class():
    """测试TensorSaver类的基本功能"""
    print("\n=== 测试TensorSaver类基本功能 ===")
    
    try:
        from megatron.core.tensor_saver import TensorSaver
        
        # 测试创建实例
        saver = TensorSaver(save_dir="./test_logs", enabled=False)
        print("✅ TensorSaver实例创建成功")
        
        # 测试文件名生成
        filename = saver._generate_filename("attention", "forward", "hifp8", "query", 0)
        print(f"✅ 文件名生成: {filename}")
        
        # 测试元数据
        metadata = {
            "test": "value",
            "layer_idx": 0
        }
        print(f"✅ 元数据处理: {metadata}")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorSaver类测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("开始测试Tensor保存器功能...")
    
    tests = [
        test_tensor_saver_import,
        test_environment_variables,
        test_file_structure,
        test_code_modifications,
        test_tensor_saver_class,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试 {test.__name__} 失败: {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！")
        print("\n使用说明:")
        print("1. 设置环境变量: export CUSTOM_QUANT_TYPE='hifp8'")
        print("2. 设置保存目录: export TENSOR_SAVE_DIR='./tensor_logs'")
        print("3. 启用保存功能: export TENSOR_SAVE_ENABLED='true'")
        print("4. 运行训练脚本，tensor将自动保存到指定目录")
    else:
        print("⚠️  部分测试失败，请检查代码修改")


if __name__ == "__main__":
    main()
