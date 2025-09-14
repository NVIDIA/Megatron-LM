#!/usr/bin/env python3
"""
测试tensor文件加载
模拟您遇到的加载错误情况
"""

import os
import sys
import torch
import glob
from pathlib import Path

def test_tensor_loading():
    """测试tensor文件加载"""
    print("测试Tensor文件加载")
    print("=" * 80)
    
    # 查找所有.pt文件
    tensor_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith('.pt'):
                tensor_files.append(os.path.join(root, file))
    
    print(f"找到 {len(tensor_files)} 个tensor文件")
    
    # 测试加载前10个文件
    test_files = tensor_files[:10]
    
    print(f"\n测试加载前 {len(test_files)} 个文件:")
    print("-" * 80)
    
    success_count = 0
    error_count = 0
    
    for i, file_path in enumerate(test_files, 1):
        print(f"[{i:2d}] {os.path.basename(file_path)}: ", end="")
        
        try:
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                print("❌ 文件为空")
                error_count += 1
                continue
            
            # 尝试加载
            tensor = torch.load(file_path, weights_only=False)
            print(f"✅ 成功 - 形状: {tensor.shape}, 类型: {tensor.dtype}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 失败 - {str(e)[:50]}...")
            error_count += 1
    
    print(f"\n测试结果:")
    print(f"  成功: {success_count} 个文件")
    print(f"  失败: {error_count} 个文件")
    
    return success_count, error_count

def create_corrupted_test_files():
    """创建一些损坏的测试文件来模拟错误"""
    print("\n创建损坏的测试文件用于测试...")
    
    test_dir = Path("./corrupted_test")
    test_dir.mkdir(exist_ok=True)
    
    # 创建空文件
    empty_file = test_dir / "empty_file.pt"
    empty_file.touch()
    
    # 创建损坏的文件（写入随机数据）
    corrupted_file = test_dir / "corrupted_file.pt"
    with open(corrupted_file, 'wb') as f:
        f.write(b"corrupted data that is not a valid pytorch file")
    
    # 创建正常文件
    normal_file = test_dir / "normal_file.pt"
    normal_tensor = torch.randn(10, 10)
    torch.save(normal_tensor, normal_file)
    
    print(f"测试文件已创建在: {test_dir}")
    return test_dir

def test_corrupted_files():
    """测试损坏的文件"""
    print("\n测试损坏的文件:")
    print("-" * 40)
    
    test_dir = create_corrupted_test_files()
    
    for file_path in test_dir.glob("*.pt"):
        print(f"测试: {file_path.name}: ", end="")
        
        try:
            tensor = torch.load(file_path, weights_only=False)
            print(f"✅ 成功 - 形状: {tensor.shape}")
        except Exception as e:
            print(f"❌ 失败 - {str(e)[:50]}...")

def main():
    print("Tensor加载测试工具")
    print("=" * 80)
    
    # 测试正常文件
    success, error = test_tensor_loading()
    
    # 测试损坏文件
    test_corrupted_files()
    
    print(f"\n总结:")
    print(f"  正常文件测试: {success} 成功, {error} 失败")
    print(f"  如果看到 'PytorchStreamReader failed reading zip archive' 错误")
    print(f"  说明文件确实损坏了，需要使用修复工具")

if __name__ == "__main__":
    main()
