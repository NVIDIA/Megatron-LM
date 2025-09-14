#!/usr/bin/env python3
"""
诊断tensor文件加载问题
检查文件完整性、格式和可能的损坏情况
"""

import os
import sys
import torch
import glob
from pathlib import Path
import traceback

def check_tensor_file(file_path):
    """检查单个tensor文件的完整性"""
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return {
                'status': 'missing',
                'error': '文件不存在',
                'size': 0
            }
        
        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return {
                'status': 'empty',
                'error': '文件为空',
                'size': file_size
            }
        
        # 尝试加载tensor
        try:
            tensor = torch.load(file_path, weights_only=False)
            return {
                'status': 'success',
                'error': None,
                'size': file_size,
                'shape': tensor.shape if hasattr(tensor, 'shape') else 'unknown',
                'dtype': str(tensor.dtype) if hasattr(tensor, 'dtype') else 'unknown',
                'min_val': float(tensor.min()) if hasattr(tensor, 'min') else 'unknown',
                'max_val': float(tensor.max()) if hasattr(tensor, 'max') else 'unknown'
            }
        except Exception as e:
            return {
                'status': 'corrupted',
                'error': str(e),
                'size': file_size
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'error': f'检查文件时出错: {str(e)}',
            'size': 0
        }

def find_tensor_files(directory, pattern="*.pt"):
    """查找指定目录下的tensor文件"""
    tensor_files = []
    
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.pt'):
                    tensor_files.append(os.path.join(root, file))
    else:
        print(f"目录不存在: {directory}")
    
    return tensor_files

def diagnose_directory(directory, max_files=50):
    """诊断目录中的tensor文件"""
    print(f"诊断目录: {directory}")
    print("=" * 80)
    
    # 查找tensor文件
    tensor_files = find_tensor_files(directory)
    
    if not tensor_files:
        print("未找到任何.pt文件")
        return
    
    print(f"找到 {len(tensor_files)} 个tensor文件")
    
    # 限制检查的文件数量
    files_to_check = tensor_files[:max_files]
    if len(tensor_files) > max_files:
        print(f"只检查前 {max_files} 个文件")
    
    # 统计结果
    results = {
        'success': 0,
        'missing': 0,
        'empty': 0,
        'corrupted': 0,
        'error': 0
    }
    
    corrupted_files = []
    
    print(f"\n检查 {len(files_to_check)} 个文件...")
    print("-" * 80)
    
    for i, file_path in enumerate(files_to_check, 1):
        print(f"[{i:3d}/{len(files_to_check)}] ", end="")
        
        result = check_tensor_file(file_path)
        results[result['status']] += 1
        
        if result['status'] == 'success':
            print(f"✅ {os.path.basename(file_path)} - 形状: {result['shape']}, 类型: {result['dtype']}")
        elif result['status'] == 'corrupted':
            print(f"❌ {os.path.basename(file_path)} - 损坏: {result['error']}")
            corrupted_files.append((file_path, result['error']))
        elif result['status'] == 'missing':
            print(f"⚠️  {os.path.basename(file_path)} - 文件不存在")
        elif result['status'] == 'empty':
            print(f"⚠️  {os.path.basename(file_path)} - 文件为空 ({result['size']} bytes)")
        else:
            print(f"❌ {os.path.basename(file_path)} - 错误: {result['error']}")
    
    # 打印统计结果
    print("\n" + "=" * 80)
    print("诊断结果统计:")
    print("-" * 40)
    print(f"成功加载: {results['success']} 个文件")
    print(f"文件缺失: {results['missing']} 个文件")
    print(f"文件为空: {results['empty']} 个文件")
    print(f"文件损坏: {results['corrupted']} 个文件")
    print(f"其他错误: {results['error']} 个文件")
    
    if corrupted_files:
        print(f"\n损坏的文件详情:")
        print("-" * 40)
        for file_path, error in corrupted_files[:10]:  # 只显示前10个
            print(f"文件: {os.path.basename(file_path)}")
            print(f"错误: {error}")
            print()
        
        if len(corrupted_files) > 10:
            print(f"... 还有 {len(corrupted_files) - 10} 个损坏的文件")
    
    return results

def check_specific_files(file_patterns):
    """检查特定的文件模式"""
    print("检查特定文件模式...")
    print("=" * 80)
    
    for pattern in file_patterns:
        print(f"\n搜索模式: {pattern}")
        files = glob.glob(pattern)
        
        if not files:
            print("  未找到匹配的文件")
            continue
        
        print(f"  找到 {len(files)} 个文件")
        
        for file_path in files[:5]:  # 只检查前5个
            result = check_tensor_file(file_path)
            print(f"  {os.path.basename(file_path)}: {result['status']}")
            if result['status'] != 'success':
                print(f"    错误: {result['error']}")

def main():
    print("Tensor文件诊断工具")
    print("=" * 80)
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python diagnose_tensor_files.py <目录路径> [文件模式...]")
        print("示例:")
        print("  python diagnose_tensor_files.py ./enhanced_tensor_logs")
        print("  python diagnose_tensor_files.py ./enhanced_tensor_logs '*20250914*'")
        sys.exit(1)
    
    directory = sys.argv[1]
    file_patterns = sys.argv[2:] if len(sys.argv) > 2 else []
    
    # 检查目录
    if not os.path.exists(directory):
        print(f"错误: 目录不存在 - {directory}")
        sys.exit(1)
    
    # 诊断目录
    results = diagnose_directory(directory)
    
    # 检查特定文件模式
    if file_patterns:
        check_specific_files(file_patterns)
    
    print("\n诊断完成!")

if __name__ == "__main__":
    main()
