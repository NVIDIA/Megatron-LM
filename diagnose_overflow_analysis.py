#!/usr/bin/env python3
"""
诊断溢出分析问题的工具
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import traceback
from typing import Dict, List, Optional

def diagnose_tensor_files(tensor_dir: str, max_files: int = 100):
    """诊断tensor文件问题"""
    tensor_dir = Path(tensor_dir)
    
    print(f"=== 诊断tensor文件问题 ===")
    print(f"目录: {tensor_dir}")
    
    # 统计文件数量
    total_files = 0
    quant_type_counts = {}
    
    for quant_type in ['bf16', 'mxfp8', 'mxfp4', 'hifp8']:
        quant_dir = tensor_dir / quant_type
        if quant_dir.exists():
            pt_files = list(quant_dir.glob('*.pt'))
            count = len(pt_files)
            quant_type_counts[quant_type] = count
            total_files += count
            print(f"{quant_type}: {count} 个文件")
        else:
            print(f"{quant_type}: 目录不存在")
    
    print(f"总计: {total_files} 个文件")
    
    if total_files == 0:
        print("❌ 没有找到任何tensor文件")
        return
    
    # 测试文件加载
    print(f"\n=== 测试文件加载 ===")
    test_files = []
    for quant_type in ['bf16', 'mxfp8', 'mxfp4', 'hifp8']:
        quant_dir = tensor_dir / quant_type
        if quant_dir.exists():
            pt_files = list(quant_dir.glob('*.pt'))
            if pt_files:
                test_files.append(pt_files[0])
                if len(test_files) >= 4:
                    break
    
    success_count = 0
    error_count = 0
    
    for i, file_path in enumerate(test_files):
        print(f"\n测试文件 {i+1}: {file_path.name}")
        try:
            # 检查文件大小
            file_size = file_path.stat().st_size
            print(f"  文件大小: {file_size / 1024 / 1024:.2f} MB")
            
            if file_size == 0:
                print("  ❌ 文件为空")
                error_count += 1
                continue
            
            # 尝试加载
            tensor = torch.load(file_path, map_location='cpu', weights_only=False)
            print(f"  加载成功，类型: {type(tensor)}")
            
            if isinstance(tensor, dict):
                print(f"  字典键: {list(tensor.keys())}")
                if 'tensor' in tensor:
                    if isinstance(tensor['tensor'], torch.Tensor):
                        print(f"  tensor形状: {tensor['tensor'].shape}")
                        print(f"  tensor数据类型: {tensor['tensor'].dtype}")
                        success_count += 1
                    else:
                        print(f"  ❌ tensor不是torch.Tensor: {type(tensor['tensor'])}")
                        error_count += 1
                else:
                    print(f"  ❌ 没有tensor键")
                    error_count += 1
            elif isinstance(tensor, torch.Tensor):
                print(f"  直接是tensor: {tensor.shape}, {tensor.dtype}")
                success_count += 1
            else:
                print(f"  ❌ 未知类型: {type(tensor)}")
                error_count += 1
                
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            error_count += 1
    
    print(f"\n=== 加载测试结果 ===")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    
    if error_count > 0:
        print("❌ 发现文件加载问题")
    else:
        print("✅ 文件加载正常")

def diagnose_overflow_analysis(tensor_dir: str, max_files: int = 50):
    """诊断溢出分析问题"""
    print(f"\n=== 诊断溢出分析问题 ===")
    
    try:
        sys.path.append('.')
        from script.visualization.overflow_detection_analyzer_improved import OverflowDetectionAnalyzer
        
        # 创建分析器
        analyzer = OverflowDetectionAnalyzer(tensor_dir, './test_overflow_output')
        
        # 获取文件列表
        print("获取文件列表...")
        all_files = analyzer.load_tensor_data()
        print(f"找到 {len(all_files)} 个文件")
        
        if len(all_files) == 0:
            print("❌ 没有找到任何文件")
            return
        
        # 测试单个文件分析
        print("\n测试单个文件分析...")
        test_file = all_files[0]
        print(f"测试文件: {test_file['filename']}")
        
        try:
            result = analyzer.analyze_tensor_file(test_file)
            if 'error' in result:
                print(f"❌ 单个文件分析失败: {result['error']}")
            else:
                print(f"✅ 单个文件分析成功")
                print(f"  总数值: {result.get('total_values', 0)}")
                print(f"  有限数值: {result.get('finite_values', 0)}")
                print(f"  溢出百分比: {result.get('overflow_percentage', 0):.4f}")
        except Exception as e:
            print(f"❌ 单个文件分析异常: {e}")
            traceback.print_exc()
            return
        
        # 测试小批量分析
        print(f"\n测试小批量分析 ({min(max_files, len(all_files))} 个文件)...")
        test_files = all_files[:max_files]
        
        try:
            results = analyzer.analyze_all_tensors(test_files)
            print(f"分析完成，结果数量: {len(results)}")
            
            # 统计结果
            success_count = 0
            error_count = 0
            for result in results:
                if 'error' in result:
                    error_count += 1
                else:
                    success_count += 1
            
            print(f"成功: {success_count}")
            print(f"失败: {error_count}")
            
            if error_count > 0:
                print("\n❌ 发现分析问题，错误示例:")
                for i, result in enumerate(results[:5]):
                    if 'error' in result:
                        print(f"  {i+1}. {result['filename']}: {result['error']}")
            else:
                print("✅ 小批量分析正常")
                
        except Exception as e:
            print(f"❌ 小批量分析异常: {e}")
            traceback.print_exc()
            
    except ImportError as e:
        print(f"❌ 无法导入溢出分析模块: {e}")
    except Exception as e:
        print(f"❌ 诊断过程异常: {e}")
        traceback.print_exc()

def check_memory_usage():
    """检查内存使用情况"""
    print(f"\n=== 检查内存使用情况 ===")
    
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"当前进程内存使用: {memory_mb:.2f} MB")
        
        # 检查系统内存
        system_memory = psutil.virtual_memory()
        print(f"系统总内存: {system_memory.total / 1024 / 1024 / 1024:.2f} GB")
        print(f"系统可用内存: {system_memory.available / 1024 / 1024 / 1024:.2f} GB")
        print(f"系统内存使用率: {system_memory.percent:.1f}%")
        
        if system_memory.percent > 90:
            print("⚠️  系统内存使用率过高，可能导致问题")
        elif system_memory.percent > 80:
            print("⚠️  系统内存使用率较高，需要注意")
        else:
            print("✅ 系统内存使用正常")
            
    except ImportError:
        print("无法检查内存使用情况（需要psutil模块）")
    except Exception as e:
        print(f"内存检查异常: {e}")

def main():
    if len(sys.argv) != 2:
        print("用法: python diagnose_overflow_analysis.py <tensor_directory>")
        sys.exit(1)
    
    tensor_dir = sys.argv[1]
    
    if not os.path.exists(tensor_dir):
        print(f"❌ 目录不存在: {tensor_dir}")
        sys.exit(1)
    
    # 诊断tensor文件
    diagnose_tensor_files(tensor_dir)
    
    # 检查内存使用
    check_memory_usage()
    
    # 诊断溢出分析
    diagnose_overflow_analysis(tensor_dir)
    
    print(f"\n=== 诊断完成 ===")
    print("如果发现问题，请根据上述信息进行修复")

if __name__ == "__main__":
    main()


