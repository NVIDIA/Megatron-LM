#!/usr/bin/env python3
"""
修复sample和layer映射问题的工具
分析现有的tensor文件，重新组织sample和layer的对应关系
"""

import os
import sys
import torch
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
from collections import defaultdict

def analyze_tensor_files(tensor_dir: str) -> Dict[str, List[Dict]]:
    """分析tensor文件，按layer和sample分组"""
    tensor_dir = Path(tensor_dir)
    
    # 按layer和sample分组
    layer_sample_groups = defaultdict(lambda: defaultdict(list))
    
    for quant_dir in tensor_dir.iterdir():
        if not quant_dir.is_dir():
            continue
            
        print(f"分析量化目录: {quant_dir.name}")
        
        for pt_file in quant_dir.glob("*.pt"):
            # 解析文件名
            filename = pt_file.name
            parts = filename.split('_')
            
            # 查找layer和sample信息
            layer_idx = None
            sample_idx = None
            
            for part in parts:
                if part.startswith('L') and part[1:].isdigit():
                    layer_idx = int(part[1:])
                elif part.startswith('sample') and part[6:].isdigit():
                    sample_idx = int(part[6:])
            
            if layer_idx is not None and sample_idx is not None:
                # 加载tensor信息
                try:
                    data = torch.load(pt_file, map_location='cpu', weights_only=True)
                    if isinstance(data, dict) and 'tensor' in data:
                        tensor_info = {
                            'filename': filename,
                            'quant_type': quant_dir.name,
                            'layer_idx': layer_idx,
                            'sample_idx': sample_idx,
                            'tensor_name': data.get('metadata', {}).get('tensor_name', 'unknown'),
                            'operation': data.get('metadata', {}).get('operation', 'unknown'),
                            'phase': data.get('metadata', {}).get('phase', 'unknown'),
                            'file_path': pt_file
                        }
                        layer_sample_groups[layer_idx][sample_idx].append(tensor_info)
                except Exception as e:
                    print(f"无法加载文件 {filename}: {e}")
    
    return layer_sample_groups

def analyze_tensor_distribution(layer_sample_groups: Dict) -> None:
    """分析tensor分布，找出问题"""
    print("\n=== Tensor分布分析 ===")
    
    for layer_idx in sorted(layer_sample_groups.keys()):
        print(f"\nLayer {layer_idx}:")
        
        for sample_idx in sorted(layer_sample_groups[layer_idx].keys()):
            tensors = layer_sample_groups[layer_idx][sample_idx]
            tensor_names = [t['tensor_name'] for t in tensors]
            
            print(f"  Sample {sample_idx}: {len(tensors)} tensors")
            print(f"    Tensor types: {set(tensor_names)}")
            
            # 分析tensor类型分布
            type_count = defaultdict(int)
            for tensor in tensors:
                type_count[tensor['tensor_name']] += 1
            
            for tensor_type, count in type_count.items():
                print(f"      {tensor_type}: {count} files")

def detect_mapping_issues(layer_sample_groups: Dict) -> List[Dict]:
    """检测映射问题"""
    issues = []
    
    for layer_idx in sorted(layer_sample_groups.keys()):
        layer_data = layer_sample_groups[layer_idx]
        
        # 检查sample 0和sample 1的tensor类型分布
        if 0 in layer_data and 1 in layer_data:
            sample_0_types = set(t['tensor_name'] for t in layer_data[0])
            sample_1_types = set(t['tensor_name'] for t in layer_data[1])
            
            # 如果sample 0主要是output，sample 1主要是input/weight，说明有问题
            if 'output' in sample_0_types and 'input' in sample_1_types and 'weight' in sample_1_types:
                if len(sample_0_types & {'input', 'weight'}) < len(sample_0_types & {'output'}):
                    issues.append({
                        'layer_idx': layer_idx,
                        'issue_type': 'sample_mapping',
                        'description': f'Layer {layer_idx}: Sample 0 mainly has output tensors, Sample 1 has input/weight tensors',
                        'sample_0_types': sample_0_types,
                        'sample_1_types': sample_1_types
                    })
    
    return issues

def fix_sample_mapping(layer_sample_groups: Dict, output_dir: str) -> None:
    """修复sample映射问题"""
    print("\n=== 修复Sample映射 ===")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for layer_idx in sorted(layer_sample_groups.keys()):
        print(f"\n处理Layer {layer_idx}:")
        
        layer_data = layer_sample_groups[layer_idx]
        
        # 重新组织tensor
        reorganized_tensors = defaultdict(list)
        
        for sample_idx, tensors in layer_data.items():
            for tensor in tensors:
                # 根据tensor类型重新分配sample
                tensor_name = tensor['tensor_name']
                
                if tensor_name in ['input', 'weight']:
                    new_sample = 0  # input和weight放在sample 0
                elif tensor_name in ['output', 'query', 'key', 'value', 'attention_weights', 'weights']:
                    new_sample = 1  # output相关放在sample 1
                else:
                    new_sample = sample_idx  # 其他保持原样
                
                reorganized_tensors[new_sample].append(tensor)
        
        # 保存重新组织的tensor
        for new_sample, tensors in reorganized_tensors.items():
            print(f"  Sample {new_sample}: {len(tensors)} tensors")
            
            for tensor in tensors:
                # 创建新的文件名
                old_filename = tensor['filename']
                new_filename = old_filename.replace(f"sample{tensor['sample_idx']:03d}", f"sample{new_sample:03d}")
                
                # 复制文件到新位置
                old_path = tensor['file_path']
                new_path = output_path / new_filename
                
                shutil.copy2(old_path, new_path)
                print(f"    复制: {old_filename} -> {new_filename}")

def main():
    if len(sys.argv) != 2:
        print("用法: python fix_sample_layer_mapping.py <tensor_directory>")
        sys.exit(1)
    
    tensor_dir = sys.argv[1]
    
    if not os.path.exists(tensor_dir):
        print(f"错误: 目录 {tensor_dir} 不存在")
        sys.exit(1)
    
    print(f"分析tensor目录: {tensor_dir}")
    
    # 分析tensor文件
    layer_sample_groups = analyze_tensor_files(tensor_dir)
    
    # 分析分布
    analyze_tensor_distribution(layer_sample_groups)
    
    # 检测问题
    issues = detect_mapping_issues(layer_sample_groups)
    
    if issues:
        print(f"\n=== 检测到 {len(issues)} 个问题 ===")
        for issue in issues:
            print(f"Layer {issue['layer_idx']}: {issue['description']}")
            print(f"  Sample 0 types: {issue['sample_0_types']}")
            print(f"  Sample 1 types: {issue['sample_1_types']}")
        
        # 询问是否修复
        response = input("\n是否要修复这些问题? (y/n): ")
        if response.lower() == 'y':
            fix_sample_mapping(layer_sample_groups, f"{tensor_dir}_fixed")
            print(f"\n修复完成! 修复后的文件保存在: {tensor_dir}_fixed")
        else:
            print("跳过修复")
    else:
        print("\n未检测到明显的映射问题")

if __name__ == "__main__":
    main()


