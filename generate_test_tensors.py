#!/usr/bin/env python3
"""
生成测试tensor数据用于可视化测试
模拟enhanced_tensor_logs目录结构：bf16, mxfp8, mxfp4, hifp8
包含Sample (0,1,2) 和 Layer (1-16) 的数据
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import random
from datetime import datetime

def create_test_directory_structure():
    """创建测试目录结构"""
    base_dir = Path("./test_enhanced_tensor_logs")
    
    # 创建量化类型目录
    quant_types = ['bf16', 'mxfp8', 'mxfp4', 'hifp8']
    for quant_type in quant_types:
        quant_dir = base_dir / quant_type
        quant_dir.mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {quant_dir}")
    
    return base_dir

def generate_tensor_data(quant_type, layer, sample, layer_type, operation, tensor_name):
    """根据量化类型生成模拟tensor数据"""
    
    # 设置随机种子确保可重现性
    random.seed(42 + layer * 100 + sample * 10 + hash(tensor_name) % 1000)
    np.random.seed(42 + layer * 100 + sample * 10 + hash(tensor_name) % 1000)
    
    # 根据量化类型设置数据范围
    if quant_type == 'bf16':
        # bf16: 最大65504, 最小6.104e-05
        min_val, max_val = 1e-4, 1000.0
        normal_range = (6.104e-05, 65504.0)
    elif quant_type == 'hifp8':
        # hifp8: 最大32768, 最小3.052e-05
        min_val, max_val = 1e-4, 500.0
        normal_range = (3.052e-05, 32768.0)
    elif quant_type == 'mxfp8':
        # mxfp8: 最大448, 最小0.015625
        min_val, max_val = 0.01, 50.0
        normal_range = (0.015625, 448.0)
    elif quant_type == 'mxfp4':
        # mxfp4: 最大12, 最小0.25
        min_val, max_val = 0.1, 8.0
        normal_range = (0.25, 12.0)
    else:
        min_val, max_val = 0.001, 100.0
        normal_range = (0.001, 100.0)
    
    # 根据层类型调整数据特征
    if layer_type == 'attention':
        # attention层通常有更大的数值范围
        scale_factor = 2.0
        noise_level = 0.1
    elif layer_type == 'linear':
        # linear层数值相对稳定
        scale_factor = 1.0
        noise_level = 0.05
    else:
        scale_factor = 1.5
        noise_level = 0.08
    
    # 根据操作类型调整数据
    if operation == 'forward':
        # forward pass数据
        base_values = np.random.normal(1.0, 0.5, (32, 64))  # 模拟batch_size=32, hidden_size=64
    elif operation == 'backward':
        # backward pass数据（梯度）
        base_values = np.random.normal(0.0, 0.3, (32, 64))
    else:
        base_values = np.random.normal(0.5, 0.4, (32, 64))
    
    # 应用量化类型特定的缩放
    base_values = base_values * scale_factor
    
    # 添加层相关的特征
    layer_factor = 1.0 + (layer - 8) * 0.1  # 中间层数值稍大
    base_values = base_values * layer_factor
    
    # 添加样本相关的变化
    sample_factor = 0.8 + sample * 0.2  # sample 0,1,2 对应 0.8, 1.0, 1.2
    base_values = base_values * sample_factor
    
    # 确保数值在合理范围内
    base_values = np.clip(base_values, min_val, max_val)
    
    # 添加一些溢出情况用于测试
    overflow_prob = 0.05  # 5%的概率产生溢出
    if random.random() < overflow_prob:
        # 随机选择一些位置产生溢出
        overflow_mask = np.random.random(base_values.shape) < 0.1
        if quant_type in ['bf16', 'hifp8']:
            # 上溢出
            base_values[overflow_mask] = normal_range[1] * (1.5 + random.random())
        elif quant_type == 'mxfp8':
            # 上溢出
            base_values[overflow_mask] = normal_range[1] * (1.2 + random.random())
        elif quant_type == 'mxfp4':
            # 上溢出
            base_values[overflow_mask] = normal_range[1] * (1.1 + random.random())
    
    # 添加一些下溢出情况
    underflow_prob = 0.03  # 3%的概率产生下溢出
    if random.random() < underflow_prob:
        underflow_mask = np.random.random(base_values.shape) < 0.08
        base_values[underflow_mask] = normal_range[0] * (0.1 + random.random() * 0.5)
    
    # 添加一些特殊值
    special_prob = 0.02  # 2%的概率产生特殊值
    if random.random() < special_prob:
        special_mask = np.random.random(base_values.shape) < 0.05
        if quant_type in ['bf16', 'hifp8']:
            # 添加一些NaN和Inf
            base_values[special_mask] = np.nan if random.random() < 0.5 else np.inf
        else:
            # mxfp8和mxfp4不支持NaN/Inf，添加极值
            base_values[special_mask] = normal_range[1] * 2 if random.random() < 0.5 else normal_range[0] * 0.1
    
    return torch.tensor(base_values, dtype=torch.float32)

def generate_filename(timestamp, quant_type, layer, layer_type, operation, tensor_name, sample, rank=0):
    """生成符合命名规范的filename"""
    return f"{timestamp}_iter000_{layer_type}_L{layer}_{operation}_{quant_type}_rank{rank:02d}_sample{sample:03d}_group000_{tensor_name}.pt"

def generate_test_tensors():
    """生成所有测试tensor文件"""
    print("开始生成测试tensor数据...")
    
    # 创建目录结构
    base_dir = create_test_directory_structure()
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 定义参数
    quant_types = ['bf16', 'mxfp8', 'mxfp4', 'hifp8']
    layers = list(range(1, 17))  # 1-16层
    samples = [0, 1, 2]
    layer_types = ['attention', 'linear']
    operations = ['forward', 'backward']
    
    # 定义tensor名称
    attention_tensors = ['query', 'key', 'value', 'attention_weights']
    linear_tensors = ['input', 'output', 'weight']
    
    total_files = 0
    
    for quant_type in quant_types:
        print(f"\n生成 {quant_type} 量化类型数据...")
        quant_dir = base_dir / quant_type
        
        for layer in layers:
            for sample in samples:
                for layer_type in layer_types:
                    for operation in operations:
                        # 选择tensor名称
                        if layer_type == 'attention':
                            tensor_names = attention_tensors
                        else:
                            tensor_names = linear_tensors
                        
                        for tensor_name in tensor_names:
                            # 生成tensor数据
                            tensor_data = generate_tensor_data(
                                quant_type, layer, sample, layer_type, operation, tensor_name
                            )
                            
                            # 生成文件名
                            filename = generate_filename(
                                timestamp, quant_type, layer, layer_type, operation, 
                                tensor_name, sample
                            )
                            
                            # 保存tensor
                            file_path = quant_dir / filename
                            torch.save(tensor_data, file_path)
                            total_files += 1
                            
                            if total_files % 100 == 0:
                                print(f"  已生成 {total_files} 个文件...")
    
    print(f"\n✅ 测试tensor数据生成完成!")
    print(f"总文件数: {total_files}")
    print(f"保存目录: {base_dir}")
    
    # 显示目录结构
    print(f"\n目录结构:")
    for quant_type in quant_types:
        quant_dir = base_dir / quant_type
        file_count = len(list(quant_dir.glob("*.pt")))
        print(f"  {quant_type}/: {file_count} 个文件")
    
    return base_dir

def generate_summary_report(base_dir):
    """生成数据摘要报告"""
    print(f"\n生成数据摘要报告...")
    
    report_path = base_dir / "test_data_summary.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("测试Tensor数据摘要报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("数据生成时间: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("数据目录: {}\n\n".format(base_dir))
        
        quant_types = ['bf16', 'mxfp8', 'mxfp4', 'hifp8']
        
        f.write("量化类型信息:\n")
        f.write("-" * 50 + "\n")
        f.write("bf16: 最大65504, 最小6.104e-05, 指数范围[-14,15]\n")
        f.write("hifp8: 最大32768, 最小3.052e-05, 指数范围[-15,15]\n")
        f.write("mxfp8: 最大448, 最小0.015625, 指数范围[-6,8]\n")
        f.write("mxfp4: 最大12, 最小0.25, 指数范围[-2,3]\n\n")
        
        f.write("数据统计:\n")
        f.write("-" * 50 + "\n")
        total_files = 0
        for quant_type in quant_types:
            quant_dir = base_dir / quant_type
            file_count = len(list(quant_dir.glob("*.pt")))
            total_files += file_count
            f.write(f"{quant_type}: {file_count} 个文件\n")
        
        f.write(f"总计: {total_files} 个文件\n\n")
        
        f.write("数据特征:\n")
        f.write("-" * 50 + "\n")
        f.write("- 层数: 1-16 (16层)\n")
        f.write("- 样本数: 0, 1, 2 (3个样本)\n")
        f.write("- 层类型: attention, linear\n")
        f.write("- 操作类型: forward, backward\n")
        f.write("- 包含溢出情况用于测试\n")
        f.write("- 包含特殊值(NaN/Inf)用于测试\n")
        f.write("- 每个tensor形状: (32, 64)\n\n")
        
        f.write("使用方法:\n")
        f.write("-" * 50 + "\n")
        f.write("1. 运行多线程可视化:\n")
        f.write("   ./run_tensor_draw.sh ./test_enhanced_tensor_logs ./test_draw\n\n")
        f.write("2. 运行溢出检测分析:\n")
        f.write("   ./run_overflow_analysis.sh ./test_enhanced_tensor_logs ./test_draw\n\n")
    
    print(f"摘要报告已保存: {report_path}")

def main():
    print("=" * 80)
    print("测试Tensor数据生成器")
    print("=" * 80)
    
    # 生成测试数据
    base_dir = generate_test_tensors()
    
    # 生成摘要报告
    generate_summary_report(base_dir)
    
    print("\n" + "=" * 80)
    print("测试数据生成完成!")
    print("=" * 80)
    print("现在可以运行以下命令查看可视化效果:")
    print("1. ./run_tensor_draw.sh ./test_enhanced_tensor_logs ./test_draw")
    print("2. ./run_overflow_analysis.sh ./test_enhanced_tensor_logs ./test_draw")
    print("=" * 80)

if __name__ == "__main__":
    main()
