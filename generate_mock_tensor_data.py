#!/usr/bin/env python3
"""
生成模拟tensor数据用于测试可视化效果
"""

import os
import sys
import torch
import numpy as np
import tempfile
from pathlib import Path
import json
import time

# 添加项目路径
sys.path.append('/data/charles/codes/Megatron-LM')

def create_mock_tensor_data():
    """创建模拟的tensor数据"""
    print("=== 创建模拟tensor数据 ===")
    
    # 创建输出目录
    enhanced_logs_dir = Path("/data/charles/codes/Megatron-LM/enhanced_tensor_logs")
    draw_dir = Path("/data/charles/codes/Megatron-LM/draw")
    
    enhanced_logs_dir.mkdir(exist_ok=True)
    draw_dir.mkdir(exist_ok=True)
    
    # 创建子目录
    subdirs = [
        "overflow_analysis", "fp8_analysis", "bf16_analysis", 
        "backward_analysis", "rank_analysis"
    ]
    
    for subdir in subdirs:
        (enhanced_logs_dir / subdir).mkdir(exist_ok=True)
        (draw_dir / subdir).mkdir(exist_ok=True)
    
    # 模拟不同层和操作的数据
    layers = [
        {"type": "linear", "idx": 0, "operation": "forward"},
        {"type": "linear", "idx": 1, "operation": "forward"},
        {"type": "linear", "idx": 0, "operation": "backward"},
        {"type": "attention", "idx": 0, "operation": "forward"},
        {"type": "attention", "idx": 1, "operation": "forward"},
    ]
    
    quant_types = ["hifp8", "mxfp8", "mxfp4", "bf16", "fp16", "fp32"]
    
    # 生成模拟数据
    for layer in layers:
        for quant_type in quant_types:
            for rank in range(2):  # 模拟2个GPU
                for sample_idx in range(3):  # 模拟3个样本
                    for iteration in range(2):  # 模拟2个iteration
                        
                        # 生成不同形状的tensor
                        if layer["type"] == "linear":
                            if layer["operation"] == "forward":
                                shapes = [
                                    (32, 1024),  # input
                                    (1024, 4096),  # weight
                                    (32, 4096),   # output
                                ]
                                tensor_names = ["input", "weight", "output"]
                            else:  # backward
                                shapes = [
                                    (32, 4096),  # grad_output
                                    (1024, 4096),  # grad_weight
                                    (32, 1024),   # grad_input
                                ]
                                tensor_names = ["grad_output", "grad_weight", "grad_input"]
                        else:  # attention
                            shapes = [
                                (32, 8, 128, 64),  # query
                                (32, 8, 128, 64),  # key
                                (32, 8, 128, 64),  # value
                                (32, 8, 128, 128), # attention_weights
                                (32, 8, 128, 64),  # output
                            ]
                            tensor_names = ["query", "key", "value", "attention_weights", "output"]
                        
                        # 为每个tensor生成数据
                        for i, (shape, tensor_name) in enumerate(zip(shapes, tensor_names)):
                            # 根据量化类型生成不同的数据分布
                            if quant_type in ["hifp8", "mxfp8"]:
                                # FP8数据：较小的数值范围
                                data = torch.randn(shape) * 0.1
                            elif quant_type == "mxfp4":
                                # FP4数据：更小的数值范围
                                data = torch.randn(shape) * 0.05
                            elif quant_type == "bf16":
                                # BF16数据：中等数值范围，可能有特殊分布
                                data = torch.randn(shape) * 0.5
                                # 添加一些特殊值
                                if np.random.random() < 0.1:
                                    data = data * 10  # 偶尔有较大值
                            elif quant_type == "fp16":
                                # FP16数据：标准范围
                                data = torch.randn(shape) * 1.0
                            else:  # fp32, fp64
                                # 高精度数据：较大范围
                                data = torch.randn(shape) * 2.0
                            
                            # 添加一些溢出值用于测试
                            if quant_type in ["fp16", "bf16"] and np.random.random() < 0.05:
                                # 5%的概率添加溢出值
                                overflow_indices = torch.randperm(data.numel())[:data.numel()//100]
                                data.view(-1)[overflow_indices] = torch.randn(len(overflow_indices)) * 1000
                            
                            # 生成文件名
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            counter = np.random.randint(1000, 9999)
                            group_idx = layer["idx"]  # 同一层使用相同的group_idx
                            
                            filename = f"{timestamp}_{counter:04d}_iter{iteration:03d}_{layer['type']}_L{layer['idx']}_{layer['operation']}_post_{layer['type']}_{quant_type}_rank{rank:02d}_sample{sample_idx:03d}_group{group_idx:03d}_{tensor_name}.pt"
                            
                            # 创建元数据
                            metadata = {
                                "tensor_shape": list(shape),
                                "quant_type": quant_type,
                                "layer_type": layer["type"],
                                "operation": layer["operation"],
                                "layer_idx": layer["idx"],
                                "rank": rank,
                                "sample_idx": sample_idx,
                                "iteration": iteration,
                                "group_idx": group_idx,
                                "tensor_name": tensor_name,
                                "phase": "post",
                                "component": layer["type"],
                                "overflow_info": {
                                    "upper_overflow_count": np.random.randint(0, 100),
                                    "lower_overflow_count": np.random.randint(0, 50),
                                    "upper_overflow_ratio": np.random.random() * 0.1,
                                    "lower_overflow_ratio": np.random.random() * 0.05,
                                    "total_overflow_ratio": np.random.random() * 0.15,
                                },
                                "statistics": {
                                    "mean": float(data.mean()),
                                    "std": float(data.std()),
                                    "min": float(data.min()),
                                    "max": float(data.max()),
                                    "skewness": float(torch.tensor(np.random.random() * 2 - 1)),
                                    "kurtosis": float(torch.tensor(np.random.random() * 4 + 1)),
                                }
                            }
                            
                            # 保存tensor数据
                            tensor_data = {
                                "tensor": data,
                                "metadata": metadata
                            }
                            
                            filepath = enhanced_logs_dir / filename
                            torch.save(tensor_data, filepath)
                            
                            print(f"生成: {filename}")
    
    print(f"模拟数据已生成到: {enhanced_logs_dir}")
    return enhanced_logs_dir

def generate_visualization_results():
    """生成可视化结果"""
    print("\n=== 生成可视化结果 ===")
    
    # 导入可视化模块
    try:
        from script.visualization.enhanced_tensor_visualizer import EnhancedTensorVisualizer
        
        # 创建可视化器
        visualizer = EnhancedTensorVisualizer()
        
        # 运行所有可视化
        print("运行tensor可视化...")
        visualizer.run_visualization()
        
        print("可视化结果已生成到: /data/charles/codes/Megatron-LM/draw")
        
    except ImportError as e:
        print(f"无法导入可视化模块: {e}")
        print("请确保enhanced_tensor_visualizer.py存在且可导入")
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")

def create_sample_analysis_files():
    """创建示例分析文件"""
    print("\n=== 创建示例分析文件 ===")
    
    enhanced_logs_dir = Path("/data/charles/codes/Megatron-LM/enhanced_tensor_logs")
    
    # 创建tensor_analysis_summary.json
    summary = {
        "total_tensors": 0,
        "by_layer_type": {},
        "by_quant_type": {},
        "by_operation": {},
        "overflow_statistics": {
            "total_overflow_tensors": 0,
            "overflow_by_quant_type": {},
            "overflow_by_layer_type": {}
        },
        "file_list": []
    }
    
    # 统计文件
    for file_path in enhanced_logs_dir.rglob("*.pt"):
        try:
            data = torch.load(file_path, map_location='cpu')
            metadata = data['metadata']
            
            summary["total_tensors"] += 1
            
            # 按层类型统计
            layer_type = metadata.get('layer_type', 'unknown')
            summary["by_layer_type"][layer_type] = summary["by_layer_type"].get(layer_type, 0) + 1
            
            # 按量化类型统计
            quant_type = metadata.get('quant_type', 'unknown')
            summary["by_quant_type"][quant_type] = summary["by_quant_type"].get(quant_type, 0) + 1
            
            # 按操作类型统计
            operation = metadata.get('operation', 'unknown')
            summary["by_operation"][operation] = summary["by_operation"].get(operation, 0) + 1
            
            # 溢出统计
            overflow_info = metadata.get('overflow_info', {})
            if overflow_info.get('total_overflow_ratio', 0) > 0:
                summary["overflow_statistics"]["total_overflow_tensors"] += 1
                summary["overflow_statistics"]["overflow_by_quant_type"][quant_type] = \
                    summary["overflow_statistics"]["overflow_by_quant_type"].get(quant_type, 0) + 1
                summary["overflow_statistics"]["overflow_by_layer_type"][layer_type] = \
                    summary["overflow_statistics"]["overflow_by_layer_type"].get(layer_type, 0) + 1
            
            summary["file_list"].append({
                "filename": file_path.name,
                "layer_type": layer_type,
                "quant_type": quant_type,
                "operation": operation,
                "rank": metadata.get('rank', 0),
                "sample_idx": metadata.get('sample_idx', 0),
                "iteration": metadata.get('iteration', 0),
                "group_idx": metadata.get('group_idx', 0),
                "tensor_name": metadata.get('tensor_name', 'unknown'),
                "has_overflow": overflow_info.get('total_overflow_ratio', 0) > 0
            })
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    # 保存统计文件
    with open(enhanced_logs_dir / "tensor_analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"分析摘要已保存到: {enhanced_logs_dir / 'tensor_analysis_summary.json'}")
    print(f"总共生成了 {summary['total_tensors']} 个tensor文件")

def main():
    """主函数"""
    print("开始生成模拟tensor数据和可视化结果...")
    
    # 1. 创建模拟数据
    enhanced_logs_dir = create_mock_tensor_data()
    
    # 2. 创建分析文件
    create_sample_analysis_files()
    
    # 3. 生成可视化结果
    generate_visualization_results()
    
    print("\n=== 生成完成 ===")
    print("请查看以下目录:")
    print(f"- 模拟数据: {enhanced_logs_dir}")
    print("- 可视化结果: /data/charles/codes/Megatron-LM/draw")
    print("- 分析摘要: {enhanced_logs_dir}/tensor_analysis_summary.json")

if __name__ == "__main__":
    main()
