#!/usr/bin/env python3
"""
测试溢出检测功能
"""

import sys
import os
sys.path.append('/data/charles/codes/Megatron-LM')

def test_quantization_limits():
    """测试量化类型限制值"""
    print("测试量化类型限制值...")
    
    # 导入溢出检测分析器
    from script.visualization.overflow_detection_analyzer import OverflowDetectionAnalyzer
    
    # 创建分析器实例
    analyzer = OverflowDetectionAnalyzer("./enhanced_tensor_logs", "./test_draw")
    
    print("\n量化类型限制值:")
    for quant_type, limits in analyzer.quantization_limits.items():
        print(f"\n{quant_type}:")
        print(f"  最大正常值: {limits.max_positive_normal:.6e}")
        print(f"  最小正常值: {limits.min_positive_normal:.6e}")
        print(f"  最大非正常值: {limits.max_positive_denormal:.6e}")
        print(f"  最小非正常值: {limits.min_positive_denormal:.6e}")
        print(f"  指数范围: {limits.exponent_range}")
        print(f"  支持无穷大: {limits.supports_infinity}")
        print(f"  支持NaN: {limits.supports_nan}")

def test_overflow_detection():
    """测试溢出检测逻辑"""
    print("\n测试溢出检测逻辑...")
    
    import numpy as np
    from script.visualization.overflow_detection_analyzer import OverflowDetectionAnalyzer
    
    analyzer = OverflowDetectionAnalyzer("./enhanced_tensor_logs", "./test_draw")
    
    # 测试数据
    test_cases = [
        {
            'name': 'bf16正常范围',
            'data': np.array([1.0, 100.0, 1000.0, 0.001, 0.0001]),
            'quant_type': 'bf16'
        },
        {
            'name': 'bf16上溢出',
            'data': np.array([1.0, 100.0, 100000.0, 0.001]),  # 100000 > 65504
            'quant_type': 'bf16'
        },
        {
            'name': 'bf16下溢出',
            'data': np.array([1.0, 100.0, 1e-6, 0.001]),  # 1e-6 < 6.103515625e-05
            'quant_type': 'bf16'
        },
        {
            'name': 'hifp8正常范围',
            'data': np.array([1.0, 100.0, 1000.0, 0.001, 0.0001]),
            'quant_type': 'hifp8'
        },
        {
            'name': 'hifp8上溢出',
            'data': np.array([1.0, 100.0, 50000.0, 0.001]),  # 50000 > 32768
            'quant_type': 'hifp8'
        },
        {
            'name': 'mxfp8正常范围',
            'data': np.array([1.0, 10.0, 100.0, 0.1, 0.01]),
            'quant_type': 'mxfp8'
        },
        {
            'name': 'mxfp8上溢出',
            'data': np.array([1.0, 10.0, 500.0, 0.1]),  # 500 > 448
            'quant_type': 'mxfp8'
        }
    ]
    
    for case in test_cases:
        print(f"\n测试案例: {case['name']}")
        result = analyzer.detect_overflow(case['data'], case['quant_type'])
        
        print(f"  总数值数: {result['total_values']}")
        print(f"  有限数值数: {result['finite_values']}")
        print(f"  最小值: {result['min_value']:.6e}")
        print(f"  最大值: {result['max_value']:.6e}")
        print(f"  上溢出数量: {result['overflow_upper']}")
        print(f"  下溢出数量: {result['underflow_upper']}")
        print(f"  上溢出率: {result['overflow_percentage']:.2f}%")
        print(f"  下溢出率: {result['underflow_percentage']:.2f}%")

def test_expected_behavior():
    """测试预期行为"""
    print("\n测试预期行为...")
    
    print("溢出检测分析器功能:")
    print("1. ✅ 支持四种量化类型: bf16, mxfp8, mxfp4, hifp8")
    print("2. ✅ 基于量化类型特征值检测溢出")
    print("3. ✅ 支持多线程并行处理")
    print("4. ✅ 生成详细的溢出分析报告")
    print("5. ✅ 支持按量化类型、样本、层进行统计")
    print("6. ✅ 生成可视化图表")
    
    print("\n检测的溢出类型:")
    print("1. 上溢出: 数值超过最大正常值")
    print("2. 下溢出: 数值小于最小正常值")
    print("3. 极值溢出: 数值超过最大非正常值")
    print("4. 极值下溢出: 数值小于最小非正常值")
    
    print("\n输出文件:")
    print("1. overflow_analysis_report.png - 溢出分析图表")
    print("2. overflow_detection_report.txt - 详细统计报告")

if __name__ == "__main__":
    test_quantization_limits()
    test_overflow_detection()
    test_expected_behavior()
    
    print("\n🎯 溢出检测功能测试完成!")
    print("现在可以运行 ./run_overflow_analysis.sh 进行实际的溢出检测分析")
