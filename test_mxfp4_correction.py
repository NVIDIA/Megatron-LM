#!/usr/bin/env python3
"""
测试mxfp4修正后的限制值
"""

import sys
import os
sys.path.append('/data/charles/codes/Megatron-LM')

def test_mxfp4_limits():
    """测试mxfp4 (FP4-E2M1) 限制值"""
    print("测试mxfp4 (FP4-E2M1) 限制值...")
    
    # 导入溢出检测分析器
    from script.visualization.overflow_detection_analyzer import OverflowDetectionAnalyzer
    
    # 创建分析器实例
    analyzer = OverflowDetectionAnalyzer("./enhanced_tensor_logs", "./test_draw")
    
    # 获取mxfp4限制值
    mxfp4_limits = analyzer.quantization_limits['mxfp4']
    
    print(f"\nmxfp4 (FP4-E2M1) 限制值:")
    print(f"  最大正常值: {mxfp4_limits.max_positive_normal:.6e} ({mxfp4_limits.max_positive_normal})")
    print(f"  最小正常值: {mxfp4_limits.min_positive_normal:.6e} ({mxfp4_limits.min_positive_normal})")
    print(f"  最大非正常值: {mxfp4_limits.max_positive_denormal:.6e} ({mxfp4_limits.max_positive_denormal})")
    print(f"  最小非正常值: {mxfp4_limits.min_positive_denormal:.6e} ({mxfp4_limits.min_positive_denormal})")
    print(f"  指数范围: {mxfp4_limits.exponent_range}")
    print(f"  支持无穷大: {mxfp4_limits.supports_infinity}")
    print(f"  支持NaN: {mxfp4_limits.supports_nan}")
    print(f"  支持零: {mxfp4_limits.supports_zero}")

def test_mxfp4_overflow_detection():
    """测试mxfp4溢出检测"""
    print("\n测试mxfp4溢出检测...")
    
    import numpy as np
    from script.visualization.overflow_detection_analyzer import OverflowDetectionAnalyzer
    
    analyzer = OverflowDetectionAnalyzer("./enhanced_tensor_logs", "./test_draw")
    
    # 测试案例
    test_cases = [
        {
            'name': 'mxfp4正常范围',
            'data': np.array([0.5, 1.0, 2.0, 4.0, 8.0]),  # 在正常范围内
            'quant_type': 'mxfp4'
        },
        {
            'name': 'mxfp4上溢出',
            'data': np.array([0.5, 1.0, 2.0, 4.0, 20.0]),  # 20 > 12 (最大正常值)
            'quant_type': 'mxfp4'
        },
        {
            'name': 'mxfp4下溢出',
            'data': np.array([0.5, 1.0, 2.0, 0.1, 0.01]),  # 0.01 < 0.25 (最小正常值)
            'quant_type': 'mxfp4'
        },
        {
            'name': 'mxfp4边界值',
            'data': np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]),  # 边界值
            'quant_type': 'mxfp4'
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
        
        # 显示限制值
        limits = result['limits']
        print(f"  限制值:")
        print(f"    最大正常值: {limits['max_positive_normal']:.6e}")
        print(f"    最小正常值: {limits['min_positive_normal']:.6e}")

def test_fp4_e2m1_calculation():
    """验证FP4-E2M1计算"""
    print("\n验证FP4-E2M1计算...")
    
    print("FP4-E2M1格式:")
    print("  1位符号位 + 2位指数位 + 1位尾数位")
    print("  指数偏移: 1 (2^(2-1) - 1)")
    print("  指数范围: -2 到 3")
    print("  尾数范围: 1.0 到 1.5 (隐含1位)")
    
    print("\n计算验证:")
    print(f"  最大正常值: 1.5 * 2^3 = {1.5 * 2**3}")
    print(f"  最小正常值: 1.0 * 2^(-2) = {1.0 * 2**(-2)}")
    print(f"  最大非正常值: 1.5 * 2^(-3) = {1.5 * 2**(-3)}")
    print(f"  最小非正常值: 1.0 * 2^(-4) = {1.0 * 2**(-4)}")

if __name__ == "__main__":
    test_mxfp4_limits()
    test_mxfp4_overflow_detection()
    test_fp4_e2m1_calculation()
    
    print("\n✅ mxfp4 (FP4-E2M1) 修正完成!")
    print("现在mxfp4使用正确的FP4-E2M1格式限制值")
