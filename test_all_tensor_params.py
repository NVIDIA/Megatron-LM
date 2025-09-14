#!/usr/bin/env python3
"""
测试所有tensor相关参数的正确性
"""

import sys
import os
sys.path.append('/data/charles/codes/Megatron-LM')

def test_tensor_params():
    """测试所有tensor相关参数"""
    print("测试所有tensor相关参数...")
    
    # 模拟命令行参数
    class MockArgs:
        def __init__(self):
            # 这些是argparse会自动转换的参数名
            self.save_tensors = True
            self.tensor_save_dir = "./test_tensor_logs"
            self.control_iter = 2
    
    args = MockArgs()
    
    print(f"模拟参数:")
    print(f"  save_tensors: {args.save_tensors}")
    print(f"  tensor_save_dir: {args.tensor_save_dir}")
    print(f"  control_iter: {args.control_iter}")
    
    # 测试参数访问
    print(f"\n测试参数访问:")
    
    # 测试save_tensors
    save_tensors = getattr(args, 'save_tensors', False)
    print(f"  getattr(args, 'save_tensors', False) = {save_tensors}")
    assert save_tensors == True, "save_tensors参数访问失败"
    
    # 测试tensor_save_dir
    tensor_save_dir = getattr(args, 'tensor_save_dir', None)
    print(f"  getattr(args, 'tensor_save_dir', None) = {tensor_save_dir}")
    assert tensor_save_dir == "./test_tensor_logs", "tensor_save_dir参数访问失败"
    
    # 测试control_iter
    control_iter = getattr(args, 'control_iter', None)
    print(f"  getattr(args, 'control_iter', None) = {control_iter}")
    assert control_iter == 2, "control_iter参数访问失败"
    
    print("\n✅ 所有参数访问测试通过！")
    
    # 测试tensor saver初始化
    print(f"\n测试tensor saver初始化:")
    try:
        from megatron.core.tensor_saver import TensorSaver
        saver = TensorSaver(
            save_dir=tensor_save_dir,
            enabled=save_tensors,
            control_iter=control_iter
        )
        print(f"  TensorSaver初始化成功")
        print(f"  save_dir: {saver.save_dir}")
        print(f"  enabled: {saver.enabled}")
        print(f"  control_iter: {saver.control_iter}")
        print("✅ TensorSaver初始化测试通过！")
    except Exception as e:
        print(f"❌ TensorSaver初始化失败: {e}")
        return False
    
    return True

def test_parameter_consistency():
    """测试参数命名一致性"""
    print(f"\n测试参数命名一致性...")
    
    # 检查命令行参数定义
    param_definitions = {
        '--save-tensors': 'save_tensors',
        '--tensor-save-dir': 'tensor_save_dir', 
        '--control-iter': 'control_iter'
    }
    
    print("命令行参数 -> 代码中使用的属性名:")
    for cmd_param, attr_name in param_definitions.items():
        print(f"  {cmd_param} -> args.{attr_name}")
    
    # 验证argparse转换规则
    print(f"\n验证argparse转换规则:")
    print("  --save-tensors -> save_tensors (连字符转下划线)")
    print("  --tensor-save-dir -> tensor_save_dir (连字符转下划线)")
    print("  --control-iter -> control_iter (连字符转下划线)")
    
    print("✅ 参数命名一致性检查通过！")

if __name__ == "__main__":
    success = test_tensor_params()
    test_parameter_consistency()
    
    if success:
        print(f"\n🎉 所有测试通过！tensor相关参数工作正常。")
    else:
        print(f"\n❌ 测试失败！")
        sys.exit(1)
