#!/usr/bin/env python3
"""
测试Python代码中的参数支持
"""

import sys
import os

# 添加Megatron路径
sys.path.insert(0, '/data/charles/codes/Megatron-LM')

def test_arguments_parsing():
    """测试参数解析"""
    print("=== 测试参数解析 ===")
    
    try:
        from megatron.training.arguments import parse_args
        from megatron.training.global_vars import set_global_variables
        
        # 模拟命令行参数
        test_args = [
            '--save-tensors',
            '--tensor-save-dir', './test_tensors',
            '--control-iter', '3',
            '--collect-micro-batches', '2'
        ]
        
        print(f"测试参数: {test_args}")
        
        # 解析参数
        args = parse_args(test_args)
        
        print(f"✅ 参数解析成功!")
        print(f"  - save_tensors: {getattr(args, 'save_tensors', 'Not found')}")
        print(f"  - tensor_save_dir: {getattr(args, 'tensor_save_dir', 'Not found')}")
        print(f"  - control_iter: {getattr(args, 'control_iter', 'Not found')}")
        print(f"  - collect_micro_batches: {getattr(args, 'collect_micro_batches', 'Not found')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 参数解析失败: {e}")
        return False

def test_tensor_saver_initialization():
    """测试TensorSaver初始化"""
    print("\n=== 测试TensorSaver初始化 ===")
    
    try:
        from megatron.core.tensor_saver import get_tensor_saver
        
        # 设置环境变量
        os.environ['TENSOR_SAVE_ENABLED'] = 'true'
        os.environ['TENSOR_SAVE_DIR'] = './test_tensors'
        os.environ['COLLECT_MICRO_BATCHES'] = '2'
        
        print("环境变量设置:")
        print(f"  - TENSOR_SAVE_ENABLED: {os.environ.get('TENSOR_SAVE_ENABLED')}")
        print(f"  - TENSOR_SAVE_DIR: {os.environ.get('TENSOR_SAVE_DIR')}")
        print(f"  - COLLECT_MICRO_BATCHES: {os.environ.get('COLLECT_MICRO_BATCHES')}")
        
        # 获取tensor saver
        saver = get_tensor_saver()
        
        print(f"✅ TensorSaver初始化成功!")
        print(f"  - save_dir: {saver.save_dir}")
        print(f"  - enabled: {saver.enabled}")
        print(f"  - control_micro_batches: {saver.control_micro_batches}")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorSaver初始化失败: {e}")
        return False

def test_micro_batch_control():
    """测试micro_batch控制功能"""
    print("\n=== 测试micro_batch控制功能 ===")
    
    try:
        from megatron.core.tensor_saver import get_tensor_saver
        
        # 设置环境变量
        os.environ['TENSOR_SAVE_ENABLED'] = 'true'
        os.environ['TENSOR_SAVE_DIR'] = './test_tensors'
        os.environ['COLLECT_MICRO_BATCHES'] = '2'
        
        saver = get_tensor_saver()
        
        print(f"初始状态:")
        print(f"  - micro_batch_count: {saver.micro_batch_count}")
        print(f"  - control_micro_batches: {saver.control_micro_batches}")
        print(f"  - should_continue_collection: {saver.should_continue_collection()}")
        
        # 模拟增加micro_batch
        print(f"\n模拟增加micro_batch:")
        for i in range(3):
            saver.increment_micro_batch()
            print(f"  - 第{i+1}次增加后: micro_batch_count={saver.micro_batch_count}, should_continue={saver.should_continue_collection()}")
        
        return True
        
    except Exception as e:
        print(f"❌ micro_batch控制测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("==================================================================================")
    print("测试Python代码中的参数支持")
    print("==================================================================================")
    
    success_count = 0
    total_tests = 3
    
    # 测试1: 参数解析
    if test_arguments_parsing():
        success_count += 1
    
    # 测试2: TensorSaver初始化
    if test_tensor_saver_initialization():
        success_count += 1
    
    # 测试3: micro_batch控制
    if test_micro_batch_control():
        success_count += 1
    
    print(f"\n==================================================================================")
    print(f"测试结果: {success_count}/{total_tests} 通过")
    if success_count == total_tests:
        print("✅ 所有测试通过！Python代码中的参数支持正常工作。")
    else:
        print("❌ 部分测试失败，需要检查代码。")
    print("==================================================================================")

if __name__ == "__main__":
    main()
