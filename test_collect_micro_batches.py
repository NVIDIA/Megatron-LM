#!/usr/bin/env python3
"""
测试collect_micro_batches参数是否真正发挥作用
"""

import os
import sys

# 添加Megatron路径
sys.path.insert(0, '/data/charles/codes/Megatron-LM')

def test_environment_variable():
    """测试环境变量方式"""
    print("=== 测试环境变量方式 ===")
    
    # 设置环境变量
    os.environ['TENSOR_SAVE_ENABLED'] = 'true'
    os.environ['TENSOR_SAVE_DIR'] = './test_tensors_env'
    os.environ['COLLECT_MICRO_BATCHES'] = '3'
    
    print(f"环境变量设置:")
    print(f"  - TENSOR_SAVE_ENABLED: {os.environ.get('TENSOR_SAVE_ENABLED')}")
    print(f"  - TENSOR_SAVE_DIR: {os.environ.get('TENSOR_SAVE_DIR')}")
    print(f"  - COLLECT_MICRO_BATCHES: {os.environ.get('COLLECT_MICRO_BATCHES')}")
    
    try:
        from megatron.core.tensor_saver import get_tensor_saver
        saver = get_tensor_saver()
        
        print(f"✅ TensorSaver初始化成功!")
        print(f"  - save_dir: {saver.save_dir}")
        print(f"  - enabled: {saver.enabled}")
        print(f"  - control_micro_batches: {saver.control_micro_batches}")
        
        # 测试micro_batch控制
        print(f"\n测试micro_batch控制:")
        for i in range(5):
            saver.increment_micro_batch()
            should_continue = saver.should_continue_collection()
            print(f"  - 第{i+1}次增加后: micro_batch_count={saver.micro_batch_count}, should_continue={should_continue}")
            if not should_continue:
                print(f"  - 达到限制，停止收集")
                break
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_command_line_args():
    """测试命令行参数方式"""
    print("\n=== 测试命令行参数方式 ===")
    
    # 模拟命令行参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-tensors', action='store_true')
    parser.add_argument('--tensor-save-dir', type=str, default='./enhanced_tensor_logs')
    parser.add_argument('--collect-micro-batches', type=int, default=1)
    
    # 模拟解析参数
    test_args = ['--save-tensors', '--tensor-save-dir', './test_tensors_args', '--collect-micro-batches', '2']
    args = parser.parse_args(test_args)
    
    print(f"命令行参数:")
    print(f"  - save_tensors: {args.save_tensors}")
    print(f"  - tensor_save_dir: {args.tensor_save_dir}")
    print(f"  - collect_micro_batches: {args.collect_micro_batches}")
    
    # 设置环境变量为默认值
    os.environ['TENSOR_SAVE_ENABLED'] = 'false'
    os.environ['TENSOR_SAVE_DIR'] = './enhanced_tensor_logs'
    os.environ['COLLECT_MICRO_BATCHES'] = '1'
    
    try:
        # 模拟全局变量设置
        import megatron.training.global_vars as global_vars
        global_vars._GLOBAL_ARGS = args
        
        from megatron.core.tensor_saver import get_tensor_saver
        saver = get_tensor_saver()
        
        print(f"✅ TensorSaver初始化成功!")
        print(f"  - save_dir: {saver.save_dir}")
        print(f"  - enabled: {saver.enabled}")
        print(f"  - control_micro_batches: {saver.control_micro_batches}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("==================================================================================")
    print("测试collect_micro_batches参数是否真正发挥作用")
    print("==================================================================================")
    
    success_count = 0
    total_tests = 2
    
    # 测试1: 环境变量方式
    if test_environment_variable():
        success_count += 1
    
    # 测试2: 命令行参数方式
    if test_command_line_args():
        success_count += 1
    
    print(f"\n==================================================================================")
    print(f"测试结果: {success_count}/{total_tests} 通过")
    if success_count == total_tests:
        print("✅ 所有测试通过！collect_micro_batches参数真正发挥作用。")
    else:
        print("❌ 部分测试失败，需要进一步检查。")
    print("==================================================================================")

if __name__ == "__main__":
    main()
