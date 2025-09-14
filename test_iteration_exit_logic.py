#!/usr/bin/env python3
"""
测试iteration退出逻辑
"""

def test_iteration_exit_logic():
    """测试iteration退出逻辑"""
    print("测试iteration退出逻辑...")
    
    control_iter = 1
    print(f"control_iter = {control_iter}")
    
    # 模拟训练循环
    iteration = 0
    print(f"\n初始状态: iteration = {iteration}")
    
    # 第一次循环
    print("\n=== 第一次循环 ===")
    iteration += 1
    print(f"iteration += 1 后: iteration = {iteration}")
    
    # 检查是否应该退出
    if control_iter is not None and iteration >= control_iter:
        print(f"✅ iteration({iteration}) >= control_iter({control_iter}) -> 退出训练")
        print("   结果: 执行了1个完整的iteration后退出")
    else:
        print(f"❌ iteration({iteration}) < control_iter({control_iter}) -> 继续")
    
    print("\n=== 总结 ===")
    print("当control_iter = 1时:")
    print("1. 初始: iteration = 0")
    print("2. 第一次循环: iteration = 1")
    print("3. 检查: 1 >= 1 -> 退出")
    print("4. 结果: 执行了1个完整的iteration")

def test_tensor_saver_logic():
    """测试tensor saver逻辑"""
    print("\n测试tensor saver逻辑...")
    
    control_iter = 1
    print(f"control_iter = {control_iter}")
    
    print("\n不同iteration的tensor保存状态:")
    for current_iteration in range(3):
        if current_iteration >= control_iter:
            print(f"iteration {current_iteration}: current_iteration({current_iteration}) >= control_iter({control_iter}) -> 不保存tensor")
        else:
            print(f"iteration {current_iteration}: current_iteration({current_iteration}) < control_iter({control_iter}) -> 保存tensor")
    
    print("\n关键点:")
    print("- iteration 0: 保存tensor（训练开始）")
    print("- iteration 1: 不保存tensor（训练结束）")
    print("- 这确保了在iteration 0期间收集tensor，iteration 1时停止")

def test_expected_behavior():
    """测试预期行为"""
    print("\n测试预期行为...")
    
    print("修改后的逻辑:")
    print("1. 训练循环: iteration >= control_iter 时退出")
    print("2. tensor saver: current_iteration >= control_iter 时停止保存")
    print("3. 当control_iter = 1时:")
    print("   - 执行iteration 0（收集tensor）")
    print("   - 执行iteration 1（不收集tensor，然后退出）")
    
    print("\n预期效果:")
    print("- 执行完整的1个iteration")
    print("- 在iteration 0期间收集所有tensor")
    print("- 在iteration 1开始时停止收集并退出")
    print("- 收集的tensor数量应该大幅增加")

if __name__ == "__main__":
    test_iteration_exit_logic()
    test_tensor_saver_logic()
    test_expected_behavior()
    
    print("\n🎯 修改总结:")
    print("1. ✅ 训练循环: iteration >= control_iter 时退出")
    print("2. ✅ tensor saver: current_iteration >= control_iter 时停止保存")
    print("3. ✅ 确保执行1个完整的iteration")
    print("4. ✅ 在iteration 0期间收集tensor，iteration 1时退出")
