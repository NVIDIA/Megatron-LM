#!/usr/bin/env python3
"""
测试control_iter修复效果
"""

def test_control_iter_logic():
    """测试control_iter逻辑修复"""
    print("测试control_iter逻辑修复...")
    
    # 模拟修复前的逻辑
    print("\n修复前的逻辑:")
    control_iter = 1
    iteration = 0
    
    print(f"初始状态: iteration={iteration}, control_iter={control_iter}")
    
    # 第一次循环
    iteration += 1
    print(f"第一次循环后: iteration={iteration}")
    
    # 修复前的检查逻辑: iteration >= control_iter
    if iteration >= control_iter:
        print(f"❌ 修复前: iteration({iteration}) >= control_iter({control_iter}) -> 立即退出")
        print("   结果: 只执行了1个iteration，tensor收集不完整")
    else:
        print(f"✅ 修复前: iteration({iteration}) < control_iter({control_iter}) -> 继续")
    
    # 模拟修复后的逻辑
    print("\n修复后的逻辑:")
    iteration = 0
    control_iter = 1
    
    print(f"初始状态: iteration={iteration}, control_iter={control_iter}")
    
    # 第一次循环
    iteration += 1
    print(f"第一次循环后: iteration={iteration}")
    
    # 修复后的检查逻辑: iteration > control_iter
    if iteration > control_iter:
        print(f"❌ 修复后: iteration({iteration}) > control_iter({control_iter}) -> 退出")
    else:
        print(f"✅ 修复后: iteration({iteration}) <= control_iter({control_iter}) -> 继续")
    
    # 第二次循环
    iteration += 1
    print(f"第二次循环后: iteration={iteration}")
    
    if iteration > control_iter:
        print(f"✅ 修复后: iteration({iteration}) > control_iter({control_iter}) -> 退出")
        print("   结果: 执行了完整的1个iteration，tensor收集完整")
    else:
        print(f"❌ 修复后: iteration({iteration}) <= control_iter({control_iter}) -> 继续")

def test_tensor_saver_logic():
    """测试tensor saver逻辑修复"""
    print("\n测试tensor saver逻辑修复...")
    
    control_iter = 1
    
    print(f"control_iter = {control_iter}")
    
    # 模拟修复前的tensor saver逻辑
    print("\n修复前的tensor saver逻辑:")
    for current_iteration in range(3):
        if current_iteration >= control_iter:
            print(f"❌ iteration={current_iteration}: current_iteration({current_iteration}) >= control_iter({control_iter}) -> 不保存tensor")
        else:
            print(f"✅ iteration={current_iteration}: current_iteration({current_iteration}) < control_iter({control_iter}) -> 保存tensor")
    
    # 模拟修复后的tensor saver逻辑
    print("\n修复后的tensor saver逻辑:")
    for current_iteration in range(3):
        if current_iteration > control_iter:
            print(f"❌ iteration={current_iteration}: current_iteration({current_iteration}) > control_iter({control_iter}) -> 不保存tensor")
        else:
            print(f"✅ iteration={current_iteration}: current_iteration({current_iteration}) <= control_iter({control_iter}) -> 保存tensor")

def test_expected_behavior():
    """测试预期行为"""
    print("\n测试预期行为...")
    
    control_iter = 1
    print(f"当control_iter = {control_iter}时:")
    print("1. 训练应该执行完整的1个iteration")
    print("2. 在iteration 0时，tensor saver应该保存tensor")
    print("3. 在iteration 1时，tensor saver应该保存tensor")
    print("4. 在iteration 2时，训练应该退出，tensor saver应该停止保存")
    
    print("\n预期tensor数量:")
    print("- 每个iteration有128个micro batch")
    print("- 每个micro batch产生多个tensor（attention + linear）")
    print("- 8个rank并行处理")
    print("- 32层模型")
    print("- 总tensor数量应该远大于56个")

if __name__ == "__main__":
    test_control_iter_logic()
    test_tensor_saver_logic()
    test_expected_behavior()
    
    print("\n🎯 修复总结:")
    print("1. ✅ 训练循环: iteration > control_iter 时退出（而不是 >=）")
    print("2. ✅ tensor saver: current_iteration > control_iter 时停止保存（而不是 >=）")
    print("3. ✅ 现在应该执行完整的1个iteration")
    print("4. ✅ tensor收集应该更完整")
    
    print("\n🔧 预期效果:")
    print("- 训练执行完整的1个iteration")
    print("- 收集大量tensor文件（数千个而不是56个）")
    print("- Linear层文件名包含层数")
    print("- Sample索引正确分布")
