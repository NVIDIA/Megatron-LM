#!/usr/bin/env python3
"""
测试修复后的control_iter逻辑
"""

def test_control_iter_fixed():
    """测试修复后的control_iter退出逻辑"""
    print("测试修复后的control_iter退出逻辑...")
    
    # 测试不同的control_iter值
    test_cases = [
        (1, "应该执行1个iteration (0)"),
        (2, "应该执行2个iteration (0, 1)"),
        (3, "应该执行3个iteration (0, 1, 2)"),
    ]
    
    for control_iter, description in test_cases:
        print(f"\n=== 测试 control_iter={control_iter} ===")
        print(f"期望: {description}")
        
        train_iters = 100
        iteration = 0
        executed_iterations = []
        
        while iteration < train_iters:
            print(f"  开始 iteration {iteration}")
            executed_iterations.append(iteration)
            
            # 模拟tensor saver更新
            print(f"  [Training] 更新tensor saver iteration为: {iteration}")
            
            # 模拟训练步骤
            print(f"  [Training] 执行训练步骤 {iteration}")
            
            # 模拟iteration递增
            iteration += 1
            print(f"  [Training] 完成iteration {iteration-1}")
            
            # 检查control_iter限制（在iteration递增之后）
            if control_iter is not None and iteration >= control_iter:
                print(f"  [Training] 达到control_iter限制 ({control_iter}), 退出训练...")
                break
        
        print(f"  实际执行了 {len(executed_iterations)} 个iteration: {executed_iterations}")
        
        # 验证结果
        expected_count = control_iter
        actual_count = len(executed_iterations)
        if actual_count == expected_count:
            print(f"  ✅ 正确: 执行了 {actual_count} 个iteration")
        else:
            print(f"  ❌ 错误: 期望 {expected_count} 个iteration，实际 {actual_count} 个")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_control_iter_fixed()
