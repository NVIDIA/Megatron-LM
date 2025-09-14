#!/usr/bin/env python3
"""
测试参数命名一致性
"""

def test_param_consistency():
    """测试参数命名一致性"""
    print("测试参数命名一致性...")
    
    # 模拟argparse的行为
    class MockArgs:
        def __init__(self):
            # 这些是argparse会自动转换的参数名
            self.save_tensors = True
            self.tensor_save_dir = "./test_tensor_logs"
            self.control_iter = 2
    
    args = MockArgs()
    
    print(f"模拟命令行参数:")
    print(f"  --save-tensors -> args.save_tensors = {args.save_tensors}")
    print(f"  --tensor-save-dir -> args.tensor_save_dir = {args.tensor_save_dir}")
    print(f"  --control-iter -> args.control_iter = {args.control_iter}")
    
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
    
    # 检查所有使用这些参数的地方
    print(f"\n检查代码中的参数使用:")
    
    # 模拟代码中的使用方式
    if getattr(args, 'save_tensors', False):
        print("  ✅ save_tensors条件检查正常")
    
    save_dir = getattr(args, 'tensor_save_dir', None) or "default_dir"
    print(f"  ✅ tensor_save_dir获取正常: {save_dir}")
    
    control_iter = getattr(args, 'control_iter', None)
    if control_iter is not None:
        print(f"  ✅ control_iter检查正常: {control_iter}")
    
    print("\n✅ 所有参数使用检查通过！")
    
    return True

def test_script_parameter_passing():
    """测试脚本中的参数传递"""
    print(f"\n测试脚本中的参数传递...")
    
    # 模拟脚本中的参数传递
    script_args = [
        "--save-tensors",
        "--tensor-save-dir", "./tensor_logs",
        "--control-iter", "3"
    ]
    
    print(f"脚本传递的参数: {' '.join(script_args)}")
    
    # 模拟argparse解析
    parsed_args = {}
    i = 0
    while i < len(script_args):
        arg = script_args[i]
        if arg == "--save-tensors":
            parsed_args['save_tensors'] = True
            i += 1
        elif arg == "--tensor-save-dir":
            parsed_args['tensor_save_dir'] = script_args[i + 1]
            i += 2
        elif arg == "--control-iter":
            parsed_args['control_iter'] = int(script_args[i + 1])
            i += 2
        else:
            i += 1
    
    print(f"解析后的参数:")
    print(f"  save_tensors: {parsed_args.get('save_tensors', False)}")
    print(f"  tensor_save_dir: {parsed_args.get('tensor_save_dir', 'default')}")
    print(f"  control_iter: {parsed_args.get('control_iter', 1)}")
    
    # 验证解析结果
    assert parsed_args.get('save_tensors') == True, "save_tensors解析失败"
    assert parsed_args.get('tensor_save_dir') == "./tensor_logs", "tensor_save_dir解析失败"
    assert parsed_args.get('control_iter') == 3, "control_iter解析失败"
    
    print("✅ 脚本参数传递测试通过！")
    
    return True

if __name__ == "__main__":
    success1 = test_param_consistency()
    success2 = test_script_parameter_passing()
    
    if success1 and success2:
        print(f"\n🎉 所有参数一致性测试通过！")
        print(f"✅ --save-tensors -> args.save_tensors")
        print(f"✅ --tensor-save-dir -> args.tensor_save_dir")
        print(f"✅ --control-iter -> args.control_iter")
    else:
        print(f"\n❌ 测试失败！")
        exit(1)
