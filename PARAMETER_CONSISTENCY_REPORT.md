# 参数命名一致性检查报告

## 概述
本次检查全面审查了Megatron-LM中所有tensor相关参数的命名一致性，确保命令行参数与代码中的使用完全匹配。

## 检查的参数

### 1. Tensor收集相关参数
| 命令行参数 | 代码中属性名 | 状态 | 说明 |
|-----------|-------------|------|------|
| `--save-tensors` | `args.save_tensors` | ✅ 正确 | argparse自动转换连字符为下划线 |
| `--tensor-save-dir` | `args.tensor_save_dir` | ✅ 正确 | argparse自动转换连字符为下划线 |
| `--control-iter` | `args.control_iter` | ✅ 正确 | argparse自动转换连字符为下划线 |

### 2. 参数定义位置
- **文件**: `megatron/training/arguments.py`
- **行号**: 1743-1748
- **函数**: `_add_tensor_args()`

```python
group.add_argument('--save-tensors', action='store_true',
                   help='Enable tensor saving for debugging and analysis.')
group.add_argument('--tensor-save-dir', type=str, default='./enhanced_tensor_logs',
                   help='Directory to save tensor logs (default: ./enhanced_tensor_logs)')
group.add_argument('--control-iter', type=int, default=1,
                   help='Number of iterations to collect tensors before stopping (default: 1)')
```

### 3. 参数使用位置

#### 3.1 `save_tensors` 参数
- **文件**: `megatron/training/training.py:2184`
- **文件**: `pretrain_gpt.py:135`
- **文件**: `megatron/core/tensor_saver.py:677`

```python
if getattr(args, 'save_tensors', False):
    # tensor saving logic
```

#### 3.2 `tensor_save_dir` 参数
- **文件**: `megatron/core/tensor_saver.py:676`

```python
save_dir = getattr(args, 'tensor_save_dir', None) or os.environ.get("TENSOR_SAVE_DIR", "./enhanced_tensor_logs")
```

#### 3.3 `control_iter` 参数
- **文件**: `megatron/training/training.py:2261`
- **文件**: `megatron/core/tensor_saver.py:678`

```python
control_iter = getattr(args, 'control_iter', None)
```

### 4. 脚本参数传递

#### 4.1 主收集脚本
- **文件**: `run_wikipedia_tensor_collection.sh`
- **传递方式**: 直接传递给训练脚本

```bash
bash examples/llama/train_llama32_1b_h100_fp8.sh \
    "$checkpoint_path" \
    "$tensorboard_path" \
    "$TOKENIZER_PATH" \
    "$DATA_PATH" \
    "$DTYPE" \
    --control-iter "$CONTROL_ITER" \
    --save-tensors \
    --tensor-save-dir "$tensor_path" \
    2>&1 | tee "${tensorboard_path}/training_${quant_type}_$(date +'%y-%m-%d_%H-%M-%S').log" &
```

#### 4.2 训练脚本
- **文件**: `examples/llama/train_llama32_1b_h100_fp8.sh`
- **处理方式**: 解析并传递给torchrun命令

```bash
while [[ $# -gt 0 ]]; do
    case $1 in
        --control-iter)
            EXTRA_ARGS+=("--control-iter" "$2")
            shift 2
            ;;
        --save-tensors)
            EXTRA_ARGS+=("--save-tensors")
            shift
            ;;
        --tensor-save-dir)
            EXTRA_ARGS+=("--tensor-save-dir" "$2")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done
```

## 检查结果

### ✅ 通过的项目
1. **参数定义正确**: 所有tensor相关参数在`arguments.py`中正确定义
2. **命名转换正确**: argparse自动将连字符转换为下划线
3. **参数访问正确**: 代码中使用`getattr(args, 'param_name', default)`正确访问
4. **脚本传递正确**: 所有脚本都正确传递参数
5. **参数使用正确**: 所有参数在代码中都被正确使用

### 🔧 已修复的问题
1. **control_iter时机问题**: 将检查移到iteration递增之后，确保正确退出
2. **sample_idx跟踪问题**: 添加了正确的sample索引更新逻辑

### 📋 其他参数检查
检查了其他带连字符的参数，发现它们都遵循相同的命名转换规则：
- `--fp8-format` → `args.fp8_format`
- `--transformer-impl` → `args.transformer_impl`
- `--enable-cuda-graph` → `args.enable_cuda_graph`
- 等等...

## 结论

**所有tensor相关参数的命名一致性检查通过！** 

- ✅ 参数定义与使用完全匹配
- ✅ 脚本参数传递正确
- ✅ argparse转换规则正确应用
- ✅ 代码中参数访问方式正确

没有发现任何参数命名不一致的问题。所有tensor收集功能应该能够正常工作。

## 测试验证

创建了以下测试脚本验证参数一致性：
- `test_param_consistency.py`: 验证参数命名转换
- `test_all_tensor_params.py`: 验证参数访问
- `test_control_iter_fixed.py`: 验证control_iter逻辑

所有测试都通过，确认参数系统工作正常。
