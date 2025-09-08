# 工具脚本

这个目录包含了各种辅助工具和实用脚本。

## 脚本文件

### 工具脚本
- **`quant_type_modifier.py`** - 量化类型修改工具
- **`update_scripts_with_pattern_v2.py`** - 脚本模式更新工具

## 功能说明

### 1. 量化类型修改工具 (quant_type_modifier.py)

用于批量修改脚本中的量化类型设置。

#### 功能特性
- 批量修改多个脚本文件
- 支持多种量化类型 (hifp8, mxfp8, mxfp4, bf16, fp16)
- 自动备份原文件
- 支持正则表达式匹配

#### 使用方法
```bash
# 修改单个文件
python quant_type_modifier.py \
    --file script/llama32-1b/pretrain_llama32-1b_dolma_bf16.sh \
    --old_quant_type bf16 \
    --new_quant_type hifp8

# 批量修改目录下所有文件
python quant_type_modifier.py \
    --directory script/llama32-1b/ \
    --old_quant_type bf16 \
    --new_quant_type hifp8

# 使用正则表达式匹配
python quant_type_modifier.py \
    --directory script/ \
    --pattern ".*_bf16\.sh$" \
    --new_quant_type hifp8
```

#### 参数说明
- `--file`: 要修改的单个文件路径
- `--directory`: 要修改的目录路径
- `--pattern`: 文件匹配的正则表达式
- `--old_quant_type`: 旧的量化类型
- `--new_quant_type`: 新的量化类型
- `--backup`: 是否创建备份文件 (默认: true)
- `--dry_run`: 预览模式，不实际修改文件

### 2. 脚本模式更新工具 (update_scripts_with_pattern_v2.py)

用于批量更新训练脚本，应用统一的模式。

#### 功能特性
- 自动识别脚本类型和量化类型
- 应用统一的脚本模式
- 支持多种模型类型 (llama32-1b, llama31-8b, deepseek2_lite)
- 自动生成新的脚本文件

#### 使用方法
```bash
# 更新所有脚本
python update_scripts_with_pattern_v2.py

# 更新特定目录
python update_scripts_with_pattern_v2.py \
    --target_dir script/llama32-1b/

# 预览模式
python update_scripts_with_pattern_v2.py \
    --dry_run
```

#### 参数说明
- `--target_dir`: 目标目录路径
- `--template_file`: 模板文件路径
- `--dry_run`: 预览模式，不实际修改文件
- `--backup`: 是否创建备份文件

## 使用场景

### 1. 量化类型切换
当需要批量切换量化类型时：
```bash
# 将所有bf16脚本改为hifp8
python quant_type_modifier.py \
    --directory script/ \
    --old_quant_type bf16 \
    --new_quant_type hifp8
```

### 2. 脚本模式统一
当需要应用新的脚本模式时：
```bash
# 应用新的脚本模式
python update_scripts_with_pattern_v2.py
```

### 3. 批量文件操作
当需要对大量文件进行相同操作时：
```bash
# 批量修改特定模式的文件
python quant_type_modifier.py \
    --directory script/ \
    --pattern ".*_mxfp4\.sh$" \
    --new_quant_type mxfp8
```

## 注意事项

### 1. 备份文件
- 工具会自动创建备份文件
- 备份文件以`.bak`后缀命名
- 建议在批量操作前手动备份重要文件

### 2. 文件权限
- 确保脚本有读取和写入权限
- 使用`chmod +x`设置执行权限

### 3. 正则表达式
- 使用正确的正则表达式语法
- 测试模式匹配是否正确
- 使用`--dry_run`预览结果

### 4. 错误处理
- 检查文件路径是否正确
- 确保目标文件存在
- 查看错误日志定位问题

## 故障排除

### 常见问题
1. **文件不存在**: 检查文件路径是否正确
2. **权限不足**: 使用`chmod`设置正确权限
3. **正则表达式错误**: 检查正则表达式语法
4. **编码问题**: 确保文件编码为UTF-8

### 调试技巧
- 使用`--dry_run`预览操作结果
- 先在小范围文件上测试
- 查看详细的错误信息
- 检查备份文件是否正确创建

## 扩展功能

### 自定义修改规则
可以修改脚本以支持自定义的修改规则：

```python
# 在quant_type_modifier.py中添加自定义规则
def custom_modification(content, old_type, new_type):
    # 自定义修改逻辑
    return modified_content
```

### 批量操作
可以结合使用多个工具进行复杂的批量操作：

```bash
# 1. 先更新脚本模式
python update_scripts_with_pattern_v2.py

# 2. 再修改量化类型
python quant_type_modifier.py \
    --directory script/ \
    --old_quant_type bf16 \
    --new_quant_type hifp8
```
