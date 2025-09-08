# 模板文件

这个目录包含了各种脚本模板文件。

## 模板文件

### 脚本模板
- **`improved_script_template.sh`** - 改进的训练脚本模板

## 模板说明

### 1. 改进的训练脚本模板 (improved_script_template.sh)

这是一个功能完整的训练脚本模板，包含以下特性：

#### 核心功能
- **环境变量管理**: 自动设置和导出必要的环境变量
- **量化类型控制**: 动态修改量化类型设置
- **日志记录**: 带时间戳的详细日志记录
- **错误处理**: 完善的错误检查和处理机制
- **参数验证**: 输入参数的验证和默认值设置

#### 主要特性
1. **HOST_TENSORBOARD_LOGS_PATH**: 自动设置tensorboard日志路径
2. **量化类型修改**: 使用sed命令动态修改量化类型
3. **命令构建**: 智能构建训练命令
4. **日志捕获**: 使用tee命令捕获训练日志
5. **时间戳**: 自动添加时间戳到日志文件名

#### 使用方法
```bash
# 直接使用模板
cp improved_script_template.sh my_training_script.sh
chmod +x my_training_script.sh

# 修改参数后运行
./my_training_script.sh
```

#### 参数说明
脚本接受以下参数：
1. **CHECKPOINT_PATH** - 检查点保存路径
2. **TENSORBOARD_LOGS_PATH** - TensorBoard日志路径
3. **MODEL_PATH** - 模型路径
4. **DATA_PATH** - 数据路径
5. **PRECISION** - 精度类型 (bf16, fp16, fp8等)

#### 环境变量
- `HOST_TENSORBOARD_LOGS_PATH`: TensorBoard日志路径
- `CUSTOM_QUANT_TYPE`: 自定义量化类型
- `TENSOR_SAVE_DIR`: Tensor保存目录
- `TENSOR_SAVE_ENABLED`: 是否启用tensor保存

## 模板定制

### 1. 修改默认参数
```bash
# 在模板中修改默认值
DEFAULT_CHECKPOINT_PATH="./checkpoints/my_model"
DEFAULT_TENSORBOARD_LOGS_PATH="./tensorboard_logs/my_model"
DEFAULT_MODEL_PATH="./model/my_model"
DEFAULT_DATA_PATH="./dataset/my_data"
DEFAULT_PRECISION="bf16"
```

### 2. 添加新的量化类型
```bash
# 在modify_quantization_types函数中添加
if [ "$QUANT_TYPE" = "new_quant_type" ]; then
    sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'new_quant_type'/" \
        megatron/core/tensor_parallel/layers.py
    sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'new_quant_type'/" \
        megatron/core/transformer/dot_product_attention.py
fi
```

### 3. 修改训练命令
```bash
# 在build_and_run_command函数中修改
TRAINING_CMD="bash examples/my_model/train_my_model.sh"
```

## 最佳实践

### 1. 脚本命名
- 使用描述性的文件名
- 包含模型名称、数据集、量化类型等信息
- 例如: `pretrain_llama32-1b_dolma_hifp8.sh`

### 2. 参数验证
- 检查必需参数是否存在
- 验证路径是否有效
- 设置合理的默认值

### 3. 错误处理
- 使用`set -e`在错误时退出
- 检查命令执行结果
- 提供清晰的错误信息

### 4. 日志记录
- 记录开始和结束时间
- 保存命令执行日志
- 使用时间戳命名日志文件

## 扩展模板

### 1. 创建新的模板
```bash
# 基于现有模板创建新模板
cp improved_script_template.sh new_template.sh

# 修改新模板的内容
# 添加新的功能或修改现有功能
```

### 2. 模板版本管理
- 使用版本号管理模板
- 记录模板的修改历史
- 保持向后兼容性

### 3. 模板测试
- 使用小数据集测试模板
- 验证所有功能正常工作
- 检查日志输出是否正确

## 注意事项

1. **权限设置**: 确保模板文件有执行权限
2. **路径检查**: 验证所有路径是否正确
3. **环境依赖**: 确保所需环境已正确设置
4. **资源限制**: 根据系统资源调整参数
5. **备份策略**: 定期备份重要的模板文件
