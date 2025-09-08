# Wikipedia Tensor收集指南

## 🎯 目标

收集wikipedia数据集在mxfp8、mxfp4和hifp8量化类型下各个位置的tensor，用于量化研究和分析。

## 📋 前置条件

### 1. 环境要求
- 已配置的Megatron-LM环境
- 已安装的依赖包（torch, matplotlib, seaborn等）
- 可用的GPU资源

### 2. 数据要求
- Wikipedia数据集已预处理完成
- 数据路径：`dataset/wikipedia_processed/wikipedia_processed_text_document`
- 分词器路径：`model/llama3.2-1b`

### 3. 文件要求
- 训练脚本：`examples/llama/train_llama32_1b_h100_fp8.sh`
- 已修改的tensor保存代码

## 🚀 使用方法

### 方法1：单次运行特定量化类型（推荐）

```bash
# 运行mxfp8量化类型
./run_single_quant_type.sh mxfp8

# 运行mxfp4量化类型
./run_single_quant_type.sh mxfp4

# 运行hifp8量化类型
./run_single_quant_type.sh hifp8
```

**优点**：
- 简单易用
- 可以单独控制每个量化类型
- 便于调试和监控
- 可以随时停止和重启

### 方法2：快速批量收集

```bash
# 自动运行所有量化类型
./quick_tensor_collection.sh
```

**优点**：
- 自动化程度高
- 一次性收集所有量化类型
- 自动监控和停止

### 方法3：完整批量收集

```bash
# 完整的tensor收集流程
./run_wikipedia_tensor_collection.sh
```

**优点**：
- 功能最完整
- 详细的日志记录
- 完整的错误处理

## 📊 收集的Tensor类型

### 1. Attention层Tensor
- **pre-FA**: Flash Attention操作前的输入tensor
  - `query`: Query tensor
  - `key`: Key tensor  
  - `value`: Value tensor
- **post-FA**: Flash Attention操作后的输出tensor
  - `output`: Attention输出tensor

### 2. Linear层Tensor
- **pre-linear**: Linear层操作前的输入tensor
  - `input`: 输入tensor
  - `weight`: 权重tensor
- **post-linear**: Linear层操作后的输出tensor
  - `output`: Linear输出tensor

### 3. Backward Tensor
- **pre-linear**: Backward操作前的输入tensor
- **post-linear**: Backward操作后的输出tensor

## 📁 文件命名规则

```
{timestamp}_{counter}_{layer_type}_{layer_idx}_{operation}_{phase}_{component}_{quant_type}_{tensor_name}.pt
```

### 示例文件名
```
20250908_095220_0001_attention_L0_forward_pre_FA_mxfp8_query.pt
20250908_095220_0002_attention_L0_forward_pre_FA_mxfp8_key.pt
20250908_095220_0003_attention_L0_forward_pre_FA_mxfp8_value.pt
20250908_095220_0004_attention_L0_forward_post_FA_mxfp8_output.pt
20250908_095220_0005_linear_L1_forward_pre_linear_mxfp8_input.pt
20250908_095220_0006_linear_L1_forward_pre_linear_mxfp8_weight.pt
20250908_095220_0007_linear_L1_forward_post_linear_mxfp8_output.pt
```

## 📂 输出目录结构

```
enhanced_tensor_logs/
├── mxfp8/
│   ├── 20250908_095220_0001_attention_L0_forward_pre_FA_mxfp8_query.pt
│   ├── 20250908_095220_0002_attention_L0_forward_pre_FA_mxfp8_key.pt
│   └── ...
├── mxfp4/
│   ├── 20250908_095220_0001_attention_L0_forward_pre_FA_mxfp4_query.pt
│   ├── 20250908_095220_0002_attention_L0_forward_pre_FA_mxfp4_key.pt
│   └── ...
└── hifp8/
    ├── 20250908_095220_0001_attention_L0_forward_pre_FA_hifp8_query.pt
    ├── 20250908_095220_0002_attention_L0_forward_pre_FA_hifp8_key.pt
    └── ...
```

## 🔍 分析收集到的Tensor

### 1. 快速查看
```bash
# 查看所有tensor文件
ls -la enhanced_tensor_logs/*/

# 查看特定量化类型的tensor
ls -la enhanced_tensor_logs/mxfp8/
ls -la enhanced_tensor_logs/mxfp4/
ls -la enhanced_tensor_logs/hifp8/
```

### 2. 使用可视化脚本
```bash
# 快速可视化
python script/visualization/quick_visualize.py --tensor_dir enhanced_tensor_logs

# 一键可视化
bash script/visualization/one_click_visualize.sh enhanced_tensor_logs

# 完整可视化
python script/visualization/visualize_tensors.py --tensor_dir enhanced_tensor_logs
```

### 3. 手动分析
```python
import torch
import glob

# 加载特定量化类型的tensor
tensor_files = glob.glob("enhanced_tensor_logs/mxfp8/*.pt")
for file_path in tensor_files:
    data = torch.load(file_path, map_location='cpu')
    tensor = data['tensor']
    metadata = data['metadata']
    
    print(f"文件: {file_path}")
    print(f"形状: {tensor.shape}")
    print(f"数据类型: {tensor.dtype}")
    print(f"量化类型: {metadata['quant_type']}")
    print(f"阶段: {metadata['phase']}")
    print(f"组件: {metadata['component']}")
    print(f"操作: {metadata['operation']}")
    print("---")
```

## 📈 预期结果

### 1. 文件数量
- 每个量化类型预计收集20-50个tensor文件
- 包含pre/post阶段的tensor
- 包含FA/linear组件的tensor
- 包含forward/backward操作的tensor

### 2. 文件大小
- 每个tensor文件大小取决于tensor形状
- 通常每个文件几KB到几MB
- 总存储空间需求：几百MB到几GB

### 3. 收集时间
- 每个量化类型：5-15分钟
- 总收集时间：15-45分钟
- 取决于GPU性能和数据集大小

## 🛠️ 故障排除

### 1. 常见问题

#### 训练脚本不存在
```bash
[ERROR] 训练脚本不存在: examples/llama/train_llama32_1b_h100_fp8.sh
```
**解决方案**：确保训练脚本存在且路径正确

#### 数据路径不存在
```bash
[ERROR] 数据路径不存在: dataset/wikipedia_processed/wikipedia_processed_text_document
```
**解决方案**：确保Wikipedia数据集已预处理完成

#### 权限问题
```bash
Permission denied
```
**解决方案**：给脚本添加执行权限
```bash
chmod +x run_single_quant_type.sh
```

### 2. 调试技巧

#### 检查环境变量
```bash
echo $TENSOR_SAVE_ENABLED
echo $TENSOR_SAVE_DIR
```

#### 检查量化类型修改
```bash
grep "custom_quant_type" megatron/core/tensor_parallel/layers.py
grep "custom_quant_type" megatron/core/transformer/dot_product_attention.py
```

#### 监控tensor生成
```bash
watch -n 5 "find enhanced_tensor_logs -name '*.pt' | wc -l"
```

## 📝 使用示例

### 示例1：收集mxfp8量化类型的tensor
```bash
# 1. 运行收集脚本
./run_single_quant_type.sh mxfp8

# 2. 查看结果
ls -la enhanced_tensor_logs/mxfp8/

# 3. 分析tensor
python script/visualization/quick_visualize.py --tensor_dir enhanced_tensor_logs/mxfp8
```

### 示例2：批量收集所有量化类型
```bash
# 1. 运行批量收集
./quick_tensor_collection.sh

# 2. 查看所有结果
ls -la enhanced_tensor_logs/*/

# 3. 对比分析
python script/visualization/visualize_tensors.py --tensor_dir enhanced_tensor_logs
```

### 示例3：分析特定阶段的tensor
```bash
# 查看pre阶段的tensor
find enhanced_tensor_logs -name "*_pre_*" | head -10

# 查看post阶段的tensor
find enhanced_tensor_logs -name "*_post_*" | head -10

# 查看FA组件的tensor
find enhanced_tensor_logs -name "*_FA_*" | head -10

# 查看linear组件的tensor
find enhanced_tensor_logs -name "*_linear_*" | head -10
```

## 🎯 下一步操作

1. **运行收集脚本**：选择合适的方法收集tensor
2. **验证结果**：检查收集到的tensor文件
3. **可视化分析**：使用可视化脚本分析tensor
4. **对比研究**：比较不同量化类型的tensor特性
5. **深入分析**：根据研究需求进行特定分析

## 📞 支持

如果遇到问题，请检查：
1. 环境配置是否正确
2. 数据路径是否存在
3. 脚本权限是否正确
4. 日志文件中的错误信息

---

**创建时间**: 2024年9月8日  
**版本**: 1.0.0  
**适用场景**: Wikipedia数据集tensor收集和分析
