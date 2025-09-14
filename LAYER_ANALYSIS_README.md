# Layer Distribution Analysis Tool

## 概述

专门分析某个层的tensor分布的工具，支持attention和linear层的q,k,v,output和input,weight,output分析。使用正则表达式匹配tensor文件，生成详细的分布图表和统计报告。

## 功能特性

### 1. 层分析功能
- **Attention层分析**: 分析query, key, value, output, attention_weights的分布
- **Linear层分析**: 分析input, weight, output, bias, hidden的分布
- **多子图显示**: 一个大图包含6个子图，展示不同tensor类型的分布
- **统计信息**: 每个子图显示均值、标准差等关键统计信息

### 2. 量化对比功能
- **多量化类型对比**: 同时显示bf16, mxfp8, mxfp4, hifp8的分布对比
- **特定tensor分析**: 可以针对特定tensor类型进行量化对比
- **2x2子图布局**: 清晰展示4种量化类型的分布差异

### 3. 统计报告
- **详细统计信息**: 包含均值、标准差、分位数等完整统计
- **文件计数**: 显示找到的文件数量
- **数据质量**: 显示有效数据点数量

## 使用方法

### 基本用法

#### Python脚本直接调用
```bash
# 分析attention层
python analyze_layer_distribution.py --layer 1 --sample 0 --layer_type attention

# 分析linear层
python analyze_layer_distribution.py --layer 2 --sample 1 --layer_type linear

# 量化对比分析
python analyze_layer_distribution.py --layer 1 --sample 0 --layer_type attention --tensor_type query --quantization_comparison
```

#### Shell脚本调用
```bash
# 基本用法
./run_layer_analysis.sh <tensor_dir> <output_dir> <layer> <sample> <layer_type>

# 示例
./run_layer_analysis.sh ./enhanced_tensor_logs ./layer_output 1 0 attention
./run_layer_analysis.sh ./enhanced_tensor_logs ./layer_output 2 1 linear

# 带量化对比
./run_layer_analysis.sh ./enhanced_tensor_logs ./layer_output 1 0 attention query true
```

### 参数说明

#### Python脚本参数
- `--tensor_dir`: 张量文件目录 (默认: ./enhanced_tensor_logs)
- `--output_dir`: 输出目录 (默认: ./layer_analysis_output)
- `--layer`: 层号 (必需, 如: 1, 2, 3, ...)
- `--sample`: 样本号 (必需, 如: 0, 1, 2)
- `--layer_type`: 层类型 (必需, attention 或 linear)
- `--tensor_type`: 特定tensor类型 (可选, 用于量化对比)
- `--quantization_comparison`: 启用量化对比 (可选)

#### Shell脚本参数
1. `tensor_dir`: 张量文件目录
2. `output_dir`: 输出目录
3. `layer`: 层号
4. `sample`: 样本号
5. `layer_type`: 层类型 (attention/linear)
6. `tensor_type`: 特定tensor类型 (可选)
7. `quantization_comparison`: 是否启用量化对比 (true/false)

## 输出文件

### 1. 层分析图表
- **文件名格式**: `layer_{layer}_sample_{sample}_{layer_type}_analysis.png`
- **内容**: 6个子图显示不同tensor类型的分布
- **统计信息**: 每个子图包含均值、标准差等统计信息

### 2. 量化对比图表
- **文件名格式**: `quantization_comparison_layer_{layer}_sample_{sample}_{layer_type}_{tensor_type}.png`
- **内容**: 2x2子图显示4种量化类型的分布对比
- **适用场景**: 需要比较不同量化类型对同一tensor的影响

### 3. 统计报告
- **文件名格式**: `statistics_layer_{layer}_sample_{sample}_{layer_type}.txt`
- **内容**: 详细的数值统计信息
- **包含信息**: 文件数量、数据点数量、均值、标准差、分位数等

## 支持的Tensor类型

### Attention层
- `query`: Query张量
- `key`: Key张量
- `value`: Value张量
- `output`: 输出张量
- `attention_weights`: 注意力权重矩阵

### Linear层
- `input`: 输入张量
- `weight`: 权重张量
- `output`: 输出张量
- `bias`: 偏置张量
- `hidden`: 隐藏层张量

## 文件命名格式支持

工具支持以下文件命名格式：
```
YYYYMMDD_HHMMSS_XXXX_iterXXX_layer_type_LX_operation_phase_component_quant_type_rankXX_sampleXXX_groupXXX_tensor_name.pt
```

示例：
```
20250914_075006_1399_iter000_attention_L1_forward_post_FA_bf16_rank07_sample000_group000_attention_weights.pt
```

## 使用示例

### 示例1: 分析第1层第0个样本的attention分布
```bash
python analyze_layer_distribution.py --layer 1 --sample 0 --layer_type attention
```

### 示例2: 分析第2层第1个样本的linear分布
```bash
python analyze_layer_distribution.py --layer 2 --sample 1 --layer_type linear
```

### 示例3: 对比第1层第0个样本query张量的量化效果
```bash
python analyze_layer_distribution.py --layer 1 --sample 0 --layer_type attention --tensor_type query --quantization_comparison
```

### 示例4: 使用shell脚本分析
```bash
# 分析attention层
./run_layer_analysis.sh ./enhanced_tensor_logs ./output 1 0 attention

# 分析linear层并启用量化对比
./run_layer_analysis.sh ./enhanced_tensor_logs ./output 2 1 linear weight true
```

## 技术特性

### 数据处理
- **自动数据清理**: 自动移除NaN和Inf值
- **数据采样**: 大数据集自动采样以提高性能
- **多文件合并**: 自动合并同一类型的多个tensor文件

### 可视化质量
- **高分辨率**: 300 DPI输出
- **专业配色**: 科学可视化标准配色
- **清晰标注**: 详细的图表标签和统计信息

### 错误处理
- **文件损坏处理**: 自动跳过损坏的tensor文件
- **格式兼容性**: 支持多种tensor文件格式
- **优雅降级**: 数据缺失时显示友好提示

## 依赖要求

### Python包
- torch
- matplotlib
- numpy
- pandas
- seaborn

### 安装依赖
```bash
pip install torch matplotlib numpy pandas seaborn scipy
```

## 注意事项

1. **文件格式**: 确保tensor文件格式正确且可读
2. **内存使用**: 处理大量tensor文件时注意内存使用
3. **输出目录**: 确保对输出目录有写权限
4. **层和样本**: 确保指定的层和样本存在对应的tensor文件

## 故障排除

### 常见问题
1. **No data found**: 检查层号、样本号和层类型是否正确
2. **No valid data**: 检查tensor文件是否损坏或格式不正确
3. **Import error**: 安装缺失的Python包
4. **Permission denied**: 检查输出目录的写权限

### 调试建议
1. 检查tensor文件是否存在
2. 验证文件名格式是否正确
3. 确认Python环境配置
4. 查看详细错误信息

## 版本历史

### v1.0.0 (当前版本)
- 基础层分析功能
- 支持attention和linear层
- 量化对比功能
- 统计报告生成
- Shell脚本封装
