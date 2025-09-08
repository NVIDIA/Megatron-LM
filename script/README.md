# Script目录结构说明

这个目录包含了Megatron-LM项目的所有脚本文件，按功能分类组织。

## 📁 目录结构

```
script/
├── data_processing/          # 数据处理脚本
│   ├── process_dolma_data.sh
│   ├── process_wikipedia_data.sh
│   ├── process_c4_data.sh
│   ├── process_custom_data.sh
│   ├── data_processing_utils.py
│   └── README.md
├── visualization/            # 可视化脚本
│   ├── visualize_tensors.py
│   ├── quick_visualize.py
│   ├── one_click_visualize.sh
│   └── README.md
├── utils/                   # 工具脚本
│   ├── quant_type_modifier.py
│   ├── update_scripts_with_pattern_v2.py
│   └── README.md
├── templates/               # 模板文件
│   ├── improved_script_template.sh
│   └── README.md
├── training/                # 训练脚本（按模型分类）
│   ├── llama32-1b/
│   ├── llama31-8b/
│   └── deepseek2_lite/
└── README.md               # 本文件
```

## 🚀 快速开始

### 1. 数据处理
```bash
# 处理Dolma数据集
cd data_processing
./process_dolma_data.sh

# 使用工具函数
python data_processing_utils.py --action check
```

### 2. 模型训练
```bash
# 使用模板创建训练脚本
cp templates/improved_script_template.sh my_training.sh
chmod +x my_training.sh
./my_training.sh
```

### 3. 结果可视化
```bash
# 一键可视化
cd visualization
./one_click_visualize.sh
```

## 📋 功能概览

### 🔧 数据处理 (data_processing/)
- **数据集处理**: 支持Dolma、Wikipedia、C4等主流数据集
- **格式转换**: 将原始数据转换为Megatron-LM训练格式
- **批量处理**: 支持大规模数据集的并行处理
- **工具函数**: 提供环境检查、时间估算等辅助功能

### 📊 可视化 (visualization/)
- **Tensor分析**: 可视化训练过程中的tensor数据
- **量化研究**: 分析不同量化类型的影响
- **统计图表**: 生成分布图、热力图、对比图等
- **一键操作**: 自动生成所有分析图表

### 🛠️ 工具脚本 (utils/)
- **量化类型管理**: 批量修改脚本中的量化类型
- **脚本模式更新**: 应用统一的脚本模式
- **批量操作**: 支持大规模脚本文件的批量处理

### 📝 模板文件 (templates/)
- **训练脚本模板**: 功能完整的训练脚本模板
- **标准化配置**: 统一的参数设置和错误处理
- **易于定制**: 支持快速创建新的训练脚本

### 🏋️ 训练脚本 (training/)
- **模型分类**: 按模型类型组织训练脚本
- **量化支持**: 支持多种量化类型 (hifp8, mxfp8, mxfp4, bf16等)
- **数据集适配**: 支持多种数据集的训练配置

## 🎯 使用场景

### 1. 量化研究
```bash
# 1. 处理数据
cd data_processing
./process_dolma_data.sh

# 2. 运行训练（保存tensor）
cd ../training/llama32-1b
./pretrain_llama32-1b_dolma_hifp8.sh

# 3. 可视化分析
cd ../../visualization
./one_click_visualize.sh
```

### 2. 模型训练
```bash
# 1. 使用模板创建脚本
cp templates/improved_script_template.sh my_training.sh

# 2. 修改参数
vim my_training.sh

# 3. 运行训练
./my_training.sh
```

### 3. 批量操作
```bash
# 1. 批量修改量化类型
cd utils
python quant_type_modifier.py --directory ../training/ --old_quant_type bf16 --new_quant_type hifp8

# 2. 更新脚本模式
python update_scripts_with_pattern_v2.py
```

## 📚 详细文档

每个子目录都包含详细的README文档：

- **[数据处理文档](data_processing/README.md)** - 数据处理脚本的详细说明
- **[可视化文档](visualization/README.md)** - 可视化工具的完整指南
- **[工具文档](utils/README.md)** - 工具脚本的使用方法
- **[模板文档](templates/README.md)** - 模板文件的定制指南

## 🔧 环境要求

### 基础环境
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+

### Python依赖
```bash
pip install matplotlib seaborn pandas scipy
```

### 环境变量
```bash
export CUSTOM_QUANT_TYPE="hifp8"
export TENSOR_SAVE_DIR="./tensor_logs"
export TENSOR_SAVE_ENABLED="true"
```

## 🚨 注意事项

### 1. 文件权限
```bash
# 设置脚本执行权限
chmod +x *.sh
```

### 2. 路径配置
- 确保所有路径配置正确
- 检查数据集和模型文件是否存在
- 验证输出目录的写入权限

### 3. 资源管理
- 根据系统资源调整工作进程数
- 监控磁盘空间使用情况
- 注意内存使用峰值

### 4. 错误处理
- 查看详细的错误日志
- 使用dry_run模式预览操作
- 定期备份重要文件

## 🤝 贡献指南

### 添加新脚本
1. 选择合适的子目录
2. 遵循现有的命名规范
3. 添加详细的文档说明
4. 测试脚本功能

### 修改现有脚本
1. 创建备份文件
2. 测试修改后的功能
3. 更新相关文档
4. 提交变更说明

## 📞 支持

如果遇到问题，请：
1. 查看相关子目录的README文档
2. 检查错误日志和输出信息
3. 验证环境配置和依赖
4. 参考使用示例和最佳实践

---

**最后更新**: 2024年9月8日  
**版本**: 1.0.0