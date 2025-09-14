# Enhanced Multi-threaded Tensor Visualization

## 概述

增强版多线程张量可视化工具，支持bf16、mxfp8、mxfp4、hifp8量化类型的多维度分析，包含进度条和高级分析功能。

## 新功能特性

### 1. HiFP8分布分析
- **折线图分布**: 按层和样本显示HiFP8值的分布曲线
- **统计度量**: 均值、标准差、最小值、最大值随层变化趋势
- **热力图**: Layer-Sample矩阵显示平均值的分布
- **分位数分析**: 详细的数值分位数统计

### 2. Layer-Sample分析
- **选定层分析**: 重点分析层1、4、8、12、16
- **均值热力图**: Layer vs Sample的平均值分布
- **标准差热力图**: Layer vs Sample的标准差分布
- **分布曲线**: 每层的数值分布密度曲线
- **统计对比**: 各层统计度量的柱状图对比

### 3. 全局统计分析
- **JSON格式**: 机器可读的详细统计数据
- **文本报告**: 人类可读的统计报告
- **多维度统计**: 按量化类型、层、样本、层类型的详细统计
- **分位数分析**: Q25、Q75、Q95、Q99等关键分位数

### 4. 进度条支持
- **tqdm集成**: 所有操作都有进度条显示
- **多级进度**: 文件加载、数据处理、图表生成都有独立进度条
- **实时反馈**: 显示当前处理的任务和完成百分比

## 使用方法

### 基本用法
```bash
# 使用默认参数
./run_tensor_draw.sh

# 指定参数
./run_tensor_draw.sh <tensor_dir> <output_dir> <max_workers>
```

### 参数说明
- `tensor_dir`: 张量文件目录 (默认: ./enhanced_tensor_logs)
- `output_dir`: 输出目录 (默认: ./draw)
- `max_workers`: 最大工作线程数 (默认: 4)

### 输出文件结构
```
output_dir/
├── quantization_analysis/
│   └── quantization_comparison.png
├── sample_analysis/
│   └── sample_analysis.png
├── layer_analysis/
│   └── layer_analysis.png
├── comparison_analysis/
│   └── comprehensive_comparison.png
├── hifp8_analysis/                    # 新增
│   └── hifp8_distribution_analysis.png
├── layer_sample_analysis/             # 新增
│   └── layer_sample_analysis.png
├── global_statistics/                 # 新增
│   ├── global_statistics.json
│   └── global_statistics_report.txt
└── statistics/
    └── detailed_statistics_report.txt
```

## 新增分析图表详解

### HiFP8分布分析图表
1. **按层分布折线图**: 显示每层HiFP8值的密度分布曲线
2. **按样本分布折线图**: 显示每个样本HiFP8值的密度分布曲线
3. **统计度量趋势图**: 均值、标准差、最小值、最大值随层变化
4. **整体分布图**: 所有HiFP8值的密度分布
5. **Layer-Sample热力图**: 平均值的二维分布
6. **分位数分析图**: 关键分位数的数值分布

### Layer-Sample分析图表
1. **均值热力图**: Layer vs Sample的平均值分布
2. **标准差热力图**: Layer vs Sample的标准差分布
3. **按层分布曲线**: 选定层的数值分布密度
4. **统计度量柱状图**: 各层均值、标准差的对比

### 全局统计报告
- **JSON格式**: 包含所有统计数据的结构化文件
- **文本报告**: 人类可读的详细统计信息
- **多维度分析**: 按不同维度组织的统计数据

## 技术特性

### 性能优化
- **多线程处理**: 支持多线程并行处理
- **内存优化**: 大数据集的分块处理
- **进度监控**: 实时显示处理进度

### 错误处理
- **文件损坏处理**: 自动跳过损坏的张量文件
- **格式兼容性**: 支持多种张量文件格式
- **降级处理**: 主功能失败时的备用方案

### 可视化质量
- **高分辨率**: 300 DPI输出
- **专业配色**: 科学可视化标准配色
- **清晰标注**: 详细的图表标签和说明

## 依赖要求

### Python包
- torch
- matplotlib
- numpy
- pandas
- seaborn
- scipy
- tqdm

### 安装依赖
```bash
pip install torch matplotlib numpy pandas seaborn scipy tqdm
```

## 版本历史

### v3.0.0 (当前版本)
- 新增HiFP8分布分析功能
- 新增Layer-Sample分析功能
- 新增全局统计分析功能
- 集成tqdm进度条
- 增强错误处理和性能优化

### v2.0.0
- 基础多线程可视化功能
- 支持bf16、mxfp8、mxfp4、hifp8量化类型
- 多维度比较分析

## 注意事项

1. **内存使用**: 处理大量张量文件时注意内存使用
2. **文件格式**: 确保张量文件格式正确
3. **输出目录**: 确保有足够的磁盘空间存储输出文件
4. **线程数**: 根据系统性能调整max_workers参数

## 故障排除

### 常见问题
1. **内存不足**: 减少max_workers或处理文件数量
2. **文件损坏**: 检查张量文件完整性
3. **依赖缺失**: 安装所需的Python包
4. **权限问题**: 确保对输出目录有写权限

### 调试模式
设置环境变量启用详细日志：
```bash
export PYTHONPATH=/data/charles/codes/Megatron-LM:$PYTHONPATH
python script/visualization/enhanced_multi_threaded_visualizer.py --tensor_dir <dir> --output_dir <dir>
```
