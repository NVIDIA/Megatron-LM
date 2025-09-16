# 综合可视化脚本使用指南

## 概述

`run_draw_all.sh` 是一个综合的tensor可视化脚本，实现了所有当前支持的可视化功能，包括量化对比、HiFP8分析、全局统计、层分析、溢出检测等。

## 功能特性

### 🎯 支持的可视化功能

1. **量化类型对比分析** - 比较bf16, mxfp8, mxfp4, hifp8的分布
2. **HiFP8分布分析** - 详细的HiFP8数值分布和统计
3. **全局统计分析** - 全面的统计报告和JSON数据
4. **层分析** - 特定层和rank的详细分析
5. **溢出检测分析** - 检测各量化类型的溢出情况
6. **多维度分析** - 按层、rank、类型等多维度分析

### 📊 输出文件结构

```
draw/
├── quantization_analysis/          # 量化对比分析
│   └── quantization_comparison.png
├── hifp8_analysis/                 # HiFP8分析
│   └── hifp8_distribution_analysis.png
├── global_statistics/              # 全局统计
│   ├── global_statistics.json
│   └── global_statistics_report.txt
├── layer_analysis/                 # 层分析
│   └── layer_*_rank_*_*_analysis.png
├── overflow_analysis/              # 溢出分析
│   └── overflow_analysis_report.png
├── quant_analysis_*/               # 各量化类型分析
└── comprehensive_analysis_report.txt  # 综合报告
```

## 使用方法

### 基本用法

```bash
# 运行所有可视化（使用默认参数）
./run_draw_all.sh

# 显示帮助信息
./run_draw_all.sh --help
```

### 高级用法

```bash
# 指定目录和参数
./run_draw_all.sh --tensor-dir ./my_tensors --output-dir ./my_draw --layer 2 --rank 1

# 跳过某些分析
./run_draw_all.sh --skip-layer-analysis --skip-overflow-analysis

# 只运行层分析
./run_draw_all.sh --skip-global-analysis --skip-overflow-analysis --layer 1 --rank 0 --layer-type attention

# 启用量化对比
./run_draw_all.sh --quantization-comparison --layer 2 --rank 1 --tensor-type output

# 使用高效模式
./run_draw_all.sh --efficient-mode --layer 1 --rank 0
```

## 参数说明

### 基本参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tensor-dir` | `./enhanced_tensor_logs` | Tensor文件目录 |
| `--output-dir` | `./draw` | 输出目录 |
| `--max-workers` | `4` | 最大工作线程数 |

### 层分析参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--layer` | `1` | 层号 |
| `--rank` | `0` | GPU rank |
| `--layer-type` | `attention` | 层类型 (attention\|linear) |
| `--tensor-type` | 空 | 特定tensor类型 |

### 控制参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--quantization-comparison` | `true` | 启用量化对比分析 |
| `--efficient-mode` | `true` | 使用高效模式 |
| `--skip-layer-analysis` | `false` | 跳过层分析 |
| `--skip-overflow-analysis` | `false` | 跳过溢出分析 |
| `--skip-global-analysis` | `false` | 跳过全局分析 |

## 使用示例

### 示例1: 基本可视化

```bash
# 运行所有可视化功能
./run_draw_all.sh
```

### 示例2: 指定层分析

```bash
# 分析第2层、rank 1的attention层
./run_draw_all.sh --layer 2 --rank 1 --layer-type attention
```

### 示例3: 只运行全局分析

```bash
# 跳过层分析和溢出分析，只运行全局分析
./run_draw_all.sh --skip-layer-analysis --skip-overflow-analysis
```

### 示例4: 高效模式分析

```bash
# 使用高效模式分析特定层和rank
./run_draw_all.sh --efficient-mode --layer 1 --rank 0 --tensor-type output
```

### 示例5: 自定义目录

```bash
# 使用自定义的tensor目录和输出目录
./run_draw_all.sh --tensor-dir /path/to/tensors --output-dir /path/to/output
```

## 输出说明

### 主要输出文件

- **量化对比分析**: `quantization_analysis/quantization_comparison.png`
- **HiFP8分布分析**: `hifp8_analysis/hifp8_distribution_analysis.png`
- **全局统计 (JSON)**: `global_statistics/global_statistics.json`
- **全局统计报告**: `global_statistics/global_statistics_report.txt`
- **层分析**: `layer_analysis/layer_*_rank_*_*_analysis.png`
- **溢出分析**: `overflow_analysis/overflow_analysis_report.png`
- **综合报告**: `comprehensive_analysis_report.txt`

### 报告内容

综合报告包含：
- 分析时间
- 参数设置
- 分析状态
- 生成的文件列表
- 统计摘要

## 依赖要求

### Python包

- torch
- matplotlib
- numpy
- pandas
- seaborn
- scipy
- tqdm
- concurrent.futures

### 系统要求

- Linux系统
- Python 3.6+
- 足够的磁盘空间存储输出文件

## 故障排除

### 常见问题

1. **Python包缺失**
   ```bash
   pip install matplotlib numpy pandas seaborn scipy tqdm
   ```

2. **Tensor目录不存在**
   - 确保 `enhanced_tensor_logs` 目录存在
   - 或使用 `--tensor-dir` 指定正确的目录

3. **权限问题**
   ```bash
   chmod +x run_draw_all.sh
   ```

4. **内存不足**
   - 减少 `--max-workers` 参数
   - 使用 `--efficient-mode` 启用高效模式

### 调试模式

```bash
# 显示详细输出
bash -x ./run_draw_all.sh
```

## 相关脚本

- `run_tensor_collection.sh` - Tensor收集脚本
- `run_tensor_draw.sh` - 基础可视化脚本
- `run_layer_analysis.sh` - 层分析脚本
- `run_draw_all_example.sh` - 使用示例脚本

## 版本信息

- 脚本版本: 1.0.0
- 支持的分析类型: all, overflow, layer, distribution
- 支持的量化类型: bf16, mxfp8, mxfp4, hifp8
