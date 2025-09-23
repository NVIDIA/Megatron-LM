# Scaling Factor Analysis Tool

这个Python程序用于分析所有存储的log文件，检查是否所有tensor的最推荐Scaling都是其最大值。

## 功能特点

- 🔍 自动扫描所有scaling_analysis目录下的log文件
- 📊 解析每个tensor的对齐范围(max_align, min_align)和推荐scaling factor
- ✅ 检查推荐的scale exponent是否等于max_align(最大值)
- 📈 生成详细的统计报告，包括按层和按tensor类型的分析
- 💾 将结果保存为JSON格式便于进一步分析

## 使用方法

### 方法1：使用简单脚本
```bash
python3 run_analysis.py
```

### 方法2：使用完整程序
```bash
python3 analyze_scaling_factors.py
```

### 方法3：指定目录
```bash
python3 run_analysis.py /path/to/your/data/directory
```

## 输出说明

程序会输出以下信息：

1. **总体摘要**: 显示分析的tensor总数和在最大值的tensor数量
2. **按类型分类**: 显示不同tensor类型(input, weight, output等)的统计
3. **按层分类**: 显示不同层的统计信息
4. **性能统计**: 显示平均composite score和MSE
5. **详细列表**: 如果有tensor不在最大值，会列出详细信息

## 输出文件

- `scaling_analysis_results.json`: 包含完整分析结果的JSON文件

## 分析结果

根据当前数据的分析结果：

🎉 **所有140个tensor的推荐scaling factor都是其最大值！**

这意味着：
- 所有tensor都在使用最优的scaling factor
- 没有overflow风险
- 精度损失最小

## 文件结构

```
draw/
├── analyze_scaling_factors.py  # 主分析程序
├── run_analysis.py            # 简单包装脚本
├── scaling_analysis_results.json  # 分析结果
└── scaling_analysis/          # 包含所有log文件的目录
    ├── 20250915_040631_0001_.../
    ├── 20250915_040632_0002_.../
    └── ...
```

## 技术细节

程序通过正则表达式解析log文件中的关键信息：
- `Calculated alignment: max_align=X, min_align=Y`
- `⭐ RECOMMENDED Scaling Factor: X`  
- `Scale Exponent: Y`

然后检查推荐的scale exponent是否等于max_align值(容差1e-6)。

## 依赖

- Python 3.6+
- 标准库模块：os, re, glob, json, pathlib, typing, dataclasses, collections




