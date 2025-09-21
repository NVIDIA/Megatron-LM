# MXFP Scaling Analysis Tools

这个文档介绍了新增的MXFP量化缩放分析工具，用于分析和优化MXFP量化过程中的缩放策略。

## 新增功能

### 1. 下溢出分析功能

在`quant/mxfp.py`中新增了`_analyze_underflow_before_quantization`函数，该函数会在量化前分析张量的下溢出情况。

**功能特点：**
- 在量化前检测潜在的下溢出问题
- 分析scaling对下溢出的影响
- 提供详细的下溢出统计信息
- 不会影响量化过程的正常执行

**触发条件：**
- 当检测到下溢出比例 > 0.1% 时自动输出分析报告
- 提供高、中等下溢出率的警告

### 2. MXFP缩放测试工具

创建了`quant/mxfp_scaling_test.py`工具，用于测试不同缩放策略对量化精度的影响。

**主要功能：**
- 测试从最大值对齐到最小值对齐的不同缩放级别
- 计算多种精度指标（MSE、余弦相似度、PSNR等）
- 生成详细的折线图可视化结果
- 支持多种MXFP格式

## 使用方法

### 基本用法

```bash
# 测试单个张量文件的缩放效果
python quant/mxfp_scaling_test.py input_tensor.pt

# 测试多个张量文件
python quant/mxfp_scaling_test.py tensor1.pt tensor2.pt tensor3.pt

# 指定输出目录和参数
python quant/mxfp_scaling_test.py input_tensor.pt --output-dir ./results/ --elem-format fp8_e4m3 --num-levels 31
```

### 参数说明

- `input_tensor`: 输入的BF16张量文件路径（支持多个文件）
- `--output-dir`: 输出结果目录（默认：./draw/scaling_analysis/{tensor_name}/）
- `--elem-format`: 量化格式（fp8_e4m3, fp8_e5m2, fp4_e2m1等）
- `--scale-bits`: 缩放位数（默认：8）
- `--max-scale-exp`: 最大缩放指数（默认：自动计算，基于tensor最大值对齐）
- `--min-scale-exp`: 最小缩放指数（默认：自动计算，基于tensor最小值对齐）
- `--num-levels`: 测试的缩放级别数量（默认：21）
- `--no-plots`: 跳过生成图表

### 多tensor处理特性

当提供多个tensor文件时，工具会：

1. **独立处理**: 每个tensor文件独立进行缩放测试和分析
2. **独立输出**: 为每个tensor创建独立的输出目录和日志文件
3. **进度显示**: 实时显示处理进度 `[1/3] Processing: tensor1.pt`
4. **状态反馈**: 显示每个tensor的处理结果（✅成功 / ❌失败）
5. **最终汇总**: 提供所有tensor的处理汇总统计

**示例输出：**
```
Processing 3 tensor(s)...
================================================================================

[1/3] Processing: tensor1.pt
------------------------------------------------------------
✅ Successfully processed: tensor1.pt

[2/3] Processing: tensor2.pt
------------------------------------------------------------
✅ Successfully processed: tensor2.pt

[3/3] Processing: tensor3.pt
------------------------------------------------------------
✅ Successfully processed: tensor3.pt

================================================================================
FINAL SUMMARY
================================================================================
Total tensors: 3
Successful: 3
Failed: 0
🎉 All tests completed successfully!
```

### 输出结果

工具会生成以下文件，默认保存在 `draw/scaling_analysis/{tensor_name}/` 目录下：

1. **详细结果文件**: `mxfp_scaling_results_<format>.txt`
   - 包含所有缩放级别的详细指标数据

2. **综合图表**: `mxfp_scaling_test_<format>.png`
   - 6个子图展示不同指标随缩放指数的变化

3. **摘要图表**: `mxfp_scaling_summary_<format>.png`
   - 关键指标（MSE、余弦相似度、PSNR）的汇总图

4. **日志文件**: `mxfp_scaling_test_{tensor_name}_{format}.log`
   - 完整的测试过程日志，包含所有输出信息
   - 同时显示在控制台和保存到文件中
   - 包含详细的缩放因子分析和推荐

**输出目录结构示例：**
```
draw/scaling_analysis/
├── tensor1/
│   ├── mxfp_scaling_results_fp8_e4m3.txt
│   ├── mxfp_scaling_test_fp8_e4m3.png
│   ├── mxfp_scaling_summary_fp8_e4m3.png
│   └── mxfp_scaling_test_tensor1_fp8_e4m3.log
├── tensor2/
│   ├── mxfp_scaling_results_fp8_e5m2.txt
│   ├── mxfp_scaling_test_fp8_e5m2.png
│   ├── mxfp_scaling_summary_fp8_e5m2.png
│   └── mxfp_scaling_test_tensor2_fp8_e5m2.log
└── ...
```

### 日志功能

工具会自动生成详细的日志文件，记录完整的测试过程：

**日志特点：**
- **双重输出**: 同时显示在控制台和保存到日志文件
- **时间戳**: 每条日志都包含精确的时间戳
- **完整记录**: 记录从测试开始到结束的所有信息
- **结构化格式**: 清晰的日志格式，便于分析和调试

**日志内容包含：**
- 测试开始和结束时间
- 输入张量信息（形状、数据类型、数值范围）
- 测试参数（格式、缩放位数、指数范围等）
- 每个缩放级别的测试进度和结果
- **详细的缩放因子分析和推荐**
- 最终的最佳结果汇总
- 文件保存位置信息

### 智能分析功能

工具会自动分析测试结果并提供智能推荐：

**分析内容：**
- **个体指标最优解**: 找出MSE、余弦相似度、PSNR等指标的最佳缩放因子
- **综合评分推荐**: 基于加权综合评分推荐最佳缩放因子
- **性能稳定性分析**: 分析不同缩放因子下的性能变化范围
- **实用性建议**: 根据性能变化程度提供使用建议

**推荐算法：**
- 综合评分权重：MSE(30%) + 余弦相似度(30%) + PSNR(20%) + MAE(10%) + 相对误差(10%)
- 自动识别性能稳定性和关键选择点
- 提供基于数据特征的个性化建议

### 智能溢出分析功能

工具会自动分析每个缩放级别的上溢出和下溢出情况并提供详细报告：

**溢出分析内容：**
- **上溢出检测**: 检测值超出格式最大表示范围的情况
- **下溢出检测**: 检测值小于格式最小表示范围的情况
- **严重程度分类**: 高严重程度(>1%)、中等严重程度(0.1-1%)、无显著溢出(<0.1%)
- **详细统计**: 每个缩放级别的溢出数量、百分比和刷新到零的统计
- **张量范围分析**: 显示量化前后的张量数值范围
- **最优范围推荐**: 基于溢出分析推荐最佳的缩放范围

**溢出分析输出示例：**
```
================================================================================
OVERFLOW/UNDERFLOW ANALYSIS SUMMARY
================================================================================
Format: fp8_e4m3
Analyzed 7 scaling levels
Significant overflow/underflow detected in 7 levels
--------------------------------------------------------------------------------
🔴 OVERFLOW ISSUES:
----------------------------------------
  Scale Exp: 5.00 (Factor: 32.000000)
    Overflow: 10 (5.00%)
    Max Normal: 4.48e+02
    Tensor Range: [-7.76e+02, 6.32e+02]
    Severity: HIGH

🟡 UNDERFLOW ISSUES:
----------------------------------------
  Scale Exp: 5.00 (Factor: 32.000000)
    Underflow: 100 (50.00%)
    Flush to Zero: 100 (50.00%)
    Min Normal: 1.56e-02
    Tensor Range: [-7.76e+02, 6.32e+02]
    Severity: HIGH

OVERFLOW EXTREMES:
----------------------------------------
Worst Overflow: Scale Exp -6.67
  50.00% overflow

UNDERFLOW EXTREMES:
----------------------------------------
Worst Underflow: Scale Exp 5.00
  50.00% underflow, 50.00% flushed to zero
Best Underflow: Scale Exp -30.00
  0.00% underflow, 0.50% flushed to zero
--------------------------------------------------------------------------------
OVERFLOW/UNDERFLOW RECOMMENDATIONS:
----------------------------------------
⚠️  AVOID scaling factors with HIGH overflow/underflow severity
   These factors cause significant precision loss
🔴 OVERFLOW WARNING:
   Avoid scaling factors that cause overflow
   These values will be saturated to max representable value
🟡 UNDERFLOW CONSIDERATIONS:
   Moderate underflow may be acceptable depending on use case
   Balance between underflow and overflow risks
⚠️  All scaling levels have some overflow/underflow - choose least problematic
💡 Least problematic scaling: 5.00
   Overflow: 5.00%, Underflow: 50.00%
================================================================================
```

**日志文件命名规则：**
```
mxfp_scaling_test_{tensor_name}_{format}.log
```

例如：`mxfp_scaling_test_my_tensor_fp8_e4m3.log`

**分析输出示例：**
```
================================================================================
SCALING FACTOR ANALYSIS & RECOMMENDATIONS
================================================================================
Format: fp8_e4m3 (e4m5)
Tested 7 scaling levels from -2.00 to 2.00
--------------------------------------------------------------------------------
INDIVIDUAL METRIC OPTIMA:
----------------------------------------
🏆 Best MSE: Scale Exp = 0.00, Factor = 1.000000
    MSE: 1.765694e+00, Cosine: 0.999654, PSNR: 41.34 dB
🎯 Best Cosine Similarity: Scale Exp = 0.00, Factor = 1.000000
    MSE: 1.765694e+00, Cosine: 0.999654, PSNR: 41.34 dB
📊 Best PSNR: Scale Exp = 0.00, Factor = 1.000000
    MSE: 1.765694e+00, Cosine: 0.999654, PSNR: 41.34 dB
--------------------------------------------------------------------------------
COMPOSITE RECOMMENDATION:
----------------------------------------
⭐ RECOMMENDED Scaling Factor: 1.000000
   Scale Exponent: 0.00
   Composite Score: 0.9999
   Balanced Performance:
     - MSE: 1.765694e+00
     - Cosine Similarity: 0.999654
     - PSNR: 41.34 dB
     - MAE: 8.851570e-01
     - Relative Error: 2.20%
--------------------------------------------------------------------------------
PERFORMANCE ANALYSIS:
----------------------------------------
MSE Range: 1.765694e+00 to 1.117014e+01 (Δ: 9.404443e+00)
Cosine Range: 0.997942 to 0.999654 (Δ: 0.001712)
PSNR Range: 33.33 to 41.34 dB (Δ: 8.01 dB)
MSE Stability (std): 3.263742e+00
Cosine Stability (std): 0.000594
--------------------------------------------------------------------------------
RECOMMENDATIONS:
----------------------------------------
⚠️  MSE varies significantly with scaling - choose the recommended factor carefully
✅ Cosine similarity is very stable - scaling factor has minimal impact on direction preservation
✅ Small PSNR range - scaling factor has limited impact on quality
--------------------------------------------------------------------------------
FINAL RECOMMENDATION:
----------------------------------------
🎯 Use scaling factor: 1.000000
   This provides the best balance of accuracy and stability for fp8_e4m3 quantization
   Scale exponent: 0.00
   📍 This is a balanced middle ground between overflow and underflow
================================================================================
```

### 计算的指标

- **MSE (Mean Squared Error)**: 均方误差
- **RMSE (Root Mean Squared Error)**: 均方根误差
- **Cosine Similarity**: 余弦相似度
- **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比
- **MAE (Mean Absolute Error)**: 平均绝对误差
- **Max Absolute Error**: 最大绝对误差
- **Relative Error**: 相对误差百分比

## 测试和验证

### 运行测试脚本

```bash
# 运行基本功能测试
python test_mxfp_scaling.py
```

### 示例用法

```bash
# 测试FP8 E4M3格式的缩放效果
python quant/mxfp_scaling_test.py your_tensor.pt --elem-format fp8_e4m3 --num-levels 21

# 测试FP8 E5M2格式，扩大测试范围
python quant/mxfp_scaling_test.py your_tensor.pt --elem-format fp8_e5m2 --max-scale-exp 15 --min-scale-exp -15 --num-levels 31

# 测试FP4格式，更多缩放级别
python quant/mxfp_scaling_test.py your_tensor.pt --elem-format fp4_e2m1 --num-levels 51
```

## 技术细节

### 缩放策略

工具测试从最大值对齐（maximum alignment）到最小值对齐（minimum alignment）的不同缩放策略：

1. **最大值对齐**: 缩放使张量的绝对值最大值刚好在格式的最大可表示值范围内，避免上溢出
2. **最小值对齐**: 缩放使张量的绝对值最小值刚好在格式的最小可表示值范围内，避免下溢出
3. **中间级别**: 在这两个极端之间均匀分布的缩放级别

**对齐计算逻辑：**
- **最大对齐指数**: `floor(log2(tensor.abs().max() / format.max_norm))`
- **最小对齐指数**: `ceil(log2(tensor.abs().min() / format.min_norm))`

其中`tensor.abs().min()`只考虑非零值，确保下溢出分析的正确性。

### 下溢出分析

下溢出分析在量化前进行，检测：
- 非零值但小于最小可表示值的元素
- 会被刷新为零的元素
- 下溢出和刷新统计信息

### 可视化特性

- 动态调整图表范围以适应数据分布
- 突出显示关键边界值
- 提供详细的统计信息框
- 支持对数刻度的误差指标

## 注意事项

1. **输入格式**: 工具期望输入为BF16格式的张量文件
2. **内存使用**: 大张量可能需要较多内存，建议在GPU上运行
3. **计算时间**: 更多缩放级别会增加计算时间
4. **精度**: 指标计算使用float32精度以确保准确性

## 故障排除

### 常见问题

1. **导入错误**: 确保在Megatron-LM根目录运行脚本
2. **内存不足**: 减小张量大小或缩放级别数量
3. **格式不支持**: 检查elem-format参数是否支持
4. **文件格式**: 确保输入文件是有效的PyTorch张量文件

### 输出目录管理

**默认行为：**
- 如果不指定`--output-dir`，工具会自动创建`draw/scaling_analysis/{tensor_name}/`目录
- `{tensor_name}`是输入文件名（不包含扩展名）

**自定义输出目录：**
```bash
# 使用默认目录（基于tensor名称）
python quant/mxfp_scaling_test.py my_tensor.pt

# 指定自定义输出目录
python quant/mxfp_scaling_test.py my_tensor.pt --output-dir ./my_results/

# 为不同格式指定不同目录
python quant/mxfp_scaling_test.py my_tensor.pt --elem-format fp8_e4m3 --output-dir ./results_fp8_e4m3/
python quant/mxfp_scaling_test.py my_tensor.pt --elem-format fp8_e5m2 --output-dir ./results_fp8_e5m2/
```

### 调试选项

使用`--no-plots`选项跳过图表生成以加快测试：
```bash
python quant/mxfp_scaling_test.py input.pt --no-plots
```

## 扩展功能

工具设计为可扩展的，可以轻松添加：
- 新的量化格式支持
- 额外的精度指标
- 不同的可视化选项
- 批量处理多个文件
