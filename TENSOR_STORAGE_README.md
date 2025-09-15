# Tensor存储代码整理总结

## 整理完成情况

已成功将tensor存储相关代码进行了整理简化，保留了功能最强的一版本。

### ✅ 已完成的工作

1. **分析了所有tensor存储代码** - 识别出核心功能和冗余部分
2. **创建了统一存储脚本** - 整合了所有tensor收集功能
3. **更新了主脚本** - 添加了tensor收集和可视化的统一接口
4. **清理了冗余文件** - 删除了4个重复的存储脚本

### 🎯 最终结果

**保留的核心文件：**
- `megatron/core/tensor_saver.py` - 核心tensor保存工具模块（751行）
- `run_tensor_collection.sh` - 统一tensor收集脚本（新增）
- `run_tensor_draw.sh` - 统一分析脚本（已更新）

**删除的冗余文件：**
- `quick_tensor_collection.sh`
- `run_single_quant_type.sh`
- `run_wikipedia_tensor_collection.sh`
- `fixed_data_collection.sh`

### 🚀 统一脚本的主要特性

#### 1. `run_tensor_collection.sh` - 统一Tensor收集脚本
- **多种模式支持**：single（单次）、batch（批量）、quick（快速测试）
- **智能监控**：自动监控tensor生成进度，稳定后自动停止
- **错误处理**：完善的错误检查和恢复机制
- **统计报告**：详细的收集结果统计和分析

#### 2. `run_tensor_draw.sh` - 统一分析脚本（已更新）
- **三种模式**：collect（收集）、visualize（可视化）、both（两者）
- **自动集成**：可以自动调用tensor收集脚本
- **参数统一**：统一的参数接口和错误处理

### 📁 简化的目录结构

```
Megatron-LM/
├── megatron/core/tensor_saver.py    # 核心tensor保存工具
├── run_tensor_collection.sh         # 统一tensor收集脚本
├── run_tensor_draw.sh              # 统一分析脚本
└── script/visualization/
    ├── tensor_visualizer.py        # 统一可视化工具
    └── README.md                   # 可视化文档
```

### 🎯 使用方式

#### 1. 只收集tensor
```bash
# 收集单个量化类型（默认1个iteration）
./run_tensor_collection.sh single mxfp8

# 收集3个micro_batch的数据（使用命令行参数）
./run_tensor_collection.sh --mode single --quant-type mxfp8 --control-iter 3 --collect-micro-batches 2

# 批量收集所有类型
./run_tensor_collection.sh batch

# 快速测试收集
./run_tensor_collection.sh quick hifp8
```

#### 2. 只进行可视化
```bash
# 可视化现有tensor
./run_tensor_draw.sh visualize

# 使用命令行参数
./run_tensor_draw.sh --mode visualize --tensor-dir ./enhanced_tensor_logs --output-dir ./draw

# 使用默认参数
./run_tensor_draw.sh
```

#### 3. 收集+可视化一体化
```bash
# 收集并可视化（默认1个iteration）
./run_tensor_draw.sh both mxfp8

# 收集3个micro_batch的数据并可视化（使用命令行参数）
./run_tensor_draw.sh --mode both --quant-type mxfp8 --control-iter 3 --collect-micro-batches 2

# 使用默认参数
./run_tensor_draw.sh both
```

### 🔧 核心功能

#### Tensor收集功能
- **自动量化类型修改**：自动修改linear和attention层的量化类型
- **智能进度监控**：实时监控tensor生成进度，避免过度收集
- **多模式支持**：支持单次、批量、快速测试等不同模式
- **micro_batch控制**：通过 `control_iter` 参数控制收集的micro_batch数量
- **详细统计报告**：提供完整的收集结果统计和分析

#### 可视化功能
- **量化类型比较分析**：对比不同量化类型的tensor分布
- **HiFP8分布分析**：专门的HiFP8量化类型深度分析
- **全局统计信息**：生成JSON和文本格式的详细统计报告
- **多线程处理**：支持多线程加速处理大量tensor文件

### 📊 支持的量化类型

- **bf16**：Brain Float 16
- **mxfp8**：MX FP8
- **mxfp4**：MX FP4  
- **hifp8**：Hi FP8

### 🔧 control_iter 参数说明

`control_iter` 参数用于控制tensor收集的micro_batch数量，这是一个重要的性能控制参数：

#### 参数作用
- **控制收集范围**：决定收集多少个micro_batch的tensor数据
- **性能优化**：避免收集过多数据导致存储空间不足
- **测试友好**：可以快速收集少量数据进行测试

#### 参数值说明
- **1**（默认）：只收集第1个micro_batch的数据，适合快速测试
- **3-5**：收集前3-5个micro_batch的数据，适合中等规模分析
- **10+**：收集更多micro_batch的数据，适合深度分析
- **0**：收集所有iteration的数据（不推荐，可能导致存储问题）

#### 使用示例
```bash
# 快速测试（1个micro_batch）
./run_tensor_collection.sh single mxfp8

# 中等规模分析（3个micro_batch）
./run_tensor_collection.sh --mode single --quant-type mxfp8 --control-iter 3

# 深度分析（10个micro_batch）
./run_tensor_collection.sh --mode single --quant-type mxfp8 --control-iter 10
```

#### 注意事项
1. **存储空间**：micro_batch数量越多，需要的存储空间越大
2. **收集时间**：micro_batch数量越多，收集时间越长
3. **分析质量**：更多micro_batch的数据可以提供更全面的分析
4. **测试建议**：建议先用小值（1-3）进行测试，确认无误后再使用大值

### 🎉 主要改进

1. **代码简化**：从5个分散的脚本整合为2个统一脚本
2. **功能增强**：添加了智能监控、自动停止、详细统计等功能
3. **使用便捷**：统一的参数接口，支持多种使用模式
4. **维护性提升**：集中管理，减少重复代码，便于维护
5. **错误处理**：完善的错误检查和恢复机制

### 💡 使用建议

1. **首次使用**：建议先运行 `./run_tensor_collection.sh quick mxfp8` 进行快速测试
2. **批量收集**：使用 `./run_tensor_collection.sh batch` 收集所有量化类型
3. **一体化分析**：使用 `./run_tensor_draw.sh both` 进行收集和可视化一体化操作
4. **自定义分析**：使用 `./run_tensor_draw.sh visualize` 对现有数据进行可视化分析

现在tensor存储和可视化功能已经完全整合，使用更加便捷，维护更加简单！
