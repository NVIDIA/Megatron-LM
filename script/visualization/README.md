# 可视化脚本

这个目录包含了用于可视化tensor数据的脚本和工具。

## 脚本文件

### 可视化脚本
- **`visualize_tensors.py`** - 完整的tensor可视化工具
- **`quick_visualize.py`** - 快速可视化脚本
- **`one_click_visualize.sh`** - 一键可视化脚本

## 功能特性

### 1. 完整的tensor可视化工具 (visualize_tensors.py)
- **分布图**: tensor数值分布直方图、箱线图、Q-Q图
- **热力图**: tensor数据的热力图可视化
- **对比图**: 不同量化类型的对比分析
- **统计图**: 统计信息汇总图表
- **Attention分析**: 专门的attention tensor分析

### 2. 快速可视化脚本 (quick_visualize.py)
- 生成基本的统计图表
- 快速分析tensor数据分布
- 生成统计信息文本文件

### 3. 一键可视化脚本 (one_click_visualize.sh)
- 自动检测tensor文件
- 运行快速和完整可视化
- 生成所有分析图表

## 使用方法

### 1. 一键可视化（推荐）
```bash
# 基本用法
./one_click_visualize.sh

# 自定义参数
./one_click_visualize.sh ./tensor_logs ./draw
```

### 2. 快速可视化
```bash
# 基本用法
python quick_visualize.py

# 自定义参数
python quick_visualize.py \
    --tensor_dir ./tensor_logs \
    --output_dir ./draw
```

### 3. 完整可视化
```bash
# 基本用法
python visualize_tensors.py

# 自定义参数
python visualize_tensors.py \
    --tensor_dir ./tensor_logs \
    --output_dir ./draw \
    --max_files 50
```

## 输出文件

### 目录结构
```
draw/
├── quick_analysis.png          # 快速分析图
├── tensor_stats.txt           # 统计信息文本
├── distributions/             # 分布图目录
├── heatmaps/                  # 热力图目录
├── comparisons/               # 对比图目录
├── statistics/                # 统计图目录
└── attention_maps/            # Attention分析图目录
```

### 图表类型
- **quick_analysis.png**: 包含4个子图的综合分析
  - 所有tensor数值分布直方图
  - 量化类型分布饼图
  - 层类型分布饼图
  - 操作类型分布饼图

- **distributions/**: 详细的tensor分布分析图
- **heatmaps/**: tensor数据的热力图
- **comparisons/**: 不同量化类型的对比图
- **statistics/**: 统计信息汇总图
- **attention_maps/**: attention tensor专门分析图

## 环境要求

### Python依赖
```bash
pip install matplotlib seaborn pandas scipy
```

### 环境变量
```bash
export TENSOR_SAVE_DIR="./tensor_logs"
export TENSOR_SAVE_ENABLED="true"
```

## 参数说明

### visualize_tensors.py
- `--tensor_dir`: tensor文件目录 (默认: ./tensor_logs)
- `--output_dir`: 输出图片目录 (默认: ./draw)
- `--max_files`: 最大处理文件数 (默认: 50)

### quick_visualize.py
- `--tensor_dir`: tensor文件目录 (默认: ./tensor_logs)
- `--output_dir`: 输出目录 (默认: ./draw)

### one_click_visualize.sh
- `$1`: tensor文件目录 (默认: ./tensor_logs)
- `$2`: 输出目录 (默认: ./draw)

## 使用场景

### 1. 量化研究
- 分析不同量化类型对tensor分布的影响
- 比较forward和backward pass的tensor特性
- 研究attention和linear层的tensor行为

### 2. 模型调试
- 可视化tensor数值分布
- 检测异常值和数值范围
- 分析tensor的统计特性

### 3. 性能分析
- 比较不同量化方法的性能
- 分析tensor的内存使用模式
- 优化量化策略

## 注意事项

1. **文件格式**: 支持.pt格式的tensor文件
2. **内存使用**: 大文件会自动采样以避免内存问题
3. **BFloat16支持**: 自动转换为Float32以支持numpy操作
4. **中文字体**: 可能需要安装中文字体以正确显示中文标签
5. **文件权限**: 确保脚本有执行权限

## 故障排除

### 常见问题
1. **ModuleNotFoundError**: 安装缺失的Python包
2. **字体警告**: 忽略中文字体警告，不影响功能
3. **内存不足**: 减少max_files参数或增加系统内存
4. **文件权限**: 使用chmod +x设置执行权限

### 调试技巧
- 使用quick_visualize.py进行快速测试
- 检查tensor文件是否正确生成
- 查看错误日志定位问题
- 使用小数据集进行测试
