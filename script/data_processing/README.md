# 数据处理脚本

这个目录包含了用于处理各种数据集的脚本和工具。

## 脚本文件

### 数据处理脚本
- **`process_dolma_data.sh`** - 处理Dolma数据集
- **`process_wikipedia_data.sh`** - 处理Wikipedia数据集  
- **`process_c4_data.sh`** - 处理C4数据集
- **`process_custom_data.sh`** - 处理自定义数据集

### 工具脚本
- **`data_processing_utils.py`** - 数据处理工具函数

## 使用方法

### 1. Dolma数据处理
```bash
# 基本用法
./process_dolma_data.sh

# 自定义参数
./process_dolma_data.sh \
    "./dataset/dolma/**/*.json.gz" \
    "./dataset/dolma_processed" \
    32 \
    8 \
    "./model/llama3/" \
    "HuggingFaceTokenizer"
```

### 2. Wikipedia数据处理
```bash
# 基本用法
./process_wikipedia_data.sh

# 自定义参数
./process_wikipedia_data.sh \
    "./dataset/wikipedia/**/*.json" \
    "./dataset/wikipedia_processed" \
    16 \
    4 \
    "./model/llama3/"
```

### 3. C4数据处理
```bash
# 基本用法
./process_c4_data.sh

# 自定义参数
./process_c4_data.sh \
    "./dataset/c4/**/*.json" \
    "./dataset/c4_processed" \
    24 \
    6 \
    "./model/llama3/"
```

### 4. 自定义数据处理
```bash
# 基本用法
./process_custom_data.sh

# 自定义参数
./process_custom_data.sh \
    "./dataset/custom/**/*.json" \
    "./dataset/custom_processed" \
    16 \
    4 \
    "./model/llama3/" \
    "HuggingFaceTokenizer" \
    "true" \
    2048 \
    "false" \
    "text"
```

## 参数说明

### 必需参数
1. **输入路径** - 数据文件的路径（支持通配符）
2. **输出前缀** - 处理后文件的输出前缀
3. **工作进程数** - 并行处理的工作进程数
4. **分区数** - 数据分区的数量
5. **分词器模型** - 分词器模型路径

### 可选参数
- **分词器类型** - 默认为"HuggingFaceTokenizer"
- **追加EOD** - 是否在序列末尾追加EOD token
- **序列长度** - 最大序列长度，默认为2048
- **覆盖输出** - 是否覆盖已存在的输出文件

## 工具函数

### data_processing_utils.py

提供以下功能：
- 环境检查
- 数据集列表
- 模型列表
- 处理时间估算
- 最优参数推荐
- 数据验证
- 脚本生成

#### 使用方法
```bash
# 检查环境
python data_processing_utils.py --action check

# 列出可用数据集和模型
python data_processing_utils.py --action list

# 估算处理时间
python data_processing_utils.py --action estimate --input "./dataset/dolma/**/*.json.gz"

# 运行数据处理
python data_processing_utils.py --action process \
    --input "./dataset/dolma/**/*.json.gz" \
    --output "./dataset/dolma_processed" \
    --tokenizer "./model/llama3/" \
    --workers 32 \
    --partitions 8
```

## 注意事项

1. **数据格式**: 支持JSON、JSONL、TXT格式
2. **内存使用**: 大数据集建议使用更多分区
3. **磁盘空间**: 确保有足够的磁盘空间存储处理后的数据
4. **分词器**: 确保分词器模型路径正确
5. **权限**: 确保脚本有执行权限

## 输出文件

处理完成后会生成以下文件：
- `{output_prefix}.bin` - 二进制数据文件
- `{output_prefix}.idx` - 索引文件

这些文件可以直接用于Megatron-LM训练。
