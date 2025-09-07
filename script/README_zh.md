I'll help you reorganize and improve the robustness of all scripts in the `script` directory to make them more suitable for different experiments. Let me first explore the current structure and understand what scripts exist.


[1 tool called]

Now let me examine the content of these scripts to understand their current structure and identify areas for improvement:


[4 tools called]

Let me examine a few more scripts to understand the patterns and variations:


[3 tools called]

Now let me check the actual training scripts to understand the structure better:


[2 tools called]

Now I have a good understanding of the current script structure. Let me create a comprehensive plan to reorganize and improve the robustness of all scripts:


[1 tool called]

Based on my analysis, I can see several issues with the current script structure:

1. **Code duplication**: Many scripts have similar patterns with minor variations
2. **Hardcoded values**: Parameters are scattered and hard to modify
3. **Poor error handling**: No validation or error recovery
4. **Inconsistent structure**: Different naming conventions and organization
5. **Manual file modifications**: Using `sed` to modify source files is fragile
6. **No experiment tracking**: Limited logging and experiment management

Let me start by creating a robust, modular structure:


[4 tools called]

Now let me create the base training script:


[2 tools called]

Now let me create an experiment launcher script:


[3 tools called]

Now let me improve the data processing script:


[2 tools called]

Now let me create utility scripts for common operations:


[3 tools called]

Now let me create a comprehensive documentation file:


[2 tools called]

Now let me create a migration script to help users transition from the old scripts to the new structure:


[2 tools called]

Now let me test the new system to make sure everything works correctly:


[4 tools called]

Perfect! Let me now create a final summary script and complete the migration:


[3 tools called]

Let me run a final test to make sure everything works:


[1 tool called]

Perfect! Let me create a final summary of what has been accomplished:


[3 tools called]

## 总结

我已经成功重新整理了 `script` 目录下的所有脚本，使其健壮性更强且便于进行不同的实验。以下是完成的工作：

### �� 主要改进

#### 1. **模块化配置系统**
- **`config/common.sh`**: 通用工具函数、错误处理、日志系统
- **`config/models.sh`**: 模型特定配置（LLaMA 3 8B, LLaMA 3.2 1B）
- **`config/training.sh`**: 训练配置（标准、快速）、数据类型配置、分布式配置

#### 2. **统一的核心脚本**
- **`train_base.sh`**: 健壮的基础训练脚本，支持所有模型和配置
- **`experiment_launcher.sh`**: 简化的实验启动器，预定义常用实验
- **`process_data_improved.sh`**: 增强的数据处理脚本，支持验证和错误处理

#### 3. **实用工具脚本**
- **`utils/check_system.sh`**: 系统健康检查和验证
- **`utils/cleanup.sh`**: 自动清理工具，管理磁盘空间

#### 4. **迁移和设置工具**
- **`migrate_scripts.sh`**: 从旧脚本结构迁移到新结构
- **`setup.sh`**: 系统初始化和测试工具

### 🚀 新功能特性

#### **健壮性改进**
- ✅ 全面的错误处理和验证
- ✅ 输入参数验证
- ✅ 路径和文件存在性检查
- ✅ 系统依赖检查
- ✅ 优雅的错误恢复

#### **易用性提升**
- ✅ 统一的命令行接口
- ✅ 详细的帮助文档
- ✅ 干运行模式（dry-run）
- ✅ 彩色日志输出
- ✅ 进度跟踪和状态报告

#### **实验管理**
- ✅ 预定义实验配置
- ✅ 灵活的配置组合
- ✅ 实验验证和比较
- ✅ 自动日志记录

#### **维护工具**
- ✅ 系统健康检查
- ✅ 自动清理工具
- ✅ 磁盘空间管理
- ✅ 依赖验证

### �� 新的目录结构

```
script/
├── config/                    # 配置文件
│   ├── common.sh             # 通用工具和函数
│   ├── models.sh             # 模型配置
│   └── training.sh           # 训练配置
├── utils/                     # 工具脚本
│   ├── check_system.sh       # 系统检查
│   └── cleanup.sh            # 清理工具
├── train_base.sh             # 基础训练脚本
├── experiment_launcher.sh    # 实验启动器
├── process_data_improved.sh  # 数据处理脚本
├── migrate_scripts.sh        # 迁移工具
├── setup.sh                  # 设置工具
└── README.md                 # 详细文档
```

### 🔄 迁移映射

| 旧脚本 | 新命令 |
|--------|--------|
| `llama31-8b/pretrain_llama_wikipedia_fp8.sh` | `experiment_launcher.sh run llama3_8b_wikipedia_fp8` |
| `llama31-8b/pretrain_llama_wikipedia_bf16.sh` | `experiment_launcher.sh run llama3_8b_wikipedia_bf16` |
| `llama32-1b/wikipedia/pretrain_llama_wikipedia_fp8.sh` | `experiment_launcher.sh run llama32_1b_wikipedia_fp8` |
| `process_data.sh` | `process_data_improved.sh` |

### �� 快速开始

#### 1. **系统检查**
```bash
./script/utils/check_system.sh
```

#### 2. **初始化系统**
```bash
./script/setup.sh init
```

#### 3. **测试系统**
```bash
./script/setup.sh test
```

#### 4. **运行实验**
```bash
# 列出可用实验
./script/experiment_launcher.sh list

# 运行预定义实验
./script/experiment_launcher.sh run llama3_8b_mock_fast

# 自定义训练
./script/train_base.sh --model llama3_8b --experiment-name my_exp \
    --checkpoint-path checkpoints/llama3_8b/my_exp \
    --tensorboard-path tensorboard_logs/llama3_8b/my_exp \
    --use-mock-data --training-config fast
```

#### 5. **数据处理**
```bash
./script/process_data_improved.sh \
    --input './dataset/dolma/**/*.json.gz' \
    --output-prefix ./dataset/dolma_processed \
    --tokenizer-path ./model/llama3/ \
    --workers 32 --partitions 8
```

### 📚 完整文档

所有功能都有详细的文档说明，包括：
- 使用示例
- 配置选项
- 故障排除
- 最佳实践
- 迁移指南

这个新的脚本系统大大提高了 Megatron-LM 的可用性和维护性，使研究人员能够更轻松地进行各种实验，同时确保系统的稳定性和可靠性。