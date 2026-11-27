<div align="center">

Megatron-LM 与 Megatron Core
=============================

<h4>面向大规模 Transformer 模型训练的 GPU 优化计算库</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.19.0-green)](https://github.com/NVIDIA/Megatron-LM/releases)
[![license](https://img.shields.io/badge/license-Apache-blue)](./LICENSE)

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

<div align="left">

## 关于项目 (About)

本仓库包含两个核心组件：**Megatron-LM** 与 **Megatron Core**。

**Megatron-LM** 是一个参考范例项目，包含 Megatron Core 以及预先配置好的端到端训练脚本，非常适合科研团队、分布式训练学习者以及快速开展算法实验。

**Megatron Core** 是一个高度可组合的算法库，为定制化训练框架提供面向 GPU 深度优化的基础构件。它提供 Transformer 模块构件、前沿并行策略（张量并行 TP、流水线并行 PP、数据并行 DP、专家并行 EP 和上下文并行 CP）、混合精度训练支持（FP16、BF16、FP8 与 FP4）以及多种前沿模型架构，特别适合框架开发者和构建自研训练流水线的机器学习工程师。

**[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** 提供 Hugging Face ↔ Megatron 双向权重检查点转换工具以及生产就绪的训练配方。

## 快速开始 (Getting Started)

**兼容性**：推荐使用 Python **3.12+** 与 PyTorch **2.6.0+**。对于从源码开发，经过全面测试验证的 NGC PyTorch 基础镜像版本已固定在 [docker/.ngc_version.dev](https://github.com/sbhavani/Megatron-LM/blob/codex/readme-refresh/docker/.ngc_version.dev) 与 [docker/.ngc_version.lts](https://github.com/sbhavani/Megatron-LM/blob/codex/readme-refresh/docker/.ngc_version.lts) 中。

**通过 PyPI 安装：**

```bash
uv pip install megatron-core
```

**或克隆源码本地安装：**

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
uv pip install -e .
```

> **说明**：从源码编译可能消耗较多内存。若在编译过程中遭遇内存不足（OOM），可通过配置 `MAX_JOBS` 限制并行编译并发数（例如：`MAX_JOBS=4 uv pip install -e .`）。

关于 NVIDIA GPU Cloud (NGC) 容器环境搭建及全量安装配置选项，请参阅 **[安装指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html)**。

- **[首次训练上手指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)** - 包含数据预处理的端到端完整训练示例
- **[并行策略指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)** - 借助 TP、PP、DP、EP 与 CP 实现跨 GPU 扩展训练
- **[贡献指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)** - 如何向 Megatron Core 贡献代码
- **[代码规范指南](style-guide.md)** - Python 编码风格规范说明

# 最新动态 (Latest News)

- **[2026/10]** **[使用 NVIDIA Megatron Core 实现大规模按位确定性预训练](https://developer.nvidia.com/blog/scale-bitwise-deterministic-pretraining-with-nvidia-megatron-core/)** - 详解如何借助 Megatron Core 在大规模集群上严格复现训练结果、验证权重断点续训，并大幅削减确定性机制的性能开销。
- **[2026/08]** **[Megatron Core 0.19 正式发布](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.19.0)** - 引入 HybridModel 混合模型、基于分位数的 MoE 门控路由平衡算法、紧凑序列（Packed-Sequence）MoE 分发机制以及兼容 CUDA Graph 的激活值卸载（Activation Offloading）。
- **[2026/07]** **[在 GB300 NVL72 上进行 DeepSeek-V3 预训练](https://developer.nvidia.com/blog/setting-a-world-record-for-moe-pre-training-on-nvidia-gb300-nvl72/)** - Megatron Core 在 256 卡集群上训练 DeepSeek-V3 671B 创下每卡 1,648 TFLOPs/s 吞吐量的新纪录。
- **[2026/05]** **[DeepSeek-V4 初步支持上线](https://github.com/NVIDIA/Megatron-LM/issues/4468)** - Megatron Core 的 `dev` 分支已集成 DeepSeek-V4 初版实现；Megatron Bridge 同步提供[模型转换、推理以及预训练配方](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/deepseek_v4)。
- **[2026/04]** **[通过 NVIDIA Megatron 探索前沿优化器以加速大语言模型训练](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/)** - Megatron Core 现已通过全新开源的 **[Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers)** 算法库正式支持 Muon 以及其他前沿新兴优化器。

[历史动态 (Previous News)](docs/discussions/README.md#previous-news)

# 项目结构 (Project Structure)

```
Megatron-LM/
├── megatron/
│   ├── core/                    # Megatron Core（底层算子、并行策略、核心构件）
│   │   ├── models/              # Transformer、混合架构与多模态模型
│   │   ├── transformer/         # Transformer 基础模块
│   │   ├── tensor_parallel/     # 张量并行 (Tensor Parallelism)
│   │   ├── pipeline_parallel/   # 流水线并行 (Pipeline Parallelism)
│   │   ├── context_parallel/    # 上下文并行 (Context Parallelism)
│   │   ├── distributed/         # 分布式训练（Megatron FSDP、DDP）
│   │   ├── dist_checkpointing/  # 分布式检查点存储、加载与重分片 (Resharding)
│   │   ├── optimizer/           # 分布式优化器
│   │   ├── datasets/            # 数据集加载器
│   │   ├── inference/           # 推理引擎与服务端口
│   │   └── export/              # 模型导出（例如 TensorRT-LLM）
│   ├── training/                # 端到端训练脚本
│   └── post_training/           # 训练后优化（量化、蒸馏、剪枝等）
└── examples/                    # 开箱即用的训练样例
```

# 性能基准测试 (Performance Benchmarking)

[NVIDIA Megatron Bridge 性能总结](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html) 发布了训练吞吐量、系统硬件拓扑、并行策略配置以及指向[性能评测配方与复现说明](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/scripts/performance)的链接。在复现评测结果时，请确保使用上述指南所指定的官方容器及对应代码版本。

精选基于 **26.08.01 NeMo 官方容器**（2026 年 8 月）测得的基准数据：

| 模型 | 硬件系统 | GPU 数量 | 计算精度 | 序列长度 | 每卡每秒词元数 (Tokens/s/GPU) | 每卡模型算力 (Model TFLOPs/s/GPU) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| DeepSeek-V3 | DGX-GB300 | 256 | MXFP8 | 4,096 | 6,288 | 1,636 |
| DeepSeek-V3 | DGX-GB200 | 256 | MXFP8 | 4,096 | 4,912 | 1,277 |
| Qwen3-235B-A22B | DGX-GB300 | 256 | MXFP8 | 4,096 | 8,832 | 1,306 |
| Nemotron 3 Ultra | DGX-GB300 | 256 | NVFP4 | 8,192 | 3,744 | 1,348 |

有关完整 Batch Size 与并行策略配置，请参阅官方性能总结。MoE 评测采用强制平衡专家路由（Force-balanced expert routing）且不丢弃 Token。上述结果基于公开发布的基准测试环境测得；实际吞吐量取决于具体模型架构、计算精度、底层硬件规格与运行配置。

# 技术路线图 (Roadmaps)

- **[2026 Q3 Megatron Core 路线图](https://github.com/NVIDIA/Megatron-LM/issues/5676)**
- **[2026 Q3 MoE 路线图](https://github.com/NVIDIA/Megatron-LM/issues/6757)** [优先在 `dev` 分支开发]
- **[2026 Q3 Megatron FSDP v2 路线图](https://github.com/NVIDIA/Megatron-LM/issues/6812)** [实验性功能]

# 相关资源 (Resources)

## 获取帮助 (Getting Help)

- 📖 **[官方文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)** - 官方用户指南与 API 参考手册
- 🐛 **[Issue 反馈](https://github.com/NVIDIA/Megatron-LM/issues)** - Bug 报告与功能需求建议

## 参与贡献 (Contributing)

我们非常欢迎社区开发者的参与和贡献。常见贡献途径：

- 🐛 **报告 Bug** - 协助提升系统稳定性与可靠度
- 💡 **建议新功能** - 共同塑造 Megatron Core 的技术演进
- 📝 **完善文档** - 降低全球开发者的使用门槛
- 🔧 **提交 PR** - 贡献代码优化与特性实现

**→ [贡献指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)**

## 学术引用 (Citation)

如果 Megatron 在你的学术研究或工程项目中发挥了作用，欢迎使用以下 BibTeX 进行引用：

```bibtex
@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}
```

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月28日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
