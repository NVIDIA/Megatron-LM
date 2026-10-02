<div align="center">

Megatron-LM and Megatron Core
=============================

<h4>GPU-optimized library for training transformer models at scale</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.19.0-green)](https://github.com/NVIDIA/Megatron-LM/releases)
[![license](https://img.shields.io/badge/license-Apache-blue)](./LICENSE)

<div align="left">

## About

This repository contains two components: **Megatron-LM** and **Megatron Core**.

**Megatron-LM** is a reference example that includes Megatron Core plus pre-configured training scripts, ideal for research teams, learning distributed training, and quick experimentation.

**Megatron Core** is a composable library with GPU-optimized building blocks for custom training frameworks. It provides transformer building blocks, advanced parallelism strategies (TP, PP, DP, EP, and CP), mixed precision support (FP16, BF16, FP8, and FP4), and model architectures, ideal for framework developers and ML engineers building custom training pipelines.

**[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** provides bidirectional Hugging Face ↔ Megatron checkpoint conversion with production-ready recipes.

## Getting Started

**Compatibility:** Python **3.12+** and PyTorch **2.6.0+** recommended. For source development, the tested NGC PyTorch base images are pinned in [docker/.ngc_version.dev](https://github.com/sbhavani/Megatron-LM/blob/codex/readme-refresh/docker/.ngc_version.dev) and [docker/.ngc_version.lts](https://github.com/sbhavani/Megatron-LM/blob/codex/readme-refresh/docker/.ngc_version.lts).

**Install from PyPI:**

```bash
uv pip install megatron-core
```

**Or clone and install from source:**

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
uv pip install -e .
```

> **Note:** Building from source can use a lot of memory. If the build runs out of memory, limit parallel compilation jobs by setting `MAX_JOBS` (for example, `MAX_JOBS=4 uv pip install -e .`).

For NVIDIA GPU Cloud (NGC) container setup and all installation options, review the **[Installation Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html)**.

- **[Your First Training Run](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)** - End-to-end training examples with data preparation
- **[Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)** - Scale training across GPUs with TP, PP, DP, EP, and CP
- **[Contribution Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)** - How to contribute to Megatron Core
- **[Style Guide](style-guide.md)** - Python style overrides

# Latest News

- **[2026/08]** **[Megatron Core 0.19](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.19.0)** - Introduces HybridModel, quantile-based MoE router balancing, packed-sequence MoE dispatch, and CUDA Graph-compatible activation offloading.
- **[2026/08]** **[Megatron Bridge performance summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html)** - August NeMo container benchmarks cover DeepSeek, Qwen3, GPT-OSS, and Nemotron on GB200 and GB300, with configurations and performance recipes.
- **[2026/07]** **[DeepSeek-V3 pretraining on GB300 NVL72](https://developer.nvidia.com/blog/setting-a-world-record-for-moe-pre-training-on-nvidia-gb300-nvl72/)** - Megatron Core achieves 1,648 TFLOPs/s per GPU for DeepSeek-V3 671B on 256 GPUs.
- **[2026/05]** **[DeepSeek-V4 initial support](https://github.com/NVIDIA/Megatron-LM/issues/4468)** - Megatron Core's `dev` branch includes the initial DeepSeek-V4 implementation; Megatron Bridge provides [conversion, inference, and pretraining recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/deepseek_v4).
- **[2026/04]** **[Advancing Emerging Optimizers for Accelerated LLM Training with NVIDIA Megatron](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/)** - Muon and other emerging optimizers are now supported in Megatron Core via the new **[Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers)** library.
- **[2026/03]** **[Scalable Training of Mixture-of-Experts Models with Megatron Core](https://arxiv.org/abs/2603.07685)** - Technical report on scaling MoE training with integrated optimizations for memory, communication, and computation.
- **[2026/03]** **[Implementing Falcon-H1 Hybrid Architecture in Megatron Core](https://developer.nvidia.com/blog/implementing-falcon-h1-hybrid-architecture-in-nvidia-megatron-core/)** - Technology Innovation Institute (TII) contributes Falcon-H1 hybrid transformer-Mamba architecture and BitNet ternary quantization support to Megatron Core.
- **[2026/03]** **[Megatron Core Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/4003)** - Roadmap for upcoming Megatron Core features and improvements.
- **[2026/01]** **[Dynamic Context Parallelism](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)** - Up to 1.48x speedup for variable-length sequence training with adaptive CP sizing.
- **[2025/12]** **Megatron Core development has moved to GitHub.** All development and CI now happen in the open, and community contributions are welcome.
- **[2025/10]** **[Megatron Dev Branch](https://github.com/NVIDIA/Megatron-LM/tree/dev)** - Early access branch with experimental features.
- **[2025/10]** **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - Bidirectional converter for interoperability between Hugging Face and Megatron checkpoints, featuring production-ready recipes for popular models.
- **[2025/08]** **[Mixture of Experts (MoE) Q3–Q4 2025 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - Comprehensive roadmap for MoE features including DeepSeek-V3, Qwen3, advanced parallelism strategies, FP8 optimizations, and Blackwell performance enhancements.
- **[2025/08]** **[GPT-OSS Model](https://github.com/NVIDIA/Megatron-LM/issues/1739)** - Megatron Core integrates advanced features including YaRN RoPE scaling, attention sinks, and custom activation functions.
- **[2025/06]** **[Megatron MoE Model Zoo](https://github.com/yanring/Megatron-MoE-ModelZoo)** - Best practices and optimized configurations for training DeepSeek-V3, Mixtral, and Qwen3 MoE models with performance benchmarking and checkpoint conversion tools.

[Previous News](docs/discussions/README.md#previous-news)

# Project Structure

```
Megatron-LM/
├── megatron/
│   ├── core/                    # Megatron Core (kernels, parallelism, building blocks)
│   │   ├── models/              # Transformer models
│   │   ├── transformer/         # Transformer building blocks
│   │   ├── tensor_parallel/     # Tensor parallelism
│   │   ├── pipeline_parallel/   # Pipeline parallelism
│   │   ├── distributed/         # Distributed training (FSDP, DDP)
│   │   ├── optimizer/           # Optimizers
│   │   ├── datasets/            # Dataset loaders
│   │   ├── inference/           # Inference engines and server
│   │   └── export/              # Model export (example: TensorRT-LLM)
│   ├── training/                # Training scripts
│   ├── legacy/                  # Legacy components
│   ├── post_training/           # Post-training (quantization, distillation, pruning, etc.)
│   └── rl/                      # Reinforcement learning (including RLHF)
├── examples/                    # Ready-to-use training examples
├── tools/                       # Utility tools
├── tests/                       # Comprehensive test suite
└── docs/                        # Documentation
```

# Performance Benchmarking

The [NVIDIA Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html) publishes training throughput, system configurations, parallelism settings, and links to [performance recipes and reproduction instructions](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/scripts/performance). Use the container and matching code version specified by those instructions when reproducing a result.

Selected results from the **26.08.01 NeMo container** (August 2026):

| Model | System | GPUs | Precision | Sequence length | Tokens/s/GPU | Model TFLOPs/s/GPU |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| DeepSeek-V3 | DGX-GB300 | 256 | MXFP8 | 4,096 | 6,288 | 1,636 |
| DeepSeek-V3 | DGX-GB200 | 256 | MXFP8 | 4,096 | 4,912 | 1,277 |
| Qwen3-235B-A22B | DGX-GB300 | 256 | MXFP8 | 4,096 | 8,832 | 1,306 |
| Nemotron 3 Ultra | DGX-GB300 | 256 | NVFP4 | 8,192 | 3,744 | 1,348 |

See the summary for the full batch sizes and parallelism configurations. MoE benchmarks use force-balanced expert routing and do not drop tokens. These results measure performance under the published benchmark conditions; throughput depends on the model, precision, hardware, and configuration.

# Roadmaps

- **[2026 Q2 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/4997)**
- **[2026 Q2 MoE-Specific Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/4815)** [`dev` branch first developments]

# Resources

## Getting Help

- 📖 **[Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)** - Official guides and API reference
- 🐛 **[Issues](https://github.com/NVIDIA/Megatron-LM/issues)** - Bug reports and feature requests

## Contributing

Contributions are welcome. Ways to contribute:

- 🐛 **Report bugs** - Help improve reliability
- 💡 **Suggest features** - Shape the future of Megatron Core
- 📝 **Improve docs** - Make Megatron Core more accessible
- 🔧 **Submit PRs** - Contribute code improvements

**→ [Contributing Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)**

## Citation

If you use Megatron in your research or project, use the following citation:

```bibtex
@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}
```
