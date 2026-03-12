<div align="center">

Megatron-LM and Megatron Core
=============================

<h4>GPU-optimized library for training transformer models at scale</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.15.0-green)](./CHANGELOG.md)
[![license](https://img.shields.io/badge/license-Apache-blue)](./LICENSE)

<div align="left">

## About

This repository contains two components: **Megatron-LM** and **Megatron Core**.

**Megatron-LM** is a reference example that includes Megatron Core plus pre-configured training scripts. Best for research teams, learning distributed training, and quick experimentation.

**Megatron Core** is a composable library with GPU-optimized building blocks for custom training frameworks. It provides transformer building blocks, advanced parallelism strategies (TP, PP, DP, EP, CP), mixed precision support (FP16, BF16, FP8, FP4), and model architectures. Best for framework developers and ML engineers building custom training pipelines.

**[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** provides bidirectional Hugging Face ↔ Megatron checkpoint conversion with production-ready recipes.


## Getting Started

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

> **Note:** Building from source can use a lot of memory. If the build runs out of memory, limit parallel compilation jobs by setting `MAX_JOBS` (e.g. `MAX_JOBS=4 uv pip install -e .`).

For NGC container setup and all installation options, see the **[Installation Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html)**.

- **[Your First Training Run](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)** - End-to-end training examples with data preparation
- **[Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)** - Scale training across GPUs with TP, PP, DP, EP, and CP
- **[Contribution Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)** - How to contribute to Megatron Core


# Latest News

- **[2026/01]** **[Dynamic Context Parallelism](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)** - Up to 1.48x speedup for variable-length sequence training with adaptive CP sizing.
- **[2025/12]** **Megatron Core development has moved to GitHub!** All development and CI now happens in the open. We welcome community contributions.
- **[2025/10]** **[Megatron Dev Branch](https://github.com/NVIDIA/Megatron-LM/tree/dev)** - early access branch with experimental features.
- **[2025/10]** **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - Bidirectional converter for interoperability between Hugging Face and Megatron checkpoints, featuring production-ready recipes for popular models.
- **[2025/08]** **[MoE Q3-Q4 2025 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - Comprehensive roadmap for MoE features including DeepSeek-V3, Qwen3, advanced parallelism strategies, FP8 optimizations, and Blackwell performance enhancements.
- **[2025/08]** **[GPT-OSS Model](https://github.com/NVIDIA/Megatron-LM/issues/1739)** - Advanced features including YaRN RoPE scaling, attention sinks, and custom activation functions are being integrated into Megatron Core.
- **[2025/06]** **[Megatron MoE Model Zoo](https://github.com/yanring/Megatron-MoE-ModelZoo)** - Best practices and optimized configurations for training DeepSeek-V3, Mixtral, and Qwen3 MoE models with performance benchmarking and checkpoint conversion tools.
- **[2025/05]** Megatron Core v0.11.0 brings new capabilities for multi-data center LLM training ([blog](https://developer.nvidia.com/blog/turbocharge-llm-training-across-long-haul-data-center-networks-with-nvidia-nemo-framework/)).

<details>
<summary>Previous News</summary>

- **[2024/07]** Megatron Core v0.7 improves scalability and training resiliency and adds support for multimodal training ([blog](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-Megatron-Core-functionalities/)).
- **[2024/06]** Megatron Core added supports for Mamba-based models. Check out our paper [An Empirical Study of Mamba-based Language Models](https://arxiv.org/pdf/2406.07887) and [code example](https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba).
- **[2024/01 Announcement]** NVIDIA has released the core capabilities in **Megatron-LM** into [**Megatron Core**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) in this repository. Megatron Core expands upon Megatron-LM's GPU-optimized techniques with more cutting-edge innovations on system-level optimizations, featuring composable and modular APIs.

</details>


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
│   │   └── export/              # Model export (e.g. TensorRT-LLM)
│   ├── training/                # Training scripts
│   ├── legacy/                  # Legacy components
│   ├── post_training/           # Post-training (quantization, distillation, pruning, etc.)
│   └── rl/                      # Reinforcement learning (RLHF, etc.)
├── examples/                    # Ready-to-use training examples
├── tools/                       # Utility tools
├── tests/                       # Comprehensive test suite
└── docs/                        # Documentation
```

### Megatron-LM: Reference Implementation

**Reference implementation** that includes Megatron Core plus everything needed to train models.

**Best for:**

- **Training state-of-the-art foundation models** at scale with cutting-edge performance on latest NVIDIA hardware
- **Research teams** exploring new architectures and training techniques
- **Learning distributed training** concepts and best practices
- **Quick experimentation** with proven model configurations

**What you get:**

- Pre-configured training scripts for GPT, LLama, DeepSeek, Qwen, and more.
- End-to-end examples from data prep to evaluation
- Research-focused tools and utilities

### Megatron Core: Composable Library

**Composable library** with GPU-optimized building blocks for custom training frameworks.

**Best for:**

- **Framework developers** building on top of modular and optimized components
- **Research teams** needing custom training loops, optimizers, or data pipelines
- **ML engineers** requiring fault-tolerant training pipelines

**What you get:**

- Composable transformer building blocks (attention, MLP, etc.)
- Advanced parallelism strategies (TP, PP, DP, EP, CP)
- Pipeline schedules and distributed optimizers
- Mixed precision support (FP16, BF16, FP8)
- GPU-optimized kernels and memory management
- High-performance dataloaders and dataset utilities
- Model architectures (LLaMA, Qwen, GPT, Mixtral, Mamba, etc.)

## Ecosystem Libraries

**Libraries used by Megatron Core:**

- **[Megatron Energon](https://github.com/NVIDIA/Megatron-Energon)** 📣 **NEW!** - Multi-modal data loader (text, images, video, audio) with distributed loading and dataset blending
- **[Transformer Engine](https://github.com/NVIDIA/TransformerEngine)** - Optimized kernels and FP8 mixed precision support
- **[Resiliency Extension (NVRx)](https://github.com/NVIDIA/nvidia-resiliency-ext)** - Fault tolerant training with failure detection and recovery

**Libraries using Megatron Core:**

- **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - Training library with bidirectional Hugging Face ↔ Megatron checkpoint conversion, flexible training loops, and production-ready recipes
- **[NeMo RL](https://github.com/NVIDIA-NeMo/RL)** - Scalable toolkit for efficient reinforcement learning with RLHF, DPO, and other post-training methods
- **[NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)** - Enterprise framework with cloud-native support and end-to-end examples
- **[TensorRT Model Optimizer (ModelOpt)](https://github.com/NVIDIA/TensorRT-Model-Optimizer)** - Model optimization toolkit for quantization, pruning, distillation, speculative decoding, and more. Checkout end-to-end examples in [examples/post_training/modelopt](./examples/post_training/modelopt/).

**Compatible with:** [Hugging Face Accelerate](https://github.com/huggingface/accelerate), [Colossal-AI](https://github.com/hpcaitech/ColossalAI), [DeepSpeed](https://github.com/microsoft/DeepSpeed)

# Installation

## 🐳 Docker (Recommended)

We strongly recommend using the previous releases of [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) rather than the latest one for optimal compatibility with Megatron Core release and testing. Our releases are always based on the previous month's NGC container, so this ensures compatibility and stability.

**Note:** The NGC PyTorch container constraints the python environment globally via `PIP_CONSTRAINT`. In the following examples we will unset the variable.

This container comes with all dependencies pre-installed with compatible versions and optimized configurations for NVIDIA GPUs:

- PyTorch (latest stable version)
- CUDA, cuDNN, NCCL (latest stable versions)
- Support for FP8 on NVIDIA Hopper, Ada, and Blackwell GPUs
- For best performance, use NVIDIA Turing GPU architecture generations and later

```bash
# Run container with mounted directories
docker run --runtime --nvidia --gpus all -it --rm \
  -v /path/to/megatron:/workspace/megatron \
  -v /path/to/dataset:/workspace/dataset \
  -v /path/to/checkpoints:/workspace/checkpoints \
  -e PIP_CONSTRAINT= \
  nvcr.io/nvidia/pytorch:25.04-py3
```

## Pip Installation

Megatron Core offers support for two NGC PyTorch containers:

- `dev`: Moving head that supports the most recent upstream dependencies
- `lts`: Long-term support of NGC PyTorch 24.01

Both containers can be combined with `mlm` which adds package dependencies for Megatron-LM on top of Megatron Core.

```bash
# Install the latest release dependencies
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation megatron-core[dev]
# For running an M-LM application:
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation megatron-core[mlm,dev]
```

```bash
# Install packages for LTS support NGC PyTorch 24.01
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation megatron-core[lts]
# For running an M-LM application:
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation megatron-core[mlm,lts]
```

For a version of Megatron Core with only torch, run:

```bash
pip install megatron-core
```

### Optional MoE Dependencies

For Mixture of Experts (MoE) training with the legacy Grouped GEMM support:

```bash
pip install --no-build-isolation megatron-core[moe]
```

**Note:** 
1. The `nv-grouped-gemm` package requires:
- CUDA toolkit (nvcc) with CUTLASS headers
- On Ubuntu/Debian: `apt-get install libcutlass-dev`
- GPU with compute capability >= 8.0

2. We recommend installing Transformer Engine (>=1.9) for more comprehensive grouped gemm support.
The legacy grouped gemm will only be used when
    - `--moe-grouped-gemm` is specified and
    - TE (>= 1.9) is not available or `--moe-use-legacy-grouped-gemm` is specified

## System Requirements

### Hardware Requirements

- **FP8 Support**: NVIDIA Hopper, Ada, Blackwell GPUs
- **Recommended**: NVIDIA Turing architecture or later

### Software Requirements

- **CUDA/cuDNN/NCCL**: Latest stable versions
- **PyTorch**: Latest stable version
- **Transformer Engine**: Latest stable version
- **Python**: 3.12 recommended

# Performance Benchmarking

For our latest performance benchmarking results, please refer to [NVIDIA Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html).

Our codebase efficiently trains models from 2B to 462B parameters across thousands of GPUs, achieving up to **47% Model FLOP Utilization (MFU)** on H100 clusters.

![Model table](images/model_table.png)

**Benchmark Configuration:**

- **Vocabulary size**: 131,072 tokens
- **Sequence length**: 4096 tokens
- **Model scaling**: Varied hidden size, attention heads, and layers to achieve target parameter counts
- **Communication optimizations**: Fine-grained overlapping with DP (`--overlap-grad-reduce`, `--overlap-param-gather`), TP (`--tp-comm-overlap`), and PP (enabled by default)

**Key Results:**

- **6144 H100 GPUs**: Successfully benchmarked 462B parameter model training
- **Superlinear scaling**: MFU increases from 41% to 47-48% with model size
- **End-to-end measurement**: Throughputs include all operations (data loading, optimizer steps, communication, logging)
- **Production ready**: Full training pipeline with checkpointing and fault tolerance
- *Note: Performance results measured without training to convergence*

## Weak Scaling Results

Our weak scaled results show superlinear scaling (MFU increases from 41% for the smallest model considered to 47-48% for the largest models); this is because larger GEMMs have higher arithmetic intensity and are consequently more efficient to execute.

![Weak scaling](images/weak_scaling.png)

## Strong Scaling Results

We also strong scaled the standard GPT-3 model (our version has slightly more than 175 billion parameters due to larger vocabulary size) from 96 H100 GPUs to 4608 GPUs, using the same batch size of 1152 sequences throughout. Communication becomes more exposed at larger scale, leading to a reduction in MFU from 47% to 42%.

![Strong scaling](images/strong_scaling.png)


# Roadmaps

- **[MoE Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - DeepSeek-V3, Qwen3, advanced parallelism, FP8 optimizations, and Blackwell enhancements


# Resources

## Getting Help

- 📖 **[Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)** - Official documentation
- 🐛 **[Issues](https://github.com/NVIDIA/Megatron-LM/issues)** - Bug reports and feature requests

## Contributing

We ❤️ contributions! Ways to contribute:

- 🐛 **Report bugs** - Help us improve reliability
- 💡 **Suggest features** - Shape the future of Megatron Core
- 📝 **Improve docs** - Make Megatron Core more accessible
- 🔧 **Submit PRs** - Contribute code improvements

**→ [Contributing Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)**

## Citation

If you use Megatron in your research or project, we appreciate that you use the following citations:

```bibtex
@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}
```
