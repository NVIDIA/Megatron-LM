<div align="center">

Megatron Core
=============
<h4>Production-ready library for building custom training frameworks</h4>

<div align="left">

## ⚡ Quick Start

```bash
# Install Megatron Core with required dependencies
pip install --no-build-isolation megatron-core[dev]

# Distributed training example (2 GPUs, mock data)
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

# What is Megatron Core?

**Megatron Core** is an open-source PyTorch-based library that contains GPU-optimized techniques and cutting-edge system-level optimizations. It abstracts them into composable and modular APIs, allowing full flexibility for developers and model researchers to train custom transformers at-scale on NVIDIA accelerated computing infrastructure.

## 🚀 Key Components

### GPU-Optimized Building Blocks
- **Transformer Components**: Attention mechanisms, MLP layers, embeddings
- **Memory Management**: Activation recomputation
- **FP8 Precision**: Optimized for NVIDIA Hopper, Ada, and Blackwell GPUs

### Parallelism Strategies
- **Tensor Parallelism (TP)**: Layer-wise parallelization (activation memory footprint can be further reduced using sequence parallelism)
- **Pipeline Parallelism (PP)**: Depth-wise model splitting and pipelining of microbatches to improve efficiency
- **Context Parallelism (CP)**: Long sequence handling ([documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html))
- **Expert Parallelism (EP)**: Split experts of an MoE model across multiple GPUs


## 🔗 Examples & Documentation

**Examples:**
- **[Simple Training Loop](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/run_simple_mcore_train_loop.py)** - Basic usage
- **[Multimodal Training](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/multimodal/)** - Vision-language models
- **[Mixture-of-Experts](https://github.com/yanring/Megatron-MoE-ModelZoo)** - MoE examples
- **[Mamba Models](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/mamba/)** - State-space models

**Documentation:**
- **[📚 API Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/index.html)** - Complete API documentation
- **[💡 Developer Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)** - Custom framework development

---

*For complete installation instructions, performance benchmarks, and ecosystem information, see the [main README](../README.md).*
