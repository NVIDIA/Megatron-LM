# Megatron Core User Guide

**Megatron Core** is a GPU-optimized library for training large language models at scale. It provides modular, composable building blocks for creating custom training frameworks with state-of-the-art parallelism strategies and performance optimizations.

Megatron Core offers a flexible, reusable foundation for building large-scale transformer training systems. **Megatron-LM** serves as a reference implementation demonstrating how to use Megatron Core components to train models with billions to trillions of parameters across distributed GPU clusters.

## Key Features

* Composable transformer building blocks (attention, MLP, etc.)
* Advanced parallelism strategies (TP, PP, DP, EP, CP)
* Pipeline schedules and distributed optimizers
* Mixed precision support (FP16, BF16, FP8)
* GPU-optimized kernels and memory management
* High-performance dataloaders and dataset utilities
* Model architectures (LLaMA, Qwen, DeepSeek, GPT, Mamba, etc.)

```{toctree}
:maxdepth: 2
:caption: Get Started

get-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: Basic Usage

user-guide/data-preparation
user-guide/training-examples
user-guide/parallelism-guide
```

```{toctree}
:maxdepth: 2
:caption: Supported Models

models/index
```

```{toctree}
:maxdepth: 2
:caption: Advanced Features

user-guide/features/moe
user-guide/features/context_parallel
user-guide/features/custom_fsdp
user-guide/features/dist_optimizer
user-guide/features/optimizer_cpu_offload
user-guide/features/pipeline_parallel_layout
user-guide/features/megatron_energon
user-guide/features/megatron_rl
user-guide/features/tokenizers
```

```{toctree}
:maxdepth: 1
:caption: Developer Guide

developer/contribute
developer/submit
developer/oncall
```

```{toctree}
:maxdepth: 2
:caption: Discussions

advanced/index
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api-guide/index
apidocs/index.rst
```
