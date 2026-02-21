<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Megatron Core User Guide

**Megatron Core** is a GPU-optimized library for training large language models at scale. It provides modular, composable building blocks for creating custom training frameworks with state-of-the-art parallelism strategies and performance optimizations.

Megatron Core offers a flexible, reusable foundation for building large-scale transformer training systems. **Megatron-LM** serves as a reference implementation demonstrating how to use Megatron Core components to train models with billions to trillions of parameters across distributed GPU clusters.

## Key Features

* Composable transformer building blocks (attention, MLP)
* Advanced parallelism strategies (TP, PP, DP, EP, CP)
* Pipeline schedules and distributed optimizers
* Mixed precision support (FP16, BF16, FP8)
* GPU-optimized kernels and memory management
* High-performance dataloaders and dataset utilities
* Model architectures (LLaMA, Qwen, DeepSeek, GPT, Mamba)


```{toctree}
:maxdepth: 2
:hidden:
:caption: About Megatron Core

get-started/overview
get-started/releasenotes
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Get Started

get-started/quickstart
get-started/install
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Basic Usage

user-guide/index
user-guide/parallelism-guide
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Supported Models

models/index
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Advanced Features

user-guide/features/index
user-guide/features/tokenizers
user-guide/features/fine_grained_activation_offloading
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Developer Guide

developer/contribute
developer/submit
developer/oncall
developer/generate_docs
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API Reference

api-guide/index
api-guide/router_replay
apidocs/index.rst
api-backwards-compatibility-check
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Resources

advanced/index
discussions/README
discussions/megatron-fsdp-user-guide/megatron-fsdp-user-guide
documentation
```