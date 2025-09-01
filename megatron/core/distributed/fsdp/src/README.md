<div align="center">

# 🚀 Megatron-FSDP

</div>

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

</div>

## ✨ What is Megatron-FSDP?

**Megatron-FSDP** is an NVIDIA-developed PyTorch extension that provides a high-performance implementation of Fully Sharded Data Parallelism (FSDP). It offers seamless cross-compatibility with major deep learning frameworks and parallelism libraries, making it easy to scale your PyTorch models across multiple GPUs and nodes.

Megatron-FSDP can provide up to 25% speed up and 23% memory savings compared to FSDP2.

### Compatibility

- **[PyTorch DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html)**
- **[Megatron Core](https://github.com/NVIDIA/Megatron-LM)**
- **[TransformerEngine](https://github.com/NVIDIA/TransformerEngine)**

## ✨ Features

- **Easy Integration**: Simple `fully_shard` function for quick model parallelization
- **High Performance**: Optimized for NVIDIA GPUs with efficient memory management
- **Cross-Framework**: Works seamlessly with PyTorch, Huggingface Transformers, Megatron-LM, Megatron Bridge and TransformerEngine
- **Scalable**: Supports both single-node multi-GPU and multi-node distributed training
- **Flexible Configuration**: Configurable sharding strategies and process groups

## ⚡ Optimizations

- **Advanced Bucketing**: Data-type aware bucketing system to minimize the overhead of collective operations
- **Buffer Management**: Zero copy communication is achieved by reorganizing the storage of parameters and main grad with `ParamAndGradBuffer` class
- **Communication Overlapping**: Improved communication overlap of paramter all-gather and gradient reduce-scatter
- **User-Buffer-Registration NCCL communication**: Offload NCCL collective communication to NVL/IB Sharp to reduce GPU SM usage for communication
- **FP8 Mixed Precision with Transformer Engine**: Compatibility with Transformer Engine enables efficient FP8 mixed precision training
- **Gradient accumulate fusion support with Transformer Engine**: Remove the explicit gradient copy to the communication buffer in backwards pass

<!-- ## 📊 Performance  -->

<!-- ## 📦 Installation -->

## 🚀 Quick Start

### Basic Usage

Transform your PyTorch model to use Fully Sharded Data Parallelism with just a few lines:

```python
import torch
from megatron_fsdp import fully_shard

# Your existing model and optimizer
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Enable FSDP with Megatron-FSDP
model, optimizer = fully_shard(
    model,
    optimizer,
    fsdp_unit_modules=[YourTransformerBlock], # Modules to shard
)

# Your model is now ready for distributed training!
```

### Comparison with FSDP-2

We provide a similar approach for sharding the model with `fully_shard` function:

- No need to call `fully_shard` on all the submodules.
- One liner for the sharding change

Here is an FSDP2 usage example for better comparison

```python
import torch
from torch.distributed.fsdp import fully_shard

# Your existing model and optimizer
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Enable FSDP with FSDP2
for module in model.modules():
    if isinstance(module, YourTransformerBlock): # Sub-Modules to shard
        fully_shard(module)
fully_shard(model)

# Your model is now ready for distributed training!
```
