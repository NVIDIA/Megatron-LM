# Overview

Megatron-Core and Megatron-LM are open-source tools that are typically used together to train LLMs at scale across GPUs. Megatron-Core expands the capability of Megatron-LM. Megatron Bridge connects Megatron-Core and Megatron-LM to other popular training models, such as Hugging Face.

## Megatron Core

NVIDIA Megatron Core is a library of essential building blocks for highly efficient large-scale generative AI training. It can be used to train models with unparalleled speed at scale across thousands of GPUs. It provides an extensive set of tools for multimodal and speech AI. It expands Megatron LM capabilities.

Megatron-Core contains GPU-optimized techniques featuring advanced parallelism strategies, optimizations like FP8 training, and support for the latest LLM, MoE, and multimodal architectures. It abstracts these techniques into composable and modular APIs.

Megatron-Core is compatible with all NVIDIA Tensor Core GPUs and popular LLM architectures such as GPT, BERT, T5, and RETRO.


**Composable library** with GPU-optimized building blocks for custom training frameworks.

**Best for:**

- **Framework developers** building on top of modular and optimized components
- **Research teams** needing custom training loops, optimizers, or data pipelines
- **ML engineers** requiring fault-tolerant training pipelines

**What you get:**

- Composable transformer building blocks (attention, MLP)
- Advanced parallelism strategies (TP, PP, DP, EP, CP)
- Pipeline schedules and distributed optimizers
- Mixed precision support (FP16, BF16, FP8)
- GPU-optimized kernels and memory management
- High-performance dataloaders and dataset utilities
- Model architectures (LLaMA, Qwen, GPT, Mixtral, Mamba)

## Megatron-LM

Megatron-LM is a reference implementation, with a lightweight large-scale LLM training framework. It offers a customizable native PyTorch training loop with fewer abstraction layers. It was designed for scaling transformer models to the multi-billion and trillion-parameter regimes under realistic memory and compute constraints. **It serves as a straightforward entry point for exploring Megatron-Core.**

It uses advanced parallelization techniques including model parallelism (tensor and pipeline), to allow models with billions of parameters to fit and train across large GPU clusters. It enables breakthroughs in large-scale NLP tasks. It splits model computations across many GPUs, overcoming single-GPU memory limits for training huge models, like GPT-style transformers.  


**Reference implementation** that includes Megatron Core plus everything needed to train models.

**Best for:**

- **Training state-of-the-art foundation models** at scale with cutting-edge performance on latest NVIDIA hardware
- **Research teams** exploring new architectures and training techniques
- **Learning distributed training** concepts and best practices
- **Quick experimentation** with proven model configurations

**What you get:**

- Pre-configured training scripts for GPT, LLaMA, DeepSeek, Qwen, and more.
- End-to-end examples from data prep to evaluation
- Research-focused tools and utilities



## Megatron Bridge

Megatron Bridge provides out-of-the-box bridges and training recipes for models built on top of base model architectures from Megatron Core.  

Megatron Bridge provides a robust, parallelism-aware pathway to convert models and checkpoints. This bidirectional converter performs on-the-fly, model-parallel-aware, per-parameter conversion, and full in-memory loading.

After training or modifying a Megatron model, you can convert it again for deployment or sharing.  

[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)



## Ecosystem Libraries

**Libraries used by Megatron Core:**

- **[Megatron Energon](https://github.com/NVIDIA/Megatron-Energon)** - Multi-modal data loader (text, images, video, audio) with distributed loading and dataset blending
- **[Transformer Engine](https://github.com/NVIDIA/TransformerEngine)** - Optimized kernels and FP8 mixed precision support
- **[Resiliency Extension (NVRx)](https://github.com/NVIDIA/nvidia-resiliency-ext)** - Fault tolerant training with failure detection and recovery

**Libraries using Megatron Core:**

- **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - Training library with bidirectional Hugging Face ↔ Megatron checkpoint conversion, flexible training loops, and production-ready recipes
- **[NeMo RL](https://github.com/NVIDIA-NeMo/RL)** - Scalable toolkit for efficient reinforcement learning with RLHF, DPO, and other post-training methods
- **[NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)** - Enterprise framework with cloud-native support and end-to-end examples
- **[Model Optimizer (ModelOpt)](https://github.com/NVIDIA/Model-Optimizer)** - Model optimization toolkit for quantization, pruning, distillation, speculative decoding, and more. Checkout end-to-end examples in [examples/post_training/modelopt](./examples/post_training/modelopt/).

**Compatible with:** [Hugging Face Accelerate](https://github.com/huggingface/accelerate), [Colossal-AI](https://github.com/hpcaitech/ColossalAI), [DeepSpeed](https://github.com/microsoft/DeepSpeed)

