<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Megatron RL

Reinforcement learning library for post-training large language models at scale.

## Overview

[**Megatron RL**](https://github.com/NVIDIA/Megatron-LM/tree/dev/megatron/rl) adds native reinforcement learning capabilities to Megatron-LM for large-scale RL-based post-training of foundation models.

> **Note:** Megatron RL is under active development and primarily designed for research teams exploring RL post-training on modern NVIDIA hardware. For production deployments, use [**NeMo RL**](https://github.com/NVIDIA-NeMo/RL).

## Key Features

- **Decoupled Design** - Separates agent and environment logic from the core RL implementation
- **Inference Backends** - Megatron, OpenAI, and Hugging Face inference stacks
- **Trainer or Evaluator** - Manages rollout generation and coordinates with inference systems
- **Megatron Integration** - Native integration with Megatron Core inference system

## Architecture

### Components

**Agents and Environments**
- Accept inference handles
- Return experience rollouts with rewards
- Implement custom RL logic

**Trainer or Evaluator**
- Controls rollout generation
- Coordinates with inference systems
- Manages training loops

**Inference Interface**
- Exposes a `.generate(prompt, **generation_args)` endpoint
- Supports multiple backends (Megatron, OpenAI, Hugging Face)

## Use Cases

- RLHF (Reinforcement Learning from Human Feedback)
- Custom reward-based fine-tuning
- Policy optimization for specific tasks
- Research on RL post-training techniques

## Selected-token logprobs with tensor parallelism

RL training normally gathers tensor-parallel vocabulary shards before selecting the
next-token logprob. Set `--rl-use-vocab-parallel-selected-logprobs` to compute the
selected-token values directly from local vocabulary shards with Megatron Core's
vocabulary-parallel cross entropy. The option is disabled by default and does not
change the existing gathered-logits path.

The optimized path avoids materializing full-vocabulary logits on every tensor-parallel
rank and can increase feasible microbatch capacity. It does not use Liger kernels and
does not modify the vocabulary-parallel cross-entropy primitive.

Sequence packing is supported when context parallelism is disabled. The implementation
explicitly uses the gathered-logits path for CUDA Graph execution, batch-invariant mode,
context or pipeline parallelism greater than one, MTP, nonzero label smoothing, and
consumers that require full logits, entropy, top-N logprobs, or another output processor.

## Resources

- **[Megatron RL GitHub](https://github.com/NVIDIA/Megatron-LM/tree/dev/megatron/rl)**: Source code and documentation
- **[Megatron Core Inference](../../api-guide/core/transformer.md)**: Native inference integration
