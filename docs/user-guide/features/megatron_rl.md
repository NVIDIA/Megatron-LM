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

> **Note**: Megatron RL is under active development and primarily designed for research teams exploring RL post-training on modern NVIDIA hardware. For production deployments, use [**NeMo RL**](https://github.com/NVIDIA-NeMo/RL).

## Key Features

- **Decoupled Design** - Clean separation between agent/environment logic and RL implementation
- **Flexible Inference** - Support for Megatron, OpenAI, and HuggingFace inference backends
- **Trainer/Evaluator** - Manages rollout generation and coordinates with inference systems
- **Megatron Integration** - Native integration with Megatron Core inference system

## Architecture

### Components

**Agents & Environments**
- Accept inference handles
- Return experience rollouts with rewards
- Implement custom RL logic

**Trainer/Evaluator**
- Controls rollout generation
- Coordinates with inference systems
- Manages training loops

**Inference Interface**
- Provides `.generate(prompt, **generation_args)` endpoint
- Supports multiple backends (Megatron, OpenAI, HuggingFace)

## Use Cases

- RLHF (Reinforcement Learning from Human Feedback)
- Custom reward-based fine-tuning
- Policy optimization for specific tasks
- Research on RL post-training techniques

## Resources

- **[Megatron RL GitHub](https://github.com/NVIDIA/Megatron-LM/tree/dev/megatron/rl)** - Source code and documentation
- **[Megatron Core Inference](../../api-guide/core/transformer.md)** - Native inference integration
