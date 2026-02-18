<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Parallelism Strategies Guide

Megatron Core supports multiple parallelism strategies that can be combined to efficiently train models from billions to trillions of parameters across thousands of GPUs.

## Overview

| Strategy | What it parallelizes | Best for |
|----------|---------------------|----------|
| **Data Parallelism (DP)** | Batch dimension | Standard training, most common |
| **Tensor Parallelism (TP)** | Individual layers | Large layers, GPU memory constraints |
| **Pipeline Parallelism (PP)** | Model depth | Very deep models |
| **Context Parallelism (CP)** | Sequence length | Long sequences (8K+ tokens) |
| **Expert Parallelism (EP)** | MoE experts | Mixture-of-Experts models |

## Data Parallelism (DP)

Replicate the model across GPUs and split the batch.

### Standard Data Parallel (DDP)

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --data-parallel-sharding-strategy no_shard
```

Each GPU has a full copy of the model and processes a portion of the batch.

### Fully Sharded Data Parallel (FSDP)

Shard model parameters, gradients, and optimizer states to reduce memory:

```bash
# Megatron FSDP (~15% faster than PyTorch FSDP2)
--use-megatron-fsdp \
--data-parallel-sharding-strategy optim_grads_params
```

**Sharding strategies:**
- `optim` - Shard optimizer states only (ZeRO-1)
- `optim_grads` - Shard gradients + optimizer (ZeRO-2)
- `optim_grads_params` - Shard parameters + gradients + optimizer (ZeRO-3)

## Tensor Parallelism (TP)

Split individual model layers across GPUs. Recommended for large hidden dimensions.

```bash
--tensor-model-parallel-size 4  # 4-way tensor parallelism
--sequence-parallel              # Enable sequence parallelism (recommended)
```

**When to use:**
- Model layers don't fit on single GPU
- Large hidden dimensions (4096+)
- Usually combined with DP and PP

## Pipeline Parallelism (PP)

Split model layers across GPUs vertically (by depth).

```bash
--pipeline-model-parallel-size 8              # 8 pipeline stages
--num-layers-per-virtual-pipeline-stage 4     # Virtual pipeline for load balancing
```

**When to use:**
- Very deep models (50+ layers)
- Combine with TP for large models
- Helps distribute memory across GPUs

## Context Parallelism (CP)

Split long sequences across GPUs for efficient long-context training.

```bash
--context-parallel-size 2           # 2-way context parallelism
--cp-comm-type p2p                  # Communication type
```

**When to use:**
- Long sequences (8K+ tokens)
- Reduces activation memory
- Can combine with TP, PP, DP

**→ [Context Parallelism Deep Dive](features/context_parallel.md)** - Detailed guide with performance analysis

## Expert Parallelism (EP)

Distribute experts across GPUs in Mixture-of-Experts models.

```bash
--expert-model-parallel-size 8  # 8-way expert parallelism
--num-experts 64                # 64 experts per MoE layer
--moe-grouped-gemm              # Optimize expert computation
```

**Important:** When combining EP with TP, you **must enable Sequence Parallelism**:

```bash
--tensor-model-parallel-size 4
--expert-model-parallel-size 8
--sequence-parallel  # Required when using TP + EP
```

## Parallelism Selection Guide

Recommended configurations based on [NVIDIA NeMo production setups](https://github.com/NVIDIA/NeMo/tree/main/scripts/performance/recommended_model_configs):

### Language Models

| Model | Size | GPUs | TP | PP | CP | EP | Configuration Notes |
|-------|------|------|----|----|----|----|---------------------|
| **LLaMA-3** | 8B | 8 | 1 | 1 | 2 | 1 | CP=2 for long context (8K seqlen) |
| **LLaMA-3** | 70B | 64 | 4 | 4 | 2 | 1 | Balanced TP+PP for 70B scale |
| **LLaMA-3.1** | 405B | 1024 | 8 | 8 | 2 | 1 | 3D parallelism (TP+PP+CP) |
| **GPT-3** | 175B | 128-512 | 4 | 8 | 1 | 1 | Standard large model config |

### Mixture-of-Experts Models

| Model | Size | GPUs | TP | PP | CP | EP | Configuration Notes |
|-------|------|------|----|----|----|----|---------------------|
| **Mixtral** | 8x7B | 64 | 1 | 4 | 1 | 8 | EP=8 for 8 experts |
| **Mixtral** | 8x22B | 256 | 4 | 4 | 1 | 8 | TP+PP+EP for large MoE |
| **DeepSeek-V3** | 671B | 1024 | 2 | 16 | 1 | 64 | Massive MoE with 256 experts |

## Combining Strategies

### Total GPU Count

The total number of GPUs is calculated as:

```
Total GPUs = TP × PP × CP × EP × DP
```

### Example: LLaMA-3 70B on 64 GPUs

```bash
# TP=4, PP=4, CP=2, DP=2 => 4 × 4 × 2 × 2 = 64 GPUs
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 4 \
    --context-parallel-size 2 \
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 8192 \
    --micro-batch-size 1 \
    --global-batch-size 512 \
    --bf16
```

## Performance Optimizations

### Communication Overlap

Enable overlapping of communication with computation:

```bash
--overlap-grad-reduce      # Overlap gradient reduction with backward pass
--overlap-param-gather     # Overlap parameter gathering with forward pass
--tp-comm-overlap          # Overlap TP communication
```

### Distributed Optimizer

Recommended for all multi-GPU training:

```bash
--use-distributed-optimizer
```

Benefits:
- Faster checkpointing
- Reduced memory when combined with FSDP
- Better performance at scale

### Sequence Parallelism

Always enable when using TP:

```bash
--sequence-parallel
```

Reduces activation memory by sharding sequence dimension in LayerNorm and Dropout.

## Choosing the Right Strategy

### Start Simple
1. Begin with **Data Parallelism** (DP) only
2. Add **Tensor Parallelism** (TP) if model doesn't fit
3. Add **Pipeline Parallelism** (PP) for very large models
4. Add **Context Parallelism** (CP) for long sequences

### Memory Constraints
- Use **FSDP** to reduce memory per GPU
- Use **TP** to split large layers
- Use **PP** to split model depth
- Enable **activation checkpointing** for extreme cases

### Communication Bottlenecks
- Reduce **TP** degree (increases memory per GPU)
- Increase **PP** degree (may reduce efficiency)
- Use **CP** instead of larger TP for long sequences

## Next Steps

- **API Reference**: See [Tensor Parallel](../api-guide/core/tensor_parallel.md) and [Pipeline Parallel](../api-guide/core/pipeline_parallel.md) API documentation
- **Advanced Features**: Explore [Megatron FSDP](features/custom_fsdp.md) and [Distributed Optimizer](features/dist_optimizer.md)
- **Performance Tuning**: Check [NVIDIA NeMo Performance Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html)
