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

The following table summarizes supported parallelism strategies.

| Strategy | Parallelism Objective | Best For |
|----------|---------------------|----------|
| **Data Parallelism (DP)** | Batch Dimension | Data Scalability, Standard Training |
| **Tensor Parallelism (TP)** | Individual Layers | Large Layers & Activation, GPU Memory Constraints |
| **Pipeline Parallelism (PP)** | Model Depth | Very Deep Models |
| **Context Parallelism (CP)** | Sequence Length | Long Sequences (8K+ Tokens) |
| **Expert Parallelism (EP)** | MoE Experts | Mixture-of-Experts Models |
| **Fully-Sharded Data Parallelism (Megatron-FSDP)** | Model State | Extremely Large Models & DP Interchangeability |

## Data Parallelism (DP)

### Standard Distributed Data Parallel (DDP)

Replicate the model across GPUs and split the batch.

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --data-parallel-sharding-strategy no_shard
```

Each GPU has a full copy of the model and processes a portion of the batch.

### Megatron Fully-Sharded Data Parallel (Megatron-FSDP)

Shard model parameters, gradients, and optimizer states across GPUs to reduce memory utilization.

```
--use-megatron-fsdp
--data-parallel-sharding-strategy optim_grads_params
--ckpt-format fsdp_dtensor
--init-model-with-meta-device
```

**Sharding Strategies**

`--data-parallel-sharding-strategy` supports the following options:

- `optim` - Shard optimizer states only (ZeRO-1)
- `optim_grads` - Shard gradients + optimizer (ZeRO-2)
- `optim_grads_params` - Shard parameters + gradients + optimizer (ZeRO-3)

If `--num-distributed-optimizer-instances` is > 1, then hierarchical data parallelism is enabled.

`--outer-dp-sharding-strategy` supports the following options:

- `no_shard` (**Hybrid-Sharded Data Parallelism**) - Replicate the model state across outer data parallel ranks.
- `optim` (**Hybrid-FSDP**) - Shard the optimizer state across the outer data parallel ranks.
  - Requires `--data-parallel-sharding-strategy optim_grads_params`.

**When to Use**

- Large models with large or fused compute kernels to hide communications under.
- Integrated with TP, CP, EP, and easily composable with heterogeneous parallelisms.
- With SM-reducing optimizations from NCCL and activation offloading from TransformerEngine.
- Using `fully_shard` without depending on Megatron-LM.

## Tensor Parallelism (TP)

Split individual model layers across GPUs. Recommended for large hidden dimensions.

```bash
--tensor-model-parallel-size 4  # 4-way tensor parallelism
--sequence-parallel              # Enable sequence parallelism (recommended)
```

**When to Use**

- Model layers do not fit on a single GPU
- Large hidden dimensions (4096+)
- Usually combined with DP and PP

## Pipeline Parallelism (PP)

Split model layers across GPUs vertically (by depth).

```bash
--pipeline-model-parallel-size 8              # 8 pipeline stages
--num-layers-per-virtual-pipeline-stage 4     # Virtual pipeline for load balancing
```

**When to Use**

- Very deep models (50+ layers)
- Combine with TP for large models
- Helps distribute memory across GPUs

## Context Parallelism (CP)

Split long sequences across GPUs for efficient long-context training.

```bash
--context-parallel-size 2           # 2-way context parallelism
--cp-comm-type p2p                  # Communication type
```

**When to Use**

- Long sequences (8K+ tokens)
- Reduces activation memory
- Can combine with TP, PP, DP

Refer to [Context Parallelism Deep Dive](features/context_parallel.md) for a detailed guide with performance analysis.

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

For a list of supported configurations, refer to [Megatron Bridge Supported Models](https://github.com/NVIDIA-NeMo/Megatron-Bridge#supported-models).

### Language Models

Recommended language model configurations:

| Model | Size | GPUs | TP | PP | CP | EP | Configuration Notes |
|-------|------|------|----|----|----|----|---------------------|
| **LLaMA-3** | 8B | 8 | 1 | 1 | 2 | 1 | CP=2 for long context (8K seqlen) |
| **LLaMA-3** | 70B | 64 | 4 | 4 | 2 | 1 | Balanced TP+PP for 70B scale |
| **LLaMA-3.1** | 405B | 1024 | 8 | 8 | 2 | 1 | 3D parallelism (TP+PP+CP) |
| **GPT-3** | 175B | 128-512 | 4 | 8 | 1 | 1 | Standard large model config |

### Mixture-of-Experts Models

Recommended mixture-of-experts configurations:

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

**Benefits**

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
1. Begin with **Data Parallelism** (DP) only.
2. Add **Tensor Parallelism** (TP) if the model does not fit.
3. Add **Pipeline Parallelism** (PP) for very large models.
4. Add **Context Parallelism** (CP) for long sequences.

### Memory Constraints
- Use **FSDP** to split model state per GPU.
- Use **TP** to split large layers.
- Use **PP** to split model depth.
- Enable **activation checkpointing or offloading** for extreme cases.

### Communication Bottlenecks
- Reduce **TP** degree (increases memory per GPU).
- Increase **PP** degree (may reduce efficiency).
- Use **CP** instead of larger TP for long sequences.

## Next Steps

- **API Reference**: Refer to [Tensor Parallel](../api-guide/core/tensor_parallel.md) and [Pipeline Parallel](../api-guide/core/pipeline_parallel.md) in the API documentation
- **Advanced Features**: Refer to [Megatron-FSDP](features/megatron_fsdp.md), [MoE](features/moe.md), and [Distributed Optimizer](features/dist_optimizer.md)
- **Performance Tuning**: Refer to the [NVIDIA NeMo Performance Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html)
