<!---
   Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Layerwise Distributed Optimizer

> **Note**: The layerwise distributed optimizer is experimental. APIs and behavior may change in future releases.

## Overview

The layerwise distributed optimizer (`LayerWiseDistributedOptimizer`) reduces memory usage by distributing optimizer state across data-parallel (DP) ranks on a per-layer (per-parameter-tensor) basis. Unlike the [standard distributed optimizer](dist_optimizer.md), which shards optimizer state over a contiguous flat gradient buffer, the layerwise optimizer assigns entire parameter tensors to individual DP ranks. This design makes it straightforward to combine heterogeneous optimizers — for example, using [Muon](https://github.com/KellerJordan/Muon) for linear weight matrices while keeping AdamW for embeddings, biases, and LayerNorm parameters.

## Motivation

Training with the standard distributed optimizer requires all participating optimizers to operate on the same contiguous gradient buffer, which prevents mixing fundamentally different optimizers in a single training run. The layerwise optimizer lifts this restriction: because each rank owns complete parameter tensors rather than contiguous slices, any combination of Megatron-compatible optimizers can be chained together.

## How It Works

```
┌───────────────────────────────────────────────────────────────┐
│                         All DP Ranks                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Full Model │  │  Full Model │  │  Full Model │  ...       │
│  │  + Grads    │  │  + Grads    │  │  + Grads    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│       Rank 0           Rank 1           Rank 2                │
└───────────────────────────────────────────────────────────────┘
                             │
                    DDP AllReduce Grads
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│              Each Rank Updates Its Own Shard                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  p0, p7, p8 │  │  p1, p6, p9 │  │  p2, p5     │  ...       │
│  │  (updated)  │  │  (updated)  │  │  (updated)  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│       Rank 0           Rank 1           Rank 2                │
└───────────────────────────────────────────────────────────────┘
                             │
                       AllGather Params
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                All Ranks Have Full Updated Model              │
└───────────────────────────────────────────────────────────────┘
```

The step sequence for each training iteration is:

1. **Backward pass**: Each DP rank computes gradients for its full local model copy. Megatron DDP performs an all-reduce so every rank holds the globally averaged gradient.
2. **Optimizer step**: Each rank runs its optimizer only on the parameter subset assigned to it. Gradient norm and zero-count statistics are reduced globally.
3. **AllGather**: A single `all_gather_v` collects the updated parameter tensors from all ranks, so every rank again holds a fully updated model.

### Parameter Sharding

Parameters from all optimizer groups are sorted by their element count and assigned to DP ranks in a **ping-pong** pattern to balance memory usage. For example, with 4 ranks and 10 parameters `p0`–`p9` (sorted ascending by size):

| Rank 0 | Rank 1 | Rank 2 | Rank 3 |
|--------|--------|--------|--------|
| p0, p7, p8 | p1, p6, p9 | p2, p5 | p3, p4 |

For Mixture-of-Experts (MoE) models, expert parameters are sharded independently using the expert data-parallel (`expt_dp`) group, keeping expert and non-expert sharding orthogonal.

## Memory Savings

Each rank stores the full model in bf16 (as with standard DDP), but optimizer states — fp32 master weights, momentum buffers, and second moments — are only allocated for the local parameter shard. With data-parallel degree `d`, the approximate bytes per parameter are:

| Configuration | Without layerwise | With layerwise |
|---------------|:-----------------:|:--------------:|
| bf16 params, fp32 optimizer states (Adam) | 18 | 6 + 12/d |
| fp32 params, fp32 optimizer states (Adam) | 16 | 8 + 8/d  |

The savings are identical to the standard distributed optimizer for Adam. The key additional benefit is the ability to chain different optimizer types (e.g., Muon + AdamW) without modifying the gradient buffer layout.

## Current Integration: `dist_muon`

The layerwise distributed optimizer is currently exposed through the `dist_muon` optimizer choice, which pairs [Tensor-Parallel Muon](https://github.com/KellerJordan/Muon) for 2-D weight matrices with AdamW for all remaining parameters. Muon applies an orthogonalization step (Newton-Schulz iteration) to gradients before the momentum update, which can improve convergence compared to plain AdamW.

The `dist_muon` optimizer automatically:
- Identifies 2-D non-embedding weight matrices and assigns them to the Muon optimizer.
- Routes embeddings, biases, LayerNorm parameters, and 1-D tensors to AdamW.
- Wraps both optimizers inside `LayerWiseDistributedOptimizer` for coordinated parameter sharding and AllGather.

## Usage

### Command-Line

```bash
torchrun ... pretrain_gpt.py \
    --optimizer dist_muon \
    --lr 3e-4 \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --bf16 \
    # Do NOT set --use-distributed-optimizer
```

> **Important**: Do **not** use `--use-distributed-optimizer` together with `dist_muon`. The layerwise optimizer has its own sharding scheme and is incompatible with the standard distributed optimizer.

### Muon-Specific Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--muon-momentum` | `0.95` | Momentum coefficient for the internal SGD step |
| `--muon-use-nesterov` | `False` | Enable Nesterov-style momentum |
| `--muon-num-ns-steps` | `5` | Number of Newton-Schulz iteration steps for gradient orthogonalization |
| `--muon-scale-mode` | `"spectral"` | Scale factor mode applied to the orthogonalized update |
| `--muon-extra-scale-factor` | `1.0` | Additional multiplicative scale on the Muon update |
| `--muon-fp32-matmul-prec` | `"medium"` | Floating-point precision for internal fp32 matrix multiplications |
| `--muon-split-qkv` | `True` | Split QKV parameters for independent orthogonalization |
| `--muon-tp-mode` | `"blockwise"` | How Newton-Schulz is applied to tensor-parallel weights (`"blockwise"`, `"duplicated"`, `"distributed"`) |

### Programmatic Usage

```python
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection

config = OptimizerConfig(
    optimizer='adam',   # internally used for AdamW part; set by get_megatron_muon_optimizer
    lr=3e-4,
    weight_decay=0.01,
    bf16=True,
    use_distributed_optimizer=False,
    clip_grad=1.0,
)

pg_collection = ProcessGroupCollection.use_mpu_process_groups()

optimizer = get_megatron_muon_optimizer(
    config=config,
    model_chunks=model_chunks,
    layer_wise_distributed_optimizer=True,  # enables LayerWiseDistributedOptimizer
    pg_collection=pg_collection,
)
```

## Checkpointing

The layerwise optimizer supports both Megatron checkpoint formats.

### `torch_dist` format (recommended)

Uses distributed checkpointing via `sharded_state_dict`. The optimizer sets `replica_id` to `0` for the DP dimension of all `ShardedTensor` objects, ensuring only one copy of each optimizer state is written per parameter. This format supports checkpoint resharding (e.g., changing TP or PP degree between save and load).

```python
# Save
model_sharded_sd = model.sharded_state_dict()
optim_sd = optimizer.sharded_state_dict(model_sharded_sd)
megatron.core.dist_checkpointing.save(optim_sd, checkpoint_dir)

# Load
load_sharded_sd = optimizer.sharded_state_dict(model_sharded_sd, is_loading=True)
state_dict = megatron.core.dist_checkpointing.load(load_sharded_sd, checkpoint_dir)
optimizer.load_state_dict(state_dict)
```

### `torch` format

Each DP rank saves and loads its own optimizer state file:

```
checkpoint_dir/
├── model_optim_rng.pt           # model weights (shared)
├── layer_wise_optimizer_0.pt    # optimizer state for DP rank 0
├── layer_wise_optimizer_1.pt    # optimizer state for DP rank 1
└── ...
```

This is handled automatically by Megatron's `save_checkpoint` / `load_checkpoint` when `--ckpt-format torch` is set.

## Compatibility

| Feature | Supported |
|---------|:---------:|
| BF16 training | Yes |
| FP16 training | No |
| FP32 training | Yes |
| Tensor Parallelism (TP) | Yes |
| Pipeline Parallelism (PP) | Yes |
| Context Parallelism (CP) | Yes |
| Expert Parallelism (EP) / MoE | Yes |
| `--use-distributed-optimizer` | **No** (must be disabled) |
| Gradient accumulation fusion | Yes |
| Activation recomputation | Yes |
| `torch_dist` checkpointing | Yes |
| `torch` checkpointing | Yes |
| Checkpoint resharding (TP/PP change) | Yes (torch_dist only) |

## Comparison with Standard Distributed Optimizer

| Aspect | Standard Distributed Optimizer | Layerwise Distributed Optimizer |
|--------|--------------------------------|----------------------------------|
| Sharding granularity | Contiguous buffer slice | Whole parameter tensors |
| AllReduce → Reduce-Scatter | Yes (overlapped with backward) | No (full AllReduce via DDP) |
| AllGather | On contiguous buffer | On individual parameter tensors |
| Heterogeneous optimizers | No | Yes |
| Communication volume | Same | Same |
| Overlap with backward | Yes | No (AllGather is post-step) |
| Contiguous gradient buffer | Required | Not required |
| MoE expert parameters | Yes | Yes (via expt_dp group) |

The standard distributed optimizer has better communication-computation overlap because it uses reduce-scatter during the backward pass. The layerwise optimizer performs a full AllReduce via DDP and then AllGathers after the optimizer step. Future work may add overlap support.

## Limitations and Known Issues

- **Experimental**: The API is not yet stable and may change.
- **No fp16 support**: Only bf16 and fp32 training are supported.
- **No overlap with backward**: The AllGather after the optimizer step is not yet overlapped with computation.
- **Checkpointing edge cases**: Distributed checkpointing assumes a fixed DP size. Changing the number of DP ranks between save and load is not supported.
- **Insufficient parameters**: If the total number of parameters is less than the DP size, some ranks will own no parameters and participate in AllGather with empty tensors. This is handled correctly but may waste communication bandwidth.

## Implementation Reference

- **Core implementation**: `megatron/core/optimizer/layer_wise_optimizer.py`
- **Muon integration**: `megatron/core/optimizer/muon.py` — `get_megatron_muon_optimizer`
- **Training entry point**: `megatron/training/training.py`
- **Checkpointing**: `megatron/training/checkpointing.py`
- **Unit tests**: `tests/unit_tests/test_layer_wise_optimizer.py`
- **Distributed checkpointing tests**: `tests/unit_tests/dist_checkpointing/test_layer_wise_optimizer.py`
