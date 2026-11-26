{/*
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
*/}

# Muon Optimizer and LayerWiseDistributedOptimizer

Megatron Core integrates the **Muon** (Momentum Orthogonalized by Newton-Schulz) optimizer from [Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers) and pairs it with `LayerWiseDistributedOptimizer` when training with distributed optimizer sharding.

This guide covers parameter routing, distributed sharding with `LayerWiseDistributedOptimizer`, tensor-parallel orthogonalization modes, Megatron-FSDP v2 support, and known configuration constraints.

## Prerequisites

Muon and `adaptive_muon` require `emerging-optimizers >= 0.2`:

```bash
pip install emerging-optimizers
```

Additional version requirements apply to optional features:

- `emerging-optimizers >= 0.3.0`: batched Newton-Schulz (`--muon-ns-batch-size` greater than `1`) under `--muon-tp-mode layer_sharded`.
- `emerging-optimizers >= 0.4.0`: Triton SYRK Gram-matrix kernel (`--muon-use-syrk`), together with CUDA and Triton `>= 3.4.0` on validated SM architectures (`(8, 0)`, `(9, 0)`, `(10, 0)`, `(10, 3)`).
- `emerging-optimizers >= 0.5.0a0`: combining `--muon-use-syrk` with `--muon-ns-batch-size` greater than `1`.

## Parameter Routing (Matrix vs. Scalar Fallback)

When `--optimizer muon` is enabled, Megatron Core automatically splits model parameters into two optimization paths using `ParamKey` overrides:

1. **Matrix parameters** (`TensorParallelMuon`, `LayerShardedMuon`, or `FsdpMuon`): all trainable 2D weights that are not embeddings/output heads and do not carry `use_muon=False`.
2. **Scalar and excluded parameters** (`adam` or `lion`): 1D parameters (biases, LayerNorm/RMSNorm weights), embedding and output weights (`is_embedding_or_output_parameter`), and any parameter explicitly flagged with `use_muon=False`.

Select the auxiliary scalar optimizer with `--muon-scalar-optimizer {adam,lion}` (defaults to `adam`). When using `lion`, configure `--lion-beta1` (default `0.95`) and `--lion-beta2` (default `0.98`).

Note that `--optimizer dist_muon` is deprecated; use `--optimizer muon --use-distributed-optimizer` instead.

## How LayerWiseDistributedOptimizer Works

Standard [Distributed Optimizer](dist_optimizer.md) flattens parameters into a single buffer and shards them across data-parallel (DP) ranks at arbitrary byte offsets. Because Newton-Schulz operates on complete 2D matrices, slicing individual matrices across DP byte boundaries breaks matrix orthogonalization.

Passing `--optimizer muon --use-distributed-optimizer` automatically enables `OptimizerConfig.use_layer_wise_distributed_optimizer` (and clears `use_distributed_optimizer` inside `OptimizerConfig.finalize_optimizer_config()`). `LayerWiseDistributedOptimizer` shards optimizer states across DP ranks at whole-parameter granularity rather than splitting individual tensors:

1. Each rank only stores FP32 master weights and Muon momentum state for the parameters it owns.
2. Each rank runs Newton-Schulz and updates its assigned parameters independently across DP (`dp_cp` for dense weights, `expt_dp` for expert weights).
3. Updated weights are synchronized across DP ranks via all-gather (overlapped with forward when `--overlap-param-gather` is enabled).

### Split Param Layout vs. Legacy Ping-Pong Mode

By default (`use_layer_wise_param_layout=True`), Megatron separates DDP parameter buffers before wrapping:

- Muon-managed 2D weights live in shard-aligned `LayerWiseDistributedOptimizer` buffers (`FullParamLayout`).
- Non-Muon scalar parameters live in separate byte-sharded buffers managed by a standard `DistributedOptimizer`, chained via `ChainedOptimizer`.

This avoids extra alignment padding and improves communication efficiency. If your run uses non-emerging expert-parallel parameter groups or requires the legacy single-buffer behavior, pass `--no-use-layer-wise-param-layout` to fall back to the legacy ping-pong `LayerWiseDistributedOptimizer` path (note that different BF16 reduction order can change loss curves slightly relative to the split layout).

## Tensor Parallel Modes (`--muon-tp-mode`)

When 2D weights are sharded across Tensor Parallel (`TP`) or Generalized Tensor Parallel (`GTP_remat`) ranks, `--muon-tp-mode` controls how Newton-Schulz computes updates. See also [Generalized Tensor Parallelism](../../api-guide/core/generalized_tensor_parallel.md) for GTP interaction details.

| Mode | Collective Pattern | Math Equivalence to TP=1 | Notes |
|---|---|---|---|
| `duplicated` (default) | All-gather shards, run full NS on every rank, keep local slice | TP-invariant | Required for layout-invariant `--muon-split-qkv` under `GTP_remat`. |
| `blockwise` | Local NS per shard with no TP collective | Depends on TP/GTP size | Zero extra communication; alters update rule as parallelism changes. |
| `distributed` | Small-Gram all-reduce across TP/GTP per NS step | TP-invariant | Cuts redundant NS FLOPs vs `duplicated` without full tensor replication. |
| `auto` | Cost model per dense weight choosing `duplicated` vs `distributed` | TP-invariant | Uses device hardware profile when available, otherwise FLOPs/domain heuristic. |
| `layer_sharded` | Two-stage `all_to_all` over `(gtp_remat x tp)` to one NS home rank per weight | TP-invariant | Selects `LayerShardedMuon`; requires `--use-distributed-optimizer` and `--muon-no-split-qkv`. |

Under `--muon-tp-mode layer_sharded`:

- `--muon-ns-batch-size` (default `1`) fuses up to N identically shaped matrices per rank into batched Newton-Schulz calls, reducing kernel-launch overhead on MoE experts while trading bitwise parity with `baddbmm` rounding.
- `--muon-no-concurrent-groups` serializes parameter groups on one CUDA stream instead of overlapping `all_to_all` collectives and Newton-Schulz across streams, lowering transient peak memory.
- `--optimizer adaptive_muon` is not supported with `layer_sharded`.

## Configuration Options

Common CLI flags in `megatron/training/arguments.py`:

| Flag | Default | Description |
|---|---|---|
| `--optimizer` | `adam` | Set to `muon` (or `adaptive_muon` for Adam/NorMuon second-moment scaling). |
| `--muon-momentum` | `0.9` | Internal SGD momentum (`OptimizerConfig.muon_momentum` defaults to `0.95` when constructed directly in Python). |
| `--muon-nesterov` | `False` | Enable Nesterov momentum before Newton-Schulz orthogonalization. |
| `--muon-no-split-qkv` | Split enabled (`muon_split_qkv=True`) | Disable splitting fused `linear_qkv` weights into separate Q, K, and V projections before orthogonalizing. |
| `--muon-num-ns-steps` | `5` | Number of Newton-Schulz iterations. |
| `--muon-coefficient-type` | `quintic` | Coefficient schedule (validated against `NSCoeffT` in the installed `emerging-optimizers`, e.g., `simple`, `quintic`, `polar_express`, `aol`). |
| `--muon-scale-mode` | `spectral` | Output scaling rule: `spectral`, `unit_rms_norm` (often paired with MuP), or `shape_scaling`. |
| `--muon-extra-scale-factor` | `1.0` | Scalar multiplier applied after `--muon-scale-mode`. |
| `--muon-fp32-matmul-prec` | `medium` | Internal matmul precision inside Newton-Schulz (`low`, `medium`, `high`). |
| `--muon-use-syrk` | `False` | Use the Triton SYRK kernel for Gram matrices (`--muon-fp32-matmul-prec medium` only). |
| `--muon-scalar-optimizer` | `adam` | Optimizer (`adam` or `lion`) for embeddings, norms, biases, and excluded weights. |

## Usage Examples

### Standard GPT Pretraining with Distributed Muon

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --num-layers 12 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --train-iters 5000 \
    --optimizer muon \
    --use-distributed-optimizer \
    --muon-tp-mode duplicated \
    --muon-momentum 0.95 \
    --muon-nesterov \
    --lr 1e-3 \
    --min-lr 1e-4 \
    --weight-decay 0.1 \
    --bf16 \
    --ckpt-format torch_dist \
    --data-path /path/to/data \
    --save /path/to/checkpoints
```

### Layer-Sharded Muon (for TP and MoE Runs)

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --optimizer muon \
    --use-distributed-optimizer \
    --muon-tp-mode layer_sharded \
    --muon-no-split-qkv \
    --muon-ns-batch-size 32 \
    --bf16 \
    --ckpt-format torch_dist \
    ...
```

### Megatron-FSDP v2 with Muon

Megatron-FSDP v2 (`--use-megatron-fsdp --megatron-fsdp-version 2`) supports `--optimizer muon` via `FsdpMuon` paired with Adam in a `FullyShardedOptimizer` chain. Pass `--muon-no-split-qkv` and `--ckpt-format fsdp_dtensor`, and do not pass `--use-distributed-optimizer`. See the [Megatron-FSDP](megatron_fsdp.md) guide for FSDP configuration details.

## Supported Configurations and Constraints

- **Precision**: `--bf16` and FP32 are supported; `--fp16` is rejected (`emerging optimizer with fp16 is not supported`).
- **Checkpointing**: Non-FSDP emerging-optimizer runs require `--ckpt-format torch` or `--ckpt-format torch_dist`.
- **Unsupported flags**:
  - `--use-torch-fsdp2` is not supported with emerging optimizers.
  - `--overlap-param-gather-with-optimizer-step` is not supported with `LayerWiseDistributedOptimizer`.
  - `--moe-single-grouped-weight` is rejected with `LayerWiseDistributedOptimizer`.
  - `num_distributed_optimizer_instances` greater than `1` is not supported with `LayerWiseDistributedOptimizer`.
  - When Gloo process groups are disabled, `--dist-ckpt-optim-fully-reshardable` with `--distrib-optim-fully-reshardable-mem-efficient` requires `--no-use-layer-wise-param-layout` because the split scalar `DistributedOptimizer` gathers over Gloo.
