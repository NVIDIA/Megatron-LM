<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Simplified sparse attention

Simplified sparse attention (SSA) replaces the attention core while retaining the
ordinary query, key, value and output projections. A separate learned indexer
projects the normalized attention input into one query/key head. It scores causal
keys by a scaled dot product and selects the top-k keys for all attention heads.
The indexer has no positional rotation, learned head weighting or scoring ReLU.

The kernels come from the cuDNN frontend (`nvidia-cudnn-frontend` 1.33 or newer,
which requires CuTe DSL 4.8). Megatron Core owns the projections, training
policy and checkpoint integration; cuDNN owns sparse selection, attention and the
indexer loss and gradients. The kernels are bound when the attention layer is
constructed, and a missing or older frontend is reported there.

## Training

Training has two phases. Indexer inputs are detached in both, so the auxiliary
objective cannot update the backbone or its normalization weights. Indexer weights
and projections remain BF16, including inside an MXFP8 backbone.

- **Phase 1, indexer-only training** (`ssa_train_indexer_only`). Every parameter
  except the indexers is frozen, and so are router bias updates. Attention is the
  frozen dense causal attention: FlashAttention-4 computes it and its log-sum-exp,
  and cuDNN evaluates the KL from the mean dense attention distribution to the
  indexer distribution over all causal keys. Phase 1 needs `flash-attn-4` 4.0.0b20
  or newer and a positive `dsa_indexer_loss_coeff`.
- **Phase 2, joint training** (the default). The backbone trains through sparse
  attention, and the indexer trains with a KL loss from the mean attention
  distribution over the selected keys to the indexer distribution over the same keys.

Phase 1 usually starts from a dense `torch_dist` checkpoint, loaded without its
optimizer state (`--no-load-optim` or `--finetune`). Indexer weights missing from
the checkpoint keep their initialization; every other weight loads strictly. Loading
optimizer state across a phase change is rejected, because the trainable parameters
differ.

`dsa_indexer_loss_coeff` scales both the auxiliary loss and its gradients and must
be nonnegative. Mean-loss mode divides by the number of query tokens; per-token-loss
mode leaves normalization to the training loop. Indexer gradients can be clipped
separately from the backbone with `dsa_indexer_clip_grad`.

### Model FLOP accounting

Throughput counts useful model GEMMs with two FLOPs per multiply-add. For each
unpadded sequence of length `S`, causal scoring has `S*(S+1)/2` pairs and sparse
attention has `S*k-k*(k-1)/2` pairs, where `k=min(topk,S)`. Training counts sparse
attention forward and backward, full causal indexer scoring, and selected-support
indexer gradients. The two indexer projections count forward and weight gradients,
without input gradients because their inputs are detached. With a zero auxiliary
coefficient, only the indexer forward work contributes. Phase 1 counts the frozen
backbone forward only, the dense teacher's attention forward, the indexer
projections, and indexer scoring and gradients over every causal key. These are useful model
FLOPs, not executed hardware FLOPs: selection, elementwise operations,
communication, optimizer work and rematerialization are excluded.

## Selection and supported inputs

Select `experimental_attention_variant="ssa"`. For hybrid stacks, use
`ssa_stack_spec` or `gdp_ssa_stack_spec` from
`megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention_layer_specs`.
These specs select `SimplifiedSparseAttention` for attention layers; other layers
keep their existing implementations. `dsa_indexer_topk` controls the selection
size. The indexer architecture is fixed to one learned query/key head of
dimension 128.

The current integration supports:

| Property | Supported values |
| --- | --- |
| Layout | Unpadded SBHD, batch size 1 |
| Parallel attention | Context parallel size 1, tensor model parallel size 1 |
| Attention heads | 16, 32 or 96 query heads; one shared K/V head |
| Attention head dimension | 256 |
| Indexer | One learned Q/K head, dimension 128 |
| Top-k | 512, 1024 or 2048 |
| Kernel inputs | BF16 on SM100 or SM103 |
| Mask | Causal self-attention without custom mask or bias |
| QKV projection | Fused-norm `linear_qkv` (for example `TELayerNormColumnParallelLinear`) |

Attention dropout, sliding-window attention, QK clipping, logit recording,
decode caching and CUDA graph capture are not supported. Backward uses atomic
accumulation, so deterministic mode is rejected.
