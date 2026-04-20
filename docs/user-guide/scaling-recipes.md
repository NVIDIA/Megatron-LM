<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Scaling Recipes

Megatron-LM supports named scaling recipes through `--scaling-recipe`.

The current named recipes are:

- `none`: standard Megatron parameterization.
- `mup`: current Megatron width MuP behavior.
- `depth_mup`: a spectral width-depth μP Adam/AdamW recipe for dense GPT-style
  residual Transformer blocks within Megatron's current support surface.

`depth_mup` remains a narrow public recipe rather than a broad public
depth-transfer claim. Megatron enforces the intended surface and explicitly
rejects unsupported paths instead of silently inheriting behavior.

## Legacy MuP Flags

`--use-mup` remains a backward-compatible alias for exact `--scaling-recipe mup`.

`depth_mup` is intentionally distinct from `--use-mup`. Code paths that mean
"MuP-family width behavior" are keyed off the resolved scaling context rather
than `config.use_mup`.

If legacy MuP flags and canonical scaling flags are both set to conflicting
values, Megatron raises an error during argument validation.

The scaling-policy framework is intended to be reusable beyond the current
recipes. `mup` and `depth_mup` are named presets over a shared resolved scaling
context, shared model/runtime hooks, and shared optimizer override plumbing.

## `mup`

`mup` preserves the current merged Megatron MuP surface:

- width multiplier from `hidden_size / scaling_base_hidden_size`
- MuP-family attention softmax scaling through `scaling_base_head_dim`
- hidden-layer width-scaled initialization
- MuP-family embedding and logit scaling
- MuP-family optimizer overrides, including Adam epsilon handling

Example:

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --optimizer adam \
    --scaling-recipe mup \
    --scaling-base-hidden-size 1024 \
    --scaling-base-head-dim 128
```

## `depth_mup`

`depth_mup` extends MuP-family width behavior with the spectral width-depth μP
paper's AdamW-style optimizer table adapted to Megatron's current dense
GPT-style residual Transformer support surface.

The resolved recipe defaults are:

- residual branch multiplier: `depth_mult^-1`
- hidden matrix-like Adam/AdamW LR multiplier: `width_mult^-1`
- hidden matrix-like Adam/AdamW epsilon multiplier: `(width_mult * depth_mult)^-1`
- hidden vector Adam/AdamW epsilon multiplier: `(width_mult * depth_mult)^-1`
- embedding/output-class Adam/AdamW epsilon multiplier: `width_mult^-1`
- hidden matrix-like Adam/AdamW weight-decay multiplier: `width_mult`
- dense block output-projection init compensation: `depth_mult^+0.5`

The output-projection init compensation is important because Megatron already
applies layer-count-dependent initialization to residual branch output
projections. `depth_mup` rebases that initialization to
`scaling_base_num_layers` so the explicit residual-branch multiplier carries the
intended depth scaling.

Example:

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --optimizer adamw \
    --scaling-recipe depth_mup \
    --scaling-base-hidden-size 1024 \
    --scaling-base-num-layers 12 \
    --scaling-base-head-dim 128
```

### Current `depth_mup` Scope

`depth_mup` v1 is currently intended for:

- dense GPT-style residual Transformer blocks
- dense self-attention residual branches
- dense MLP residual branches
- Adam and AdamW optimizers

`depth_mup` currently rejects unsupported paths instead of silently applying
unvalidated rules. The rejected surface includes:

- SGD, Muon, and other non-Adam optimizers
- residual-branch scaling during inference
- fused TP inference residual scaling
- cross-attention
- `multi_latent_attention`
- configured experimental attention variants
- MoE
- BERT, T5, and Mamba model families

## Manual Overrides

The canonical scaling fields remain overrides on top of the named recipe. For
example, `--scaling-residual-branch-depth-power 0.0` explicitly disables the
default `depth_mup` residual multiplier instead of introducing a separate
recipe.

These overrides only change the resolved multipliers inside the supported
surface. They do not widen the supported-surface contract. For example,
`depth_mup` inference remains rejected even if
`--scaling-residual-branch-depth-power 0.0` makes the residual multiplier an
identity.

Recipe role classification prefers explicit metadata attached during model
construction and annotation. A small name-based fallback remains for older
unannotated parameters, but that fallback is compatibility-only and is not the
intended semantic source for `depth_mup`.

## Non-goals

These recipes do not currently claim:

- HyperP
- CompleteP
- MuonH / AdamH
- the full Muon-Kimi spectral width-depth training setup
- token-count LR scaling
- SqrtGate
- MoE granularity transfer
- public SGD depth transfer
- public Muon depth transfer
