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
- `depth_mup`: experimental spectral width-depth MuP behavior for dense
  GPT-style residual Transformer blocks using `--optimizer adam` with AdamW-style
  semantics. Nonzero weight decay requires `decoupled_weight_decay=True`.

`depth_mup` is intentionally narrow. Megatron rejects unsupported paths instead
of silently applying unvalidated scaling rules.

## Configuration Surface

New configs should use the canonical scaling fields:

```bash
--scaling-recipe mup \
--scaling-base-hidden-size <base-hidden-size> \
--scaling-base-head-dim <base-head-dim>
```

`--use-mup` remains a backward-compatible alias for exact
`--scaling-recipe mup`, but it is deprecated. The legacy aliases
`--mup-base-hidden-size`, `--mup-base-head-dim`, and `--mup-width-mult` are also
deprecated where they overlap with the canonical scaling surface.

`--mup-width-mult` is derived from the resolved scaling context:

```text
width_mult = hidden_size / scaling_base_hidden_size
```

If a non-default `--mup-width-mult` is supplied, it must match that derived
value. If legacy MuP fields and canonical scaling fields conflict, Megatron
raises an error during validation. CLI and YAML configs use the same alias
warning and canonicalization path.

## `mup`

`mup` preserves the current Megatron width-MuP surface:

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

`depth_mup` extends MuP-family width behavior with depth scaling for dense
GPT-style residual blocks. It is implemented only for `--optimizer adam`.
Because its weight-decay rule is AdamW-style, `depth_mup` requires
`decoupled_weight_decay=True` whenever `weight_decay` is nonzero. Coupled
Adam/L2 is allowed only with `weight_decay=0.0`.

The main default behaviors are:

- dense self-attention/MLP residual branch output scales as `depth_mult^-1`
- hidden matrix-like Adam LR scales as `width_mult^-1`
- hidden matrix-like Adam epsilon scales as `(width_mult * depth_mult)^-1`
- embedding/output-class Adam epsilon scales as `width_mult^-1`
- hidden matrix-like AdamW weight decay scales as `width_mult`
- dense residual output-projection initialization scales as `depth_mult^+0.5`

Megatron's 1-D parameters are not all hidden biases. Under `depth_mup`, hidden
linear/attention/MLP biases keep base weight decay, while normalization vectors
and otherwise unknown 1-D tensors keep Megatron's conservative no-weight-decay
behavior unless q/k layernorm is explicitly opted into weight decay with
`apply_wd_to_qk_layernorm=True`.

Example:

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --optimizer adam \
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
- `--optimizer adam`, with `decoupled_weight_decay=True` when `weight_decay`
  is nonzero

`depth_mup` currently rejects unsupported paths, including:

- SGD, Muon, and other non-Adam optimizers
- residual-branch scaling during inference
- fused TP inference residual scaling
- cross-attention
- `multi_latent_attention`
- configured experimental attention variants
- MoE
- BERT, T5, and Mamba model families

Training mode is the supported runtime. Validation loss can be enabled
explicitly with `--allow-depth-mup-eval`; this switch is for validation only and
does not make generation or inference a supported `depth_mup` path. YAML configs
default `allow_depth_mup_eval` to `False`.

## Manual Overrides

The canonical scaling fields remain overrides on top of the named recipe. For
example, `--scaling-residual-branch-depth-power 0.0` explicitly disables the
default `depth_mup` residual multiplier instead of introducing a separate
recipe.

Overrides only change resolved multipliers inside the supported surface. They do
not widen the supported-surface contract.

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
