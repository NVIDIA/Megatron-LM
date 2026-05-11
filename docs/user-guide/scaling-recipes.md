<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Scaling Recipes

Megatron-LM supports named scaling recipes through `--scaling-recipe`. A recipe
is a resolved contract over model initialization, forward-time scaling, optimizer
parameter groups, checkpoint compatibility, and validation-time behavior. It is
not only a command-line alias.

The current named recipes are:

- `none`: standard Megatron parameterization.
- `mup`: current Megatron width MuP behavior.
- `depth_mup`: experimental spectral width-depth μP behavior for dense
  GPT-style residual Transformer blocks using AdamW-style semantics through
  `--optimizer adam` with `decoupled_weight_decay=True` when weight decay is
  nonzero.

`depth_mup` is intentionally narrow. Megatron enforces the supported surface and
rejects unsupported paths instead of silently inheriting behavior from standard
Megatron, width-only MuP, inference, MoE, or non-Adam optimizers.

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

`--mup-width-mult` is derived state, not user-owned configuration. The effective
width multiplier is:

```text
width_mult = hidden_size / scaling_base_hidden_size
```

If a non-default `--mup-width-mult` is supplied, it must match that derived
value. This prevents checkpoint metadata, logs, or YAML configs from carrying a
stale width multiplier that disagrees with the actual model shape.

`depth_mup` is intentionally distinct from `--use-mup`. Code paths that mean
"MuP-family width behavior" are keyed off the resolved scaling context rather
than `config.use_mup`.

If legacy MuP flags and canonical scaling flags are both set to conflicting
values, Megatron raises an error during argument validation. CLI and YAML
validation use the same legacy-warning and alias-synchronization path so the
global args namespace has the same canonical scaling fields regardless of input
format.

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

`depth_mup` extends MuP-family width behavior with a v1 Megatron adaptation of
the spectral width-depth μP AdamW table. It deliberately targets the optimizer
and residual-block semantics Megatron can implement and validate today.

Because the weight-decay row is derived for decoupled AdamW, `depth_mup`
requires `decoupled_weight_decay=True` whenever `weight_decay` is nonzero.
Coupled Adam/L2 is allowed only with `weight_decay=0.0`. Non-Adam optimizers are
rejected for `depth_mup` rather than partially mapped.

The resolved recipe defaults are:

| Mechanism | Default multiplier |
| --- | --- |
| Dense self-attention/MLP residual branch output | `depth_mult^-1` |
| Hidden matrix-like Adam/AdamW LR | `width_mult^-1` |
| Hidden matrix-like Adam/AdamW epsilon | `(width_mult * depth_mult)^-1` |
| Hidden vector-like Adam/AdamW epsilon | `(width_mult * depth_mult)^-1` |
| Embedding/output-class Adam/AdamW epsilon | `width_mult^-1` |
| Hidden matrix-like AdamW weight decay | `width_mult` |
| Dense block output-projection initialization | `depth_mult^+0.5` |

### Parameter-Class Policy

Megatron classifies parameters by explicit parameterization metadata when it is
available, with name/shape fallbacks only for backward compatibility. The v1
`depth_mup` policy is:

| Parameter class | LR policy | Epsilon policy | Weight-decay policy |
| --- | --- | --- | --- |
| Embedding/output class | Preserve embedding/output LR policy, including `decoupled_lr` precedence | `width_mult^-1` | Base Megatron policy |
| Hidden matrix-like weights | `width_mult^-1` | `(width_mult * depth_mult)^-1` | `width_mult` |
| Hidden linear/attention/MLP biases | Base LR | `(width_mult * depth_mult)^-1` | Base weight decay |
| Norm scale/bias and unknown 1-D tensors | Base LR | `(width_mult * depth_mult)^-1` as current v1 policy | No weight decay |
| q/k layernorm vectors with `apply_wd_to_qk_layernorm=True` | Base LR | `(width_mult * depth_mult)^-1` | Base weight decay |

This table is an intentional Megatron adaptation. The spectral AdamW table gives
base weight decay to hidden biases, but Megatron's 1-D tensors are not all
hidden biases. Hidden linear/attention/MLP biases therefore keep base weight
decay, while normalization vectors and otherwise unknown 1-D tensors keep
Megatron's conservative no-weight-decay behavior unless q/k layernorm is
explicitly opted in. The hidden-vector epsilon rule is applied to hidden
vector-like parameters as a v1 policy, not as a claim that all norm-like vectors
appear in the paper table.

### Initialization and Residual Branches

Megatron already applies layer-count-dependent initialization to residual branch
output projections. `depth_mup` rebases dense transformer block output
projection initialization to `scaling_base_num_layers` so the explicit
residual-branch multiplier carries the intended depth scaling. The dense
residual hook covers self-attention output projection and dense MLP output
projection. MoE layers do not inherit this hook.

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

### Validation and Evaluation

Training mode is the supported runtime for `depth_mup`. Validation loss can be
enabled explicitly with `--allow-depth-mup-eval`; this switch is for validation
only and does not make generation/inference a supported `depth_mup` path. YAML
configs default `allow_depth_mup_eval` to `False` so older YAML files continue
to validate without adding a new field.

Checkpoint resume and distributed-optimizer preprocessing identify optimizer
parameter groups through the same tolerant identifier tuple used during
optimizer load. Optional group fields such as `eps` and per-group `optimizer`
may be absent in standard Adam/SGD-style groups, so missing fields resolve to
`None` instead of causing resume-time `KeyError`.

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
