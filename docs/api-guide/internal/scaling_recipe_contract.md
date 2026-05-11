<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Scaling Recipe Design Contract

This page documents the internal implementation contract for the current scaling
recipes. It is intended for maintainers changing model initialization, optimizer
parameter grouping, checkpoint compatibility, or argument validation.

A scaling recipe is resolved into a single scaling context. Model code,
optimizer code, checkpoint code, CLI validation, and YAML validation should read
that resolved context rather than independently interpreting legacy MuP fields.

## Legacy Alias Canonicalization

`--use-mup` is a deprecated alias for exact `--scaling-recipe mup`. The legacy
fields `mup_base_hidden_size`, `mup_base_head_dim`, and `mup_width_mult` are kept
for compatibility but should be synchronized from the canonical fields.

`mup_width_mult` is derived state:

```text
width_mult = hidden_size / scaling_base_hidden_size
```

A non-default legacy `mup_width_mult` must match the derived value. Conflicting
legacy and canonical scaling fields are validation errors. CLI and YAML
validation both warn for legacy aliases and call the same synchronization helper
so downstream global args have the same canonical shape.

## `depth_mup` Optimizer Contract

`depth_mup` is a v1 Megatron adaptation of the spectral width-depth MuP AdamW
table. It is supported only for `optimizer='adam'` because the current optimizer
overrides are defined for Adam/AdamW-style parameter groups.

Because the weight-decay row is derived for decoupled AdamW, nonzero
`weight_decay` requires `decoupled_weight_decay=True`. Coupled Adam/L2 is allowed
only when `weight_decay=0.0`. SGD, Muon, and other optimizers are intentionally
rejected rather than partially mapped.

The default multipliers are:

| Mechanism | Default multiplier |
| --- | --- |
| Dense self-attention/MLP residual branch output | `depth_mult^-1` |
| Hidden matrix-like Adam/AdamW LR | `width_mult^-1` |
| Hidden matrix-like Adam/AdamW epsilon | `(width_mult * depth_mult)^-1` |
| Hidden vector-like Adam/AdamW epsilon | `(width_mult * depth_mult)^-1` |
| Embedding/output-class Adam/AdamW epsilon | `width_mult^-1` |
| Hidden matrix-like AdamW weight decay | `width_mult` |
| Dense block output-projection initialization | `depth_mult^+0.5` |

## Parameter-Class Policy

Parameter classification should prefer explicit parameterization metadata
attached during model construction. Name/shape fallback logic exists only for
backward compatibility with older unannotated parameters.

| Parameter class | LR policy | Epsilon policy | Weight-decay policy |
| --- | --- | --- | --- |
| Embedding/output class | Preserve embedding/output LR policy, including `decoupled_lr` precedence | `width_mult^-1` | Base Megatron policy |
| Hidden matrix-like weights | `width_mult^-1` | `(width_mult * depth_mult)^-1` | `width_mult` |
| Hidden linear/attention/MLP biases | Base LR | `(width_mult * depth_mult)^-1` | Base weight decay |
| Norm scale/bias and unknown 1-D tensors | Base LR | `(width_mult * depth_mult)^-1` as current v1 policy | No weight decay |
| q/k layernorm vectors with `apply_wd_to_qk_layernorm=True` | Base LR | `(width_mult * depth_mult)^-1` | Base weight decay |

This table is deliberate Megatron behavior, not a direct claim that every row is
spelled out by the paper table. The paper gives base weight decay to hidden
biases, but Megatron's 1-D tensors also include normalization scale/bias tensors
and other vectors. Hidden linear/attention/MLP biases therefore keep base weight
decay, while normalization vectors and otherwise unknown 1-D tensors stay on the
standard no-weight-decay path unless q/k layernorm is explicitly opted in. The
hidden-vector epsilon rule applies to hidden vector-like parameters as the
current v1 policy.

## Initialization and Residual Branches

Megatron already applies layer-count-dependent initialization to residual branch
output projections. `depth_mup` rebases dense transformer block output projection
initialization to `scaling_base_num_layers` so the explicit residual-branch
multiplier carries the intended depth scaling.

The dense residual hook covers:

- self-attention output projection
- dense MLP output projection

MoE layers do not inherit this hook. Unsupported residual or model-family paths
should keep failing closed until they have explicit rules and tests.

## Runtime Scope

Training mode is the supported runtime for `depth_mup`. Validation loss can be
enabled with `allow_depth_mup_eval`, but that switch is validation-only and does
not make generation or inference a supported path.

YAML configs may not contain newly added argparse fields. YAML validation should
populate defaults for new global fields that downstream runtime code reads. For
`allow_depth_mup_eval`, the default is `False`.

## Checkpoint and Optimizer-Group Compatibility

Distributed-optimizer checkpoint preprocessing and optimizer load must identify
parameter groups through the same tolerant identifier tuple. Optional optimizer
group fields such as `eps` and per-group `optimizer` may be absent in standard
Adam/SGD groups.

Missing optional fields should resolve to `None` instead of causing resume-time
`KeyError`. Sorting code must also be `None`-safe so groups with and without
optional keys can be preprocessed deterministically.
