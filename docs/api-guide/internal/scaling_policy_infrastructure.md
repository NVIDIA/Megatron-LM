<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Scaling Policy Infrastructure

This internal policy layer centralizes Megatron's parameterization hooks behind a
scaling context.

The current public recipes are `none`, `mup`, and `depth_mup`. The policy
resolver also accepts legacy MuP aliases, then syncs them to the canonical recipe
fields so model, optimizer, YAML, and checkpoint paths see the same effective
scaling context. Standard Megatron behavior is represented as the identity
policy, so code paths can call the same hooks whether or not a scaling recipe is
active.

## Model Policy

Model code should route scaling-sensitive decisions through the model scaling
policy instead of reading `use_mup` at each call site. The policy currently
covers:

- hidden-weight initialization;
- output-projection initialization;
- attention softmax scale;
- embedding activation scaling;
- output logit scaling;
- residual branch output hooks.

For non-scaling configs, every hook returns the current Megatron default.

`depth_mup` adds depth-aware model hooks for dense GPT-style residual blocks:

- dense self-attention residual branch output scaling;
- dense MLP residual branch output scaling;
- dense block output-projection initialization rebased to the base depth.

Unsupported residual paths must fail closed. Cross-attention, MoE, fused TP
inference residual scaling, hybrid/Mamba layer patterns, MTP, and TE fused MLPs
without an explicit depth-init implementation should not silently inherit
dense-block hooks.

## Training Policy

Optimizer code should route per-parameter hyperparameter multipliers through the
training scaling policy. For `mup`, the policy preserves the existing width-MuP
rules:

- Adam-family hidden matrix parameters use `lr / mup_width_mult`;
- Adam-family hidden matrix parameters use `eps / mup_width_mult`;
- SGD vector-like parameters use `lr * mup_width_mult`;
- Muon-managed matrices stay on Muon scaling rather than Adam-style MuP LR
  overrides.
- Muon-family nonlinear and embedding-class scalar parameters are routed through
  the configured scalar optimizer, currently `adam` or `lion`.

For `depth_mup`, the policy is Adam/AdamW-only. Nonzero weight decay requires
`decoupled_weight_decay=True`; coupled Adam/L2 is allowed only with
`weight_decay=0.0`. The default multipliers are:

| Parameter class | LR policy | Epsilon policy | Weight-decay policy |
| --- | --- | --- | --- |
| Embedding/output class | Preserve embedding/output LR policy, including `decoupled_lr` precedence | `width_mult^-1` | Base Megatron policy |
| Hidden matrix-like weights | `width_mult^-1` | `(width_mult * depth_mult)^-1` | `width_mult` |
| Hidden linear/attention/MLP biases | Base LR | `(width_mult * depth_mult)^-1` | Base weight decay |
| Norm scale/bias and unknown 1-D tensors | Base LR | `(width_mult * depth_mult)^-1` as current v1 policy | No weight decay |
| q/k layernorm vectors with `apply_wd_to_qk_layernorm=True` | Base LR | `(width_mult * depth_mult)^-1` | Base weight decay |

The 1-D parameter policy is deliberate. Tensor rank alone is not semantic:
hidden biases, norm scales, q/k layernorm vectors, and unknown vectors are all
1-D tensors but do not share the same weight-decay rule.

The public compatibility function `get_mup_config_overrides` remains available
and delegates to the policy implementation for the legacy width-MuP surface.

## Parameter Metadata

Model construction may attach explicit parameterization metadata to parameters.
Optimizer grouping should prefer this metadata and keep existing name/shape
fallbacks only for compatibility with unannotated parameters.

FSDP and other parameter-rewriting paths must preserve the metadata attributes so
optimizer grouping remains stable after wrapping or sharding.

## Checkpoint Resume

Distributed optimizer resume and optimizer load must use the same tolerant
parameter-group identifier helper. The identifier includes optimizer-group fields
that can distinguish scaling-policy groups, while treating optional absent fields
as `None`.

Call sites must not sort groups with direct indexing over the identifier key
list. Standard Adam groups may not carry per-group `optimizer`, and SGD groups
may not carry `eps`; direct indexing turns those valid checkpoints into
resume-time `KeyError`s. Sorting must also use the None-safe sort key rather
than the raw identifier tuple, because Python cannot order `None` against floats
or strings when optional fields are present in only some groups.

## CLI, YAML, and Checkpoints

CLI validation, YAML validation, and checkpoint argument restore should all
resolve the same scaling context before downstream code reads global args.

- Legacy MuP aliases should warn and synchronize to canonical scaling fields.
- `mup_width_mult` is derived from `hidden_size / scaling_base_hidden_size`, not
  an independent user input.
- Checkpoint compatibility should compare effective scaling contexts
  rather than raw legacy spelling.
