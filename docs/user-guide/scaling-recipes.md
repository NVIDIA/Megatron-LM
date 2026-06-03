<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Scaling Recipes

Scaling recipes choose the parameterization used to transfer hyperparameters
between model sizes. The canonical flag is `--scaling-recipe`.

Megatron currently exposes three recipes:

| Recipe | Behavior |
| --- | --- |
| `none` | Standard Megatron parameterization. This is the default. |
| `mup` | Width MuP for hidden-size transfer. |
| `depth_mup` | Experimental dense-transformer width-depth MuP for AdamW-style training. |

## Standard Parameterization

Use `--scaling-recipe none`, or omit `--scaling-recipe`, to keep the standard
parameterization. Scaling-specific fields such as `--scaling-base-hidden-size`
are rejected unless a scaling recipe is selected.

## Width MuP

Use `--scaling-recipe mup` when transferring hyperparameters from a base width to
a target width.

```bash
--scaling-recipe mup \
--scaling-base-hidden-size 1024 \
--scaling-base-head-dim 64
```

For MuP, Megatron derives the width multiplier internally:

```text
width_mult = hidden_size / scaling_base_hidden_size
```

This derived value controls MuP initialization, attention scale, output-logit
scale, and optimizer multipliers. `--mup-width-mult` is no longer an independent
input. If it is provided on the CLI for compatibility, it must match the derived
value.

When MuP is combined with Muon-family optimizers, Muon-managed matrix parameters
keep Muon's spectral scaling. Nonlinear and embedding-class scalar parameters
are routed through `--muon-scalar-optimizer`, which currently accepts `adam` or
`lion`.

## Legacy MuP Flags

The following flags are accepted for checkpoint and script compatibility, but are
deprecated as user-facing inputs:

| Deprecated flag | Canonical replacement |
| --- | --- |
| `--use-mup` | `--scaling-recipe mup` |
| `--mup-base-hidden-size` | `--scaling-base-hidden-size` |
| `--mup-base-head-dim` | `--scaling-base-head-dim` |
| `--mup-width-mult` | derived from `hidden_size / scaling_base_hidden_size` |

`--mup-embedding-mult`, `--mup-output-mult`, and `--mup-attn-scale-power` remain
MuP-specific tuning knobs. When `--mup-output-mult` is left at `1.0`, Megatron
sets it to `1 / width_mult` for non-base widths.

## Depth MuP

Use `--scaling-recipe depth_mup` when transferring from a base width and depth to
a target dense GPT-style transformer width and depth.

```bash
--scaling-recipe depth_mup \
--scaling-base-hidden-size 1024 \
--scaling-base-num-layers 12 \
--scaling-base-head-dim 64
```

Megatron derives both multipliers internally:

```text
width_mult = hidden_size / scaling_base_hidden_size
depth_mult = num_layers / scaling_base_num_layers
```

`depth_mup` includes the width-MuP model-side behavior, plus depth-aware residual
branch scaling, dense block output-projection initialization, and Adam/AdamW
optimizer multipliers. The default depth behavior is:

| Mechanism | Default multiplier |
| --- | --- |
| Dense self-attention and dense MLP residual branch output | `depth_mult^-1` |
| Hidden matrix Adam LR | `width_mult^-1` |
| Hidden matrix Adam epsilon | `(width_mult * depth_mult)^-1` |
| Hidden vector Adam epsilon | `(width_mult * depth_mult)^-1` |
| Embedding/output-class Adam epsilon | `width_mult^-1` |
| Hidden matrix AdamW weight decay | `width_mult` |
| Dense block output-projection initialization | `depth_mult^+0.5` |

`depth_mup` is intentionally narrow. It currently supports `--optimizer adam`.
If `weight_decay` is nonzero, the optimizer must use AdamW-style decoupled
weight decay (`decoupled_weight_decay=True`). Coupled Adam/L2 is allowed only
with `weight_decay=0.0`.

Megatron also keeps the standard distinction between hidden biases and
normalization vectors. Under `depth_mup`, hidden linear/attention/MLP biases keep
base weight decay, while normalization vectors and otherwise unknown 1-D tensors
stay on the conservative no-weight-decay path. q/k layernorm vectors use weight
decay only when `apply_wd_to_qk_layernorm=True`.

The supported runtime path is training. Megatron's validation-loss path enables
the required internal scaling-policy eval context automatically. This does not
make generation, inference, or fused TP inference residual scaling supported.

The current implementation fails closed for unsupported surfaces, including
cross-attention, hybrid/Mamba layer patterns, MTP, multi-latent attention,
experimental attention variants, MoE, non-Adam optimizers, and TE fused MLPs
when nontrivial dense block output-init depth scaling would be required.

## Checkpoints and YAML

Megatron stores and compares the resolved scaling recipe, not just the raw flag
spelling. A checkpoint created with legacy MuP aliases is compatible with the
canonical spelling when both resolve to the same effective recipe and base size.

YAML configs use the same effective resolution rules as CLI configs. Existing
YAML files that omit the new canonical scaling fields default to
`--scaling-recipe none`. For compatibility with full legacy YAML files that
materialized old defaults, `mup_width_mult: 1.0` is treated as an omitted default;
non-`1.0` YAML values are still validated against the derived width multiplier.
