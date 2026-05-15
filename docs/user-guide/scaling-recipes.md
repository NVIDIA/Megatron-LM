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

Megatron currently exposes two recipes:

| Recipe | Behavior |
| --- | --- |
| `none` | Standard Megatron parameterization. This is the default. |
| `mup` | Width MuP for hidden-size transfer. |

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

## Checkpoints and YAML

Megatron stores and compares the resolved scaling recipe, not just the raw flag
spelling. A checkpoint created with legacy MuP aliases is compatible with the
canonical spelling when both resolve to the same effective recipe and base size.

YAML configs use the same effective resolution rules as CLI configs. Existing
YAML files that omit the new canonical scaling fields default to
`--scaling-recipe none`. For compatibility with full legacy YAML files that
materialized old defaults, `mup_width_mult: 1.0` is treated as an omitted default;
non-`1.0` YAML values are still validated against the derived width multiplier.
