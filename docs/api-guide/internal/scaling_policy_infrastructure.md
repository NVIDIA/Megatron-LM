<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Scaling Policy Infrastructure

This internal policy layer centralizes Megatron's existing parameterization hooks
without adding a new user-facing recipe surface.

The current policy resolves the legacy `use_mup` fields into model-side and
optimizer-side decisions. Standard Megatron behavior is represented as the
identity policy, so code paths can call the same hooks whether or not MuP is
active.

## Model Policy

Model code should route scaling-sensitive decisions through the resolved model
policy instead of reading `use_mup` at each call site. The policy currently
covers:

- hidden-weight initialization;
- output-projection initialization;
- attention softmax scale;
- embedding activation scaling;
- output logit scaling;
- residual branch output hooks.

For non-MuP configs, every hook returns the current Megatron default.

## Training Policy

Optimizer code should route per-parameter hyperparameter multipliers through the
resolved training policy. The policy currently preserves the existing MuP rules:

- Adam-family hidden matrix parameters use `lr / mup_width_mult`;
- Adam-family hidden matrix parameters use `eps / mup_width_mult`;
- SGD vector-like parameters use `lr * mup_width_mult`;
- Muon-managed matrices stay on Muon scaling rather than Adam-style MuP LR
  overrides.

The public compatibility function `get_mup_config_overrides` remains available
and delegates to the policy implementation.

## Parameter Metadata

Model construction may attach explicit parameterization metadata to parameters.
Optimizer grouping should prefer this metadata and keep existing name/shape
fallbacks only for compatibility with unannotated parameters.

FSDP and other parameter-rewriting paths must preserve the metadata attributes so
optimizer grouping remains stable after wrapping or sharding.

## Checkpoint Resume

Distributed optimizer resume uses a shared helper for the existing stable
parameter-group identity: `wd_mult`, `lr_mult`, `is_expert_parallel`, and
`is_decoupled_lr`. The helper tolerates NeMo-style `pre_` field names and missing
legacy fields without adding newer optional fields such as `eps`, `max_lr`,
`min_lr`, or per-group `optimizer` to the PR1 resume identity.
