<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Optimizer Parameters Scheduler

This api is used to calculate the learning rate and weight decay for the optimizer.

## Weight decay on RMSNorm gamma

By default, the optimizer excludes biases and one-dimensional parameters from weight decay.
Set `OptimizerConfig.rmsnorm_weight_decay_mult` (or `--rmsnorm-weight-decay-mult`) to
apply a multiple of the scheduled global weight decay specifically to RMSNorm gamma:

```python
config = OptimizerConfig(weight_decay=0.1, rmsnorm_weight_decay_mult=0.5)
```

Equivalently, pass `--weight-decay 0.1 --rmsnorm-weight-decay-mult 0.5` to training.
This applies weight decay of `0.05` to RMSNorm gamma when the global schedule is at `0.1`.
The multiplier continues to apply if the global weight decay changes during training.
Ordinary matrix weights retain global weight decay; LayerNorm parameters, biases, and
other one-dimensional parameters retain their existing policy.

Leaving the multiplier unset (`None`) preserves existing behavior, including
`apply_wd_to_qk_layernorm`. An explicit multiplier takes precedence over that option
for RMSNorm gamma. Use `1.0` for full decay or `0.0` to disable RMSNorm gamma decay.
Values must be finite and nonnegative.

Megatron's PyTorch, Transformer Engine (including fused norm-linear and residual norms),
vision, ModelOpt, and gated Mamba/GDP RMSNorm constructors mark gamma with
`is_rmsnorm_weight = True`. Custom RMSNorm implementations should mark their gamma
parameters with the same attribute. This identifies the actual parameter independently
of its name or the surrounding model's normalization setting. With zero-centered gamma,
decay acts on the stored parameter, as with the other optimizer parameters.

Callers supplying custom `config_overrides` should start from
`get_standard_config_overrides(config)` to include this policy.
