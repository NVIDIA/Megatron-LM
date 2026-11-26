# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optimizer param-group overrides for scale-learning QAD (LSQ)."""

import math

from megatron.core.optimizer import ParamKey
from megatron.core.optimizer_param_scheduler import ParamGroupOverride

LSQ_SCALE_PARAM_NAMES = ("*_amax_pre", "*_amax_post")


def get_lsq_config_overrides(
    lsq_scale_lr: float | None,
    *,
    base_lr: float | None,
    scale_min_lr: float | None = None,
    base_min_lr: float | None = None,
    scale_wd_mult: float = 1.0,
) -> dict[ParamKey, ParamGroupOverride]:
    """Optimizer overrides for scale-learning QAD (LSQ).

    Only settings that differ from the base policy are emitted, so identical scale
    and ordinary groups do not collide on checkpoint load. Unset LR bounds inherit
    the base schedule. The default weight-decay multiplier preserves the existing
    policy; callers can explicitly disable weight decay on scales with 0.0.

    Args:
        lsq_scale_lr: Peak scale learning rate, or None to inherit the base rate.
        base_lr: Peak learning rate in the optimizer configuration.
        scale_min_lr: Minimum scale learning rate, or None to inherit the base rate.
        base_min_lr: Minimum learning rate in the optimizer configuration.
        scale_wd_mult: Weight-decay multiplier for the scales.

    Raises:
        ValueError: If a scale setting is negative/nonfinite or both explicit LR bounds are inverted.

    Returns:
        Scale-specific contributions, without standard weight-decay or decoupled-LR rules.
    """
    for name, value in (
        ("max_lr", lsq_scale_lr),
        ("min_lr", scale_min_lr),
        ("wd_mult", scale_wd_mult),
    ):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"LSQ {name} must be finite and nonnegative, got {value}.")
    # Inherited bounds can still be overridden by a recipe after this contribution.
    # Compare only explicit bounds here; validate the composed policy at setup.
    maximum = lsq_scale_lr
    minimum = scale_min_lr
    if maximum is not None and minimum is not None and minimum > maximum:
        raise ValueError("LSQ minimum learning rate must not exceed its maximum learning rate.")
    override = ParamGroupOverride()
    if lsq_scale_lr is not None and lsq_scale_lr != base_lr:
        override["max_lr"] = lsq_scale_lr
    if scale_min_lr is not None and scale_min_lr != base_min_lr:
        override["min_lr"] = scale_min_lr
    if scale_wd_mult != 1.0:
        override["wd_mult"] = scale_wd_mult
    return {ParamKey(name=LSQ_SCALE_PARAM_NAMES): override} if override else {}
