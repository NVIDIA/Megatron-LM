# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optimizer param-group overrides for scale-learning QAD (LSQ)."""

from typing import Dict

from megatron.core.optimizer import ParamKey
from megatron.core.optimizer_param_scheduler import ParamGroupOverride

LSQ_SCALE_PARAM_NAMES = ("*_amax_pre", "*_amax_post")


def get_lsq_config_overrides(
    lsq_scale_lr: float | None, *, base_lr: float | None
) -> Dict[ParamKey, ParamGroupOverride]:
    """Optimizer overrides for scale-learning QAD (LSQ).

    Puts the learnable scale (amax) params in their own param group with
    max_lr=lsq_scale_lr. Returns no override when it equals the optimizer config's
    base_lr: the group would then match the default group on every field that identifies
    a group on checkpoint load, and the two would collide on resume.
    """
    if lsq_scale_lr is None or lsq_scale_lr == base_lr:
        return {}
    return {ParamKey(name=LSQ_SCALE_PARAM_NAMES): ParamGroupOverride(max_lr=lsq_scale_lr)}
