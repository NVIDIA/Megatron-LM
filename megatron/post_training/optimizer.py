# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optimizer param-group overrides for scale-learning QAD (LSQ)."""

from typing import Dict

from megatron.core.optimizer import ParamKey
from megatron.core.optimizer_param_scheduler import ParamGroupOverride

LSQ_SCALE_PARAM_NAMES = ("*_amax_pre", "*_amax_post")


def get_lsq_config_overrides(args) -> Dict[ParamKey, ParamGroupOverride]:
    """Optimizer overrides for scale-learning QAD (LSQ).

    Puts the learnable scale (amax) params in their own param group with
    max_lr=--lsq-scale-lr. Returns no override when it equals --lr: the group would then
    match the default group on every field that identifies a group on checkpoint load,
    and the two would collide on resume.
    """
    lsq_scale_lr = getattr(args, "lsq_scale_lr", None)
    if lsq_scale_lr is None or lsq_scale_lr == args.lr:
        return {}
    return {ParamKey(name=LSQ_SCALE_PARAM_NAMES): ParamGroupOverride(max_lr=lsq_scale_lr)}


def install_lsq_optimizer_overrides():
    """Wrap megatron.training.training.get_megatron_optimizer_config to add the LSQ overrides.

    setup_model_and_optimizer() looks the function up by module global at call time, so this
    takes effect if installed before the optimizer is built (it is called from the model
    builder, which runs inside get_model()). Same pattern as
    megatron/elastification/pretrain_hybrid_flex.py. Idempotent: the builder runs once per
    model chunk (and again for a KD teacher).
    """
    import megatron.training.training as training

    orig = training.get_megatron_optimizer_config
    if getattr(orig, "_lsq_wrapped", False):
        return

    def get_megatron_optimizer_config(args):
        config, config_overrides = orig(args)
        config_overrides = {**(config_overrides or {}), **get_lsq_config_overrides(args)}
        return config, config_overrides

    get_megatron_optimizer_config._lsq_wrapped = True
    training.get_megatron_optimizer_config = get_megatron_optimizer_config
