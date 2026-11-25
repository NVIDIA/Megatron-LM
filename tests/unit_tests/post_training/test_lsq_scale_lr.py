# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for --lsq-scale-lr and the LSQ scale-param handling in model_hooks."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import (
    OptimizerConfig,
    _get_param_groups,
    get_standard_config_overrides,
)
from megatron.core.optimizer.optimizer import param_group_identifier_keys
from megatron.core.optimizer.optimizer_config import ParamKey
from megatron.post_training.model_hooks import _propagate_expert_allreduce_to_lsq_params, lsq_hook
from megatron.post_training.optimizer import get_lsq_config_overrides
from megatron.training.config.optimizer_config import build_optimizer_config_overrides
from tests.unit_tests.test_utilities import Utils


class _Quantizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._amax_pre = torch.nn.Parameter(torch.ones(4, 1))
        self._amax_post = torch.nn.Parameter(torch.ones(4, 1))


class _Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 16))
        self.weight_quantizer = _Quantizer()


def _toy_model():
    model = torch.nn.Module()
    model.linear = _Linear()
    model.config = SimpleNamespace(use_mup=False)
    return model


def test_no_override_when_unset():
    assert get_lsq_config_overrides(None, base_lr=3e-5) == {}


def test_no_override_when_equal_to_base_lr():
    # An override equal to the optimizer config's LR would create a group that collides
    # with the default group on checkpoint load, so none is created.
    assert get_lsq_config_overrides(3e-5, base_lr=3e-5) == {}


def test_override_when_set():
    overrides = get_lsq_config_overrides(1e-4, base_lr=3e-5)
    assert len(overrides) == 1
    (override,) = overrides.values()
    assert override == {"max_lr": 1e-4}


@pytest.mark.parametrize(
    "scale_lr,scale_min_lr,scale_wd_mult,expected",
    [
        (3e-5, 3e-6, 1.0, {}),  # Equal settings must not create a duplicate group.
        (None, None, 1.0, {}),
        (None, 1e-6, 1.0, {"min_lr": 1e-6}),
        (3e-5, None, 0.0, {"wd_mult": 0.0}),
        (1e-4, 0.0, 0.0, {"max_lr": 1e-4, "min_lr": 0.0, "wd_mult": 0.0}),
    ],
)
def test_shared_scale_policy_only_emits_differences(
    scale_lr, scale_min_lr, scale_wd_mult, expected
):
    overrides = get_lsq_config_overrides(
        scale_lr,
        base_lr=3e-5,
        base_min_lr=3e-6,
        scale_min_lr=scale_min_lr,
        scale_wd_mult=scale_wd_mult,
    )
    scale_key = ParamKey(name=("*_amax_pre", "*_amax_post"))
    assert overrides == ({scale_key: expected} if expected else {})


@pytest.mark.parametrize("scale_lr", [1e-4, 3e-5])
def test_lsq_hook_uses_model_settings_without_leaking_to_another_run(scale_lr):
    config = OptimizerConfig(lr=3e-5, min_lr=0.0)
    first, second = [_toy_model()], [_toy_model()]
    lsq_hook(scale_lr)(first)
    lsq_hook(scale_lr)(first)
    resolved = build_optimizer_config_overrides(config, first)
    standard = get_standard_config_overrides(config)
    assert build_optimizer_config_overrides(config, second) == standard
    for key, value in standard.items():
        assert resolved[key] == value
    extra = set(resolved) - set(standard)
    if scale_lr == config.lr:
        assert not extra
    else:
        assert len(extra) == 1
        assert resolved[extra.pop()] == {"max_lr": scale_lr}
    lsq_hook(None)(first)
    assert build_optimizer_config_overrides(config, first) == standard


def test_propagate_expert_allreduce_to_lsq_params():
    model = _toy_model()
    model.linear.weight.allreduce = False
    _propagate_expert_allreduce_to_lsq_params(model)
    assert model.linear.weight_quantizer._amax_pre.allreduce is False
    assert model.linear.weight_quantizer._amax_post.allreduce is False


class TestLSQParamGroups:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_minimum_rate_and_weight_decay_reach_scale_groups(self):
        config = OptimizerConfig(lr=3e-5, min_lr=3e-6, weight_decay=0.1)
        model = _toy_model()
        scales = get_lsq_config_overrides(
            config.lr,
            base_lr=config.lr,
            scale_min_lr=1e-6,
            base_min_lr=config.min_lr,
            scale_wd_mult=0.0,
        )
        overrides = build_optimizer_config_overrides(config, [model], scales)
        groups = _get_param_groups([model], config, overrides)
        by_param = {id(param): group for group in groups for param in group["params"]}
        weight_group = by_param[id(model.linear.weight)]
        scale_group = by_param[id(model.linear.weight_quantizer._amax_pre)]
        assert (weight_group["min_lr"], weight_group["wd_mult"]) == (3e-6, 1.0)
        assert (scale_group["max_lr"], scale_group["min_lr"], scale_group["wd_mult"]) == (
            3e-5,
            1e-6,
            0.0,
        )
        identities = [tuple(g.get(k) for k in param_group_identifier_keys) for g in groups]
        assert len(set(identities)) == len(identities)

    @pytest.mark.parametrize("lsq_scale_lr", [1e-4, 3e-5])
    def test_scale_params_get_own_group(self, lsq_scale_lr):
        config = OptimizerConfig(lr=3e-5, min_lr=0.0, weight_decay=0.0)
        overrides = get_lsq_config_overrides(lsq_scale_lr, base_lr=config.lr)
        groups = _get_param_groups([_toy_model()], config, overrides)

        if lsq_scale_lr == config.lr:
            # No override: everything stays in the default group.
            assert len(groups) == 1
            return
        assert len(groups) == 2
        by_lr = {g["max_lr"]: g for g in groups}
        assert len(by_lr[3e-5]["params"]) == 1  # weight
        assert len(by_lr[lsq_scale_lr]["params"]) == 2  # _amax_pre, _amax_post
        # The two groups must stay distinguishable when a checkpoint is loaded.
        identities = [tuple(g.get(k) for k in param_group_identifier_keys) for g in groups]
        assert len(set(identities)) == len(identities)
