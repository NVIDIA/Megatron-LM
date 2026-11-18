# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for --lsq-scale-lr and the LSQ scale-param handling in model_builder."""

from argparse import Namespace

import pytest
import torch

import megatron.training.training as training
from megatron.core.optimizer import (
    OptimizerConfig,
    _get_param_groups,
    get_standard_config_overrides,
)
from megatron.core.optimizer.optimizer import param_group_identifier_keys
from megatron.post_training.model_builder import _propagate_expert_allreduce_to_lsq_params
from megatron.post_training.optimizer import (
    get_lsq_config_overrides,
    install_lsq_optimizer_overrides,
)
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
    return model


def _args(lr=3e-5, min_lr=0.0, lsq_scale_lr=None):
    return Namespace(lr=lr, min_lr=min_lr, lsq_scale_lr=lsq_scale_lr)


@pytest.fixture
def restore_optimizer_config_override_fn(monkeypatch):
    """Restore training.get_standard_config_overrides after the test wraps it."""
    monkeypatch.setattr(training, "get_standard_config_overrides", get_standard_config_overrides)


def test_no_override_when_unset():
    assert get_lsq_config_overrides(_args(), base_lr=3e-5) == {}


def test_no_override_when_equal_to_base_lr():
    # An override equal to the optimizer config's LR would create a group that collides
    # with the default group on checkpoint load, so none is created.
    assert get_lsq_config_overrides(_args(lsq_scale_lr=3e-5), base_lr=3e-5) == {}


def test_override_when_set():
    overrides = get_lsq_config_overrides(_args(lsq_scale_lr=1e-4), base_lr=3e-5)
    assert len(overrides) == 1
    (override,) = overrides.values()
    assert override == {"max_lr": 1e-4}


def test_install_is_idempotent(restore_optimizer_config_override_fn):
    install_lsq_optimizer_overrides()
    wrapped = training.get_standard_config_overrides
    install_lsq_optimizer_overrides()
    assert training.get_standard_config_overrides is wrapped


@pytest.mark.parametrize(
    "args_values, expected_lsq_lr",
    [
        pytest.param({"lsq_scale_lr": 1e-4}, 1e-4, id="args-without-lr"),
        pytest.param({"lr": 1e-4, "lsq_scale_lr": 3e-5}, None, id="scale-lr-equals-config-lr"),
        pytest.param({"lr": 1e-4, "lsq_scale_lr": 1e-4}, 1e-4, id="scale-lr-equals-stale-args-lr"),
    ],
)
def test_wrapped_config_adds_lsq_override(
    monkeypatch, restore_optimizer_config_override_fn, args_values, expected_lsq_lr
):
    args = Namespace(**args_values)
    monkeypatch.setattr(training, "get_args", lambda: args)
    config = OptimizerConfig(lr=3e-5, min_lr=0.0)
    base_overrides = training.get_standard_config_overrides(config)
    install_lsq_optimizer_overrides()
    overrides = training.get_standard_config_overrides(config)
    if expected_lsq_lr is None:
        assert overrides == base_overrides
    else:
        for key, value in base_overrides.items():
            assert overrides[key] == value
        added_keys = set(overrides) - set(base_overrides)
        assert len(added_keys) == 1
        assert overrides[added_keys.pop()] == {"max_lr": expected_lsq_lr}


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

    @pytest.mark.parametrize("lsq_scale_lr", [1e-4, 3e-5])
    def test_scale_params_get_own_group(self, lsq_scale_lr):
        config = OptimizerConfig(lr=3e-5, min_lr=0.0, weight_decay=0.0)
        overrides = get_lsq_config_overrides(_args(lsq_scale_lr=lsq_scale_lr), base_lr=config.lr)
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
