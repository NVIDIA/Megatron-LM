# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for --lsq-scale-lr and the LSQ scale-param handling in model_builder."""

from argparse import Namespace

import pytest
import torch

import megatron.training.training as training
from megatron.core.optimizer import OptimizerConfig, _get_param_groups
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
def restore_optimizer_config_fn(monkeypatch):
    """Restore training.get_megatron_optimizer_config after the test wraps it."""
    monkeypatch.setattr(
        training, "get_megatron_optimizer_config", training.get_megatron_optimizer_config
    )


def test_no_override_when_unset():
    assert get_lsq_config_overrides(_args()) == {}


def test_no_override_when_equal_to_base_lr():
    # An override equal to --lr would create a group that collides with the default group
    # on checkpoint load, so none is created.
    assert get_lsq_config_overrides(_args(lr=3e-5, lsq_scale_lr=3e-5)) == {}


def test_override_when_set():
    overrides = get_lsq_config_overrides(_args(lsq_scale_lr=1e-4))
    assert len(overrides) == 1
    (override,) = overrides.values()
    assert override == {"max_lr": 1e-4}


def test_install_is_idempotent(restore_optimizer_config_fn):
    install_lsq_optimizer_overrides()
    wrapped = training.get_megatron_optimizer_config
    install_lsq_optimizer_overrides()
    assert training.get_megatron_optimizer_config is wrapped


def test_wrapped_config_adds_lsq_override(restore_optimizer_config_fn):
    base_config, base_overrides = training.get_megatron_optimizer_config(
        Namespace(lr=3e-5, min_lr=0.0)
    )
    install_lsq_optimizer_overrides()
    config, overrides = training.get_megatron_optimizer_config(
        Namespace(lr=3e-5, min_lr=0.0, lsq_scale_lr=1e-4)
    )
    assert config == base_config
    assert set(base_overrides).issubset(overrides)
    assert len(overrides) == len(base_overrides) + 1


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
        overrides = get_lsq_config_overrides(_args(lsq_scale_lr=lsq_scale_lr))
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
