# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""RMSNorm gamma decay selection, scheduling, and argument validation."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import _get_param_groups, get_standard_config_overrides
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.training.arguments import _add_regularization_args


def _norm(normalization):
    config = SimpleNamespace(
        normalization=normalization,
        layernorm_zero_centered_gamma=False,
        persist_layer_norm=False,
        sequence_parallel=False,
        memory_efficient_layer_norm=False,
    )
    return WrappedTorchNorm(config=config, hidden_size=4)


def _model():
    model = torch.nn.Module()
    model.linear = torch.nn.Linear(4, 4)
    # Names deliberately do not identify the normalization type.
    model.arbitrary_name = _norm("RMSNorm")
    model.misleading_rmsnorm = _norm("LayerNorm")
    model.q_layernorm = _norm("RMSNorm")
    model.k_layernorm = _norm("LayerNorm")
    model.vector = torch.nn.Parameter(torch.ones(4))
    model.retention = torch.nn.Parameter(torch.ones(4))
    model.retention.is_wide_residual_retention_parameter = True
    model.frozen = torch.nn.Parameter(torch.ones(4), requires_grad=False)
    return model


def _groups(model, config, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda **_: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, keys, **_: output.__setitem__(0, keys),
    )
    return _get_param_groups([model], config, get_standard_config_overrides(config))


def _group_for(groups, parameter):
    matches = [group for group in groups if any(p is parameter for p in group["params"])]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("multiplier", [None, 0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("apply_wd_to_qk_layernorm", [False, True])
def test_rmsnorm_weight_decay_selection(monkeypatch, multiplier, apply_wd_to_qk_layernorm):
    """Only marked gamma opts out of the generic 1D skip; Q/K policy composes."""
    model = _model()
    config = OptimizerConfig(
        lr=0.1,
        rmsnorm_weight_decay_mult=multiplier,
        apply_wd_to_qk_layernorm=apply_wd_to_qk_layernorm,
    )
    groups = _groups(model, config, monkeypatch)
    expected_rmsnorm = 0.0 if multiplier is None else multiplier
    expected_qk_rmsnorm = float(apply_wd_to_qk_layernorm) if multiplier is None else multiplier
    expected = {
        model.arbitrary_name.weight: expected_rmsnorm,
        model.q_layernorm.weight: expected_qk_rmsnorm,
        model.k_layernorm.weight: float(apply_wd_to_qk_layernorm),
        model.k_layernorm.bias: float(apply_wd_to_qk_layernorm),
        model.misleading_rmsnorm.weight: 0.0,
        model.misleading_rmsnorm.bias: 0.0,
        model.linear.bias: 0.0,
        model.vector: 0.0,
        model.linear.weight: 1.0,
        model.retention: 1.0,
    }
    for parameter, wd_mult in expected.items():
        assert _group_for(groups, parameter)["wd_mult"] == wd_mult
    assert all(p is not model.frozen for group in groups for p in group["params"])


@pytest.mark.parametrize("global_weight_decay", [0.0, 0.1])
def test_rmsnorm_decay_follows_global_schedule_and_adamw(monkeypatch, global_weight_decay):
    """Exercise the actual scheduler and zero-gradient AdamW decay updates on CPU."""
    model = _model()
    config = OptimizerConfig(lr=0.1, rmsnorm_weight_decay_mult=0.5)
    groups = _groups(model, config, monkeypatch)
    optimizer = torch.optim.AdamW(groups, lr=config.lr)
    scheduler = OptimizerParamScheduler(
        optimizer=optimizer,
        init_lr=config.lr,
        max_lr=config.lr,
        min_lr=config.lr,
        lr_warmup_steps=0,
        lr_decay_steps=10,
        lr_decay_style="constant",
        start_wd=global_weight_decay,
        end_wd=2 * global_weight_decay,
        wd_incr_steps=10,
        wd_incr_style="linear",
    )

    for increment, expected_wd in [
        (0, global_weight_decay),
        (5, 1.5 * global_weight_decay),
        (5, 2 * global_weight_decay),
    ]:
        scheduler.step(increment)
        before = {p: p.detach().clone() for group in groups for p in group["params"]}
        for parameter in before:
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()

        for parameter, wd_mult in [
            (model.arbitrary_name.weight, 0.5),
            (model.q_layernorm.weight, 0.5),
            (model.linear.weight, 1.0),
            (model.misleading_rmsnorm.weight, 0.0),
            (model.linear.bias, 0.0),
            (model.vector, 0.0),
        ]:
            assert _group_for(groups, parameter)["weight_decay"] == pytest.approx(
                expected_wd * wd_mult
            )
            torch.testing.assert_close(
                parameter, before[parameter] * (1 - config.lr * expected_wd * wd_mult)
            )


@pytest.mark.parametrize("multiplier", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_rmsnorm_weight_decay_rejects_invalid_multiplier(multiplier):
    with pytest.raises(ValueError, match="rmsnorm_weight_decay_mult.*finite nonnegative"):
        OptimizerConfig(rmsnorm_weight_decay_mult=multiplier)


def test_rmsnorm_weight_decay_cli():
    parser = _add_regularization_args(argparse.ArgumentParser())
    assert parser.parse_args([]).rmsnorm_weight_decay_mult is None
    args = parser.parse_args(["--weight-decay", "0.1", "--rmsnorm-weight-decay-mult", "0.5"])
    config = OptimizerConfig(
        weight_decay=args.weight_decay, rmsnorm_weight_decay_mult=args.rmsnorm_weight_decay_mult
    )
    assert config.weight_decay == 0.1
    assert config.rmsnorm_weight_decay_mult == 0.5
