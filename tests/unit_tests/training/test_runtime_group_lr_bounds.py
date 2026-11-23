# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime custom LR bounds survive dictionary replacement and group reordering."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.training.checkpointing import (
    _restore_optimizer_group_lr_bounds,
    _snapshot_optimizer_group_lr_bounds,
)


@pytest.mark.parametrize("custom_decoupled", [False, True])
@pytest.mark.parametrize("multiplier", [0.25, 16.0])
@pytest.mark.parametrize("resume_samples", [0, 42])
def test_custom_bounds_survive_load_and_reordering(custom_decoupled, multiplier, resume_samples):
    params = [torch.nn.Parameter(torch.ones(1)) for _ in range(3)]
    groups = [
        {"params": [params[0]], "max_lr": 1.0, "min_lr": 0.1, "default_config": True},
        {
            "params": [params[1]],
            "max_lr": multiplier,
            "min_lr": multiplier * 0.1,
            "default_config": False,
            "is_decoupled_lr": custom_decoupled,
        },
        {
            "params": [params[2]],
            "max_lr": 0.5,
            "min_lr": 0.05,
            "default_config": True,
            "is_decoupled_lr": True,
        },
    ]
    optimizer = torch.optim.SGD(groups, lr=1.0)
    runtime = _snapshot_optimizer_group_lr_bounds(optimizer)
    saved = deepcopy(optimizer.state_dict())
    for group in saved["param_groups"]:
        group.update(max_lr=0.01, min_lr=0.001, default_config=True)
    optimizer.load_state_dict(saved)
    optimizer.param_groups.reverse()
    config = OptimizerConfig(lr=1.0, min_lr=0.1, decoupled_lr=0.5, decoupled_min_lr=0.05)
    _restore_optimizer_group_lr_bounds(optimizer, runtime, config)
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=1.0,
        min_lr=0.1,
        lr_warmup_steps=10,
        lr_decay_steps=100,
        lr_decay_style="linear",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=100,
        wd_incr_style="constant",
    )
    by_param = {id(g["params"][0]): g for g in optimizer.param_groups}
    for increment in (resume_samples, 1, 1):
        scheduler.step(increment=increment)
        assert by_param[id(params[1])]["lr"] == pytest.approx(
            multiplier * by_param[id(params[0])]["lr"]
        )
        assert by_param[id(params[2])]["lr"] == pytest.approx(0.5 * by_param[id(params[0])]["lr"])
        for param in params:
            param.grad = torch.ones_like(param)
        before = [param.detach().clone() for param in params]
        optimizer.step()
        for param, previous in zip(params, before):
            torch.testing.assert_close(param, previous - by_param[id(param)]["lr"])
    assert by_param[id(params[1])]["default_config"] is False


def test_ambiguous_empty_groups_fail_without_partial_updates():
    optimizer = SimpleNamespace(
        param_groups=[
            {"params": [], "max_lr": 1.0, "min_lr": 0.1, "default_config": False},
            {"params": [], "max_lr": 2.0, "min_lr": 0.2, "default_config": False},
        ]
    )
    runtime = _snapshot_optimizer_group_lr_bounds(optimizer)
    optimizer.param_groups[1].update(max_lr=3.0, min_lr=0.3)
    before = deepcopy(optimizer.param_groups)
    with pytest.raises(ValueError, match="unambiguously"):
        _restore_optimizer_group_lr_bounds(optimizer, runtime, OptimizerConfig(lr=1.0))
    assert optimizer.param_groups == before


def test_changed_parameter_membership_is_rejected():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    runtime = _snapshot_optimizer_group_lr_bounds(optimizer)
    optimizer.param_groups[0]["params"] = [torch.nn.Parameter(torch.ones(1))]
    with pytest.raises(ValueError, match="membership changed"):
        _restore_optimizer_group_lr_bounds(optimizer, runtime, OptimizerConfig(lr=1.0))
