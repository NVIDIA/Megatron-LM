# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest

from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.training.checkpointing import (
    _restore_optimizer_param_group_lr_bounds,
    _snapshot_optimizer_param_group_lr_bounds,
)


@pytest.mark.parametrize("num_steps", [5, 42, 100])
def test_scheduler_override_preserves_runtime_parameter_group_bounds(num_steps: int) -> None:
    """Runtime group overrides survive checkpoint metadata and native scheduler loading."""
    optimizer = SimpleNamespace(
        param_groups=[
            {"params": [], "max_lr": 1.0, "min_lr": 0.1},
            {"params": [], "max_lr": 5.0, "min_lr": 0.5, "lr_mult": 5.0},
            {"params": [], "max_lr": 0.5, "min_lr": 0.05, "is_decoupled_lr": True},
            {"params": [], "max_lr": 0.7, "min_lr": 0.25, "lr_mult": 1.0},
            {"params": []},
        ]
    )
    scheduler = OptimizerParamScheduler(
        optimizer=optimizer,
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
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=True,
    )
    runtime_bounds = _snapshot_optimizer_param_group_lr_bounds(optimizer.param_groups)
    # PyTorch optimizer load_state_dict replaces group dictionaries while keeping
    # their native order. Empty PP groups also need their original LR schedule.
    optimizer.param_groups = [
        dict(group, max_lr=99.0, min_lr=88.0, checkpoint_marker=True)
        for group in optimizer.param_groups
    ]
    scheduler_state = dict(scheduler.state_dict(), max_lr=99.0, min_lr=88.0, num_steps=num_steps)
    scheduler.load_state_dict(scheduler_state)
    _restore_optimizer_param_group_lr_bounds(optimizer.param_groups, runtime_bounds)
    scheduler.step(0)

    assert scheduler.num_steps == num_steps
    for group, (max_lr, min_lr) in zip(
        optimizer.param_groups, [(1.0, 0.1), (5.0, 0.5), (0.5, 0.05), (0.7, 0.25), (1.0, 0.1)]
    ):
        expected = (
            max_lr * num_steps / 10
            if num_steps <= 10
            else max_lr - (max_lr - min_lr) * (num_steps - 10) / 90
        )
        assert group["lr"] == pytest.approx(expected)
        assert group["checkpoint_marker"]
    assert "max_lr" not in optimizer.param_groups[-1]
    assert "min_lr" not in optimizer.param_groups[-1]
    assert optimizer.param_groups[3]["min_lr"] == 0.25


def test_scheduler_override_rejects_changed_parameter_group_count() -> None:
    """Do not silently apply runtime schedules to an incompatible optimizer layout."""
    with pytest.raises(ValueError, match="parameter-group count changed"):
        _restore_optimizer_param_group_lr_bounds([{}], [{}, {}])
