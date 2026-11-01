# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from unittest.mock import MagicMock

import pytest

from megatron.core.optimizer_param_scheduler import (
    OptimizerParamScheduler,
    get_canonical_lr_for_logging,
    parse_step_lr_rescale_schedule,
)


@pytest.fixture
def mock_optimizer():
    optimizer = MagicMock()
    optimizer.param_groups = [{'lr': 0.0, 'weight_decay': 0.0}]
    return optimizer


def test_initialization(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    assert scheduler.init_lr == 0.01
    assert scheduler.max_lr == 0.1
    assert scheduler.min_lr == 0.001
    assert scheduler.lr_warmup_steps == 100
    assert scheduler.lr_decay_steps == 1000
    assert scheduler.lr_decay_style == 'linear'
    assert scheduler.start_wd == 0.0
    assert scheduler.end_wd == 0.1
    assert scheduler.wd_incr_steps == 1000
    assert scheduler.wd_incr_style == 'linear'


def test_get_wd_constant(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='constant',
    )

    scheduler.step(500)
    wd = scheduler.get_wd()
    assert wd == 0.1


def test_get_wd_linear(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    scheduler.step(500)
    wd = scheduler.get_wd()
    assert wd == 0.05


def test_get_wd_cosine(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='cosine',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='cosine',
    )

    scheduler.step(500)
    wd = scheduler.get_wd()
    expected_wd = 0.05 * (math.cos(math.pi * (1 - 0.5)) + 1.0)
    assert math.isclose(wd, expected_wd, rel_tol=1e-5)


def test_get_lr_linear(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    param_group = {'max_lr': 0.1, 'min_lr': 0.001}

    scheduler.step(50)
    lr = scheduler.get_lr(param_group)
    expected_lr = 0.01 + (0.1 - 0.01) * (50 / 100)
    assert math.isclose(lr, expected_lr, rel_tol=1e-5)

    scheduler.step(450)
    lr = scheduler.get_lr(param_group)
    expected_lr = 0.1 - ((0.1 - 0.001) * ((500 - 100) / (1000 - 100)))
    assert math.isclose(lr, expected_lr, rel_tol=1e-5)

    scheduler.step(501)
    lr = scheduler.get_lr(param_group)
    expected_lr = 0.001
    assert math.isclose(lr, expected_lr, rel_tol=1e-5)


def test_get_lr_cosine(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='cosine',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    scheduler.step(500)
    param_group = {'max_lr': 0.1, 'min_lr': 0.001}
    lr = scheduler.get_lr(param_group)
    expected_lr = 0.001 + (0.1 - 0.001) * 0.5 * (
        math.cos(math.pi * ((500 - 100) / (1000 - 100))) + 1.0
    )
    assert math.isclose(lr, expected_lr, rel_tol=1e-5)


def test_step_function(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    scheduler.step(100)
    assert scheduler.num_steps == 100
    param_group = mock_optimizer.param_groups[0]
    assert math.isclose(param_group['lr'], 0.01 + (0.1 - 0.01) * (100 / 100), rel_tol=1e-5)
    assert math.isclose(param_group['weight_decay'], 0.01, rel_tol=1e-5)


def test_step_lr_rescale_schedule(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.0,
        max_lr=0.1,
        min_lr=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=1000,
        lr_decay_style="constant",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=1000,
        wd_incr_style="constant",
        step_lr_rescale_schedule="0:0.5,100:0.75 200:1.0",
    )

    assert mock_optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    scheduler.step(99)
    assert mock_optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    scheduler.step(1)
    assert mock_optimizer.param_groups[0]["lr"] == pytest.approx(0.075)
    scheduler.step(100)
    assert mock_optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert scheduler.state_dict()["step_lr_rescale_schedule"] == ((0, 0.5), (100, 0.75), (200, 1.0))
    assert scheduler.state_dict()["step_lr_rescale_warmup_samples"] == 0


def test_step_lr_rescale_schedule_is_disabled_by_default(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.0,
        max_lr=0.1,
        min_lr=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=1000,
        lr_decay_style="constant",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=1000,
        wd_incr_style="constant",
    )

    assert scheduler.step_lr_rescale_schedule == ()
    assert scheduler.step_lr_rescale_warmup_samples == 0
    assert scheduler.get_lr_rescale() == 1.0
    assert mock_optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert "step_lr_rescale_schedule" not in scheduler.state_dict()
    assert "step_lr_rescale_warmup_samples" not in scheduler.state_dict()


def test_step_lr_rescale_without_transition_warmup_only_warms_initial_stage(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.0,
        max_lr=0.1,
        min_lr=0.0,
        lr_warmup_steps=20,
        lr_decay_steps=1000,
        lr_decay_style="constant",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=1000,
        wd_incr_style="constant",
        step_lr_rescale_schedule="0:0.5,100:0.75",
    )

    for samples, expected_lr in [(0, 0.0), (10, 0.025), (20, 0.05), (99, 0.05), (100, 0.075)]:
        scheduler.step(samples - scheduler.num_steps)
        assert mock_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)


def test_step_lr_rescale_transition_warmup_composes_with_initial_warmup():
    regular_group = {"lr": -1.0, "weight_decay": -1.0}
    optimizer = MagicMock(param_groups=[regular_group])
    scheduler = OptimizerParamScheduler(
        optimizer=optimizer,
        init_lr=0.0,
        max_lr=0.1,
        min_lr=0.0,
        lr_warmup_steps=20,
        lr_decay_steps=1000,
        lr_decay_style="constant",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=1000,
        wd_incr_style="constant",
        step_lr_rescale_schedule="0:0.5,100:0.75,200:1.0",
        step_lr_rescale_warmup_samples=40,
    )

    checkpoint_state = None
    for samples, expected_lr in [
        (0, 0.0),
        (10, 0.025),
        (20, 0.05),
        (99, 0.05),
        (100, 0.05),
        (120, 0.0625),
        (140, 0.075),
        (199, 0.075),
        (200, 0.075),
        (220, 0.0875),
        (240, 0.1),
    ]:
        scheduler.step(samples - scheduler.num_steps)
        assert regular_group["lr"] == pytest.approx(expected_lr)
        if samples == 120:
            checkpoint_state = scheduler.state_dict()

    state_dict = scheduler.state_dict()
    assert state_dict["step_lr_rescale_warmup_samples"] == 40

    restored_regular_group = {"lr": -1.0, "weight_decay": -1.0}
    restored_scheduler = OptimizerParamScheduler(
        optimizer=MagicMock(param_groups=[restored_regular_group]),
        init_lr=0.0,
        max_lr=0.1,
        min_lr=0.0,
        lr_warmup_steps=20,
        lr_decay_steps=1000,
        lr_decay_style="constant",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=1000,
        wd_incr_style="constant",
        use_checkpoint_opt_param_scheduler=True,
        step_lr_rescale_schedule="0:0.5,100:0.75,200:1.0",
        step_lr_rescale_warmup_samples=20,
    )
    assert checkpoint_state is not None
    restored_scheduler.load_state_dict(checkpoint_state)
    assert restored_scheduler.step_lr_rescale_warmup_samples == 40
    assert restored_scheduler.num_steps == 120
    assert restored_regular_group["lr"] == pytest.approx(0.0625)


def test_step_lr_rescale_transition_warmup_resumes_checkpoint_without_warmup_field():
    old_optimizer = MagicMock(param_groups=[{"lr": -1.0, "weight_decay": -1.0}])
    old_scheduler = OptimizerParamScheduler(
        optimizer=old_optimizer,
        init_lr=0.0,
        max_lr=0.1,
        min_lr=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=1000,
        lr_decay_style="constant",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=1000,
        wd_incr_style="constant",
        step_lr_rescale_schedule="0:0.5,100:0.75,200:1.0",
    )
    old_scheduler.step(100)
    checkpoint_state = old_scheduler.state_dict()
    # Checkpoints written before the warmup option existed carry only the schedule.
    del checkpoint_state["step_lr_rescale_warmup_samples"]

    restored_group = {"lr": -1.0, "weight_decay": -1.0}
    restored_scheduler = OptimizerParamScheduler(
        optimizer=MagicMock(param_groups=[restored_group]),
        init_lr=0.0,
        max_lr=0.1,
        min_lr=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=1000,
        lr_decay_style="constant",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=1000,
        wd_incr_style="constant",
        step_lr_rescale_schedule="0:0.5,100:0.75,200:1.0",
        step_lr_rescale_warmup_samples=40,
    )
    restored_scheduler.load_state_dict(checkpoint_state)

    assert restored_scheduler.step_lr_rescale_warmup_samples == 40
    assert restored_scheduler.num_steps == 100
    assert restored_group["lr"] == pytest.approx(0.05)

    restored_scheduler.step(20)
    assert restored_group["lr"] == pytest.approx(0.0625)


def _make_constant_rescale_scheduler(param_group, warmup_samples, **kwargs):
    return OptimizerParamScheduler(
        optimizer=MagicMock(param_groups=[param_group]),
        init_lr=0.0,
        max_lr=0.1,
        min_lr=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=1000,
        lr_decay_style="constant",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=1000,
        wd_incr_style="constant",
        step_lr_rescale_schedule="0:0.5,100:0.75,200:1.0",
        step_lr_rescale_warmup_samples=warmup_samples,
        **kwargs,
    )


@pytest.fixture
def zero_warmup_checkpoint_state():
    scheduler = _make_constant_rescale_scheduler({"lr": -1.0, "weight_decay": -1.0}, 0)
    scheduler.step(100)
    state_dict = scheduler.state_dict()
    assert state_dict["step_lr_rescale_warmup_samples"] == 0
    return state_dict


def test_step_lr_rescale_restores_zero_transition_warmup_from_checkpoint(
    zero_warmup_checkpoint_state,
):
    restored_group = {"lr": -1.0, "weight_decay": -1.0}
    restored_scheduler = _make_constant_rescale_scheduler(
        restored_group, 40, use_checkpoint_opt_param_scheduler=True
    )
    restored_scheduler.load_state_dict(zero_warmup_checkpoint_state)

    assert restored_scheduler.step_lr_rescale_warmup_samples == 0
    assert restored_group["lr"] == pytest.approx(0.075)


def test_step_lr_rescale_rejects_mismatched_zero_transition_warmup(zero_warmup_checkpoint_state):
    restored_scheduler = _make_constant_rescale_scheduler(
        {"lr": -1.0, "weight_decay": -1.0}, 40, use_checkpoint_opt_param_scheduler=False
    )
    with pytest.raises(AssertionError, match="step LR rescale warmup samples"):
        restored_scheduler.load_state_dict(zero_warmup_checkpoint_state)


def test_step_lr_rescale_override_keeps_runtime_transition_warmup(zero_warmup_checkpoint_state):
    restored_group = {"lr": -1.0, "weight_decay": -1.0}
    restored_scheduler = _make_constant_rescale_scheduler(
        restored_group,
        40,
        override_opt_param_scheduler=True,
        use_checkpoint_opt_param_scheduler=False,
    )
    restored_scheduler.load_state_dict(zero_warmup_checkpoint_state)

    assert restored_scheduler.step_lr_rescale_warmup_samples == 40
    assert restored_group["lr"] == pytest.approx(0.05)


@pytest.mark.parametrize("lr_decay_style", ["linear", "cosine", "inverse-square-root", "WSD"])
def test_step_lr_rescale_composes_with_decay_and_group_bounds(lr_decay_style):
    def make_scheduler(schedule):
        param_groups = [
            {"params": [], "default_config": True},
            {"params": [object()], "max_lr": 0.2, "min_lr": 0.02},
        ]
        scheduler = OptimizerParamScheduler(
            optimizer=MagicMock(param_groups=param_groups),
            init_lr=0.0,
            max_lr=0.1,
            min_lr=0.01,
            lr_warmup_steps=120,
            lr_decay_steps=1000,
            lr_decay_style=lr_decay_style,
            start_wd=0.0,
            end_wd=0.0,
            wd_incr_steps=1000,
            wd_incr_style="constant",
            wsd_decay_steps=200,
            lr_wsd_decay_style="linear",
            step_lr_rescale_schedule=schedule,
            step_lr_rescale_warmup_samples=40 if schedule else 0,
        )
        return scheduler, param_groups

    base_scheduler, base_groups = make_scheduler(None)
    rescaled_scheduler, rescaled_groups = make_scheduler("0:0.5,100:0.75")
    # Covers the initial warmup, the overlapping transition ramp, decay, and the LR floor.
    for samples, factor in [
        (50, 0.5),
        (100, 0.5),
        (120, 0.625),
        (140, 0.75),
        (500, 0.75),
        (900, 0.75),
        (1001, 0.75),
    ]:
        base_scheduler.step(samples - base_scheduler.num_steps)
        rescaled_scheduler.step(samples - rescaled_scheduler.num_steps)
        for base_group, rescaled_group in zip(base_groups, rescaled_groups):
            assert rescaled_group["lr"] == pytest.approx(base_group["lr"] * factor)
            assert rescaled_group["weight_decay"] == base_group["weight_decay"]
        assert get_canonical_lr_for_logging(rescaled_groups) == pytest.approx(
            rescaled_groups[0]["lr"]
        )

    assert rescaled_groups[0]["lr"] == pytest.approx(0.0075)
    assert rescaled_groups[1]["lr"] == pytest.approx(0.015)


@pytest.mark.parametrize(
    ("schedule", "warmup_samples", "message"),
    [
        ("0:0.5,100:1.0", -1, "must be nonnegative"),
        (None, 10, "requires a schedule with at least two entries"),
        ("0:0.5", 10, "requires a schedule with at least two entries"),
        ("0:0.5,100:0.75,150:1.0", 60, "must not exceed the gap between transitions"),
    ],
)
def test_step_lr_rescale_transition_warmup_rejects_invalid_values(
    mock_optimizer, schedule, warmup_samples, message
):
    with pytest.raises(ValueError, match=message):
        OptimizerParamScheduler(
            optimizer=mock_optimizer,
            init_lr=0.0,
            max_lr=0.1,
            min_lr=0.0,
            lr_warmup_steps=0,
            lr_decay_steps=1000,
            lr_decay_style="constant",
            start_wd=0.0,
            end_wd=0.0,
            wd_incr_steps=1000,
            wd_incr_style="constant",
            step_lr_rescale_schedule=schedule,
            step_lr_rescale_warmup_samples=warmup_samples,
        )


@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        ("100:1.0", "must begin at sample 0"),
        ("0:1.0,0:0.5", "must be strictly increasing"),
        ("0:0", "must be finite and positive"),
        ("0:nan", "must be finite and positive"),
        ("0.5:1.0", "expected SAMPLES:MULTIPLIER"),
    ],
)
def test_parse_step_lr_rescale_schedule_rejects_invalid_values(schedule, message):
    with pytest.raises(ValueError, match=message):
        parse_step_lr_rescale_schedule(schedule)


def test_step_updates_empty_param_groups():
    """Empty param groups (rank-alignment stubs) must still receive lr updates.

    get_canonical_lr_for_logging reads lr from default_config groups regardless
    of whether they hold parameters, so step() must not skip them.
    """
    optimizer = MagicMock()
    # lr and weight_decay are set by the scheduler's step() method
    optimizer.param_groups = [
        # Non-default group with its own max_lr override (lr will differ from the canonical schedule)
        {'params': [1, 2], "min_lr": 0.001, "max_lr": 0.2, "default_config": False},
        # Model parallelism may leave default_config groups empty on some ranks
        {'params': [], "wd_mult": 0.0, 'default_config': True},
    ]
    scheduler = OptimizerParamScheduler(
        optimizer=optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    scheduler.step(100)
    non_empty, empty = optimizer.param_groups

    # Verify learning rates: at step 100 warmup is complete so lr == max_lr
    assert "lr" in non_empty, "non-empty param group must have an lr"
    assert "lr" in empty, "empty param group must have an lr"
    assert non_empty['lr'] == pytest.approx(0.2)  # warmup complete → this group's max_lr override
    assert empty['lr'] == pytest.approx(0.1)  # warmup complete → scheduler's default max_lr
    assert get_canonical_lr_for_logging(optimizer.param_groups) == pytest.approx(0.1)

    # Verify weight decay: linear from 0.0 to 0.1 over 1000 steps → base wd is 0.01 at step 100
    assert "weight_decay" in non_empty, "non-empty param group must have a weight decay"
    assert "weight_decay" in empty, "empty param group must have a weight decay"
    assert non_empty['weight_decay'] == pytest.approx(0.01)  # base wd, no wd_mult override
    assert empty['weight_decay'] == pytest.approx(0.0)  # base wd * wd_mult=0.0


def test_state_dict(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    state_dict = scheduler.state_dict()
    assert state_dict['max_lr'] == 0.1
    assert state_dict['lr_warmup_steps'] == 100
    assert state_dict['num_steps'] == 0
    assert state_dict['lr_decay_style'] == 'linear'
    assert state_dict['lr_decay_steps'] == 1000
    assert state_dict['min_lr'] == 0.001
    assert state_dict['start_wd'] == 0.0
    assert state_dict['end_wd'] == 0.1
    assert state_dict['wd_incr_style'] == 'linear'
    assert state_dict['wd_incr_steps'] == 1000


def test_load_state_dict(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    state_dict = {
        'max_lr': 0.2,
        'min_lr': 0.0005,
        'lr_warmup_steps': 200,
        'lr_decay_steps': 2000,
        'lr_decay_style': 'cosine',
        'num_steps': 500,
        'start_wd': 0.01,
        'end_wd': 0.2,
        'wd_incr_steps': 500,
        'wd_incr_style': 'cosine',
    }

    scheduler.load_state_dict(state_dict)
    assert scheduler.max_lr == 0.2
    assert scheduler.min_lr == 0.0005
    assert scheduler.lr_warmup_steps == 200
    assert scheduler.lr_decay_steps == 2000
    assert scheduler.lr_decay_style == 'cosine'
    assert scheduler.num_steps == 500
    assert scheduler.start_wd == 0.01
    assert scheduler.end_wd == 0.2
    assert scheduler.wd_incr_steps == 500
    assert scheduler.wd_incr_style == 'cosine'


# ── get_canonical_lr_for_logging tests ──────────────────────────────────────
#
# Returns the lr of the first default_config=True param group.  In practice
# the scheduler always sets a valid lr on every group (including empty
# rank-alignment stubs), so a default_config=True group with a float lr is
# always present.


class TestGetCanonicalLrForLogging:
    """Tests for get_canonical_lr_for_logging."""

    def test_single_default_config_group(self):
        """Typical case: one default_config group with a valid lr."""
        param_groups = [{'lr': 0.05, 'default_config': True}]
        assert get_canonical_lr_for_logging(param_groups) == 0.05

    def test_default_config_with_non_default_groups(self):
        """default_config group is returned even when non-default groups are present."""
        param_groups = [{'lr': 0.001, 'default_config': True}, {'lr': 0.999}]
        assert get_canonical_lr_for_logging(param_groups) == 0.001

    def test_default_config_after_non_default(self):
        """default_config group is found even when it is not first in the list."""
        param_groups = [{'lr': 0.50}, {'lr': 0.01, 'default_config': True}]
        assert get_canonical_lr_for_logging(param_groups) == 0.01

    def test_no_default_config_groups(self):
        """Returns None when no group has default_config=True."""
        param_groups = [{'lr': 0.50}, {'lr': 0.01}]
        assert get_canonical_lr_for_logging(param_groups) is None

    def test_missing_lr_key(self):
        """Returns None (not KeyError) when the default_config group lacks an 'lr' key."""
        param_groups = [{'default_config': True}]
        assert get_canonical_lr_for_logging(param_groups) is None

    def test_empty_param_groups(self):
        """Returns None when there are no param groups at all."""
        assert get_canonical_lr_for_logging([]) is None

    def test_no_default_config_no_lr(self):
        """Returns None when groups exist but none are default_config."""
        param_groups = [{'params': []}]
        assert get_canonical_lr_for_logging(param_groups) is None

    def test_lr_zero_is_valid(self):
        """lr=0.0 is a legitimate value, not to be confused with None."""
        param_groups = [{'lr': 0.0, 'default_config': True}]
        assert get_canonical_lr_for_logging(param_groups) == 0.0
