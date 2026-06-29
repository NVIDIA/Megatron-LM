# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from unittest.mock import MagicMock

import pytest

from megatron.core.optimizer_param_scheduler import (
    OptimizerParamScheduler,
    get_canonical_lr_for_logging,
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
