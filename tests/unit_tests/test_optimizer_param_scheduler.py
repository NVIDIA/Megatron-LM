import math
from unittest.mock import MagicMock

import pytest

from megatron.core.optimizer_param_scheduler import (  # Adjust import according to your module path
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


class TestGetCanonicalLrForLogging:
    """Tests for get_canonical_lr_for_logging."""

    # ── Case 4: no groups at all → None (stub optimizer) ────────────────────

    def test_empty_param_groups(self):
        """Return None when the param_groups list is empty (stub optimizer)."""
        assert get_canonical_lr_for_logging([]) is None

    def test_all_lrs_none(self):
        """Return None when every group has lr=None."""
        param_groups = [
            {'params': [1], 'lr': None, 'default_config': True},
            {'params': [2], 'lr': None},
        ]
        assert get_canonical_lr_for_logging(param_groups) is None

    # ── Case 1: default_config groups ───────────────────────────────────────

    def test_single_default_config_group(self):
        """A single default_config group with a valid lr is returned."""
        param_groups = [{'params': [1, 2, 3], 'lr': 0.05, 'default_config': True}]
        assert get_canonical_lr_for_logging(param_groups) == 0.05

    def test_default_config_empty_params_still_used(self):
        """Empty default_config groups still have valid lr from the scheduler."""
        param_groups = [{'params': [], 'lr': 0.05, 'default_config': True}]
        assert get_canonical_lr_for_logging(param_groups) == 0.05

    def test_multiple_default_config_groups_first_valid(self):
        """First non-None lr among default_config groups is returned."""
        param_groups = [
            {'params': [1], 'lr': 0.01, 'default_config': True},
            {'params': [2], 'lr': 0.02, 'default_config': True},
        ]
        assert get_canonical_lr_for_logging(param_groups) == 0.01

    def test_default_config_first_none_lr(self):
        """Skip None lr in default_config groups, return the next valid one."""
        param_groups = [
            {'params': [1], 'lr': None, 'default_config': True},
            {'params': [2], 'lr': 0.03, 'default_config': True},
        ]
        assert get_canonical_lr_for_logging(param_groups) == 0.03

    def test_default_config_all_none_lr_falls_through(self):
        """If every default_config group has lr=None, fall through to case 2/3."""
        param_groups = [
            {'params': [1], 'lr': None, 'default_config': True},
            {'params': [1, 2, 3], 'lr': 0.07},
        ]
        # default_config group has None lr so falls through; non-default group wins.
        assert get_canonical_lr_for_logging(param_groups) == 0.07

    def test_default_config_all_none_lr_no_fallback(self):
        """All default_config lr=None and no non-default groups → None."""
        param_groups = [{'params': [1], 'lr': None, 'default_config': True}]
        assert get_canonical_lr_for_logging(param_groups) is None

    # ── Case 2: non-default_config groups (fallback by param count) ─────────

    def test_single_non_default_group(self):
        """A single non-default group with valid lr is returned."""
        param_groups = [{'params': [1, 2], 'lr': 0.1}]
        assert get_canonical_lr_for_logging(param_groups) == 0.1

    def test_non_default_picks_group_with_most_params(self):
        """Among non-default groups, lr from the group with the most params wins."""
        param_groups = [
            {'params': [1], 'lr': 0.01},
            {'params': [1, 2, 3, 4, 5], 'lr': 0.05},
            {'params': [1, 2], 'lr': 0.02},
        ]
        assert get_canonical_lr_for_logging(param_groups) == 0.05

    def test_non_default_skips_none_lr(self):
        """Groups with lr=None are skipped even if they have the most params."""
        param_groups = [{'params': [1, 2, 3, 4, 5], 'lr': None}, {'params': [1, 2], 'lr': 0.04}]
        assert get_canonical_lr_for_logging(param_groups) == 0.04

    def test_non_default_all_none_lr(self):
        """All non-default groups with lr=None → None."""
        param_groups = [{'params': [1], 'lr': None}, {'params': [1, 2], 'lr': None}]
        assert get_canonical_lr_for_logging(param_groups) is None

    # ── Case 1 takes priority over Case 2 ──────────────────────────────────

    def test_default_config_takes_priority_over_non_default(self):
        """default_config group lr is preferred even if a non-default group has more params."""
        param_groups = [
            {'params': [1], 'lr': 0.001, 'default_config': True},
            {'params': [1, 2, 3, 4, 5, 6], 'lr': 0.999},
        ]
        assert get_canonical_lr_for_logging(param_groups) == 0.001

    # ── Case 3: non-default empty groups as last resort ─────────────────────

    def test_non_default_empty_group_used_as_fallback(self):
        """Empty non-default groups are used when no non-empty non-default group has lr."""
        param_groups = [{'params': [], 'lr': 0.42}]
        assert get_canonical_lr_for_logging(param_groups) == 0.42

    def test_non_default_prefers_nonempty_over_empty(self):
        """Non-empty non-default groups are preferred over empty ones."""
        param_groups = [{'params': [], 'lr': 0.42}, {'params': [1, 2], 'lr': 0.07}]
        assert get_canonical_lr_for_logging(param_groups) == 0.07

    def test_non_default_all_empty_picks_first_valid(self):
        """Among all-empty non-default groups, the first non-None lr wins."""
        param_groups = [
            {'params': [], 'lr': None},
            {'params': [], 'lr': 0.15},
            {'params': [], 'lr': 0.25},
        ]
        assert get_canonical_lr_for_logging(param_groups) == 0.15

    # ── Edge cases ──────────────────────────────────────────────────────────

    def test_lr_zero_is_valid(self):
        """lr=0.0 is a legitimate value, not to be confused with None."""
        param_groups = [{'params': [1], 'lr': 0.0, 'default_config': True}]
        assert get_canonical_lr_for_logging(param_groups) == 0.0

    def test_default_config_false_treated_as_non_default(self):
        """Explicitly setting default_config=False makes the group non-default."""
        param_groups = [{'params': [1, 2], 'lr': 0.08, 'default_config': False}]
        assert get_canonical_lr_for_logging(param_groups) == 0.08

    def test_missing_default_config_key_treated_as_non_default(self):
        """Groups without the default_config key are non-default."""
        param_groups = [{'params': [1], 'lr': 0.06}]
        assert get_canonical_lr_for_logging(param_groups) == 0.06

    def test_mixed_empty_and_nonempty_default_config(self):
        """Empty default_config group lr is used (scheduler sets it correctly)."""
        param_groups = [
            {'params': [], 'lr': 0.99, 'default_config': True},
            {'params': [1], 'lr': 0.11},
        ]
        # default_config group is preferred regardless of param count.
        assert get_canonical_lr_for_logging(param_groups) == 0.99

    def test_tie_in_param_count_picks_first(self):
        """When two non-default groups have equal param counts, the first seen wins."""
        param_groups = [{'params': [1, 2], 'lr': 0.10}, {'params': [3, 4], 'lr': 0.20}]
        assert get_canonical_lr_for_logging(param_groups) == 0.10

    def test_rank_with_only_empty_groups(self):
        """Simulates a rank where all groups are empty (rank-alignment stubs).

        The scheduler still writes lr on these groups, so we should find it.
        """
        param_groups = [
            {'params': [], 'lr': 0.001, 'default_config': True},
            {'params': [], 'lr': 0.002, 'default_config': False},
        ]
        assert get_canonical_lr_for_logging(param_groups) == 0.001

    def test_missing_lr_key(self):
        """Groups that have never been through the scheduler lack the 'lr' key."""
        param_groups = [{'params': [1], 'default_config': True}]
        # param_group.get('lr') returns None → treated as no valid lr.
        assert get_canonical_lr_for_logging(param_groups) is None
