import math
from unittest.mock import MagicMock

import pytest

from megatron.core.optimizer_param_scheduler import (  # Adjust import according to your module path
    OptimizerParamScheduler,
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
