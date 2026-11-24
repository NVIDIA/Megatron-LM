# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""training_log() reads its stats on device and post-processes them on the host.

The tensorboard block and the stdout block are independent `if`s that both read
and post-process the same stats. These tests pin the contract that the
post-processing -- notably rooting the squared params norm -- is applied exactly
once no matter which of the two blocks run.
"""

import math
import re
from argparse import ArgumentParser
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.training import arguments, training
from megatron.training.config import LoggerConfig
from megatron.training.state import TrainState


class _Writer:
    """Records add_scalar calls like a tensorboard SummaryWriter."""

    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, step):
        self.scalars[name] = value


class _Timer:
    def elapsed(self, *args, **kwargs):
        return 1.0


class _Timers:
    def __call__(self, name, log_level=None):
        return _Timer()

    def log(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass


_ARGS = SimpleNamespace(freeze_all_layers=False)


@pytest.fixture
def log_env(monkeypatch, run_config):
    """Stub every global training_log() reaches for; capture the stdout line."""
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args(['--micro-batch-size', '1'])
    args.data_parallel_size = args.world_size = args.gtp_weight_remat_size = 1
    args.dsa_indexer_loss_coeff = None
    run_config.train.micro_batch_size = 1
    run_config.train.train_iters = 10
    printed = []
    env = SimpleNamespace(writer=None, printed=printed, run_config=run_config)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_train_state", lambda: TrainState())
    monkeypatch.setattr(training, "get_timers", lambda: _Timers())
    monkeypatch.setattr(training, "get_tensorboard_writer", lambda: env.writer)
    for getter in ("get_wandb_writer", "get_one_logger", "get_energy_monitor", "get_telemetry"):
        monkeypatch.setattr(training, getter, lambda: None)
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(training, "num_floating_point_operations", lambda *a, **k: 0.0)
    monkeypatch.setattr(training, "one_logger_utils", Mock())
    # Single rank: the model-parallel max of a stat is the stat.
    monkeypatch.setattr(
        training, "reduce_max_stat_across_model_parallel_group", lambda stat, **k: stat
    )
    monkeypatch.setattr(training, "print_rank_last", printed.append)
    return env


def _run_training_log(env, log_interval, tensorboard_log_interval, writer, params_norm_sq):
    env.run_config.logger = LoggerConfig(
        log_interval=log_interval, tensorboard_log_interval=tensorboard_log_interval
    )
    env.writer = writer
    training.training_log(
        loss_dict={},
        total_loss_dict={},
        learning_rate=1e-3,
        iteration=2,
        loss_scale=1.0,
        report_memory_flag=False,
        skipped_iter=0,
        grad_norm=torch.tensor([3.0]),
        params_norm=torch.tensor([params_norm_sq]),
        num_zeros_in_grad=None,
        max_attention_logit=None,
    )


def _stdout_params_norm(printed):
    assert len(printed) == 1, printed
    match = re.search(r"params norm: ([0-9.]+)", printed[0])
    assert match, printed[0]
    return float(match.group(1))


def test_params_norm_rooted_once_when_writer_and_log_intervals_coincide(log_env):
    """Both blocks run: the stdout norm must equal the tensorboard norm, i.e. sqrt once."""
    writer = _Writer()
    _run_training_log(log_env, 1, 1, writer, params_norm_sq=10000.0)
    assert writer.scalars["params-norm"] == pytest.approx(100.0)
    assert _stdout_params_norm(log_env.printed) == pytest.approx(100.0)
    assert writer.scalars["grad-norm"] == pytest.approx(3.0)
    assert "grad norm: 3.000" in log_env.printed[0]


def test_params_norm_rooted_once_without_writer(log_env):
    _run_training_log(log_env, 1, 1, None, params_norm_sq=10000.0)
    assert _stdout_params_norm(log_env.printed) == pytest.approx(100.0)


def test_params_norm_rooted_once_when_only_writer_runs(log_env):
    """iteration 2: writer interval hits, log interval does not."""
    writer = _Writer()
    _run_training_log(log_env, 5, 2, writer, params_norm_sq=49.0)
    assert writer.scalars["params-norm"] == pytest.approx(math.sqrt(49.0))
    assert log_env.printed == []


def _stats(**overrides):
    values = dict(
        loss_scale=torch.tensor([2.0]),
        params_norm=torch.tensor([9.0]),
        grad_norm=torch.tensor([-1.0]),
        num_zeros_in_grad=torch.tensor([5.0]),
        learning_rate=torch.tensor([0.5]),
    )
    values.update(overrides)
    return training._LoggingStats(**values)


def test_read_logging_stats_resolves_each_stat_and_loss_by_name():
    stats = _stats()
    totals = training._read_logging_stats(
        _ARGS, stats, {"lm loss": torch.tensor([4.0]), "z loss": torch.tensor([6.0])}
    )
    assert stats.loss_scale == 2.0
    assert stats.params_norm == pytest.approx(3.0)
    assert stats.grad_norm is None  # the -1.0 "no rank had this stat" sentinel
    assert stats.num_zeros_in_grad == 5.0
    assert stats.learning_rate == 0.5
    assert totals == {"lm loss": 4.0, "z loss": 6.0}


def test_read_logging_stats_finishes_the_stats_once():
    """A second read resolves new loss totals and leaves the finished stats alone."""
    args = _ARGS
    stats = _stats()
    training._read_logging_stats(args, stats)
    totals = training._read_logging_stats(args, stats, {"lm loss": torch.tensor([4.0])})
    assert stats.params_norm == pytest.approx(3.0)
    assert totals == {"lm loss": 4.0}


def test_read_logging_stats_is_one_device_to_host_copy(monkeypatch):
    copies = []
    cat = torch.cat
    monkeypatch.setattr(torch, "cat", lambda *a, **k: copies.append(1) or cat(*a, **k))
    training._read_logging_stats(_ARGS, _stats(), {"lm loss": torch.tensor([4.0])})
    assert len(copies) == 1


def test_read_logging_stats_passes_host_values_through():
    stats = _stats(loss_scale=1.0, params_norm=None, learning_rate=None)
    totals = training._read_logging_stats(_ARGS, stats, {"lm loss": 4.0})
    assert stats.loss_scale == 1.0
    assert stats.params_norm is None
    assert stats.learning_rate is None
    assert totals == {"lm loss": 4.0}


def test_host_stat_reads_a_tensor_and_decodes_the_absent_sentinel():
    assert training._host_stat(torch.tensor([2.0])) == 2.0
    assert training._host_stat(torch.tensor([-1.0])) is None
    assert training._host_stat(2.0) == 2.0
    assert training._host_stat(None) is None
