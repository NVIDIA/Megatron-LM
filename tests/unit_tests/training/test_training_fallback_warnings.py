# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Training code warns when a caller omits its process groups, and uses the groups it is given.

Each fallback to the global parallel state emits a ProcessGroupFallbackWarning that names the
function whose caller omitted the groups. The tests replace the global read with a stub, so they
run without torch.distributed.
"""

import warnings
from argparse import ArgumentParser
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.training import arguments, training
from megatron.training.state import TrainState


class _GlobalRead(Exception):
    """Raised by a stubbed read of the global parallel state."""


def _raise_global_read(*args, **kwargs):
    raise _GlobalRead


class _Group:
    def __init__(self, size=2, rank=0):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


@pytest.fixture(autouse=True)
def fresh_fallback_record(monkeypatch):
    """Forget which fallbacks warned, so that each test sees its own warning."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


@contextmanager
def _record_fallbacks():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", ProcessGroupFallbackWarning)
        yield record


def _owners(record):
    """The functions named by the fallback warnings, in order."""
    return [
        str(w.message).split()[0]
        for w in record
        if issubclass(w.category, ProcessGroupFallbackWarning)
    ]


def _assert_warns_then_reads_global(owner, call):
    """``call`` warns that ``owner`` fell back, then reads the global parallel state."""
    with _record_fallbacks() as record, pytest.raises(_GlobalRead):
        call()
    assert _owners(record) == [owner]


class TestModelCollection:

    def test_prefers_the_model_collection_then_the_run_collection(self, monkeypatch):
        model_pg_collection, run_pg_collection = ProcessGroupCollection(), ProcessGroupCollection()
        monkeypatch.setattr(
            ProcessGroupCollection, "use_mpu_process_groups", classmethod(_raise_global_read)
        )

        with _record_fallbacks() as record:
            assert (
                training._get_model_pg_collection(
                    [SimpleNamespace(pg_collection=model_pg_collection)],
                    run_pg_collection,
                    owner="owner",
                )
                is model_pg_collection
            )
            assert (
                training._get_model_pg_collection(
                    [SimpleNamespace(pg_collection=None)], run_pg_collection, owner="owner"
                )
                is run_pg_collection
            )
        assert _owners(record) == []

    def test_without_either_warns_with_the_owner(self, monkeypatch):
        monkeypatch.setattr(
            ProcessGroupCollection, "use_mpu_process_groups", classmethod(_raise_global_read)
        )
        _assert_warns_then_reads_global(
            "evaluate",
            lambda: training._get_model_pg_collection(
                [SimpleNamespace(pg_collection=None)], None, owner="evaluate"
            ),
        )


def test_dummy_train_step_without_a_collection_warns(monkeypatch, run_config):
    monkeypatch.setattr(training, "get_args", lambda: SimpleNamespace())
    monkeypatch.setattr(
        ProcessGroupCollection, "use_mpu_process_groups", classmethod(_raise_global_read)
    )
    _assert_warns_then_reads_global("dummy_train_step", lambda: training.dummy_train_step(iter([])))


def test_training_log_without_a_collection_warns(monkeypatch, run_config):
    monkeypatch.setattr(training, "get_args", lambda: SimpleNamespace())
    monkeypatch.setattr(training, "get_train_state", mock.Mock())
    monkeypatch.setattr(training, "get_timers", mock.Mock())
    for getter in (
        "get_tensorboard_writer",
        "get_wandb_writer",
        "get_one_logger",
        "get_energy_monitor",
    ):
        monkeypatch.setattr(training, getter, lambda: None)
    monkeypatch.setattr(
        ProcessGroupCollection, "use_mpu_process_groups", classmethod(_raise_global_read)
    )

    _assert_warns_then_reads_global(
        "training_log",
        lambda: training.training_log({}, {}, None, 1, 1.0, False, 0, None, None, None, None),
    )


@pytest.mark.parametrize("pass_collection", [True, False], ids=["collection", "omitted"])
def test_training_log_reports_memory_on_rank_0_of_its_data_parallel_group(
    monkeypatch, run_config, pass_collection
):
    # The run's collection gates the report on its dp group. Without a collection the report
    # keeps the rank of mpu.get_data_parallel_rank(), which spans the GTP weight-remat axis.
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args(["--micro-batch-size", "1"])
    args.data_parallel_size = args.world_size = args.gtp_weight_remat_size = 1
    run_config.logger.log_interval = run_config.logger.log_memory_interval = 1
    run_config.train.micro_batch_size = run_config.train.train_iters = 1
    timers = mock.Mock()
    timers.return_value.elapsed.return_value = 1.0
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_train_state", TrainState)
    monkeypatch.setattr(training, "get_timers", lambda: timers)
    for getter in (
        "get_tensorboard_writer",
        "get_wandb_writer",
        "get_one_logger",
        "get_energy_monitor",
        "get_telemetry",
    ):
        monkeypatch.setattr(training, getter, lambda: None)
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(training, "one_logger_utils", mock.Mock())
    monkeypatch.setattr(training, "num_floating_point_operations", lambda *a, **kw: 1)
    monkeypatch.setattr(training, "reduce_max_stat_across_model_parallel_group", lambda x, **kw: x)
    monkeypatch.setattr(training, "print_rank_last", mock.Mock())
    report_memory = mock.Mock()
    monkeypatch.setattr(training, "report_memory", report_memory)
    groups = {name: _Group() for name in ("mp", "pp", "dp", "dp_gtp_remat", "dp_cp_gtp_remat")}

    def use_mpu_process_groups(cls, required_pgs=None):
        return ProcessGroupCollection(**{name: groups[name] for name in required_pgs})

    monkeypatch.setattr(
        ProcessGroupCollection, "use_mpu_process_groups", classmethod(use_mpu_process_groups)
    )
    pg_collection = ProcessGroupCollection(**groups) if pass_collection else None

    with _record_fallbacks() as record:
        training.training_log(
            {}, {}, 0.01, 1, 1.0, False, 0, None, None, None, None, pg_collection=pg_collection
        )

    expected = groups["dp"] if pass_collection else groups["dp_gtp_remat"]
    assert report_memory.call_args.kwargs["process_group"] is expected
    assert _owners(record) == ([] if pass_collection else ["training_log"])


def test_get_model_without_a_collection_warns_and_builds_one(monkeypatch, run_config):
    args = SimpleNamespace(
        virtual_pipeline_model_parallel_size=None, init_model_with_meta_device=False
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "has_nvidia_modelopt", False)
    built = ProcessGroupCollection(pp=_Group(size=1))
    build = mock.Mock(return_value=built)
    monkeypatch.setattr(training, "_build_pg_collection_from_parallel_state", build)
    model_provider = mock.Mock(side_effect=_GlobalRead)

    _assert_warns_then_reads_global("get_model", lambda: training.get_model(model_provider))
    build.assert_called_once_with(args)
    assert model_provider.call_args.kwargs["pg_collection"] is built


def test_wrapping_without_a_collection_warns_before_sizing_the_layout(monkeypatch):
    monkeypatch.setattr(
        ProcessGroupCollection, "use_mpu_process_groups", classmethod(_raise_global_read)
    )
    ddp_config = SimpleNamespace(bucket_size=None, use_distributed_optimizer=True)

    _assert_warns_then_reads_global(
        "wrap_model_chunks_with_ddp",
        lambda: training.wrap_model_chunks_with_ddp([torch.nn.Linear(1, 1)], None, ddp_config),
    )


def test_sequence_statistics_without_a_collection_warn(monkeypatch):
    monkeypatch.setattr(training, "_seqlen_stats_in_iteration", torch.zeros(2, dtype=torch.float64))
    monkeypatch.setattr(training, "_seqlen_stats_active", True)
    monkeypatch.setattr(training, "_seqlen_stats_are_global", False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", mock.Mock())
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", _raise_global_read)

    _assert_warns_then_reads_global(
        "consume_seqlen_stats_in_iteration", training.consume_seqlen_stats_in_iteration
    )


def test_sequence_statistics_with_a_collection_do_not_warn(monkeypatch):
    monkeypatch.setattr(
        training, "_seqlen_stats_in_iteration", torch.tensor([8.0, 32.0], dtype=torch.float64)
    )
    monkeypatch.setattr(training, "_seqlen_stats_active", True)
    monkeypatch.setattr(training, "_seqlen_stats_are_global", False)
    monkeypatch.setattr(torch.distributed, "all_reduce", mock.Mock())
    pg_collection = ProcessGroupCollection(tp=_Group(2), cp=_Group(1), pp=_Group(2))

    with _record_fallbacks() as record:
        stats = training.consume_seqlen_stats_in_iteration(pg_collection=pg_collection)

    assert stats == (2.0, 8.0)
    assert _owners(record) == []


def test_batch_accounting_without_a_collection_warns(monkeypatch):
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state, "get_data_parallel_world_size", _raise_global_read)

    _assert_warns_then_reads_global(
        "train", lambda: training._get_data_parallel_world_size(None, SimpleNamespace())
    )


def test_full_dataset_probe_without_a_collection_warns(monkeypatch, run_config):
    provider = mock.Mock(return_value=(None, None, None), supports_train_full_dataset=True)
    monkeypatch.setattr(
        ProcessGroupCollection, "use_mpu_process_groups", classmethod(_raise_global_read)
    )

    _assert_warns_then_reads_global(
        "_get_train_full_dataset_sample_count",
        lambda: training._get_train_full_dataset_sample_count(SimpleNamespace(), provider),
    )


def test_data_loaders_without_a_collection_warn(monkeypatch, run_config):
    args = SimpleNamespace(phase_transition_iterations=None)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_train_state", TrainState)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", _raise_global_read)

    _assert_warns_then_reads_global(
        "build_train_valid_test_data_loaders",
        lambda: training.build_train_valid_test_data_loaders(lambda num_samples: None),
    )
