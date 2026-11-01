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
from megatron.training import arguments, gpu_sniff_test, initialize, training
from megatron.training.distillation import utils as distillation_utils
from megatron.training.logging import packed_sequence_stats
from megatron.training.state import TrainState
from megatron.training.utils import common_utils


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
                    SimpleNamespace(pg_collection=model_pg_collection),
                    run_pg_collection,
                    owner="owner",
                )
                is model_pg_collection
            )
            assert (
                training._get_model_pg_collection(
                    SimpleNamespace(pg_collection=None), run_pg_collection, owner="owner"
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
                SimpleNamespace(pg_collection=None), None, owner="evaluate"
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


@pytest.mark.parametrize(
    "owner, accessor, call",
    [
        (
            "reduce_max_stat_across_model_parallel_group",
            "get_model_parallel_group",
            lambda: common_utils.reduce_max_stat_across_model_parallel_group(1.0),
        ),
        (
            "logical_and_across_model_parallel_group",
            "get_model_parallel_group",
            lambda: common_utils.logical_and_across_model_parallel_group(True),
        ),
        (
            "average_losses_across_data_parallel_group",
            "get_data_parallel_group",
            lambda: common_utils.average_losses_across_data_parallel_group([torch.ones(1)]),
        ),
        (
            "calc_params_l2_norm",
            "get_gtp_weight_remat_rank",
            lambda: common_utils.calc_params_l2_norm([torch.nn.Linear(1, 1)]),
        ),
        (
            "set_jit_fusion_options",
            "get_tensor_model_parallel_world_size",
            lambda: initialize._warmup_jit_function(),
        ),
        (
            "_set_random_seed",
            "get_pipeline_model_parallel_rank",
            lambda: initialize._set_random_seed(1234),
        ),
        (
            "update_packed_sequence_stats",
            "is_pipeline_last_stage",
            lambda: packed_sequence_stats.update_packed_sequence_stats(
                torch.tensor([3]), torch.ones(3)
            ),
        ),
        (
            "_broadcast_without_pp",
            "get_tensor_and_data_parallel_group",
            lambda: distillation_utils._broadcast_without_pp(lambda: None),
        ),
    ],
    ids=[
        "reduce_max_stat",
        "logical_and",
        "average_losses",
        "params_norm",
        "jit_warmup",
        "random_seed",
        "packed_sequence_stats",
        "distillation_listing",
    ],
)
def test_utilities_warn_before_reading_the_global_grid(
    monkeypatch, run_config, owner, accessor, call
):
    monkeypatch.setattr(parallel_state, accessor, _raise_global_read)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *args: 2)
    args = SimpleNamespace(
        use_megatron_fsdp=False,
        bf16=False,
        fp16=False,
        use_te_activation_func=False,
        swiglu=False,
        bias_gelu_fusion=False,
        bias_swiglu_fusion=False,
        sequence_parallel=True,
        ffn_hidden_size=4,
        tensor_model_parallel_size=1,
        seq_length=4,
        context_parallel_size=1,
        hidden_size=4,
    )
    run_config.train.micro_batch_size = 1
    monkeypatch.setattr(common_utils, "get_args", lambda: args)
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    # The JIT warmup draws its tensors on the GPU; only their shapes matter here.
    monkeypatch.setattr(
        initialize.torch,
        "rand",
        lambda *size, **kwargs: torch.zeros(size[0] if len(size) == 1 else size),
    )

    _assert_warns_then_reads_global(owner, call)


@pytest.mark.parametrize(
    "owner, call",
    [
        ("run_gpu_sniff_test", lambda: gpu_sniff_test.run_gpu_sniff_test("tag")),
        (
            "LogitsSaverHooks",
            lambda: distillation_utils.resolve_distillation_process_groups(
                None, "LogitsSaverHooks"
            ),
        ),
    ],
    ids=["run_gpu_sniff_test", "distillation"],
)
def test_omitted_collection_warns_before_building_one(monkeypatch, owner, call):
    monkeypatch.setattr(
        ProcessGroupCollection, "use_mpu_process_groups", classmethod(_raise_global_read)
    )
    _assert_warns_then_reads_global(owner, call)


def test_report_memory_without_a_group_warns(monkeypatch, run_config):
    for name in (
        "memory_allocated",
        "max_memory_allocated",
        "memory_reserved",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(parallel_state, "get_data_parallel_rank", _raise_global_read)

    _assert_warns_then_reads_global("report_memory", lambda: common_utils.report_memory("x"))


def test_jit_warmup_takes_the_tensor_parallel_size_it_is_given(monkeypatch, run_config):
    args = SimpleNamespace(
        bf16=False,
        fp16=False,
        use_te_activation_func=False,
        swiglu=False,
        bias_gelu_fusion=False,
        sequence_parallel=True,
        ffn_hidden_size=4,
        tensor_model_parallel_size=1,
        seq_length=8,
        context_parallel_size=1,
        hidden_size=4,
    )
    run_config.train.micro_batch_size = 1
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", _raise_global_read)
    shapes = []

    def fake_rand(*size, **kwargs):
        shape = size[0] if len(size) == 1 else size
        shapes.append(shape)
        return torch.zeros(shape)

    monkeypatch.setattr(initialize.torch, "rand", fake_rand)
    monkeypatch.setattr(initialize, "bias_dropout_add_fused_train", mock.Mock())
    monkeypatch.setattr(initialize.torch.cuda, "empty_cache", mock.Mock())

    with _record_fallbacks() as record:
        initialize._warmup_jit_function(tp_size=2)

    # The sequence-parallel input of the fused bias+dropout+add is split over 2 ranks.
    assert (4, 1, 4) in shapes
    assert _owners(record) == []


def test_grid_initialized_before_initialize_megatron_is_returned_for_seeding(
    monkeypatch, run_config
):
    # The seeding reads its groups from the returned collection, so it does not warn.
    grid = ProcessGroupCollection()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *args: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *args: 4)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        ProcessGroupCollection, "use_mpu_process_groups", classmethod(lambda cls: grid)
    )
    monkeypatch.setattr(initialize, "get_args", lambda: SimpleNamespace())

    assert initialize._initialize_distributed(None, None, None) is grid
