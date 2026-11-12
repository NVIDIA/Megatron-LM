# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Data-order and CPU-worker contracts, independent of model size."""

import hashlib
import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.context_parallel.groups import get_valid_dynamic_context_parallel_group_sizes
from megatron.core.datasets import data_schedule, data_schedule_utils, dcp_runtime_worker
from megatron.core.datasets import dcp_schedule_prefetch as prefetch
from megatron.core.datasets.dcp_runtime_scheduler import models
from megatron.core.datasets.dcp_runtime_scheduler.models.base import PackLimits, QuadraticCostModel
from megatron.core.datasets.dcp_runtime_scheduler.models.benchmark import BenchmarkCostModel
from megatron.core.datasets.dcp_runtime_scheduler.runtime.config import (
    SCHEDULERS,
    allowed_keys,
    load_runtime_config,
    validate_runtime_config,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.lightweight import (
    LightweightRuntimePlanner,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import (
    RuntimePlanGeometry,
    RuntimeScheduleError,
    validate_rows,
)


class _ReplayIterator:
    def __init__(self, items):
        self.iterable = iter(items)
        self.replaying = False
        self.saved = []
        self.position = 0

    def __next__(self):
        if self.replaying:
            result = self.saved[self.position]
            self.position += 1
            return result
        result = next(self.iterable)
        self.saved.append(result)
        return result


def _prepare(iterator, count, group, include_original_seqlens=False):
    samples = [next(iterator) for _ in range(count)]
    physical = [((sample + 15) // 16) * 16 for sample in samples]
    return samples, list(enumerate(physical)), [], [], physical, samples


def _consume(owner, count):
    """Read ``count`` items as DpBalancedScheduler.run does and hand them to the runtime."""
    return owner.consume([next(owner.iterator) for _ in range(count)])


def _solve(lengths, padded, baseline_rows):
    return dict(
        rows=[[list(range(len(lengths)))] * 8],
        statistics={},
        predicted_ms=1,
        solve_start={},
        solve_end={},
        worker_pid=123,
        worker_cuda_initialized=False,
        worker_distributed_initialized=False,
    )


@pytest.fixture
def make_prefetcher(monkeypatch):
    original_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(prefetch, 'get_batch_and_global_seqlens', _prepare)
    monkeypatch.setattr(torch, 'tensor', cpu_tensor)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.distributed, 'broadcast_object_list', lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, 'get_global_rank', lambda *args: 0)
    owners = []

    def create(items, asynchronous=True):
        owner = prefetch.DcpSchedulePrefetcher.__new__(prefetch.DcpSchedulePrefetcher)
        owner.iterator = _ReplayIterator(items)
        owner.buffer = prefetch._BufferedIterator(owner.iterator.iterable)
        owner.iterator.iterable = owner.buffer
        owner.spec = {'geometry': {'dataset_alignment': 16}}
        owner.group = owner.dp_group = SimpleNamespace()
        owner.asynchronous = asynchronous
        owner.pending = None
        owner.sequence_number = 0
        owner.trace_path = None
        owner.executor = prefetch._DaemonExecutor('dcp-planner-io-test')
        owner.worker = SimpleNamespace(solve=_solve, stop=lambda: None, close=lambda: None)
        owner.baseline = lambda lengths: [[list(range(len(lengths)))] * 8]
        owners.append(owner)
        return owner

    yield create
    for owner in owners:
        owner.close()


def test_buffered_iterator_answers_for_its_source():
    """The training loop asks the iterable for more than items, e.g. a loader's save_state."""

    class Loader:
        is_save_state_rank_independent = True

        def __init__(self):
            self.values = iter(range(3))

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.values)

        def save_state(self):
            return {'position': 1}

    buffered = prefetch._BufferedIterator(Loader())
    assert buffered.peek(2) == [0, 1]
    assert buffered.save_state() == {'position': 1}
    assert buffered.is_save_state_rank_independent is True
    assert not hasattr(buffered, 'load_state')
    assert [next(buffered), next(buffered), next(buffered)] == [0, 1, 2]


def test_lookahead_does_not_consume_rerun_data(make_prefetcher):
    owner = make_prefetcher([1, 2, 3, 4, 5, 6])
    assert _consume(owner, 2)[0][0] == [1, 2]
    owner.prefetch(2)
    assert owner.iterator.saved == [1, 2]
    assert len(owner.buffer.items) == 2
    assert _consume(owner, 2)[0][0] == [3, 4]
    owner.prefetch(2)
    assert _consume(owner, 2)[0][0] == [5, 6]
    assert owner.iterator.saved == [1, 2, 3, 4, 5, 6]


def test_trace_record_names_the_scheduler_type(make_prefetcher):
    owner = make_prefetcher([1, 2])
    owner.spec['type'] = 'dcp_benchmark'
    _consume(owner, 2)
    assert owner.last_telemetry['scheduler_type'] == 'dcp_benchmark'
    assert 'solver' not in owner.last_telemetry and 'baseline' not in owner.last_telemetry


@pytest.mark.parametrize('packed_count', [1, 2])
def test_scheduler_prefetches_raw_count_after_packing(make_prefetcher, monkeypatch, packed_count):
    raw_count = 4
    owner = make_prefetcher(list(range(1, 2 * raw_count + 1)))

    def prepare(iterator, count, group, include_original_seqlens=False):
        samples, pairs, ids, offsets, padded, lengths = _prepare(iterator, count, group)
        batch = [{'padded_seq_len': physical} for physical in padded]
        return batch, pairs, ids, offsets, padded, lengths

    calls = []

    def solve(lengths, padded, baseline_rows):
        calls.append(list(lengths))
        result = _solve(lengths, padded, baseline_rows)
        width = len(lengths) // packed_count
        result['rows'] = [
            [list(range(start, start + width))] * 8 for start in range(0, len(lengths), width)
        ]
        return result

    monkeypatch.setattr(prefetch, 'get_batch_and_global_seqlens', prepare)
    monkeypatch.setattr(owner.worker, 'solve', solve)
    monkeypatch.setattr(data_schedule, 'mtp_on_this_pipeline_rank', lambda *args, **kwargs: False)
    monkeypatch.setattr(data_schedule, 'reroute_samples_to_dcp_ranks', lambda *args, **kwargs: [])
    monkeypatch.setattr(data_schedule, 'build_packed_microbatches', lambda *args, **kwargs: [])
    monkeypatch.setattr(data_schedule, 'broadcast_scalars', lambda values, group, device: values)
    monkeypatch.setattr(data_schedule, 'create_data_iterator', lambda *args, **kwargs: iter([]))
    monkeypatch.setattr(
        data_schedule, 'get_global_real_seqlen_stats', lambda batch, group, device: (0.0, 0.0)
    )
    scheduler = data_schedule.DpBalancedScheduler.__new__(data_schedule.DpBalancedScheduler)
    scheduler.runtime = owner
    scheduler.prefetch_next = True
    scheduler.is_dynamic_cp = True
    scheduler.get_required_sample_keys = lambda: ['padded_seq_len']
    single = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    parent = SimpleNamespace(rank=lambda: 0, size=lambda: 8)
    config = SimpleNamespace(virtual_pipeline_model_parallel_size=None)

    _, actual_packed_count, _, _ = scheduler.run(
        owner.iterator, raw_count, single, single, single, parent, torch.device('cpu'), config
    )
    assert actual_packed_count == packed_count
    assert owner.pending['count'] == raw_count
    owner.pending['future'].result()
    assert calls == [list(range(1, 5)), list(range(5, 9))]
    prepared, _ = _consume(owner, raw_count)
    assert len(prepared[0]) == raw_count
    assert calls == [list(range(1, 5)), list(range(5, 9))]
    assert owner.last_telemetry['ready_when_consumed']
    assert owner.iterator.saved == list(range(1, 9))


@pytest.mark.parametrize('next_count', [1, 3])
def test_batch_size_change_keeps_every_sample(make_prefetcher, next_count):
    owner = make_prefetcher(list(range(1, 10)))
    _consume(owner, 2)
    owner.prefetch(2)
    assert _consume(owner, next_count)[0][0] == list(range(3, 3 + next_count))
    assert next(owner.iterator) == 3 + next_count


def test_replay_reads_current_batch_not_prefetched_batch(make_prefetcher):
    owner = make_prefetcher([1, 2, 3, 4])
    _consume(owner, 2)
    owner.prefetch(2)
    owner.iterator.replaying = True
    assert _consume(owner, 2)[0][0] == [1, 2]
    owner.iterator.replaying = False
    assert _consume(owner, 2)[0][0] == [3, 4]


def test_sync_control_does_not_read_ahead(make_prefetcher):
    owner = make_prefetcher([1, 2, 3, 4], asynchronous=False)
    _consume(owner, 2)
    owner.prefetch(2)
    assert owner.pending is None and not owner.buffer.items
    assert _consume(owner, 2)[0][0] == [3, 4]


def test_short_tail_is_retained(make_prefetcher):
    owner = make_prefetcher([1, 2, 3])
    _consume(owner, 2)
    owner.prefetch(2)
    assert owner.pending is None
    assert _consume(owner, 1)[0][0] == [3]


def test_worker_error_is_broadcast_as_an_error(make_prefetcher, monkeypatch):
    def fail(*args):
        raise ValueError('bad cost model')

    owner = make_prefetcher([1, 2])
    monkeypatch.setattr(owner.worker, 'solve', fail)
    with pytest.raises(RuntimeError, match='bad cost model'):
        _consume(owner, 2)


@pytest.mark.parametrize('length,build_baseline', [(192, True), (200, False)])
def test_cp_cap_capacity_failure_reaches_worker_before_default_baseline(
    make_prefetcher, monkeypatch, length, build_baseline
):
    owner = make_prefetcher([length])
    owner.spec['geometry'].update(tokens_per_rank=100, allowed_cp_sizes=(1, 2), max_cp_size=2)
    baseline_calls, received = [], []
    baseline_rows = [[[0]] * 8]

    def baseline(lengths):
        baseline_calls.append(lengths)
        return baseline_rows

    def solve(logical, padded, incumbent):
        received.append((logical, padded, incumbent))
        return _solve(logical, padded, incumbent)

    owner.baseline = baseline
    monkeypatch.setattr(owner.worker, 'solve', solve)
    prepared = _prepare(owner.iterator, 1, None)
    owner._submit(prepared, 1)['future'].result()
    assert bool(baseline_calls) is build_baseline
    assert received == [([length], prepared[4], baseline_rows if build_baseline else None)]


def test_batch_normalization_does_not_mutate_raw_snapshot(monkeypatch):
    raw = {'tokens': torch.ones(1, 16), 'padded_seq_len': torch.tensor([[16]])}
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 'cpu')
    monkeypatch.setattr(
        data_schedule_utils,
        '_get_global_seqlens_and_ids',
        lambda lengths, group: ([(0, 16)], [], [], lengths.tolist()),
    )
    prepared = data_schedule_utils.get_batch_and_global_seqlens(iter([raw]), 1, None, True)
    assert raw['tokens'].ndim == 2 and 'original_seq_len' not in raw
    assert prepared[0][0]['tokens'].ndim == 1
    assert prepared[-1] == [16]


_NON_ATTENTION = {'forward_ms_per_layer': 0.5, 'backward_ms_per_layer': 1.0}
_BENCHMARK = {'type': 'dcp_benchmark', 'hardware': 'h100', 'non_attention': _NON_ATTENTION}
_DYNAMIC_USP = {'type': 'dynamic_usp'}


def _runtime_config(tmp_path, payload):
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps(payload))
    return path


def _runtime_training_config(path, **overrides):
    values = dict(
        dynamic_cp_runtime_config=str(path),
        max_seqlen_per_dp_cp_rank=4096,
        min_dynamic_context_parallel_size=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _gqa_training_config(path, **overrides):
    """Ordinary-GQA training configuration inside the packaged h100 predictor's domain."""
    values = dict(
        num_attention_heads=32,
        num_query_groups=4,
        kv_channels=128,
        num_layers=48,
        thd_tail_padding_policy='extend_last',
        pad_packed_seq_alignment='max',
        bf16=True,
        attention_backend='fused',
        cp_comm_type='p2p',
    )
    values.update(overrides)
    return _runtime_training_config(path, **values)


@pytest.fixture
def geometry_only_model(monkeypatch):
    """The one Python-only seam: every scheduler type gets a model that only needs geometry.

    Geometry, caching and plumbing tests use it. Worker subprocess tests use the real models.
    """

    class GeometryOnlyModel(QuadraticCostModel):
        non_attention_layer_types = ('gqa',)
        # What a cost model class declares: the largest CP group it prices.
        largest_cp_size = 64

    monkeypatch.setattr(models, 'model_class', lambda name: GeometryOnlyModel)
    return GeometryOnlyModel


def _wrap_environment(monkeypatch, world=8, pp=1):
    """Process groups and collectives for one CPU process acting as the DPxCP leader."""
    single = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    pipeline = SimpleNamespace(rank=lambda: 0, size=lambda: pp)
    parent = SimpleNamespace(rank=lambda: 0, size=lambda: world)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.distributed, 'broadcast_object_list', lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, 'get_global_rank', lambda *args: 0)
    # The all_reduce that counts ranks whose runtime file differs from the leader's.
    monkeypatch.setattr(prefetch, '_count_over_group', lambda flag, group: int(bool(flag)))
    return SimpleNamespace(
        dp=single,
        tp=single,
        pp=pipeline,
        dp_cp=parent,
        # No GTP_remat peers; one dynamic CP group per size this DPxCP domain can form.
        dp_gtp_remat=None,
        dp_cp_gtp_remat=None,
        dynamic_cp=dict.fromkeys(get_valid_dynamic_context_parallel_group_sizes(world)),
    )


# What wrap_data_iterator itself reads from the training configuration.
_WRAP = dict(
    sequence_parallel=False,
    sequence_packing_scheduler='default_dynamic_cp',
    virtual_pipeline_model_parallel_size=None,
)


def _wrap_config(path, **overrides):
    values = dict(_WRAP, max_seqlen_per_dp_cp_rank=1024)
    values.update(overrides)
    return _runtime_training_config(path, **values)


def _record_outcomes(monkeypatch):
    sent = []
    monkeypatch.setattr(
        torch.distributed,
        'broadcast_object_list',
        lambda payload, **kwargs: sent.append(payload[0]),
    )
    return sent


def _count_file_reads(monkeypatch):
    reads = []
    load = prefetch.load_runtime_config
    monkeypatch.setattr(
        prefetch, 'load_runtime_config', lambda path: reads.append(path) or load(path)
    )
    return reads


# --------------------------------------------------------------------------------------
# The CPU worker process, driven by a specification built from the runtime JSON
# --------------------------------------------------------------------------------------
def test_planner_process_stays_cpu_only(tmp_path):
    from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler

    config = _gqa_training_config(_runtime_config(tmp_path, _BENCHMARK))
    spec = prefetch.runtime_spec(config, 8)
    baseline = DefaultDynamicCPScheduler(4096, 8, 1, None, max_num_seqs=64, min_cp_size=1)
    worker = prefetch._CpuPlanner(spec)
    try:
        result = worker.solve(
            [10400, 18000],
            [10400, 18000],
            baseline.get_groups_and_subsamples([(0, 10400), (1, 18000)]),
        )
        again = worker.solve(
            [4000, 24000, 2000],
            [4000, 24000, 2000],
            baseline.get_groups_and_subsamples([(0, 4000), (1, 24000), (2, 2000)]),
        )
    finally:
        worker.close()
    assert len(result['rows'][0]) == 8
    assert result['model_metadata']['name'] == 'dcp_benchmark'
    # Every reply says which NVLink domain size the cost model took from the hardware.
    assert result['model_metadata']['hardware'] == 'h100'
    assert result['model_metadata']['nvlink_domain_size'] == 8
    assert 'LightweightRuntimePlanner.plan_padded' in result['solver_entry']
    assert result['predicted_ms'] > 0 and again['predicted_ms'] > 0
    assert result['worker_pid'] == again['worker_pid']
    assert not result['worker_cuda_initialized'] and not again['worker_cuda_initialized']
    assert (
        not result['worker_distributed_initialized'] and not again['worker_distributed_initialized']
    )


def test_worker_reports_initialization_error_after_process_exit(tmp_path):
    spec = prefetch.runtime_spec(_gqa_training_config(_runtime_config(tmp_path, _BENCHMARK)), 8)
    # Start-up validation built this model in-process; break only what the worker receives.
    worker = prefetch._CpuPlanner(dict(spec, model='not_a_model'))
    try:
        worker.process.wait(timeout=30)
        with pytest.raises(RuntimeError, match='Unknown in-tree DCP cost model: not_a_model'):
            worker.solve([1000], [1008], [[[0]] * 8])
    finally:
        worker.close()


def test_solver_options_are_a_python_only_knob_of_the_specification(tmp_path):
    path = _runtime_config(tmp_path, _BENCHMARK)
    spec = prefetch.runtime_spec(_gqa_training_config(path), 8)
    assert spec['solver_options'] == {}
    spec['solver_options'] = {'max_scored_candidates': 2, 'time_budget_s': 0.25}
    planner = dcp_runtime_worker._initialize_planner(spec)
    assert (planner.options.max_scored_candidates, planner.options.time_budget_s) == (2, 0.25)
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['solver_options'\]"):
        load_runtime_config(
            str(_runtime_config(tmp_path, dict(_BENCHMARK, solver_options={'time_budget_s': 1})))
        )


# --------------------------------------------------------------------------------------
# Geometry derived from the training configuration
# --------------------------------------------------------------------------------------
def test_runtime_respects_real_sequence_cap_after_dummy_tail(tmp_path, geometry_only_model):
    config = _runtime_training_config(
        _runtime_config(tmp_path, _BENCHMARK), thd_max_packed_sequences=4
    )
    assert prefetch.runtime_spec(config, 8, max_num_seqs=3)['geometry']['max_group_seqs'] == 3


@pytest.mark.parametrize(
    'world,minimum,allowed', [(8, 1, (1, 2, 4, 8)), (32, 8, (8, 16, 32)), (16, 2, (2, 4, 8, 16))]
)
def test_runtime_spec_matches_existing_classic_cp_groups(
    tmp_path, geometry_only_model, world, minimum, allowed
):
    config = _runtime_training_config(
        _runtime_config(tmp_path, _BENCHMARK), min_dynamic_context_parallel_size=minimum
    )
    spec = prefetch.runtime_spec(config, world)
    assert spec['geometry']['allowed_cp_sizes'] == allowed
    assert spec['geometry']['align_group_starts']


def test_runtime_spec_rejects_unsupported_classic_minimum(tmp_path, geometry_only_model):
    config = _runtime_training_config(
        _runtime_config(tmp_path, _BENCHMARK), min_dynamic_context_parallel_size=3
    )
    with pytest.raises(ValueError, match='power-of-two minimum'):
        prefetch.runtime_spec(config, 8)


def test_minimum_cp_above_the_cost_model_ceiling_names_the_ceiling(tmp_path, geometry_only_model):
    config = _runtime_training_config(
        _runtime_config(tmp_path, _BENCHMARK), min_dynamic_context_parallel_size=128
    )
    with pytest.raises(ValueError, match='prices CP groups of at most 64 ranks') as caught:
        prefetch.runtime_spec(config, 128)
    assert 'min_dynamic_context_parallel_size=128' in str(caught.value)


@pytest.mark.parametrize(
    'world,minimum,maximum,formable',
    [
        (8, 1, 3, [1, 2, 4, 8]),
        (8, 1, 6, [1, 2, 4, 8]),
        (8, 1, 16, [1, 2, 4, 8]),
        (8, 2, 6, [2, 4, 8]),
        # A non-power-of-two DPxCP cannot use the full group, and 12 or 6 do not stand for 4.
        (12, 1, 12, [1, 2, 4]),
        (12, 1, 6, [1, 2, 4]),
    ],
)
def test_max_cp_size_must_be_a_cp_size_the_run_can_form(
    tmp_path, geometry_only_model, world, minimum, maximum, formable
):
    path = _runtime_config(tmp_path, dict(_BENCHMARK, max_cp_size=maximum))
    config = _runtime_training_config(path, min_dynamic_context_parallel_size=minimum)
    with pytest.raises(ValueError) as caught:
        prefetch.runtime_spec(config, world)
    message = str(caught.value)
    assert f'max_cp_size={maximum} is not a CP group size this run can form' in message
    assert f'DPxCP={world} and min_dynamic_context_parallel_size={minimum}' in message
    assert f'forms groups of {formable} ranks' in message
    assert str(path.resolve()) in message
    # Each listed size is accepted.
    for size in formable:
        path.write_text(json.dumps(dict(_BENCHMARK, max_cp_size=size)))
        fresh = _runtime_training_config(path, min_dynamic_context_parallel_size=minimum)
        allowed = prefetch.runtime_spec(fresh, world)['geometry']['allowed_cp_sizes']
        assert list(allowed) == [cp for cp in formable if cp <= size]


@pytest.mark.parametrize(
    'maximum,allowed', [(1, (1,)), (2, (1, 2)), (4, (1, 2, 4)), (8, (1, 2, 4, 8))]
)
def test_runtime_config_caps_cp_candidates_without_becoming_a_model_option(
    tmp_path, geometry_only_model, maximum, allowed
):
    path = _runtime_config(tmp_path, dict(_BENCHMARK, max_cp_size=maximum))
    settings = load_runtime_config(str(path))
    assert settings['max_cp_size'] == maximum
    assert 'max_cp_size' not in settings['options']
    spec = prefetch.runtime_spec(_runtime_training_config(path), 8)
    assert spec['geometry']['allowed_cp_sizes'] == allowed
    assert spec['geometry']['max_cp_size'] == maximum
    assert 'max_cp_size' not in spec['model_options']


def test_runtime_config_rejects_a_cp_cap_below_the_training_minimum(tmp_path, geometry_only_model):
    path = _runtime_config(tmp_path, dict(_BENCHMARK, max_cp_size=2))
    config = _runtime_training_config(path, min_dynamic_context_parallel_size=4)
    with pytest.raises(ValueError, match='max_cp_size'):
        prefetch.runtime_spec(config, 8)


# --------------------------------------------------------------------------------------
# One argument, one file: schema of the runtime JSON
# --------------------------------------------------------------------------------------
def test_runtime_scheduler_adds_exactly_one_argument(capsys):
    import argparse
    import dataclasses

    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.training.arguments import add_megatron_arguments

    fields = [field.name for field in dataclasses.fields(ModelParallelConfig)]
    assert [name for name in fields if name.startswith('dynamic_cp_runtime')] == [
        'dynamic_cp_runtime_config'
    ]
    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    flags = [flag for action in parser._actions for flag in action.option_strings]
    assert [flag for flag in flags if 'dynamic-cp-runtime' in flag] == [
        '--dynamic-cp-runtime-config'
    ]
    assert parser.parse_args([]).dynamic_cp_runtime_config is None
    args = parser.parse_args(['--dynamic-cp-runtime-config', '/tmp/runtime.json'])
    assert args.dynamic_cp_runtime_config == '/tmp/runtime.json'
    for removed in ('--no-dynamic-cp-runtime-async', '--dynamic-cp-runtime-force-powers'):
        with pytest.raises(SystemExit):
            parser.parse_args([removed])
    capsys.readouterr()


def _validate_command_line(monkeypatch, *arguments):
    import sys

    from megatron.training.arguments import parse_args, validate_args

    model = '--num-layers 2 --hidden-size 64 --num-attention-heads 4 --seq-length 64'
    batch = '--max-position-embeddings 64 --micro-batch-size 1 --global-batch-size 8'
    monkeypatch.setattr(sys, 'argv', ['pretrain_gpt.py', *model.split(), *batch.split()])
    sys.argv.extend(arguments)
    monkeypatch.setenv('WORLD_SIZE', '1')
    return validate_args(parse_args(ignore_unknown_args=True))


_DYNAMIC_CP_ARGUMENTS = (
    '--dynamic-context-parallel',
    '--calculate-per-token-loss',
    '--max-seqlen-per-dp-cp-rank',
    '1024',
)


def test_command_line_validation_checks_the_runtime_file_before_the_model_is_built(
    tmp_path, monkeypatch, capsys
):
    path = _runtime_config(tmp_path, dict(_DYNAMIC_USP, solver='dynamic_usp'))
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['solver'\]"):
        _validate_command_line(
            monkeypatch, *_DYNAMIC_CP_ARGUMENTS, '--dynamic-cp-runtime-config', str(path)
        )
    # Only the schema is checked here; the model and parallel configuration follow later.
    path.write_text(json.dumps(_DYNAMIC_USP))
    args = _validate_command_line(
        monkeypatch, *_DYNAMIC_CP_ARGUMENTS, '--dynamic-cp-runtime-config', str(path)
    )
    assert args.dynamic_cp_runtime_config == str(path)
    capsys.readouterr()


def test_command_line_validation_requires_dynamic_cp_for_the_runtime_file(
    tmp_path, monkeypatch, capsys
):
    path = _runtime_config(tmp_path, _DYNAMIC_USP)
    with pytest.raises(ValueError, match='requires --dynamic-context-parallel'):
        _validate_command_line(monkeypatch, '--dynamic-cp-runtime-config', str(path))
    capsys.readouterr()


def test_command_line_warning_leaves_the_cp_sizes_to_the_runtime_file(
    tmp_path, monkeypatch, capsys
):
    path = _runtime_config(tmp_path, _DYNAMIC_USP)
    heuristic_range = 'will be used as the maximum dynamic CP group size'
    with pytest.warns(UserWarning) as caught:
        _validate_command_line(
            monkeypatch, *_DYNAMIC_CP_ARGUMENTS, '--dynamic-cp-runtime-config', str(path)
        )
    messages = [str(warning.message) for warning in caught]
    assert any(
        f'Dynamic CP enabled with the runtime scheduler file {path}: the CP group sizes it '
        'forms are checked against the model and printed once the model is built.' in message
        for message in messages
    )
    assert not any(heuristic_range in message for message in messages)
    # Without the file the default heuristic's range is still announced.
    with pytest.warns(UserWarning) as caught:
        _validate_command_line(monkeypatch, *_DYNAMIC_CP_ARGUMENTS)
    assert any(heuristic_range in str(warning.message) for warning in caught)
    capsys.readouterr()


def test_runtime_config_requires_dynamic_context_parallel():
    from megatron.core.model_parallel_config import ModelParallelConfig

    with pytest.raises(ValueError, match='requires dynamic context parallelism'):
        ModelParallelConfig(dynamic_cp_runtime_config='/nonexistent/runtime.json')
    # Building the configuration object does not open the file.
    config = ModelParallelConfig(
        dynamic_cp_runtime_config='/nonexistent/runtime.json',
        dynamic_context_parallel=True,
        max_seqlen_per_dp_cp_rank=1024,
    )
    assert config.sequence_packing_scheduler == 'default_dynamic_cp'


def test_runtime_config_defaults_and_tracks_file_contents(tmp_path):
    path = _runtime_config(tmp_path, _BENCHMARK)
    assert load_runtime_config(str(path)) == {
        'type': 'dcp_benchmark',
        'options': {
            'hardware': 'h100',
            'non_attention': dict(_NON_ATTENTION, source=str(path.resolve())),
        },
        'max_cp_size': None,
        'async': True,
        'trace_dir': None,
        'config_path': str(path.resolve()),
        'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    # The same schema check is available without a file.
    assert validate_runtime_config(dict(_DYNAMIC_USP), tmp_path / 'memory.json') == {
        'type': 'dynamic_usp',
        'options': {},
        'max_cp_size': None,
        'async': True,
        'trace_dir': None,
    }


@pytest.mark.parametrize(
    'payload,solver,planner',
    [
        (_BENCHMARK, 'lightweight', 'LightweightRuntimePlanner'),
        (_DYNAMIC_USP, 'dynamic_usp', 'DynamicUSPRuntimePlanner'),
    ],
)
def test_each_scheduler_type_binds_one_cost_model_family_to_one_solver(
    tmp_path, geometry_only_model, payload, solver, planner
):
    assert sorted(SCHEDULERS) == ['dcp_benchmark', 'dynamic_usp']
    assert SCHEDULERS[payload['type']]['solver'] == solver
    config = _runtime_training_config(
        _runtime_config(tmp_path, payload), min_dynamic_context_parallel_size=8
    )
    spec = prefetch.runtime_spec(config, 8)
    assert spec['type'] == payload['type'] and 'solver' not in spec
    assert spec['model'] == ('usp_calib_v1' if payload is _DYNAMIC_USP else 'dcp_benchmark')
    assert type(dcp_runtime_worker._initialize_planner(spec)).__name__ == planner
    with pytest.raises(ValueError, match='Unknown scheduler type'):
        dcp_runtime_worker._initialize_planner(dict(spec, type='usp_calib_v1'))


@pytest.mark.parametrize(
    'scheduler_type',
    [
        'usp_calib_v1',
        'calibrated_attention',
        'qwen35_full',
        'qwen35_gb200',
        'lightweight',
        'extension.models:create',
        'Dcp_Benchmark',
        ' dcp_benchmark',
        'dcp_benchmrk',
    ],
)
def test_runtime_config_type_is_a_closed_set(tmp_path, scheduler_type):
    path = _runtime_config(tmp_path, dict(_BENCHMARK, type=scheduler_type))
    with pytest.raises(ValueError) as caught:
        load_runtime_config(str(path))
    assert "'type' must be one of ['dcp_benchmark', 'dynamic_usp']" in str(caught.value)
    assert repr(scheduler_type) in str(caught.value)


@pytest.mark.parametrize(
    'payload',
    [[], None, True, 1, 'model', {}, {'type': None}, {'type': []}, {'type': 1}, {'type': ''}],
)
def test_runtime_config_rejects_missing_or_invalid_model_type(tmp_path, payload):
    path = _runtime_config(tmp_path, payload)
    with pytest.raises(ValueError):
        load_runtime_config(str(path))


@pytest.mark.parametrize('payload', [_BENCHMARK, _DYNAMIC_USP])
@pytest.mark.parametrize(
    'key,value',
    [
        ('solver', 'lightweight'),
        ('solver', 'dynamic_usp'),
        ('solver_options', {}),
        ('baseline', 'module:factory'),
        ('force_powers', False),
        ('profile', 'legacy_frozen_20260922'),
        ('source', 'measured last week'),
        ('allow_provisional', True),
        ('recompute_forward', True),
        ('layer_types', ['gqa']),
        ('query_heads', 32),
        ('legacy_model_family', 'registry'),
        # NVLink topology follows from "hardware"; no key states a domain size.
        ('legacy_domain_size', 8),
        ('nvlink_domain_size', 8),
        ('domain_size', 72),
        ('first_forward_dp_cp_sync', True),
        ('model', 'dcp_benchmark'),
        ('model_options', {}),
    ],
)
def test_runtime_config_rejects_removed_and_internal_keys(tmp_path, payload, key, value):
    path = _runtime_config(tmp_path, dict(payload, **{key: value}))
    with pytest.raises(ValueError, match=rf"unknown key\(s\) \['{key}'\]"):
        load_runtime_config(str(path))


def test_unknown_keys_list_the_keys_the_selected_type_accepts(tmp_path):
    path = _runtime_config(tmp_path, dict(_BENCHMARK, max_cp=4, calibration='table.json'))
    with pytest.raises(ValueError) as caught:
        load_runtime_config(str(path))
    assert "unknown key(s) ['calibration', 'max_cp'] for type 'dcp_benchmark'" in str(caught.value)
    assert (
        "allowed keys: ['type', 'hardware', 'non_attention', 'max_cp_size', 'async', 'trace_dir']"
    ) in str(caught.value)
    path = _runtime_config(tmp_path, dict(_DYNAMIC_USP, non_attention=_NON_ATTENTION))
    with pytest.raises(ValueError) as caught:
        load_runtime_config(str(path))
    assert "unknown key(s) ['non_attention'] for type 'dynamic_usp'" in str(caught.value)
    assert f"allowed keys: {list(allowed_keys('dynamic_usp'))}" in str(caught.value)
    assert 'calibration' in allowed_keys('dynamic_usp')
    assert 'non_attention' not in allowed_keys('dynamic_usp')


@pytest.mark.parametrize('missing', ['hardware', 'non_attention'])
def test_missing_required_keys_are_named(tmp_path, missing):
    payload = {key: value for key, value in _BENCHMARK.items() if key != missing}
    path = _runtime_config(tmp_path, payload)
    with pytest.raises(
        ValueError, match=rf"type 'dcp_benchmark' requires key\(s\) \['{missing}'\]"
    ):
        load_runtime_config(str(path))


@pytest.mark.parametrize(
    'non_attention,fragment',
    [
        ('1.0', 'non_attention must be an object'),
        ([1.0, 2.0], 'non_attention must be an object'),
        ({'forward_ms_per_layer': 1.0}, "missing: ['backward_ms_per_layer']"),
        ({}, "missing: ['forward_ms_per_layer', 'backward_ms_per_layer']"),
        (dict(_NON_ATTENTION, source='mine'), "unknown: ['source']"),
        (dict(_NON_ATTENTION, microbatch_ms=1.0), "unknown: ['microbatch_ms']"),
        (dict(_NON_ATTENTION, iteration_tail_ms=1.0), "unknown: ['iteration_tail_ms']"),
        (dict(_NON_ATTENTION, forward_ms_per_layer=None), 'forward_ms_per_layer must be a finite'),
        (dict(_NON_ATTENTION, forward_ms_per_layer='1'), 'forward_ms_per_layer must be a finite'),
        (dict(_NON_ATTENTION, forward_ms_per_layer=True), 'forward_ms_per_layer must be a finite'),
        (dict(_NON_ATTENTION, backward_ms_per_layer=-1), 'backward_ms_per_layer must be a finite'),
        (dict(_NON_ATTENTION, backward_ms_per_layer=[1]), 'backward_ms_per_layer must be a finite'),
        # A huge integer is a ValueError like any other value outside the domain.
        (dict(_NON_ATTENTION, backward_ms_per_layer=10**400), 'backward_ms_per_layer must be a'),
        (dict(_NON_ATTENTION, forward_ms_per_layer={}), 'nonempty layer-type mapping'),
        (
            dict(_NON_ATTENTION, forward_ms_per_layer={'gqa': '1'}),
            'forward_ms_per_layer.gqa must be a finite',
        ),
    ],
)
def test_non_attention_inputs_have_exactly_two_validated_keys(tmp_path, non_attention, fragment):
    path = _runtime_config(tmp_path, dict(_BENCHMARK, non_attention=non_attention))
    with pytest.raises(ValueError) as caught:
        load_runtime_config(str(path))
    assert fragment in str(caught.value)


@pytest.mark.parametrize('maximum', [None, False, True, 0, -1, 2.0, '2'])
def test_runtime_config_rejects_invalid_maximum_cp_size(tmp_path, maximum):
    path = _runtime_config(tmp_path, dict(_DYNAMIC_USP, max_cp_size=maximum))
    with pytest.raises(ValueError, match='max_cp_size'):
        load_runtime_config(str(path))


@pytest.mark.parametrize(
    'field,value',
    [
        ('async', 1),
        ('async', 'false'),
        ('async', None),
        ('trace_dir', []),
        ('trace_dir', False),
        ('trace_dir', ''),
        ('calibration', 3),
        ('calibration', ''),
    ],
)
def test_runtime_config_rejects_invalid_control_types(tmp_path, field, value):
    path = _runtime_config(tmp_path, dict(_DYNAMIC_USP, **{field: value}))
    with pytest.raises(ValueError, match=field):
        load_runtime_config(str(path))


@pytest.mark.parametrize('hardware', [5, None, '', ['h100']])
def test_runtime_config_rejects_a_hardware_that_is_not_a_name(tmp_path, hardware):
    path = _runtime_config(tmp_path, dict(_BENCHMARK, hardware=hardware))
    with pytest.raises(ValueError, match='hardware must be a nonempty string'):
        load_runtime_config(str(path))


@pytest.mark.parametrize(
    'contents',
    [
        '{"type":"dynamic_usp","type":"dcp_benchmark"}',
        '{"type":"dcp_benchmark","hardware":"h100","non_attention":'
        '{"forward_ms_per_layer":1,"forward_ms_per_layer":2,"backward_ms_per_layer":1}}',
        '{"type":"dcp_benchmark","hardware":"h100","non_attention":'
        '{"forward_ms_per_layer":NaN,"backward_ms_per_layer":1}}',
        '{"type":"dcp_benchmark","hardware":"h100","non_attention":'
        '{"forward_ms_per_layer":Infinity,"backward_ms_per_layer":1}}',
        '{"type":"dcp_benchmark","hardware":"h100","non_attention":'
        '{"forward_ms_per_layer":1,"backward_ms_per_layer":-Infinity}}',
        '{"type":"dynamic_usp"',
    ],
)
def test_runtime_config_rejects_ambiguous_or_non_finite_json(tmp_path, contents):
    path = tmp_path / 'runtime.json'
    path.write_text(contents)
    with pytest.raises(ValueError, match='Cannot read runtime configuration'):
        load_runtime_config(str(path))


def test_runtime_spec_carries_execution_controls_and_no_solver_choice(
    tmp_path, geometry_only_model
):
    trace_dir = str(tmp_path / 'trace')
    path = _runtime_config(
        tmp_path, dict(_DYNAMIC_USP, **{'async': False, 'trace_dir': trace_dir, 'max_cp_size': 8})
    )
    spec = prefetch.runtime_spec(
        _runtime_training_config(path, min_dynamic_context_parallel_size=8), 8
    )
    assert spec['async'] is False and spec['trace_dir'] == trace_dir
    assert spec['geometry']['max_cp_size'] == 8
    # The file names no hardware, and the geometry-only model of this test has no NVLink domain.
    assert spec['hardware'] is None and spec['nvlink_domain_size'] is None
    assert set(spec) == {
        'type',
        'geometry',
        'model',
        'model_options',
        'solver_options',
        'hardware',
        'nvlink_domain_size',
        'async',
        'trace_dir',
        'config_path',
        'config_sha256',
    }


# --------------------------------------------------------------------------------------
# Binding the file to the training configuration
# --------------------------------------------------------------------------------------
def test_benchmark_runtime_config_inherits_training_geometry_and_records_source(
    tmp_path, monkeypatch
):
    checked = []
    check_domain = BenchmarkCostModel.check_domain

    def recording_check(self, geometry, config=None):
        checked.append(config)
        return check_domain(self, geometry, config)

    monkeypatch.setattr(BenchmarkCostModel, 'check_domain', recording_check)
    measurements = {'forward_ms_per_layer': {'gqa': 0.5}, 'backward_ms_per_layer': {'gqa': 1.0}}
    path = _runtime_config(tmp_path, dict(_BENCHMARK, non_attention=measurements))
    config = _gqa_training_config(path, num_layers=4, expert_model_parallel_size=4)
    spec = prefetch.runtime_spec(config, 8)
    options = spec['model_options']
    assert spec['model'] == 'dcp_benchmark'
    assert spec['geometry']['ep_size'] == 4
    assert spec['geometry']['allowed_cp_sizes'] == (1, 2, 4, 8)
    assert options['hardware'] == 'h100'
    assert (options['query_heads'], options['kv_heads'], options['head_dim']) == (32, 4, 128)
    assert options['layer_count'] == 4
    assert options['legacy_transport_backend'] == 'nccl_p2p'
    assert options['non_attention'] == dict(measurements, source=str(path.resolve()))
    # The model's domain check saw the training configuration.
    assert checked[-1] is not None and checked[-1].num_layers == 4


@pytest.mark.parametrize('granularity,recompute_forward', [(None, False), ('full', True)])
def test_recomputation_follows_the_training_configuration(tmp_path, granularity, recompute_forward):
    config = _gqa_training_config(
        _runtime_config(tmp_path, _BENCHMARK),
        recompute_granularity=granularity,
        recompute_method='uniform',
    )
    spec = prefetch.runtime_spec(config, 8)
    assert spec['model_options']['recompute_forward'] is recompute_forward


def test_selective_recomputation_is_reported_as_outside_the_layer_program(tmp_path):
    config = _gqa_training_config(
        _runtime_config(tmp_path, _BENCHMARK), recompute_granularity='selective'
    )
    with pytest.raises(ValueError, match="recompute_granularity None or 'full', got 'selective'"):
        prefetch.runtime_spec(config, 8)


def test_benchmark_runtime_config_does_not_offer_manual_profile_selection(tmp_path):
    path = _runtime_config(tmp_path, dict(_BENCHMARK, profile='experimental_cp1_bounded'))
    with pytest.raises(ValueError, match='profile'):
        prefetch.runtime_spec(_gqa_training_config(path), 8)


def test_layer_type_names_come_from_the_cost_model(tmp_path):
    measurements = {'forward_ms_per_layer': {'fa': 1.0}, 'backward_ms_per_layer': 1.0}
    path = _runtime_config(tmp_path, dict(_BENCHMARK, non_attention=measurements))
    with pytest.raises(ValueError) as caught:
        prefetch.runtime_spec(_gqa_training_config(path), 8)
    assert (
        "non_attention.forward_ms_per_layer must be one number, or one number per layer type "
        "['gqa']; got ['fa']"
    ) in str(caught.value)


def test_layer_type_check_fails_closed_for_a_model_without_declared_types(tmp_path, monkeypatch):
    class Undeclared(QuadraticCostModel):
        pass

    monkeypatch.setattr(models, 'model_class', lambda name: Undeclared)
    config = _runtime_training_config(_runtime_config(tmp_path, _BENCHMARK))
    with pytest.raises(ValueError, match='does not declare non_attention_layer_types'):
        prefetch.runtime_spec(config, 8)


def test_faults_of_the_file_alone_are_not_blamed_on_the_training_configuration(tmp_path):
    path = _runtime_config(tmp_path, dict(_BENCHMARK, hardware='gb300'))
    with pytest.raises(ValueError, match='hardware must be h100 or gb200') as caught:
        prefetch.runtime_spec(_gqa_training_config(path), 8)
    assert 'training configuration' not in str(caught.value)
    path = _runtime_config(tmp_path, dict(_DYNAMIC_USP, calibration='missing-table.json'))
    config = _runtime_training_config(path, min_dynamic_context_parallel_size=8)
    with pytest.raises(ValueError, match='Cannot read DynamicUSP calibration') as caught:
        prefetch.runtime_spec(config, 8)
    assert 'training configuration' not in str(caught.value)
    assert "(type 'dynamic_usp')" in str(caught.value)


def test_a_training_configuration_outside_the_model_domain_is_named_as_such(tmp_path):
    config = _gqa_training_config(
        _runtime_config(tmp_path, _BENCHMARK), thd_tail_padding_policy='append_dummy_seq'
    )
    with pytest.raises(ValueError) as caught:
        prefetch.runtime_spec(config, 8)
    assert (
        "(type 'dcp_benchmark') does not match the training configuration: "
        'Benchmark predictors require extend_last tail padding'
    ) in str(caught.value)


def test_dynamic_usp_domain_error_uses_the_scheduler_type_name(tmp_path):
    config = _runtime_training_config(_runtime_config(tmp_path, _DYNAMIC_USP))
    with pytest.raises(ValueError, match="type 'dynamic_usp' requires") as caught:
        prefetch.runtime_spec(config, 32)
    assert 'DPxCP=32, minimum=1' in str(caught.value)
    assert 'usp_calib_v1' not in str(caught.value)


def test_runtime_scheduler_supports_one_pipeline_stage(tmp_path, geometry_only_model):
    config = _runtime_training_config(_runtime_config(tmp_path, _BENCHMARK))
    with pytest.raises(ValueError, match='PP1/VPP1 only'):
        prefetch.runtime_spec(config, 8, pp_size=2)
    config = _runtime_training_config(
        _runtime_config(tmp_path, _BENCHMARK), virtual_pipeline_model_parallel_size=2
    )
    with pytest.raises(ValueError, match='PP1/VPP1 only'):
        prefetch.runtime_spec(config, 8)


# --------------------------------------------------------------------------------------
# Read once, validated once: the specification kept on the training configuration
# --------------------------------------------------------------------------------------
def test_runtime_file_is_read_once_per_training_configuration(
    tmp_path, monkeypatch, geometry_only_model
):
    reads = _count_file_reads(monkeypatch)
    path = _runtime_config(tmp_path, _BENCHMARK)
    config = _runtime_training_config(path)
    first = prefetch.runtime_spec(config, 8)
    path.write_text('{"type": "dynamic_usp"}')  # An edit while training runs has no effect.
    second = prefetch.runtime_spec(config, 8)
    path.unlink()
    assert prefetch.runtime_spec(config, 8) == first == second
    assert len(reads) == 1
    # Every caller gets its own copy of the validated specification.
    first['solver_options']['time_budget_s'] = 0.25
    first['geometry']['allowed_cp_sizes'] = (1,)
    assert prefetch.runtime_spec(config, 8) == second


@pytest.mark.parametrize(
    'field,value',
    [
        ('min_dynamic_context_parallel_size', 2),
        ('max_seqlen_per_dp_cp_rank', 8192),
        ('expert_model_parallel_size', 2),
        ('dynamic_cp_runtime_config', '/elsewhere/runtime.json'),
    ],
)
def test_configuration_fields_cannot_change_after_validation(
    tmp_path, geometry_only_model, field, value
):
    config = _runtime_training_config(_runtime_config(tmp_path, _BENCHMARK))
    prefetch.runtime_spec(config, 8)
    setattr(config, field, value)
    with pytest.raises(ValueError, match='cannot change after it was validated') as caught:
        prefetch.runtime_spec(config, 8)
    assert field in str(caught.value)


def test_fingerprint_covers_fields_read_by_the_cost_model_domain_check(tmp_path):
    config = _gqa_training_config(_runtime_config(tmp_path, _BENCHMARK))
    prefetch.runtime_spec(config, 8)
    config.unrelated_training_option = 'changed'
    prefetch.runtime_spec(config, 8)
    config.cp_comm_type = 'a2a'
    with pytest.raises(ValueError, match=r"validated: configuration fields \['cp_comm_type'\]"):
        prefetch.runtime_spec(config, 8)


@pytest.mark.parametrize('request_change', [dict(world_size=16), dict(max_num_seqs=3)])
def test_geometry_cannot_change_after_validation(tmp_path, geometry_only_model, request_change):
    config = _runtime_training_config(_runtime_config(tmp_path, _BENCHMARK))
    prefetch.runtime_spec(config, 8)
    arguments = dict(dict(world_size=8, max_num_seqs=None), **request_change)
    with pytest.raises(ValueError, match='cannot change after it was validated'):
        prefetch.runtime_spec(
            config, arguments['world_size'], max_num_seqs=arguments['max_num_seqs']
        )


def test_a_failed_validation_is_not_kept(tmp_path, geometry_only_model):
    path = _runtime_config(tmp_path, dict(_BENCHMARK, type='dcp_benchmrk'))
    config = _runtime_training_config(path)
    with pytest.raises(ValueError, match="'type' must be one of"):
        prefetch.runtime_spec(config, 8)
    path.write_text(json.dumps(_BENCHMARK))
    assert prefetch.runtime_spec(config, 8)['type'] == 'dcp_benchmark'


# --------------------------------------------------------------------------------------
# Start-up validation on every rank, and its reuse by wrap_data_iterator
# --------------------------------------------------------------------------------------
def test_wrap_activates_runtime_with_classic_process_groups(
    monkeypatch, tmp_path, geometry_only_model
):
    trace_dir = str(tmp_path / 'trace')
    path = _runtime_config(tmp_path, dict(_BENCHMARK, **{'async': False, 'trace_dir': trace_dir}))
    config = _wrap_config(path)
    groups = _wrap_environment(monkeypatch)
    reads = _count_file_reads(monkeypatch)
    iterator = SimpleNamespace()
    created = []

    def create(*args):
        runtime = SimpleNamespace(spec=args[3], asynchronous=args[4], trace_dir=args[5])
        created.append(runtime)
        return runtime

    def run(self, *args):
        assert self.runtime is iterator._dcp_schedule_prefetcher
        assert self.prefetch_next
        return iter([]), 1, 1, 1

    monkeypatch.setattr(prefetch, 'DcpSchedulePrefetcher', create)
    monkeypatch.setattr(data_schedule.DefaultDynamicCPScheduler, 'run', run)
    data_schedule.wrap_data_iterator(iterator, config, 1, groups, prefetch_next=True)
    path.unlink()  # Later steps do not read the file again.
    data_schedule.wrap_data_iterator(iterator, config, 1, groups, prefetch_next=True)
    assert len(created) == 1 and len(reads) == 1
    assert created[0].asynchronous is False
    assert created[0].trace_dir == trace_dir
    assert created[0].spec['type'] == 'dcp_benchmark'


def test_startup_validation_needs_no_data_iterator_and_is_reused_by_every_step(
    monkeypatch, tmp_path, geometry_only_model
):
    path = _runtime_config(tmp_path, _BENCHMARK)
    config = _wrap_config(path)
    groups = _wrap_environment(monkeypatch)
    reads = _count_file_reads(monkeypatch)
    created, schedulers = [], []
    monkeypatch.setattr(
        prefetch, 'DcpSchedulePrefetcher', lambda *args: created.append(args[3]) or object()
    )
    monkeypatch.setattr(
        data_schedule.DefaultDynamicCPScheduler,
        'run',
        lambda self, *args: schedulers.append(self) or (iter([]), 1, 1, 1),
    )
    # What training does on every rank before the first step, data iterator or not.
    data_schedule.validate_dynamic_cp_runtime(config, groups)
    assert config._dynamic_cp_runtime['coverage_checked'] and len(reads) == 1
    path.unlink()
    # A rank without a data iterator (TP rank > 0) only confirms the validated result.
    data_schedule.wrap_data_iterator(None, config, 1, groups)
    assert created == [] and not hasattr(schedulers[-1], 'runtime')
    data_schedule.wrap_data_iterator(SimpleNamespace(), config, 1, groups)
    data_schedule.wrap_data_iterator(SimpleNamespace(), config, 1, groups)  # Evaluation data.
    assert len(created) == 2 and len(reads) == 1


def test_startup_validation_reports_a_bad_file_without_a_data_iterator(monkeypatch, tmp_path):
    groups = _wrap_environment(monkeypatch)
    config = _wrap_config(_runtime_config(tmp_path, dict(_BENCHMARK, type='dcp_benchmrk')))
    with pytest.raises(ValueError, match="'type' must be one of"):
        data_schedule.validate_dynamic_cp_runtime(config, groups)
    config = _gqa_training_config(
        _runtime_config(tmp_path, _BENCHMARK), pad_packed_seq_alignment=None, **_WRAP
    )
    with pytest.raises(ValueError, match='does not match the training configuration'):
        data_schedule.validate_dynamic_cp_runtime(config, groups)


def test_startup_validation_without_a_runtime_file_does_nothing():
    data_schedule.validate_dynamic_cp_runtime(SimpleNamespace(), None)
    data_schedule.validate_dynamic_cp_runtime(SimpleNamespace(dynamic_cp_runtime_config=None), None)


def test_startup_validation_needs_the_models_dp_cp_and_pp_groups(tmp_path):
    from unittest import mock

    from megatron.core.process_groups_config import MultiModuleProcessGroupCollection

    config = _wrap_config(_runtime_config(tmp_path, _BENCHMARK))
    # A multi-module schedule carrier has no dp_cp group; training passes the model's own.
    carrier = mock.MagicMock(spec=MultiModuleProcessGroupCollection)
    for collection in (SimpleNamespace(dp_cp=None, pp=None), carrier):
        with pytest.raises(ValueError, match='dp_cp and pp process groups'):
            data_schedule.validate_dynamic_cp_runtime(config, collection)


def test_runtime_file_is_not_ignored_without_the_dynamic_cp_scheduler(
    monkeypatch, tmp_path, geometry_only_model
):
    groups = _wrap_environment(monkeypatch)
    config = _wrap_config(
        _runtime_config(tmp_path, _BENCHMARK), sequence_packing_scheduler='dp_balanced'
    )
    with pytest.raises(ValueError, match='requires dynamic context parallelism'):
        data_schedule.validate_dynamic_cp_runtime(config, groups)
    with pytest.raises(ValueError, match='requires dynamic context parallelism'):
        data_schedule.wrap_data_iterator(SimpleNamespace(), config, 1, groups)


def test_startup_validation_uses_the_dp_cp_group_that_wrap_schedules_over(
    monkeypatch, tmp_path, geometry_only_model
):
    groups = _wrap_environment(monkeypatch)
    # With GTP_remat peers the packing domain is the larger group, at start-up and per step.
    groups.dp_gtp_remat = SimpleNamespace(rank=lambda: 0, size=lambda: 2)
    groups.dp_cp_gtp_remat = SimpleNamespace(rank=lambda: 0, size=lambda: 16)
    groups.dynamic_cp = dict.fromkeys((1, 2, 4, 8, 16))
    config = _wrap_config(_runtime_config(tmp_path, _BENCHMARK))
    assert data_schedule.validate_dynamic_cp_runtime(config, groups)['geometry']['world_size'] == 16
    monkeypatch.setattr(prefetch, 'DcpSchedulePrefetcher', lambda *args: object())
    monkeypatch.setattr(
        data_schedule.DefaultDynamicCPScheduler, 'run', lambda self, *args: (iter([]), 1, 1, 1)
    )
    # The step asks for the geometry that was validated: no "cannot change" error.
    data_schedule.wrap_data_iterator(SimpleNamespace(), config, 1, groups)


def test_wrap_needs_a_process_group_for_every_cp_size_the_planner_may_choose(
    monkeypatch, tmp_path, geometry_only_model
):
    groups = _wrap_environment(monkeypatch)
    groups.dynamic_cp = dict.fromkeys((1, 2, 8))  # A collection built without CP4 groups.
    config = _wrap_config(_runtime_config(tmp_path, _BENCHMARK))
    with pytest.raises(ValueError) as caught:
        data_schedule.wrap_data_iterator(SimpleNamespace(), config, 1, groups)
    assert str(caught.value) == (
        'The runtime DCP scheduler forms CP groups of [4] ranks, but this run has dynamic CP '
        'process groups only for sizes [1, 2, 8]'
    )


def test_wrap_needs_a_token_capacity_that_packed_padding_keeps(
    monkeypatch, tmp_path, geometry_only_model
):
    groups = _wrap_environment(monkeypatch)
    # Padding packs to multiples of 32 tokens leaves 992 of 1000; the planner would fill 1000.
    config = _wrap_config(
        _runtime_config(tmp_path, _BENCHMARK),
        max_seqlen_per_dp_cp_rank=1000,
        pad_packed_seq_alignment=32,
    )
    with pytest.raises(ValueError) as caught:
        data_schedule.wrap_data_iterator(SimpleNamespace(), config, 1, groups)
    assert str(caught.value) == (
        'The runtime DCP scheduler fills packs up to max_seqlen_per_dp_cp_rank=1000, which must '
        'be a multiple of the packed sequence alignment 32'
    )


def test_startup_validation_rejects_pipeline_parallelism(
    monkeypatch, tmp_path, geometry_only_model
):
    groups = _wrap_environment(monkeypatch, pp=2)
    config = _wrap_config(_runtime_config(tmp_path, _BENCHMARK))
    with pytest.raises(ValueError, match='PP1/VPP1 only'):
        data_schedule.validate_dynamic_cp_runtime(config, groups)


# --------------------------------------------------------------------------------------
# Start-up coverage probe: every allowed CP size must have cost-model parameters
# --------------------------------------------------------------------------------------
def test_probe_packs_fill_a_cp_group_without_padding():
    geometry = RuntimePlanGeometry(8, 4096, 1, (1, 2, 4, 8), 16, 64, True)
    assert dcp_runtime_worker._probe_packs(geometry, 1) == [(4096,)]
    for cp in (2, 4, 8):
        # One sequence across the whole group, then one sequence per rank.
        assert dcp_runtime_worker._probe_packs(geometry, cp) == [(4096 * cp,), (4096,) * cp]
    # Fewer, longer sequences when the per-group sequence cap is below the CP size.
    capped = RuntimePlanGeometry(8, 4096, 1, (8,), 16, 3, True)
    packs = dcp_runtime_worker._probe_packs(capped, 8)
    assert [len(pack) for pack in packs] == [1, 3]
    assert all(sum(pack) <= capped.capacity(8) and pack[0] % 16 == 0 for pack in packs)
    # Nothing aligned fits one rank: there is no pack to probe.
    assert dcp_runtime_worker._probe_packs(RuntimePlanGeometry(8, 8, 1, (1,), 16, 64), 1) == []


class _ManySequenceLimit(QuadraticCostModel):
    """Prices any CP1/CP2 pack, only single sequences at CP4, and nothing at CP8."""

    def forward(self, lengths, cp, *, ranks=None):
        if cp == 8 or (cp == 4 and len(lengths) > 1):
            raise ValueError(f'no parameters for {len(lengths)} sequences at CP{cp}')
        return super().forward(lengths, cp, ranks=ranks)


class _DeclaredSequenceLimit(_ManySequenceLimit):
    """The same model, telling the planner what it prices at CP4."""

    def pack_limits(self, cp):
        return PackLimits(max_sequences=1 if cp == 4 else None)


def test_a_cp_size_is_covered_when_the_model_prices_the_largest_pack_the_planner_builds():
    geometry = RuntimePlanGeometry(8, 4096, 1, (1, 2, 4, 8), 16, 64, True)
    # Pricing one full pack is not enough: the planner would fill CP4 groups with up to 64
    # sequences, which this model rejects. Without a declared limit that is a gap.
    planner = LightweightRuntimePlanner(geometry, _ManySequenceLimit(geometry))
    assert dcp_runtime_worker.coverage_gaps(planner) == {
        4: (
            'no parameters for 64 sequences at CP4 (for a full pack of 64 sequences, the most '
            'the planner puts in one group of this size)'
        ),
        8: 'no parameters for 1 sequences at CP8',
    }
    # A model that declares its limit keeps the CP size: the planner honours the limit,
    # and the probe prices exactly the largest pack the planner may build.
    planner = LightweightRuntimePlanner(geometry, _DeclaredSequenceLimit(geometry))
    assert dcp_runtime_worker.coverage_gaps(planner) == {8: 'no parameters for 1 sequences at CP8'}
    assert [planner.sequence_limit(cp) for cp in (1, 2, 4)] == [64, 64, 1]


def test_cpu_worker_asks_for_pack_limits_before_it_serves_requests(monkeypatch):
    geometry = RuntimePlanGeometry(8, 4096, 1, (1, 2, 4, 8), 16, 64, True)
    asked = []

    class Recording(QuadraticCostModel):
        def pack_limits(self, cp):
            asked.append(cp)
            return PackLimits()

    planner = LightweightRuntimePlanner(geometry, Recording(geometry))
    monkeypatch.setattr(dcp_runtime_worker, '_initialize_planner', lambda spec: planner)
    # The specification line only: the process starts, serves no request and ends.
    monkeypatch.setattr(sys, 'stdin', io.StringIO('{}\n'))
    monkeypatch.setattr(sys, 'stdout', io.StringIO())
    dcp_runtime_worker.main()
    # A model may price packs to answer; that belongs to start-up, not to the time budget
    # of the first plan.
    assert asked == [1, 2, 4, 8]


def test_a_cp_size_whose_declared_limit_is_zero_is_reported_as_a_gap():
    geometry = RuntimePlanGeometry(8, 4096, 1, (1, 2), 16, 64, True)

    class NothingAtCp2(QuadraticCostModel):
        def pack_limits(self, cp):
            return PackLimits(max_sequences=0 if cp == 2 else None)

    planner = LightweightRuntimePlanner(geometry, NothingAtCp2(geometry))
    assert dcp_runtime_worker.coverage_gaps(planner) == {
        2: 'the cost model declares that it prices no pack of this CP size'
    }


def test_coverage_probe_finds_cp_sizes_without_predictor_parameters(tmp_path):
    path = _runtime_config(tmp_path, _BENCHMARK)
    covered = prefetch.runtime_spec(_gqa_training_config(path), 8)
    assert prefetch._coverage_error(covered) is None
    # The h100 parameters cover multi-CP groups at 4096 tokens per rank only. The model's
    # own domain check accepts this configuration; planning would fail at some later step.
    spec = prefetch.runtime_spec(_gqa_training_config(path, max_seqlen_per_dp_cp_rank=8192), 8)
    message = prefetch._coverage_error(spec)
    assert (
        "The 'dcp_benchmark' cost model cannot price CP groups of [2, 4, 8] ranks with "
        'max_seqlen_per_dp_cp_rank=8192'
    ) in message
    assert 'no measured h100' in message and '(CP sizes [2, 4, 8])' in message
    assert 'set "max_cp_size": 1' in message
    # With that cap every remaining size is covered.
    capped = _runtime_config(tmp_path, dict(_BENCHMARK, max_cp_size=1))
    spec = prefetch.runtime_spec(_gqa_training_config(capped, max_seqlen_per_dp_cp_rank=8192), 8)
    assert prefetch._coverage_error(spec) is None


def test_coverage_probe_says_when_no_cap_can_help(tmp_path):
    # gb200 single-rank parameters stop at 8192 tokens.
    path = _runtime_config(tmp_path, dict(_BENCHMARK, hardware='gb200'))
    spec = prefetch.runtime_spec(_gqa_training_config(path, max_seqlen_per_dp_cp_rank=16384), 8)
    message = prefetch._coverage_error(spec)
    assert 'cannot price CP groups of [1, 2, 4, 8] ranks' in message
    assert 'The smallest allowed CP size is among them, so no "max_cp_size" avoids them' in message
    assert 'set "max_cp_size"' not in message


def test_dynamic_usp_table_covers_its_cp_tiers(tmp_path):
    path = _runtime_config(tmp_path, _DYNAMIC_USP)
    config = _runtime_training_config(
        path, max_seqlen_per_dp_cp_rank=1024, min_dynamic_context_parallel_size=8
    )
    spec = prefetch.runtime_spec(config, 32)
    assert spec['geometry']['allowed_cp_sizes'] == (8, 16, 32)
    assert prefetch._coverage_error(spec) is None


def _coverage_configs(tmp_path, tokens_per_rank):
    path = _runtime_config(tmp_path, _BENCHMARK)
    configs = [
        _gqa_training_config(path, max_seqlen_per_dp_cp_rank=tokens_per_rank, **_WRAP)
        for _ in range(2)
    ]
    for config in configs:
        prefetch.runtime_spec(config, 8)
    return configs


def test_coverage_outcome_is_broadcast_so_every_rank_raises_the_same_error(tmp_path, monkeypatch):
    groups = _wrap_environment(monkeypatch)
    leader, follower = _coverage_configs(tmp_path, 8192)
    wire, probes = [], []
    probe = prefetch._coverage_error

    def broadcast(payload, **kwargs):
        if wire:
            payload[:] = wire[0]  # A follower receives what the leader sent.
        else:
            wire.append(list(payload))

    monkeypatch.setattr(torch.distributed, 'broadcast_object_list', broadcast)
    monkeypatch.setattr(prefetch, '_coverage_error', lambda spec: probes.append(1) or probe(spec))
    raised = []
    for config, rank in ((leader, 0), (follower, 5)):
        group = SimpleNamespace(rank=lambda rank=rank: rank, size=lambda: 8)
        with pytest.raises(ValueError, match='cannot price CP groups') as caught:
            prefetch.check_cost_model_coverage(config, group)
        raised.append(str(caught.value))
    assert raised[0] == raised[1] and len(probes) == 1  # Only the leader probes.
    # The outcome is final: a caller that survived the first error gets it again, on every
    # rank, and the collective is not repeated (a later call may be made by one rank only).
    for config in (leader, follower):
        with pytest.raises(ValueError) as caught:
            prefetch.check_cost_model_coverage(config, groups.dp_cp)
        assert str(caught.value) == raised[0]
    assert len(wire) == 1 and len(probes) == 1
    with pytest.raises(ValueError, match='cannot price CP groups'):
        data_schedule.wrap_data_iterator(SimpleNamespace(), leader, 1, groups)


def test_a_passed_coverage_check_is_not_repeated(tmp_path, monkeypatch):
    groups = _wrap_environment(monkeypatch)
    sent = _record_outcomes(monkeypatch)
    config = _gqa_training_config(_runtime_config(tmp_path, _BENCHMARK), **_WRAP)
    prefetch.runtime_spec(config, 8)
    for _ in range(3):  # Start-up, then one prefetcher each for training and evaluation data.
        prefetch.check_cost_model_coverage(config, groups.dp_cp)
    assert sent == [None]
    assert config._dynamic_cp_runtime['coverage_checked']
    assert config._dynamic_cp_runtime['coverage_error'] is None


def test_a_failing_probe_on_the_leader_still_reaches_every_rank(tmp_path, monkeypatch):
    groups = _wrap_environment(monkeypatch)
    leader, _ = _coverage_configs(tmp_path, 4096)
    sent = _record_outcomes(monkeypatch)

    def broken(spec):
        raise OSError(5, 'cannot read predictor parameters')

    monkeypatch.setattr(prefetch, '_coverage_error', broken)
    with pytest.raises(ValueError, match='cannot read predictor parameters'):
        prefetch.check_cost_model_coverage(leader, groups.dp_cp)
    assert len(sent) == 1


def test_covered_configuration_passes_startup_and_wrap_without_error(tmp_path, monkeypatch):
    groups = _wrap_environment(monkeypatch)
    sent = _record_outcomes(monkeypatch)
    config = _gqa_training_config(_runtime_config(tmp_path, _BENCHMARK), **_WRAP)
    data_schedule.validate_dynamic_cp_runtime(config, groups)
    assert sent == [None]


def test_wrap_runs_the_coverage_probe_when_startup_validation_was_skipped(tmp_path, monkeypatch):
    groups = _wrap_environment(monkeypatch)
    config = _gqa_training_config(
        _runtime_config(tmp_path, _BENCHMARK), max_seqlen_per_dp_cp_rank=8192, **_WRAP
    )
    created = []
    monkeypatch.setattr(prefetch, 'DcpSchedulePrefetcher', lambda *args: created.append(args))
    with pytest.raises(ValueError, match=r'cannot price CP groups of \[2, 4, 8\] ranks'):
        data_schedule.wrap_data_iterator(SimpleNamespace(), config, 1, groups)
    assert created == []


# --------------------------------------------------------------------------------------
# Leader-only failures reach every rank through the schedule broadcast (RC-5)
# --------------------------------------------------------------------------------------
def _record_broadcasts(monkeypatch):
    sent = []
    monkeypatch.setattr(
        torch.distributed,
        'broadcast_object_list',
        lambda payload, **kwargs: sent.append(dict(payload[0])),
    )
    return sent


@pytest.mark.parametrize('asynchronous', [False, True])
def test_default_baseline_failure_drops_the_incumbent_on_the_leader(
    make_prefetcher, monkeypatch, asynchronous
):
    owner = make_prefetcher([1, 2, 3, 4], asynchronous=asynchronous)
    sent = _record_broadcasts(monkeypatch)
    received = []

    def baseline(lengths):
        # data_schedule_utils.next_hdp_group_packing_aware on a batch it cannot place.
        raise AssertionError('Sequence length 36864 requires CP size 16, but only 12 DPxCP ranks')

    def solve(logical, padded, incumbent):
        received.append(incumbent)
        return _solve(logical, padded, incumbent)

    owner.baseline = baseline
    monkeypatch.setattr(owner.worker, 'solve', solve)
    _consume(owner, 2)
    owner.prefetch(2)  # Asynchronous planning runs the baseline one step ahead.
    _consume(owner, 2)
    assert received == [None, None]
    assert len(sent) == 2 and all('error' not in payload for payload in sent)
    assert 'requires CP size 16' in owner.last_telemetry['incumbent_error']


@pytest.mark.parametrize('asynchronous', [False, True])
def test_leader_submission_failure_is_broadcast_before_it_is_raised(
    make_prefetcher, monkeypatch, asynchronous
):
    owner = make_prefetcher([1, 2, 3, 4], asynchronous=asynchronous)
    sent = _record_broadcasts(monkeypatch)
    _consume(owner, 2)

    def refuse(*args, **kwargs):
        raise RuntimeError('cannot schedule new futures after shutdown')

    monkeypatch.setattr(owner.executor, 'submit', refuse)
    owner.prefetch(2)  # Must not raise on the leader while the other ranks train.
    with pytest.raises(RuntimeError, match='cannot schedule new futures after shutdown'):
        _consume(owner, 2)
    assert len(sent) == 2 and 'error' in sent[1]


def test_schedule_error_has_the_same_type_and_message_on_every_rank(make_prefetcher, monkeypatch):
    message = (
        'sample 0: seq_len=36864 requires CP >= 9 before padding, but no allowed CP size fits '
        'after alignment; max_cp_size=4, allowed_cp_sizes=(1, 2, 4), max_seqlen_per_dp_cp_rank=4096'
    )
    leader, follower = make_prefetcher([1, 2]), make_prefetcher([1, 2])
    follower.executor.shutdown()
    follower.executor = follower.worker = follower.baseline = None
    wire = []

    def broadcast(payload, **kwargs):
        if wire:
            payload[0] = wire[0]  # A follower receives what the leader sent.
        else:
            wire.append(payload[0])

    def fail(*args):
        raise RuntimeScheduleError(message)

    monkeypatch.setattr(torch.distributed, 'broadcast_object_list', broadcast)
    monkeypatch.setattr(leader.worker, 'solve', fail)
    raised = []
    for owner in (leader, follower):
        with pytest.raises(RuntimeScheduleError) as caught:
            _consume(owner, 2)
        raised.append(str(caught.value))
    assert raised == [message, message]


def test_leader_start_failure_is_reported_through_the_first_broadcast(make_prefetcher, monkeypatch):
    sent = _record_broadcasts(monkeypatch)

    def no_worker(spec):
        raise OSError(12, 'Cannot allocate memory')

    monkeypatch.setattr(prefetch, '_CpuPlanner', no_worker)
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 8)
    geometry = dict(
        world_size=8,
        tokens_per_rank=1024,
        allowed_cp_sizes=(1, 2, 4, 8),
        max_group_seqs=64,
        dataset_alignment=16,
    )
    # Raising here would stop the leader alone, before any collective.
    owner = prefetch.DcpSchedulePrefetcher(
        _ReplayIterator([1, 2]), group, group, {'geometry': geometry}, False, None
    )
    try:
        with pytest.raises(RuntimeError, match='Cannot allocate memory'):
            _consume(owner, 2)
        assert len(sent) == 1 and 'error' in sent[0]
    finally:
        owner.close()


def _dynamic_usp_spec(tmp_path, **training):
    path = _runtime_config(tmp_path, {'type': 'dynamic_usp'})
    values = dict(max_seqlen_per_dp_cp_rank=16384, min_dynamic_context_parallel_size=8)
    values.update(training)
    return prefetch.runtime_spec(_runtime_training_config(path, **values), 8)


def test_cpu_worker_reports_an_unschedulable_sample_as_a_schedule_error(tmp_path):
    worker = prefetch._CpuPlanner(_dynamic_usp_spec(tmp_path))
    try:
        with pytest.raises(RuntimeScheduleError, match='sample 0: seq_len=131088') as caught:
            worker.solve([131088], [131088], None)
        assert 'Traceback' not in str(caught.value)
        assert 'max_seqlen_per_dp_cp_rank=16384' in str(caught.value)
        # The persistent worker survives a rejected batch.
        assert len(worker.solve([1000], [1008], None)['rows'][0]) == 8
    finally:
        worker.close()


@pytest.mark.parametrize('thd_max_packed_sequences,cap', [(None, 64), (8, 8)])
def test_default_incumbent_over_the_sequence_cap_does_not_stop_planning(
    tmp_path, thd_max_packed_sequences, cap
):
    """The real default heuristic ignores the per-group sequence cap (RC-2)."""
    from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler

    spec = prefetch.runtime_spec(
        _runtime_training_config(
            _runtime_config(tmp_path, {'type': 'dynamic_usp'}),
            max_seqlen_per_dp_cp_rank=16384,
            min_dynamic_context_parallel_size=8,
        ),
        8,
        max_num_seqs=thd_max_packed_sequences,
    )
    geometry = spec['geometry']
    assert geometry['max_group_seqs'] == cap
    lengths = [16] * 65
    incumbent = DefaultDynamicCPScheduler(
        geometry['tokens_per_rank'], 8, 1, None, max_num_seqs=cap, min_cp_size=8
    ).get_groups_and_subsamples(list(enumerate(lengths)))
    assert len(incumbent) == 1 and len(incumbent[0][0]) == 65  # One CP8 group, 65 sequences.
    worker = prefetch._CpuPlanner(spec)
    try:
        result = worker.solve(lengths, lengths, incumbent)
    finally:
        worker.close()
    groups = validate_rows(
        result['rows'], dict(enumerate(lengths)), RuntimePlanGeometry(**geometry), 8
    )
    assert max(len(ids) for row in groups for _, _, ids in row) <= cap
    assert (
        result['statistics']['incumbent_rejection'] == 'default_incumbent_violates_runtime_contract'
    )
    assert 'packs 65 sequences' in result['statistics']['incumbent_rejection_detail']


# --------------------------------------------------------------------------------------
# Start-up: no implicit CP cap; the real-sequence cap follows the tail-padding policy
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize('world,minimum,largest', [(12, 1, 4), (24, 2, 8), (6, 1, 2)])
def test_runtime_spec_needs_an_explicit_cap_when_the_full_dp_cp_group_is_unavailable(
    tmp_path, geometry_only_model, world, minimum, largest
):
    training = dict(max_seqlen_per_dp_cp_rank=1024, min_dynamic_context_parallel_size=minimum)
    config = _runtime_training_config(_runtime_config(tmp_path, _BENCHMARK), **training)
    with pytest.raises(ValueError, match=f'"max_cp_size": {largest}') as caught:
        prefetch.runtime_spec(config, world)
    assert f'DPxCP={world}' in str(caught.value)
    assert f'{largest * 1024} tokens' in str(caught.value)
    path = _runtime_config(tmp_path, dict(_BENCHMARK, max_cp_size=largest))
    spec = prefetch.runtime_spec(_runtime_training_config(path, **training), world)
    assert max(spec['geometry']['allowed_cp_sizes']) == largest


@pytest.mark.parametrize(
    'limit,policy,alignment,expected',
    [
        (4, None, 'max', 3),
        (4, 'append_dummy_seq', 'max', 3),
        (4, 'extend_last', 'max', 4),
        (4, None, None, 4),
        # The usual 64-entry THD buffer: 63 real sequences next to a dummy tail sequence.
        (64, 'append_dummy_seq', 'max', 63),
        (64, 'extend_last', 'max', 64),
        (128, 'extend_last', 'max', 64),
    ],
)
def test_wrap_passes_the_real_sequence_cap_to_the_runtime(
    monkeypatch, tmp_path, geometry_only_model, limit, policy, alignment, expected
):
    config = _wrap_config(
        _runtime_config(tmp_path, _BENCHMARK),
        thd_max_packed_sequences=limit,
        thd_tail_padding_policy=policy,
        pad_packed_seq_alignment=alignment,
    )
    groups = _wrap_environment(monkeypatch)
    specs = []

    def create(*args):
        specs.append(args[3])
        return SimpleNamespace(spec=args[3])

    monkeypatch.setattr(prefetch, 'DcpSchedulePrefetcher', create)
    monkeypatch.setattr(
        data_schedule.DefaultDynamicCPScheduler, 'run', lambda self, *args: (iter([]), 1, 1, 1)
    )
    data_schedule.wrap_data_iterator(SimpleNamespace(), config, 1, groups)
    assert specs[0]['geometry']['max_group_seqs'] == expected


def test_closing_the_cpu_planner_does_not_wait_for_a_worker_that_ignores_eof(monkeypatch):
    worker = prefetch._CpuPlanner.__new__(prefetch._CpuPlanner)
    # Stands for a planner whose request pipe is still held open by a later fork.
    worker.process = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(600)'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    monkeypatch.setattr(prefetch._CpuPlanner, 'close_timeout_s', 0.5)
    begin = time.monotonic()
    worker.close()
    assert time.monotonic() - begin < 30
    assert worker.process.poll() is not None


# --------------------------------------------------------------------------------------
# What the user sees at start-up: one line, and files that every rank read alike
# --------------------------------------------------------------------------------------
def test_startup_line_names_what_the_runtime_file_selected(tmp_path):
    path = _runtime_config(tmp_path, dict(_BENCHMARK, max_cp_size=4))
    spec = prefetch.runtime_spec(_gqa_training_config(path), 8, max_num_seqs=63)
    assert spec['hardware'] == 'h100' and spec['nvlink_domain_size'] == 8
    assert prefetch.describe_runtime_spec(spec) == (
        'Runtime DCP scheduler: type=dcp_benchmark, cost model=dcp_benchmark, '
        'hardware=h100 (NVLink domain: 8 consecutive ranks), '
        'CP group sizes=[1, 2, 4], at most 63 sequences per CP group, async=on, '
        f'runtime file={path.resolve()}'
    )
    path = _runtime_config(tmp_path, dict(_BENCHMARK, hardware='gb200'))
    spec = prefetch.runtime_spec(_gqa_training_config(path), 8)
    assert spec['hardware'] == 'gb200' and spec['nvlink_domain_size'] == 72
    assert prefetch.describe_runtime_spec(spec) == (
        'Runtime DCP scheduler: type=dcp_benchmark, cost model=dcp_benchmark, '
        'hardware=gb200 (NVLink domain: 72 consecutive ranks), '
        'CP group sizes=[1, 2, 4, 8], at most 64 sequences per CP group, async=on, '
        f'runtime file={path.resolve()}'
    )
    path = _runtime_config(tmp_path, dict(_DYNAMIC_USP, **{'async': False}))
    config = _runtime_training_config(
        path, max_seqlen_per_dp_cp_rank=1024, min_dynamic_context_parallel_size=8
    )
    # No "hardware" in the file: none in the line. The domain is the one of the packaged table.
    assert prefetch.describe_runtime_spec(prefetch.runtime_spec(config, 32)) == (
        'Runtime DCP scheduler: type=dynamic_usp, cost model=usp_calib_v1, '
        'NVLink domain: 8 consecutive ranks, '
        'CP group sizes=[8, 16, 32], at most 64 sequences per CP group, async=off, '
        f'runtime file={path.resolve()}'
    )
    # A table given with "calibration" does not say where it was measured: no domain in the line.
    table = tmp_path / 'table.json'
    table.write_bytes((Path(models.__file__).parent / 'usp_calib_v1.json').read_bytes())
    path = _runtime_config(tmp_path, dict(_DYNAMIC_USP, calibration='table.json'))
    config = _runtime_training_config(
        path, max_seqlen_per_dp_cp_rank=1024, min_dynamic_context_parallel_size=8
    )
    spec = prefetch.runtime_spec(config, 32)
    assert spec['nvlink_domain_size'] is None
    assert prefetch.describe_runtime_spec(spec) == (
        'Runtime DCP scheduler: type=dynamic_usp, cost model=usp_calib_v1, '
        'CP group sizes=[8, 16, 32], at most 64 sequences per CP group, async=on, '
        f'runtime file={path.resolve()}'
    )


def test_startup_validation_returns_the_validated_specification(tmp_path, monkeypatch):
    groups = _wrap_environment(monkeypatch)
    config = _gqa_training_config(_runtime_config(tmp_path, _BENCHMARK), **_WRAP)
    spec = data_schedule.validate_dynamic_cp_runtime(config, groups)
    assert spec == prefetch.runtime_spec(config, 8)
    assert data_schedule.validate_dynamic_cp_runtime(SimpleNamespace(), None) is None


def _two_rank_wire(monkeypatch, differing):
    """Leader then follower in one process: the broadcast wire, and the rank count's inputs."""
    wire, flags = [], []

    def broadcast(payload, **kwargs):
        if wire:
            payload[:] = wire[0]  # A follower receives what the leader sent.
        else:
            wire.append(list(payload))

    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.distributed, 'get_global_rank', lambda *args: 0)
    monkeypatch.setattr(torch.distributed, 'broadcast_object_list', broadcast)
    # The all_reduce returns the group-wide count on every rank.
    monkeypatch.setattr(
        prefetch, '_count_over_group', lambda flag, group: flags.append(bool(flag)) or differing
    )
    return wire, flags


def test_ranks_that_read_different_runtime_files_all_stop_with_the_same_error(
    tmp_path, monkeypatch, geometry_only_model
):
    wire, flags = _two_rank_wire(monkeypatch, differing=1)
    path = _runtime_config(tmp_path, _BENCHMARK)
    leader = _runtime_training_config(path)
    prefetch.runtime_spec(leader, 8)
    # Another node sees other bytes under the same path ("async" alone changes the collectives).
    path.write_text(json.dumps(dict(_BENCHMARK, **{'async': False})))
    follower = _runtime_training_config(path)
    prefetch.runtime_spec(follower, 8)
    raised = []
    for config, rank in ((leader, 0), (follower, 5)):
        group = SimpleNamespace(rank=lambda rank=rank: rank, size=lambda: 8)
        with pytest.raises(
            ValueError, match='differs from the one the DPxCP leader read'
        ) as caught:
            prefetch.check_cost_model_coverage(config, group)
        raised.append(str(caught.value))
    assert raised[0] == raised[1]
    assert raised[0].startswith(
        '1 of 8 DPxCP ranks read a runtime configuration that differs from the one the DPxCP '
        f'leader read ({str(path.resolve())!r}). The file must be identical on every rank'
    )
    # The leader agrees with itself; the follower is the rank that is counted.
    assert flags == [False, True]
    # This outcome is final as well: the same error again, without another collective.
    for config in (leader, follower):
        with pytest.raises(ValueError) as caught:
            prefetch.check_cost_model_coverage(
                config, SimpleNamespace(rank=lambda: 0, size=lambda: 8)
            )
        assert str(caught.value) == raised[0]
    assert flags == [False, True] and len(wire) == 1
    assert wire[0][1] == dict(
        sha256=leader._dynamic_cp_runtime['spec']['config_sha256'], path=str(path.resolve())
    )


def test_ranks_that_read_the_same_runtime_file_pass(tmp_path, monkeypatch, geometry_only_model):
    wire, flags = _two_rank_wire(monkeypatch, differing=0)
    path = _runtime_config(tmp_path, _BENCHMARK)
    for rank in (0, 5):
        config = _runtime_training_config(path)
        prefetch.runtime_spec(config, 8)
        group = SimpleNamespace(rank=lambda rank=rank: rank, size=lambda: 8)
        prefetch.check_cost_model_coverage(config, group)
        assert config._dynamic_cp_runtime['coverage_checked']
    assert flags == [False, False]


def test_differing_ranks_are_counted_with_one_all_reduce_over_the_group(monkeypatch):
    original_tensor = torch.tensor
    monkeypatch.setattr(
        torch,
        'tensor',
        lambda *args, **kwargs: original_tensor(*args, **{**kwargs, 'device': 'cpu'}),
    )
    contributed = []

    def all_reduce(tensor, group=None):
        contributed.append((int(tensor.item()), group))
        tensor.add_(2)  # Two other ranks of the group raised their flag.

    monkeypatch.setattr(torch.distributed, 'all_reduce', all_reduce)
    group = object()
    assert prefetch._count_over_group(True, group) == 3
    assert prefetch._count_over_group(False, group) == 2
    assert contributed == [(1, group), (0, group)]


def _calibrated_dynamic_usp(tmp_path):
    packaged = Path(prefetch.__file__).parent / 'dcp_runtime_scheduler/models/usp_calib_v1.json'
    table = tmp_path / 'table.json'
    table.write_bytes(packaged.read_bytes())
    path = _runtime_config(tmp_path, dict(_DYNAMIC_USP, calibration='table.json'))
    config = _runtime_training_config(
        path, max_seqlen_per_dp_cp_rank=1024, min_dynamic_context_parallel_size=8
    )
    return table, config


def test_calibration_digest_is_pinned_at_startup_and_checked_by_the_worker(tmp_path):
    table, config = _calibrated_dynamic_usp(tmp_path)
    spec = prefetch.runtime_spec(config, 8)
    pinned = hashlib.sha256(table.read_bytes()).hexdigest()
    assert spec['calibration_sha256'] == pinned
    assert dcp_runtime_worker._initialize_planner(spec).model.calibration_sha256 == pinned
    # The same coefficients in other bytes: only the file that start-up validated is accepted.
    table.write_text(json.dumps(json.loads(table.read_text()), indent=1))
    changed = hashlib.sha256(table.read_bytes()).hexdigest()
    message = (
        f'The calibration file {str(table.resolve())!r} changed after start-up validation '
        f'(sha256 {pinned} then, {changed} now); it must not change during a run'
    )
    with pytest.raises(ValueError) as caught:
        dcp_runtime_worker._initialize_planner(spec)
    assert str(caught.value) == message
    # The CPU worker process refuses; consume() broadcasts this error to every rank.
    worker = prefetch._CpuPlanner(spec)
    try:
        with pytest.raises(RuntimeError) as caught:
            worker.solve([1000] * 8, [1008] * 8, None)
    finally:
        worker.close()
    assert message in str(caught.value)


def test_calibration_changed_before_the_coverage_probe_stops_every_rank(tmp_path, monkeypatch):
    groups = _wrap_environment(monkeypatch)
    sent = _record_outcomes(monkeypatch)
    table, config = _calibrated_dynamic_usp(tmp_path)
    prefetch.runtime_spec(config, 8)
    table.write_text(json.dumps(json.loads(table.read_text()), indent=1))
    with pytest.raises(ValueError, match='changed after start-up validation'):
        prefetch.check_cost_model_coverage(config, groups.dp_cp)
    assert len(sent) == 1 and 'changed after start-up validation' in sent[0]


def test_packaged_parameters_carry_no_calibration_pin(tmp_path):
    spec = _dynamic_usp_spec(tmp_path)
    assert 'calibration_sha256' not in spec


# --------------------------------------------------------------------------------------
# trace_dir: checked on every rank at start-up; one file per rank, records per iterator
# --------------------------------------------------------------------------------------
def test_trace_dir_is_expanded_and_created_at_startup(tmp_path, monkeypatch, geometry_only_model):
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    path = _runtime_config(tmp_path, dict(_BENCHMARK, trace_dir='~/traces/run1'))
    expected = tmp_path / 'home' / 'traces' / 'run1'
    assert load_runtime_config(str(path))['trace_dir'] == str(expected)
    assert not expected.exists()
    # Every rank validates, also one that never gets a data iterator.
    spec = prefetch.runtime_spec(_runtime_training_config(path), 8)
    assert spec['trace_dir'] == str(expected) and expected.is_dir()
    assert list(expected.iterdir()) == []


def test_relative_trace_dir_stays_relative_to_the_working_directory(
    tmp_path, monkeypatch, geometry_only_model
):
    run = tmp_path / 'run'
    run.mkdir()
    monkeypatch.chdir(run)
    path = _runtime_config(tmp_path, dict(_BENCHMARK, trace_dir='traces/a'))
    assert prefetch.runtime_spec(_runtime_training_config(path), 8)['trace_dir'] == 'traces/a'
    assert (run / 'traces' / 'a').is_dir()


@pytest.mark.parametrize('trace_dir', ['occupied/trace', 'occupied', 'nul\x00byte'])
def test_unusable_trace_dir_is_reported_at_startup(
    tmp_path, monkeypatch, geometry_only_model, trace_dir
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'occupied').write_text('a file, not a directory')
    path = _runtime_config(tmp_path, dict(_BENCHMARK, trace_dir=trace_dir))
    config = _runtime_training_config(path)
    with pytest.raises(ValueError) as caught:
        prefetch.runtime_spec(config, 8)
    assert str(caught.value).startswith(
        f'trace_dir {trace_dir!r} from the runtime configuration cannot be created or written '
        'on this rank: '
    )
    # Nothing is kept from the failed validation.
    assert not hasattr(config, '_dynamic_cp_runtime')
    path.write_text(json.dumps(dict(_BENCHMARK, trace_dir='free/trace')))
    assert prefetch.runtime_spec(config, 8)['trace_dir'] == 'free/trace'


def test_trace_records_of_two_iterators_are_told_apart(tmp_path, monkeypatch, make_prefetcher):
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    monkeypatch.setattr(
        prefetch,
        '_CpuPlanner',
        lambda spec: SimpleNamespace(solve=_solve, stop=lambda: None, close=lambda: None),
    )
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 8)
    geometry = dict(
        world_size=8,
        tokens_per_rank=1024,
        allowed_cp_sizes=(1, 2, 4, 8),
        max_group_seqs=64,
        dataset_alignment=16,
    )
    spec = {'type': 'dcp_benchmark', 'geometry': geometry}
    trace_dir = tmp_path / 'trace'
    # Training data, then evaluation data: two prefetchers that share this rank's file.
    owners = [
        prefetch.DcpSchedulePrefetcher(
            _ReplayIterator([16] * 6), group, group, spec, True, str(trace_dir)
        )
        for _ in range(2)
    ]
    try:
        training, evaluation = owners
        assert evaluation.iterator_index == training.iterator_index + 1
        _consume(training, 2)
        training.prefetch(2)
        _consume(evaluation, 2)
        _consume(training, 2)
        _consume(evaluation, 2)
    finally:
        for owner in owners:
            owner.close()
    assert [entry.name for entry in trace_dir.iterdir()] == ['rank00000.jsonl']
    records = [
        json.loads(line) for line in (trace_dir / 'rank00000.jsonl').read_text().splitlines()
    ]
    first, second = training.iterator_index, evaluation.iterator_index
    assert [
        (record['event'], record['iterator'], record['sequence_number']) for record in records
    ] == [
        ('consume', first, 0),
        ('submit', first, 1),
        ('consume', second, 0),
        ('consume', first, 1),
        ('consume', second, 1),
    ]


def test_trace_record_of_a_plan_names_the_nvlink_domain_of_the_cost_model(
    tmp_path, monkeypatch, make_prefetcher
):
    # The real CPU worker behind a prefetcher of the DPxCP leader; only the collectives are
    # replaced (by the make_prefetcher fixture).
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    spec = prefetch.runtime_spec(_gqa_training_config(_runtime_config(tmp_path, _BENCHMARK)), 8)
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 8)
    trace_dir = tmp_path / 'trace'
    owner = prefetch.DcpSchedulePrefetcher(
        _ReplayIterator([10400, 18000]), group, group, spec, False, str(trace_dir)
    )
    try:
        _consume(owner, 2)
    finally:
        owner.close()
    (record,) = [
        json.loads(line) for line in (trace_dir / 'rank00000.jsonl').read_text().splitlines()
    ]
    assert record['event'] == 'consume' and record['predicted_ms'] > 0
    assert record['model_metadata']['hardware'] == 'h100'
    assert record['model_metadata']['nvlink_domain_size'] == 8


def test_trace_says_whether_a_batch_was_planned_one_step_ahead(make_prefetcher):
    owner = make_prefetcher([1, 2, 3, 4, 5, 6, 7])
    _consume(owner, 2)
    # The first batch of an iterator is always planned when it is consumed.
    assert owner.last_telemetry['asynchronous'] is True
    assert owner.last_telemetry['lookahead'] is False
    owner.prefetch(2)
    _consume(owner, 2)
    assert owner.last_telemetry['lookahead'] is True
    # Without a prefetch call from the training loop, "async" alone plans nothing ahead.
    _consume(owner, 2)
    assert owner.last_telemetry['asynchronous'] is True
    assert owner.last_telemetry['lookahead'] is False
    # A batch prepared for another microbatch count is planned again.
    owner.prefetch(2)
    _consume(owner, 1)
    assert owner.last_telemetry['lookahead'] is False


def test_synchronous_planning_is_never_recorded_as_lookahead(make_prefetcher):
    owner = make_prefetcher([1, 2, 3, 4], asynchronous=False)
    _consume(owner, 2)
    owner.prefetch(2)
    _consume(owner, 2)
    assert owner.last_telemetry['asynchronous'] is False
    assert owner.last_telemetry['lookahead'] is False


# --------------------------------------------------------------------------------------
# Faults of the file or of a path in it surface as one kind of error
# --------------------------------------------------------------------------------------
def test_unknown_home_directory_user_is_a_configuration_error(tmp_path):
    with pytest.raises(ValueError, match='Cannot read runtime configuration') as caught:
        load_runtime_config('~dcp_no_such_user/runtime.json')
    assert "'~dcp_no_such_user/runtime.json'" in str(caught.value)
    for key in ('trace_dir', 'calibration'):
        path = _runtime_config(tmp_path, dict(_DYNAMIC_USP, **{key: '~dcp_no_such_user/data'}))
        with pytest.raises(ValueError) as caught:
            load_runtime_config(str(path))
        assert (
            f"Runtime configuration {str(path.resolve())!r}: {key} '~dcp_no_such_user/data' "
            'cannot be resolved'
        ) in str(caught.value)


def test_calibration_path_the_system_cannot_represent_is_a_configuration_error(tmp_path):
    path = _runtime_config(tmp_path, dict(_DYNAMIC_USP, calibration='table\x00.json'))
    with pytest.raises(ValueError, match='calibration .* cannot be resolved'):
        load_runtime_config(str(path))


def test_deeply_nested_json_is_a_configuration_error(tmp_path):
    path = tmp_path / 'runtime.json'
    path.write_text('{"type": "dynamic_usp", "trace_dir": ' + '[' * 200000)
    with pytest.raises(ValueError, match='Cannot read runtime configuration') as caught:
        load_runtime_config(str(path))
    assert str(path) in str(caught.value)


@pytest.mark.parametrize(
    'value,label',
    [
        (100000.5, 'forward_ms_per_layer'),
        (1e9, 'forward_ms_per_layer'),
        (10**12, 'forward_ms_per_layer'),
        ({'gqa': 2e5}, 'forward_ms_per_layer.gqa'),
    ],
)
def test_non_attention_times_above_the_bound_are_rejected_with_their_unit(tmp_path, value, label):
    non_attention = dict(_NON_ATTENTION, forward_ms_per_layer=value)
    path = _runtime_config(tmp_path, dict(_BENCHMARK, non_attention=non_attention))
    with pytest.raises(ValueError) as caught:
        load_runtime_config(str(path))
    number = value['gqa'] if isinstance(value, dict) else value
    assert (
        f'non_attention.{label} is {number}; the unit is milliseconds per layer, per '
        'microbatch, per rank, and at most 100000 is accepted'
    ) in str(caught.value)


def test_non_attention_times_up_to_the_bound_are_accepted(tmp_path):
    non_attention = {'forward_ms_per_layer': 1e5, 'backward_ms_per_layer': {'gqa': 100000}}
    path = _runtime_config(tmp_path, dict(_BENCHMARK, non_attention=non_attention))
    loaded = load_runtime_config(str(path))['options']['non_attention']
    assert loaded['forward_ms_per_layer'] == 1e5
    assert loaded['backward_ms_per_layer'] == {'gqa': 100000}


def test_invalid_non_attention_times_state_the_unit(tmp_path):
    non_attention = dict(_NON_ATTENTION, backward_ms_per_layer=-1)
    path = _runtime_config(tmp_path, dict(_BENCHMARK, non_attention=non_attention))
    with pytest.raises(ValueError) as caught:
        load_runtime_config(str(path))
    assert (
        'non_attention.backward_ms_per_layer must be a finite nonnegative number of '
        'milliseconds per layer, per microbatch, per rank'
    ) in str(caught.value)


# --------------------------------------------------------------------------------------
# Messages name the training setting that is at fault
# --------------------------------------------------------------------------------------
def test_unsupported_head_geometry_names_the_three_training_settings(tmp_path):
    path = _runtime_config(tmp_path, _BENCHMARK)
    config = _gqa_training_config(path, num_attention_heads=16, num_query_groups=16, kv_channels=64)
    with pytest.raises(ValueError) as caught:
        prefetch.runtime_spec(config, 8)
    message = str(caught.value)
    assert (
        "(type 'dcp_benchmark') does not match the training configuration: the 'dcp_benchmark' "
        'cost model has attention parameters for (num_attention_heads, num_query_groups, '
        f'kv_channels) in {list(BenchmarkCostModel.attention_geometries)}; this model has '
        '(16, 16, 64)'
    ) in message
    assert (32, 4, 128) in BenchmarkCostModel.attention_geometries
    # Every listed geometry is one that the packaged predictor accepts.
    for heads, groups, channels in BenchmarkCostModel.attention_geometries:
        supported = _gqa_training_config(
            path, num_attention_heads=heads, num_query_groups=groups, kv_channels=channels
        )
        assert prefetch.runtime_spec(supported, 8)['model'] == 'dcp_benchmark'


@pytest.mark.parametrize('layers', [None, 0, -2, 2.0, True])
def test_layer_count_is_named_as_a_training_setting(tmp_path, layers):
    config = _gqa_training_config(_runtime_config(tmp_path, _BENCHMARK), num_layers=layers)
    with pytest.raises(ValueError) as caught:
        prefetch.runtime_spec(config, 8)
    assert (
        'does not match the training configuration: num_layers must be a positive integer, '
        f'got {layers!r}'
    ) in str(caught.value)


def test_sequence_parallel_message_names_the_real_settings(tmp_path, geometry_only_model):
    path = _runtime_config(tmp_path, _BENCHMARK)
    config = _runtime_training_config(path, sequence_parallel=True, tensor_model_parallel_size=2)
    with pytest.raises(ValueError) as caught:
        prefetch.runtime_spec(config, 8)
    assert str(caught.value) == (
        'The runtime DCP scheduler currently requires sequence_parallel off or '
        'tensor_model_parallel_size=1'
    )
    for accepted in (
        dict(sequence_parallel=True, tensor_model_parallel_size=1),
        dict(sequence_parallel=False, tensor_model_parallel_size=2),
    ):
        assert prefetch.runtime_spec(_runtime_training_config(path, **accepted), 8)


@pytest.mark.parametrize('payload,minimum', [(_BENCHMARK, 1), (_DYNAMIC_USP, 8)])
def test_parallel_geometry_faults_are_not_blamed_on_the_runtime_file(tmp_path, payload, minimum):
    path = _runtime_config(tmp_path, payload)
    training = dict(min_dynamic_context_parallel_size=minimum, expert_model_parallel_size=3)
    config = (
        _gqa_training_config(path, **training)
        if payload is _BENCHMARK
        else _runtime_training_config(path, max_seqlen_per_dp_cp_rank=1024, **training)
    )
    with pytest.raises(ValueError) as caught:
        prefetch.runtime_spec(config, 8)
    assert (
        f"(type {payload['type']!r}) does not match the training configuration: "
        'expert_model_parallel_size (3) must divide dp_size*cp_size (8)'
    ) in str(caught.value)


def test_a_fault_raised_by_a_models_domain_check_is_a_training_configuration_fault(
    tmp_path, monkeypatch
):
    class Narrow(QuadraticCostModel):
        non_attention_layer_types = ('gqa',)

        def __init__(self, geometry, **options):
            super().__init__(geometry)
            self.check_domain(geometry)

        def check_domain(self, geometry, config=None):
            if geometry.world_size != 16:
                raise ValueError(f'measured with DPxCP=16, got {geometry.world_size}')

    class Unreadable(QuadraticCostModel):
        non_attention_layer_types = ('gqa',)

        def __init__(self, geometry, **options):
            raise ValueError('parameter file is malformed')

    config = _runtime_training_config(_runtime_config(tmp_path, _BENCHMARK))
    monkeypatch.setattr(models, 'model_class', lambda name: Narrow)
    with pytest.raises(ValueError) as caught:
        prefetch.runtime_spec(config, 8)
    assert (
        "(type 'dcp_benchmark') does not match the training configuration: measured with "
        'DPxCP=16, got 8'
    ) in str(caught.value)
    # A failure to build the model from the file's own inputs keeps the plain label.
    monkeypatch.setattr(models, 'model_class', lambda name: Unreadable)
    with pytest.raises(ValueError, match='parameter file is malformed') as caught:
        prefetch.runtime_spec(config, 8)
    assert 'training configuration' not in str(caught.value)


def test_sequence_cap_message_says_where_the_cap_comes_from():
    geometry = RuntimePlanGeometry(8, 1024, 1, (8,), 16, 3, True)
    rows = [[[0, 1, 2, 3]] * 8]
    with pytest.raises(RuntimeScheduleError) as caught:
        validate_rows(rows, {index: 16 for index in range(4)}, geometry, 8)
    assert (
        'packs 4 sequences, cap is 3 (thd_max_packed_sequences, less one entry when a dummy '
        "tail sequence is appended, and at most the cost models' 64 sequences)"
    ) in str(caught.value)


# --------------------------------------------------------------------------------------
# The default heuristic scores compute only: its two communication options are gone
# --------------------------------------------------------------------------------------
def test_removed_heuristic_options_are_not_config_fields():
    import dataclasses

    from megatron.core.model_parallel_config import ModelParallelConfig

    names = {field.name for field in dataclasses.fields(ModelParallelConfig)}
    removed = {'dynamic_cp_nvlink_domain_size', 'dynamic_cp_communication_cost'}
    assert not removed & names
    dynamic = dict(dynamic_context_parallel=True, max_seqlen_per_dp_cp_rank=1024)
    for name in sorted(removed):
        with pytest.raises(TypeError, match=name):
            ModelParallelConfig(**dynamic, **{name: 4})
    assert ModelParallelConfig(dynamic_cp_runtime_config='/data/runtime.json', **dynamic)


# --------------------------------------------------------------------------------------
# Ordinary GQA: DPxCP above 64 is outside the attention parameters
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize('world,cap', [(128, 64), (96, 32), (72, 8)])
def test_ordinary_gqa_rejects_dp_cp_above_64_on_every_rank(tmp_path, monkeypatch, world, cap):
    path = _runtime_config(tmp_path, dict(_BENCHMARK, max_cp_size=cap))
    # No cost model is built for it: every rank reports this from the spec builder.
    monkeypatch.setattr(
        dcp_runtime_worker, '_initialize_planner', lambda spec: pytest.fail('planner was built')
    )
    with pytest.raises(ValueError) as caught:
        prefetch.runtime_spec(_gqa_training_config(path), world)
    assert str(caught.value) == (
        f"Runtime configuration {str(path.resolve())!r} (type 'dcp_benchmark') does not match "
        "the training configuration: the 'dcp_benchmark' cost model has ordinary-GQA attention "
        'parameters for DPxCP (data-parallel size times context-parallel size) up to 64; this '
        f'run has DPxCP={world}'
    )


def test_ordinary_gqa_accepts_dp_cp_64(tmp_path):
    spec = prefetch.runtime_spec(_gqa_training_config(_runtime_config(tmp_path, _BENCHMARK)), 64)
    assert spec['geometry']['allowed_cp_sizes'] == (1, 2, 4, 8, 16, 32, 64)
    assert BenchmarkCostModel.max_dp_cp_size == 64


# --------------------------------------------------------------------------------------
# Ordinary GQA on DPxCP=64: the predictor prices at most 32 sequences in a CP64 pack
# --------------------------------------------------------------------------------------
def test_gqa_dp_cp_64_plans_a_batch_whose_cp64_pack_would_exceed_the_priced_sequences(tmp_path):
    """One long sample plus 40 short ones: start-up passed, then this batch stopped the run."""
    path = _runtime_config(tmp_path, dict(_BENCHMARK, hardware='gb200'))
    # Only the full group, so that this test prices a handful of packs (seconds each).
    config = _gqa_training_config(path, min_dynamic_context_parallel_size=64)
    planner = dcp_runtime_worker._initialize_planner(prefetch.runtime_spec(config, 64))
    model = planner.model
    # The predictor's own limit, which no table states: 33 sequences in a CP64 pack.
    with pytest.raises(ValueError, match='maximum_logical_tokens must be a positive integer'):
        model.program_costs((172104,) + (2600,) * 32, 64, ranks=list(range(64)))
    # The model tells the planner how many it prices ...
    assert model.pack_limits(64) == PackLimits(max_sequences=32)
    assert planner.sequence_limit(64) == 32
    # ... the start-up probe prices that 32-sequence pack, not only a single sequence ...
    assert dcp_runtime_worker.coverage_gaps(planner) == {}
    # ... and the planner never fills a CP64 pack beyond it. The seed is what the default
    # heuristic may build, a valid CP64 group of 40 sequences: it cannot be priced, so it
    # is dropped and the batch is planned all the same.
    lengths = [172104] + [2600] * 40
    padded = {i: (n + 127) // 128 * 128 for i, n in enumerate(lengths)}
    plan = planner.plan_padded(
        padded,
        logical_lengths=dict(enumerate(lengths)),
        incumbent_rows=[[[0]] * 64, [list(range(1, 41))] * 64],
    )
    groups = validate_rows(plan.rows, padded, planner.geometry, 64)
    assert sorted(len(ids) for row in groups for _, _, ids in row) == [9, 32]
    assert plan.predicted_ms > 0
    statistics = planner.last_statistics
    assert statistics['model_sequence_limits'] == {64: 32}
    assert not statistics['selected_incumbent']
    assert any(
        'cost model rejected CP64 pack' in text for text in statistics['rejected_candidates']
    )


# --------------------------------------------------------------------------------------
# The planner I/O thread never keeps the process from exiting; a silent worker is reported
# --------------------------------------------------------------------------------------
def test_planner_io_runs_on_a_daemon_thread_that_interpreter_exit_does_not_join():
    from concurrent.futures import thread as pool_threads

    executor = prefetch._DaemonExecutor('dcp-planner-io-test')
    try:
        assert executor.submit(lambda a, b: a + b, 2, 3).result(timeout=30) == 5
        failure = executor.submit(lambda: 1 // 0)
        with pytest.raises(ZeroDivisionError):
            failure.result(timeout=30)
        assert executor._thread.daemon
        # concurrent.futures joins the threads it registered before atexit callbacks run.
        assert executor._thread not in pool_threads._threads_queues
    finally:
        executor.shutdown()
    assert not executor._thread.is_alive()
    with pytest.raises(RuntimeError, match='after shutdown'):
        executor.submit(print)


def test_executor_shutdown_drops_queued_calls_and_does_not_wait_forever(monkeypatch):
    monkeypatch.setattr(prefetch._DaemonExecutor, 'join_timeout_s', 0.2)
    executor = prefetch._DaemonExecutor('dcp-planner-io-test')
    release = threading.Event()
    running = executor.submit(release.wait)
    queued = executor.submit(lambda: 'never runs')
    _wait_until(running.running)
    begin = time.monotonic()
    executor.shutdown(wait=True, cancel_futures=True)
    assert time.monotonic() - begin < 30  # The running call is abandoned, not awaited.
    assert queued.cancelled() and not running.done()
    release.set()
    assert running.result(timeout=30) is True


def _silent_worker():
    """A worker process that reads requests and never answers."""
    worker = prefetch._CpuPlanner.__new__(prefetch._CpuPlanner)
    worker.process = subprocess.Popen(
        [sys.executable, '-c', 'import sys, time\nfor line in sys.stdin:\n    time.sleep(600)'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    return worker


def _wait_until(condition, seconds=30):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert condition()


def test_close_with_a_request_in_flight_ends_the_worker_at_once(make_prefetcher, monkeypatch):
    # The graceful wait would take this long; an abandoned request must not wait for it.
    monkeypatch.setattr(prefetch._CpuPlanner, 'close_timeout_s', 300.0)
    owner = make_prefetcher([1, 2])
    owner.worker = worker = _silent_worker()
    try:
        future = owner.executor.submit(worker.solve, [1000], [1008], None)
        _wait_until(lambda: worker.in_flight)
        begin = time.monotonic()
        owner.close()  # What atexit calls while the reply is still outstanding.
        assert time.monotonic() - begin < 30
        assert worker.process.poll() is not None
        with pytest.raises(RuntimeError, match='CPU DCP worker exited unexpectedly'):
            future.result(timeout=30)
        assert not worker.in_flight
    finally:
        worker.process.kill()
        worker.process.wait()


def test_an_idle_worker_is_still_closed_gracefully(make_prefetcher):
    owner = make_prefetcher([1, 2])
    owner.worker = worker = prefetch._CpuPlanner.__new__(prefetch._CpuPlanner)
    worker.process = subprocess.Popen(
        [sys.executable, '-c', 'import sys\nfor line in sys.stdin:\n    pass\nsys.exit(7)'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    owner.close()
    assert worker.process.returncode == 7  # It saw the end of its request stream and left.


@pytest.mark.parametrize('asynchronous', [False, True])
def test_a_worker_that_never_answers_stops_every_rank_with_one_error(
    make_prefetcher, monkeypatch, asynchronous
):
    monkeypatch.setattr(prefetch, 'PLANNER_REPLY_TIMEOUT_S', 0.3)
    owner = make_prefetcher([1, 2, 3, 4], asynchronous=asynchronous)
    sent = _record_broadcasts(monkeypatch)
    _consume(owner, 2)
    release = threading.Event()
    monkeypatch.setattr(owner.worker, 'solve', lambda *args: release.wait() and _solve(*args))
    owner.prefetch(2)
    try:
        begin = time.monotonic()
        with pytest.raises(RuntimeError) as caught:
            _consume(owner, 2)
        assert time.monotonic() - begin < 30
    finally:
        release.set()
    # The leader gave up waiting and broadcast the error, so no rank is left in the collective.
    assert len(sent) == 2 and sent[1] == dict(
        error=(
            'The CPU planner did not answer within 0.3 s. The DPxCP leader stops waiting so '
            'that every rank raises this error instead of blocking in the schedule broadcast'
        )
    )
    assert str(caught.value) == 'CPU DCP planner failed:\n' + sent[1]['error']


def test_the_reply_limit_names_the_worker_process(make_prefetcher, monkeypatch):
    monkeypatch.setattr(prefetch, 'PLANNER_REPLY_TIMEOUT_S', 0.3)
    owner = make_prefetcher([1, 2])
    owner.worker = worker = _silent_worker()
    try:
        with pytest.raises(RuntimeError) as caught:
            _consume(owner, 2)
        assert (
            f'The CPU planner did not answer within 0.3 s (worker process {worker.process.pid} '
            'is still running)'
        ) in str(caught.value)
    finally:
        owner.close()
        worker.process.kill()
        worker.process.wait()


def test_the_shipped_reply_limit_is_generous():
    # Plans take seconds at most; the limit only has to recognise a worker that is stuck.
    assert prefetch.PLANNER_REPLY_TIMEOUT_S == 600.0
