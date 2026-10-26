# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Data-order and CPU-worker contracts, independent of model size."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets import data_schedule, data_schedule_utils
from megatron.core.datasets import dcp_schedule_prefetch as prefetch


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
        owner.executor = ThreadPoolExecutor(max_workers=1)
        owner.worker = SimpleNamespace(solve=_solve, close=lambda: None)
        owner.baseline = lambda lengths: [[list(range(len(lengths)))] * 8]
        owners.append(owner)
        return owner

    yield create
    for owner in owners:
        owner.close()


def test_lookahead_does_not_consume_rerun_data(make_prefetcher):
    owner = make_prefetcher([1, 2, 3, 4, 5, 6])
    assert owner.consume(2)[0][0] == [1, 2]
    owner.prefetch(2)
    assert owner.iterator.saved == [1, 2]
    assert len(owner.buffer.items) == 2
    assert owner.consume(2)[0][0] == [3, 4]
    owner.prefetch(2)
    assert owner.consume(2)[0][0] == [5, 6]
    assert owner.iterator.saved == [1, 2, 3, 4, 5, 6]


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
    monkeypatch.setattr(data_schedule, 'mtp_on_this_rank', lambda *args, **kwargs: False)
    monkeypatch.setattr(data_schedule, 'reroute_samples_to_dcp_ranks', lambda *args: [])
    monkeypatch.setattr(data_schedule, 'build_packed_microbatches', lambda *args, **kwargs: [])
    monkeypatch.setattr(data_schedule, 'broadcast_scalars', lambda values, group, device: values)
    monkeypatch.setattr(data_schedule, 'create_data_iterator', lambda *args: iter([]))
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
    prepared, _ = owner.consume(raw_count)
    assert len(prepared[0]) == raw_count
    assert calls == [list(range(1, 5)), list(range(5, 9))]
    assert owner.last_telemetry['ready_when_consumed']
    assert owner.iterator.saved == list(range(1, 9))


@pytest.mark.parametrize('next_count', [1, 3])
def test_batch_size_change_keeps_every_sample(make_prefetcher, next_count):
    owner = make_prefetcher(list(range(1, 10)))
    owner.consume(2)
    owner.prefetch(2)
    assert owner.consume(next_count)[0][0] == list(range(3, 3 + next_count))
    assert next(owner.iterator) == 3 + next_count


def test_replay_reads_current_batch_not_prefetched_batch(make_prefetcher):
    owner = make_prefetcher([1, 2, 3, 4])
    owner.consume(2)
    owner.prefetch(2)
    owner.iterator.replaying = True
    assert owner.consume(2)[0][0] == [1, 2]
    owner.iterator.replaying = False
    assert owner.consume(2)[0][0] == [3, 4]


def test_sync_control_does_not_read_ahead(make_prefetcher):
    owner = make_prefetcher([1, 2, 3, 4], asynchronous=False)
    owner.consume(2)
    owner.prefetch(2)
    assert owner.pending is None and not owner.buffer.items
    assert owner.consume(2)[0][0] == [3, 4]


def test_short_tail_is_retained(make_prefetcher):
    owner = make_prefetcher([1, 2, 3])
    owner.consume(2)
    owner.prefetch(2)
    assert owner.pending is None
    assert owner.consume(1)[0][0] == [3]


def test_worker_error_is_broadcast_as_an_error(make_prefetcher, monkeypatch):
    def fail(*args):
        raise ValueError('bad cost model')

    owner = make_prefetcher([1, 2])
    monkeypatch.setattr(owner.worker, 'solve', fail)
    with pytest.raises(RuntimeError, match='bad cost model'):
        owner.consume(2)


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


def test_planner_process_stays_cpu_only():
    from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler

    config = SimpleNamespace(max_seqlen_per_dp_cp_rank=1024, min_dynamic_context_parallel_size=1)
    spec = prefetch.runtime_spec(config, 8)
    baseline = DefaultDynamicCPScheduler(1024, 8, 1, None, max_num_seqs=64, min_cp_size=1)
    worker = prefetch._CpuPlanner(spec)
    try:
        result = worker.solve(
            [2600, 4500], [2608, 4512], baseline.get_groups_and_subsamples([(0, 2608), (1, 4512)])
        )
        again = worker.solve(
            [1000, 6000, 500],
            [1008, 6000, 512],
            baseline.get_groups_and_subsamples([(0, 1008), (1, 6000), (2, 512)]),
        )
    finally:
        worker.close()
    assert len(result['rows'][0]) == 8
    assert result['worker_pid'] == again['worker_pid']
    assert not result['worker_cuda_initialized'] and not again['worker_cuda_initialized']
    assert (
        not result['worker_distributed_initialized'] and not again['worker_distributed_initialized']
    )


def test_worker_reports_initialization_error_after_process_exit():
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=1024,
        min_dynamic_context_parallel_size=1,
        dynamic_cp_runtime_model='calibrated_attention',
        dynamic_cp_runtime_model_options='{"allow_provisional":true,"query_heads":7}',
    )
    worker = prefetch._CpuPlanner(prefetch.runtime_spec(config, 8))
    try:
        worker.process.wait(timeout=30)
        with pytest.raises(RuntimeError, match='No forward coefficient set for geometry'):
            worker.solve([1000], [1008], [[[0]] * 8])
    finally:
        worker.close()


def test_runtime_respects_real_sequence_cap_after_dummy_tail():
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=1024,
        min_dynamic_context_parallel_size=1,
        thd_max_packed_sequences=4,
    )
    assert prefetch.runtime_spec(config, 8, max_num_seqs=3)['geometry']['max_group_seqs'] == 3


@pytest.mark.parametrize(
    'world,minimum,allowed', [(8, 1, (1, 2, 4, 8)), (32, 8, (8, 16, 32)), (12, 2, (2, 4))]
)
def test_runtime_spec_matches_existing_classic_cp_groups(world, minimum, allowed):
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=1024, min_dynamic_context_parallel_size=minimum
    )
    spec = prefetch.runtime_spec(config, world)
    assert spec['geometry']['allowed_cp_sizes'] == allowed
    assert spec['geometry']['align_group_starts']
    assert not spec['force_powers']


def test_runtime_spec_rejects_unsupported_classic_minimum():
    config = SimpleNamespace(max_seqlen_per_dp_cp_rank=1024, min_dynamic_context_parallel_size=3)
    with pytest.raises(ValueError, match='power-of-two minimum'):
        prefetch.runtime_spec(config, 8)


@pytest.mark.parametrize('minimum', [1, 3, 5])
def test_native_runtime_relaxes_widths_and_group_starts(minimum):
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=1024,
        min_dynamic_context_parallel_size=minimum,
        use_native_cp_transport=True,
    )
    spec = prefetch.runtime_spec(config, 8)
    assert spec['geometry']['allowed_cp_sizes'] == tuple(range(minimum, 9))
    assert not spec['geometry']['align_group_starts']
    assert not spec['force_powers']


def test_wrap_activates_runtime_with_classic_process_groups(monkeypatch):
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=1024,
        min_dynamic_context_parallel_size=1,
        sequence_parallel=False,
        sequence_packing_scheduler='default_dynamic_cp',
        virtual_pipeline_model_parallel_size=None,
        dynamic_cp_runtime_model='module:model',
    )
    iterator = SimpleNamespace()
    single = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    parent = SimpleNamespace(rank=lambda: 0, size=lambda: 8)
    groups = SimpleNamespace(dp=single, tp=single, pp=single, dp_cp=parent)
    created = []

    def create(*args):
        runtime = SimpleNamespace(spec=args[3])
        created.append(runtime)
        return runtime

    def run(self, *args):
        assert self.runtime is iterator._dcp_schedule_prefetcher
        assert self.prefetch_next
        assert not self.allow_arbitrary_group_starts
        return iter([]), 1, 1, 1

    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(prefetch, 'DcpSchedulePrefetcher', create)
    monkeypatch.setattr(data_schedule.DefaultDynamicCPScheduler, 'run', run)
    for _ in range(2):
        data_schedule.wrap_data_iterator(iterator, config, 1, groups, prefetch_next=True)
    assert len(created) == 1


def test_runtime_options_are_injectable_and_keep_contiguous(tmp_path):
    options = tmp_path / 'solver.json'
    options.write_text('{"time_budget_s": 0.25, "max_scored_candidates": 2}')
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=16384,
        min_dynamic_context_parallel_size=1,
        cp_partition_mode='contiguous',
        thd_tail_padding_policy='extend_last',
        dynamic_cp_runtime_solver='test_planner:create',
        dynamic_cp_runtime_solver_options=str(options),
        dynamic_cp_runtime_baseline='pinned_dev_baseline:create',
    )
    spec = prefetch.runtime_spec(config, 32, max_num_seqs=64)
    assert spec['geometry']['max_group_seqs'] == 64
    assert spec['solver'] == 'test_planner:create'
    assert spec['solver_options'] == {'time_budget_s': 0.25, 'max_scored_candidates': 2}
    assert spec['baseline'] == 'pinned_dev_baseline:create'
    assert (
        config.cp_partition_mode == 'contiguous' and config.thd_tail_padding_policy == 'extend_last'
    )


def test_runtime_cli_supports_both_async_directions():
    import argparse
    import dataclasses

    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.training.argument_utils import ArgumentGroupFactory

    fields = dataclasses.fields(ModelParallelConfig)
    factory = ArgumentGroupFactory(
        ModelParallelConfig,
        exclude=[f.name for f in fields if not f.name.startswith('dynamic_cp_runtime_')],
    )
    parser = argparse.ArgumentParser()
    factory.build_group(parser)
    assert parser.parse_args([]).dynamic_cp_runtime_async
    assert parser.parse_args(['--dynamic-cp-runtime-async']).dynamic_cp_runtime_async
    args = parser.parse_args(
        [
            '--no-dynamic-cp-runtime-async',
            '--dynamic-cp-runtime-model',
            'module:model',
            '--dynamic-cp-runtime-solver-options',
            '{"time_budget_s":0.8}',
            '--dynamic-cp-runtime-force-powers',
            '--dynamic-cp-runtime-trace-dir',
            '/tmp/dcp-telemetry',
        ]
    )
    assert not args.dynamic_cp_runtime_async and args.dynamic_cp_runtime_force_powers
    assert args.dynamic_cp_runtime_model == 'module:model'
    assert args.dynamic_cp_runtime_trace_dir == '/tmp/dcp-telemetry'


def test_worker_accepts_custom_planner_factory(monkeypatch):
    import sys

    from megatron.core.datasets.dcp_runtime_worker import _initialize_planner

    expected = object()
    received = {}

    def create(geometry, model, **options):
        received.update(world_size=geometry.world_size, model_name=model.name, options=options)
        return expected

    monkeypatch.setitem(sys.modules, 'test_dcp_cpu_factory', SimpleNamespace(create=create))
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=1024,
        min_dynamic_context_parallel_size=1,
        dynamic_cp_runtime_solver='test_dcp_cpu_factory:create',
        dynamic_cp_runtime_solver_options='{"time_budget_s":0.25}',
    )
    assert _initialize_planner(prefetch.runtime_spec(config, 8)) is expected
    assert received == {
        'world_size': 8,
        'model_name': 'megatron_quadratic_work',
        'options': {'time_budget_s': 0.25},
    }
