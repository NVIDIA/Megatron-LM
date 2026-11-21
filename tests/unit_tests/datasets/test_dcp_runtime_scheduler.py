# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.datasets import dcp_runtime_worker as worker
from megatron.core.datasets import dcp_schedule_prefetch as prefetch
from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
from megatron.core.datasets.dcp_runtime_scheduler.models.base import QuadraticCostModel
from megatron.core.datasets.dcp_runtime_scheduler.runtime import dynamic_usp, lightweight, planner
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import RuntimeScheduleError
from megatron.core.datasets.dcp_runtime_scheduler.runtime.timeline import TimelineModel, _freeze_op

_LIGHT = lightweight.LightweightRuntimePlanner
# The other solver: the one planner class that the ``dynamic_usp`` module defines.
[_USP] = [c for c in _LIGHT.__base__.__subclasses__() if c.__module__ == dynamic_usp.__name__]
both_solvers = pytest.mark.parametrize('solver', [_LIGHT, _USP])
_TIMES = {'non_attention': {'forward_ms_per_layer': 0.5, 'backward_ms_per_layer': 1.0}}
_GQA = dict(_TIMES, hardware='h100', query_heads=32, kv_heads=4, head_dim=128, layer_count=48)
_PACKAGED = [  # scheduler type, cost model, its options, DPxCP, EP, CP sizes
    ('dcp_benchmark', 'dcp_benchmark', _GQA, 8, 1, (1, 2, 4, 8)),
    ('dcp_benchmark', 'qwen35_full', dict(_TIMES, hardware='gb200'), 32, 32, (1, 2, 4, 8, 16, 32)),
    ('dynamic_usp', 'usp_calib_v1', dict(allow_provisional=True), 32, 1, (8, 16, 32)),
]
_SAMPLES = (4, 3, 3, 2, 2, 1)  # lengths of six samples, in capacities of the smallest CP group
_PLANS = [  # their plan as (CP, samples) per group of every microbatch, and its predicted ms
    ([[(4, [0]), (4, [1, 5])], [(4, [3, 4]), (4, [2])]], 599.296343744),
    ([[(4, [0]), (4, [1, 5]), (8, [2]), (16, [3, 4])]], 1546.75686155),
    ([[(16, [4]), (16, [3])], [(32, [1, 5])], [(32, [0])], [(32, [2])]], 62563.425239),
]
_FROZEN = list(zip(_PACKAGED, _PLANS, strict=True))  # one plan for every packaged scheduler


def _geometry(world=8, capacity=1024, allowed=(1, 2, 4, 8), cap=64, max_cp=None):
    """Size-aligned groups, as the runtime forms them; the dataset pads to 2 x DPxCP."""
    return planner.RuntimePlanGeometry(world, capacity, 2, allowed, 2 * world, cap, True, max_cp)


def _solve(monkeypatch, scheduler, lengths, default):
    """One request to the handler of the CPU worker, without the Torch state of this process."""
    monkeypatch.setattr(worker, 'sys', SimpleNamespace(modules={}))
    return worker._solve(scheduler, {}, list(lengths), list(lengths), default)


@pytest.mark.parametrize('packaged,frozen', _FROZEN, ids=[p[1] for p in _PACKAGED])
def test_every_packaged_scheduler_plans_a_frozen_batch_also_in_the_cpu_worker(packaged, frozen):
    kind, name, options, world, ep, allowed = packaged
    geometry = planner.RuntimePlanGeometry(world, 512 * world, ep, allowed, 64, 64, True)
    spec = dict(type=kind, model=name, model_options=options, geometry=asdict(geometry))
    lengths = [512 * world * allowed[0] * count for count in _SAMPLES]
    plan = worker._initialize_planner(spec).plan_padded(dict(enumerate(lengths)))
    assert [[(cp, ids) for _, cp, ids in row] for row in plan.groups] == frozen[0]
    assert plan.predicted_ms == pytest.approx(frozen[1], rel=1e-9)
    process = prefetch._CpuPlanner(spec)  # the process that plans during training
    try:
        with pytest.raises(RuntimeScheduleError, match='sample 0: seq_len='):
            process.solve([lengths[0] * 16], [lengths[0] * 16], None)  # fits no CP size
        result = process.solve(lengths, lengths, None)
    finally:
        process.close()
    assert (result['rows'], result['predicted_ms']) == (plan.rows, plan.predicted_ms)
    with pytest.raises(ValueError):  # at construction or when asked for a pack of 65 sequences
        spec['geometry']['max_group_seqs'] = 65
        model = worker._initialize_planner(spec).model
        model.forward((64,) * 65, allowed[-1], ranks=tuple(range(allowed[-1])))


@both_solvers
def test_a_batch_without_a_layout_raises_the_schedule_error_with_the_limits(solver):
    geometry = _geometry(capacity=100, max_cp=2)
    # A CP2 group holds 200 tokens; the dataset stores 193 tokens as 208, a multiple of 16.
    with pytest.raises(RuntimeScheduleError, match='sample 7: seq_len=193 .*max_cp_size=2'):
        solver(geometry, QuadraticCostModel(geometry)).plan({7: 193})
    # Seven samples cannot fill eight CP1 groups, whatever the capacity.
    geometry = _geometry(capacity=4096, max_cp=1)
    with pytest.raises(RuntimeScheduleError, match='no layout gives each of the 8 DPxCP ranks'):
        solver(geometry, QuadraticCostModel(geometry)).plan(dict.fromkeys(range(7), 1024))


@both_solvers
@pytest.mark.parametrize(
    'maximum,values,microbatches',
    [(2, [1023, 2047, 4095, 6143, 7167], 1), (8, [4096] * 3, 1), (1, [2000] * 17, 2)],
)
def test_every_rank_gets_a_sample_whatever_the_cp_cap(solver, maximum, values, microbatches):
    geometry = _geometry(capacity=4096, max_cp=maximum)
    lengths = dict(enumerate(values))
    plan = solver(geometry, QuadraticCostModel(geometry)).plan(lengths)
    groups = planner.validate_rows(plan.rows, lengths, geometry, 1)
    assert plan.microbatches == microbatches and plan.predicted_ms > 0
    assert all(cp <= maximum for row in groups for _, cp, _ in row)


_G12, _G248 = _geometry(12, 4096, (1, 2, 4)), _geometry(allowed=(2, 4, 8))
_CONTRACT, _DOMAIN = 'violates_runtime_contract', 'outside_predictor_cp_domain'
_INVALID_DEFAULT_LAYOUTS = [  # geometry, lengths, layout, why the trace says it was discarded
    # More sequences in a group than the cost models price (64) or than the user allows (2).
    (_geometry(allowed=(8,)), [16] * 65, [[list(range(65))] * 8], _CONTRACT),
    (_geometry(cap=2), [16] * 24, [[[r, r + 8, r + 16] for r in range(8)]], _CONTRACT),
    # CP 8 is not a size of a run with DPxCP=12, and is above a max_cp_size of 2.
    (_G12, [480] * 12, [[list(range(6))] * 8 + [list(range(6, 12))] * 4], _DOMAIN),
    (_geometry(max_cp=2), [96] * 8, [[list(range(8))] * 8], 'exceeds_max_cp_size'),
    (_G248, [96] * 3, [[[0]] * 2 + [[1]] * 4 + [[2]] * 2], _CONTRACT),  # CP4 starts at rank 2
    (_geometry(), [1040] * 8, [[[r] for r in range(8)]], _CONTRACT),  # a rank holds 1024 tokens
    (_geometry(), [96] * 8, [[[0]] * 8], _CONTRACT),  # seven samples are missing
    (_geometry(), [96] * 8, [[[r + 1] for r in range(8)]], _CONTRACT),  # there is no sample 8
    (_geometry(), [96] * 8, [[[r] for r in range(8)], [[]] * 8], _CONTRACT),  # eight idle ranks
    (_geometry(), [96] * 8, [[list(range(8))] * 4], _CONTRACT),  # a microbatch of four ranks
]


@both_solvers
@pytest.mark.parametrize('geometry,lengths,layout,reason', _INVALID_DEFAULT_LAYOUTS)
def test_worker_discards_an_invalid_default_layout_and_still_plans(
    solver, geometry, lengths, layout, reason, monkeypatch
):
    result = _solve(monkeypatch, solver(geometry, QuadraticCostModel(geometry)), lengths, layout)
    # The plan satisfies the whole contract, the violated constraint included.
    planner.validate_rows(result['rows'], dict(enumerate(lengths)), geometry, 1)
    assert result['statistics']['incumbent_rejection'] == 'default_incumbent_' + reason


@both_solvers
@pytest.mark.parametrize('default', [None, [[[rank] for rank in range(8)]]])
def test_a_search_without_a_layout_keeps_the_default_layout_or_falls_back_on_capacity(
    solver, default, monkeypatch
):
    geometry = _geometry()
    scheduler = solver(geometry, QuadraticCostModel(geometry))
    if solver is _LIGHT:
        for family in ('_pack', '_global_pack', '_cost_pack'):
            monkeypatch.setattr(scheduler, family, Mock(return_value=None))
    else:
        failure = dynamic_usp.InfeasibleLayout('no candidate count produced a layout')
        monkeypatch.setattr(scheduler.scheduler, 'solve', Mock(side_effect=failure))
    result = _solve(monkeypatch, scheduler, [1024] * 8, default)
    # Without a default layout: one full group, the widest that holds every sample.
    assert result['rows'] == (default or [[list(range(8))] * 8])
    stats = result['statistics']
    assert (stats['selected_incumbent'], stats['capacity_fallback']) == (bool(default), not default)


@pytest.mark.parametrize('default,scored', [(None, 1), ([[[rank] for rank in range(8)]], 2)])
def test_a_spent_time_budget_still_scores_the_default_layout_and_one_candidate(default, scored):
    geometry = _geometry()
    options = lightweight.LightweightOptions(time_budget_s=1e-9)
    scheduler = _LIGHT(geometry, QuadraticCostModel(geometry), options)
    plan = scheduler.plan_padded(dict.fromkeys(range(8), 1024), incumbent_rows=default)
    stats = scheduler.last_statistics
    assert (stats['generated_candidates'], stats['scored_candidates']) == (4, scored)
    assert stats['budget_overrun'] and plan.rows == [[list(range(8))] * 8]


@pytest.mark.parametrize('world', [4, 8, 16, 32])
def test_the_default_scheduler_proposes_a_layout_the_runtime_scheduler_accepts(world, monkeypatch):
    # Lengths from a quarter of a rank to the whole DPxCP: the layout has groups of every CP size.
    lengths = [64 << sample % (world.bit_length() + 2) for sample in range(16)]
    default = DefaultDynamicCPScheduler(256, world, 1, None)
    layout = default.get_groups_and_subsamples(list(enumerate(lengths)))
    geometry = _geometry(world, 256, tuple(cp for cp in (1, 2, 4, 8, 16, 32) if cp <= world))
    result = _solve(monkeypatch, _LIGHT(geometry, QuadraticCostModel(geometry)), lengths, layout)
    assert 'incumbent_rejection' not in result['statistics']


def test_timeline_step_times_of_mixed_cp_layouts_are_frozen():
    # Groups of CP 1, 1, 2 and 4 on eight ranks: (start, CP, attention ms, rest ms).
    sizes = ((0, 1, 1.0, 2.5), (1, 1, 2.0, 3.0), (2, 2, 3.0, 4.0), (4, 4, 4.0, 6.0))
    groups = [
        (start, cp, f, 2 * f, dict(att_f=f, att_b=2 * f, rest_f=rest, rest_b=rest / 2))
        for start, cp, f, rest in sizes
    ]
    mean = ((('cp', 'att_f'), ('ep_mean', 'rest_f')), (('ep_mean', 'rest_b'), ('cp', 'att_b')))
    timeline = TimelineModel(8, 8, 1, iteration_tail_ms=0.25, program=mean)
    # Forward: slowest attention 4 + rank-weighted EP mean 4.6875; backward: 2.34375 + 8.
    # The second microbatch starts from the unequal completion times of the first.
    assert [timeline.evaluate([groups] * count) for count in (1, 2)] == [19.28125, 38.3125]
    # A second EP block of eight ranks. A block starts at its own slowest rank; all ranks
    # wait for each other once, after the forward pass of the first microbatch.
    blocks = groups + [(8, 8, 0.0, 0.0, dict(att_b=0.5, rest_f=12.0))]
    fixed = ((('ep_mean', 'rest_f'), ('ep', 0.5)), (('cp', 'att_b'), ('local', 0.25)))
    timeline = TimelineModel(16, 8, 1, program=fixed)
    state = [7.0] + [0.0] * 7 + [3.0] + [0.0] * 7
    assert timeline.makespan(timeline.advance(state, blocks, False)) == 20.4375
    assert [timeline.evaluate([blocks] * count) for count in (1, 2)] == [20.75, 34.1875]
    with pytest.raises(ValueError, match='explicit mean'):
        _freeze_op({'sync': 'ep', 'cost': 'rest'})
