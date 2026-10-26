# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Contracts of the in-tree CPU runtime and its frozen timing models."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatron.core.datasets import dcp_schedule_prefetch as prefetch
from megatron.core.datasets.dcp_runtime_scheduler.models import (
    AttentionCostModel,
    CorePredictor,
    make_model,
    qwen35_new_core_model,
)
from megatron.core.datasets.dcp_runtime_scheduler.models.attention.dispatch import (
    dispatch_backend,
    dispatch_chain,
)
from megatron.core.datasets.dcp_runtime_scheduler.models.base import (
    DcpCostModel,
    QuadraticCostModel,
)
from megatron.core.datasets.dcp_runtime_scheduler.models.cache import PackCostModel
from megatron.core.datasets.dcp_runtime_scheduler.runtime.dynamic_usp import (
    DynamicUSPRuntimePlanner,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.lightweight import (
    LightweightOptions,
    LightweightRuntimePlanner,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import (
    RuntimePlanGeometry,
    RuntimePlanner,
    RuntimeScheduleError,
    validate_rows,
)


def _geometry(world=8, capacity=1024, ep=2):
    return RuntimePlanGeometry(world, capacity, ep, tuple(range(1, world + 1)), 2 * world, 64)


@pytest.mark.parametrize("planner_type", [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
@pytest.mark.parametrize("model_name", ["calibrated_attention", "qwen35_gb200", "usp_calib_v1"])
def test_models_and_schedulers_share_one_injected_contract(planner_type, model_name):
    geometry = RuntimePlanGeometry(32, 16384, 4, (8, 16, 32), 64, 64)
    options = {"allow_provisional": True} if model_name != "qwen35_gb200" else {}
    model = make_model(model_name, geometry, **options)
    planner = planner_type(geometry, model)
    assert isinstance(model, DcpCostModel)
    assert isinstance(planner, RuntimePlanner)
    assert planner.model is model
    assert not issubclass(DynamicUSPRuntimePlanner, LightweightRuntimePlanner)
    plan = planner.plan_padded({0: 2304, 1: 1408}, logical_lengths={0: 2300, 1: 1400})
    validate_rows(plan.rows, {0: 2304, 1: 1408}, geometry, 8)
    assert plan.predicted_ms > 0


@pytest.mark.parametrize('cp', [1, 2, 3, 5, 6, 7, 8])
def test_every_cp_width_and_unaligned_tail_are_legal(cp):
    geometry = _geometry()
    lengths = {0: 511}
    rows = [[[0]] * cp + [[1]] * (8 - cp)] if cp < 8 else [[[0]] * 8]
    if cp < 8:
        lengths[1] = 501
    groups = validate_rows(rows, lengths, geometry, 1)
    assert groups[0][0][:2] == (0, cp)
    if cp < 8:
        assert groups[0][1][:2] == (cp, 8 - cp)


def test_classic_geometry_rejects_cp4_starting_at_rank2():
    geometry = RuntimePlanGeometry(8, 1024, 2, (2, 4, 8), 16, 64, True)
    rows = [[[0]] * 2 + [[1]] * 4 + [[2]] * 2]
    with pytest.raises(RuntimeScheduleError, match='size-aligned start'):
        validate_rows(rows, {0: 256, 1: 256, 2: 256}, geometry, 2)


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
@pytest.mark.parametrize('world', [4, 8, 16, 32])
def test_solvers_search_only_supported_classic_intervals(planner_type, world):
    allowed = tuple(cp for cp in range(1, world + 1) if cp & (cp - 1) == 0 and world % cp == 0)
    geometry = RuntimePlanGeometry(world, 256, 2, allowed, 2 * world, 64, True)
    planner = planner_type(geometry, QuadraticCostModel(geometry))
    rng = random.Random(20261026)
    for _ in range(8):
        # Several feasible packs, including widths that compete on the cost model.
        lengths = {sid: rng.randint(1, max(allowed) * 128) for sid in range(2 * world)}
        plan = planner.plan(lengths)
        validate_rows(plan.rows, lengths, geometry, 1)
        assert all(start % cp == 0 and cp in allowed for row in plan.groups for start, cp, _ in row)
        assert not planner.last_statistics['selected_incumbent']
        if planner_type is LightweightRuntimePlanner:
            assert planner.last_statistics['scored_candidates'] > 0
        else:
            assert planner.last_statistics['candidates']


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
def test_non_power_domain_uses_only_existing_power_groups(planner_type):
    geometry = RuntimePlanGeometry(12, 256, 2, (1, 2, 4), 24, 64, True)
    lengths = {sid: 240 for sid in range(12)}
    plan = planner_type(geometry, QuadraticCostModel(geometry)).plan(lengths)
    validate_rows(plan.rows, lengths, geometry, 1)
    assert all(cp in (1, 2, 4) and start % cp == 0 for row in plan.groups for start, cp, _ in row)


def test_rotation_scores_only_supported_classic_starts(monkeypatch):
    geometry = RuntimePlanGeometry(8, 1024, 2, (2, 4, 8), 16, 64, True)
    model = QuadraticCostModel(geometry)
    model.placement_invariant = True
    planner = LightweightRuntimePlanner(geometry, model)
    advance = planner.timeline.advance
    placements = []

    def checked_advance(state, groups, first):
        assert all(geometry.supports_group(start, cp) for start, cp, *_ in groups)
        placements.append([(start, cp) for start, cp, *_ in groups])
        return advance(state, groups, first)

    monkeypatch.setattr(
        type(planner.timeline), 'advance', lambda self, *args: checked_advance(*args)
    )
    rounds = [[dict(cp=4, sample_ids=[0]), dict(cp=2, sample_ids=[1]), dict(cp=2, sample_ids=[2])]]
    planner._balance_exact(rounds, {0: 1000, 1: 500, 2: 300})
    assert len(placements) == 2  # The third cyclic rotation would put CP4 at rank 2.


def test_padded_entry_rejects_invalid_inputs_and_keeps_incumbent():
    geometry = _geometry()
    planner = LightweightRuntimePlanner(geometry, QuadraticCostModel(geometry))
    for invalid in ({0: 17}, [(0, 16), (0, 16)], {0: 0}, {-1: 16}):
        with pytest.raises(ValueError):
            planner.plan_padded(invalid)
    incumbent = [[[0, 1]] * 8]
    logical = {0: 2300, 1: 1400}
    plan = planner.plan_padded(
        {0: 2304, 1: 1408}, logical_lengths=logical, incumbent_rows=incumbent
    )
    before = planner.timeline.evaluate(planner._costed([[dict(cp=8, sample_ids=[0, 1])]], logical))
    assert plan.predicted_ms <= before
    with pytest.raises(RuntimeScheduleError):
        validate_rows([[[0]] * 8], logical, geometry, 1)


@pytest.mark.parametrize('model_name', ['calibrated_attention', 'qwen35_gb200'])
def test_frozen_models_include_both_phases_for_odd_cp(model_name):
    geometry = _geometry(world=32, capacity=16384, ep=4)
    model = (
        qwen35_new_core_model()
        if model_name == 'qwen35_gb200'
        else AttentionCostModel(geometry, allow_provisional=True)
    )
    for cp in (1, 3, 5, 32):
        values = [1025, 511]
        ranks = tuple(range(32 - cp, 32))
        assert model.forward(values, cp, ranks=ranks) > 0
        assert model.backward(values, cp, ranks=ranks) > 0
        assert model.program_costs(values, cp, ranks=ranks)
    assert model.metadata()['provisional']


@pytest.mark.parametrize(
    'case', json.loads(Path(__file__).with_name('dcp_frozen_costs.json').read_text())['cases']
)
@pytest.mark.parametrize('kernel_backend', ['python', 'numba'])
def test_scalar_costs_match_the_frozen_reference(case, kernel_backend):
    if kernel_backend == 'numba':
        pytest.importorskip('numba')
    hardware, geometry, cp, lengths, forward_ms, backward_ms = case
    predictor = CorePredictor(
        hardware=hardware,
        query_heads=geometry[0],
        kv_heads=geometry[1],
        head_dim=geometry[2],
        kernel_backend=kernel_backend,
    )
    assert predictor(lengths, cp) == {'forward_ms': forward_ms, 'backward_ms': backward_ms}


def test_prediction_scope_is_not_silently_claimed_as_calibrated():
    with pytest.raises(ValueError, match='allow_provisional'):
        AttentionCostModel(_geometry())
    with pytest.raises(ValueError, match='NVL72'):
        AttentionCostModel(_geometry(world=80), allow_provisional=True)


def test_builtin_rpc_needs_no_external_scheduler_package():
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=1024,
        min_dynamic_context_parallel_size=1,
        dynamic_cp_runtime_model='calibrated_attention',
        dynamic_cp_runtime_model_options='{"allow_provisional":true}',
        dynamic_cp_runtime_solver_options='{"max_scored_candidates":2}',
    )
    worker = prefetch._CpuPlanner(prefetch.runtime_spec(config, 8))
    try:
        result = worker.solve([2300, 1400], [2304, 1408], [[[0, 1]] * 8])
    finally:
        worker.close()
    assert not result['worker_cuda_initialized']
    assert not result['worker_distributed_initialized']
    assert not result['external_scheduler_imported']
    assert 'dcp_runtime_scheduler.runtime.lightweight' in result['solver_entry']
    assert result['planner_options']['max_scored_candidates'] == 2


def test_explicit_solver_options_remain_supported():
    options = LightweightOptions(
        time_budget_s=0.25, max_scored_candidates=2, max_guide_candidates=2, cp_floors=(1, 3, 5, 8)
    )
    assert options.cp_floors == (1, 3, 5, 8)


def test_pack_costs_enforce_sequence_cap_without_a_wrapper():
    geometry = _geometry()
    model = QuadraticCostModel(geometry)
    costs = PackCostModel(geometry, model.forward, model.backward, max_group_seqs=2)
    assert costs.fits((16, 16), 8)
    assert not costs.fits((16, 16, 16), 8)
    assert not costs.fits((geometry.capacity(8) + 1,), 8)


@pytest.mark.parametrize('backend', ['python', 'numba'])
@pytest.mark.parametrize(
    'units,slots,expected',
    [
        ((), 4, (0.0, 0, 0, 0, 0)),
        ((1, 4, 2), 4, (4.5, 1, 4, 3, 4)),
        ((4,) * 9, 4, (13.5, 3, 12, 9, 4)),
        ((8, 1, 3), 1, (13.5, 3, 12, 3, 8)),
    ],
)
def test_shared_python_numba_dispatch_keeps_exact_chain_quantities(backend, units, slots, expected):
    if backend == 'numba':
        pytest.importorskip('numba')
    with dispatch_backend(backend):
        assert dispatch_chain(units, slots, 0.5) == expected
