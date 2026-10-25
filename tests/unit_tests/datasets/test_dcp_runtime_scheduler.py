# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Contracts of the in-tree CPU runtime and its frozen timing models."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatron.core.datasets import dcp_schedule_prefetch as prefetch
from megatron.core.datasets.dcp_runtime_scheduler.attention.dispatch import (
    dispatch_backend,
    dispatch_chain,
)
from megatron.core.datasets.dcp_runtime_scheduler.models import (
    AttentionCostModel,
    CorePredictor,
    qwen35_new_core_model,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.cost_model import QuadraticCostModel
from megatron.core.datasets.dcp_runtime_scheduler.runtime.lightweight import (
    LightweightOptions,
    LightweightRuntimePlanner,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import (
    RuntimePlanGeometry,
    RuntimeScheduleError,
    validate_rows,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.sched.costs import PackCostModel


def _geometry(world=8, capacity=1024, ep=2):
    return RuntimePlanGeometry(world, capacity, ep, tuple(range(1, world + 1)), 2 * world, 64)


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
