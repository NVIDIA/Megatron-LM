# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Generic hot-switch policy and its existing async CPU-worker integration."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from megatron.core.datasets import dcp_schedule_prefetch as prefetch
from megatron.core.datasets.dcp_runtime_scheduler.runtime.cost_model import QuadraticCostModel
from megatron.core.datasets.dcp_runtime_scheduler.runtime.dynamic_usp import (
    DynamicUSPRuntimePlanner,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.hot_switch import (
    HotSwitchOptions,
    HotSwitchScheduler,
    InfeasibleLayout,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import (
    RuntimePlanGeometry,
    RuntimeScheduleError,
    validate_rows,
)


def _geometry(world=8, capacity=1024, max_seqs=64):
    return RuntimePlanGeometry(world, capacity, 1, tuple(range(1, world + 1)), 2 * world, max_seqs)


def _layout(layout):
    return [[(g.start, g.size, g.sample_ids) for g in row] for row in layout]


def test_generic_policy_preserves_upstream_long_and_short_order():
    tokens = {0: 350, 1: 350, 2: 30, 3: 30}
    scheduler = HotSwitchScheduler(8, (1, 2, 4, 8), 100)
    rows = scheduler.allocate(
        tokens,
        {0: 8.0, 1: 8.0, 2: 1.0, 3: 1.0},
        1,
        lambda ids, size: sum(tokens[i] for i in ids) <= size * 100,
    )
    assert _layout(rows) == [[(0, 4, [1, 3]), (4, 4, [0, 2])]]


def test_equal_weights_preserve_input_order_not_sorted_sample_ids():
    tokens = {9: 100, 2: 100, 7: 100, 1: 100}
    scheduler = HotSwitchScheduler(4, (1, 2, 4), 100)
    rows = scheduler.allocate(
        tokens,
        dict.fromkeys(tokens, 1.0),
        1,
        lambda ids, size: sum(tokens[i] for i in ids) <= size * 100,
    )
    assert _layout(rows) == [[(0, 1, [9]), (1, 1, [2]), (2, 1, [7]), (3, 1, [1])]]


@pytest.mark.parametrize("world", [1, 3, 8, 12, 32])
def test_runtime_is_independent_of_attention_or_communication_attributes(world):
    geometry = _geometry(world)
    model = QuadraticCostModel(geometry)
    planner = DynamicUSPRuntimePlanner(geometry, model)
    logical = {i: 15 + i % 7 for i in range(64)}
    padded = {
        i: (n + geometry.dataset_alignment - 1)
        // geometry.dataset_alignment
        * geometry.dataset_alignment
        for i, n in logical.items()
    }
    plan = planner.plan_padded(padded, logical_lengths=logical)
    validate_rows(plan.rows, padded, geometry, 1)
    assert all(group[1] & (group[1] - 1) == 0 for row in plan.groups for group in row)
    assert planner.last_statistics["cost_length_source"] == "after_lm_logical"


def test_physical_padding_and_group_sequence_caps_are_enforced():
    geometry = replace(_geometry(capacity=48, max_seqs=2), dataset_alignment=16)
    planner = DynamicUSPRuntimePlanner(geometry, QuadraticCostModel(geometry))
    lengths = {i: 17 for i in range(24)}
    padded = dict.fromkeys(lengths, 32)
    plan = planner.plan_padded(padded, logical_lengths=lengths)
    validate_rows(plan.rows, padded, geometry, 1)
    assert all(len(ids) <= 2 for row in plan.groups for _, _, ids in row)


def test_incumbent_is_preserved_when_model_prefers_it():
    geometry = _geometry()
    planner = DynamicUSPRuntimePlanner(geometry, QuadraticCostModel(geometry))
    incumbent = [[[0, 1]] * 8]
    padded, logical = {0: 2304, 1: 1408}, {0: 2300, 1: 1400}
    before = planner.timeline.evaluate(planner._costed([[dict(cp=8, sample_ids=[0, 1])]], logical))
    plan = planner.plan_padded(padded, logical_lengths=logical, incumbent_rows=incumbent)
    assert plan.predicted_ms <= before


def test_invalid_inputs_and_incumbents_fail_before_execution():
    geometry = _geometry()
    planner = DynamicUSPRuntimePlanner(geometry, QuadraticCostModel(geometry))
    for invalid in ({0: 17}, [(0, 16), (0, 16)], {0: 0}, {-1: 16}):
        with pytest.raises(ValueError):
            planner.plan_padded(invalid)
    with pytest.raises(RuntimeScheduleError, match="lost or duplicated"):
        planner.plan_padded({0: 16, 1: 16}, incumbent_rows=[[[0]] * 8])
    with pytest.raises(RuntimeScheduleError, match="exceeds every allowed"):
        planner.plan_padded({0: 8208})


@pytest.mark.parametrize(
    "options",
    [dict(candidate_count=0), dict(outlier_threshold=float("nan")), dict(column_offset=1)],
)
def test_invalid_options_are_not_silently_ignored(options):
    with pytest.raises(ValueError):
        HotSwitchOptions(**options)


def test_no_empty_group_or_partial_cover_is_returned():
    scheduler = HotSwitchScheduler(8, (1, 2), 100)
    with pytest.raises(InfeasibleLayout):
        scheduler.solve({0: 1}, {0: 1.0}, lambda ids, size: True, lambda layout: 1.0)


def test_builtin_worker_reuses_one_cpu_process_and_independent_model_interface():
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=1024,
        min_dynamic_context_parallel_size=1,
        dynamic_cp_runtime_model="calibrated_attention",
        dynamic_cp_runtime_model_options='{"allow_provisional":true}',
        dynamic_cp_runtime_solver="dynamic_usp",
        dynamic_cp_runtime_solver_options='{"candidate_count":2}',
    )
    worker = prefetch._CpuPlanner(prefetch.runtime_spec(config, 8))
    try:
        first = worker.solve([2300, 1400], [2304, 1408], [[[0, 1]] * 8])
        second = worker.solve([2300, 1400], [2304, 1408], [[[0, 1]] * 8])
    finally:
        worker.close()
    assert first["rows"] == second["rows"]
    assert first["worker_pid"] == second["worker_pid"]
    for result in (first, second):
        assert not result["worker_cuda_initialized"]
        assert not result["worker_distributed_initialized"]
        assert not result["external_scheduler_imported"]
        assert "DynamicUSPRuntimePlanner.plan_padded" in result["solver_entry"]
        assert result["planner_options"]["candidate_count"] == 2
