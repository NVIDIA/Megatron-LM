# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Contracts of the in-tree CPU runtime and its frozen timing models."""

import json
import random
from types import SimpleNamespace

import pytest

from megatron.core.datasets import dcp_schedule_prefetch as prefetch
from megatron.core.datasets.dcp_runtime_scheduler.models import make_model
from megatron.core.datasets.dcp_runtime_scheduler.models.base import (
    DcpCostModel,
    PackLimits,
    QuadraticCostModel,
    largest_priced_sequences,
)
from megatron.core.datasets.dcp_runtime_scheduler.models.cache import PackCostModel
from megatron.core.datasets.dcp_runtime_scheduler.runtime.dynamic_usp import (
    DynamicUSPRuntimePlanner,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.hot_switch import InfeasibleLayout
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


@pytest.mark.parametrize(
    "planner_type,model_name",
    [(LightweightRuntimePlanner, "qwen35_full"), (DynamicUSPRuntimePlanner, "usp_calib_v1")],
)
def test_bound_models_and_solvers_share_one_injected_contract(planner_type, model_name):
    geometry = RuntimePlanGeometry(32, 16384, 32, (8, 16, 32), 64, 64)
    options = (
        {"allow_provisional": True}
        if model_name == "usp_calib_v1"
        else {
            "hardware": "gb200",
            "non_attention": {"forward_ms_per_layer": 0.5, "backward_ms_per_layer": 1.0},
        }
    )
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


def test_maximum_cp_size_filters_geometry_and_final_row_validation():
    geometry = RuntimePlanGeometry(8, 100, 2, (1, 2, 4, 8), 16, 64, True, max_cp_size=3)
    assert geometry.allowed_cp_sizes == (1, 2)
    assert geometry.supports_group(0, 2)
    assert not geometry.supports_group(0, 4)
    with pytest.raises(RuntimeScheduleError, match='cp=8'):
        validate_rows([[[0]] * 8], {0: 16}, geometry, 1)


def test_maximum_cp_size_rejects_an_empty_cp_domain():
    with pytest.raises(ValueError, match='max_cp_size'):
        RuntimePlanGeometry(8, 100, 2, (4, 8), 16, 64, True, max_cp_size=2)


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
def test_both_solvers_can_cover_a_domain_using_only_capped_cp_groups(planner_type):
    geometry = RuntimePlanGeometry(8, 100, 2, (1, 2, 4, 8), 16, 64, True, max_cp_size=2)
    lengths = dict.fromkeys(range(8), 192)
    planner = planner_type(geometry, QuadraticCostModel(geometry))
    plan = planner.plan_padded(lengths)
    validate_rows(plan.rows, lengths, geometry, 1)
    assert plan.microbatches == 2
    assert all(cp == 2 for row in plan.groups for _, cp, _ in row)


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
@pytest.mark.parametrize(
    'maximum,values,microbatches,lightweight_fallback',
    [
        (2, [1023, 2047, 4095, 6143, 7167], 1, True),
        (1, [1023] * 16, 1, True),
        (2, [1023] * 16, 1, False),
        (1, [2000] * 17, 2, True),
    ],
)
def test_capped_packing_keeps_enough_nonempty_groups_to_fill_every_row(
    planner_type, maximum, values, microbatches, lightweight_fallback
):
    geometry = RuntimePlanGeometry(8, 4096, 2, (1, 2, 4, 8), 16, 64, True, maximum)
    lengths = dict(enumerate(values))
    planner = planner_type(geometry, QuadraticCostModel(geometry))
    plan = planner.plan(lengths)
    groups = validate_rows(plan.rows, lengths, geometry, 1)
    assert plan.microbatches == microbatches
    assert all(cp <= maximum and ids for row in groups for _, cp, ids in row)
    assert sorted(i for row in groups for _, _, ids in row for i in ids) == list(lengths)
    assert plan.predicted_ms > 0
    assert type(planner.last_statistics['capacity_fallback']) is bool
    if planner_type is LightweightRuntimePlanner:
        assert planner.last_statistics['capacity_fallback'] is lightweight_fallback
        assert planner.last_statistics['scored_candidates'] <= planner.options.max_scored_candidates


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
def test_a_cp_cap_cannot_be_satisfied_by_replicating_short_samples(planner_type):
    geometry = RuntimePlanGeometry(8, 4096, 2, (1, 2, 4, 8), 16, 64, True, max_cp_size=1)
    planner = planner_type(geometry, QuadraticCostModel(geometry))
    # Seven samples cannot fill eight nonempty CP1 groups, regardless of capacity.
    with pytest.raises(RuntimeScheduleError):
        planner.plan(dict.fromkeys(range(7), 1024))


@pytest.mark.parametrize('world,allowed', [(8, (1, 2, 4, 8)), (12, (1, 2, 3, 4, 6))])
def test_dynamic_usp_recovers_a_capacity_plan_after_policy_failure(world, allowed, monkeypatch):
    geometry = RuntimePlanGeometry(world, 4096, 2, allowed, 2 * world, 64, True)
    planner = DynamicUSPRuntimePlanner(geometry, QuadraticCostModel(geometry))
    lengths = dict.fromkeys(range(world), 100)

    def rejected_search(*args, **kwargs):
        raise InfeasibleLayout('Policy rejected this feasible layout')

    monkeypatch.setattr(planner.scheduler, 'solve', rejected_search)
    plan = planner.plan(lengths)
    groups = validate_rows(plan.rows, lengths, geometry, 1)
    assert planner.last_statistics['capacity_fallback'] is True
    assert all(cp in planner.scheduler.sizes for row in groups for _, cp, _ in row)
    assert all(cp != 6 for row in groups for _, cp, _ in row)


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
@pytest.mark.parametrize('incumbent', [False, True])
def test_capacity_fallback_is_unused_when_search_or_an_incumbent_is_feasible(
    planner_type, incumbent, monkeypatch
):
    geometry = RuntimePlanGeometry(8, 4096, 2, (1, 2, 4, 8), 16, 64, True, max_cp_size=1)
    planner = planner_type(geometry, QuadraticCostModel(geometry))

    def unexpected_fallback(*args, **kwargs):
        pytest.fail('A valid search result or incumbent must avoid the capacity fallback')

    monkeypatch.setattr(planner, '_capacity_fallback', unexpected_fallback)
    if incumbent:
        lengths = dict.fromkeys(range(16), 1024)
        rows = [[[2 * rank, 2 * rank + 1] for rank in range(8)]]
    else:
        lengths, rows = dict.fromkeys(range(8), 4096), None
    plan = planner.plan_padded(lengths, incumbent_rows=rows)
    validate_rows(plan.rows, lengths, geometry, 1)
    assert planner.last_statistics['capacity_fallback'] is False


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
def test_both_solvers_reject_incumbents_exceeding_the_cp_cap(planner_type):
    geometry = RuntimePlanGeometry(8, 100, 2, (1, 2, 4, 8), 16, 64, True, max_cp_size=2)
    lengths = dict.fromkeys(range(4), 192)
    planner = planner_type(geometry, QuadraticCostModel(geometry))
    # This CP8 pack fits physical capacity, but the user capped every group at CP2.
    with pytest.raises(RuntimeScheduleError, match='cp=8'):
        planner.plan_padded(lengths, incumbent_rows=[[[0, 1, 2, 3]] * 8])


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
@pytest.mark.parametrize('padded_entry', [False, True])
def test_alignment_capacity_failure_under_a_cp_cap_precedes_model_pricing(
    planner_type, padded_entry, monkeypatch
):
    geometry = RuntimePlanGeometry(8, 100, 2, (1, 2, 4, 8), 16, 64, True, max_cp_size=2)
    model = QuadraticCostModel(geometry)

    def unexpected_price(*args, **kwargs):
        pytest.fail('An infeasible sample must be rejected before model pricing')

    monkeypatch.setattr(model, 'forward', unexpected_price)
    monkeypatch.setattr(model, 'backward', unexpected_price)
    planner = planner_type(geometry, model)
    # 200 logical tokens fit the bare CP2 capacity, but alignment stores 208.
    with pytest.raises(RuntimeScheduleError, match='max_cp_size') as error:
        if padded_entry:
            planner.plan_padded({7: 208}, logical_lengths={7: 200})
        else:
            planner.plan({7: 200})
    assert 'sample 7' in str(error.value)
    assert '200' in str(error.value)
    assert 'max_seqlen_per_dp_cp_rank=100' in str(error.value)


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
def test_worker_replaces_an_uncapped_default_incumbent_for_both_solvers(planner_type, monkeypatch):
    from megatron.core.datasets import dcp_runtime_worker as worker

    # Exercise the CPU worker handler without inheriting the training test's Torch state.
    monkeypatch.setattr(worker, 'sys', SimpleNamespace(modules={}))
    geometry = RuntimePlanGeometry(8, 100, 2, (1, 2, 4, 8), 16, 64, True, max_cp_size=2)
    planner = planner_type(geometry, QuadraticCostModel(geometry))
    spec = {'model': 'test_quadratic'}
    baseline = [[list(range(8))] * 8]
    result = worker._solve(planner, spec, [96] * 8, [96] * 8, baseline)
    groups = validate_rows(result['rows'], dict.fromkeys(range(8), 96), geometry, 1)
    assert all(cp <= 2 for row in groups for _, cp, _ in row)
    assert result['statistics']['incumbent_rejection'] == 'default_incumbent_exceeds_max_cp_size'


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


def test_frozen_models_include_both_phases_for_odd_cp():
    geometry = _geometry(world=32, capacity=16384, ep=32)
    options = {
        "hardware": "gb200",
        "non_attention": {"forward_ms_per_layer": 0.5, "backward_ms_per_layer": 1.0},
    }
    model = make_model('qwen35_full', geometry, **options)
    for cp in (1, 3, 5, 32):
        values = [1025, 511]
        ranks = tuple(range(32 - cp, 32))
        assert model.forward(values, cp, ranks=ranks) > 0
        assert model.backward(values, cp, ranks=ranks) > 0
        assert model.program_costs(values, cp, ranks=ranks)
    assert model.metadata()['attention_prediction_scope'] == 'WHOLE_MODULE_ONLY'


def test_builtin_rpc_needs_no_external_scheduler_package(tmp_path):
    path = tmp_path / 'runtime.json'
    path.write_text(
        json.dumps(
            {
                'type': 'dcp_benchmark',
                'hardware': 'h100',
                'non_attention': {'forward_ms_per_layer': 0.5, 'backward_ms_per_layer': 1.0},
            }
        )
    )
    config = SimpleNamespace(
        dynamic_cp_runtime_config=str(path),
        max_seqlen_per_dp_cp_rank=4096,
        min_dynamic_context_parallel_size=1,
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
    spec = prefetch.runtime_spec(config, 8)
    # Solver options are not part of the runtime JSON; tests set them on the specification.
    spec['solver_options'] = {'max_scored_candidates': 2}
    worker = prefetch._CpuPlanner(spec)
    try:
        result = worker.solve([9200, 5600], [9200, 5600], [[[0, 1]] * 8])
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
    # A per-CP-size limit takes the place of the single cap.
    costs = PackCostModel(
        geometry, model.forward, model.backward, max_group_seqs=2, sequence_limit={8: 3, 4: 1}.get
    )
    assert costs.sequence_cap(8) == 3 and costs.fits((16, 16, 16), 8)
    assert not costs.fits((16, 16, 16, 16), 8)
    assert costs.fits((16,), 4) and not costs.fits((16, 16), 4)


# --------------------------------------------------------------------------------------
# Invalid default incumbents never stop training (RC-2)
# --------------------------------------------------------------------------------------
def _worker_solve(monkeypatch, planner, lengths, incumbent):
    from megatron.core.datasets import dcp_runtime_worker as worker

    monkeypatch.setattr(worker, 'sys', SimpleNamespace(modules={}))
    spec = {'model': 'test_quadratic'}
    return worker._solve(planner, spec, list(lengths), list(lengths), incumbent)


_INVALID_DEFAULT_INCUMBENTS = {
    # 65 sequences in the only CP8 group; the heuristic does not read the sequence cap.
    'sequence_cap': (
        RuntimePlanGeometry(8, 1024, 2, (8,), 16, 64, True),
        [16] * 65,
        [[list(range(65))] * 8],
        'default_incumbent_violates_runtime_contract',
        'packs 65 sequences',
    ),
    # The cap the user configured (thd_max_packed_sequences) is violated in the same way.
    'user_sequence_cap': (
        RuntimePlanGeometry(8, 1024, 2, (1, 2, 4, 8), 16, 2, True),
        [16] * 24,
        [[[3 * rank, 3 * rank + 1, 3 * rank + 2] for rank in range(8)]],
        'default_incumbent_violates_runtime_contract',
        'packs 3 sequences',
    ),
    # DPxCP=12: the heuristic doubles a CP4 group to CP8, which is not a width of this domain.
    'cp8_on_dp_cp_12': (
        RuntimePlanGeometry(12, 4096, 1, (1, 2, 4), 24, 64, True),
        [480] * 12,
        [[list(range(6))] * 8 + [list(range(6, 12))] * 4],
        'default_incumbent_outside_predictor_cp_domain',
        'cp=8',
    ),
    # DPxCP=12: the heuristic's full DPxCP group has no power-of-two runtime width.
    'full_group_on_dp_cp_12': (
        RuntimePlanGeometry(12, 4096, 1, (1, 2, 4), 24, 64, True),
        [480] * 12,
        [[list(range(12))] * 12],
        'default_incumbent_outside_predictor_cp_domain',
        'cp=12',
    ),
    'above_max_cp_size': (
        RuntimePlanGeometry(8, 1024, 2, (1, 2, 4, 8), 16, 64, True, max_cp_size=2),
        [96] * 8,
        [[list(range(8))] * 8],
        'default_incumbent_exceeds_max_cp_size',
        'cp=8',
    ),
    'unknown_sample': (
        RuntimePlanGeometry(8, 1024, 2, (1, 2, 4, 8), 16, 64, True),
        [96] * 8,
        [[[rank + 1] for rank in range(8)]],
        'default_incumbent_violates_runtime_contract',
        'unknown sample id 8',
    ),
}


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
@pytest.mark.parametrize('case', sorted(_INVALID_DEFAULT_INCUMBENTS))
def test_worker_discards_any_invalid_default_incumbent_and_still_plans(
    planner_type, case, monkeypatch
):
    geometry, lengths, incumbent, reason, detail = _INVALID_DEFAULT_INCUMBENTS[case]
    planner = planner_type(geometry, QuadraticCostModel(geometry))
    result = _worker_solve(monkeypatch, planner, lengths, incumbent)
    # The returned layout satisfies the whole contract, including the violated constraint.
    validate_rows(
        result['rows'], dict(enumerate(lengths)), geometry, min(geometry.allowed_cp_sizes)
    )
    statistics = result['statistics']
    assert statistics['incumbent_rejection'] == reason
    assert detail in statistics['incumbent_rejection_detail']
    assert not statistics['selected_incumbent']


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
@pytest.mark.parametrize('search', ['ties', 'finds_nothing', 'cannot_be_priced'])
def test_worker_uses_a_valid_default_incumbent_when_search_finds_nothing_better(
    planner_type, search, monkeypatch
):
    geometry = RuntimePlanGeometry(8, 1024, 2, (1, 2, 4, 8), 16, 64, True)
    planner = planner_type(geometry, QuadraticCostModel(geometry))
    if search == 'ties':
        # One sample that needs every rank: the search can only rebuild the incumbent.
        lengths, incumbent = [8192], [[[0]] * 8]
    else:
        lengths, incumbent = [1024] * 8, [[[rank] for rank in range(8)]]
        error = (
            InfeasibleLayout('no candidate count produced a complete feasible layout')
            if search == 'finds_nothing'
            else RuntimeScheduleError('cost model rejected CP2 pack (1024, 1024)')
        )

        def no_result(*args, **kwargs):
            raise error

        if planner_type is DynamicUSPRuntimePlanner:
            monkeypatch.setattr(planner.scheduler, 'solve', no_result)
        else:
            if search == 'finds_nothing':
                error = RuntimeScheduleError('allowed CP widths cannot fill every rank')
            monkeypatch.setattr(planner, '_pack', no_result)
            monkeypatch.setattr(planner, '_global_pack', lambda *args, **kwargs: None)
            monkeypatch.setattr(planner, '_cost_pack', lambda *args, **kwargs: None)
    result = _worker_solve(monkeypatch, planner, lengths, incumbent)
    assert result['rows'] == incumbent
    assert result['statistics']['selected_incumbent']
    assert result['statistics']['capacity_fallback'] is False
    assert 'incumbent_rejection' not in result['statistics']


class _NoCp8Model(QuadraticCostModel):
    """A cost model whose prediction domain is narrower than the geometry."""

    def forward(self, lengths, cp, *, ranks=None):
        if cp == 8:
            raise ValueError('model has no CP8 communication calibration')
        return super().forward(lengths, cp, ranks=ranks)


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
def test_planners_drop_an_incumbent_the_cost_model_cannot_price(planner_type):
    geometry = RuntimePlanGeometry(8, 1024, 2, (1, 2, 4, 8), 16, 64, True)
    lengths = dict.fromkeys(range(8), 1000)
    planner = planner_type(geometry, _NoCp8Model(geometry))
    # A legal CP8 pack: valid rows, but outside the model's prediction domain.
    plan = planner.plan_padded(
        {i: 1008 for i in lengths}, logical_lengths=lengths, incumbent_rows=[[list(range(8))] * 8]
    )
    groups = validate_rows(plan.rows, {i: 1008 for i in lengths}, geometry, 1)
    assert all(cp != 8 for row in groups for _, cp, _ in row)
    assert not planner.last_statistics['selected_incumbent']
    assert plan.predicted_ms > 0


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
def test_a_layout_always_exists_when_the_full_dp_cp_group_is_allowed(planner_type, monkeypatch):
    """Power-of-two DPxCP without max_cp_size: one full-width pack per microbatch always fits."""
    rng = random.Random(20261112)
    for _ in range(60):
        world = rng.choice([2, 4, 8, 16])
        capacity = rng.choice([256, 1024])
        widths = [cp for cp in (1, 2, 4, 8, 16) if cp <= world]
        minimum = rng.choice(widths)
        allowed = tuple(cp for cp in widths if cp >= minimum)
        sequence_cap = rng.choice([64, 1, 2, 3])
        geometry = RuntimePlanGeometry(world, capacity, 1, allowed, 2 * world, sequence_cap, True)
        count = rng.choice([1, 2, 3, world, 3 * world])
        top = world * capacity
        lengths = [
            rng.randint(1, top) if rng.random() < 0.3 else rng.randint(1, capacity)
            for _ in range(count)
        ]
        padded = [(n + 2 * world - 1) // (2 * world) * (2 * world) for n in lengths]
        # An arbitrary, usually invalid, seed: everything in one full-width group.
        incumbent = rng.choice([None, [[list(range(count))] * world]])
        planner = planner_type(geometry, QuadraticCostModel(geometry))
        from megatron.core.datasets import dcp_runtime_worker as worker

        monkeypatch.setattr(worker, 'sys', SimpleNamespace(modules={}))
        spec = {'model': 'test_quadratic'}
        result = worker._solve(planner, spec, lengths, padded, incumbent)
        validate_rows(result['rows'], dict(enumerate(padded)), geometry, minimum)


# --------------------------------------------------------------------------------------
# The packaged cost models price packs of 1..64 sequences; any cap in that range is valid
# --------------------------------------------------------------------------------------
_NON_ATTENTION = {"forward_ms_per_layer": 0.5, "backward_ms_per_layer": 1.0}
_PACKAGED_MODELS = {
    # name: (world, tokens_per_rank, ep, allowed CP sizes, CP size priced below, options)
    'dcp_benchmark': (
        8,
        4096,
        1,
        (1, 2, 4, 8),
        1,
        dict(
            hardware='h100',
            query_heads=32,
            kv_heads=4,
            head_dim=128,
            layer_count=48,
            non_attention=dict(_NON_ATTENTION, source='synthetic test input'),
        ),
    ),
    'usp_calib_v1': (32, 16384, 1, (8, 16, 32), 8, dict(allow_provisional=True)),
    'qwen35_full': (
        32,
        16384,
        32,
        (1, 2, 4, 8, 16, 32),
        1,
        dict(hardware='gb200', non_attention=_NON_ATTENTION),
    ),
}


@pytest.mark.parametrize('cap', [1, 63, 64])
@pytest.mark.parametrize('model_name', sorted(_PACKAGED_MODELS))
def test_packaged_cost_models_accept_any_sequence_cap_up_to_64(model_name, cap):
    """thd_max_packed_sequences=64 leaves 63 real sequences once a dummy tail is appended."""
    world, capacity, ep, allowed, cp, options = _PACKAGED_MODELS[model_name]
    geometry = RuntimePlanGeometry(world, capacity, ep, allowed, 64, cap, True)
    model = make_model(model_name, geometry, **options)
    model.check_domain(geometry, None)
    pack, ranks = (64,) * cap, tuple(range(cp))
    assert model.forward(pack, cp, ranks=ranks) > 0
    assert model.backward(pack, cp, ranks=ranks) > 0


@pytest.mark.parametrize('model_name', sorted(_PACKAGED_MODELS))
def test_packaged_cost_models_reject_a_sequence_cap_above_64(model_name):
    world, capacity, ep, allowed, cp, options = _PACKAGED_MODELS[model_name]
    geometry = RuntimePlanGeometry(world, capacity, ep, allowed, 64, 65, True)
    with pytest.raises(ValueError):
        # Either the model rejects the geometry, or it refuses to price a 65-sequence pack.
        model = make_model(model_name, geometry, **options)
        model.forward((64,) * 65, max(allowed), ranks=tuple(range(max(allowed))))


# --------------------------------------------------------------------------------------
# A cost model declares what it prices in one pack; no planner builds a pack beyond it
# --------------------------------------------------------------------------------------
class _SequenceLimitedModel(QuadraticCostModel):
    """Prices at most ``limits[cp]`` sequences in one pack of ``cp`` ranks, and says so."""

    def __init__(self, geometry, limits):
        super().__init__(geometry)
        self.limits, self.asked = dict(limits), []

    def pack_limits(self, cp):
        self.asked.append(cp)
        return PackLimits(max_sequences=self.limits.get(cp))

    def forward(self, lengths, cp, *, ranks=None):
        if len(lengths) > self.limits.get(cp, 64):
            raise ValueError(f'no parameters for {len(lengths)} sequences at CP{cp}')
        return super().forward(lengths, cp, ranks=ranks)


def _classic_geometry(world=8, capacity=1024, allowed=(1, 2, 4, 8), cap=64):
    return RuntimePlanGeometry(world, capacity, 1, allowed, 2 * world, cap, True)


def test_the_declared_limit_is_asked_lazily_and_once_per_cp_size():
    geometry = _classic_geometry()
    model = _SequenceLimitedModel(geometry, {8: 2, 4: 100})
    planner = LightweightRuntimePlanner(geometry, model)
    # Building a planner asks nothing: every rank builds one at start-up, only the leader
    # and the CPU worker plan.
    assert model.asked == []
    assert [planner.sequence_limit(cp) for cp in (8, 8, 4, 2)] == [2, 2, 64, 64]
    assert model.asked == [8, 4, 2]
    assert planner.model_sequence_limits() == {8: 2}
    # The geometry's own cap still applies when it is the smaller one.
    capped = _classic_geometry(cap=1)
    assert LightweightRuntimePlanner(capped, model).sequence_limit(8) == 1


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
@pytest.mark.parametrize('limits', [{8: 2}, {8: 1, 4: 3}, {2: 1, 4: 1, 8: 1}])
def test_no_planner_packs_more_sequences_than_the_cost_model_prices(planner_type, limits):
    geometry = _classic_geometry()
    planner = planner_type(geometry, _SequenceLimitedModel(geometry, limits))
    rng = random.Random(20261112)
    for _ in range(40):
        count = rng.choice([1, 2, 3, 8, 9, 24, 40])
        lengths = {
            i: rng.randint(1, 8 * 1024) if rng.random() < 0.2 else rng.randint(1, 600)
            for i in range(count)
        }
        padded = {i: (n + 15) // 16 * 16 for i, n in lengths.items()}
        # Whatever the seed: here everything in one full group, usually above the limit.
        incumbent = [[list(range(count))] * 8] if sum(padded.values()) <= 8 * 1024 else None
        plan = planner.plan_padded(padded, logical_lengths=lengths, incumbent_rows=incumbent)
        groups = validate_rows(plan.rows, padded, geometry, 1)
        assert all(len(ids) <= limits.get(cp, 64) for row in groups for _, cp, ids in row)
        assert planner.last_statistics['model_sequence_limits'] == {
            cp: limit for cp, limit in limits.items() if cp in planner._sequence_limits
        }


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
def test_the_model_is_only_asked_about_cp_sizes_the_planner_may_use(planner_type):
    # DPxCP=16 capped at CP 4: the full group is not a size of this run.
    geometry = RuntimePlanGeometry(16, 1024, 1, (1, 2, 4, 8, 16), 32, 64, True, max_cp_size=4)
    model = _SequenceLimitedModel(geometry, {})
    planner = planner_type(geometry, model)
    rng = random.Random(5)
    for _ in range(12):
        lengths = {i: rng.randint(1, 4096) for i in range(rng.choice([16, 24, 40]))}
        padded = {i: (n + 31) // 32 * 32 for i, n in lengths.items()}
        planner.plan_padded(padded, logical_lengths=lengths)
    assert model.asked and set(model.asked) <= {1, 2, 4}
    assert planner.last_statistics['model_sequence_limits'] == {}


def test_widening_a_pack_to_the_full_group_keeps_the_declared_limit():
    # Three short samples under a minimum CP of 4: one CP4 pack that has to become the full
    # CP8 group, which prices only two sequences. The third sample moves to a second
    # microbatch in both candidate families.
    geometry = _classic_geometry(allowed=(4, 8))
    planner = LightweightRuntimePlanner(geometry, _SequenceLimitedModel(geometry, {8: 2}))
    lengths = {0: 96, 1: 64, 2: 32}
    for rounds in (planner._pack(lengths, 1), planner._global_pack(lengths, 1)):
        assert [[(g['cp'], g['sample_ids']) for g in row] for row in rounds] == [
            [(8, [0, 1])],
            [(8, [2])],
        ]
    plan = planner.plan(lengths)
    assert plan.microbatches == 2 and plan.predicted_ms > 0


@pytest.mark.parametrize('planner_type', [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
def test_the_last_resort_fallback_keeps_the_declared_limit(planner_type):
    geometry = _classic_geometry(allowed=(8,))
    planner = planner_type(geometry, _SequenceLimitedModel(geometry, {8: 2}))
    lengths = dict.fromkeys(range(5), 32)
    rounds = planner._capacity_fallback(lengths)
    assert sorted(len(group['sample_ids']) for row in rounds for group in row) == [1, 2, 2]
    # A width at which the model prices nothing is not used for the fallback at all.
    geometry = _classic_geometry(allowed=(4, 8))
    planner = planner_type(geometry, _SequenceLimitedModel(geometry, {8: 0}))
    lengths = dict.fromkeys(range(4), 32)
    rounds = planner._capacity_fallback(lengths)
    assert {group['cp'] for row in rounds for group in row} == {4}
    planner = planner_type(geometry, _SequenceLimitedModel(geometry, {4: 0, 8: 0}))
    assert planner._capacity_fallback(lengths) is None


def test_largest_priced_sequences_finds_the_count_by_pricing_full_packs():
    geometry = _classic_geometry()
    priced = []

    class Counting(_SequenceLimitedModel):
        def forward(self, lengths, cp, *, ranks=None):
            priced.append((cp, len(lengths)))
            return super().forward(lengths, cp, ranks=ranks)

    model = Counting(geometry, {8: 5, 4: 0})
    # No limit of its own: the largest full pack is priced, and that is the only pricing.
    assert largest_priced_sequences(model, geometry, 2, 64) is None
    assert priced == [(2, 64), (2, 64)]  # forward, then backward through forward
    del priced[:]
    assert largest_priced_sequences(model, geometry, 8, 64) == 5
    assert max(count for _, count in priced) == 64 and len({count for _, count in priced}) <= 8
    assert largest_priced_sequences(model, geometry, 4, 64) == 0
    # A cap above what fits: CP1 holds at most 1024 / 16 = 64 aligned sequences, 32 with a
    # coarser alignment; the count that is priced is the one that exists.
    coarse = RuntimePlanGeometry(8, 1024, 1, (1,), 32, 64, True)
    del priced[:]
    assert largest_priced_sequences(Counting(coarse, {}), coarse, 1, 64) is None
    assert priced[0] == (1, 32)
    assert largest_priced_sequences(model, geometry, 8, 3) is None  # A cap below the limit.
