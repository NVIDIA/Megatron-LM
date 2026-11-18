# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The dynamic_usp solver against pinned results of the upstream solver and of the runtime."""

import hashlib
import json
from pathlib import Path

import pytest

from megatron.core.datasets import dcp_runtime_worker
from megatron.core.datasets.dcp_runtime_scheduler.models import solver_section
from megatron.core.datasets.dcp_runtime_scheduler.runtime import pipeline, planner

HERE, TABLES = Path(__file__).parent, Path(solver_section.__file__).parent
PINS = json.loads((HERE / 'dcp_usp_frozen_plans.json').read_text())
# The solver's plans under Megatron's order of a step without virtual pipeline stages.
MEGATRON = json.loads((HERE / 'dcp_non_interleaved_pins.json').read_text())
STAGE2 = json.loads((HERE / 'dcp_usp_stage2_pins.json').read_text())
# The constant that the runtime adds to every step time; the objective of the solver has none.
TAIL_MS = json.loads((TABLES / 'usp_calib_v1.json').read_text())['global']['C_global_clean_ms']
SECTIONS = ('plans', 'runtime_sequence_cap', 'pipeline_plans', 'runtime_pipeline')
PLANS = {case['name']: case for section in SECTIONS for case in PINS[section]['cases']}
PLANS.update((case['name'], case) for case in MEGATRON['plans']['cases'])
H100 = dict(model='qwen3_h100_p2p', tokens_per_rank=4096, cp_sizes=[1, 2, 4, 8], ep_size=8)
FITTED_AT = dict(pipeline_parallel=2, virtual_stages=1, schedule='non_interleaved')
STAGE2_PLANS = STAGE2['plans']['cases'] + STAGE2['ring_guard_plans']['cases']


def _planner(case, tmp_path, **changes):
    """The planner that the CPU worker builds for the run of a pinned case."""
    case = dict(case, **changes)
    name = case.get('model', 'usp_calib_v1')  # the original profile, unless the case names one
    geometry = dict(world_size=case['world_size'], tokens_per_rank=case['tokens_per_rank'])
    geometry.update(ep_size=case.get('ep_size', 1), allowed_cp_sizes=case['cp_sizes'])
    geometry.update(dataset_alignment=2 * case['world_size'], align_group_starts=True)
    geometry.update(max_group_seqs=case.get('max_group_sequences', 64))
    model = dict(allow_provisional=True) if name == 'usp_calib_v1' else {}
    if case.get('solver'):  # coefficients of the search: the solver section of a table
        table = json.loads((TABLES / f'{name}.json').read_text())
        model['calibration'] = str(tmp_path / 'calibration.json')
        Path(model['calibration']).write_text(json.dumps(dict(table, solver=case['solver'])))
    options = dict(case.get('options', {}), keep_candidate_layouts=True)
    spec = dict(type='dynamic_usp', model=name, geometry=geometry)
    spec.update(model_options=model, solver_options=options)
    sizes = case.get('pipeline', (case.get('pipeline_parallel', 1), case.get('virtual_stages')))
    if sizes[1] == 1:  # the solver's order for one chunk per stage, which no run takes
        options['pipeline'] = tuple(sizes)
    elif sizes[0] > 1:  # as in a run: the solver's order with virtual stages, else Megatron's
        spec['pipeline'] = dict(stages=sizes[0], virtual_stages=sizes[1] or 1)
    return dcp_runtime_worker._initialize_planner(spec)


def _layout(rows):
    return [[[start, size, list(ids)] for start, size, ids in row] for row in rows]


def _rounds(layout):
    return [[dict(cp=size, sample_ids=list(ids)) for _, size, ids in row] for row in layout]


def _trail(candidates):
    return [[c['microbatches'], c.get('rejection', 'ok')] for c in candidates]


def _ms(objective_ms):
    """A pinned objective of the solver as a step time of the runtime."""
    return pytest.approx(objective_ms + TAIL_MS, rel=1e-12)


@pytest.mark.parametrize('name', list(PLANS))
def test_plan_reproduces_every_pinned_candidate_and_the_chosen_plan(tmp_path, name):
    case = PLANS[name]
    usp = _planner(case, tmp_path)
    plan = usp.plan(dict(enumerate(case['lengths'])))
    # Where the sequence cap raises the bound inside the solver's window, 'upstream' has its result.
    statistics, solver = usp.last_statistics, case.get('upstream', case).get('candidates', [])
    pinned = {candidate['microbatches']: candidate for candidate in solver}
    assert _trail(statistics['candidates']) == case.get('candidate_trail', _trail(solver))
    for candidate in statistics['candidates']:
        expected = pinned.get(candidate['microbatches'], {})
        if 'layout' in expected:
            assert _layout(candidate['layout']) == expected['layout']
            assert candidate['predicted_ms'] == _ms(expected['objective_ms'])
    # The fastest count is planned; ten cases of the runtime's own rules pin its time, no layout.
    best = pinned.get(case['chosen_microbatches'], {})
    step_ms = best['objective_ms'] + TAIL_MS if 'objective_ms' in best else case['predicted_ms']
    assert plan.microbatches == case['chosen_microbatches']
    assert plan.predicted_ms == pytest.approx(step_ms, rel=1e-12)
    assert best.get('layout', case.get('layout')) in (None, _layout(plan.groups))
    assert statistics['capacity_fallback'] is case.get('capacity_fallback', False)
    # No group holds more sequences than the cap of the run, 64 by default.
    largest = max(len(ids) for row in plan.groups for _, _, ids in row)
    assert largest == case.get('max_sequences_in_a_group', largest) <= usp.geometry.max_group_seqs


@pytest.mark.parametrize('case', STAGE2_PLANS + [PLANS['B_P4_stage2']], ids=lambda c: c['name'])
def test_plan_with_stage2_reproduces_the_solver(tmp_path, case):
    # The case on Megatron's order has its Stage 2 result in a dictionary of its own.
    pinned = case['stage2'] if isinstance(case['stage2'], dict) else case
    codes = pinned.get('codes') or [microbatch['code'] for microbatch in case['stage2']]
    on = {'stage2': {'enabled': True, 'keep_threshold': pinned['keep_threshold']}}
    usp, lengths = _planner(case, tmp_path, solver=on), dict(enumerate(case['lengths']))
    plan, stage1 = usp.plan(lengths), _planner(case, tmp_path).plan(lengths)
    record = usp.last_statistics['stage2']
    # The re-packed layout is the result of the search, also when its step time is higher.
    assert _layout(plan.groups) == pinned['layout']
    times = (stage1.predicted_ms, record['stage1_predicted_ms'], plan.predicted_ms)
    before, after = _ms(pinned['stage1_objective_ms']), _ms(pinned['objective_after_stage2_ms'])
    assert times == (before, stage1.predicted_ms, after)
    assert record['counts'] == {code: codes.count(code) for code in set(codes)}
    assert record['planned'] == ('stage2' if 'ADOPTED' in codes else 'stage1')
    assert (stage1.rows != plan.rows) == (record['adopted_microbatches'] > 0)
    # The Stage 1 layout as the default layout: the faster of it and the re-packed is planned.
    again = usp.plan(lengths, incumbent_rounds=_rounds(stage1.groups))
    kept = stage1.predicted_ms <= plan.predicted_ms
    assert usp.last_statistics['stage2']['planned'] == ('default' if kept else 'stage2')
    assert again.rows == (stage1.rows if kept else plan.rows)


def test_solver_quantities_of_the_original_profile_equal_every_pin(tmp_path):
    weights = {name: s for name, s in PINS['sample_weight'].items() if name != 'description'}
    for section in weights.values():
        coefficients = tuple(section['coefficients'][key] for key in 'ABC')
        for row in section['weights']:
            assert solver_section.sample_weight(row['tokens'], coefficients) == row['weight']
    default = tuple(weights['default']['coefficients'].values())
    model = _planner(PLANS['A'], tmp_path).model
    assert model.sample_weight_coefficients == solver_section.DEFAULT_SAMPLE_WEIGHT == default
    for table in PINS['tier_time_factors']['tables']:
        derived = model.derive_tier_time_factors(table['reference_tokens'], table['cp_sizes'])
        factors = {int(size): factor for size, factor in table['factors'].items()}
        assert derived == pytest.approx(factors, rel=1e-12)
    for pinned in PINS['pipeline_group_times']['groups']:
        shares = {key: pinned[key] for key in ('compute_fwd_share', 'comm_fwd_share')}
        solver = {'pipeline': dict(shares, fitted_at=FITTED_AT)}  # the record changes no time
        model = _planner(PLANS['A'], tmp_path, solver=solver).model
        times = model.pipeline_group_times(tuple(pinned['lengths']), pinned['cp'])
        assert times == pytest.approx((pinned['forward_ms'], pinned['backward_ms']), rel=1e-12)
    assert model.pipeline_terms['fitted_at'] == FITTED_AT


def test_both_plan_builders_reproduce_the_pinned_order_of_every_stage():
    # 256 plans with virtual stages: operation, predecessor and has-previous flag in plan order.
    for pinned in PINS['pipeline_schedule']['digests']:
        sizes = pinned['stages'], pinned['virtual_stages']
        plans = [pipeline.interleaved_plan(count, *sizes) for count in pinned['counts']]
        document = [[[list(o), n and list(n), flag] for o, n, flag in plan] for plan in plans]
        text = json.dumps(document, separators=(',', ':'))
        assert hashlib.sha256(text.encode()).hexdigest() == pinned['sha256']
    for order in MEGATRON['orders']:  # stage r warms up with min(stages - r - 1, n) forwards
        plan = pipeline.non_interleaved_plan(order['microbatches'], order['stages'])
        rows = [[f'{o[0]}{o[1]}' for o, _, _ in plan if o[3] == s] for s in range(order['stages'])]
        assert [' '.join(row) for row in rows] == order['per_stage']


def test_step_time_reproduces_the_pinned_replays_of_both_schedules():
    pinned = PINS['pipeline_step_time']
    for case in pinned['cases']:
        arguments = case['rows'], case['world_size'], case['stages'], case['virtual_stages']
        assert pipeline.step_time(*arguments, case['per_mb_overhead_ms']) == case['step_ms']
    for run in pinned['uniform']:  # identical microbatches
        rows = [[(0, 8, 30.0, 70.0)]] * run['microbatches']
        assert pipeline.step_time(rows, 8, run['stages'], run['virtual_stages']) == run['step_ms']
    toy = pinned['lock_step_toy']  # the ranks of one CP group start a chunk together
    for result in toy['results']:
        sizes = toy['world_size'], result['stages'], result['virtual_stages']
        assert pipeline.step_time(toy['rows'], *sizes) == result['step_ms']
    for case in MEGATRON['step_time']['cases']:
        arguments = case['rows'], case['world_size'], case['stages'], 1, case['per_mb_overhead_ms']
        assert pipeline.step_time(*arguments, schedule='non_interleaved') == case['step_ms']


def test_batch_without_a_layout_is_planned_by_capacity_or_is_a_schedule_error(tmp_path):
    # One group of two sequences per microbatch, 42 samples that fill it and 42 short ones: the
    # search gives up 16 counts past its window of four, and the capacity fallback plans.
    case = dict(world_size=8, tokens_per_rank=1024, cp_sizes=[8], max_group_sequences=2)
    usp, lengths = _planner(case, tmp_path), dict(enumerate([8192] * 42 + [64] * 42))
    plan, statistics = usp.plan(lengths), usp.last_statistics
    assert _trail(statistics['candidates']) == [[n, 'group_capacity'] for n in range(43, 63)]
    assert (statistics['capacity_fallback'], plan.microbatches) == (True, 63)
    planner.validate_rows(plan.rows, lengths, usp.geometry, 8)
    with pytest.raises(planner.RuntimeScheduleError, match='sample 0: .* no allowed CP size fits'):
        usp.plan({0: 8192 + 16})
    # Virtual stages take multiples of the stages; Megatron's order plans these four samples.
    case, few = PLANS['four_samples_P8'], 'takes at least 8 samples; this batch has 4'
    with pytest.raises(planner.RuntimeScheduleError, match=few):
        _planner(case, tmp_path, virtual_stages=2).plan(dict(enumerate(case['lengths'])))


def test_default_layout_is_kept_unless_the_search_is_predicted_faster(tmp_path):
    case = PLANS['A']
    slow, best = (_rounds(c['layout']) for c in case['candidates'][:2])  # 3 and 4 microbatches
    usp, lengths = _planner(case, tmp_path), dict(enumerate(case['lengths']))
    kept = usp.plan(lengths, incumbent_rounds=best)
    assert usp.last_statistics['selected_incumbent'] is True
    assert kept.predicted_ms == _ms(case['candidates'][1]['objective_ms'])
    replaced = usp.plan(lengths, incumbent_rounds=slow)
    assert (usp.last_statistics['selected_incumbent'], replaced.rows) == (False, kept.rows)
    # With virtual stages a count that is no multiple of the stages cannot run: not priced.
    usp = _planner(PLANS['A_P4_V2'], tmp_path)
    assert usp.plan(lengths, incumbent_rounds=slow).microbatches == 4
    assert usp.last_statistics['incumbent_unpriced'] == '3 microbatches, not a multiple of 4'


@pytest.mark.parametrize(
    'changes,phrase',
    [
        (dict(solver={'weights': 1}), r"Unknown key\(s\) \['weights'\] in the solver section"),
        (dict(solver={'sample_weight': {'A': True, 'B': 2, 'C': 3}}), 'A must be a finite'),
        (dict(solver={'tier_time_factors': {'12': 1.5}}), r'CP sizes of the cost model \[8,'),
        (dict(solver={'pipeline': {'compute_fwd_share': 1.5}}), r'a number in \[0, 1\]'),
        (dict(solver={'pipeline': {'fitted_at': {}}}), 'fitted_at requires the keys'),
        (dict(solver={'stage2': {'enabled': 1}}), 'stage2.enabled must be true or false'),
        # The measured H100 calibration: one pipeline stage, no Stage 2, its own overhead term.
        (dict(H100, pipeline_parallel=2), 'has no step time for more than one'),
        (dict(H100, solver={'stage2': {'enabled': True}}), 'is not offered for this'),
        (dict(H100, solver={'pipeline': {}}), 'solver.pipeline is not accepted by this'),
    ],
)
def test_bad_solver_section_or_run_shape_stops_start_up(tmp_path, changes, phrase):
    with pytest.raises(ValueError, match=phrase):
        _planner(PLANS['A'], tmp_path, **changes)


def test_stage2_obeys_the_sequence_limit_and_an_error_inside_it_never_loses_a_layout(tmp_path):
    case = STAGE2['runtime']['cases'][1]  # the runtime's own result: four sequences per group
    on = {'stage2': {'enabled': True, 'keep_threshold': case['keep_threshold']}}
    usp, lengths = _planner(case, tmp_path, solver=on), dict(enumerate(case['lengths']))
    plan, stage1 = usp.plan(lengths), _planner(case, tmp_path).plan(lengths)
    record = usp.last_statistics['stage2']
    assert (_layout(plan.groups), record['counts']) == (case['layout'], case['counts'])
    assert plan.predicted_ms == pytest.approx(case['predicted_ms'], rel=1e-12)
    marked = [(m['index'], m['code'], m['runtime_capacity']) for m in record['microbatches']]
    assert marked == [(m['index'], m['code'], True) for m in case['stage2']]
    usp.model.stage2_group_time = lambda group, cp: 1 / 0  # a hook of the cost model fails
    plan, record = usp.plan(lengths), usp.last_statistics['stage2']
    assert (plan.rows, plan.predicted_ms) == (stage1.rows, stage1.predicted_ms)
    assert record['counts'] == {'PROFILE_EXECUTION_ERROR': 5} and record['planned'] == 'stage1'
    assert record['first_error'] == 'ZeroDivisionError: division by zero'


def test_stage2_drops_a_finished_layout_that_leaves_a_ring_group_below_the_guard(tmp_path):
    tie = STAGE2['ring_guard_shipped']['keep_rule_tie']  # eight equal samples, one group of 32
    on = {'stage2': {'enabled': True, 'keep_threshold': tie['keep_threshold']}}
    usp = _planner(tie, tmp_path, solver=on)
    for length in tie['lengths']:
        usp.plan(dict.fromkeys(range(tie['samples']), length))
        assert usp.last_statistics['stage2']['counts'] == {tie['codes'][0]: 1}
