# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cost models of the runtime DCP scheduler: frozen prices, start-up selection and limits."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatron.core.datasets import dcp_runtime_worker as worker
from megatron.core.datasets import dcp_schedule_prefetch as prefetch
from megatron.core.datasets.dcp_runtime_scheduler import models
from megatron.core.datasets.dcp_runtime_scheduler.models import attention
from megatron.core.datasets.dcp_runtime_scheduler.models.full35b import gdn_features, geometry
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import RuntimeScheduleError

PACKAGE = Path(models.__file__).parents[1]
TIMES = dict(forward_ms_per_layer=0.5, backward_ms_per_layer=1.0)
# fmt: off
# Launch settings inside the domain of each cost model, as the README of the package lists them.
P2P = dict(cp_comm_type='p2p', pad_packed_seq_alignment='max', min_dynamic_context_parallel_size=1)
GQA = dict(
    P2P, num_attention_heads=32, num_query_groups=4, kv_channels=128, attention_backend='fused',
    num_layers=2, max_seqlen_per_dp_cp_rank=4096, thd_tail_padding_policy='extend_last', bf16=True,
)
QWEN35 = dict(
    P2P, num_layers=40, mtp_num_layers=1, hidden_size=2048, num_attention_heads=16,
    num_query_groups=2, kv_channels=256, qk_layernorm=True, attention_output_gate=True,
    attention_dropout=0.0, experimental_attention_variant='gdn', linear_attention_freq=4,
    linear_cp_mode='chunkwise', linear_conv_kernel_dim=4, linear_num_key_heads=16,
    linear_num_value_heads=32, linear_key_head_dim=128, linear_value_head_dim=128,
    gdn_kernel_backend='fla', expert_model_parallel_size=32, attention_backend='auto',
    cp_partition_mode='contiguous', max_seqlen_per_dp_cp_rank=16384, bf16=True,
)
H100 = json.loads((PACKAGE / 'models/qwen3_h100_p2p.json').read_text())['domain']['config']
H100 = dict(H100, **P2P, thd_tail_padding_policy='extend_last')
USP = dict(max_seqlen_per_dp_cp_rank=1024, min_dynamic_context_parallel_size=8)
# fmt: on
# The four cost models: runtime file, launch settings and DPxCP of a run that selects each.
RUNS = dict(
    gqa=(dict(type='dcp_benchmark', hardware='h100', non_attention=TIMES), GQA, 64),
    qwen35=(dict(type='dcp_benchmark', hardware='gb200', non_attention=TIMES), QWEN35, 32),
    h100=(dict(type='dynamic_usp', hardware='h100', max_cp_size=8), H100, 32),
    usp=(dict(type='dynamic_usp'), USP, 32),
)


def _pins(name):
    return json.loads((Path(__file__).parent / f'dcp_{name}.json').read_text())


def _spec(tmp_path, run, world=None, file=(), **settings):
    """Start-up validation of one of RUNS after changes to its runtime file or its launch."""
    runtime, launch, measured = RUNS[run]
    runtime = {key: value for key, value in {**runtime, **dict(file)}.items() if value is not None}
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps(runtime))
    config = SimpleNamespace(dynamic_cp_runtime_config=str(path), **{**launch, **settings})
    return prefetch.runtime_spec(config, world or measured)


def _planner(tmp_path, run, **changes):
    """The planner the CPU worker builds for that run: the solver bound to the cost model."""
    return worker._initialize_planner(_spec(tmp_path, run, **changes))


@pytest.mark.parametrize('case', _pins('attention_frozen_predictions')['cases'])
def test_attention_predictor_gives_its_frozen_predictions(case):
    options = {name: case[name] for name in ('sequence_alignment', 'pack_alignment', 'ranks')}
    pack = (case['hardware'], *case['geometry'], case['lengths'], case['cp_size'])
    pinned = (case['forward_ms'], case['backward_ms'])
    assert attention.predict_pair(*pack, **options) == pytest.approx(pinned, rel=1e-12)


def test_full_qwen35_modules_give_the_frozen_predictions_of_the_held_out_points(tmp_path):
    model = _planner(tmp_path, 'qwen35').model
    for case in _pins('qwen35_full_frozen_predictions')['cases']:
        point, phase = dict(case, logical_lengths=case['lengths']), case['phase']
        if case['kind'] == 'fa':
            price = geometry.dot(model.full_attention[phase], geometry.fa_features(point))
        else:  # GDN: the hot pass, and one constant more for a cold forward pass.
            cold = (phase, case['cache_pass']) == ('forward', 0)
            price = gdn_features.predict(model.gdn[phase], point, phase)
            price += model.gdn['first_layer_cold_forward_ms'] * cold
        assert price == pytest.approx(case['predicted_ms'], rel=2e-13), case


@pytest.mark.parametrize('case', _pins('qwen3_h100_p2p_frozen_costs')['cases'])
def test_h100_calibration_gives_the_predictions_of_its_fitter(tmp_path, case):
    model, lengths = _planner(tmp_path, 'h100').model, tuple(case['lengths'])
    priced = model.forward(lengths, case['cp']), model.backward(lengths, case['cp'])
    assert priced == pytest.approx((case['forward_ms'], case['backward_ms']), abs=1e-9)


def test_original_profile_gives_the_frozen_upstream_group_costs_for_every_table(tmp_path):
    table = json.loads((PACKAGE / 'models/usp_calib_v1.json').read_text())
    variants = [dict(moe_rate_lut=None, groups=_pins('usp_frozen_costs')['cases'])]
    for variant in variants + _pins('usp_frozen_plans')['table_variants']['cases']:
        lut = variant['moe_rate_lut']
        if lut:  # A table beside the runtime file replaces the coefficients, not the formulas.
            table['compute']['moe_rate_lut'].update(lut)
            (tmp_path / 'table.json').write_text(json.dumps(table))
        model = _planner(tmp_path, 'usp', file=dict(calibration=lut and 'table.json')).model
        for group in variant['groups']:
            lengths, cp = tuple(group['lengths']), group['cp']
            parts = model.components(lengths, cp)
            for name in ('compute_ms', 'a2a_ms', 'ring_ms'):
                assert parts[name] == pytest.approx(group[name], rel=2e-15, abs=1e-12)
            stack = 44 * (model.forward(lengths, cp) + model.backward(lengths, cp))  # ms per layer
            assert stack == pytest.approx(parts['compute_ms'] + parts['a2a_ms'] + parts['ring_ms'])


def test_h100_calibration_plans_are_the_pinned_ones_and_other_batches_are_refused(tmp_path):
    spec = _spec(tmp_path, 'h100')
    planner = worker._initialize_planner(spec)
    for case in _pins('usp_frozen_plans')['runtime_h100_plans']['cases']:
        lengths = case['lengths']
        plan, statistics = planner.plan(dict(enumerate(lengths))), planner.last_statistics
        assert [[list(group) for group in row] for row in plan.groups] == case['layout']
        assert plan.predicted_ms == pytest.approx(case['predicted_ms'], rel=1e-12)
        assert statistics['candidates'] == [pytest.approx(c, rel=1e-12) for c in case['candidates']]
        assert statistics['stage1_weight'] == 'cost_model_sample_work'  # No A, B, C were fitted.
        rows = [[ids for _, cp, ids in row for _ in range(cp)] for row in case['layout']]
        assert worker._solve(planner, spec, lengths, lengths, None)['rows'] == rows  # Per rank.
    # The worker refuses a batch outside the measurement: 256 samples of up to 16384 tokens.
    with pytest.raises(RuntimeScheduleError, match='256 samples per global batch; this .* 255'):
        worker._solve(planner, spec, lengths[1:], lengths[1:], None)
    lengths = [20032] + lengths[1:]
    with pytest.raises(RuntimeScheduleError, match='sample 0: seq_len=20032 is longer than the'):
        worker._solve(planner, spec, lengths, lengths, None)


def test_h100_refit_file_beside_the_runtime_file_brings_its_own_domain(tmp_path):
    refit = json.loads((PACKAGE / 'models/qwen3_h100_p2p.json').read_text())
    refit['domain'].update(step_overhead_global_batch_size=64, max_logical_length=8192)
    refit['domain']['config']['hidden_dropout'] = 0.1  # The launch is compared with the refit.
    (tmp_path / 'refit.json').write_text(json.dumps(refit))
    spec = _spec(tmp_path, 'h100', file=dict(calibration='refit.json'), hidden_dropout=0.1)
    limits = worker._initialize_planner(spec).model.pack_limits(8)
    assert (spec['required_global_batch_size'], limits.max_sample_length) == (64, 8192)


@pytest.mark.parametrize(
    'run,selected',  # Cost model, solver, CP group sizes, NVLink domain, required global batch.
    [
        ('gqa', ('dcp_benchmark', 'lightweight', (1, 2, 4, 8, 16, 32, 64), 8, None)),
        ('qwen35', ('qwen35_full', 'lightweight', (1, 2, 4, 8, 16, 32), 72, None)),
        ('h100', ('qwen3_h100_p2p', 'dynamic_usp', (1, 2, 4, 8), 8, 256)),
        ('usp', ('usp_calib_v1', 'dynamic_usp', (8, 16, 32), 8, None)),
    ],
)
def test_runtime_file_and_launch_select_the_cost_model_and_its_limits(tmp_path, run, selected):
    spec = _spec(tmp_path, run)
    planner = worker._initialize_planner(spec)
    solver, sizes = type(planner).__module__.rsplit('.')[-1], spec['geometry']['allowed_cp_sizes']
    domain, batch = spec['nvlink_domain_size'], spec.get('required_global_batch_size')
    assert (spec['model'], solver, sizes, domain, batch) == selected
    assert planner.model.metadata()['nvlink_domain_size'] == domain  # In every plan record.
    assert prefetch._coverage_error(spec) is None  # The start-up probe prices every CP size.
    assert planner.costs.pack((1024,), 3, 0) is None  # No price for a CP size outside the list.


@pytest.mark.parametrize(
    'run,changes,phrase',
    [
        ('gqa', dict(recompute_granularity='selective'), "recompute_granularity None or 'full'"),
        ('qwen35', dict(file=dict(hardware='h100')), '"hardware" must be "gb200" for full Qwen3.5'),
        ('qwen35', dict(num_layers=4), 'require num_layers=40, got 4'),
        ('qwen35', dict(cp_comm_type='a2a'), "cp_comm_type='p2p', got 'a2a'"),
        ('qwen35', dict(world=64), 'measured with DPxCP=32, got 64'),
        ('h100', dict(file=dict(hardware='gb200')), "is packaged for hardware 'gb200'"),
        ('h100', dict(file=dict(max_cp_size=None)), 'Set "max_cp_size": 8'),
        ('h100', dict(expert_model_parallel_size=4), 'expert_model_parallel_size=8, got 4'),
        ('h100', dict(num_layers=4), 'num_layers=48, got 4'),
        ('h100', dict(mtp_num_layers=1), 'mtp_num_layers=None or 0, got 1'),
        ('h100', dict(pad_packed_seq_alignment=6144), 'fixed per-rank buffer'),
    ],
)
def test_start_up_names_what_is_outside_the_cost_model(tmp_path, run, changes, phrase):
    with pytest.raises(ValueError) as caught:
        _spec(tmp_path, run, **changes)
    # Only a fault of the launch is reported as a mismatch with the training configuration.
    mismatch = 'does not match the training configuration' in str(caught.value)
    assert phrase in str(caught.value) and mismatch == ('file' not in changes)


def test_ordinary_gqa_prices_a_group_on_its_ranks_in_a_buffer_that_grows_with_its_size(tmp_path):
    costs, pack = _planner(tmp_path, 'gqa').costs, (1001, 2001)
    # Ranks 6 to 9 lie on two H100 nodes; DPxCP 64 stores a sequence in a multiple of 128 tokens.
    for cp, ranks in ((1, [8]), (4, [6, 7, 8, 9]), (8, [*range(8, 16)])):
        stored = dict(sequence_alignment=128, pack_alignment=cp * 4096, ranks=ranks)
        priced = attention.predict_pair('h100', 32, 4, 128, pack, cp, **stored)
        assert costs.pack(pack, cp, ranks[0])[:2] == priced


# Two microbatches on four ranks (four CP1 groups; two CP1 groups and one CP2 group), simulated
# rank by rank: a layer is the attention of each CP group, then fixed work that ends with the
# slowest rank of an EP block; full recomputation replays each forward layer in the backward pass.
@pytest.mark.parametrize('ep,expected', [(1, (46.0, 60.0)), (2, (49.0, 65.0)), (4, (51.5, 69.5))])
def test_ordinary_gqa_step_time_follows_ep_blocks_and_recomputation(tmp_path, ep, expected):
    rounds = [[(1, [0]), (1, [1]), (1, [2]), (1, [3])], [(1, [4]), (1, [5]), (2, [6])]]
    rounds = [[dict(cp=cp, sample_ids=ids) for cp, ids in row] for row in rounds]
    lengths = dict(enumerate((3072, 2048, 4096, 4096, 3072, 4096, 3072)))
    for recompute, expected_ms in zip((None, 'full'), expected):
        launch = dict(recompute_granularity=recompute, recompute_method='uniform')
        planner = _planner(tmp_path, 'gqa', world=4, expert_model_parallel_size=ep, **launch)
        planner.model.predictor = lambda pack, cp, _: (sum(pack) / 1024 / cp, sum(pack) / 512 / cp)
        step_ms = planner.timeline.evaluate(planner._costed(rounds, lengths))
        assert step_ms == pytest.approx(expected_ms)


def test_full_qwen35_step_is_its_41_attention_modules_plus_the_times_of_each_layer_type(tmp_path):
    forward, backward = {'gdn': 0.5, 'gqa': 1.0, 'mtp': 3.0}, {'gdn': 1.5, 'gqa': 3.0}
    times = dict(forward_ms_per_layer=forward, backward_ms_per_layer=backward)
    planner = _planner(tmp_path, 'qwen35', file=dict(non_attention=times))
    model, lengths, rounds = planner.model, {0: 12472, 1: 12492}, [[dict(cp=2, sample_ids=[0, 1])]]
    assert [layer['kind'] for layer in model.layers] == ['gdn', 'gdn', 'gdn', 'fa'] * 10 + ['fa']
    # The run stores a sequence in a multiple of 64 tokens and of 2 x CP; the tail extends the last.
    point, phases = geometry.pack_geometry(tuple(lengths.values()), 2), ('forward', 'backward')
    assert point['storage_lengths'] == [12480, 20288]
    fa = sum(geometry.dot(model.full_attention[p], geometry.fa_features(point)) for p in phases)
    gdn = sum(gdn_features.predict(model.gdn[p], point, p) for p in phases)
    # 30 GDN modules, the first with a cold forward pass at CP above 1, and 11 of full attention.
    modules_ms = 30 * gdn + 11 * fa + model.gdn['first_layer_cold_forward_ms']
    # 30 GDN layers, 10 full-attention layers and the MTP layer, which takes the gqa backward time.
    fixed_ms = 30 * (0.5 + 1.5) + 10 * (1.0 + 3.0) + (3.0 + 3.0)
    step_ms = planner.timeline.evaluate(planner._costed(rounds, lengths))
    assert step_ms == pytest.approx(modules_ms + fixed_ms, rel=1e-12)
    hot = model.program_costs((1025,), 1, ranks=[0])  # At CP 1 the first GDN layer is hot as well.
    assert len({hot[layer['name'] + ':attention:forward'] for layer in model.layers[:3]}) == 1


def test_origin_records_the_digest_of_every_packaged_json_file():
    recorded = json.loads((PACKAGE / 'ORIGIN.json').read_text())['runtime_parameters_sha256']
    files = set(PACKAGE.rglob('*.json')) - {PACKAGE / 'ORIGIN.json'}
    on_disk = {path.relative_to(PACKAGE).as_posix(): path.read_bytes() for path in files}
    assert recorded == {name: hashlib.sha256(raw).hexdigest() for name, raw in on_disk.items()}
