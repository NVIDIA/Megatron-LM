# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime integration of the benchmark's core-attention and E2E predictors.

Synthetic timings below test scheduling semantics, not hardware accuracy.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from megatron.core.datasets.dcp_runtime_scheduler.models import benchmark as benchmark_module
from megatron.core.datasets.dcp_runtime_scheduler.models import make_model
from megatron.core.datasets.dcp_runtime_scheduler.models._benchmark.model_e2e import (
    ModelConfig,
    ModelEvaluator,
    NonAttentionTimes,
    PackGroup,
)
from megatron.core.datasets.dcp_runtime_scheduler.models.benchmark import BenchmarkCostModel
from megatron.core.datasets.dcp_runtime_scheduler.runtime.dynamic_usp import (
    DynamicUSPRuntimePlanner,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.lightweight import (
    LightweightRuntimePlanner,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import (
    RuntimePlanGeometry,
    validate_rows,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.timeline import build_timeline


class SyntheticCore:
    """Independent F/B, placement-aware CPU fixture with an explicit core scope."""

    def __init__(self, kind="gqa", scale=1.0, scope="core_attention"):
        self.attention_kind = kind
        self.scale = scale
        self.scope = scope
        self.calls = []

    def predict(self, lengths, cp_size=1, *, phase, ranks):
        lengths, ranks = tuple(lengths), tuple(ranks)
        self.calls.append((lengths, cp_size, phase, ranks))
        direction = {"forward": 1.0, "backward": 2.0}[phase]
        return {
            "time_ms": self.scale * direction * sum(lengths) / (1024 * cp_size),
            "scope": self.scope,
            "provisional": True,
        }


def _geometry(world=4, ep=2, allowed=None):
    if allowed is None:
        allowed = tuple(cp for cp in (1, 2, 4, 8, 16, 32) if cp <= world)
    return RuntimePlanGeometry(world, 4096, ep, allowed, 64, 64, True)


def _non_attention(forward=0.5, backward=1.0, tail=0.0):
    return dict(
        forward_ms_per_layer=forward,
        backward_ms_per_layer=backward,
        source="SYNTHETIC TEST INPUT; not a measurement",
        iteration_tail_ms=tail,
    )


def _options(**overrides):
    options = dict(
        hardware="h100",
        profile="legacy_frozen_20260922",
        query_heads=32,
        kv_heads=4,
        head_dim=128,
        layer_count=2,
        non_attention=_non_attention(),
        allow_provisional=True,
    )
    options.update(overrides)
    return options


def _model(geometry=None, predictor=None, **options):
    geometry = _geometry() if geometry is None else geometry
    if predictor is None:
        predictor = SyntheticCore()
    options.setdefault("predictors", {"gqa": predictor})
    return BenchmarkCostModel(geometry, **_options(**options))


def _source_prediction(geometry, rounds, lengths, predictors, **options):
    options = _options(**options)
    layer_types = options.get("layer_types") or ("gqa",) * options["layer_count"]
    evaluator = ModelEvaluator(
        ModelConfig(
            layer_types,
            world_size=geometry.world_size,
            ep_size=geometry.ep_size,
            first_forward_dp_cp_sync=options.get("first_forward_dp_cp_sync", True),
            recompute_forward=options.get("recompute_forward", False),
        ),
        NonAttentionTimes(**options["non_attention"]),
        predictors,
    )
    groups = []
    for row in rounds:
        start, source_row = 0, []
        for group in row:
            cp = group["cp"]
            source_row.append(
                PackGroup(
                    tuple(lengths[i] for i in group["sample_ids"]), tuple(range(start, start + cp))
                )
            )
            start += cp
        groups.append(source_row)
    return evaluator.predict(groups)


def test_non_attention_times_remain_required():
    options = _options()
    del options["non_attention"]
    with pytest.raises(TypeError):
        BenchmarkCostModel(_geometry(), **options)


@pytest.mark.parametrize("allowed", [(1,), (1, 2, 4)])
def test_benchmark_defaults_need_only_hardware_geometry_and_non_attention_times(allowed):
    predictor = SyntheticCore()
    measurements = dict(forward_ms_per_layer=0.5, backward_ms_per_layer=1.0)
    model = BenchmarkCostModel(
        _geometry(allowed=allowed),
        hardware="h100",
        query_heads=32,
        kv_heads=4,
        head_dim=128,
        non_attention=measurements,
        predictors={"gqa": predictor},
    )
    assert model.profile == "legacy_frozen_20260922"
    assert model.predictor_options["allow_provisional"] is True
    assert model.non_attention.source == "user-provided non-attention times"
    assert model.metadata()["provisional"] is True
    assert predictor.calls == []
    assert "source" not in measurements


def test_explicit_non_attention_source_must_be_nonempty():
    for source in ("", "  ", None):
        with pytest.raises((TypeError, ValueError)):
            _model(non_attention=dict(_non_attention(), source=source)).timing()


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_invalid_fixed_measurements_fail_before_schedule_execution(value, phase):
    measurements = _non_attention()
    measurements[phase + "_ms_per_layer"] = value
    with pytest.raises((TypeError, ValueError)):
        _model(non_attention=measurements).timing()


def test_serial_step_charges_core_and_fixed_non_attention_once():
    geometry = _geometry(world=1, ep=1)
    model = _model(geometry, non_attention=_non_attention(tail=7.0))
    planner = DynamicUSPRuntimePlanner(geometry, model)
    rounds = [[dict(cp=1, sample_ids=[0])]]
    # Two layers, each F=1, B=2, fixed non-attention F=.5/B=1, plus one tail.
    assert planner.timeline.evaluate(planner._costed(rounds, {0: 1024})) == pytest.approx(16.0)


def test_backward_reverses_non_attention_and_attention_dependencies():
    program = _model(layer_count=1).timing()["execution_program"][0]
    assert program["forward"] == [
        {"sync": "cp", "cost": "gqa:forward"},
        {"sync": "ep", "duration_ms": 0.5},
    ]
    assert program["backward"] == [
        {"sync": "ep", "duration_ms": 1.0},
        {"sync": "cp", "cost": "gqa:backward"},
    ]


@pytest.mark.parametrize("first_sync", [False, True])
@pytest.mark.parametrize("ep", [1, 2, 4])
@pytest.mark.parametrize("recompute", [False, True])
def test_runtime_scores_match_source_e2e_for_mixed_cp_and_ep(ep, first_sync, recompute):
    geometry = _geometry(ep=ep)
    core = SyntheticCore()
    options = dict(
        first_forward_dp_cp_sync=first_sync,
        recompute_forward=recompute,
        non_attention=_non_attention(tail=3.0),
    )
    model = _model(geometry, core, **options)
    planner = DynamicUSPRuntimePlanner(geometry, model)
    lengths = {0: 3072, 1: 1024, 2: 2048, 3: 4096, 4: 1024, 5: 4096}
    rounds = [
        [dict(cp=1, sample_ids=[0]), dict(cp=1, sample_ids=[1]), dict(cp=2, sample_ids=[2])],
        [dict(cp=4, sample_ids=[3])],
        [dict(cp=2, sample_ids=[4]), dict(cp=2, sample_ids=[5])],
    ]
    expected = _source_prediction(geometry, rounds, lengths, {"gqa": core}, **options)
    costed = planner._costed(rounds, lengths)
    assert planner.timeline.evaluate(costed) == pytest.approx(expected.time_ms, abs=1e-10)
    state = planner.timeline.initial_state()
    for index, groups in enumerate(costed):
        state = planner.timeline.advance(state, groups, index == 0)
    assert planner.timeline.makespan(state) == pytest.approx(expected.time_ms, abs=1e-10)
    assert any(call[3] == (2, 3) for call in core.calls)


def test_ep_rendezvous_changes_e2e_even_with_the_same_attention_calls():
    rounds = [
        [dict(cp=1, sample_ids=[0]), dict(cp=1, sample_ids=[1])],
        [dict(cp=1, sample_ids=[2]), dict(cp=1, sample_ids=[3])],
    ]
    lengths = {0: 4096, 1: 1024, 2: 1024, 3: 4096}
    times = []
    for ep in (1, 2):
        geometry = _geometry(world=2, ep=ep)
        planner = DynamicUSPRuntimePlanner(
            geometry,
            _model(
                geometry,
                layer_count=1,
                first_forward_dp_cp_sync=False,
                non_attention=_non_attention(0.0, 0.0),
            ),
        )
        times.append(planner.timeline.evaluate(planner._costed(rounds, lengths)))
    assert times[0] == pytest.approx(15.0)
    assert times[1] > times[0]


def test_mixed_layers_require_their_own_core_predictors_and_fixed_inputs():
    geometry = _geometry(world=1, ep=1)
    kinds = ("gqa", "gdn")
    with pytest.raises(ValueError, match="gdn"):
        _model(geometry, layer_types=kinds).timing()
    predictors = {"gqa": SyntheticCore(), "gdn": SyntheticCore("gdn", scale=3.0)}
    measurements = _non_attention({"gqa": 0.5, "gdn": 2.0}, {"gqa": 1.0, "gdn": 4.0})
    options = dict(layer_types=kinds, non_attention=measurements)
    model = _model(geometry, predictors=predictors, **options)
    timeline = build_timeline(model, geometry)
    planner = DynamicUSPRuntimePlanner(geometry, model)
    rounds, lengths = [[dict(cp=1, sample_ids=[0])]], {0: 1024}
    expected = _source_prediction(geometry, rounds, lengths, predictors, **options)
    assert expected.time_ms == pytest.approx(19.5)
    assert timeline.evaluate(planner._costed(rounds, lengths)) == pytest.approx(expected.time_ms)


def test_full_attention_module_cannot_be_injected_as_core_attention():
    geometry = _geometry(world=1, ep=1)
    with pytest.raises(ValueError, match="scope"):
        model = _model(geometry, SyntheticCore(scope="self_attention_module"))
        model.program_costs((1024,), 1, ranks=(0,))


def test_predictor_mapping_contract_matches_the_source_e2e_api():
    class MappingCore(SyntheticCore):
        def predict(self, lengths, cp_size=1, *, phase, ranks):
            return MappingProxyType(super().predict(lengths, cp_size, phase=phase, ranks=ranks))

    geometry = _geometry(world=1, ep=1)
    model = _model(geometry, MappingCore(), layer_count=1)
    assert model.program_costs((1024,), 1, ranks=(0,)) == {"gqa:forward": 1.0, "gqa:backward": 2.0}


def test_cp1_pack_cache_preserves_a_custom_predictors_physical_rank():
    class RankSensitiveCore(SyntheticCore):
        def predict(self, lengths, cp_size=1, *, phase, ranks):
            estimate = super().predict(lengths, cp_size, phase=phase, ranks=ranks)
            estimate["time_ms"] *= ranks[0] + 1
            return estimate

    geometry = _geometry(world=2, ep=1, allowed=(1,))
    core = RankSensitiveCore()
    planner = DynamicUSPRuntimePlanner(geometry, _model(geometry, core, layer_count=1))
    first = planner.costs.pack((1024,), 1, 0)
    second = planner.costs.pack((1024,), 1, 1)
    assert first[:2] == (1.0, 2.0)
    assert second[:2] == (2.0, 4.0)
    assert {call[3] for call in core.calls} == {(0,), (1,)}
    calls = list(core.calls)
    assert planner.costs.pack((1024,), 1, 1) == second
    assert core.calls == calls
    assert planner.costs.stats["calls"] == 2
    assert planner.costs.stats["hits"] == 1


@pytest.mark.parametrize("planner_type", [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
def test_both_solvers_search_using_the_benchmark_cost_model(planner_type):
    geometry = _geometry()
    core = SyntheticCore()
    model = _model(geometry, core)
    planner = planner_type(geometry, model)
    lengths = {i: 512 + 64 * (i % 5) for i in range(16)}
    plan = planner.plan(lengths)
    validate_rows(plan.rows, lengths, geometry, 1)
    rounds = [[dict(cp=cp, sample_ids=ids) for _, cp, ids in row] for row in plan.groups]
    expected = _source_prediction(geometry, rounds, lengths, {"gqa": core})
    assert planner.model is model
    assert model.name == "dcp_benchmark"
    assert core.calls
    assert plan.predicted_ms == pytest.approx(expected.time_ms, abs=1e-10)


def test_public_attention_wrapper_preserves_per_rank_capacity_and_real_ranks(monkeypatch):
    instances, calls = [], []

    class RecordingPredictor:
        attention_kind = "gqa"

        def __init__(self, **options):
            self.__dict__.update(options)
            instances.append(options)

        def predict(self, lengths, cp_size=1, *, phase, ranks):
            from megatron.core.datasets.dcp_runtime_scheduler.models._benchmark.attention_predictor import (
                Pack,
            )

            pack = Pack.build(
                lengths, self.sequence_alignment, self.pack_alignment, cp_size=cp_size
            )
            calls.append((tuple(lengths), cp_size, phase, tuple(ranks), pack))
            return SimpleNamespace(
                time_ms=1.0 if phase == "forward" else 2.0,
                scope="core_attention",
                pack=pack,
                details={"synthetic": True},
                provisional=True,
            )

    monkeypatch.setattr(benchmark_module, "AttentionPredictor", RecordingPredictor)
    geometry = _geometry(world=32, ep=8, allowed=(1, 2, 4, 8))
    model = BenchmarkCostModel(geometry, **_options())
    for cp in geometry.allowed_cp_sizes:
        ranks = tuple(range(8, 8 + cp))
        assert model.forward((1001, 2001), cp, ranks=ranks) == 1.0
    assert {item["pack_alignment"] for item in instances} == {4096, 8192, 16384, 32768}
    for lengths, cp, phase, ranks, pack in calls:
        assert lengths == (1001, 2001)
        assert phase in ("forward", "backward")
        assert ranks == tuple(range(8, 8 + cp))
        assert pack.storage_tokens == cp * geometry.tokens_per_rank
        assert pack.storage_lengths[:-1] == pack.aligned_lengths[:-1]
        assert pack.storage_lengths[-1] >= pack.aligned_lengths[-1]
    assert {phase for _, _, phase, _, _ in calls} == {"forward", "backward"}


def test_experimental_cp1_profile_rejects_multicp_without_fallback():
    geometry = _geometry(allowed=(1, 2))
    with pytest.raises(ValueError, match="CP1|CP>1|multi.CP|unfinished"):
        model = BenchmarkCostModel(geometry, **_options(profile="experimental_cp1_bounded"))
        model.forward((1024,), 2, ranks=(0, 1))


def test_public_factory_selects_the_new_model():
    model = make_model(
        "dcp_benchmark", _geometry(), **_options(predictors={"gqa": SyntheticCore()})
    )
    assert isinstance(model, BenchmarkCostModel)
    assert model.name == "dcp_benchmark"


def test_cpu_worker_loads_the_model_by_file_path_without_gpu_libraries():
    root = Path(__file__).resolve().parents[3]
    worker = root / "megatron/core/datasets/dcp_runtime_worker.py"
    geometry = _geometry(world=1, ep=1, allowed=(1,))
    spec = dict(
        geometry=dict(
            world_size=geometry.world_size,
            tokens_per_rank=geometry.tokens_per_rank,
            ep_size=geometry.ep_size,
            allowed_cp_sizes=geometry.allowed_cp_sizes,
            dataset_alignment=geometry.dataset_alignment,
            max_group_seqs=geometry.max_group_seqs,
            align_group_starts=geometry.align_group_starts,
        ),
        type="dcp_benchmark",
        model="dcp_benchmark",
        model_options=_options(profile="experimental_cp1_bounded", layer_count=1),
        solver_options={"max_scored_candidates": 1},
    )
    script = """
import importlib.abc
import importlib.util
import json
import sys

class RejectGpuLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformer_engine', 'ortools', 'dcp_scheduler'}:
            raise ImportError('unexpected GPU or external scheduler dependency: ' + fullname)

sys.meta_path.insert(0, RejectGpuLibraries())
worker_path, planner_spec = sys.argv[1], json.loads(sys.argv[2])
sys.path.insert(0, worker_path.rsplit('/', 1)[0])
module_spec = importlib.util.spec_from_file_location('isolated_runtime_worker', worker_path)
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
planner = module._initialize_planner(planner_spec)
plan = planner.plan({0: 1024})
assert planner.model.name == 'dcp_benchmark'
assert plan.predicted_ms > 0
assert 'torch' not in sys.modules
print(json.dumps({'name': planner.model.name, 'solver': type(planner).__name__, 'rows': plan.rows}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(worker), json.dumps(spec)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert json.loads(result.stdout) == {
        "name": "dcp_benchmark",
        "solver": "LightweightRuntimePlanner",
        "rows": [[[0]]],
    }


# A corrupt default incumbent (unknown sample id) is discarded like an unsupported one.
@pytest.mark.parametrize("unknown_id", [False, True])
def test_cp1_worker_discards_an_invalid_default_incumbent(unknown_id):
    root = Path(__file__).resolve().parents[3]
    worker = root / "megatron/core/datasets/dcp_runtime_worker.py"
    spec = dict(
        geometry=dict(
            world_size=2,
            tokens_per_rank=4096,
            ep_size=1,
            allowed_cp_sizes=(1,),
            dataset_alignment=64,
            max_group_seqs=64,
            align_group_starts=True,
        ),
        type="dcp_benchmark",
        model="dcp_benchmark",
        model_options=_options(profile="experimental_cp1_bounded", layer_count=1),
        solver_options={"max_scored_candidates": 1},
    )
    ids = [0, 2] if unknown_id else [0, 1]
    request = [[1024, 2048], [1024, 2048], [[ids, ids]]]
    script = """
import importlib.util
import json
import sys

worker_path, spec, request = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
sys.path.insert(0, worker_path.rsplit('/', 1)[0])
module_spec = importlib.util.spec_from_file_location('isolated_runtime_worker', worker_path)
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
planner = module._initialize_planner(spec)
try:
    result = module._solve(planner, spec, *request)
except Exception as exc:
    print(json.dumps({'error': type(exc).__name__, 'message': str(exc)}))
else:
    print(json.dumps(result))
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(worker), json.dumps(spec), json.dumps(request)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    result = json.loads(process.stdout)
    assert "error" not in result
    assert result["statistics"]["incumbent_rejection"] == (
        "default_incumbent_outside_predictor_cp_domain"
    )
    assert result["statistics"]["incumbent_unsupported_cp_sizes"] == [2]
    assert result["predicted_ms"] > 0
    geometry = RuntimePlanGeometry(**spec["geometry"])
    groups = validate_rows(result["rows"], {0: 1024, 1: 2048}, geometry, 1)
    assert all(cp == 1 for row in groups for _, cp, _ in row)


class _FewSequencesAtCp4(SyntheticCore):
    """Like the packaged predictor at CP64 with DPxCP=64: many sequences are not priced."""

    def predict(self, lengths, cp_size=1, *, phase, ranks):
        if cp_size == 4 and len(lengths) > 3:
            raise ValueError('maximum_logical_tokens must be a positive integer, got -95')
        return super().predict(lengths, cp_size, phase=phase, ranks=ranks)


def test_benchmark_model_declares_how_many_sequences_its_predictor_prices():
    predictor = _FewSequencesAtCp4()
    model = _model(predictor=predictor)
    assert model.pack_limits(1).max_sequences is None
    assert model.pack_limits(2).max_sequences is None
    assert model.pack_limits(4).max_sequences == 3
    # Found once per CP size; a new global batch (clear) does not price the packs again.
    calls = len(predictor.calls)
    model.clear()
    assert model.pack_limits(4).max_sequences == 3 and len(predictor.calls) == calls


@pytest.mark.parametrize("planner_type", [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
def test_both_solvers_keep_packs_within_what_the_benchmark_predictor_prices(planner_type):
    geometry = _geometry()
    planner = planner_type(geometry, _model(geometry, predictor=_FewSequencesAtCp4()))
    lengths = {i: 64 * (1 + i % 5) for i in range(40)}
    plan = planner.plan_padded(lengths, incumbent_rows=[[list(range(40))] * 4])
    groups = validate_rows(plan.rows, lengths, geometry, 1)
    assert all(len(ids) <= 3 for row in groups for _, cp, ids in row if cp == 4)
    assert planner.sequence_limit(4) == 3
    # The trace lists the limits of the CP sizes this planner has considered so far.
    assert planner.last_statistics["model_sequence_limits"] in ({}, {4: 3})
