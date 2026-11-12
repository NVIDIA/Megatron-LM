# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Original DynamicUSP costs and their selection by the dynamic_usp scheduler type."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets import dcp_runtime_worker
from megatron.core.datasets import dcp_schedule_prefetch as prefetch
from megatron.core.datasets.dcp_runtime_scheduler.models import make_model, usp_calib_v1
from megatron.core.datasets.dcp_runtime_scheduler.models.usp_calib_v1 import USPCalibV1CostModel
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import RuntimePlanGeometry
from megatron.core.datasets.dcp_runtime_scheduler.runtime.timeline import build_timeline


def _geometry(world=32):
    return RuntimePlanGeometry(world, 1024, 1, tuple(c for c in (8, 16, 32) if c <= world), 64, 64)


def _json_training_config(tmp_path, *, minimum=8, **options):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(dict(type="dynamic_usp", **options)))
    return SimpleNamespace(
        dynamic_cp_runtime_config=str(path),
        max_seqlen_per_dp_cp_rank=1024,
        min_dynamic_context_parallel_size=minimum,
    )


def _calibration_table():
    return json.loads(Path(usp_calib_v1.__file__).with_suffix(".json").read_text())


def _write_calibration(tmp_path, table):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(table))
    return path


@pytest.mark.parametrize("world", [8, 16, 32])
def test_upstream_model_only_exposes_supported_powers(tmp_path, world):
    config = _json_training_config(tmp_path)
    spec = prefetch.runtime_spec(config, world)
    assert spec["geometry"]["allowed_cp_sizes"] == tuple(c for c in (8, 16, 32) if c <= world)
    assert spec["type"] == "dynamic_usp" and spec["model"] == "usp_calib_v1"
    # No communication option is introduced or rewritten by cost-model selection.
    assert not hasattr(config, "cp_comm_type")


def test_internal_model_name_is_not_a_scheduler_type(tmp_path):
    config = _json_training_config(tmp_path)
    Path(config.dynamic_cp_runtime_config).write_text('{"type": "usp_calib_v1"}')
    with pytest.raises(ValueError, match="'type' must be one of") as caught:
        prefetch.runtime_spec(config, 32)
    assert "'dynamic_usp'" in str(caught.value)


@pytest.mark.parametrize(
    "minimum,maximum,allowed",
    [(8, 8, (8,)), (8, 16, (8, 16)), (8, 32, (8, 16, 32)), (16, 16, (16,)), (32, 32, (32,))],
)
def test_single_json_selects_the_upstream_scheme_with_a_cp_cap(tmp_path, minimum, maximum, allowed):
    config = _json_training_config(tmp_path, minimum=minimum, max_cp_size=maximum)
    spec = prefetch.runtime_spec(config, 32)
    assert spec["type"] == "dynamic_usp"
    assert spec["model"] == "usp_calib_v1"
    assert spec["model_options"] == {"allow_provisional": True}
    assert spec["geometry"]["allowed_cp_sizes"] == allowed
    assert spec["geometry"]["max_cp_size"] == maximum
    planner = dcp_runtime_worker._initialize_planner(spec)
    assert isinstance(planner.model, USPCalibV1CostModel)
    assert planner.geometry.allowed_cp_sizes == allowed
    assert planner.model.metadata()["modeled_transport"] == "U8 all-to-all + ring"
    assert type(planner).__name__ == "DynamicUSPRuntimePlanner"
    assert "solver" not in spec
    assert not hasattr(config, "cp_comm_type")


@pytest.mark.parametrize("world,minimum", [(4, 8), (8, 16), (24, 8), (64, 8), (32, 1), (32, 24)])
def test_single_json_rejects_unsupported_upstream_domain_before_worker_start(
    tmp_path, world, minimum
):
    config = _json_training_config(tmp_path, minimum=minimum)
    with pytest.raises(ValueError, match="type 'dynamic_usp' requires"):
        prefetch.runtime_spec(config, world)


@pytest.mark.parametrize("minimum,maximum", [(8, 12), (8, 24), (16, 24)])
def test_single_json_cap_must_be_one_of_the_tiers(tmp_path, minimum, maximum):
    config = _json_training_config(tmp_path, minimum=minimum, max_cp_size=maximum)
    with pytest.raises(ValueError, match=f"max_cp_size={maximum} is not a CP group size") as caught:
        prefetch.runtime_spec(config, 32)
    assert str([cp for cp in (8, 16, 32) if cp >= minimum]) in str(caught.value)


def test_single_json_cannot_cap_upstream_model_below_minimum_cp(tmp_path):
    config = _json_training_config(tmp_path, max_cp_size=4)
    with pytest.raises(ValueError, match="max_cp_size"):
        prefetch.runtime_spec(config, 32)


def test_proxy_is_explicit_and_keeps_the_original_asset_and_scope():
    with pytest.raises(ValueError, match="allow_provisional"):
        USPCalibV1CostModel(_geometry())
    with pytest.raises(ValueError, match="minimum CP8"):
        USPCalibV1CostModel(replace(_geometry(), allowed_cp_sizes=(4, 8)), allow_provisional=True)
    model = make_model("usp_calib_v1", _geometry(), allow_provisional=True)
    assert isinstance(model, USPCalibV1CostModel)
    assert model.source_hashes["calib_table_v1.json"] == (
        "d25989552ab9700f07261231fa13d66672d98684435fff57838442ca12c19cd7"
    )
    assert model.metadata()["provisional"]
    assert model.metadata()["modeled_transport"] == "U8 all-to-all + ring"
    assert "not a Qwen calibration" in model.metadata()["shape"]


@pytest.mark.parametrize("cp", [8, 16, 32])
def test_external_clone_preserves_original_costs_and_records_loaded_bytes(tmp_path, cp):
    path = _write_calibration(tmp_path, _calibration_table())
    bundled = USPCalibV1CostModel(_geometry(), allow_provisional=True)
    external = USPCalibV1CostModel(_geometry(), calibration=path, allow_provisional=True)
    assert external.components((7001, 1025), cp) == bundled.components((7001, 1025), cp)
    assert external.metadata()["calibration_path"] == str(path.resolve())
    assert (
        external.metadata()["calibration_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_external_calibration_changes_compute_and_global_tail_without_refitting_shape(tmp_path):
    table = _calibration_table()
    table["compute"]["fa_additive_law"]["fwd"]["P_tflops"] /= 2
    table["global"]["C_global_clean_ms"] = 123
    path = _write_calibration(tmp_path, table)
    geometry = _geometry()
    bundled = USPCalibV1CostModel(geometry, allow_provisional=True)
    model = USPCalibV1CostModel(geometry, calibration=path, allow_provisional=True)
    lengths, cp = (7001, 1025), 32
    parts = model.components(lengths, cp)
    assert model.forward(lengths, cp) > bundled.forward(lengths, cp)
    assert model.backward(lengths, cp) == bundled.backward(lengths, cp)
    assert parts["a2a_ms"] == bundled.components(lengths, cp)["a2a_ms"]
    assert parts["ring_ms"] == bundled.components(lengths, cp)["ring_ms"]
    assert model.layer_count == 44
    groups = [
        (
            0,
            cp,
            model.forward(lengths, cp),
            model.backward(lengths, cp),
            model.program_costs(lengths, cp),
        )
    ]
    assert build_timeline(model, geometry).evaluate([groups]) == pytest.approx(
        sum(parts[name] for name in ("compute_ms", "a2a_ms", "ring_ms")) + 123, rel=1e-13
    )


def test_runtime_resolves_external_calibration_relative_to_the_runtime_json(tmp_path):
    path = _write_calibration(tmp_path, _calibration_table())
    config = _json_training_config(tmp_path, calibration=path.name)
    spec = prefetch.runtime_spec(config, 32)
    assert spec["model_options"]["calibration"] == str(path.resolve())
    planner = dcp_runtime_worker._initialize_planner(spec)
    assert (
        planner.model.metadata()["calibration_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "keys,value",
    [
        (("compute", "fa_additive_law", "fwd", "P_tflops"), float("nan")),
        (("compute", "fa_additive_law", "bwd", "P_tflops"), 0),
        (("a2a", "fits", "U8_pre", "bw_gbps"), -1),
        (("global", "gamma_comp", "pure_compute", "sp16_ring", "fwd"), True),
        (("global", "C_global_clean_ms"), -1),
        (("compute", "moe_rate_lut", "tokens"), [128, 128, 16384]),
        (("compute", "moe_rate_lut", "tokens"), [128, 8192]),
        (("compute", "moe_rate_lut", "tflops"), [1]),
        (("ring", "beta_by_chunk_bytes"), {"note": "no fitted sizes"}),
        (("ring", "beta_by_chunk_bytes"), {"1MB": 0}),
    ],
)
def test_external_calibration_rejects_invalid_consumed_coefficients(tmp_path, keys, value):
    table = _calibration_table()
    target = table
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path = _write_calibration(tmp_path, table)
    with pytest.raises(ValueError):
        USPCalibV1CostModel(_geometry(), calibration=path, allow_provisional=True)


@pytest.mark.parametrize(
    "keys",
    [
        ("compute", "fa_additive_law", "fwd", "P_tflops"),
        ("a2a", "fits", "U8_pre", "bw_gbps"),
        ("ring", "fits", "dkv", "t0_ms"),
        ("global", "gamma_comp", "pure_compute", "sp32_ring", "bwd"),
    ],
)
def test_external_calibration_rejects_missing_consumed_coefficients(tmp_path, keys):
    table = _calibration_table()
    target = table
    for key in keys[:-1]:
        target = target[key]
    del target[keys[-1]]
    path = _write_calibration(tmp_path, table)
    with pytest.raises(ValueError):
        USPCalibV1CostModel(_geometry(), calibration=path, allow_provisional=True)


@pytest.mark.parametrize("contents", [None, "{", "[]"])
def test_external_calibration_rejects_unreadable_or_invalid_json(tmp_path, contents):
    path = tmp_path / "calibration.json"
    if contents is not None:
        path.write_text(contents)
    with pytest.raises(ValueError):
        USPCalibV1CostModel(_geometry(), calibration=path, allow_provisional=True)


@pytest.mark.parametrize("cp", [8, 16, 32])
def test_full_stack_is_counted_once_and_global_constant_once(cp):
    geometry = _geometry(cp)
    model = USPCalibV1CostModel(geometry, allow_provisional=True)
    lengths = (7001, 1025)
    parts = model.components(lengths, cp)
    total = sum(parts[name] for name in ("compute_ms", "a2a_ms", "ring_ms"))
    assert model.layer_count * (model.forward(lengths, cp) + model.backward(lengths, cp)) == (
        pytest.approx(total, rel=1e-14)
    )
    program = model.program_costs(lengths, cp)
    timeline = build_timeline(model, geometry)
    groups = [(0, cp, model.forward(lengths, cp), model.backward(lengths, cp), program)]
    assert timeline.evaluate([groups]) == pytest.approx(total + 2000, rel=1e-13)


@pytest.mark.parametrize("lengths,cp", [((1,), 24), ((1,), 4), ((0,), 8), ((), 8), ((True,), 8)])
def test_invalid_model_queries_do_not_silently_extrapolate(lengths, cp):
    model = USPCalibV1CostModel(_geometry(), allow_provisional=True)
    with pytest.raises(ValueError):
        model.forward(lengths, cp)


def test_persistent_worker_uses_proxy_without_importing_upstream_or_cuda(tmp_path):
    spec = prefetch.runtime_spec(_json_training_config(tmp_path), 32)
    lengths, padded, baseline = [7200, 7300, 15000], [7232, 7360, 15040], [[[0, 1, 2]] * 32]
    worker = prefetch._CpuPlanner(spec)
    try:
        first = worker.solve(lengths, padded, baseline)
        with ThreadPoolExecutor(max_workers=1) as executor:
            second = executor.submit(worker.solve, lengths, padded, baseline).result(timeout=30)
    finally:
        worker.close()
    assert first["rows"] == second["rows"]
    assert first["predicted_ms"] == second["predicted_ms"]
    assert first["worker_pid"] == second["worker_pid"]
    for result in (first, second):
        assert result["statistics"]["group_sizes"] == [8, 16, 32]
        assert not result["worker_cuda_initialized"]
        assert not result["worker_distributed_initialized"]
        assert not result["external_scheduler_imported"]
        assert result["model_metadata"]["name"] == "usp_calib_v1"
        assert result["model_metadata"]["nvlink_domain_size"] == 8
        assert "DynamicUSPRuntimePlanner.plan_padded" in result["solver_entry"]


def test_cpu_only_guard_also_covers_model_metadata(monkeypatch):
    state = {"cuda": False}

    def metadata():
        state["cuda"] = True

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: state["cuda"])
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    planner = SimpleNamespace(
        geometry=_geometry(8),
        model=SimpleNamespace(metadata=metadata),
        last_statistics={},
        plan_padded=lambda *args, **kwargs: SimpleNamespace(rows=[[[0]] * 8], predicted_ms=0.0),
    )
    with pytest.raises(RuntimeError, match="must not initialize CUDA"):
        dcp_runtime_worker._solve(planner, {}, [64], [64], None)


@pytest.mark.parametrize(
    "case", json.loads(Path(__file__).with_name("dcp_usp_frozen_costs.json").read_text())["cases"]
)
def test_default_profile_matches_frozen_upstream_group_costs(case):
    model = USPCalibV1CostModel(_geometry(), allow_provisional=True)
    parts = model.components(tuple(case["lengths"]), case["cp"])
    for name in ("compute_ms", "a2a_ms", "ring_ms"):
        assert parts[name] == pytest.approx(case[name], rel=2e-15, abs=1e-12)
