# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Temporary upstream U8 cost proxy, not U8 communication or a P2P calibration."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets import dcp_runtime_worker
from megatron.core.datasets import dcp_schedule_prefetch as prefetch
from megatron.core.datasets.dcp_runtime_scheduler.models import make_model
from megatron.core.datasets.dcp_runtime_scheduler.runtime.cost_model import build_timeline
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import RuntimePlanGeometry
from megatron.core.datasets.dcp_runtime_scheduler.usp_calib_v1 import USPCalibV1CostModel


def _geometry(world=32):
    return RuntimePlanGeometry(world, 1024, 1, tuple(c for c in (8, 16, 32) if c <= world), 64, 64)


@pytest.mark.parametrize("world", [8, 16, 32])
def test_temporary_recipe_only_exposes_supported_powers(world):
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=1024,
        min_dynamic_context_parallel_size=8,
        dynamic_cp_runtime_model="usp_calib_v1",
        dynamic_cp_runtime_model_options='{"allow_provisional":true}',
        dynamic_cp_runtime_solver="dynamic_usp",
    )
    spec = prefetch.runtime_spec(config, world)
    assert spec["geometry"]["allowed_cp_sizes"] == tuple(c for c in (8, 16, 32) if c <= world)
    assert spec["solver"] == "dynamic_usp"
    # No communication option is introduced or rewritten by cost-model selection.
    assert not hasattr(config, "cp_comm_type")


@pytest.mark.parametrize("world,minimum", [(4, 8), (8, 16), (24, 8), (64, 8), (32, 1), (32, 24)])
def test_out_of_profile_recipe_is_rejected_before_worker_start(world, minimum):
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=1024,
        min_dynamic_context_parallel_size=minimum,
        dynamic_cp_runtime_model="usp_calib_v1",
    )
    with pytest.raises(ValueError, match="usp_calib_v1 requires"):
        prefetch.runtime_spec(config, world)


def test_proxy_is_explicit_and_keeps_the_original_asset_and_scope():
    with pytest.raises(ValueError, match="allow_provisional"):
        USPCalibV1CostModel(_geometry())
    with pytest.raises(ValueError, match="minimum CP8"):
        USPCalibV1CostModel(replace(_geometry(), allowed_cp_sizes=(4, 8)), allow_provisional=True)
    model = make_model("usp_calib_v1", _geometry(), allow_provisional=True)
    assert model.source_hashes["calib_table_v1.json"] == (
        "d25989552ab9700f07261231fa13d66672d98684435fff57838442ca12c19cd7"
    )
    assert model.metadata()["provisional"]
    assert model.metadata()["modeled_transport"] == "U8 all-to-all + ring"
    assert "not a Qwen calibration" in model.metadata()["shape"]


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


def test_persistent_worker_uses_proxy_without_importing_upstream_or_cuda():
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=1024,
        min_dynamic_context_parallel_size=8,
        dynamic_cp_runtime_model="usp_calib_v1",
        dynamic_cp_runtime_model_options='{"allow_provisional":true}',
        dynamic_cp_runtime_solver="dynamic_usp",
    )
    spec = prefetch.runtime_spec(config, 32)
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


def test_cpu_only_guard_also_covers_model_metadata(monkeypatch):
    state = {"cuda": False}

    def metadata():
        state["cuda"] = True

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: state["cuda"])
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    planner = SimpleNamespace(geometry=_geometry(8), model=SimpleNamespace(metadata=metadata))
    with pytest.raises(RuntimeError, match="must not initialize CUDA"):
        dcp_runtime_worker._solve(planner, {"force_powers": True}, [64], [64], [[[0]] * 8])


@pytest.mark.parametrize(
    "case", json.loads(Path(__file__).with_name("dcp_usp_frozen_costs.json").read_text())["cases"]
)
def test_default_profile_matches_frozen_upstream_group_costs(case):
    model = USPCalibV1CostModel(_geometry(), allow_provisional=True)
    parts = model.components(tuple(case["lengths"]), case["cp"])
    for name in ("compute_ms", "a2a_ms", "ring_ms"):
        assert parts[name] == pytest.approx(case[name], rel=2e-15, abs=1e-12)
