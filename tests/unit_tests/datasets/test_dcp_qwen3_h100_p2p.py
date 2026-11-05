# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Measured-model contracts using explicitly synthetic coefficients, not GPU evidence."""

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatron.core.datasets import dcp_schedule_prefetch as prefetch
from megatron.core.datasets.dcp_runtime_scheduler.models import make_model
from megatron.core.datasets.dcp_runtime_scheduler.models.qwen3_h100_p2p import (
    Qwen3H100P2PCostModel,
    _zigzag_max_causal_work,
)
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


@pytest.fixture
def profile_data():
    """A dimensioned arithmetic fixture; every coefficient is deliberately synthetic."""
    table = {
        "source": {"scope": "synthetic unit-test fixture; not calibration evidence"},
        "validation": {"status": "synthetic fixture"},
        "attention": {
            str(cp): {"forward": [1.0, 2.0, 3.0, 4.0], "backward": [2.0, 3.0, 4.0, 5.0]}
            for cp in (1, 2, 4, 8)
        },
        "remainder_ms": {"forward": [0.5, 1.0], "backward": [0.75, 2.0]},
        "microbatch_overhead_ms": {"forward": 1.25, "backward": 1.75},
        "iteration_tail_ms": 2.0,
    }
    return {
        "schema_version": 1,
        "name": "qwen3_h100_p2p",
        "unit": "ms_per_layer",
        "domain": {
            "hardware": "h100",
            "nodes": 4,
            "ranks_per_node": 8,
            "world_size": 32,
            "ep_size": 8,
            "tokens_per_rank": 4096,
            "dataset_alignment": 64,
            "cp_sizes": [1, 2, 4, 8],
            "min_logical_length": 1,
            "max_logical_length": 16384,
            "max_group_seqs": 64,
            "tail_padding": "extend_last",
            "partition": "zigzag",
            "config": {
                "num_layers": 48,
                "hidden_size": 2048,
                "num_attention_heads": 32,
                "num_query_groups": 4,
                "kv_channels": 128,
                "num_moe_experts": 128,
                "moe_router_topk": 8,
                "moe_ffn_hidden_size": 768,
                "moe_router_force_load_balancing": True,
                "cp_partition_mode": "zigzag",
                "experimental_attention_variant": None,
                "multi_latent_attention": False,
                "qk_layernorm": True,
                "normalization": "RMSNorm",
                "attention_backend": "fused",
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "expert_model_parallel_size": 8,
                "expert_tensor_parallel_size": 1,
                "bf16": True,
                "fp16": False,
                "fp8": None,
                "fp4": None,
            },
        },
        "features": {
            "names": ["bias", "padded_s2_per_rank", "logical_s1_per_rank", "sequence_count"],
            "token_unit": 4096,
        },
        "profiles": {"classic_p2p": table, "native_p2p": copy.deepcopy(table)},
    }


@pytest.fixture
def profile_path(tmp_path, profile_data):
    path = tmp_path / "synthetic_profile.json"
    path.write_text(json.dumps(profile_data))
    return path


def _geometry():
    return RuntimePlanGeometry(32, 4096, 8, (1, 2, 4, 8), 64, 64, True)


def _config(profile_data, profile_path):
    return SimpleNamespace(
        **profile_data["domain"]["config"],
        virtual_pipeline_model_parallel_size=None,
        use_native_cp_transport=False,
        cp_comm_type="p2p",
        max_seqlen_per_dp_cp_rank=4096,
        min_dynamic_context_parallel_size=1,
        pad_packed_seq_alignment="max",
        thd_tail_padding_policy="extend_last",
        dynamic_cp_runtime_model="qwen3_h100_p2p",
        dynamic_cp_runtime_model_options=json.dumps({"profile_path": str(profile_path)}),
        dynamic_cp_runtime_solver="dynamic_usp",
    )


def test_attention_units_padding_and_remainders_are_separate(profile_path):
    model = make_model("qwen3_h100_p2p", _geometry(), profile_path=profile_path)
    # CP4 extends (1024, 1024) to (1024, 15360), giving padded S2/CP = 3.53125
    # in units of 4096 tokens. Coefficients include no EP remainder or layer count.
    assert model.forward((1024, 1024), 4) == 16.4375
    assert model.backward((1024, 1024), 4) == 23.09375
    assert model.program_costs((1024, 1024), 4) == {
        "att_f": 16.4375,
        "att_b": 23.09375,
        "rest_f": 0.625,
        "rest_b": 1.0,
    }
    timeline = build_timeline(model, _geometry())
    groups = [
        (
            start,
            4,
            16.4375,
            23.09375,
            {"att_f": 16.4375, "att_b": 23.09375, "rest_f": 0.625, "rest_b": 1.0},
        )
        for start in range(0, 32, 4)
    ]
    one_microbatch = 48 * (16.4375 + 23.09375 + 0.625 + 1.0) + 1.25 + 1.75
    assert timeline.evaluate([groups]) == one_microbatch + 2.0
    assert timeline.evaluate([groups, groups]) == 2 * one_microbatch + 2.0
    # The backward EP phase must finish before its attention backward starts.
    assert timeline.program[1][0] == ("ep_mean", "rest_b")
    assert timeline.program[1][1] == ("cp", "att_b")


def test_phase_and_cp_coefficients_are_independent(profile_path, profile_data):
    profile_data["features"]["names"] = ["bias"]
    profile_data["profiles"]["classic_p2p"]["attention"] = {
        str(cp): {"forward": [float(cp)], "backward": [float(10 * cp)]} for cp in (1, 2, 4, 8)
    }
    profile_path.write_text(json.dumps(profile_data))
    model = Qwen3H100P2PCostModel(_geometry(), profile_path=profile_path)
    for cp in (1, 2, 4, 8):
        assert model.forward((512,), cp) == cp
        assert model.backward((512,), cp) == 10 * cp


def test_original_profile_digest_and_evidence_are_reported(profile_path):
    raw_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    model = Qwen3H100P2PCostModel(_geometry(), profile_path=profile_path, profile_sha256=raw_hash)
    metadata = model.metadata()
    assert metadata["source_hashes"] == {"qwen3_h100_p2p.json": raw_hash}
    assert metadata["validation"] == {"status": "synthetic fixture"}
    assert metadata["transport"] == "classic_p2p"
    with pytest.raises(ValueError, match="hash mismatch"):
        Qwen3H100P2PCostModel(_geometry(), profile_path=profile_path, profile_sha256="0" * 64)


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.update(unit="seconds"),
        lambda data: data["profiles"].pop("classic_p2p"),
        lambda data: data["domain"]["config"].update(num_layers=4),
        lambda data: data["profiles"]["classic_p2p"]["attention"].pop("8"),
        lambda data: data["profiles"]["classic_p2p"]["attention"]["1"].update(
            forward=[float("nan")] * 4
        ),
        lambda data: data["profiles"]["classic_p2p"]["attention"]["1"].update(forward=[-1.0] * 4),
        lambda data: data["profiles"]["classic_p2p"]["attention"]["1"].update(forward=[0.0] * 4),
        lambda data: data["profiles"]["classic_p2p"]["attention"]["1"].update(backward=[1.0]),
        lambda data: data["features"].update(names=["unmeasured_feature"]),
        lambda data: data["features"].update(token_unit=0),
        lambda data: data["profiles"]["classic_p2p"]["remainder_ms"].update(forward=[-1.0, 1.0]),
        lambda data: data["profiles"]["classic_p2p"]["remainder_ms"].update(forward=[1.0]),
    ],
)
def test_invalid_profiles_are_rejected(profile_path, profile_data, change):
    change(profile_data)
    profile_path.write_text(json.dumps(profile_data))
    with pytest.raises(ValueError):
        Qwen3H100P2PCostModel(_geometry(), profile_path=profile_path)


@pytest.mark.parametrize(
    "changes",
    [
        {"world_size": 16},
        {"ep_size": 4},
        {"tokens_per_rank": 6144},
        {"dataset_alignment": 16},
        {"allowed_cp_sizes": (1, 2, 3, 4, 8)},
        {"align_group_starts": False},
        {"max_group_seqs": 65},
    ],
)
def test_unmeasured_geometry_is_rejected(profile_path, changes):
    with pytest.raises(ValueError, match="calibration domain"):
        Qwen3H100P2PCostModel(replace(_geometry(), **changes), profile_path=profile_path)


@pytest.mark.parametrize(
    "changes",
    [
        {"num_layers": 4},
        {"num_moe_experts": 8},
        {"bf16": False},
        {"fp8": "hybrid"},
        {"cp_comm_type": "a2a"},
        {"cp_comm_type": ["p2p", "a2a"]},
        {"multi_latent_attention": True},
        {"experimental_attention_variant": "gated_delta_net"},
        {"moe_router_force_load_balancing": False},
        {"cp_partition_mode": "contiguous"},
        {"attention_backend": "flash"},
        {"pad_packed_seq_alignment": 6144},
        {"thd_tail_padding_policy": "append_dummy_seq"},
        {"virtual_pipeline_model_parallel_size": 2},
    ],
)
def test_runtime_rejects_unmeasured_recipe_before_worker_start(profile_path, profile_data, changes):
    config = _config(profile_data, profile_path)
    for name, value in changes.items():
        setattr(config, name, value)
    with pytest.raises(ValueError):
        prefetch.runtime_spec(config, 32)


def test_runtime_filters_candidates_to_measured_cp_tiers(profile_path, profile_data):
    config = _config(profile_data, profile_path)
    spec = prefetch.runtime_spec(config, 32)
    assert spec["geometry"]["allowed_cp_sizes"] == (1, 2, 4, 8)
    assert spec["geometry"]["align_group_starts"]
    assert spec["solver"] == "dynamic_usp"
    assert spec["model_options"]["transport"] == "classic_p2p"
    config.cp_comm_type = ["p2p"] * config.num_layers
    assert prefetch.runtime_spec(config, 32)["model_options"]["transport"] == "classic_p2p"


def test_attention_backend_enum_is_compared_without_importing_torch(profile_path, profile_data):
    class Backend(Enum):
        fused = 1
        flash = 2

    config = _config(profile_data, profile_path)
    config.attention_backend = Backend.fused
    assert prefetch.runtime_spec(config, 32)["model_options"]["transport"] == "classic_p2p"
    config.attention_backend = Backend.flash
    with pytest.raises(ValueError, match="attention_backend"):
        prefetch.runtime_spec(config, 32)


def test_native_transport_requires_and_selects_its_own_measurements(profile_path, profile_data):
    native = profile_data["profiles"]["native_p2p"]
    native["attention"]["4"]["forward"][0] = 101.0
    profile_path.write_text(json.dumps(profile_data))
    config = _config(profile_data, profile_path)
    config.use_native_cp_transport = True
    spec = prefetch.runtime_spec(config, 32)
    assert spec["model_options"]["transport"] == "native_p2p"
    # Native capability must not enable unmeasured odd widths or cross-node placements.
    assert spec["geometry"]["allowed_cp_sizes"] == (1, 2, 4, 8)
    assert spec["geometry"]["align_group_starts"]
    model = make_model("qwen3_h100_p2p", _geometry(), **spec["model_options"])
    assert model.forward((1024, 1024), 4) == 116.4375
    assert model.metadata()["transport"] == "native_p2p"
    profile_data["profiles"].pop("native_p2p")
    profile_path.write_text(json.dumps(profile_data))
    with pytest.raises(ValueError, match="No measured.*native_p2p"):
        prefetch.runtime_spec(config, 32)


def test_transport_option_cannot_override_actual_execution(profile_path, profile_data):
    config = _config(profile_data, profile_path)
    config.dynamic_cp_runtime_model_options = json.dumps(
        {"profile_path": str(profile_path), "transport": "native_p2p"}
    )
    with pytest.raises(ValueError, match="execution transport"):
        prefetch.runtime_spec(config, 32)


@pytest.mark.parametrize(
    "lengths,cp,ranks",
    [
        ((16385,), 8, None),
        ((4097,), 1, None),
        ((0,), 1, None),
        ((True,), 1, None),
        ((1024,), 3, None),
        ((1024,), 16, None),
        ((1024,), 4, (6, 7, 8, 9)),
        ((1024,), 4, (0, 1, 3, 4)),
        ((1024,), 4, (32, 33, 34, 35)),
    ],
)
def test_queries_do_not_extrapolate_past_calibration(profile_path, lengths, cp, ranks):
    model = Qwen3H100P2PCostModel(_geometry(), profile_path=profile_path)
    with pytest.raises(ValueError):
        model.forward(lengths, cp, ranks=ranks)


@pytest.mark.parametrize("planner_type", [LightweightRuntimePlanner, DynamicUSPRuntimePlanner])
def test_both_solvers_consume_the_model_without_changing_their_search(profile_path, planner_type):
    geometry = _geometry()
    model = Qwen3H100P2PCostModel(geometry, profile_path=profile_path)
    planner = planner_type(geometry, model)
    lengths = dict.fromkeys(range(64), 1024)
    incumbent = [[list(range(2 * rank, 2 * rank + 2)) for rank in range(32)]]
    plan = planner.plan_padded(lengths, logical_lengths=lengths, incumbent_rows=incumbent)
    validate_rows(plan.rows, lengths, geometry, 1)
    assert plan.predicted_ms > 0
    assert planner.model is model
    assert all(
        cp in (1, 2, 4, 8) and start % cp == 0 for row in plan.groups for start, cp, _ in row
    )


def test_registered_model_runs_inside_the_persistent_cpu_worker(profile_path, profile_data):
    config = _config(profile_data, profile_path)
    config.dynamic_cp_runtime_solver_options = '{"candidate_count":2}'
    lengths = [1024] * 64
    baseline = [[list(range(2 * rank, 2 * rank + 2)) for rank in range(32)]]
    worker = prefetch._CpuPlanner(prefetch.runtime_spec(config, 32))
    try:
        first = worker.solve(lengths, lengths, baseline)
        second = worker.solve(lengths, lengths, baseline)
    finally:
        worker.close()
    assert first["worker_pid"] == second["worker_pid"]
    assert first["rows"] == second["rows"]
    assert first["predicted_ms"] == second["predicted_ms"]
    for result in (first, second):
        assert not result["worker_cuda_initialized"]
        assert not result["worker_distributed_initialized"]
        assert not result["external_scheduler_imported"]
        assert result["model_metadata"]["name"] == "qwen3_h100_p2p"
        assert result["model_metadata"]["transport"] == "classic_p2p"
        assert result["statistics"]["group_sizes"] == [1, 2, 4, 8]
        assert "incumbent_rejection" not in result["statistics"]


def _wide_incumbent(cp, count=64):
    groups = 32 // cp
    assert count % groups == 0
    per_group = count // groups
    return [
        [
            list(range(group * per_group, (group + 1) * per_group))
            for group in range(groups)
            for _ in range(cp)
        ]
    ]


@pytest.mark.parametrize("solver", [None, "dynamic_usp"])
@pytest.mark.parametrize("cp", [16, 32])
def test_only_unpriced_default_incumbent_is_skipped(profile_path, profile_data, solver, cp):
    config = _config(profile_data, profile_path)
    config.dynamic_cp_runtime_solver = solver
    worker = prefetch._CpuPlanner(prefetch.runtime_spec(config, 32))
    lengths = [1024] * 64
    try:
        result = worker.solve(lengths, lengths, _wide_incumbent(cp))
    finally:
        worker.close()
    validate_rows(result["rows"], dict(enumerate(lengths)), _geometry(), 1)
    statistics = result["statistics"]
    assert statistics["incumbent_rejection"] == "default_incumbent_outside_measured_cp_domain"
    assert statistics["incumbent_unsupported_cp_sizes"] == [cp]
    assert not statistics["selected_incumbent"]
    assert result["predicted_ms"] > 0
    assert not result["worker_cuda_initialized"]


@pytest.mark.parametrize("control", ["custom_baseline", "force_powers"])
@pytest.mark.parametrize("cp", [16, 32])
def test_explicit_baseline_controls_still_reject_unpriced_widths(
    profile_path, profile_data, control, cp
):
    spec = prefetch.runtime_spec(_config(profile_data, profile_path), 32)
    if control == "custom_baseline":
        # The training process owns this factory; the worker must respect its
        # explicit provenance without importing or substituting a factory here.
        spec["baseline"] = "explicit:test_baseline_factory"
    else:
        spec["force_powers"] = True
    worker = prefetch._CpuPlanner(spec)
    try:
        with pytest.raises(RuntimeError, match=f"cp={cp}"):
            worker.solve([1024] * 64, [1024] * 64, _wide_incumbent(cp))
    finally:
        worker.close()


@pytest.mark.parametrize(
    "invalid",
    [
        "unknown",
        "duplicate",
        "missing",
        "empty",
        "row_length",
        "capacity",
        "sequence_cap",
        "alignment",
    ],
)
def test_unpriced_default_incumbent_does_not_hide_invalid_rows(profile_path, profile_data, invalid):
    lengths = [1024] * 64
    rows = _wide_incumbent(32)
    if invalid == "unknown":
        rows = [[list(range(63)) + [64] for _ in range(32)]]
        message = "unknown sample"
    elif invalid == "duplicate":
        rows = [[list(range(63)) + [0] for _ in range(32)]]
        message = "lost or duplicated"
    elif invalid == "missing":
        rows = [[list(range(63)) for _ in range(32)]]
        message = "lost or duplicated"
    elif invalid == "empty":
        rows = [[[] for _ in range(32)]]
        message = "every rank must pack"
    elif invalid == "row_length":
        rows[0].pop()
        message = "rank entries"
    elif invalid == "capacity":
        lengths = [4096] * 64
        message = "capacity"
    elif invalid == "sequence_cap":
        lengths = [1024] * 65
        rows = _wide_incumbent(32, 65)
        message = "cap is 64"
    else:
        rows = [
            [list(range(16)) for _ in range(8)]
            + [list(range(16, 48)) for _ in range(16)]
            + [list(range(48, 64)) for _ in range(8)]
        ]
        message = "size-aligned start"
    worker = prefetch._CpuPlanner(prefetch.runtime_spec(_config(profile_data, profile_path), 32))
    try:
        with pytest.raises(RuntimeError, match=message):
            worker.solve(lengths, lengths, rows)
    finally:
        worker.close()


def test_public_planner_validation_remains_strict_for_unpriced_incumbents(profile_path):
    geometry = _geometry()
    model = Qwen3H100P2PCostModel(geometry, profile_path=profile_path)
    lengths = dict.fromkeys(range(64), 1024)
    planner = DynamicUSPRuntimePlanner(geometry, model)
    with pytest.raises(RuntimeError, match="cp=32"):
        planner.plan_padded(lengths, logical_lengths=lengths, incumbent_rows=_wide_incumbent(32))


def _enumerated_causal_work(logical, physical, cp, unit):
    work = [0] * cp
    for length, storage in zip(logical, physical):
        block_size = storage // (2 * cp)
        for query in range(length):
            block = query // block_size
            owner = block if block < cp else 2 * cp - 1 - block
            work[owner] += query + 1
    return 2 * max(work) / unit**2


@pytest.mark.parametrize(
    "cp,logical,physical,unit",
    [
        (1, (1, 5), (2, 8), 16),
        (2, (3, 9), (4, 12), 16),
        (4, (7, 20, 1), (8, 24, 8), 32),
        (8, (3, 17, 31), (16, 32, 48), 64),
        (8, (8192,), (32768,), 4096),
        (8, (4096,) * 8, (4096,) * 8, 4096),
        (1, (0,), (2,), 4),
    ],
)
def test_zigzag_causal_closed_form_matches_independent_query_enumeration(
    cp, logical, physical, unit
):
    assert _zigzag_max_causal_work(logical, physical, cp, unit) == (
        _enumerated_causal_work(logical, physical, cp, unit)
    )


def test_zigzag_causal_work_captures_tail_padding_rank_imbalance():
    assert _zigzag_max_causal_work((8192,), (32768,), 8, 4096) == 1.7501220703125
    assert _zigzag_max_causal_work((4096,) * 8, (4096,) * 8, 8, 4096) == 1.000244140625


@pytest.mark.parametrize(
    "logical,physical,cp,unit",
    [
        ((), (), 1, 1),
        ((1,), (2, 2), 1, 1),
        ((3,), (2,), 1, 1),
        ((1,), (7,), 2, 1),
        ((-1,), (2,), 1, 1),
        ((1,), (0,), 1, 1),
        ((1,), (2,), 0, 1),
        ((1,), (2,), 1, 0),
    ],
)
def test_zigzag_causal_work_rejects_invalid_physical_partitions(logical, physical, cp, unit):
    with pytest.raises(ValueError):
        _zigzag_max_causal_work(logical, physical, cp, unit)


def test_optional_causal_feature_uses_actual_tail_policy(profile_path, profile_data):
    profile_data["features"]["names"] = ["zigzag_max_causal_work"]
    for table in profile_data["profiles"].values():
        table["attention"] = {str(cp): {"forward": [1.0], "backward": [2.0]} for cp in (1, 2, 4, 8)}
    profile_path.write_text(json.dumps(profile_data))
    model = Qwen3H100P2PCostModel(_geometry(), profile_path=profile_path)
    assert model.forward((8192,), 8) == 1.7501220703125
    assert model.backward((8192,), 8) == 3.500244140625
    profile_data["domain"]["tail_padding"] = "append_dummy_seq"
    profile_path.write_text(json.dumps(profile_data))
    model = Qwen3H100P2PCostModel(_geometry(), profile_path=profile_path)
    assert model.forward((8192,), 8) == _enumerated_causal_work(
        (8192, 24576), (8192, 24576), 8, 4096
    )


def test_existing_feature_profiles_do_not_compute_optional_causal_work(monkeypatch, profile_path):
    module = importlib.import_module(Qwen3H100P2PCostModel.__module__)

    def unexpected(*args):
        raise AssertionError("The optional feature must remain lazy")

    monkeypatch.setattr(module, "_zigzag_max_causal_work", unexpected)
    model = Qwen3H100P2PCostModel(_geometry(), profile_path=profile_path)
    assert model.forward((1024, 1024), 4) == 16.4375


def test_cpu_only_import_and_prediction_need_no_megatron_or_torch(profile_path):
    datasets = Path(prefetch.__file__).parent
    script = """
import json
import sys
sys.path.insert(0, sys.argv[1])
from dcp_runtime_scheduler.models.qwen3_h100_p2p import Qwen3H100P2PCostModel
from dcp_runtime_scheduler.runtime.planner import RuntimePlanGeometry
geometry = RuntimePlanGeometry(32, 4096, 8, (1, 2, 4, 8), 64, 64, True)
model = Qwen3H100P2PCostModel(geometry, profile_path=sys.argv[2])
print(json.dumps({
    "forward_ms": model.forward((1024, 1024), 4),
    "torch_imported": "torch" in sys.modules,
    "megatron_imported": "megatron" in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(datasets), str(profile_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        check=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "forward_ms": 16.4375,
        "torch_imported": False,
        "megatron_imported": False,
    }
