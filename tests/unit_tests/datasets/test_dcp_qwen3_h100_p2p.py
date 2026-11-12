# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The measured H100 classic-P2P calibration of the dynamic_usp scheduler type."""

import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets import data_schedule, dcp_runtime_worker
from megatron.core.datasets import dcp_schedule_prefetch as prefetch
from megatron.core.datasets.dcp_runtime_scheduler import models
from megatron.core.datasets.dcp_runtime_scheduler.models import (
    make_model,
    measured_dynamic_usp_model,
    qwen3_h100_p2p,
)
from megatron.core.datasets.dcp_runtime_scheduler.models.base import PackLimits
from megatron.core.datasets.dcp_runtime_scheduler.models.qwen3_h100_p2p import (
    PACKAGED_SHA256,
    Qwen3H100P2PCostModel,
    _zigzag_max_causal_work,
)
from megatron.core.datasets.dcp_runtime_scheduler.models.usp_calib_v1 import USPCalibV1CostModel
from megatron.core.datasets.dcp_runtime_scheduler.runtime.config import (
    allowed_keys,
    load_runtime_config,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.dynamic_usp import (
    DynamicUSPRuntimePlanner,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import (
    RuntimePlanGeometry,
    RuntimeScheduleError,
    longest_priced_sample,
    sample_length_limits,
    unpriced_sample,
    validate_rows,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.timeline import build_timeline
from megatron.core.transformer import transformer_config
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.transformer_config import TransformerConfig

PACKAGED = Path(qwen3_h100_p2p.__file__).with_suffix(".json")
# Canonical digest of the consumed numbers (domain, features and every coefficient) of
# the coefficient set that was executed on the measured cluster.
EXECUTED_PAYLOAD_SHA256 = "68e4ceb28eb667ee2a09b37e2fb0482798a9de7e34310371a211731dcf2be932"
DYNAMIC_USP_COMMIT = "18eb0f53dbdb3a13b19e49c02fd1782178947c9a"
# Regression pin: the attention predictions the offline fitter wrote for its fitting cases.
FROZEN_PATH = Path(__file__).with_name("dcp_qwen3_h100_p2p_frozen_costs.json")
FROZEN = json.loads(FROZEN_PATH.read_text())


@pytest.fixture
def profile_data():
    """A refit-format file: the packaged domain with deliberately synthetic coefficients."""
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
    # A refit lists the settings of the packaged file and states its own global batch size.
    domain = _packaged()["domain"]
    domain["step_overhead_global_batch_size"] = 64
    return {
        "schema_version": 1,
        "name": "qwen3_h100_p2p",
        "unit": "ms_per_layer",
        "domain": domain,
        "features": {
            "names": ["bias", "padded_s2_per_rank", "logical_s1_per_rank", "sequence_count"],
            "token_unit": 4096,
        },
        "profiles": {"classic_p2p": table},
    }


@pytest.fixture
def profile_path(tmp_path, profile_data):
    path = tmp_path / "synthetic_calibration.json"
    path.write_text(json.dumps(profile_data))
    return path


def _geometry():
    return RuntimePlanGeometry(32, 4096, 8, (1, 2, 4, 8), 64, 64, True)


def _packaged():
    return json.loads(PACKAGED.read_text())


def _training_config(tmp_path, domain_config, runtime=None, **overrides):
    """A training configuration of the measured recipe plus the runtime file selecting it."""
    payload = {"type": "dynamic_usp", "hardware": "h100", "max_cp_size": 8}
    payload.update(runtime or {})
    payload = {key: value for key, value in payload.items() if value is not None}
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload))
    values = dict(
        domain_config,
        dynamic_cp_runtime_config=str(path),
        virtual_pipeline_model_parallel_size=None,
        cp_comm_type="p2p",
        max_seqlen_per_dp_cp_rank=4096,
        min_dynamic_context_parallel_size=1,
        pad_packed_seq_alignment="max",
        thd_tail_padding_policy="extend_last",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _config(tmp_path, profile_data, profile_path, runtime=None, **overrides):
    """The recipe with the synthetic coefficient file as its ``calibration``."""
    runtime = dict({"calibration": str(profile_path)}, **(runtime or {}))
    return _training_config(tmp_path, profile_data["domain"]["config"], runtime, **overrides)


def _packaged_config(tmp_path, runtime=None, **overrides):
    """The recipe with the packaged coefficients, as a user selects them."""
    return _training_config(tmp_path, _packaged()["domain"]["config"], runtime, **overrides)


# --------------------------------------------------------------------------------------
# Cost arithmetic and the coefficient file
# --------------------------------------------------------------------------------------
def test_attention_units_padding_and_remainders_are_separate(profile_path):
    model = make_model("qwen3_h100_p2p", _geometry(), calibration=profile_path)
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
    model = Qwen3H100P2PCostModel(_geometry(), calibration=profile_path)
    for cp in (1, 2, 4, 8):
        assert model.forward((512,), cp) == cp
        assert model.backward((512,), cp) == 10 * cp


def test_loaded_file_digest_and_evidence_are_reported(profile_path):
    raw_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    metadata = Qwen3H100P2PCostModel(_geometry(), calibration=str(profile_path)).metadata()
    assert metadata["name"] == "qwen3_h100_p2p"
    assert metadata["source_hashes"] == {"qwen3_h100_p2p.json": raw_hash}
    assert metadata["calibration_sha256"] == raw_hash
    assert metadata["calibration_path"] == str(profile_path.resolve())
    assert metadata["validation"] == {"status": "synthetic fixture"}
    assert metadata["modeled_transport"] == "classic P2P"


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.update(unit="seconds"),
        lambda data: data.update(name="usp_calib_v1"),
        lambda data: data["profiles"].pop("classic_p2p"),
        lambda data: data["profiles"].update(other_transport=data["profiles"]["classic_p2p"]),
        lambda data: data["domain"]["config"].update(num_layers=4),
        lambda data: data["domain"].update(world_size=64),
        lambda data: data["domain"].update(step_overhead_global_batch_size=0),
        lambda data: data["domain"].pop("step_overhead_global_batch_size"),
        lambda data: data["domain"]["config"].pop("hidden_dropout"),
        lambda data: data["domain"]["config"].update(another_setting=1),
        lambda data: data["profiles"]["classic_p2p"]["attention"].pop("8"),
        lambda data: data["profiles"]["classic_p2p"]["attention"]["1"].update(
            forward=[float("nan")] * 4
        ),
        lambda data: data["profiles"]["classic_p2p"]["attention"]["1"].update(forward=[-1.0] * 4),
        lambda data: data["profiles"]["classic_p2p"]["attention"]["1"].update(forward=[0.0] * 4),
        lambda data: data["profiles"]["classic_p2p"]["attention"]["1"].update(backward=[1.0]),
        lambda data: data["profiles"]["classic_p2p"].pop("validation"),
        lambda data: data["features"].update(names=["unmeasured_feature"]),
        lambda data: data["features"].update(token_unit=0),
        lambda data: data["profiles"]["classic_p2p"]["remainder_ms"].update(forward=[-1.0, 1.0]),
        lambda data: data["profiles"]["classic_p2p"]["remainder_ms"].update(forward=[1.0]),
    ],
)
def test_invalid_calibration_files_are_rejected(profile_path, profile_data, change):
    change(profile_data)
    profile_path.write_text(json.dumps(profile_data))
    with pytest.raises(ValueError):
        Qwen3H100P2PCostModel(_geometry(), calibration=profile_path)


def test_refit_file_lists_the_packaged_settings_and_states_its_global_batch_size(
    profile_path, profile_data
):
    settings = profile_data["domain"]["config"]
    del settings["hidden_dropout"], settings["layernorm_epsilon"]
    settings["another_setting"] = 1
    profile_path.write_text(json.dumps(profile_data))
    with pytest.raises(ValueError) as caught:
        Qwen3H100P2PCostModel(_geometry(), calibration=profile_path)
    assert str(caught.value) == (
        "The H100 calibration file must list the same domain.config settings as the packaged "
        "file (values may differ); missing: ['hidden_dropout', 'layernorm_epsilon']; not in "
        "the packaged file: ['another_setting']"
    )
    profile_data["domain"]["config"] = _packaged()["domain"]["config"]
    del profile_data["domain"]["step_overhead_global_batch_size"]
    profile_path.write_text(json.dumps(profile_data))
    with pytest.raises(ValueError, match="must state domain.step_overhead_global_batch_size"):
        Qwen3H100P2PCostModel(_geometry(), calibration=profile_path)
    # The packaged file is the reference for the list, and it is complete by that measure.
    assert len(_packaged()["domain"]["config"]) == 54
    assert Qwen3H100P2PCostModel(_geometry()).required_global_batch_size == 256


def test_refit_file_values_define_the_enforced_domain(tmp_path, profile_data, profile_path):
    profile_data["domain"]["config"]["hidden_dropout"] = 0.1
    profile_data["domain"]["step_overhead_global_batch_size"] = 128
    profile_path.write_text(json.dumps(profile_data))
    # A training configuration with the file's own values passes; the packaged value fails.
    spec = prefetch.runtime_spec(_config(tmp_path, profile_data, profile_path), 32)
    assert spec["required_global_batch_size"] == 128
    packaged_value = _config(tmp_path, profile_data, profile_path, hidden_dropout=0.0)
    with pytest.raises(ValueError, match="was measured with hidden_dropout=0.1, got 0.0"):
        prefetch.runtime_spec(packaged_value, 32)


def test_model_declares_its_measured_sample_length_range():
    geometry = _geometry()
    model = Qwen3H100P2PCostModel(geometry)
    for cp in (1, 2, 4, 8):
        assert model.pack_limits(cp) == PackLimits(max_sample_length=16384)
    assert sample_length_limits(geometry, model) == {1: 16384, 2: 16384, 4: 16384, 8: 16384}
    # CP 4 holds 16384 tokens; CP 8 holds more than the model prices.
    assert longest_priced_sample(geometry, model) == 16384
    assert unpriced_sample(geometry, model, {0: 16384, 1: 4096}) is None
    assert unpriced_sample(geometry, model, {0: 16384, 1: 16385, 2: 20000}) == 1
    # With CP 1 and 2 only, the group capacity is the smaller limit.
    narrow = replace(geometry, allowed_cp_sizes=(1, 2))
    assert longest_priced_sample(narrow, Qwen3H100P2PCostModel(narrow)) == 8192
    # The sequence cap is not lowered, and CP 8 stays a size the planner can rely on.
    planner = DynamicUSPRuntimePlanner(geometry, model)
    assert planner.sequence_limit(8) == 64
    assert dcp_runtime_worker.coverage_gaps(planner) == {}


def test_refit_file_states_its_own_sample_length_range(profile_path, profile_data):
    profile_data["domain"]["max_logical_length"] = 8192
    profile_path.write_text(json.dumps(profile_data))
    model = Qwen3H100P2PCostModel(_geometry(), calibration=profile_path)
    assert model.pack_limits(8) == PackLimits(max_sample_length=8192)
    assert longest_priced_sample(_geometry(), model) == 8192


@pytest.mark.parametrize("contents", ["", "{", "[]", "null"])
def test_unreadable_calibration_files_are_rejected(tmp_path, contents):
    path = tmp_path / "broken.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match="H100 calibration"):
        Qwen3H100P2PCostModel(_geometry(), calibration=path)
    with pytest.raises(ValueError, match="Cannot read H100 calibration"):
        Qwen3H100P2PCostModel(_geometry(), calibration=tmp_path / "missing.json")
    with pytest.raises(ValueError, match="coefficient file path"):
        Qwen3H100P2PCostModel(_geometry(), calibration={"attention": {}})


@pytest.mark.parametrize(
    "changes,named",
    [
        ({"world_size": 16}, "DPxCP=32, got 16"),
        ({"ep_size": 4}, "expert_model_parallel_size=8, got 4"),
        ({"tokens_per_rank": 6144}, "max_seqlen_per_dp_cp_rank=4096, got 6144"),
        ({"dataset_alignment": 16}, "dataset_alignment=64, got 16"),
        ({"allowed_cp_sizes": (1, 2, 3, 4, 8)}, r"CP groups of \[1, 2, 4, 8\] ranks"),
        ({"allowed_cp_sizes": (8, 16)}, r"CP groups of \[1, 2, 4, 8\] ranks"),
        ({"align_group_starts": False}, "size-aligned CP groups"),
        ({"max_group_seqs": None}, "at most 64 sequences"),
    ],
)
def test_unmeasured_geometry_is_rejected_and_named(profile_path, changes, named):
    with pytest.raises(ValueError, match=named):
        Qwen3H100P2PCostModel(replace(_geometry(), **changes), calibration=profile_path)


@pytest.mark.parametrize(
    "lengths,cp,ranks",
    [
        ((16385,), 8, None),
        ((4097,), 1, None),
        ((0,), 1, None),
        ((True,), 1, None),
        ((1024,), 3, None),
        ((1024,), 16, None),
        ((64,) * 65, 8, None),
        ((1024,), 4, (6, 7, 8, 9)),
        ((1024,), 4, (0, 1, 3, 4)),
        ((1024,), 4, (32, 33, 34, 35)),
    ],
)
def test_queries_do_not_extrapolate_past_calibration(profile_path, lengths, cp, ranks):
    model = Qwen3H100P2PCostModel(_geometry(), calibration=profile_path)
    with pytest.raises(ValueError):
        model.forward(lengths, cp, ranks=ranks)


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
    profile_data["profiles"]["classic_p2p"]["attention"] = {
        str(cp): {"forward": [1.0], "backward": [2.0]} for cp in (1, 2, 4, 8)
    }
    profile_path.write_text(json.dumps(profile_data))
    model = Qwen3H100P2PCostModel(_geometry(), calibration=profile_path)
    assert model.forward((8192,), 8) == 1.7501220703125
    assert model.backward((8192,), 8) == 3.500244140625
    profile_data["domain"]["tail_padding"] = "append_dummy_seq"
    profile_path.write_text(json.dumps(profile_data))
    model = Qwen3H100P2PCostModel(_geometry(), calibration=profile_path)
    assert model.forward((8192,), 8) == _enumerated_causal_work(
        (8192, 24576), (8192, 24576), 8, 4096
    )


def test_existing_feature_profiles_do_not_compute_optional_causal_work(monkeypatch, profile_path):
    module = importlib.import_module(Qwen3H100P2PCostModel.__module__)

    def unexpected(*args):
        raise AssertionError("The optional feature must remain lazy")

    monkeypatch.setattr(module, "_zigzag_max_causal_work", unexpected)
    model = Qwen3H100P2PCostModel(_geometry(), calibration=profile_path)
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
model = Qwen3H100P2PCostModel(geometry, calibration=sys.argv[2])
packaged = Qwen3H100P2PCostModel(geometry)
print(json.dumps({
    "forward_ms": model.forward((1024, 1024), 4),
    "packaged_positive": packaged.forward((1024, 1024), 4) > 0,
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
        "packaged_positive": True,
        "torch_imported": False,
        "megatron_imported": False,
    }


# --------------------------------------------------------------------------------------
# The packaged coefficient file
# --------------------------------------------------------------------------------------
def _consumed_payload(data):
    table = data["profiles"]["classic_p2p"]
    return {
        "domain": data["domain"],
        "features": data["features"],
        "attention": table["attention"],
        "remainder_ms": table["remainder_ms"],
        "microbatch_overhead_ms": table["microbatch_overhead_ms"],
        "iteration_tail_ms": table["iteration_tail_ms"],
    }


def test_packaged_file_digest_is_pinned_and_recorded():
    digest = hashlib.sha256(PACKAGED.read_bytes()).hexdigest()
    origin = json.loads((PACKAGED.parents[1] / "ORIGIN.json").read_text())
    assert digest == PACKAGED_SHA256
    assert origin["runtime_parameters_sha256"]["models/qwen3_h100_p2p.json"] == digest
    model = Qwen3H100P2PCostModel(_geometry())
    assert model.metadata()["calibration_sha256"] == digest
    assert model.metadata()["calibration_path"] == str(PACKAGED.resolve())


def test_packaged_file_hash_mismatch_is_reported(monkeypatch):
    monkeypatch.setattr(qwen3_h100_p2p, "PACKAGED_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="hash mismatch"):
        Qwen3H100P2PCostModel(_geometry())


def test_packaged_coefficients_are_the_executed_set():
    canonical = json.dumps(_consumed_payload(_packaged()), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == EXECUTED_PAYLOAD_SHA256


def test_packaged_file_describes_classic_p2p_only_and_states_its_fit():
    data = _packaged()
    assert list(data["profiles"]) == ["classic_p2p"]
    assert data["domain"]["cp_sizes"] == [1, 2, 4, 8]
    assert data["domain"]["step_overhead_global_batch_size"] == 256
    assert data["domain"]["config"]["moe_router_force_load_balancing"] is True
    validation = data["profiles"]["classic_p2p"]["validation"]
    assert (validation["training_case_count"], validation["heldout_case_count"]) == (60, 14)
    assert validation["step_overhead_fit"]["fitted_rows"] == 11
    model = Qwen3H100P2PCostModel(_geometry())
    assert model.required_global_batch_size == 256
    assert model.metadata()["validation"] == validation


# What only records of one measurement run contain: scheduler bookkeeping, host names
# and host ranges, absolute paths. Generic patterns, so that this list names no machine.
_RUN_RECORD_PATTERNS = (
    r"\bjobs?\b|\bjob[_-]?ids?\b",
    r"\bnode[ _-]?lists?\b",
    r"\bclusters?\b",
    r"\baccount",
    r"\breceipt",
    r"\bmanifest",
    # A host name or host range: letters, a hyphen, then at least three digits or a list.
    r"\b[a-z][a-z0-9]*-(?:\d{3,}|\[\d)",
    # An absolute file-system path with at least two components.
    r"(?<![\w.:/])/(?:[\w.-]+/)+[\w.-]*",
)


def _run_records(text):
    return [
        found.group(0)
        for pattern in _RUN_RECORD_PATTERNS
        for found in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]


def test_run_record_patterns_recognise_what_they_are_for():
    assert _run_records("job 123456 on the cluster, job_id=7") == ["job", "job_id", "cluster"]
    assert _run_records('"nodelist": "host-[001,002]"') == ["nodelist", "host-[0"]
    assert _run_records("host worker-0042 wrote /srv/shared/run-1/out.json") == [
        "worker-0042",
        "/srv/shared/run-1/out.json",
    ]
    # Not run records: model names, ratios, repository-relative paths and URLs.
    assert _run_records("Qwen3-30B-A3B TP1/PP1/EP8, CP 1/2/4/8, 2*CP-1-r") == []
    assert _run_records("models/qwen3_h100_p2p.json at https://github.com/org/repo") == []


@pytest.mark.parametrize(
    "path",
    [PACKAGED, FROZEN_PATH, PACKAGED.with_suffix(".py")],
    ids=["coefficients", "fixture", "loader"],
)
def test_shipped_files_carry_no_internal_identifiers(path):
    text = path.read_text()
    assert _run_records(text) == []
    # Content digests are 64 hex digits: the only one is the loader's pin of the packaged
    # file. The only 40-digit identifier is the DynamicUSP commit of the procedure.
    digests = set(re.findall(r"\b[0-9a-f]{64}\b", text))
    assert digests == ({PACKAGED_SHA256} if path.suffix == ".py" else set())
    assert set(re.findall(r"\b[0-9a-f]{40}\b", text)) <= {DYNAMIC_USP_COMMIT}


def test_origin_entry_carries_no_internal_identifiers():
    origin = json.loads((PACKAGED.parents[1] / "ORIGIN.json").read_text())
    entry = origin["dynamic_usp"]["measured_calibrations"]["h100"]
    assert entry["parameters"] == "models/qwen3_h100_p2p.json"
    assert entry["methodology"]["commit"] == DYNAMIC_USP_COMMIT
    assert entry["measurements"]["module_heldout_cases"] == 14
    text = json.dumps(entry)
    assert _run_records(text) == []
    assert "sha256" not in text.lower()
    assert set(re.findall(r"\b[0-9a-f]{40,64}\b", text)) == {DYNAMIC_USP_COMMIT}


@pytest.mark.parametrize("case", FROZEN["cases"])
def test_packaged_model_matches_the_fitter_predictions(case):
    model = Qwen3H100P2PCostModel(_geometry())
    lengths = tuple(case["lengths"])
    assert model.forward(lengths, case["cp"]) == pytest.approx(case["forward_ms"], abs=1e-9)
    assert model.backward(lengths, case["cp"]) == pytest.approx(case["backward_ms"], abs=1e-9)


def test_packaged_whole_step_is_layers_times_block_plus_overhead_and_tail():
    data = _packaged()
    table = data["profiles"]["classic_p2p"]
    names = data["features"]["names"]
    # One full 4096-token sample on one rank: no padding, one sequence, and a causal work of
    # 2 * (1 + 2 + ... + 4096) / 4096**2.
    features = {
        "bias": 1.0,
        "zigzag_max_causal_work": 4097 / 4096,
        "padded_s2_per_rank": 1.0,
        "sequence_count": 1.0,
    }
    attention = {
        phase: sum(c * features[name] for c, name in zip(table["attention"]["1"][phase], names))
        for phase in ("forward", "backward")
    }
    remainder = {phase: sum(table["remainder_ms"][phase]) for phase in ("forward", "backward")}
    block = sum(attention.values()) + sum(remainder.values())
    overhead = sum(table["microbatch_overhead_ms"].values())
    model = Qwen3H100P2PCostModel(_geometry())
    costs = model.program_costs((4096,), 1)
    assert costs["att_f"] == pytest.approx(attention["forward"], rel=1e-12)
    assert costs["att_b"] == pytest.approx(attention["backward"], rel=1e-12)
    assert costs["rest_f"] == pytest.approx(remainder["forward"], rel=1e-12)
    assert costs["rest_b"] == pytest.approx(remainder["backward"], rel=1e-12)
    groups = [(rank, 1, costs["att_f"], costs["att_b"], costs) for rank in range(32)]
    timeline = build_timeline(model, _geometry())
    # 256 samples as eight microbatches of 32 single-rank groups.
    expected = 8 * (48 * block + overhead) + table["iteration_tail_ms"]
    assert timeline.evaluate([groups] * 8) == pytest.approx(expected, rel=1e-12)
    assert 4700 < expected < 4900


# --------------------------------------------------------------------------------------
# Selection through the runtime file
# --------------------------------------------------------------------------------------
def test_hardware_is_a_dynamic_usp_key_and_names_a_packaged_calibration(tmp_path):
    assert allowed_keys("dynamic_usp") == (
        "type",
        "hardware",
        "calibration",
        "max_cp_size",
        "async",
        "trace_dir",
    )
    path = tmp_path / "runtime.json"
    path.write_text('{"type": "dynamic_usp", "hardware": "h100", "max_cp_size": 8}')
    assert load_runtime_config(str(path))["options"] == {"hardware": "h100"}
    assert measured_dynamic_usp_model(None) is None
    assert measured_dynamic_usp_model("h100") == "qwen3_h100_p2p"


@pytest.mark.parametrize("hardware", ["gb200", "H100", "h100 "])
def test_hardware_without_a_packaged_calibration_is_rejected_at_startup(tmp_path, hardware):
    config = _packaged_config(tmp_path, {"hardware": hardware})
    with pytest.raises(ValueError, match="No DynamicUSP calibration is packaged") as caught:
        prefetch.runtime_spec(config, 32)
    assert f"for hardware {hardware!r}; packaged: ['h100']" in str(caught.value)
    assert "(type 'dynamic_usp')" in str(caught.value)
    assert "training configuration" not in str(caught.value)


@pytest.mark.parametrize("hardware", [5, "", ["h100"]])
def test_hardware_must_be_a_name(tmp_path, hardware):
    config = _packaged_config(tmp_path, {"hardware": hardware})
    with pytest.raises(ValueError, match="hardware must be a nonempty string"):
        prefetch.runtime_spec(config, 32)


def test_runtime_file_selects_the_packaged_calibration_and_the_dynamic_usp_solver(tmp_path):
    spec = prefetch.runtime_spec(_packaged_config(tmp_path), 32, max_num_seqs=64)
    assert spec["type"] == "dynamic_usp"
    assert spec["model"] == "qwen3_h100_p2p"
    assert spec["model_options"] == {}
    assert spec["geometry"]["allowed_cp_sizes"] == (1, 2, 4, 8)
    assert spec["geometry"]["max_cp_size"] == 8
    assert spec["geometry"]["align_group_starts"]
    assert "solver" not in spec
    planner = dcp_runtime_worker._initialize_planner(spec)
    assert isinstance(planner, DynamicUSPRuntimePlanner)
    assert isinstance(planner.model, Qwen3H100P2PCostModel)
    assert planner.model.metadata()["calibration_sha256"] == PACKAGED_SHA256
    # The start-up probe prices every CP size the planner may use.
    assert prefetch._coverage_error(spec) is None
    assert dcp_runtime_worker.coverage_gaps(planner) == {}
    # The training entry compares this with the configured global batch sizes at start-up.
    assert spec["required_global_batch_size"] == 256
    # The line that rank 0 prints once start-up validation has passed.
    assert spec["hardware"] == "h100" and spec["nvlink_domain_size"] == 8
    assert prefetch.describe_runtime_spec(spec) == (
        "Runtime DCP scheduler: type=dynamic_usp, cost model=qwen3_h100_p2p, "
        "hardware=h100 (NVLink domain: 8 consecutive ranks), "
        "CP group sizes=[1, 2, 4, 8], at most 64 sequences per CP group, async=on, "
        f"runtime file={Path(spec['config_path'])}"
    )


@pytest.mark.parametrize(
    "minimum,maximum,allowed",
    [(1, 8, (1, 2, 4, 8)), (1, 4, (1, 2, 4)), (1, 2, (1, 2)), (2, 8, (2, 4, 8)), (8, 8, (8,))],
)
def test_cp_sizes_are_the_measured_ones_within_minimum_and_cap(tmp_path, minimum, maximum, allowed):
    config = _packaged_config(
        tmp_path, {"max_cp_size": maximum}, min_dynamic_context_parallel_size=minimum
    )
    assert prefetch.runtime_spec(config, 32)["geometry"]["allowed_cp_sizes"] == allowed


def test_the_cap_must_be_explicit_because_the_calibration_stops_at_cp8(tmp_path):
    config = _packaged_config(tmp_path, {"max_cp_size": None})
    with pytest.raises(ValueError, match='Set "max_cp_size": 8') as caught:
        prefetch.runtime_spec(config, 32)
    assert "CP groups of [1, 2, 4, 8] ranks for DPxCP=32" in str(caught.value)
    assert "power-of-two DPxCP, or" not in str(caught.value)
    # The smaller of the two limits: CP 8 holds 32768 tokens, the calibration prices 16384.
    assert (
        "so samples longer than 16384 tokens, the longest sample the 'dynamic_usp' cost model "
        "prices (less than 8 * max_seqlen_per_dp_cp_rank = 32768 tokens) cannot be scheduled"
    ) in str(caught.value)
    # A cap is one of the sizes this calibration forms; 6, 16 or 32 are not rounded down.
    for maximum in (6, 16, 32):
        config = _packaged_config(tmp_path, {"max_cp_size": maximum})
        with pytest.raises(ValueError, match=f"max_cp_size={maximum} is not a CP group size"):
            prefetch.runtime_spec(config, 32)


def test_minimum_cp_above_the_measured_sizes_is_rejected(tmp_path):
    config = _packaged_config(tmp_path, {"max_cp_size": 16}, min_dynamic_context_parallel_size=16)
    with pytest.raises(ValueError, match="CP groups of at most 8 ranks"):
        prefetch.runtime_spec(config, 32)


def test_without_hardware_the_original_profile_and_its_tiers_apply(tmp_path):
    config = _packaged_config(tmp_path, {"hardware": None})
    with pytest.raises(ValueError, match="type 'dynamic_usp' requires") as caught:
        prefetch.runtime_spec(config, 32)
    assert '"hardware": "h100"' in str(caught.value)
    config = SimpleNamespace(
        dynamic_cp_runtime_config=config.dynamic_cp_runtime_config,
        max_seqlen_per_dp_cp_rank=1024,
        min_dynamic_context_parallel_size=8,
    )
    spec = prefetch.runtime_spec(config, 32)
    assert spec["model"] == "usp_calib_v1"
    assert spec["model_options"] == {"allow_provisional": True}
    assert spec["geometry"]["allowed_cp_sizes"] == (8,)
    assert "required_global_batch_size" not in spec
    planner = dcp_runtime_worker._initialize_planner(spec)
    assert isinstance(planner.model, USPCalibV1CostModel)
    assert planner.model.metadata()["modeled_transport"] == "U8 all-to-all + ring"


def test_calibration_overrides_the_coefficients_of_the_selected_recipe(
    tmp_path, profile_data, profile_path
):
    config = _config(tmp_path, profile_data, profile_path, {"calibration": profile_path.name})
    spec = prefetch.runtime_spec(config, 32)
    # A relative path is resolved beside the runtime file.
    assert spec["model_options"] == {"calibration": str(profile_path.resolve())}
    planner = dcp_runtime_worker._initialize_planner(spec)
    metadata = planner.model.metadata()
    assert metadata["calibration_sha256"] == hashlib.sha256(profile_path.read_bytes()).hexdigest()
    assert planner.model.forward((1024, 1024), 4) == 16.4375
    missing = _config(tmp_path, profile_data, profile_path, {"calibration": "missing.json"})
    with pytest.raises(ValueError, match="Cannot read H100 calibration") as caught:
        prefetch.runtime_spec(missing, 32)
    assert "training configuration" not in str(caught.value)
    # An upstream-schema table belongs to the original profile, not to this recipe.
    table = Path(prefetch.__file__).parent / "dcp_runtime_scheduler/models/usp_calib_v1.json"
    wrong = _config(tmp_path, profile_data, profile_path, {"calibration": str(table)})
    with pytest.raises(ValueError, match="H100 calibration"):
        prefetch.runtime_spec(wrong, 32)


@pytest.mark.parametrize(
    "changes,named",
    [
        ({"num_layers": 4}, "num_layers=48, got 4"),
        ({"num_moe_experts": 8}, "num_moe_experts=128, got 8"),
        ({"bf16": False}, "bf16=True, got False"),
        ({"fp8": "hybrid"}, "fp8=None, got 'hybrid'"),
        ({"cp_comm_type": "a2a"}, "cp_comm_type='p2p', got 'a2a'"),
        ({"cp_comm_type": ["p2p", "a2a"]}, "cp_comm_type='p2p'"),
        ({"multi_latent_attention": True}, "multi_latent_attention=False, got True"),
        ({"experimental_attention_variant": "gated_delta_net"}, "experimental_attention_variant"),
        ({"moe_router_force_load_balancing": False}, "moe_router_force_load_balancing=True"),
        ({"cp_partition_mode": "contiguous"}, "cp_partition_mode='zigzag', got 'contiguous'"),
        ({"attention_backend": "flash"}, "attention_backend='fused', got 'flash'"),
        ({"pad_packed_seq_alignment": 6144}, "fixed per-rank buffer"),
        ({"thd_tail_padding_policy": "append_dummy_seq"}, "thd_tail_padding_policy"),
        ({"thd_tail_padding_policy": None}, "thd_tail_padding_policy"),
        ({"recompute_granularity": "full"}, "recompute_granularity=None, got 'full'"),
        ({"cuda_graph_impl": "local"}, "cuda_graph_impl='none', got 'local'"),
    ],
)
def test_training_configuration_outside_the_measured_recipe_is_named_at_startup(
    tmp_path, changes, named
):
    config = _packaged_config(tmp_path, **changes)
    with pytest.raises(ValueError, match="does not match the training configuration") as caught:
        prefetch.runtime_spec(config, 32)
    message = str(caught.value)
    assert named in message
    assert "The H100 calibration (Qwen3-30B-A3B, classic P2P) was measured with" in message


@pytest.mark.parametrize(
    "changes,named",
    [
        ({"mtp_num_layers": 1}, "mtp_num_layers=None or 0, got 1"),
        ({"moe_layer_freq": 2}, "moe_layer_freq=1, got 2"),
        ({"moe_layer_freq": [1, 0] * 24}, "moe_layer_freq=1, got [1, 0, 1, 0, "),
        ({"moe_layer_freq": []}, "moe_layer_freq=1, got []"),
        (
            {"moe_shared_expert_intermediate_size": 768},
            "moe_shared_expert_intermediate_size=None, got 768",
        ),
        ({"attention_output_gate": True}, "attention_output_gate=False, got True"),
        ({"is_hybrid_model": True}, "is_hybrid_model=False, got True"),
    ],
)
def test_model_features_the_measurement_did_not_use_are_named_at_startup(tmp_path, changes, named):
    # These settings are not entries of the calibration file; the model class requires them.
    assert not set(changes) & set(_packaged()["domain"]["config"])
    config = _packaged_config(tmp_path, **changes)
    with pytest.raises(ValueError, match="does not match the training configuration") as caught:
        prefetch.runtime_spec(config, 32)
    message = str(caught.value)
    assert named in message
    assert "The H100 calibration (Qwen3-30B-A3B, classic P2P) was measured with" in message


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"mtp_num_layers": None},
        {"mtp_num_layers": 0},
        {"moe_layer_freq": 1},
        {"moe_layer_freq": [1] * 48},
        {
            "moe_shared_expert_intermediate_size": None,
            "attention_output_gate": False,
            "is_hybrid_model": False,
        },
    ],
)
def test_unused_model_features_switched_off_or_absent_are_accepted(tmp_path, changes):
    spec = prefetch.runtime_spec(_packaged_config(tmp_path, **changes), 32)
    assert spec["model"] == "qwen3_h100_p2p"


def test_unused_model_features_also_bind_a_refit_file(tmp_path, profile_data, profile_path):
    config = _config(tmp_path, profile_data, profile_path, mtp_num_layers=1)
    with pytest.raises(ValueError, match="was measured with mtp_num_layers=None or 0, got 1"):
        prefetch.runtime_spec(config, 32)


def test_every_cost_model_class_declares_the_largest_cp_size_it_prices():
    names = ("dcp_benchmark", "qwen35_full", "usp_calib_v1", "qwen3_h100_p2p")
    assert [models.model_class(name).largest_cp_size for name in names] == [64, 32, 32, 8]
    # The spec builder reads the declaration; it keeps no table of its own by model name.
    assert not hasattr(prefetch, "_MODEL_CP_CEILING")


def _register_measured_model(monkeypatch, cost_model):
    """Register ``cost_model`` as the packaged calibration of hardware "next"."""
    known = models.model_class
    monkeypatch.setitem(models._MEASURED_DYNAMIC_USP, "next", "next_p2p")
    monkeypatch.setattr(
        models, "model_class", lambda name: cost_model if name == "next_p2p" else known(name)
    )


def test_newly_registered_measured_model_brings_its_own_cp_ceiling(monkeypatch, tmp_path):
    class NextCostModel(Qwen3H100P2PCostModel):
        largest_cp_size = 4

    _register_measured_model(monkeypatch, NextCostModel)
    spec = prefetch.runtime_spec(
        _packaged_config(tmp_path, {"hardware": "next", "max_cp_size": 4}), 32
    )
    assert (spec["model"], spec["hardware"]) == ("next_p2p", "next")
    assert spec["geometry"]["allowed_cp_sizes"] == (1, 2, 4)
    with pytest.raises(ValueError, match="prices CP groups of at most 4 ranks"):
        prefetch.runtime_spec(
            _packaged_config(tmp_path, {"hardware": "next"}, min_dynamic_context_parallel_size=8),
            32,
        )


def test_measured_model_without_a_declared_cp_ceiling_is_checked_by_its_domain(
    monkeypatch, tmp_path
):
    class UndeclaredCostModel(Qwen3H100P2PCostModel):
        largest_cp_size = None

    _register_measured_model(monkeypatch, UndeclaredCostModel)
    # Every power-of-two size up to DPxCP is a candidate; the model's own domain check
    # then names the sizes it was measured with, at start-up.
    config = _packaged_config(tmp_path, {"hardware": "next", "max_cp_size": None})
    with pytest.raises(ValueError, match="does not match the training configuration") as caught:
        prefetch.runtime_spec(config, 32)
    assert "covers CP groups of [1, 2, 4, 8] ranks, got [1, 2, 4, 8, 16, 32]" in str(caught.value)
    spec = prefetch.runtime_spec(
        _packaged_config(tmp_path, {"hardware": "next", "max_cp_size": 8}), 32
    )
    assert spec["geometry"]["allowed_cp_sizes"] == (1, 2, 4, 8)


def test_unset_training_fields_are_reported_as_unset(tmp_path):
    config = _packaged_config(tmp_path)
    del config.moe_router_force_load_balancing
    with pytest.raises(ValueError, match="moe_router_force_load_balancing=True, got <unset>"):
        prefetch.runtime_spec(config, 32)


@pytest.mark.parametrize(
    "world,changes,named",
    [
        (16, {}, "DPxCP=32, got 16"),
        (64, {}, "DPxCP=32, got 64"),
        (32, {"expert_model_parallel_size": 4}, "expert_model_parallel_size=8, got 4"),
        (32, {"max_seqlen_per_dp_cp_rank": 8192}, "max_seqlen_per_dp_cp_rank=4096, got 8192"),
    ],
)
def test_parallel_geometry_outside_the_measured_recipe_is_named_at_startup(
    tmp_path, world, changes, named
):
    config = _packaged_config(tmp_path, **changes)
    with pytest.raises(ValueError, match=named) as caught:
        prefetch.runtime_spec(config, world)
    # A fault of the training configuration, not of the runtime file.
    assert "(type 'dynamic_usp') does not match the training configuration: " in str(caught.value)


def test_p2p_per_layer_lists_and_backend_enums_are_accepted(tmp_path):
    class Backend(Enum):
        fused = 1
        flash = 2

    config = _packaged_config(tmp_path, cp_comm_type=["p2p"] * 48, attention_backend=Backend.fused)
    assert prefetch.runtime_spec(config, 32)["model"] == "qwen3_h100_p2p"
    config = _packaged_config(tmp_path, attention_backend=Backend.flash)
    with pytest.raises(ValueError, match="attention_backend='fused', got 'flash'"):
        prefetch.runtime_spec(config, 32)


# --------------------------------------------------------------------------------------
# The training configuration: the real config class, validated the way training does it
# --------------------------------------------------------------------------------------
_FUSED_PERMUTATION_KERNELS = (
    "fused_permute",
    "fused_permute_with_probs",
    "fused_sort_chunks_by_index",
    "fused_sort_chunks_by_index_with_probs",
    "fused_unpermute",
)


def _recipe_config(monkeypatch, tmp_path, **overrides):
    """TransformerConfig of the measured Qwen3-30B-A3B recipe with dynamic CP on 32 ranks."""
    # The recipe enables moe_permute_fusion and TransformerConfig only checks that the fused
    # kernels import. Start-up validation never calls them, so stand in where they are absent.
    for name in _FUSED_PERMUTATION_KERNELS:
        if getattr(moe_utils, name) is None:
            monkeypatch.setattr(moe_utils, name, lambda *args, **kwargs: None)
    # TransformerConfig accepts sequence packing only with Transformer Engine >= 2.9. The
    # configuration checks under test do not depend on the installed version, so answer its
    # version predicate here and keep these tests running on images with an older one.
    monkeypatch.setattr(transformer_config, "HAVE_PACKAGING", True)
    monkeypatch.setattr(transformer_config, "is_te_min_version", lambda _version: True)
    path = tmp_path / "runtime.json"
    path.write_text('{"type": "dynamic_usp", "hardware": "h100", "max_cp_size": 8}')
    values = dict(
        # Model and execution settings exactly as the calibration records them.
        _packaged()["domain"]["config"],
        attention_backend=AttnBackend.fused,
        # Dynamic CP over P2P with a fixed 4096-token buffer on each of the 32 ranks.
        context_parallel_size=1,
        dynamic_context_parallel=True,
        # TransformerConfig accepts dynamic CP only with the per-token loss.
        calculate_per_token_loss=True,
        min_dynamic_context_parallel_size=1,
        cp_comm_type="p2p",
        pad_packed_seq_alignment=4096,
        thd_tail_padding_policy="extend_last",
        use_cpu_initialization=True,
        dynamic_cp_runtime_config=str(path),
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _start_up(monkeypatch, config, world_size=32):
    """Run the validation training performs on every rank before the first step."""
    single = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    parent = SimpleNamespace(rank=lambda: 0, size=lambda: world_size)
    # This process stands in for the DPxCP leader of the coverage broadcast.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda *args: 0)
    monkeypatch.setattr(prefetch, "_count_over_group", lambda flag, group: int(bool(flag)))
    data_schedule.validate_dynamic_cp_runtime(config, SimpleNamespace(dp_cp=parent, pp=single))
    return config._dynamic_cp_runtime["spec"]


_FULL_RECOMPUTE = dict(
    recompute_granularity="full", recompute_method="uniform", recompute_num_layers=48
)


def test_real_training_configuration_passes_start_up_validation(monkeypatch, tmp_path):
    config = _recipe_config(monkeypatch, tmp_path)
    # What TransformerConfig resolves for this run; the calibration has to agree with it.
    assert config.sequence_packing_scheduler == "default_dynamic_cp"
    assert config.attention_backend is AttnBackend.fused
    assert config.cp_partition_mode == "zigzag"
    assert config.recompute_modules == ["core_attn"]
    spec = _start_up(monkeypatch, config)
    assert spec["type"] == "dynamic_usp" and spec["model"] == "qwen3_h100_p2p"
    geometry = spec["geometry"]
    assert geometry["allowed_cp_sizes"] == (1, 2, 4, 8)
    assert (geometry["world_size"], geometry["ep_size"]) == (32, 8)
    assert (geometry["tokens_per_rank"], geometry["dataset_alignment"]) == (4096, 64)
    # extend_last keeps all 64 THD entries for real sequences.
    assert geometry["max_group_seqs"] == 64
    assert config._dynamic_cp_runtime["coverage_checked"]


@pytest.mark.parametrize(
    "overrides,named",
    [
        ({"moe_router_force_load_balancing": False}, "moe_router_force_load_balancing=True"),
        ({"num_layers": 24}, "num_layers=48, got 24"),
        ({"expert_model_parallel_size": 4}, "expert_model_parallel_size=8, got 4"),
        ({"max_seqlen_per_dp_cp_rank": 8192}, "max_seqlen_per_dp_cp_rank=4096, got 8192"),
        (_FULL_RECOMPUTE, "recompute_granularity=None, got 'full'"),
        ({"mtp_num_layers": 1}, "mtp_num_layers=None or 0, got 1"),
        ({"moe_layer_freq": [1, 0] * 24}, r"moe_layer_freq=1, got \[1, 0, "),
        (
            {"moe_shared_expert_intermediate_size": 768},
            "moe_shared_expert_intermediate_size=None, got 768",
        ),
        ({"attention_output_gate": True}, "attention_output_gate=False, got True"),
        ({"is_hybrid_model": True}, "is_hybrid_model=False, got True"),
    ],
)
def test_real_training_configuration_outside_the_recipe_is_rejected_at_start_up(
    monkeypatch, tmp_path, overrides, named
):
    config = _recipe_config(monkeypatch, tmp_path, **overrides)
    with pytest.raises(ValueError, match=named) as caught:
        _start_up(monkeypatch, config)
    assert "H100 calibration (Qwen3-30B-A3B, classic P2P)" in str(caught.value)
    assert "(type 'dynamic_usp')" in str(caught.value)


# --------------------------------------------------------------------------------------
# Planning: the DynamicUSP solver in the persistent CPU worker
# --------------------------------------------------------------------------------------
def test_dynamic_usp_solver_plans_with_the_measured_costs(profile_path):
    geometry = _geometry()
    model = Qwen3H100P2PCostModel(geometry, calibration=profile_path)
    planner = DynamicUSPRuntimePlanner(geometry, model)
    lengths = dict.fromkeys(range(64), 1024)
    incumbent = [[list(range(2 * rank, 2 * rank + 2)) for rank in range(32)]]
    plan = planner.plan_padded(lengths, logical_lengths=lengths, incumbent_rows=incumbent)
    validate_rows(plan.rows, lengths, geometry, 1)
    assert plan.predicted_ms > 0
    assert planner.model is model
    assert all(
        cp in (1, 2, 4, 8) and start % cp == 0 for row in plan.groups for start, cp, _ in row
    )


def test_measured_model_runs_inside_the_persistent_cpu_worker(tmp_path, profile_data, profile_path):
    config = _config(tmp_path, profile_data, profile_path)
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
        assert result["solver_entry"].endswith("DynamicUSPRuntimePlanner.plan_padded")
        assert result["model_metadata"]["name"] == "qwen3_h100_p2p"
        assert result["model_metadata"]["nvlink_domain_size"] == 8
        assert result["model_metadata"]["modeled_transport"] == "classic P2P"
        assert result["statistics"]["group_sizes"] == [1, 2, 4, 8]
        assert "incumbent_rejection" not in result["statistics"]


@pytest.mark.parametrize("count", [128, 255, 257])
def test_worker_rejects_other_global_batch_sizes_with_one_schedule_error(
    tmp_path, profile_data, profile_path, count
):
    profile_data["domain"]["step_overhead_global_batch_size"] = 256
    profile_path.write_text(json.dumps(profile_data))
    worker = prefetch._CpuPlanner(
        prefetch.runtime_spec(_config(tmp_path, profile_data, profile_path), 32)
    )
    try:
        with pytest.raises(RuntimeScheduleError) as caught:
            worker.solve([1024] * count, [1024] * count, None)
    finally:
        worker.close()
    assert str(caught.value) == (
        "The 'dynamic_usp' calibration was measured with 256 samples per global batch; "
        f"this global batch has {count}"
    )


@pytest.mark.parametrize("length", [16385, 20000, 32768])
def test_worker_names_a_sample_longer_than_the_measured_range(tmp_path, length):
    spec = prefetch.runtime_spec(_packaged_config(tmp_path), 32, max_num_seqs=64)
    valid = [64, 512, 1024, 2048, 3072, 4096, 8192, 16384] * 32
    lengths = list(valid)
    lengths[17] = length
    aligned = [[(n + 63) // 64 * 64 for n in batch] for batch in (lengths, valid)]
    worker = prefetch._CpuPlanner(spec)
    try:
        # CP 8 holds 32768 tokens, so the sample fits a group, but the calibration prices
        # samples of at most 16384 tokens: one error that names the sample and that limit.
        with pytest.raises(RuntimeScheduleError) as caught:
            worker.solve(lengths, aligned[0], None)
        # The worker keeps serving; the next global batch within the range is planned.
        planned = worker.solve(valid, aligned[1], None)
    finally:
        worker.close()
    assert str(caught.value) == (
        f"sample 17: seq_len={length} is longer than the 16384 tokens that the 'dynamic_usp' "
        "cost model prices (its measured sample-length range); it cannot be scheduled with "
        "allowed_cp_sizes=(1, 2, 4, 8), max_seqlen_per_dp_cp_rank=4096"
    )
    assert "cost model rejected" not in str(caught.value)
    validate_rows(planned["rows"], dict(enumerate(aligned[1])), _geometry(), 1)
    assert planned["predicted_ms"] > 0


def test_worker_reports_a_sample_beyond_every_cp_group_as_before(tmp_path):
    spec = prefetch.runtime_spec(_packaged_config(tmp_path), 32, max_num_seqs=64)
    lengths = [1024] * 255 + [32769]
    padded = [(n + 63) // 64 * 64 for n in lengths]
    worker = prefetch._CpuPlanner(spec)
    try:
        # Longer than the widest group holds: the geometry's own message, unchanged.
        with pytest.raises(RuntimeScheduleError, match="sample 255: seq_len=32769") as caught:
            worker.solve(lengths, padded, None)
    finally:
        worker.close()
    assert "requires CP >= 9 before padding" in str(caught.value)
    assert "measured sample-length range" not in str(caught.value)


def test_sample_length_range_is_checked_after_the_global_batch_size(tmp_path):
    planner = dcp_runtime_worker._initialize_planner(
        prefetch.runtime_spec(_packaged_config(tmp_path), 32, max_num_seqs=64)
    )
    spec = {"type": "dynamic_usp"}
    lengths = [1024] * 254 + [20032]
    with pytest.raises(RuntimeScheduleError, match="256 samples per global batch"):
        dcp_runtime_worker._solve(planner, spec, lengths, lengths, None)
    with pytest.raises(RuntimeScheduleError, match="sample 255: seq_len=20032 is longer than"):
        dcp_runtime_worker._solve(planner, spec, [1024] + lengths, [1024] + lengths, None)


def test_packaged_calibration_plans_its_global_batch_twice_in_one_worker(tmp_path):
    spec = prefetch.runtime_spec(_packaged_config(tmp_path), 32, max_num_seqs=64)
    lengths = [64, 512, 1024, 2048, 3072, 4096, 8192, 16384] * 32
    padded = [(length + 63) // 64 * 64 for length in lengths]
    worker = prefetch._CpuPlanner(spec)
    try:
        first = worker.solve(lengths, padded, None)
        second = worker.solve(lengths, padded, None)
        with pytest.raises(RuntimeScheduleError, match="256 samples per global batch"):
            worker.solve(lengths[:-1], padded[:-1], None)
        third = worker.solve(lengths, padded, None)
    finally:
        worker.close()
    groups = validate_rows(first["rows"], dict(enumerate(padded)), _geometry(), 1)
    assert {cp for row in groups for _, cp, _ in row} <= {1, 2, 4, 8}
    assert first["predicted_ms"] > 0
    assert first["worker_pid"] == second["worker_pid"] == third["worker_pid"]
    assert first["rows"] == second["rows"] == third["rows"]
    assert first["model_metadata"]["calibration_sha256"] == PACKAGED_SHA256
    assert not first["worker_cuda_initialized"]
    assert not first["worker_distributed_initialized"]


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


@pytest.mark.parametrize("cp", [16, 32])
def test_default_incumbent_wider_than_the_measured_sizes_is_dropped(
    tmp_path, profile_data, profile_path, cp
):
    worker = prefetch._CpuPlanner(
        prefetch.runtime_spec(_config(tmp_path, profile_data, profile_path), 32)
    )
    lengths = [1024] * 64
    try:
        result = worker.solve(lengths, lengths, _wide_incumbent(cp))
    finally:
        worker.close()
    validate_rows(result["rows"], dict(enumerate(lengths)), _geometry(), 1)
    statistics = result["statistics"]
    assert statistics["incumbent_rejection"] == "default_incumbent_exceeds_max_cp_size"
    assert statistics["incumbent_unsupported_cp_sizes"] == [cp]
    assert not statistics["selected_incumbent"]
    assert result["predicted_ms"] > 0
    assert not result["worker_cuda_initialized"]


def test_planner_validation_remains_strict_for_explicit_incumbents(profile_path):
    geometry = _geometry()
    model = Qwen3H100P2PCostModel(geometry, calibration=profile_path)
    lengths = dict.fromkeys(range(64), 1024)
    planner = DynamicUSPRuntimePlanner(geometry, model)
    with pytest.raises(RuntimeScheduleError, match="cp=32"):
        planner.plan_padded(lengths, logical_lengths=lengths, incumbent_rows=_wide_incumbent(32))
