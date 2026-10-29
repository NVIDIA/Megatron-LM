# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU contract tests using explicitly synthetic GPU metadata and timings."""

import copy
import json
import statistics
import sys

import pytest

from tests.unit_tests.determinism_reporting.test_paired_performance import (
    SCRIPTS,
    benchmark,
    load_module,
)

REVISION = "a" * 40


@pytest.fixture
def author(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setitem(sys.modules, "benchmark", benchmark)
    return load_module("author_evidence")


def contract():
    return {
        "adapter": "local_activation_v1",
        "adapter_sha256": "d" * 64,
        "distribution": "local_without_collectives",
        "case": "weighted_swiglu",
        "op_id": "fused_bias_swiglu",
        "implementation": "torch.compile:weighted_swiglu",
        "input_seed": 1234,
        "upstream_gradient": "ones_like_output",
        "inputs": [
            {
                "shape": [4, 16],
                "stride": [16, 1],
                "dtype": "torch.bfloat16",
                "requires_grad": True,
                "sha256": "e" * 64,
            },
            None,
            {
                "shape": [4, 1],
                "stride": [1, 1],
                "dtype": "torch.float32",
                "requires_grad": True,
                "sha256": "f" * 64,
            },
        ],
        "runtime": {
            "python": "3.12.fixture",
            "packages": {
                "torch": "CPU-test-only",
                "triton": "fixture",
                "transformer-engine": "fixture",
            },
            "torch": "CPU-test-only",
            "cuda": "synthetic",
            "gpu": "H100 fixture",
            "capability": [9, 0],
            "driver": ["fixture-driver"],
            "deterministic_algorithms": True,
            "warn_only": False,
            "fill_uninitialized_memory": True,
            "autocast": False,
            "autocast_dtype": "torch.float16",
            "float32_matmul_precision": "highest",
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": True,
            "cudnn_deterministic": False,
            "cudnn_benchmark": False,
            "environment": {
                **benchmark.DET_ENV,
                "TRITON_CACHE_AUTOTUNING": None,
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_PROTO": None,
                "TRITON_CACHE_DIR": None,
            },
        },
    }


def coverage():
    adapter = contract()
    signature = {
        "implementation": adapter["implementation"],
        "op_id": adapter["op_id"],
        "phase": "forward_backward",
        "deterministic_algorithms": True,
        "configuration": {"kernel_case": adapter},
        "inputs": [
            (
                {key: value[key] for key in ("shape", "stride", "dtype", "requires_grad")}
                if value is not None
                else None
            )
            for value in adapter["inputs"]
        ],
    }
    observations = [
        {
            "rank": rank,
            "signature": copy.deepcopy(signature),
            "status": "verified_deterministic",
            "compared_outputs": 1,
            "compared_gradients": 2,
            "protocol": {"replays": 3, "warn_only": False, "contention": True},
        }
        for rank in range(2)
    ]
    checks = [
        {
            "rank": rank,
            "signature": copy.deepcopy(signature),
            "kind": kind,
            "status": "passed",
            "compared_outputs": 1,
            "compared_gradients": 2,
            "detected_perturbations": 3,
        }
        for rank in range(2)
        for kind in ("reference", "sensitivity")
    ]
    return {
        "schema_version": 1,
        "kind": "determinism_coverage",
        "run_id": "CPU-fixture-not-GPU-evidence",
        "context": {"revision": REVISION, "dirty": False, "world_size": 2},
        "ranks_present": [0, 1],
        "author_requirements": [
            {"op_id": adapter["op_id"], "test": "test_weighted[bf16]", "status": "passed"}
        ],
        "cases": [
            {
                "op_id": adapter["op_id"],
                "case_id": "test_weighted[bf16]",
                "status": "verified_deterministic",
                "check_status": {"reference": "passed", "sensitivity": "passed"},
                "observations": observations,
                "checks": checks,
            }
        ],
    }


def timings(phase, *, has_base=False, limit=None, regression=None, factor=1):
    measurement = {
        "kernel_case": "weighted_swiglu",
        "phase": phase,
        "tokens": 4,
        "hidden_size": 8,
        "dtype": "bfloat16",
        "warmup": 2,
        "steps": 3,
        "pairs": 3,
        "gpus": 1,
        "timing": "cuda_event_ms",
    }
    keys = ("kernel_case", "phase", "tokens", "hidden_size", "dtype", "warmup", "steps")
    runs = []
    for pair in range(3):
        for label in (("head", "base") if has_base else ("head",)):
            for mode in ("default", "det"):
                adapter = contract()
                adapter["runtime"]["deterministic_algorithms"] = mode == "det"
                policy = benchmark.DET_ENV if mode == "det" else benchmark.DEFAULT_ENV
                adapter["runtime"]["environment"].update(
                    {key: policy.get(key) for key in benchmark.MODE_ENV}
                )
                samples = [
                    (12.0 if mode == "det" else 10.0) * (factor if label == "head" else 1)
                ] * 3
                kernel = {
                    "measurement": {key: measurement[key] for key in keys},
                    "mode": mode,
                    "deterministic_algorithms": mode == "det",
                    "samples_ms": samples,
                    "case_signature": adapter,
                    "device_uuid": "GPU-fixture-0",
                }
                runs.append(
                    {
                        "pair": pair,
                        "revision_label": label,
                        "mode": mode,
                        "status": "complete",
                        "median_ms": statistics.median(samples),
                        "samples_ms": samples,
                        "kernel": kernel,
                    }
                )
    comparisons = benchmark.summarize(runs, 3, has_base, limit, regression)
    statuses = [value["status"] for key, value in comparisons.items() if key != "base_overhead"]
    status = (
        "fail"
        if "fail" in statuses
        else (
            "inconclusive"
            if "inconclusive" in statuses
            else "pass" if "pass" in statuses else "reported"
        )
    )
    sources = {"head": {"revision": REVISION, "dirty": False}}
    if has_base:
        sources["base"] = {"revision": "b" * 40, "dirty": False}
    return {
        "schema_version": 1,
        "kind": "determinism_kernel_performance",
        "measurement": measurement,
        "sources": sources,
        "machine": {"gpus": ["GPU-fixture-0"]},
        "runs": runs,
        "comparisons": comparisons,
        "status": status,
    }


@pytest.mark.parametrize(
    "has_base,limit,regression,status",
    [
        (False, None, None, "not_gated"),
        (False, 1.35, None, "not_gated"),
        (True, 1.35, None, "not_gated"),
        (True, 1.35, 1.05, "passed"),
    ],
)
def test_complete_bundle_requires_both_revision_limits_for_performance_pass(
    author, has_base, limit, regression, status
):
    reports = [
        timings(phase, has_base=has_base, limit=limit, regression=regression)
        for phase in ("forward", "backward")
    ]
    result = author.join(coverage(), reports, REVISION)
    assert result["evidence_complete"] is True
    assert result["status"] == status


def test_shared_slowdown_fails_even_when_overhead_is_unchanged(author):
    reports = [
        timings(phase, has_base=True, limit=1.35, regression=1.05, factor=1.2)
        for phase in ("forward", "backward")
    ]
    result = author.join(coverage(), reports, REVISION)
    assert result["status"] == "failed"
    comparison = result["requirements"][0]["phases"]["backward"]["comparisons"]
    assert comparison["head_overhead"]["status"] == "pass"
    assert comparison["det_regression"]["status"] == "fail"


@pytest.mark.parametrize(
    "problem",
    [
        "missing_phase",
        "duplicate_phase",
        "model_report",
        "source",
        "dirty",
        "missing_gpu",
        "uuid",
        "different_uuid",
        "hash",
        "stride",
        "gradient",
        "runtime",
        "hardware",
        "software",
        "default_policy",
        "warn_only",
        "missing_arm",
        "raw_sample",
        "median",
        "comparison",
        "status",
        "warmup",
        "pair_count",
        "shape",
        "phase",
        "dtype",
        "base_revision",
    ],
)
def test_incompatible_or_incomplete_timings_cannot_supply_author_evidence(author, problem):
    reports = [
        timings(phase, has_base=True, limit=1.35, regression=1.05)
        for phase in ("forward", "backward")
    ]
    report = reports[0]
    run = report["runs"][0]
    adapter = run["kernel"]["case_signature"]
    if problem == "missing_phase":
        reports.pop()
    elif problem == "duplicate_phase":
        reports.append(copy.deepcopy(report))
    elif problem == "model_report":
        report["kind"] = "determinism_performance"
    elif problem == "source":
        report["sources"]["head"]["revision"] = "c" * 40
    elif problem == "dirty":
        report["sources"]["base"]["dirty"] = True
    elif problem == "missing_gpu":
        report["machine"]["gpus"] = None
    elif problem == "uuid":
        run["kernel"].pop("device_uuid")
    elif problem == "different_uuid":
        run["kernel"]["device_uuid"] = "GPU-fixture-other"
    elif problem == "hash":
        adapter["inputs"][0]["sha256"] = "1" * 64
    elif problem == "stride":
        adapter["inputs"][0]["stride"] = [1, 4]
    elif problem == "gradient":
        adapter["upstream_gradient"] = "random"
    elif problem == "runtime":
        adapter["runtime"]["fill_uninitialized_memory"] = False
    elif problem == "hardware":
        adapter["runtime"]["gpu"] = "GB200 fixture"
    elif problem == "software":
        adapter["runtime"]["cuda"] = "different CUDA"
    elif problem == "default_policy":
        adapter["runtime"]["environment"]["NCCL_ALGO"] = "Ring"
    elif problem == "warn_only":
        adapter["runtime"]["warn_only"] = True
    elif problem == "missing_arm":
        report["runs"].pop()
    elif problem == "raw_sample":
        run["kernel"]["samples_ms"] = [1.0, 2.0]
    elif problem == "median":
        run["median_ms"] = 5
    elif problem == "comparison":
        report["comparisons"]["head_overhead"]["median_ratio"] = 0.9
    elif problem == "status":
        report["status"] = "reported"
    elif problem == "warmup":
        report["measurement"]["warmup"] = 0
    elif problem == "pair_count":
        report["measurement"]["pairs"] = 1
    elif problem == "shape":
        report["measurement"]["tokens"] = 8
    elif problem == "phase":
        report["measurement"]["phase"] = "forward_backward"
    elif problem == "dtype":
        report["measurement"]["dtype"] = "float32"
    else:
        report["sources"]["base"]["revision"] = "c" * 40
    result = author.join(coverage(), reports, REVISION)
    assert result["status"] == "not_verified"
    assert result["evidence_complete"] is False


@pytest.mark.parametrize(
    "problem",
    [
        "empty",
        "missing_case",
        "rank",
        "stale",
        "dirty",
        "skip",
        "reference",
        "sensitivity",
        "raw_replay",
        "raw_checks",
        "counts",
        "adapter",
        "different_rank_input",
        "missing_fingerprint",
        "input_metadata",
        "forward_only",
        "warn_only",
        "no_contention",
    ],
)
def test_missing_or_contradictory_numerical_evidence_cannot_pass(author, problem):
    evidence = coverage()
    case = evidence["cases"][0]
    signature = case["observations"][0]["signature"]
    if problem == "empty":
        evidence["author_requirements"] = []
    elif problem == "missing_case":
        evidence["cases"] = []
    elif problem == "rank":
        evidence["ranks_present"].pop()
    elif problem == "stale":
        evidence["context"]["revision"] = "b" * 40
    elif problem == "dirty":
        evidence["context"]["dirty"] = True
    elif problem == "skip":
        case["status"] = "not_verified"
    elif problem in ("reference", "sensitivity"):
        case["check_status"][problem] = "failed"
    elif problem == "raw_replay":
        case["observations"][0]["status"] = "verified_nondeterministic"
    elif problem == "raw_checks":
        case["checks"].pop()
    elif problem == "counts":
        case["checks"][0]["compared_gradients"] = 1
    elif problem == "adapter":
        signature["configuration"]["kernel_case"]["adapter"] = "collective"
    elif problem == "different_rank_input":
        signature["configuration"]["kernel_case"]["inputs"][0]["sha256"] = "1" * 64
    elif problem == "missing_fingerprint":
        signature["configuration"]["kernel_case"]["inputs"][0].pop("sha256")
    elif problem == "input_metadata":
        signature["inputs"][0]["shape"] = [8, 8]
    elif problem == "forward_only":
        signature["phase"] = "forward"
    elif problem == "warn_only":
        case["observations"][0]["protocol"]["warn_only"] = True
    else:
        case["observations"][0]["protocol"]["contention"] = False
    result = author.join(evidence, [timings("forward"), timings("backward")], REVISION)
    assert result["evidence_complete"] is False
    assert result["status"] == (
        "failed" if problem in ("reference", "sensitivity") else "not_verified"
    )


def test_cli_keeps_hashed_inputs_and_unknowns_and_separates_evidence_from_budget_gate(
    author, tmp_path
):
    coverage_file = tmp_path / "coverage.json"
    timing_file = tmp_path / "leaderboard.json"
    coverage_file.write_text(json.dumps(coverage()))
    timing_file.write_text(json.dumps([timings("forward"), timings("backward")]))
    args = [
        "--coverage",
        str(coverage_file),
        "--performance",
        str(timing_file),
        "--revision",
        REVISION,
    ]
    target = tmp_path / "author.json"
    assert author.main([*args, "--output", str(target)]) == 0
    saved = json.loads(target.read_text())
    assert saved["status"] == "not_gated"
    assert len(saved["inputs"]) == 2 and all(len(row["sha256"]) == 64 for row in saved["inputs"])
    assert (
        author.main([*args, "--output", str(tmp_path / "gated.json"), "--require-performance-pass"])
        == 2
    )
    with pytest.raises(SystemExit):
        author.main([*args, "--output", str(target)])
    timing_file.write_text("[]")
    missing = tmp_path / "missing.json"
    assert author.main([*args, "--output", str(missing)]) == 2
    assert missing.with_suffix(".md").exists()


def test_model_coverage_and_malformed_json_are_not_operator_author_evidence(author, tmp_path):
    evidence = coverage()
    evidence["kind"] = "model_determinism_replay"
    with pytest.raises(ValueError, match="model"):
        author.join(evidence, [], REVISION)
    path = tmp_path / "bad.json"
    path.write_text("{")
    target = tmp_path / "report.json"
    assert (
        author.main(
            [
                "--coverage",
                str(path),
                "--performance",
                str(path),
                "--revision",
                REVISION,
                "--output",
                str(target),
            ]
        )
        == 2
    )
    assert json.loads(target.read_text())["status"] == "not_verified"


@pytest.mark.parametrize("problem", ["null", "shape", "inputs", "nonfinite", "short_revision"])
def test_invalid_artifacts_keep_diagnostics_instead_of_crashing(author, tmp_path, problem):
    evidence = coverage()
    reports = [timings(phase, has_base=True) for phase in ("forward", "backward")]
    if problem == "null":
        evidence = None
    elif problem == "shape":
        for report in reports:
            report["runs"][0]["kernel"]["case_signature"]["inputs"][0]["shape"] = []
    elif problem == "inputs":
        for row in evidence["cases"][0]["observations"]:
            row["signature"]["configuration"]["kernel_case"]["inputs"] = [None]
            row["signature"]["inputs"] = [None]
    elif problem == "nonfinite":
        reports[0]["measurement"]["tokens"] = float("nan")
    else:
        reports[0]["sources"]["base"]["revision"] = "abc1234"
    coverage_file, timings_file = tmp_path / "coverage.json", tmp_path / "timings.json"
    coverage_file.write_text(json.dumps(evidence))
    timings_file.write_text(json.dumps(reports))
    target = tmp_path / "report.json"
    assert (
        author.main(
            [
                "--coverage",
                str(coverage_file),
                "--performance",
                str(timings_file),
                "--revision",
                REVISION,
                "--output",
                str(target),
            ]
        )
        == 2
    )
    saved = json.loads(target.read_text())
    assert saved["status"] == "not_verified"
    assert saved.get("error") or saved["requirements"][0].get("reason")


@pytest.mark.parametrize("connections,expected", [("1", True), ("32", False), (None, False)])
def test_serialized_replay_requires_explicit_launch_constraint(author, connections, expected):
    protocol = {"contention": False, "contention_requested": True}
    assert (
        author.replay_contention_supported(
            protocol, {"environment": {"CUDA_DEVICE_MAX_CONNECTIONS": connections}}
        )
        is expected
    )
    assert not author.replay_contention_supported(
        {"contention": False}, {"environment": {"CUDA_DEVICE_MAX_CONNECTIONS": connections}}
    )
