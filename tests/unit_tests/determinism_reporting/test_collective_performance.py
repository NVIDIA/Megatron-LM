# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU controls for captured collective performance, using synthetic GPU records."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.determinism_reporting.test_paired_performance import SCRIPTS, load_module


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return load_module("collective_case")


def capture_records():
    """Two ranks with distinct bytes/UUIDs; metadata is explicitly synthetic."""
    context = {
        "revision": "a" * 40,
        "dirty": False,
        "world_size": 2,
        "environment": {
            "NCCL_ALGO": "Ring",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "MAMBA_DETERMINISTIC": "1",
            "CAUSAL_CONV1D_DETERMINISTIC": "1",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUDA_DEVICE_MAX_CONNECTIONS": "32",
            "TRITON_CACHE_AUTOTUNING": "0",
            "TRITON_CACHE_DIR": "/synthetic-cache",
            "TRITON_AUTOTUNE_BLOCK_M": "16",
        },
    }
    captures = []
    for rank in range(2):
        tensor = {
            "shape": [2, 3],
            "stride": [3, 1],
            "storage_offset": 2,
            "dtype": "torch.float32",
            "requires_grad": True,
            "sha256": str(rank) * 64,
        }
        collective = {
            "case": "copy",
            "group_ranks": [0, 1],
            "group_rank": rank,
            "size": 2,
            "backend": "nccl",
            "device_uuid": f"GPU-fixture-{rank}",
            "nccl_version": [2, 30, 7],
            "nccl_environment": {
                "NCCL_ALGO": "Ring",
                "NCCL_PROTO": "Simple",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            },
            "group_options": {
                "is_high_priority_stream": True,
                "config": {"min_ctas": 2, "max_ctas": 4},
                "flags": {},
            },
            "input": tensor,
            "grad_enabled": True,
            "warn_only": False,
        }
        events = []
        for index in range(2):
            fields = copy.deepcopy(collective)
            if index:
                fields.update(
                    gradient={**tensor, "requires_grad": False}, backward_grad_enabled=False
                )
            events.append(
                {
                    "call_id": 0,
                    "signature": {
                        "op_id": "tensor_parallel_mappings",
                        "implementation": "mcore:copy_to_tensor_model_parallel_region",
                        "phase": "forward_backward" if index else "forward",
                        "deterministic_algorithms": True,
                        "inputs": [
                            {
                                key: tensor[key]
                                for key in ("shape", "stride", "dtype", "requires_grad")
                            }
                        ],
                        "runtime": {
                            "fill_uninitialized_memory": False,
                            "cudnn_deterministic": True,
                        },
                        "configuration": {"collective": fields},
                    },
                }
            )
        captures.append(
            {
                "rank": rank,
                "context": copy.deepcopy(context),
                "recipe_id": "CPU-fixture",
                "events": events,
            }
        )
    return captures


def replay_records(captures):
    """Complete synthetic producer observations and matching numerical controls."""
    cases = []
    for index in range(2):
        observations = [
            {
                "rank": rank,
                "signature": report["events"][index]["signature"],
                "status": "verified_deterministic",
                "protocol": {"replays": 3, "warn_only": False, "contention": True},
                "compared_outputs": 1,
                "compared_gradients": index,
            }
            for rank, report in enumerate(captures)
        ]
        cases.append(
            {
                "case_id": f"event-{index}",
                "status": "verified_deterministic",
                "observations": observations,
                "check_status": {"reference": "passed", "sensitivity": "passed"},
                "checks": [
                    {
                        **copy.deepcopy(row),
                        "kind": kind,
                        "status": "passed",
                        "detected_perturbations": 1 + index,
                    }
                    for row in observations
                    for kind in ("reference", "sensitivity")
                ],
            }
        )
    return {
        "schema_version": 1,
        "kind": "determinism_coverage",
        "run_id": "CPU-fixture-only",
        "context": copy.deepcopy(captures[0]["context"]),
        "ranks_present": [0, 1],
        "cases": cases,
    }


def measurement():
    """A small paired protocol; no values represent actual GPU measurements."""
    return {
        "pairs": 3,
        "steps": 3,
        "warmup": 2,
        "event_indices": [0, 1],
        "world_size": 2,
        "manifest_sha256": ["b" * 64, "c" * 64],
        "tooling": {"fixture.py": "d" * 64},
    }


def rank_results(adapter, captures, settings, mode="det", revision="a" * 40):
    """Alternate the slower rank so max-of-medians gives the wrong answer."""
    return [
        {
            "adapter": adapter.ADAPTER,
            "rank": rank,
            "mode": mode,
            "measurement": copy.deepcopy(settings),
            "context": adapter.mode_context(report["context"], mode, revision),
            "context_after": adapter.mode_context(report["context"], mode, revision),
            "capture_manifest_sha256": settings["manifest_sha256"][rank],
            "tooling": settings["tooling"],
            "communicators": {
                adapter.group_key(event["signature"]["configuration"]["collective"]): {
                    "requested": copy.deepcopy(
                        event["signature"]["configuration"]["collective"]["group_options"]
                    ),
                    "before_initialize": copy.deepcopy(
                        event["signature"]["configuration"]["collective"]["group_options"]
                    ),
                    "after_initialize": adapter.resolved_options(
                        event["signature"]["configuration"]["collective"]
                    ),
                }
                for event in report["events"]
            },
            "rows": {
                str(index): {
                    "capture_signature": copy.deepcopy(event["signature"]),
                    "actual_signature": adapter.timing_signature(event["signature"], mode),
                    "signature_after": adapter.timing_signature(event["signature"], mode),
                    "phase": "backward" if index else "forward",
                    "samples_ms": [10, 1, 1] if rank == 0 else [1, 10, 1],
                }
                for index, event in enumerate(report["events"])
            },
        }
        for rank, report in enumerate(captures)
    ]


def test_group_samples_use_aligned_maxima_not_rank_medians(adapter):
    captures, settings = capture_records(), measurement()
    results = rank_results(adapter, captures, settings)
    rows = adapter.aggregate_arm(results, captures, settings, "det", "a" * 40)
    assert rows["event-0/ranks-0-1"]["samples_ms"] == [10, 10, 1]
    assert rows["event-0/ranks-0-1"]["median_ms"] == 10
    runs = [
        {"pair": pair, "revision_label": revision, "mode": mode, "status": "complete", "rows": rows}
        for pair in range(3)
        for revision in ("base", "head")
        for mode in ("default", "det")
    ]
    compared = adapter.comparisons(runs, settings, True, None, None)
    assert all(
        value["status"] == "not_gated" for row in compared.values() for value in row.values()
    )
    assert all(
        value["paired_ratios"] == [1, 1, 1] for row in compared.values() for value in row.values()
    )
    with pytest.raises(ValueError):
        adapter.comparisons(runs[:-1], settings, True, None, None)


@pytest.mark.parametrize("nonblocking,effective", [(None, 1), ("0", 1), ("1", 0)])
def test_lazy_blocking_resolution_keeps_requested_and_effective_options(
    adapter, nonblocking, effective
):
    captures, settings = capture_records(), measurement()
    for capture in captures:
        for event in capture["events"]:
            collective = event["signature"]["configuration"]["collective"]
            collective["group_options"]["config"]["blocking"] = -(2**31)
            collective["nccl_environment"]["TORCH_NCCL_USE_COMM_NONBLOCKING"] = nonblocking
    original = copy.deepcopy(captures)
    results = rank_results(adapter, captures, settings)
    adapter.aggregate_arm(results, captures, settings, "det", "a" * 40)
    assert captures == original
    for result in results:
        for record in result["communicators"].values():
            assert record["requested"]["config"]["blocking"] == -(2**31)
            assert record["after_initialize"]["config"]["blocking"] == effective
            assert record["after_initialize"]["config"]["min_ctas"] == 2
        assert (
            result["rows"]["0"]["actual_signature"]["configuration"]["collective"]["group_options"][
                "config"
            ]["blocking"]
            == effective
        )


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "priority",
        "cta",
        "blocking",
        "flags",
        "before",
        "requested",
        "unknown_environment",
    ],
)
def test_communicator_initialization_cannot_hide_option_drift(adapter, fault):
    captures, settings = capture_records(), measurement()
    for capture in captures:
        for event in capture["events"]:
            collective = event["signature"]["configuration"]["collective"]
            collective["group_options"]["config"]["blocking"] = -(2**31)
    if fault == "unknown_environment":
        captures[0]["events"][0]["signature"]["configuration"]["collective"]["nccl_environment"][
            "TORCH_NCCL_USE_COMM_NONBLOCKING"
        ] = "unknown"
        with pytest.raises(ValueError):
            rank_results(adapter, captures, settings)
        return
    results = rank_results(adapter, captures, settings)
    record = next(iter(results[0]["communicators"].values()))
    if fault == "missing":
        results[0]["communicators"] = {}
    elif fault == "priority":
        record["after_initialize"]["is_high_priority_stream"] = False
    elif fault == "cta":
        record["after_initialize"]["config"]["min_ctas"] = 3
    elif fault == "blocking":
        record["after_initialize"]["config"]["blocking"] = 0
    elif fault == "flags":
        record["after_initialize"]["flags"] = {"changed": True}
    elif fault == "before":
        record["before_initialize"]["config"]["min_ctas"] = 3
    elif fault == "requested":
        record["requested"]["config"]["blocking"] = 1
    with pytest.raises(ValueError):
        adapter.aggregate_arm(results, captures, settings, "det", "a" * 40)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_rank",
        "duplicate_rank",
        "missing_row",
        "phase",
        "uuid",
        "input",
        "gradient",
        "options",
        "fill",
        "policy",
        "context",
        "after",
        "tooling",
        "manifest",
        "short",
        "zero",
        "negative",
        "nan",
        "inf",
        "bool",
    ],
)
def test_corrupt_timing_cannot_be_aggregated(adapter, fault):
    captures, settings = capture_records(), measurement()
    results = rank_results(adapter, captures, settings)
    row = results[0]["rows"]["1"]
    if fault == "missing_rank":
        results.pop()
    elif fault == "duplicate_rank":
        results[1]["rank"] = 0
    elif fault == "missing_row":
        results[0]["rows"].pop("0")
    elif fault == "phase":
        row["phase"] = "forward"
    elif fault in ("uuid", "input", "gradient", "options"):
        key = {"uuid": "device_uuid", "options": "group_options"}.get(fault, fault)
        row["actual_signature"]["configuration"]["collective"][key] = "changed"
    elif fault == "fill":
        row["actual_signature"]["runtime"]["fill_uninitialized_memory"] = True
    elif fault == "policy":
        row["actual_signature"]["deterministic_algorithms"] = False
    elif fault == "context":
        results[0]["context"]["revision"] = "changed"
    elif fault == "after":
        row["signature_after"]["runtime"]["cudnn_deterministic"] = False
    elif fault == "tooling":
        results[0]["tooling"] = {}
    elif fault == "manifest":
        results[0]["capture_manifest_sha256"] = "changed"
    elif fault == "short":
        row["samples_ms"].pop()
    else:
        row["samples_ms"][0] = {
            "zero": 0,
            "negative": -1,
            "nan": float("nan"),
            "inf": float("inf"),
            "bool": True,
        }[fault]
    with pytest.raises(ValueError):
        adapter.aggregate_arm(results, captures, settings, "det", "a" * 40)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "context",
        "ranks",
        "missing_case",
        "missing_observation",
        "duplicate_observation",
        "failed_case",
        "failed_replay",
        "replays",
        "contention",
        "warn_only",
        "outputs",
        "gradients",
        "missing_check",
        "failed_check",
        "perturbations",
        "check_signature",
    ],
)
def test_replay_accuracy_and_controls_are_required_for_every_rank(adapter, fault):
    captures = capture_records()
    evidence = replay_records(captures)
    case = evidence["cases"][1]
    if fault == "context":
        evidence["context"]["revision"] = "changed"
    elif fault == "ranks":
        evidence["ranks_present"].pop()
    elif fault == "missing_case":
        evidence["cases"].pop()
    elif fault == "missing_observation":
        case["observations"].pop()
    elif fault == "duplicate_observation":
        case["observations"][1] = case["observations"][0]
    elif fault == "failed_case":
        case["status"] = "verified_nondeterministic"
    elif fault == "failed_replay":
        case["observations"][1]["status"] = "not_verified"
    elif fault in ("replays", "contention", "warn_only"):
        case["observations"][1]["protocol"][fault] = {
            "replays": 1,
            "contention": False,
            "warn_only": True,
        }[fault]
    elif fault == "outputs":
        case["observations"][1]["compared_outputs"] = 0
    elif fault == "gradients":
        case["observations"][1]["compared_gradients"] = 0
    elif fault == "missing_check":
        case["checks"].pop()
    elif fault == "failed_check":
        case["checks"][-1]["status"] = "failed"
    elif fault == "perturbations":
        case["checks"][-1]["detected_perturbations"] = 1
    elif fault == "check_signature":
        case["checks"][-1]["signature"]["implementation"] = "different"
    if fault:
        with pytest.raises(ValueError):
            adapter.validate_evidence(captures, [0, 1], evidence)
    else:
        assert len(adapter.validate_evidence(captures, [0, 1], evidence)) == 2


def test_policy_changes_are_explicit_and_capture_is_immutable(adapter):
    captures = capture_records()
    before = copy.deepcopy(captures)
    signature = captures[0]["events"][1]["signature"]
    collective = signature["configuration"]["collective"]
    for mode in ("default", "det"):
        env = adapter.policy_environment(
            {
                "NCCL_UNCAPTURED": "bad",
                "TRITON_AUTOTUNE_BLOCK_N": "bad",
                "CUDA_VISIBLE_DEVICES": "1,3",
            },
            captures[0]["context"],
            collective,
            mode,
        )
        assert env["CUDA_DEVICE_MAX_CONNECTIONS"] == "32" and env["NCCL_PROTO"] == "Simple"
        assert env["TRITON_CACHE_AUTOTUNING"] == "0" and env["TRITON_AUTOTUNE_BLOCK_M"] == "16"
        assert env["CUDA_VISIBLE_DEVICES"] == "1,3"
        assert "NCCL_UNCAPTURED" not in env and "TRITON_AUTOTUNE_BLOCK_N" not in env
        assert env.get("NCCL_ALGO") == ("Ring" if mode == "det" else None)
        results = rank_results(adapter, captures, measurement(), mode, "e" * 40)
        adapter.aggregate_arm(results, captures, measurement(), mode, "e" * 40)
    assert captures == before


@pytest.mark.parametrize("layout", ["offset", "transposed", "broadcast"])
def test_input_restoration_preserves_storage_and_bytes(adapter, layout):
    worker = load_module("run_collectives")
    value = torch.arange(24, dtype=torch.float32).reshape(4, 6)[1:]
    if layout == "transposed":
        value = value.t()
    if layout == "broadcast":
        value = value[:1].expand(4, 6)
    value.requires_grad_()
    cloned = worker.clone_tensor(torch, value)
    assert cloned.stride() == value.stride() and cloned.storage_offset() == value.storage_offset()
    assert cloned.requires_grad and torch.equal(cloned, value)
    assert cloned.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()


def test_head_helpers_do_not_import_head_production_or_pytest_policy(adapter, tmp_path):
    head, base = tmp_path / "head", tmp_path / "base"
    for root in (head, base):
        (root / "megatron").mkdir(parents=True)
        (root / "megatron/__init__.py").write_text(f"ORIGIN = {root.name!r}\n")
        (root / "tools").mkdir()
        (root / "tools/__init__.py").write_text(f"ORIGIN = {root.name!r}\n")
    code = f"from collective_case import install_head_helpers; from pathlib import Path; install_head_helpers(Path({str(head)!r})); import megatron, tools, sys; assert megatron.ORIGIN == 'base'; assert tools.ORIGIN == 'head'; assert 'tests.unit_tests.determinism.kernels' not in sys.modules"
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=base,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(SCRIPTS), str(base)])},
        check=True,
        timeout=30,
    )


@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_measure_excludes_setup_and_restores_mutated_inputs(adapter, monkeypatch, phase):
    worker = load_module("run_collectives")
    events = []

    class Event:
        def __init__(self, **kwargs):
            self.index = len(events)
            events.append("event")

        def record(self):
            events.append("start" if self.index == 0 else "end")

        def synchronize(self):
            events.append("event_sync")

        def elapsed_time(self, other):
            return float(events.count("end"))

    proxy = SimpleNamespace(
        **{name: getattr(torch, name) for name in ("empty", "set_grad_enabled", "autograd")},
        cuda=SimpleNamespace(Event=Event, synchronize=lambda: events.append("sync")),
        distributed=SimpleNamespace(barrier=lambda: events.append("barrier")),
    )
    calls = []

    def function(value, group):
        calls.append(value.detach().clone())
        with torch.no_grad():
            value.add_(7)
        events.append("operator")
        return value * 2

    local = torch.ones(2, requires_grad=True)
    samples = worker.measure(
        proxy,
        function,
        local,
        torch.ones(2),
        None,
        phase=phase,
        grad_enabled=True,
        warmup=2,
        steps=3,
    )
    assert samples == [3, 4, 5] and len(calls) == 5
    assert all(torch.equal(value, local) for value in calls)
    for i, event in enumerate(events):
        if event == "start":
            assert events[i - 2 : i] == ["barrier", "sync"]
    assert events.count("barrier") == 5


@pytest.mark.parametrize("fault", [None, "missing_rank", "changed_evidence", "changed_source"])
def test_parent_launches_paired_arms_and_retains_failures(adapter, monkeypatch, tmp_path, fault):
    # This mocked parent is independent of pytest's own distributed launcher.
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    driver = load_module("benchmark_collectives")
    captures = capture_records()
    capture_root = tmp_path / "capture"
    for rank in range(2):
        path = capture_root / f"rank-{rank}/manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(captures[rank]))
    evidence_path = tmp_path / "coverage.json"
    evidence_path.write_text(json.dumps(replay_records(captures)))
    monkeypatch.setitem(
        sys.modules,
        "tools.determinism.collective_capture",
        SimpleNamespace(load_captures=lambda *a, **k: captures),
    )
    monkeypatch.setattr(driver, "install_head_helpers", lambda root: None)
    monkeypatch.setattr(driver, "_machine", lambda: {"gpus": "CPU-fixture"})
    root = SCRIPTS.parents[3]
    base = tmp_path / "base"
    base.mkdir()
    monkeypatch.chdir(root)
    mutated = False

    def source(path):
        return {
            "revision": "a" * 40 if path == root else "e" * 40,
            "dirty": mutated,
            "checkout": str(path),
        }

    monkeypatch.setattr(driver, "_source", source)
    calls = []

    def launch(command, *, cwd, env, **kwargs):
        nonlocal mutated
        path = Path(command[-1])
        request = json.loads(path.read_text())
        calls.append((cwd, request["mode"]))
        assert env["PYTHONPATH"] == str(cwd)
        assert env.get("NCCL_ALGO") == ("Ring" if request["mode"] == "det" else None)
        results = rank_results(
            adapter,
            captures,
            request["measurement"],
            request["mode"],
            request["source"]["revision"],
        )
        for rank, result in enumerate(results):
            if fault == "missing_rank" and rank == 1:
                continue
            (path.parent / f"rank-{rank}.json").write_text(json.dumps(result))
        if fault == "changed_source":
            mutated = True
        if fault == "changed_evidence":
            evidence_path.write_text("{}")

    monkeypatch.setattr(driver.subprocess, "run", launch)
    output = tmp_path / "timings"
    code = driver.main(
        [
            "--capture",
            str(capture_root),
            "--evidence",
            str(evidence_path),
            "--output",
            str(output),
            "--base-checkout",
            str(base),
            "--warmup",
            "2",
            "--steps",
            "3",
        ]
    )
    report = json.loads((output / "benchmark.json").read_text())
    if fault:
        assert code == 1 and report["status"] == "error"
        assert report["runs"][0]["status"] == "incomplete"
    else:
        assert code == 0 and report["status"] == "reported"
        first = [(base, "default"), (base, "det"), (root, "default"), (root, "det")]
        assert calls == first + first[::-1] + first
        assert len(report["comparisons"]) == 2
        assert len(report["runs"]) == 12
        assert all(len(run["rank_files"]) == 2 for run in report["runs"])


@pytest.mark.parametrize(
    "variable,value",
    [
        ("TORCHELASTIC_RUN_ID", "single-rank-torchrun"),
        ("RANK", "1"),
        ("WORLD_SIZE", "2"),
        ("LOCAL_WORLD_SIZE", "2"),
    ],
)
def test_parent_rejects_nested_distributed_launch(adapter, monkeypatch, tmp_path, variable, value):
    for key in ("TORCHELASTIC_RUN_ID", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(variable, value)
    driver = load_module("benchmark_collectives")
    with pytest.raises(SystemExit) as stopped:
        driver.main(
            ["--capture", "unused", "--evidence", "unused", "--output", str(tmp_path / "out")]
        )
    assert stopped.value.code == 2
    assert not (tmp_path / "out").exists()


def test_startup_package_does_not_select_head_production(adapter, tmp_path):
    base = tmp_path / "base"
    (base / "megatron").mkdir(parents=True)
    (base / "megatron/__init__.py").write_text("ORIGIN = 'base'\n")
    root = SCRIPTS.parents[3]
    if not (root / "megatron/determinism/__init__.py").exists():
        pytest.skip("Integration control requires the companion early startup API (#7419)")
    code = f"""
from pathlib import Path
import sys, torch
from collective_case import install_head_helpers, configure_policy
root = Path({str(root)!r})
install_head_helpers(root)
from tools.determinism.capture_recipe import runtime_signature
runtime = runtime_signature(torch)
runtime['fill_uninitialized_memory'] = False
configure_policy(root, runtime, 'det')
assert torch.are_deterministic_algorithms_enabled()
assert not torch.utils.deterministic.fill_uninitialized_memory
assert 'megatron' not in sys.modules
assert 'tests.unit_tests.determinism.kernels' not in sys.modules
import megatron
assert megatron.ORIGIN == 'base'
"""
    environment = adapter.policy_environment(
        dict(os.environ),
        capture_records()[0]["context"],
        capture_records()[0]["events"][0]["signature"]["configuration"]["collective"],
        "det",
    )
    environment["PYTHONPATH"] = os.pathsep.join([str(SCRIPTS), str(base)])
    subprocess.run([sys.executable, "-c", code], cwd=base, env=environment, check=True, timeout=30)
