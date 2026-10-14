# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Portable collective contracts using explicitly synthetic CPU-only records."""

import copy
import hashlib
import json
import shutil
import struct
import subprocess
import sys

import pytest

from tests.unit_tests.determinism_reporting.test_collective_performance import (
    capture_records,
    rank_results,
    replay_records,
)
from tests.unit_tests.determinism_reporting.test_paired_performance import (
    SCRIPTS,
    benchmark,
    load_module,
)

REVISION = "a" * 40


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def consumer(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setitem(sys.modules, "benchmark", benchmark)
    monkeypatch.setitem(sys.modules, "author_evidence", load_module("author_evidence"))
    monkeypatch.setitem(sys.modules, "baseline", load_module("baseline"))
    monkeypatch.setitem(sys.modules, "collective_case", load_module("collective_case"))
    module = load_module("collective_baseline")
    monkeypatch.setitem(sys.modules, "collective_baseline", module)
    return module


def artifacts(consumer, folder, *, has_base=True, limit=None, platform="dgx_h100"):
    """Build separate manifests, blobs, worker requests, rank files and reports."""
    captures = capture_records()
    capture = folder / "capture"
    coverage = folder / "coverage.json"
    timing = folder / "timing/benchmark.json"
    for rank, report in enumerate(captures):
        name, capability = {"dgx_h100": ("H100", [9, 0]), "dgx_gb200": ("GB200", [10, 0])}[platform]
        report["context"].update(gpu="Synthetic " + name, capability=capability)
        report.update(
            schema_version=1,
            kind="collective_recipe_capture",
            complete=True,
            truncated=False,
            capture_issues=[],
            context_after=copy.deepcopy(report["context"]),
        )
        directory = capture / f"rank-{rank}"
        directory.mkdir(parents=True)
        hashes = []
        for kind in range(2):
            raw = struct.pack("<6f", *[rank + kind + 1.0] * 6)
            digest = hashlib.sha256(raw).hexdigest()
            (directory / (digest + ".bin")).write_bytes(raw)
            hashes.append(digest)
        report["bytes_written"] = sum(p.stat().st_size for p in directory.iterdir())
        for event in report["events"]:
            signature = event["signature"]
            collective = signature["configuration"]["collective"]
            collective["capture_schema"] = 1
            collective["input"]["sha256"] = hashes[0]
            collective["group_options"]["config"]["blocking"] = -(2**31)
            collective["nccl_environment"]["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "0"
            if "gradient" in collective:
                collective["gradient"]["sha256"] = hashes[1]
        write_json(directory / "manifest.json", report)
    evidence = replay_records(captures)
    write_json(coverage, evidence)
    settings = {
        "pairs": 3,
        "warmup": 2,
        "steps": 3,
        "world_size": 2,
        "event_indices": [0, 1],
        "max_bytes": 1024,
        "timing": "cuda_event_ms",
        "aggregation": "per_sample_group_max",
        "communicator_initialization": "group_barrier_before_operator_warmup",
        "manifest_sha256": [
            consumer._digest((capture / f"rank-{r}/manifest.json").read_bytes()) for r in range(2)
        ],
        "tooling": {"synthetic-fixture.py": "f" * 64},
    }
    sources = {"head": {"revision": REVISION, "dirty": False, "checkout": "/old/head"}}
    if has_base:
        sources["base"] = {"revision": "b" * 40, "dirty": False, "checkout": "/old/base"}
    report = {
        "schema_version": 1,
        "kind": "determinism_collective_performance",
        "status": "pass" if limit else "reported",
        "machine": {"fixture": "CPU only; not GPU evidence"},
        "sources": sources,
        "measurement": settings,
        "capture": {
            "path": "/old/capture",
            "context": captures[0]["context"],
            "recipe_id": captures[0]["recipe_id"],
        },
        "replay_evidence": {
            "path": "/old/coverage.json",
            "sha256": consumer._digest(coverage.read_bytes()),
            "run_id": evidence["run_id"],
            "events": consumer.collective_case.validate_evidence(captures, [0, 1], evidence),
        },
        "runs": [],
    }
    arms = [(label, mode) for label in sorted(sources) for mode in ("default", "det")]
    for pair in range(3):
        for label, mode in (arms if pair % 2 == 0 else arms[::-1]):
            directory = timing.parent / f"pair-{pair}/{label}-{mode}"
            write_json(
                directory / "request.json",
                {
                    "head_checkout": sources["head"]["checkout"],
                    "capture": "/old/capture",
                    "source": sources[label],
                    "mode": mode,
                    "measurement": settings,
                },
            )
            (directory / "launcher.log").write_text(
                "Synthetic CPU record; never execute its command\n"
            )
            results = rank_results(
                consumer.collective_case, captures, settings, mode, sources[label]["revision"]
            )
            for rank, result in enumerate(results):
                for row in result["rows"].values():
                    factor = (1.1 if mode == "det" else 1) * (1.02 if label == "head" else 1)
                    row["samples_ms"] = [value * factor for value in row["samples_ms"]]
                write_json(directory / f"rank-{rank}.json", result)
            report["runs"].append(
                {
                    "pair": pair,
                    "revision_label": label,
                    "mode": mode,
                    "status": "complete",
                    "log_directory": f"/old/pair-{pair}/{label}-{mode}",
                    "command": ["do-not-execute", "/old/worker.py"],
                    "rank_files": {
                        f"rank-{rank}.json": consumer._digest(
                            (directory / f"rank-{rank}.json").read_bytes()
                        )
                        for rank in range(2)
                    },
                    "rows": consumer.collective_case.aggregate_arm(
                        results, captures, settings, mode, sources[label]["revision"]
                    ),
                }
            )
    report["comparisons"] = consumer.collective_case.comparisons(
        report["runs"], settings, has_base, limit, None
    )
    write_json(timing, report)
    return capture, coverage, timing


@pytest.mark.parametrize("has_base,limit", [(True, None), (False, None), (True, 1.5)])
def test_publish_transport_and_verify_without_torch_or_original_paths(
    consumer, tmp_path, has_base, limit
):
    paths = artifacts(consumer, tmp_path / "input", has_base=has_base, limit=limit)
    result = consumer.publish(*paths, tmp_path / "store", REVISION, "CPU-fixture")
    assert result["status"] == ("passed" if limit else "not_gated")
    assert result["rows"] == 2 and result["events"] == 2 and result["ranks"] == 2
    archive = shutil.make_archive(str(tmp_path / "transport"), "zip", result["path"])
    relocated = tmp_path / "relocated"
    shutil.unpack_archive(archive, relocated)
    shutil.rmtree(tmp_path / "input")
    shutil.rmtree(tmp_path / "store")
    checked = consumer.baseline.verify(relocated, result["baseline_id"])
    assert checked["rank_samples"] == (144 if has_base else 72)
    command = [
        sys.executable,
        "-S",
        str(SCRIPTS / "baseline.py"),
        "verify",
        str(relocated),
        "--expected-id",
        result["baseline_id"],
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(process.stdout) == checked


def test_repeated_publication_does_not_replace_immutable_content(consumer, tmp_path):
    paths = artifacts(consumer, tmp_path / "input")
    first = consumer.publish(*paths, tmp_path / "store", REVISION, "CPU-fixture")
    manifest = tmp_path / "store" / first["baseline_id"] / "baseline.json"
    before = manifest.stat().st_mtime_ns
    second = consumer.publish(*paths, tmp_path / "store", REVISION, "CPU-fixture")
    assert first["created"] and not second["created"]
    assert first["baseline_id"] == second["baseline_id"] and manifest.stat().st_mtime_ns == before


@pytest.mark.parametrize(
    "fault",
    [
        "missing_arm",
        "duplicate_arm",
        "order",
        "group_max",
        "ratio",
        "interval",
        "gate",
        "request",
        "samples",
        "uuid",
        "communicator",
        "raw_hash",
        "missing_log",
        "extra_attempt",
        "diagnostic",
        "missing_blob",
        "blob_bytes",
        "extra_blob",
        "capture_complete",
        "capture_rank",
        "capture_context",
        "subset",
        "reference",
        "replay",
        "sensitivity",
        "evidence_hash",
        "wrong_source",
    ],
)
def test_invalid_or_ambiguous_records_cannot_be_published(consumer, tmp_path, fault):
    capture, coverage, timing = artifacts(consumer, tmp_path / "input")
    report = json.loads(timing.read_text())
    arm = timing.parent / "pair-0/base-default"
    if fault == "missing_arm":
        report["runs"].pop()
    elif fault == "duplicate_arm":
        report["runs"][-1] = report["runs"][0]
    elif fault == "order":
        report["runs"].reverse()
    elif fault == "group_max":
        report["runs"][0]["rows"]["event-0/ranks-0-1"]["median_ms"] += 1
    elif fault == "ratio":
        report["comparisons"]["event-0/ranks-0-1"]["head_overhead"]["median_ratio"] += 1
    elif fault == "interval":
        report["comparisons"]["event-0/ranks-0-1"]["head_overhead"][
            "bootstrap_95_percent_interval"
        ] = [1, 1]
    elif fault == "gate":
        report["status"] = "pass"
    elif fault == "request":
        write_json(arm / "request.json", {})
    elif fault == "missing_log":
        (arm / "launcher.log").unlink()
    elif fault == "extra_attempt":
        (timing.parent / "retry.json").write_text("{}")
    elif fault == "subset":
        report["measurement"]["event_indices"] = [0]
    elif fault == "wrong_source":
        report["sources"]["head"]["revision"] = "c" * 40
    elif fault in ("samples", "uuid", "communicator", "raw_hash", "diagnostic"):
        path = arm / "rank-0.json"
        rank = json.loads(path.read_text())
        if fault in ("samples", "raw_hash"):
            rank["rows"]["0"]["samples_ms"][0] += 1
        elif fault == "uuid":
            rank["rows"]["0"]["actual_signature"]["configuration"]["collective"][
                "device_uuid"
            ] = "different"
        elif fault == "diagnostic":
            rank["diagnostics"] = {"status": "observed"}
        else:
            next(iter(rank["communicators"].values()))["after_initialize"]["config"]["max_ctas"] = 5
        write_json(path, rank)
        if fault != "raw_hash":
            report["runs"][0]["rank_files"]["rank-0.json"] = consumer._digest(path.read_bytes())
    elif fault in ("missing_blob", "blob_bytes", "extra_blob"):
        path = next(capture.rglob("*.bin"))
        if fault == "missing_blob":
            path.unlink()
        elif fault == "blob_bytes":
            path.write_bytes(b"changed")
        else:
            (path.parent / "unexpected.bin").write_bytes(b"x")
    elif fault.startswith("capture_"):
        path = capture / "rank-0/manifest.json"
        value = json.loads(path.read_text())
        if fault == "capture_complete":
            value["complete"] = False
        elif fault == "capture_rank":
            value["rank"] = 1
        else:
            value["context_after"]["revision"] = "c" * 40
        write_json(path, value)
        report["measurement"]["manifest_sha256"][0] = consumer._digest(path.read_bytes())
    else:
        evidence = json.loads(coverage.read_text())
        if fault == "replay":
            evidence["cases"][0]["observations"][0]["protocol"]["replays"] = 1
        else:
            evidence["cases"][0]["checks"][0]["status"] = "failed"
        if fault == "sensitivity":
            evidence["cases"][0]["check_status"]["sensitivity"] = "failed"
        write_json(coverage, evidence)
        if fault != "evidence_hash":
            report["replay_evidence"]["sha256"] = consumer._digest(coverage.read_bytes())
    write_json(timing, report)
    with pytest.raises((ValueError, KeyError)):
        consumer.publish(capture, coverage, timing, tmp_path / "store", REVISION, "CPU-fixture")
    assert not (tmp_path / "store").exists()


@pytest.mark.parametrize(
    "fault", ["changed", "missing", "extra", "symlink", "wrong_id", "rehashed_rank"]
)
def test_transport_integrity_and_rehashed_semantic_corruption(consumer, tmp_path, fault):
    paths = artifacts(consumer, tmp_path / "input")
    result = consumer.publish(*paths, tmp_path / "store", REVISION, "CPU-fixture")
    directory = tmp_path / "store" / result["baseline_id"]
    path = directory / "timing/pair-0/base-default/rank-0.json"
    expected = result["baseline_id"]
    if fault == "changed":
        path.write_text("{}")
    elif fault == "missing":
        path.unlink()
    elif fault == "extra":
        (directory / "extra.json").write_text("{}")
    elif fault == "symlink":
        path.unlink()
        path.symlink_to(paths[1])
    elif fault == "wrong_id":
        expected = "0" * 64
    else:
        value = json.loads(path.read_text())
        value["rows"]["0"]["samples_ms"][0] += 3
        write_json(path, value)
        manifest = json.loads((directory / "baseline.json").read_text())
        manifest["files"][path.relative_to(directory).as_posix()] = {
            "sha256": consumer._digest(path.read_bytes()),
            "bytes": path.stat().st_size,
        }
        write_json(directory / "baseline.json", manifest)
        expected = None
    with pytest.raises(ValueError):
        consumer.baseline.verify(directory, expected)
