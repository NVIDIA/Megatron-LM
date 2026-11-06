# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Portable captured-collective evidence, verified without Torch or a GPU.

Recheck recorded replay/accuracy evidence and recompute timings after transport.
The capture producer owns tensor reconstruction and numerical reference execution;
this consumer preserves and checks those records and the actual captured bytes.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import baseline
import collective_case

KIND = "determinism_collective_baseline"
MEASUREMENT_FIELDS = {
    "pairs",
    "warmup",
    "steps",
    "world_size",
    "event_indices",
    "max_bytes",
    "timing",
    "aggregation",
    "communicator_initialization",
    "manifest_sha256",
    "tooling",
}


def _tree(root: Path) -> dict[str, bytes]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected a regular artifact directory")
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Artifact symlinks are unsupported")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = baseline._bytes(path)
    return files


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tensor(descriptor: dict, files: dict, rank: int, limit: int) -> str:
    if set(descriptor) != {"shape", "stride", "dtype", "requires_grad", "storage_offset", "sha256"}:
        raise ValueError("Unsupported captured tensor descriptor")
    shape, stride = descriptor["shape"], descriptor["stride"]
    offset, dtype = descriptor["storage_offset"], descriptor["dtype"]
    if (
        not isinstance(shape, list)
        or not shape
        or not isinstance(stride, list)
        or len(shape) != len(stride)
        or any(type(n) is not int or n < 1 for n in shape)
        or any(type(n) is not int or n < 0 for n in stride)
        or type(offset) is not int
        or offset < 0
        or dtype not in ("torch.float32", "torch.bfloat16")
        or type(descriptor["requires_grad"]) is not bool
        or not isinstance(descriptor["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])
    ):
        raise ValueError("Invalid captured tensor metadata")
    itemsize = 4 if dtype == "torch.float32" else 2
    size = math.prod(shape) * itemsize
    extent = offset + 1 + sum((n - 1) * s for n, s in zip(shape, stride))
    if max(size, extent * itemsize) > limit:
        raise ValueError("Captured tensor exceeds its declared byte limit")
    name = f"rank-{rank}/{descriptor['sha256']}.bin"
    raw = files[name]
    if len(raw) != size or _digest(raw) != descriptor["sha256"]:
        raise ValueError("Captured blob differs from the timed/replayed input")
    return name


def _captures(files: dict[str, bytes], measurement: dict, revision: str) -> list[dict]:
    world = measurement["world_size"]
    manifests = [f"rank-{rank}/manifest.json" for rank in range(world)]
    captures = [baseline._json(files[name]) for name in manifests]
    if [_digest(files[name]) for name in manifests] != measurement["manifest_sha256"]:
        raise ValueError("Capture manifests differ from the measured captures")
    context = captures[0]["context"]
    count = len(captures[0]["events"])
    if (
        not count
        or measurement["event_indices"] != list(range(count))
        or context["revision"] != revision
        or context.get("dirty") is not False
        or type(context["world_size"]) is not int
        or context["world_size"] != world
    ):
        raise ValueError("A baseline requires every event of a clean matching capture")
    expected = set(manifests)
    for rank, capture in enumerate(captures):
        if (
            capture.get("schema_version") != 1
            or capture.get("kind") != "collective_recipe_capture"
            or type(capture.get("rank")) is not int
            or capture["rank"] != rank
            or capture["context"] != context
            or capture.get("context_after") != context
            or capture.get("complete") is not True
            or capture.get("truncated") is not False
            or capture.get("capture_issues") != []
            or len(capture["events"]) != count
            or not capture.get("recipe_id")
            or capture["recipe_id"] != captures[0]["recipe_id"]
        ):
            raise ValueError("Incomplete or inconsistent capture records")
        blobs = set()
        forwards = {}
        for event in capture["events"]:
            signature = event["signature"]
            collective = signature["configuration"]["collective"]
            members = collective["group_ranks"]
            fields = {
                "capture_schema",
                "case",
                "group_ranks",
                "group_rank",
                "size",
                "backend",
                "nccl_version",
                "nccl_environment",
                "device_uuid",
                "input",
                "grad_enabled",
                "warn_only",
                "group_options",
            }
            if signature["phase"] == "forward_backward":
                fields.update(("gradient", "backward_grad_enabled"))
            if (
                set(signature["configuration"]) != {"collective"}
                or set(collective) != fields
                or any(
                    key in signature
                    for key in (
                        "backward_runtime",
                        "backward_deterministic_algorithms",
                        "backward_collective",
                    )
                )
                or type(collective["grad_enabled"]) is not bool
                or signature["op_id"] != "tensor_parallel_mappings"
                or not signature["implementation"].startswith("mcore:")
                or signature["phase"] not in ("forward", "forward_backward")
                or signature["deterministic_algorithms"] is not True
                or type(signature["runtime"].get("fill_uninitialized_memory")) is not bool
                or collective.get("capture_schema") != 1
                or collective["backend"] != "nccl"
                or collective["warn_only"] is not False
                or not collective["device_uuid"]
                or not isinstance(members, list)
                or len(members) < 2
                or any(type(member) is not int or not 0 <= member < world for member in members)
                or members != sorted(set(members))
                or rank not in members
                or collective["group_rank"] != members.index(rank)
                or collective["size"] != len(members)
                or type(event["call_id"]) is not int
                or event["call_id"] < 0
            ):
                raise ValueError("Unsupported captured collective identity or policy")
            local = collective["input"]
            blobs.add(_tensor(local, files, rank, measurement["max_bytes"]))
            if signature["inputs"] != [
                {key: local[key] for key in ("shape", "stride", "dtype", "requires_grad")}
            ]:
                raise ValueError("Captured input descriptor differs from its signature")
            call = event["call_id"]
            if signature["phase"] == "forward":
                if call in forwards or "gradient" in collective:
                    raise ValueError("Duplicate or invalid forward event")
                forwards[call] = collective
            else:
                if (
                    call not in forwards
                    or collective.get("backward_grad_enabled") is not False
                    or collective.get("grad_enabled") is not True
                    or local["requires_grad"] is not True
                    or {
                        k: v
                        for k, v in collective.items()
                        if k not in ("gradient", "backward_grad_enabled")
                    }
                    != forwards[call]
                    or collective["gradient"]["requires_grad"] is not False
                    or collective["gradient"]["dtype"] != local["dtype"]
                ):
                    raise ValueError("Backward capture differs from its forward")
                blobs.add(_tensor(collective["gradient"], files, rank, measurement["max_bytes"]))
        size = sum(len(files[name]) for name in blobs)
        if size > measurement["max_bytes"] or capture["bytes_written"] != size:
            raise ValueError("Capture byte inventory differs or exceeds the declared limit")
        expected.update(blobs)
    if set(files) != expected:
        raise ValueError("Capture contains missing, duplicate or undeclared files")
    for index in range(count):
        events = [capture["events"][index] for capture in captures]
        if (
            len(
                {
                    (
                        event["call_id"],
                        event["signature"]["implementation"],
                        event["signature"]["phase"],
                    )
                    for event in events
                }
            )
            != 1
        ):
            raise ValueError("Captured rank schedules differ")
        for event in events:
            collective = event["signature"]["configuration"]["collective"]
            for rank in collective["group_ranks"]:
                peer = events[rank]["signature"]["configuration"]["collective"]
                if any(
                    peer[key] != collective[key]
                    for key in (
                        "case",
                        "group_ranks",
                        "size",
                        "backend",
                        "group_options",
                        "nccl_version",
                        "nccl_environment",
                    )
                ):
                    raise ValueError("Captured peers use different communicators")
    return captures


def snapshot(
    capture: Path, coverage: Path, benchmark: Path, revision: str, origin: str
) -> tuple[dict, dict]:
    """Reconstruct every rank, event/group statistic and evidence match from files."""
    if (
        not isinstance(origin, str)
        or not origin.strip()
        or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision)
    ):
        raise ValueError("Explicit source revision and origin are required")
    capture_files, timing_files = _tree(capture), _tree(benchmark.parent)
    if benchmark.name != "benchmark.json":
        raise ValueError("Expected the original benchmark.json report")
    coverage_raw = baseline._bytes(coverage)
    evidence = baseline._json(coverage_raw)
    report = baseline._json(timing_files["benchmark.json"])
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "determinism_collective_performance"
        or report.get("status") not in ("reported", "pass")
        or set(report)
        != {
            "schema_version",
            "kind",
            "status",
            "sources",
            "machine",
            "measurement",
            "capture",
            "replay_evidence",
            "runs",
            "comparisons",
        }
    ):
        raise ValueError("A baseline requires a complete supported collective timing report")
    measurement, sources = report["measurement"], report["sources"]
    if (
        set(measurement) != MEASUREMENT_FIELDS
        or any(
            type(measurement[key]) is not int or measurement[key] < 1
            for key in ("pairs", "warmup", "steps", "world_size", "max_bytes")
        )
        or measurement["pairs"] < 3
        or measurement["world_size"] < 2
        or measurement["timing"] != "cuda_event_ms"
        or measurement["aggregation"] != "per_sample_group_max"
        or measurement["communicator_initialization"] != "group_barrier_before_operator_warmup"
        or not measurement["tooling"]
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in measurement["tooling"].values()
        )
        or set(sources) not in ({"head"}, {"base", "head"})
        or sources["head"]["revision"] != revision
        or any(
            source.get("dirty") is not False
            or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source["revision"])
            for source in sources.values()
        )
    ):
        raise ValueError("Unsupported measurement protocol or dirty/mismatched sources")
    captures = _captures(capture_files, measurement, revision)
    matched = collective_case.validate_evidence(captures, measurement["event_indices"], evidence)
    if (
        report["capture"]["context"] != captures[0]["context"]
        or report["capture"]["recipe_id"] != captures[0]["recipe_id"]
        or report["replay_evidence"]["sha256"] != _digest(coverage_raw)
        or report["replay_evidence"]["run_id"] != evidence["run_id"]
        or report["replay_evidence"]["events"] != matched
    ):
        raise ValueError("Timing report differs from its original capture/replay evidence")
    arms = [(label, mode) for label in sorted(sources) for mode in ("default", "det")]
    order = [
        (pair, label, mode)
        for pair in range(measurement["pairs"])
        for label, mode in (arms if pair % 2 == 0 else arms[::-1])
    ]
    if [(run["pair"], run["revision_label"], run["mode"]) for run in report["runs"]] != order:
        raise ValueError("Timing arms are missing, duplicated or out of order")
    expected = {"benchmark.json"}
    if "benchmark.md" in timing_files:
        expected.add("benchmark.md")
    for run in report["runs"]:
        pair, label, mode = run["pair"], run["revision_label"], run["mode"]
        prefix = f"pair-{pair}/{label}-{mode}/"
        request_name = prefix + "request.json"
        request = baseline._json(timing_files[request_name])
        if request != {
            "head_checkout": sources["head"]["checkout"],
            "capture": report["capture"]["path"],
            "source": sources[label],
            "mode": mode,
            "measurement": measurement,
        }:
            raise ValueError("Recorded worker request differs from the reported arm")
        expected.update((request_name, prefix + "launcher.log"))
        rank_files = {
            f"rank-{rank}.json": _digest(timing_files[prefix + f"rank-{rank}.json"])
            for rank in range(measurement["world_size"])
        }
        expected.update(prefix + name for name in rank_files)
        results = [baseline._json(timing_files[prefix + name]) for name in rank_files]
        if (
            run.get("status") != "complete"
            or run["rank_files"] != rank_files
            or collective_case.aggregate_arm(
                results, captures, measurement, mode, sources[label]["revision"]
            )
            != run["rows"]
            or any(
                set(result)
                != {
                    "adapter",
                    "rank",
                    "mode",
                    "measurement",
                    "context",
                    "context_after",
                    "rows",
                    "tooling",
                    "communicators",
                    "capture_manifest_sha256",
                }
                for result in results
            )
        ):
            raise ValueError("Raw rank files differ from the complete measured arm")
    if set(timing_files) != expected:
        raise ValueError("Timing files are missing or contain undeclared attempts/diagnostics")
    has_base = "base" in sources
    rows = list(report["comparisons"].values())
    if not rows:
        raise ValueError("No collective comparisons")
    overhead = rows[0]["head_overhead"]["limit"]
    regression = rows[0]["det_regression"]["limit"] if has_base else None
    for limit in (overhead, regression):
        if limit is not None and (
            type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0
        ):
            raise ValueError("Invalid performance limit")
    comparisons = collective_case.comparisons(
        report["runs"], measurement, has_base, overhead, regression
    )
    if comparisons != report["comparisons"]:
        raise ValueError("Reported comparisons differ from the retained paired samples")
    statuses = [
        value["status"]
        for row in comparisons.values()
        for name, value in row.items()
        if name != "base_overhead" and value["status"] != "not_gated"
    ]
    status = (
        "pass"
        if statuses and all(value == "pass" for value in statuses)
        else "reported" if not statuses else "invalid"
    )
    if report["status"] != status:
        raise ValueError("Collective baseline has failed or inconclusive performance evidence")
    files = {
        "coverage.json": coverage_raw,
        **{"capture/" + name: raw for name, raw in capture_files.items()},
        **{"timing/" + name: raw for name, raw in timing_files.items()},
    }
    summary = {
        "status": "passed" if status == "pass" else "not_gated",
        "recipe_id": captures[0]["recipe_id"],
        "events": len(captures[0]["events"]),
        "ranks": len(captures),
        "rows": len(comparisons),
        "arms": len(report["runs"]),
        "rank_samples": len(report["runs"])
        * len(captures)
        * len(captures[0]["events"])
        * measurement["steps"],
    }
    manifest = {
        "schema_version": 1,
        "kind": KIND,
        "revision": revision,
        "origin": origin,
        "collective_evidence": summary,
        "files": {
            name: {"sha256": _digest(raw), "bytes": len(raw)} for name, raw in sorted(files.items())
        },
    }
    return manifest, files


def verify(directory: Path, expected_id: str | None = None) -> dict:
    """Verify relocated collective bytes and recompute evidence without executing code."""
    manifest, identity, count = baseline.verified_contents(directory, expected_id)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != KIND:
        raise ValueError("Unsupported collective baseline schema")
    rebuilt, _ = snapshot(
        directory / "capture",
        directory / "coverage.json",
        directory / "timing/benchmark.json",
        manifest["revision"],
        manifest["origin"],
    )
    if rebuilt != manifest:
        raise ValueError("Collective baseline summary differs from recomputed evidence")
    return {
        "baseline_id": identity,
        "revision": manifest["revision"],
        "origin": manifest["origin"],
        "files_verified": count,
        **rebuilt["collective_evidence"],
    }


def publish(
    capture: Path, coverage: Path, benchmark: Path, store: Path, revision: str, origin: str
) -> dict:
    """Publish an immutable complete capture and timing bundle, preserving unbudgeted status."""
    if any(store.resolve().is_relative_to(root.resolve()) for root in (capture, benchmark.parent)):
        raise ValueError("Baseline store must be outside the source artifacts")
    manifest, files = snapshot(capture, coverage, benchmark, revision, origin)
    return baseline.publish_snapshot(manifest, files, store, verify)
