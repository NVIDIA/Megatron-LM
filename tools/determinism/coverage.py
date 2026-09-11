# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Aggregate measured replay evidence without importing GPU dependencies.

Only an explicit replay mismatch is evidence of nondeterminism. Pytest outcomes,
source registration, and exemptions are not numerical observations.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import json
from pathlib import Path
from typing import Callable, Iterator

SCHEMA_VERSION = 1
DETERMINISTIC = "verified_deterministic"
NONDETERMINISTIC = "verified_nondeterministic"
UNVERIFIED = "not_verified"
STATUSES = (DETERMINISTIC, NONDETERMINISTIC, UNVERIFIED)
_OBSERVER: contextvars.ContextVar[Callable[[dict], None] | None] = contextvars.ContextVar(
    "determinism_observer", default=None
)


class ReplayMismatch(AssertionError):
    """Identical-input replay produced different tensors or tensor contents."""


def is_recording() -> bool:
    """Return whether a caller is collecting replay observations."""
    return _OBSERVER.get() is not None


def runtime_signature(torch) -> dict:
    """Record dispatch-affecting settings at the call, after test setup."""
    return {
        "autocast": torch.is_autocast_enabled(),
        "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


@contextlib.contextmanager
def collect_observations(sink: Callable[[dict], None]) -> Iterator[None]:
    """Route replay observations to a test-local sink."""
    token = _OBSERVER.set(sink)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


@contextlib.contextmanager
def observe_replay(signature: dict, protocol: dict) -> Iterator[dict]:
    """Record completed comparisons; preserve the original exception on failure.

    The caller fills ``compared_outputs`` and ``compared_gradients`` after replay.
    An unrelated assertion, unsupported operation, or infrastructure error remains
    unverified, even if pytest marks it as an expected failure.
    """
    observation: dict = {"signature": signature, "protocol": protocol, "status": UNVERIFIED}
    try:
        yield observation
    except ReplayMismatch as error:
        observation.update(status=NONDETERMINISTIC, reason=str(error))
        raise
    except BaseException as error:
        observation["reason"] = f"{type(error).__name__}: {error}"
        raise
    else:
        if (
            observation.get("compared_outputs", 0) > 0
            and protocol.get("replays", 0) >= 2
            and (
                signature.get("phase") != "forward_backward"
                or observation.get("compared_gradients", 0) > 0
            )
        ):
            observation["status"] = DETERMINISTIC
        else:
            observation["reason"] = "Replay count or output/gradient comparisons were insufficient"
    finally:
        observer = _OBSERVER.get()
        if observer is not None:
            observer(observation)


def case_status(observations: list[dict], test_complete: bool) -> str:
    """Classify a case from numerical observations and test completion."""
    statuses = [item["status"] for item in observations]
    if NONDETERMINISTIC in statuses:
        return NONDETERMINISTIC
    if test_complete and statuses and all(status == DETERMINISTIC for status in statuses):
        return DETERMINISTIC
    return UNVERIFIED


def aggregate(shards: list[dict], expected_revision: str | None = None) -> dict:
    """Require matching provenance and complete rank coverage before assigning D.

    The denominator is the union of declared, selected test cases. Untagged kernel
    families are reported separately, never silently represented by passing tests
    from the same source file. Missing ranks, interrupted runs, and dirty source
    trees cannot produce a verified-deterministic case.
    """
    if not shards:
        raise ValueError("No evidence shards supplied")
    first = shards[0]
    context = first["context"]
    world_size = context["world_size"]
    if not isinstance(world_size, int) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    by_rank = {}
    for shard in shards:
        if shard.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported evidence schema")
        if shard["context"] != context or shard["run_id"] != first["run_id"]:
            raise ValueError("Cannot combine different runs, source revisions, or environments")
        rank = shard["rank"]
        if not isinstance(rank, int) or not 0 <= rank < world_size or rank in by_rank:
            raise ValueError(f"Duplicate or invalid rank: {rank}")
        by_rank[rank] = shard
    case_ids = sorted({case_id for shard in shards for case_id in shard["cases"]})
    cases = []
    for case_id in case_ids:
        present = [shard["cases"][case_id] for shard in shards if case_id in shard["cases"]]
        declaration = present[0]["declaration"]
        if any(case["declaration"] != declaration for case in present):
            raise ValueError(f"Different declarations for {case_id}")
        observations = [
            {**observation, "rank": rank}
            for rank, shard in sorted(by_rank.items())
            for observation in shard["cases"].get(case_id, {}).get("observations", [])
        ]
        for observation in observations:
            if observation["status"] not in STATUSES:
                raise ValueError(f"Unknown observation status in {case_id}")
        complete = all(
            rank in by_rank
            and by_rank[rank]["complete"]
            and by_rank[rank]["cases"].get(case_id, {}).get("test_complete", False)
            and by_rank[rank]["cases"][case_id].get("observations")
            for rank in range(world_size)
        )
        stale = expected_revision is not None and context["revision"] != expected_revision
        dirty = context.get("dirty", True)
        status = case_status(observations, complete)
        reasons = sorted({case.get("reason", "") for case in present} - {""})
        if stale or dirty:
            # A mismatch remains in the raw observations, but is not evidence
            # about the requested clean revision.
            status = UNVERIFIED
            reasons.append("Source revision is stale or has uncommitted changes")
        elif not complete and status != NONDETERMINISTIC:
            reasons.append("Missing replay, failed test phase, incomplete session, or missing rank")
        cases.append(
            {
                "case_id": case_id,
                **declaration,
                "status": status,
                "reasons": reasons,
                "observations": observations,
            }
        )
    counts = {status: sum(case["status"] == status for case in cases) for status in STATUSES}
    total = len(cases)
    tagged = {case["op_id"] for case in cases}
    inventory = first.get("inventory", {})
    if any(shard.get("inventory", {}) != inventory for shard in shards):
        raise ValueError("Kernel inventories differ")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "determinism_coverage",
        "run_id": first["run_id"],
        "context": context,
        "ranks_present": sorted(by_rank),
        "scope": "Declared selected replay cases; outputs and gradients, not full training state",
        "counts": {"total": total, **counts},
        "deterministic_percent": 100 * counts[DETERMINISTIC] / total if total else None,
        "verified_percent": (
            100 * (counts[DETERMINISTIC] + counts[NONDETERMINISTIC]) / total if total else None
        ),
        "inventory_without_declared_cases": {
            name: detail for name, detail in inventory.items() if name not in tagged
        },
        "cases": cases,
    }


def markdown_report(report: dict) -> str:
    """Render a compact report, including visible unannotated inventory debt."""
    counts = report["counts"]
    lines = [
        "# Determinism coverage",
        "",
        report["scope"],
        "",
        f"Revision: `{report['context']['revision']}`. Ranks: {report['ranks_present']}.",
        "",
        "| Deterministic | Nondeterministic | Unverified | Total cases |",
        "| ---: | ---: | ---: | ---: |",
        f"| {counts[DETERMINISTIC]} | {counts[NONDETERMINISTIC]} | {counts[UNVERIFIED]} | {counts['total']} |",
        "",
    ]
    if counts["total"]:
        lines.append(
            f"Deterministic coverage: {report['deterministic_percent']:.1f}%. "
            f"Verification coverage: {report['verified_percent']:.1f}%."
        )
    else:
        lines.append("No declared cases: coverage percentages are unavailable.")
    lines.extend(["", "| Case | Operation | Status |", "| --- | --- | --- |"])
    for case in report["cases"]:
        name = case["case_id"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{name}` | `{case['op_id']}` | {case['status']} |")
    gaps = report["inventory_without_declared_cases"]
    lines.extend(
        [
            "",
            f"{len(gaps)} registered operation families have no declared cases in this selection.",
            "These families are outside the case percentage above; registration is not verification.",
            "",
        ]
    )
    for name, detail in sorted(gaps.items()):
        reason = detail.get("exempt_reason") or "No annotated case selected"
        lines.append(f"- `{name}`: {reason}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Combine per-rank JSON shards and write machine and human reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", help="Revision whose evidence is requested")
    args = parser.parse_args(argv)
    report = aggregate(
        [json.loads(path.read_text()) for path in sorted(args.shards.glob("rank-*.json"))],
        args.revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".md").write_text(markdown_report(report))
    print(markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
