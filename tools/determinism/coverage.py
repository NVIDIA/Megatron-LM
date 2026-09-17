# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Aggregate measured replay evidence without importing GPU dependencies.

Only an explicit replay mismatch is evidence of nondeterminism. Pytest outcomes,
source registration, and exemptions are not numerical observations.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import fnmatch
import json
import os
from pathlib import Path
from typing import Callable, Iterator

from tools.determinism.branch_coverage import branch_report
from tools.determinism.parallelism import parallelism_report

SCHEMA_VERSION = 1
DETERMINISTIC = "verified_deterministic"
NONDETERMINISTIC = "verified_nondeterministic"
UNVERIFIED = "not_verified"
STATUSES = (DETERMINISTIC, NONDETERMINISTIC, UNVERIFIED)
_OBSERVER: contextvars.ContextVar[Callable[[dict], None] | None] = contextvars.ContextVar(
    "determinism_observer", default=None
)
_CONFIGURATION: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "determinism_replay_configuration", default=None
)


class ReplayMismatch(AssertionError):
    """Identical-input replay produced different tensors or tensor contents."""


def is_recording() -> bool:
    """Return whether a caller is collecting replay observations."""
    return _OBSERVER.get() is not None


def triton_signature() -> dict:
    """Record cache policy and every explicit SSM autotuning block override."""
    keys = {"TRITON_CACHE_AUTOTUNING", "TRITON_CACHE_DIR"}
    keys.update(key for key in os.environ if key.startswith("TRITON_AUTOTUNE_BLOCK_"))
    return {key: os.environ.get(key) for key in sorted(keys)}


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
        "triton": triton_signature(),
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
def replay_configuration(configuration: dict) -> Iterator[None]:
    """Attach runtime configuration observed by the replay adapter, not its plan."""
    token = _CONFIGURATION.set({**(_CONFIGURATION.get() or {}), **configuration})
    try:
        yield
    finally:
        _CONFIGURATION.reset(token)


@contextlib.contextmanager
def observe_replay(signature: dict, protocol: dict) -> Iterator[dict]:
    """Record completed comparisons; preserve the original exception on failure.

    The caller fills ``compared_outputs`` and ``compared_gradients`` after replay.
    An unrelated assertion, unsupported operation, or infrastructure error remains
    unverified, even if pytest marks it as an expected failure.
    """
    signature = {**signature, **(_CONFIGURATION.get() or {})}
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
    scope = first.get("evidence_scope", "kernel")
    if scope not in ("kernel", "model"):
        raise ValueError("Unknown replay evidence scope")
    context = first["context"]
    world_size = context["world_size"]
    if not isinstance(world_size, int) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    by_rank = {}
    for shard in shards:
        if shard.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported evidence schema")
        if shard.get("evidence_scope", "kernel") != scope:
            raise ValueError("Cannot combine kernel and model replay evidence")
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
        plan = present[0].get("parallelism_plan")
        if any(case.get("parallelism_plan") != plan for case in present):
            raise ValueError(f"Different parallelism plans for {case_id}")
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
                **({"parallelism_plan": plan} if plan is not None else {}),
            }
        )
    counts = {status: sum(case["status"] == status for case in cases) for status in STATUSES}
    total = len(cases)
    tagged = {case["op_id"] for case in cases}
    inventory = first.get("inventory", {})
    if any(shard.get("inventory", {}) != inventory for shard in shards):
        raise ValueError("Kernel inventories differ")
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "determinism_coverage" if scope == "kernel" else "model_determinism_replay",
        "run_id": first["run_id"],
        "context": context,
        "ranks_present": sorted(by_rank),
        "scope": (
            f"Declared selected {scope} replay cases; same-process outputs and gradients, "
            "not full training state or independent-run reproducibility"
        ),
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
    branches = branch_report(
        shards,
        cases,
        not context.get("dirty", True)
        and (expected_revision is None or context["revision"] == expected_revision),
    )
    if branches is not None:
        report["branches"] = branches
    if scope == "model":
        report["parallelism"] = parallelism_report(cases)
    return report


def markdown_report(report: dict) -> str:
    """Render a compact report, including visible unannotated inventory debt."""
    counts = report["counts"]
    lines = [
        (
            "# Model replay evidence"
            if report["kind"] == "model_determinism_replay"
            else "# Determinism coverage"
        ),
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
    lines.extend(["", "| Case | Operation/model | Status | Reason |", "| --- | --- | --- | --- |"])
    for case in report["cases"]:
        name = case["case_id"].replace("|", "\\|").replace("\n", " ")
        reasons = "; ".join(case["reasons"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{name}` | `{case['op_id']}` | {case['status']} | {reasons} |")
    gaps = report["inventory_without_declared_cases"]
    branches = report.get("branches")
    if branches is not None:
        lines.extend(["", "## Python branch execution", ""])
        if branches["counts"] is not None:
            branch_counts = branches["counts"]
            lines.append(
                f"{branch_counts['passing_replay']}/{branch_counts['total']} source branches "
                "were exercised by cases with passing replay on all ranks. "
                f"{branch_counts['never_observed']} were never observed in declared test calls. "
                "This is execution coverage, not branch-level numerical correctness."
            )
        if not branches["complete"]:
            lines.append("Branch measurement is incomplete: " + "; ".join(branches["reasons"]))
        lines.append("The JSON retains every branch, its source hash, supporting cases and ranks.")
    if "parallelism" in report:
        view = report["parallelism"]
        lines.extend(["", "## Parallelism interactions", "", view["scope"], ""])
        lines.append(
            f"{view['counts'][DETERMINISTIC]}/{view['counts']['total']} declared value-pairs "
            f"have matching passing replay; {len(view['unplanned_cases'])} cases lack a plan."
        )
        lines.extend(["", "| Model | Axis values | Status |", "| --- | --- | --- |"])
        for pair in view["pairs"]:
            values = ", ".join(f"{axis}={size}" for axis, size in pair["values"].items())
            lines.append(f"| `{pair['model_id']}` | {values} | {pair['status']} |")
    if report["kind"] == "model_determinism_replay":
        return "\n".join(lines) + "\n"
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
    parser.add_argument(
        "--require-verified",
        action="store_true",
        help="Fail unless at least one case has passing comparisons on every rank",
    )
    parser.add_argument(
        "--require-branches",
        action="store_true",
        help="Require complete Python branch measurement with branches exercised by passing replay",
    )
    parser.add_argument(
        "--require-parallelism",
        action="store_true",
        help="Require at least one model case with matching runtime parallelism and passing replay",
    )
    parser.add_argument(
        "--require-case",
        action="append",
        default=[],
        help="Require all cases matching this node-ID glob to be verified deterministic; missing matches fail",
    )
    args = parser.parse_args(argv)
    report = aggregate(
        [json.loads(path.read_text()) for path in sorted(args.shards.glob("rank-*.json"))],
        args.revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".md").write_text(markdown_report(report))
    print(markdown_report(report))
    if args.require_branches:
        branches = report.get("branches", {})
        if not branches.get("complete") or not branches["counts"]["passing_replay"]:
            print("No complete branch measurement associated with passing replay")
            return 1
    if args.require_parallelism and not any(
        row["status"] == DETERMINISTIC for row in report.get("parallelism", {}).get("rows", [])
    ):
        print("No passing model replay with runtime parallelism matching its plan")
        return 1
    if args.require_verified and not report["counts"][DETERMINISTIC]:
        print("No case completed passing replay comparisons on every required rank")
        return 1
    for pattern in args.require_case:
        matches = [
            case for case in report["cases"] if fnmatch.fnmatchcase(case["case_id"], pattern)
        ]
        if not matches or any(case["status"] != DETERMINISTIC for case in matches):
            print(f"Required replay cases are absent or unverified: {pattern}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
