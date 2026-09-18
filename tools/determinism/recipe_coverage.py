# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Match a recipe's observed operation signatures to scoped replay evidence.

This consumer has no GPU dependencies. A complete matching inventory means only
the observed operations have evidence; independent recipe and resume replay are
still required. Source, environment, implementation and signatures must match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DETERMINISTIC = "verified_deterministic"
NONDETERMINISTIC = "verified_nondeterministic"
UNVERIFIED = "not_verified"
STATUSES = (DETERMINISTIC, NONDETERMINISTIC, UNVERIFIED)


def signature_key(signature: dict) -> str:
    """Return the exact configuration key, independent of dictionary order."""
    required = {"op_id", "implementation", "inputs", "phase", "deterministic_algorithms", "runtime"}
    if not required <= signature.keys():
        raise ValueError(f"Signature lacks fields: {sorted(required - signature.keys())}")
    if not all(
        isinstance(signature[key], str) and signature[key] for key in ("op_id", "implementation")
    ):
        raise ValueError("Operation and implementation IDs must be nonempty strings")
    if signature["phase"] not in ("forward", "forward_backward"):
        raise ValueError("Unsupported verification phase")
    if not isinstance(signature["deterministic_algorithms"], bool):
        raise ValueError("deterministic_algorithms must be a boolean")
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _has_memory_fill_policy(signature: dict) -> bool:
    """Legacy or malformed policy records cannot establish a recipe match."""
    runtimes = [signature.get("runtime")]
    if "backward_runtime" in signature:
        runtimes.append(signature["backward_runtime"])
    return all(
        isinstance(runtime, dict) and type(runtime.get("fill_uninitialized_memory")) is bool
        for runtime in runtimes
    )


def _evidence_index(
    reports: list[dict], context: dict, observed_by_rank: dict[int, set[str]]
) -> tuple[dict, list[str]]:
    index: dict[str, list[dict]] = {}
    rejected = []
    expected = set(range(context["world_size"]))
    for report in reports:
        if report.get("schema_version") != 1 or report.get("kind") != "determinism_coverage":
            raise ValueError("Unsupported coverage report schema")
        if report["context"] != context or context.get("dirty", True):
            rejected.append(str(report.get("run_id", "unknown")))
            continue
        for case in report["cases"]:
            status = case["status"]
            if status not in STATUSES:
                raise ValueError(f"Unknown case status: {status}")
            if status == UNVERIFIED:
                continue
            candidates_by_protocol: dict[tuple[str, str, str], list[dict]] = {}
            assignments: dict[tuple[str, str], dict[int, set[str]]] = {}
            for observation in case["observations"]:
                # A failed aggregate case cannot lend passing observations to
                # other operations. For N, retain only the actual mismatches.
                if observation["status"] != status:
                    continue
                rank = observation["rank"]
                if type(rank) is not int or rank not in expected:
                    raise ValueError("Evidence contains an invalid rank")
                signature = observation["signature"]
                if not _has_memory_fill_policy(signature):
                    continue
                signatures = [signature]
                # A forward+backward replay also compared its forward outputs.
                if status == DETERMINISTIC and signature.get("phase") == "forward_backward":
                    signatures.append({**signature, "phase": "forward"})
                for candidate in signatures:
                    key = signature_key(candidate)
                    protocol_key = json.dumps(
                        observation["protocol"], sort_keys=True, allow_nan=False
                    )
                    group = candidates_by_protocol.setdefault(
                        (key, protocol_key, candidate["phase"]), []
                    )
                    group.append(observation)
                    assignment = assignments.setdefault((candidate["phase"], protocol_key), {})
                    assignment.setdefault(rank, set()).add(key)
            for (key, protocol_key, phase), observations in candidates_by_protocol.items():
                ranks = {observation["rank"] for observation in observations}
                rank_assignment = None
                if status == DETERMINISTIC:
                    if set(report["ranks_present"]) != expected:
                        continue
                    if ranks != expected:
                        # A rank-dependent signature is eligible only as part
                        # of one complete case/protocol assignment. Never fill
                        # missing peers from a different case or replay run.
                        assignment = assignments[(phase, protocol_key)]
                        if not (
                            set(assignment) == expected
                            and all(len(keys) == 1 for keys in assignment.values())
                            and all(
                                keys <= observed_by_rank.get(rank, set())
                                for rank, keys in assignment.items()
                            )
                        ):
                            continue
                        rank_assignment = {
                            str(rank): next(iter(keys)) for rank, keys in sorted(assignment.items())
                        }
                index.setdefault(key, []).append(
                    {
                        "status": status,
                        "run_id": report["run_id"],
                        "case_id": case["case_id"],
                        "ranks": sorted(ranks),
                        "protocol": observations[0]["protocol"],
                        **(
                            {"rank_assignment": rank_assignment}
                            if rank_assignment is not None
                            else {}
                        ),
                    }
                )
    return index, rejected


def build_report(inventories: list[dict], evidence: list[dict]) -> dict:
    """Keep stale evidence, unknown variants and incomplete captures out of D."""
    if not inventories:
        raise ValueError("No inventory shards supplied")
    first = inventories[0]
    context = first["context"]
    if not isinstance(context.get("world_size"), int) or context["world_size"] < 1:
        raise ValueError("world_size must be a positive integer")
    by_rank = {}
    observed_by_rank: dict[int, set[str]] = {}
    operations: dict[str, dict] = {}
    issues = []
    for inventory in inventories:
        if inventory.get("schema_version") != 1 or inventory.get("kind") != "determinism_inventory":
            raise ValueError("Unsupported inventory schema")
        if inventory["context"] != context or inventory["recipe_id"] != first["recipe_id"]:
            raise ValueError("Cannot combine different recipes, revisions, or environments")
        rank = inventory["rank"]
        if not isinstance(rank, int) or not 0 <= rank < context["world_size"] or rank in by_rank:
            raise ValueError(f"Duplicate or invalid rank: {rank}")
        by_rank[rank] = inventory
        if not inventory["complete"]:
            issues.append(f"Rank {rank}: capture did not complete")
        if inventory.get("context_after", context) != context:
            issues.append(f"Rank {rank}: source or environment context changed during capture")
        if inventory.get("truncated", False):
            issues.append(f"Rank {rank}: signature limit reached")
        for operation in inventory["operations"]:
            signature = operation["signature"]
            key = signature_key(signature)
            observed_by_rank.setdefault(rank, set()).add(key)
            row = operations.setdefault(
                key, {"signature": signature, "calls": 0, "ranks": set(), "sites": set()}
            )
            if not isinstance(operation["calls"], int) or operation["calls"] < 1:
                raise ValueError("Invocation counts must be positive integers")
            row["calls"] += operation["calls"]
            row["ranks"].add(rank)
            row["sites"].add(operation.get("site", "unspecified"))
    missing = set(range(context["world_size"])) - by_rank.keys()
    if missing:
        issues.append(f"Missing ranks: {sorted(missing)}")
    if context.get("dirty", True):
        issues.append("Source tree has uncommitted changes")
    index, rejected = _evidence_index(evidence, context, observed_by_rank)
    cases = []
    for key, operation in sorted(operations.items()):
        matches = [
            match
            for match in index.get(key, [])
            if match["status"] != DETERMINISTIC or operation["ranks"] <= set(match["ranks"])
        ]
        statuses = {match["status"] for match in matches}
        if not _has_memory_fill_policy(operation["signature"]):
            status = UNVERIFIED
            reason = "Memory-fill policy is missing or not a boolean"
        elif NONDETERMINISTIC in statuses:
            status = NONDETERMINISTIC
            reason = "Matching numerical mismatch; passing evidence does not erase it"
        elif DETERMINISTIC in statuses and not issues:
            status = DETERMINISTIC
            reason = "Matching scoped replay evidence; recipe replay still required"
        else:
            status = UNVERIFIED
            reason = "Incomplete capture" if issues else "No matching current replay evidence"
        cases.append(
            {
                **operation,
                "signature_id": key,
                "ranks": sorted(operation["ranks"]),
                "sites": sorted(operation["sites"]),
                "status": status,
                "reason": reason,
                "evidence": matches,
            }
        )
    counts = {status: sum(case["status"] == status for case in cases) for status in STATUSES}
    total = len(cases)
    return {
        "schema_version": 1,
        "kind": "recipe_determinism_coverage",
        "recipe_id": first["recipe_id"],
        "context": context,
        "scope": "Unique observed signatures at explicitly bound Python entrypoints",
        "recipe_status": (
            "known_nondeterministic_operation" if counts[NONDETERMINISTIC] else "replay_required"
        ),
        "counts": {"total": total, **counts},
        "deterministic_percent": 100 * counts[DETERMINISTIC] / total if total else None,
        "verified_percent": (
            100 * (counts[DETERMINISTIC] + counts[NONDETERMINISTIC]) / total if total else None
        ),
        "capture_issues": issues,
        "rejected_evidence_runs": rejected,
        "limitations": [
            "Unbound/native/compiled graph internals are outside this inventory",
            "A matching operation test does not certify scheduling, full training state, or restart",
        ],
        "operations": cases,
    }


def markdown_report(report: dict) -> str:
    """Render failures and gaps with their test evidence."""
    lines = [
        "# Recipe determinism coverage",
        "",
        f"Recipe: `{report['recipe_id']}`. Verdict: **{report['recipe_status']}**.",
        "",
        report["scope"],
        "",
        "| Operation | Implementation | Phase | Calls | Status |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for operation in report["operations"]:
        signature = operation["signature"]
        fields = [
            signature["op_id"],
            signature["implementation"],
            signature["phase"],
            str(operation["calls"]),
            operation["status"],
        ]
        lines.append(
            "| "
            + " | ".join(field.replace("|", "\\|").replace("\n", " ") for field in fields)
            + " |"
        )
    lines.extend(["", "Capture issues: " + ("; ".join(report["capture_issues"]) or "none"), ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Join inventory shards to versioned coverage reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--evidence", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on known N; exit 2 for unknown/incomplete coverage",
    )
    args = parser.parse_args(argv)
    inventories = [
        json.loads(path.read_text()) for path in sorted(args.inventory.glob("rank-*.json"))
    ]
    report = build_report(inventories, [json.loads(path.read_text()) for path in args.evidence])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".md").write_text(markdown_report(report))
    print(markdown_report(report))
    if args.strict:
        if report["counts"][NONDETERMINISTIC]:
            return 1
        if (
            report["capture_issues"]
            or report["counts"][UNVERIFIED]
            or not report["counts"]["total"]
        ):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
