# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Interaction coverage over the declared model replay matrix, not all products."""

from itertools import combinations

AXES = ("TP", "PP", "VPP", "CP", "EP", "FSDP")


def normalize_parallelism(value: dict) -> dict[str, int]:
    """Normalize an explicit plan; reject ambiguous sizes and misspelled axes."""
    if not isinstance(value, dict) or set(value) - set(AXES):
        raise ValueError(f"Parallelism must be a mapping over {AXES}")
    if any(type(size) is not int or size < 1 for size in value.values()):
        raise ValueError("Parallelism sizes must be positive integers")
    return {axis: value.get(axis, 1) for axis in AXES}


def parallelism_report(cases: list[dict]) -> dict:
    """Require matching runtime axes on every observation before crediting a plan.

    A pair is verified only if all selected model variants containing that pair
    completed matching replay. One passing preset cannot hide another's skip.
    """
    rows = []
    pairs: dict[tuple, list[dict]] = {}
    unplanned = []
    for case in cases:
        plan = case.get("parallelism_plan")
        if plan is None:
            unplanned.append(case["case_id"])
            continue
        plan = normalize_parallelism(plan)
        observations = case["observations"]
        matched = bool(observations) and all(
            obs["signature"].get("parallelism") == plan for obs in observations
        )
        if plan["FSDP"] > 1:
            matched = matched and all(
                obs["signature"].get("fsdp", {}).get("sharding_strategy") == "optim_grads_params"
                for obs in observations
            )
        status = case["status"] if matched else "not_verified"
        row = {
            "case_id": case["case_id"],
            "model_id": case["op_id"],
            "planned": plan,
            "runtime_matches_plan": matched,
            "status": status,
            "reasons": case["reasons"]
            + ([] if matched else ["Runtime parallelism or sharding policy is missing or differs"]),
        }
        rows.append(row)
        for left, right in combinations(AXES, 2):
            key = (case["op_id"], left, plan[left], right, plan[right])
            pairs.setdefault(key, []).append(row)
    pair_rows = []
    for (model, left, left_value, right, right_value), contributors in sorted(pairs.items()):
        statuses = {row["status"] for row in contributors}
        status = (
            "verified_nondeterministic"
            if "verified_nondeterministic" in statuses
            else (
                "verified_deterministic"
                if statuses == {"verified_deterministic"}
                else "not_verified"
            )
        )
        pair_rows.append(
            {
                "model_id": model,
                "values": {left: left_value, right: right_value},
                "status": status,
                "cases": [row["case_id"] for row in contributors],
            }
        )
    counts = {
        status: sum(row["status"] == status for row in pair_rows)
        for status in ("verified_deterministic", "verified_nondeterministic", "not_verified")
    }
    return {
        "scope": (
            "Axis-value pairs present in the explicitly selected matrix, per model; "
            "all contributing cases required"
        ),
        "axes": list(AXES),
        "rows": rows,
        "unplanned_cases": unplanned,
        "pairs": pair_rows,
        "counts": {"total": len(pair_rows), **counts},
        "deterministic_percent": (
            100 * counts["verified_deterministic"] / len(pair_rows) if pair_rows else None
        ),
    }
