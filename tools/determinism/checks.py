# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Reference-correctness and comparator controls, separate from replay status."""

from __future__ import annotations

import contextlib
import contextvars
from typing import Callable, Iterator

from tools.determinism.configuration import configured_signature

PASSED = "passed"
FAILED = "failed"
UNVERIFIED = "not_verified"
KINDS = ("reference", "sensitivity")
_OBSERVER: contextvars.ContextVar[Callable[[dict], None] | None] = contextvars.ContextVar(
    "kernel_check_observer", default=None
)


@contextlib.contextmanager
def collect_checks(sink: Callable[[dict], None]) -> Iterator[None]:
    """Route checks into the current pytest case without changing replay evidence."""
    token = _OBSERVER.set(sink)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


def has_comparisons(check: dict) -> bool:
    """Require nonempty output checks, and gradients for a backward case."""
    outputs, gradients = check.get("compared_outputs", 0), check.get("compared_gradients", 0)
    return (
        type(outputs) is int
        and outputs > 0
        and type(gradients) is int
        and gradients >= 0
        and (check["signature"].get("phase") != "forward_backward" or gradients > 0)
        and (
            not check.get("protocol", {}).get("numerical_controls_required", False)
            or (
                len(check.get("numerical_controls", {})) == outputs + gradients
                and check["numerical_controls"].keys() == check.get("metrics", {}).keys()
                and all(
                    control.get("detected") is True
                    for control in check["numerical_controls"].values()
                )
            )
        )
        and (
            check["kind"] != "sensitivity"
            or (
                type(check.get("detected_perturbations")) is int
                and check["detected_perturbations"] == outputs + gradients
            )
        )
    )


@contextlib.contextmanager
def observe_check(kind: str, signature: dict, protocol: dict) -> Iterator[dict]:
    """Keep a failed reference assertion distinct from an observed replay mismatch."""
    if kind not in KINDS:
        raise ValueError(f"Unknown kernel check: {kind}")
    check = {
        "kind": kind,
        "signature": configured_signature(signature),
        "protocol": protocol,
        "status": UNVERIFIED,
    }
    try:
        yield check
    except AssertionError as error:
        check.update(status=FAILED, reason=str(error))
        raise
    except BaseException as error:
        check["reason"] = f"{type(error).__name__}: {error}"
        raise
    else:
        if has_comparisons(check):
            check["status"] = PASSED
        else:
            check["reason"] = "Required output/gradient comparisons or perturbations are missing"
    finally:
        observer = _OBSERVER.get()
        if observer is not None:
            observer(check)


def check_statuses(
    checks: list[dict], observations: list[dict], *, complete: bool, fresh: bool
) -> dict:
    """Require a matching check for every replay signature on every required rank."""
    for check in checks:
        if check.get("kind") not in KINDS or check.get("status") not in (
            PASSED,
            FAILED,
            UNVERIFIED,
        ):
            raise ValueError("Unknown kernel check kind or status")
    statuses = {}
    for kind in KINDS:
        rows = [check for check in checks if check["kind"] == kind]
        matches = [
            [
                row
                for row in rows
                if row["rank"] == obs["rank"] and row["signature"] == obs["signature"]
            ]
            for obs in observations
        ]
        if not fresh:
            status = UNVERIFIED
        elif any(row["status"] == FAILED for group in matches for row in group):
            status = FAILED
        elif (
            complete
            and matches
            and all(
                group
                and len(group)
                >= sum(
                    other["rank"] == observation["rank"]
                    and other["signature"] == observation["signature"]
                    for other in observations
                )
                and all(
                    row["status"] == PASSED
                    and has_comparisons(row)
                    and all(
                        row.get(key) == observation.get(key)
                        for key in ("compared_outputs", "compared_gradients")
                    )
                    for row in group
                )
                for observation, group in zip(observations, matches)
            )
            and all(any(row in group for group in matches) for row in rows)
            and all(obs["status"] == "verified_deterministic" for obs in observations)
        ):
            status = PASSED
        else:
            status = UNVERIFIED
        statuses[kind] = status
    return statuses


def author_requirements(inventory: dict, cases: list[dict]) -> list[dict]:
    """Evaluate manifest-declared author tests; a removed or skipped case cannot pass."""
    requirements = []
    for op_id, entry in sorted(inventory.items()):
        for node_id in entry.get("author_tests", []):
            selected = [
                case for case in cases if case["op_id"] == op_id and case["case_id"] == node_id
            ]
            statuses = [case["check_status"][kind] for case in selected for kind in KINDS]
            status = (
                FAILED
                if FAILED in statuses
                else (
                    PASSED
                    if selected and all(value == PASSED for value in statuses)
                    else UNVERIFIED
                )
            )
            requirements.append(
                {
                    "op_id": op_id,
                    "test": node_id,
                    "status": status,
                    "cases": [case["case_id"] for case in selected],
                }
            )
    return requirements
