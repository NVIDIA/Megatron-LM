# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Internal pytest worker used by the MLite local-test entry."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


class OutcomeRecorder:
    """Collect rank-local test outcomes without depending on pytest plugins."""

    def __init__(self) -> None:
        self.outcomes: dict[str, str] = {}
        self.collection_errors = 0

    def pytest_collectreport(self, report) -> None:
        if report.failed:
            self.collection_errors += 1
        elif report.skipped:
            nodeid = str(report.nodeid or "<collection>")
            self.outcomes[f"{nodeid}::<collection>"] = "skipped"

    def pytest_runtest_logreport(self, report) -> None:
        nodeid = report.nodeid
        was_xfail = hasattr(report, "wasxfail")

        if report.when == "setup" and report.skipped:
            self.outcomes[nodeid] = "xfailed" if was_xfail else "skipped"
            return
        if report.when == "setup" and report.failed:
            self.outcomes[nodeid] = "error"
            return
        if report.when == "teardown" and report.failed:
            self.outcomes[nodeid] = "error"
            return
        if report.when == "teardown" and report.skipped:
            self.outcomes[nodeid] = "xfailed" if was_xfail else "skipped"
            return
        if report.when != "call":
            return

        if report.skipped:
            self.outcomes[nodeid] = "xfailed" if was_xfail else "skipped"
        elif report.failed:
            self.outcomes[nodeid] = "failed"
        elif was_xfail:
            self.outcomes[nodeid] = "xpassed"
        else:
            self.outcomes[nodeid] = "passed"

    def as_dict(self, pytest_exit_code: int) -> dict[str, object]:
        counts = {
            key: sum(outcome == key for outcome in self.outcomes.values())
            for key in ("passed", "failed", "skipped", "xfailed", "xpassed", "error")
        }
        serialized_outcomes = json.dumps(
            self.outcomes, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        executed = sum(counts.values())
        passed = not (
            pytest_exit_code
            or self.collection_errors
            or counts["failed"]
            or counts["skipped"]
            or counts["xpassed"]
            or counts["error"]
            or executed == 0
        )
        return {
            "rank": int(os.environ.get("RANK", "0")),
            "status": "PASS" if passed else "FAIL",
            "counts": counts,
            "outcome_digest": hashlib.sha256(serialized_outcomes).hexdigest(),
        }


def _write_report(report: dict[str, object]) -> None:
    report_dir = Path(os.environ["MLITE_TEST_REPORT_DIR"])
    report_dir.mkdir(parents=True, exist_ok=True)
    rank = int(report["rank"])
    destination = report_dir / f"rank-{rank}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "MLITE_TEST_REPORT_DIR" not in os.environ:
        return 1

    import pytest

    recorder = OutcomeRecorder()
    exit_code = int(pytest.main(args, plugins=[recorder]))
    report = recorder.as_dict(exit_code)
    _write_report(report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
