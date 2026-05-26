# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare a perf test results.json against a checked-in baseline_values.json.

Exit code:
  0 — all metrics within tolerance
  1 — at least one regression OR an improvement large enough to require baseline refresh

For throughput-style metrics:
  - Fail when measured < baseline * (1 - tol).        ← regression
  - Fail when measured > baseline * (1 + UPPER_TOL).  ← improvement too large; refresh baseline
For latency-style metrics:
  - Fail when measured > baseline * (1 + tol).        ← regression (slower)

The two-sided check on throughput matches the rule the old
tests/functional_tests/python_test_utils/test_inference_regular_pipeline.py
applied. Speed-ups beyond UPPER_TOL would silently weaken regression detection
for future runs, so we force a baseline refresh instead.

Default tol is 10% (configurable via TOLERANCE_PCT in model_config.yaml).
Default UPPER_TOL is 20% (configurable via UPPER_TOLERANCE_PCT, throughput only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

THROUGHPUT_METRICS = {"throughput_tok_per_sec"}
LATENCY_METRICS = {"avg_latency_ms", "p50_latency_ms", "p99_latency_ms", "tpot_ms_per_tok"}


def _check(
    label: str,
    measured: float,
    baseline: float,
    tol: float,
    higher_is_better: bool,
    upper_tol: float | None = None,
) -> tuple[bool, str]:
    delta_pct = (measured - baseline) / baseline * 100
    if higher_is_better:
        floor = baseline * (1 - tol)
        ceiling = baseline * (1 + upper_tol) if upper_tol is not None else None
        if measured < floor:
            verdict = "FAIL"
            ok = False
            tail = f"floor={floor:10.3f}  (regression)"
        elif ceiling is not None and measured > ceiling:
            verdict = "FAIL"
            ok = False
            tail = f"ceiling={ceiling:10.3f}  (improvement too large — refresh baseline)"
        else:
            verdict = "PASS"
            ok = True
            tail = f"floor={floor:10.3f}"
            if ceiling is not None:
                tail = f"{tail}  ceiling={ceiling:10.3f}"
        return ok, (
            f"  {verdict} {label:30s} measured={measured:10.3f}  "
            f"baseline={baseline:10.3f}  delta={delta_pct:+6.2f}%  {tail}"
        )
    ceiling = baseline * (1 + tol)
    ok = measured <= ceiling
    return ok, (
        f"  {'PASS' if ok else 'FAIL'} {label:30s} measured={measured:10.3f}  "
        f"baseline={baseline:10.3f}  delta={delta_pct:+6.2f}%  ceiling={ceiling:10.3f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results.json from this run.")
    ap.add_argument("--baseline", required=True, help="Path to baseline_values.json.")
    ap.add_argument(
        "--config", required=True, help="Path to model_config.yaml (for tolerance + metrics list)."
    )
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    baseline = json.loads(Path(args.baseline).read_text())
    config = yaml.safe_load(Path(args.config).read_text())

    tol = float(config.get("TOLERANCE_PCT", 10)) / 100.0
    upper_tol = float(config.get("UPPER_TOLERANCE_PCT", 20)) / 100.0
    metrics: list[str] = list(config.get("METRICS") or sorted(THROUGHPUT_METRICS | LATENCY_METRICS))

    print(f"Comparing {args.results} vs {args.baseline}")
    print(
        f"Tolerance: -{tol * 100:.1f}% (regression) / "
        f"+{upper_tol * 100:.1f}% (improvement, throughput only)"
    )
    print(f"Metrics: {metrics}")

    all_ok = True
    for batch_key, baseline_entry in baseline.items():
        if batch_key not in results:
            print(f"FAIL: {batch_key} present in baseline but missing from results")
            all_ok = False
            continue
        print(f"\n[{batch_key}]")
        measured_entry = results[batch_key]
        for metric in metrics:
            if metric not in baseline_entry or metric not in measured_entry:
                continue
            higher_is_better = metric in THROUGHPUT_METRICS
            ok, line = _check(
                metric,
                measured_entry[metric],
                baseline_entry[metric],
                tol,
                higher_is_better,
                upper_tol=upper_tol if higher_is_better else None,
            )
            all_ok = all_ok and ok
            print(line)

    print()
    if all_ok:
        print("OK: all metrics within tolerance.")
        return 0
    print("REGRESSION: one or more metrics outside tolerance — see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
