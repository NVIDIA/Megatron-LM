"""Compare a perf test results.json against a checked-in baseline_values.json.

Exit code:
  0 — all metrics within tolerance
  1 — at least one regression

For throughput-style metrics, fail when measured < baseline * (1 - tol).
For latency-style metrics, fail when measured > baseline * (1 + tol).

Default tolerance is 10%; override per-test via TOLERANCE_PCT in model_config.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

THROUGHPUT_METRICS = {"throughput_tok_per_sec"}
LATENCY_METRICS = {
    "avg_latency_ms",
    "p50_latency_ms",
    "p99_latency_ms",
    "tpot_ms_per_tok",
}


def _check(label: str, measured: float, baseline: float, tol: float, higher_is_better: bool) -> tuple[bool, str]:
    if higher_is_better:
        floor = baseline * (1 - tol)
        ok = measured >= floor
        delta_pct = (measured - baseline) / baseline * 100
        return ok, (
            f"  {'PASS' if ok else 'FAIL'} {label:30s} measured={measured:10.3f}  "
            f"baseline={baseline:10.3f}  delta={delta_pct:+6.2f}%  floor={floor:10.3f}"
        )
    else:
        ceiling = baseline * (1 + tol)
        ok = measured <= ceiling
        delta_pct = (measured - baseline) / baseline * 100
        return ok, (
            f"  {'PASS' if ok else 'FAIL'} {label:30s} measured={measured:10.3f}  "
            f"baseline={baseline:10.3f}  delta={delta_pct:+6.2f}%  ceiling={ceiling:10.3f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results.json from this run.")
    ap.add_argument("--baseline", required=True, help="Path to baseline_values.json.")
    ap.add_argument("--config", required=True, help="Path to model_config.yaml (for tolerance + metrics list).")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    baseline = json.loads(Path(args.baseline).read_text())
    config = yaml.safe_load(Path(args.config).read_text())

    tol = float(config.get("TOLERANCE_PCT", 10)) / 100.0
    metrics: list[str] = list(config.get("METRICS") or sorted(THROUGHPUT_METRICS | LATENCY_METRICS))

    print(f"Comparing {args.results} vs {args.baseline}")
    print(f"Tolerance: ±{tol * 100:.1f}%  Metrics: {metrics}")

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
