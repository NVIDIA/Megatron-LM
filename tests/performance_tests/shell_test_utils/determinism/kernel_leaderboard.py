# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run a small per-operator leaderboard without assigning uncalibrated budgets."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import benchmark


def markdown_report(reports: list[dict]) -> str:
    """Rank completed rows by overhead, retaining failed and incomplete rows."""
    lines = [
        "# Determinism kernel performance",
        "",
        "CUDA-event operator latency; forward and backward are timed separately.",
        "Report only until hardware-specific budgets are calibrated. Replay/correctness is separate.",
        "",
        "| Case | Phase | Default us | Deterministic us | Det/default | 95% interval | Status |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    ordered = sorted(
        reports,
        key=lambda row: row.get("comparisons", {}).get("head_overhead", {}).get("median_ratio", -1),
        reverse=True,
    )
    for report in ordered:
        measurement = report["measurement"]
        comparison = report.get("comparisons", {}).get("head_overhead")
        values = []
        for mode in ("default", "det"):
            medians = [
                run["median_ms"] * 1000
                for run in report["runs"]
                if run["revision_label"] == "head"
                and run["mode"] == mode
                and run["status"] == "complete"
            ]
            values.append(f"{statistics.median(medians):.3f}" if comparison and medians else "n/a")
        ratio = f"{comparison['median_ratio']:.4f}" if comparison else "n/a"
        interval = (
            " / ".join(f"{value:.4f}" for value in comparison["bootstrap_95_percent_interval"])
            if comparison
            else "n/a"
        )
        lines.append(
            f"| {measurement['kernel_case']} | {measurement['phase']} | {values[0]} | {values[1]} | "
            f"{ratio} | {interval} | {report['status']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write a leaderboard plus the full per-case reports, even after failed arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args(argv)
    if min(args.pairs, args.steps, args.warmup) < 1:
        parser.error("pairs, steps, and warmup must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory; previous attempts must be preserved")
    args.output.mkdir(parents=True, exist_ok=True)
    reports, codes = [], []
    for case in ("bias_swiglu", "weighted_swiglu", "weighted_squared_relu"):
        for phase in ("forward", "backward"):
            output = args.output / f"{case}-{phase}"
            codes.append(
                benchmark.main(
                    [
                        "--output",
                        str(output),
                        "--kernel-case",
                        case,
                        "--phase",
                        phase,
                        "--gpus",
                        "1",
                        "--pairs",
                        str(args.pairs),
                        "--warmup",
                        str(args.warmup),
                        "--steps",
                        str(args.steps),
                    ]
                )
            )
            reports.append(json.loads((output / "benchmark.json").read_text()))
            (args.output / "leaderboard.json").write_text(
                json.dumps(reports, indent=2, allow_nan=False) + "\n"
            )
            (args.output / "leaderboard.md").write_text(markdown_report(reports))
    print(markdown_report(reports))
    return 1 if 1 in codes else 2 if 2 in codes else 0


if __name__ == "__main__":
    raise SystemExit(main())
