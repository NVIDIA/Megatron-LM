# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Paired same-allocation timings of captured direct TP/SP mapping phases.

Requires the capture/replay producer and startup API. Run in the clean head
checkout, outside torchrun, on the capture's original single-node allocation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from benchmark import _machine, _source
from collective_case import (
    aggregate_arm,
    comparisons,
    digest,
    install_head_helpers,
    policy_environment,
    validate_evidence,
)


def write_report(report: dict, output: Path) -> None:
    """Retain partial attempts and readable per-group rows."""
    temporary = output / "benchmark.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "benchmark.json")
    lines = [
        "# Captured collective performance",
        "",
        f"Status: **{report['status']}**.",
        "",
        "Isolated operator phases; each sample is the maximum latency across the captured group. "
        "Pairs are fresh processes. Warmup, input restoration, backward graph setup and alignment "
        "barriers are excluded. CUDA-event intervals include launch gaps and arrival skew. "
        "This does not measure whole-model throughput or communication overlap.",
        "",
        "| Event/group | Comparison | Median ratio | Paired bootstrap 95% interval | Limit | Status |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for key, values in report.get("comparisons", {}).items():
        for name, value in values.items():
            low, high = value["bootstrap_95_percent_interval"]
            limit = "report only" if value["limit"] is None else f"{value['limit']:.4f}"
            lines.append(
                f"| {key} | {name} | {value['median_ratio']:.4f} | [{low:.4f}, {high:.4f}] | {limit} | {value['status']} |"
            )
    if report.get("error"):
        lines.extend(["", f"Error: {report['error']}"])
    (output / "benchmark.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Require accuracy evidence, then launch alternating fresh policy/source arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-checkout", type=Path)
    parser.add_argument("--event-indices", nargs="+", type=int)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-overhead-ratio", type=float)
    parser.add_argument("--max-regression-ratio", type=float)
    args = parser.parse_args(argv)
    if min(args.pairs, args.warmup, args.steps, args.max_bytes) < 1:
        parser.error("pairs, warmup, steps and max-bytes must be positive")
    if (
        "TORCHELASTIC_RUN_ID" in os.environ
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
        or int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != 1
        or int(os.environ.get("RANK", "0")) != 0
    ):
        parser.error("Launch the parent outside torchrun")
    if args.max_regression_ratio is not None and args.base_checkout is None:
        parser.error("A revision limit requires --base-checkout")
    for limit in (args.max_overhead_ratio, args.max_regression_ratio):
        if limit is not None and (not math.isfinite(limit) or limit <= 0):
            parser.error("Ratio limits must be finite and positive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory; preserve previous attempts")
    output.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema_version": 1,
        "kind": "determinism_collective_performance",
        "status": "incomplete",
        "runs": [],
    }
    try:
        head = Path(__file__).resolve().parents[4]
        if Path.cwd().resolve() != head:
            raise ValueError("Run the benchmark from its head checkout")
        install_head_helpers(head)
        from tools.determinism.collective_capture import load_captures

        capture_root = args.capture.resolve()
        captures = load_captures(capture_root, max_bytes=args.max_bytes)
        indices = (
            args.event_indices
            if args.event_indices is not None
            else list(range(len(captures[0]["events"])))
        )
        if (
            not indices
            or indices != sorted(set(indices))
            or indices[0] < 0
            or indices[-1] >= len(captures[0]["events"])
        ):
            raise ValueError("Event indices must be unique, increasing and within the capture")
        evidence_path = args.evidence.resolve()
        evidence_hash = digest(evidence_path)
        evidence = json.loads(evidence_path.read_text())
        matched = validate_evidence(captures, indices, evidence)
        checkouts = {"head": head}
        if args.base_checkout:
            checkouts["base"] = args.base_checkout.resolve()
        sources = {label: _source(checkout) for label, checkout in checkouts.items()}
        if (
            any(source["dirty"] for source in sources.values())
            or sources["head"]["revision"] != captures[0]["context"]["revision"]
        ):
            raise ValueError("Clean source and a current-head capture are required")
        files = [
            *Path(__file__).parent.glob("*.py"),
            *(head / "tools/determinism").glob("*.py"),
            *(head / "megatron/determinism").glob("*.py"),
        ]
        measurement = {
            "pairs": args.pairs,
            "warmup": args.warmup,
            "steps": args.steps,
            "world_size": len(captures),
            "event_indices": indices,
            "max_bytes": args.max_bytes,
            "timing": "cuda_event_ms",
            "aggregation": "per_sample_group_max",
            "manifest_sha256": [
                digest(capture_root / f"rank-{rank}/manifest.json") for rank in range(len(captures))
            ],
            "tooling": {str(path.relative_to(head)): digest(path) for path in sorted(files)},
        }
        report.update(
            sources=sources,
            machine=_machine(),
            measurement=measurement,
            capture={
                "path": str(capture_root),
                "context": captures[0]["context"],
                "recipe_id": captures[0]["recipe_id"],
            },
            replay_evidence={
                "path": str(evidence_path),
                "sha256": evidence_hash,
                "run_id": evidence["run_id"],
                "events": matched,
            },
        )
        write_report(report, output)
        collective = captures[0]["events"][indices[0]]["signature"]["configuration"]["collective"]
        for pair in range(args.pairs):
            arms = [(label, mode) for label in sorted(checkouts) for mode in ("default", "det")]
            if pair % 2:
                arms.reverse()
            for label, mode in arms:
                run_dir = output / f"pair-{pair}" / f"{label}-{mode}"
                run_dir.mkdir(parents=True)
                request = {
                    "head_checkout": str(head),
                    "capture": str(capture_root),
                    "source": sources[label],
                    "mode": mode,
                    "measurement": measurement,
                }
                request_path = run_dir / "request.json"
                request_path.write_text(json.dumps(request, indent=2, allow_nan=False) + "\n")
                environment = policy_environment(
                    dict(os.environ), captures[0]["context"], collective, mode
                )
                environment["PYTHONPATH"] = str(checkouts[label])
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={len(captures)}",
                    str(Path(__file__).with_name("run_collectives.py")),
                    "--request",
                    str(request_path),
                ]
                run = {
                    "pair": pair,
                    "revision_label": label,
                    "mode": mode,
                    "status": "incomplete",
                    "log_directory": str(run_dir),
                    "command": command,
                }
                report["runs"].append(run)
                write_report(report, output)
                print(f"Collective pair {pair + 1}/{args.pairs}: {label}, {mode}", flush=True)
                with (run_dir / "launcher.log").open("w") as log:
                    subprocess.run(
                        command,
                        cwd=checkouts[label],
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                results = [
                    json.loads((run_dir / f"rank-{rank}.json").read_text())
                    for rank in range(len(captures))
                ]
                run["rows"] = aggregate_arm(
                    results, captures, measurement, mode, sources[label]["revision"]
                )
                run["rank_files"] = {
                    f"rank-{rank}.json": digest(run_dir / f"rank-{rank}.json")
                    for rank in range(len(captures))
                }
                if any(_source(checkout) != sources[name] for name, checkout in checkouts.items()):
                    raise ValueError("Source changed during the benchmark")
                if (
                    digest(evidence_path) != evidence_hash
                    or load_captures(capture_root, max_bytes=args.max_bytes) != captures
                ):
                    raise ValueError("Capture or replay evidence changed during the benchmark")
                run["status"] = "complete"
                write_report(report, output)
        report["comparisons"] = comparisons(
            report["runs"],
            measurement,
            bool(args.base_checkout),
            args.max_overhead_ratio,
            args.max_regression_ratio,
        )
        gated = [
            value["status"]
            for row in report["comparisons"].values()
            for name, value in row.items()
            if name != "base_overhead" and value["status"] != "not_gated"
        ]
        report["status"] = (
            "fail"
            if "fail" in gated
            else "inconclusive" if "inconclusive" in gated else "pass" if gated else "reported"
        )
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        ImportError,
        subprocess.CalledProcessError,
    ) as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        write_report(report, output)
    return {"pass": 0, "reported": 0, "fail": 1, "error": 1, "inconclusive": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
