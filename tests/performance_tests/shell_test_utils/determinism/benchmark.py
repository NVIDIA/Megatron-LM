# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repeated unprofiled deterministic/default training benchmarks.

Each arm starts a fresh process. With --base-checkout the same launcher and
Python environment run both source revisions on the same allocation. Raw logs,
step samples, source provenance, and paired ratios are retained as artifacts.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path

TIMING = re.compile(r"iteration\s+(\d+)/\s*\d+.*elapsed time per iteration \(ms\):\s*([^\s|]+)")
MODE_ENV = (
    "NCCL_ALGO",
    "CUBLAS_WORKSPACE_CONFIG",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "MAMBA_DETERMINISTIC",
    "CAUSAL_CONV1D_DETERMINISTIC",
    "TRITON_CACHE_AUTOTUNING",
)
DET_ENV = {
    "NCCL_ALGO": "Ring",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "MAMBA_DETERMINISTIC": "1",
    "CAUSAL_CONV1D_DETERMINISTIC": "1",
}
DEFAULT_ENV = {
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1",
    "MAMBA_DETERMINISTIC": "0",
    "CAUSAL_CONV1D_DETERMINISTIC": "0",
}


def mode_environment(parent: dict[str, str], mode: str) -> dict[str, str]:
    """Set policy before imports, avoiding inherited deterministic pins."""
    if mode not in ("det", "default"):
        raise ValueError(f"Unknown mode: {mode}")
    environment = {key: value for key, value in parent.items() if key not in MODE_ENV}
    environment.update(DET_ENV if mode == "det" else DEFAULT_ENV)
    environment.update(DETERMINISM_PERF_MODE=mode, PYTHONHASHSEED="0")
    environment.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    return environment


def read_step_times(directory: Path, warmup: int, steps: int) -> tuple[list[float], str]:
    """Require one complete rank timing log and every requested step."""
    candidates = []
    for path in sorted(directory.glob("**/stdout.log")):
        samples = {}
        for line in path.read_text(errors="replace").splitlines():
            match = TIMING.search(line)
            if match is None:
                continue
            iteration, milliseconds = int(match[1]), float(match[2])
            if iteration in samples:
                raise ValueError(f"Repeated iteration {iteration} in {path}; possible restart")
            if not math.isfinite(milliseconds) or milliseconds <= 0:
                raise ValueError(f"Invalid timing in {path}: {milliseconds}")
            samples[iteration] = milliseconds
        if samples:
            candidates.append((path, samples))
    if len(candidates) != 1:
        raise ValueError(f"Expected one rank timing log in {directory}, found {len(candidates)}")
    path, samples = candidates[0]
    missing = set(range(1, warmup + steps + 1)) - samples.keys()
    if missing:
        raise ValueError(f"Missing iterations in {path}: {sorted(missing)}")
    return [samples[index] for index in range(warmup + 1, warmup + steps + 1)], str(path)


def ratio_summary(ratios: list[float], limit: float | None) -> dict:
    """Bootstrap paired-run ratios; ambiguous or undersampled checks stay pending."""
    if not ratios or any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("Ratios must be finite and positive")
    rng = random.Random(0)
    medians = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    low, high = medians[49], medians[1949]
    if len(ratios) < 3:
        status = "inconclusive"
    elif limit is None:
        status = "not_gated"
    elif low > limit:
        status = "fail"
    elif high <= limit:
        status = "pass"
    else:
        status = "inconclusive"
    return {
        "paired_ratios": ratios,
        "median_ratio": statistics.median(ratios),
        "bootstrap_95_percent_interval": [low, high],
        "limit": limit,
        "status": status,
    }


def summarize(
    runs: list[dict],
    pairs: int,
    has_base: bool,
    overhead_limit: float | None,
    regression_limit: float | None,
) -> dict:
    """Report mode overhead and base-to-head changes as separate comparisons."""
    values = {}
    for run in runs:
        key = (run["pair"], run["revision_label"], run["mode"])
        if key in values or run.get("status") != "complete":
            raise ValueError("Duplicate or incomplete benchmark arm")
        values[key] = run["median_ms"]
    labels = ("base", "head") if has_base else ("head",)
    expected = {
        (pair, label, mode)
        for pair in range(pairs)
        for label in labels
        for mode in ("det", "default")
    }
    if set(values) != expected:
        raise ValueError("Missing or unexpected benchmark arms")

    def ratios(numerator, denominator):
        return [values[(pair, *numerator)] / values[(pair, *denominator)] for pair in range(pairs)]

    result = {
        "head_overhead": ratio_summary(ratios(("head", "det"), ("head", "default")), overhead_limit)
    }
    if has_base:
        result["base_overhead"] = ratio_summary(
            ratios(("base", "det"), ("base", "default")), overhead_limit
        )
        for mode in ("det", "default"):
            result[f"{mode}_regression"] = ratio_summary(
                ratios(("head", mode), ("base", mode)), regression_limit
            )
    return result


def _source(checkout: Path) -> dict:
    return {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--untracked-files=normal",
                    "--",
                    "megatron",
                    "tools",
                    "tests",
                    "pretrain_*.py",
                    "pyproject.toml",
                    "uv.lock",
                    "setup.py",
                    "setup.cfg",
                ],
                cwd=checkout,
            )
        ),
        "checkout": str(checkout),
    }


def _machine() -> dict:
    versions: dict[str, str | None] = {}
    for name in ("torch", "triton", "transformer-engine", "mamba-ssm", "causal-conv1d"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    hardware: list[str] | None
    try:
        hardware = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            .strip()
            .splitlines()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        hardware = None
    return {
        "host": platform.node(),
        "python": platform.python_version(),
        "versions": versions,
        "gpus": hardware,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def markdown_report(report: dict) -> str:
    """Render the leaderboard for this fixed recipe and environment."""
    lines = [
        "# Determinism performance",
        "",
        f"Workload: `{report['measurement'].get('kernel_case') or report['measurement']['recipe']}`. "
        f"Status: **{report['status']}**.",
        "",
        (
            "Diagnostic only: external telemetry accompanies event timing, followed by profiles. "
            "Ineligible for performance approval and baseline publication."
            if report["measurement"].get("diagnostic_only")
            else "Timings are unprofiled; each pair uses independent fresh processes."
        ),
        "",
        "| Comparison | Median ratio | Paired bootstrap 95% interval | Limit | Status |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for name, result in report.get("comparisons", {}).items():
        low, high = result["bootstrap_95_percent_interval"]
        limit = f"{result['limit']:.4f}" if result["limit"] is not None else "report only"
        lines.append(
            f"| {name} | {result['median_ratio']:.4f} | [{low:.4f}, {high:.4f}] | {limit} | {result['status']} |"
        )
    lines.extend(
        [
            "",
            "No numerical determinism claim is made by timing results; link replay evidence separately.",
            "",
        ]
    )
    if report.get("error"):
        lines.append(f"Error: {report['error']}\n")
    return "\n".join(lines)


def _write(report: dict, output: Path):
    temporary = output / "benchmark.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "benchmark.json")
    (output / "benchmark.md").write_text(markdown_report(report))


def main(argv: list[str] | None = None) -> int:
    """Launch paired measurements and retain incomplete/failed attempts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-checkout", type=Path)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--recipe", choices=("dense", "moe", "hybrid"), default="dense")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument(
        "--kernel-case", choices=("bias_swiglu", "weighted_swiglu", "weighted_squared_relu")
    )
    parser.add_argument("--phase", choices=("forward", "backward"), default="forward")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Kernel-only telemetry and post-timing profiles; ineligible for performance gates",
    )
    parser.add_argument("--max-overhead-ratio", type=float)
    parser.add_argument("--max-regression-ratio", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.diagnostics and (
        not args.kernel_case
        or args.command
        or args.max_overhead_ratio is not None
        or args.max_regression_ratio is not None
    ):
        parser.error(
            "Diagnostics require the built-in kernel runner and cannot use performance limits"
        )
    if min(args.pairs, args.steps, args.gpus, args.tokens, args.hidden_size) < 1 or args.warmup < 0:
        parser.error("pairs, steps, and gpus must be positive; warmup must be nonnegative")
    if not args.kernel_case:
        if args.max_overhead_ratio is None:
            args.max_overhead_ratio = 1.35
        if args.max_regression_ratio is None:
            args.max_regression_ratio = 1.05
    elif args.gpus != 1 or args.command:
        parser.error("Kernel measurements require --gpus 1 and the built-in kernel runner")
    elif args.warmup < 1:
        parser.error("Kernel measurements require at least one warmup to exclude compilation")
    for value in (args.max_overhead_ratio, args.max_regression_ratio):
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error("ratio limits must be finite and positive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory; previous attempts must be preserved")
    output.mkdir(parents=True, exist_ok=True)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.kernel_case:
        command = [sys.executable, str(Path(__file__).with_name("run_kernel.py").resolve())]
        for option in ("kernel_case", "phase", "tokens", "hidden_size", "dtype", "warmup", "steps"):
            command.extend(["--" + option.replace("_", "-"), str(getattr(args, option))])
        if args.diagnostics:
            command.append("--diagnostics")
    elif not command:
        command = [sys.executable, str(Path(__file__).with_name("run_training.py").resolve())]
    checkouts = {"head": Path.cwd()}
    if args.base_checkout:
        checkouts["base"] = args.base_checkout.resolve()
    sources = {label: _source(checkout) for label, checkout in checkouts.items()}
    report = {
        "schema_version": 1,
        "kind": "determinism_kernel_performance" if args.kernel_case else "determinism_performance",
        "status": "incomplete",
        "sources": sources,
        "machine": _machine(),
        "measurement": {
            key: getattr(args, key) for key in ("pairs", "warmup", "steps", "recipe", "gpus")
        },
        "command": command,
        "runs": [],
    }
    if args.kernel_case:
        report["measurement"].pop("recipe")
        report["measurement"].update(
            {
                key: getattr(args, key)
                for key in ("kernel_case", "phase", "tokens", "hidden_size", "dtype")
            }
        )
        report["measurement"]["timing"] = "cuda_event_ms"
        if args.diagnostics:
            report["measurement"]["diagnostic_only"] = True
    _write(report, output)
    try:
        for pair in range(args.pairs):
            arms = [(label, mode) for label in sorted(checkouts) for mode in ("default", "det")]
            if pair % 2:
                arms.reverse()
            for label, mode in arms:
                run_dir = output / f"pair-{pair}" / f"{label}-{mode}"
                run_dir.mkdir(parents=True)
                environment = mode_environment(dict(os.environ), mode)
                environment.update(
                    DETERMINISM_PERF_LOG_DIR=str(run_dir),
                    DETERMINISM_PERF_TRAIN_ITERS=str(args.warmup + args.steps),
                    DETERMINISM_PERF_RECIPE=args.recipe,
                    DETERMINISM_PERF_GPUS=str(args.gpus),
                    DETERMINISM_PERF_PROFILE="0",
                    PYTHONPATH=str(checkouts[label]),
                )
                run = {
                    "pair": pair,
                    "revision_label": label,
                    "mode": mode,
                    "status": "incomplete",
                    "log_directory": str(run_dir),
                    "environment": {
                        key: environment.get(key)
                        for key in sorted(
                            set(MODE_ENV)
                            | {"CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_PROTO", "TRITON_CACHE_DIR"}
                            | {
                                key
                                for key in environment
                                if key.startswith("TRITON_AUTOTUNE_BLOCK_")
                            }
                        )
                    },
                }
                report["runs"].append(run)
                _write(report, output)
                print(f"Benchmark pair {pair + 1}/{args.pairs}: {label}, {mode}", flush=True)
                with (run_dir / "launcher.log").open("w") as log:
                    subprocess.run(
                        command,
                        cwd=checkouts[label],
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                if args.kernel_case:
                    from run_kernel import read_result

                    result = read_result(run_dir / "kernel.json", report["measurement"], mode)
                    samples = result["samples_ms"]
                    log_path = str(run_dir / "kernel.json")
                    run["kernel"] = result
                else:
                    samples, log_path = read_step_times(run_dir, args.warmup, args.steps)
                if _source(checkouts[label]) != sources[label]:
                    raise ValueError(f"Source changed during benchmark: {label}")
                run.update(
                    status="complete", median_ms=statistics.median(samples), timing_log=log_path
                )
                run["samples_ms" if args.kernel_case else "step_times_ms"] = samples
                _write(report, output)
        report["comparisons"] = summarize(
            report["runs"],
            args.pairs,
            bool(args.base_checkout),
            args.max_overhead_ratio,
            args.max_regression_ratio,
        )
        gated = [
            result["status"]
            for name, result in report["comparisons"].items()
            if name != "base_overhead" and result["status"] != "not_gated"
        ]
        report["status"] = (
            "fail"
            if "fail" in gated
            else "inconclusive" if "inconclusive" in gated else "pass" if gated else "reported"
        )
        if args.diagnostics:
            report["status"] = "diagnostic"
        if any(source["dirty"] for source in sources.values()) or report["machine"]["gpus"] is None:
            report["status"] = "inconclusive"
            report["error"] = "Clean source and GPU provenance are required for a gating result"
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        report.update(status="error", error=str(error))
    finally:
        _write(report, output)
    print(markdown_report(report))
    return {"pass": 0, "reported": 0, "diagnostic": 0, "fail": 1, "error": 1, "inconclusive": 2}[
        report["status"]
    ]


if __name__ == "__main__":
    raise SystemExit(main())
