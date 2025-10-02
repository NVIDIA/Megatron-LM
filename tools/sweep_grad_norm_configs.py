#!/usr/bin/env python3
"""Run a grad-norm sharding sweep and summarize results.

This script iterates over tensor/pipeline/context parallel configurations that fit
within a given world size, launches `pretrain_gpt.py` for each setup, captures the
logged global gradient norm (using Megatron's `--log-global-grad-norms` flag), and
reports a comparison table against a baseline configuration.

Example usage (matches the mock-data experiments we have been running):

    python tools/sweep_grad_norm_configs.py \
        --world-size 8 \
        --common-args-file common_args.txt \
        --log-dir logs/grad_norm_sweep

Where `common_args.txt` might contain:

    --use-mcore-models
    --num-layers 4
    --hidden-size 256
    --ffn-hidden-size 1024
    --num-attention-heads 4
    --seq-length 128
    --max-position-embeddings 128
    --micro-batch-size 1
    --global-batch-size 8
    --train-iters 1
    --lr 1e-4
    --min-lr 1e-5
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --tokenizer-type NullTokenizer
    --vocab-size 8192
    --bf16
    --mock-data
    --seed 42
    --log-interval 1
    --no-load-optim
    --no-load-rng

The script produces a table with absolute / relative deviations vs. the baseline
configuration (default baseline: tp=1, pp=1, cp=1).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class ParallelConfig:
    tp: int
    pp: int
    cp: int
    dp: int

    def tag(self) -> str:
        return f"tp{self.tp}_pp{self.pp}_cp{self.cp}_dp{self.dp}"


@dataclass
class RunResult:
    config: ParallelConfig
    grad_norm: float | None
    status: str
    runtime_sec: float | None
    log_path: Path
    error: str | None = None
    abs_diff: float | None = None
    rel_diff: float | None = None


def parse_common_args(args_file: Path | None, inline_args: str | None) -> List[str]:
    tokens: List[str] = []
    if args_file:
        for raw_line in args_file.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens.extend(shlex.split(line))
    if inline_args:
        tokens.extend(shlex.split(inline_args))
    return tokens


def enumerate_configs(
    world_size: int,
    tp_values: Sequence[int],
    pp_values: Sequence[int],
    cp_values: Sequence[int],
    include_baseline: bool = True,
) -> List[ParallelConfig]:
    configs: List[ParallelConfig] = []
    for tp in tp_values:
        for pp in pp_values:
            for cp in cp_values:
                denom = tp * pp * cp
                if denom == 0 or world_size % denom != 0:
                    continue
                dp = world_size // denom
                if dp <= 0:
                    continue
                configs.append(ParallelConfig(tp=tp, pp=pp, cp=cp, dp=dp))

    configs.sort(key=lambda c: (c.tp, c.pp, c.cp, c.dp))
    if include_baseline:
        baseline = ParallelConfig(tp=1, pp=1, cp=1, dp=world_size)
        if baseline not in configs:
            configs.insert(0, baseline)
        else:
            # Move baseline to the front for convenience
            configs = [baseline] + [cfg for cfg in configs if cfg != baseline]
    return configs


def build_run_command(
    torchrun: str,
    pretrain_script: str,
    world_size: int,
    base_args: Sequence[str],
    config: ParallelConfig,
    log_path: Path,
    iteration: int,
) -> List[str]:
    cmd: List[str] = [
        torchrun,
        "--standalone",
        f"--nproc_per_node={world_size}",
        pretrain_script,
    ]
    cmd.extend(base_args)
    cmd.extend(
        [
            "--tensor-model-parallel-size",
            str(config.tp),
            "--pipeline-model-parallel-size",
            str(config.pp),
            "--context-parallel-size",
            str(config.cp),
            "--log-global-grad-norms",
            str(log_path),
            "--log-grad-norm-iteration",
            str(iteration),
        ]
    )
    return cmd


def run_configuration(
    cmd: Sequence[str],
    log_path: Path,
    timeout: float | None,
    env: dict[str, str] | None = None,
) -> tuple[str, float | None, str | None]:
    """Execute a training run and return (status, runtime, error_message)."""
    if log_path.exists():
        log_path.unlink()

    start = time.time()
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=timeout,
            env=env,
        )
        runtime = time.time() - start
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - defensive branch
        return "timeout", None, f"Command timed out after {timeout} seconds: {exc}"

    if completed.returncode != 0:
        return "failed", runtime, f"Return code {completed.returncode}"

    return "ok", runtime, None


def extract_grad_norm(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    last_record = None
    with log_path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            last_record = stripped
    if last_record is None:
        return None
    try:
        payload = json.loads(last_record)
    except json.JSONDecodeError:  # pragma: no cover - defensive
        return None
    return float(payload.get("grad_norm")) if "grad_norm" in payload else None


def summarize(results: List[RunResult]) -> None:
    header = "tp pp cp dp grad_norm abs_diff rel_diff status runtime[s] log_path"
    print(header)
    print("-" * len(header))
    for result in results:
        if result.grad_norm is None:
            grad_str = "-"
        else:
            grad_str = f"{result.grad_norm:.6e}"
        if result.abs_diff is None:
            abs_str = "-"
            rel_str = "-"
        else:
            abs_str = f"{result.abs_diff:.3e}"
            rel_str = f"{result.rel_diff:.3e}"
        runtime_str = (
            f"{result.runtime_sec:.1f}" if result.runtime_sec is not None else "-"
        )
        print(
            f"{result.config.tp:2d} {result.config.pp:2d} {result.config.cp:2d} {result.config.dp:2d} "
            f"{grad_str:>12} {abs_str:>10} {rel_str:>10} {result.status:>8} {runtime_str:>9} {result.log_path}"
        )
        if result.error:
            print(f"    error: {result.error}")


def write_csv(results: List[RunResult], csv_path: Path) -> None:
    import csv

    with csv_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "tp",
                "pp",
                "cp",
                "dp",
                "grad_norm",
                "abs_diff",
                "rel_diff",
                "status",
                "runtime_sec",
                "log_path",
                "error",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.config.tp,
                    result.config.pp,
                    result.config.cp,
                    result.config.dp,
                    result.grad_norm,
                    result.abs_diff,
                    result.rel_diff,
                    result.status,
                    result.runtime_sec,
                    str(result.log_path),
                    result.error,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--torchrun", default="torchrun", help="Path to torchrun executable"
    )
    parser.add_argument(
        "--pretrain-script",
        default="pretrain_gpt.py",
        help="Training entry point to invoke",
    )
    parser.add_argument(
        "--world-size", type=int, required=True, help="Total number of ranks per run"
    )
    parser.add_argument(
        "--common-args-file",
        type=Path,
        help="Optional path to a file containing baseline Megatron arguments (one per line)",
    )
    parser.add_argument(
        "--common-args",
        type=str,
        help="Additional inline Megatron arguments (quoted string)",
    )
    parser.add_argument(
        "--tp-values",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8),
        help="Tensor-parallel sizes to consider",
    )
    parser.add_argument(
        "--pp-values",
        type=int,
        nargs="+",
        default=(1, 2, 4),
        help="Pipeline-parallel sizes to consider",
    )
    parser.add_argument(
        "--cp-values",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8),
        help="Context-parallel sizes to consider",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Directory where per-run JSONL logs will be stored",
    )
    parser.add_argument(
        "--log-iteration",
        type=int,
        default=0,
        help="Iteration index at which Megatron should log the grad norm",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout (seconds) for each torchrun invocation",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to write the summary table as CSV",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Do not automatically prepend the (1,1,1) baseline configuration",
    )

    args = parser.parse_args()

    base_args = parse_common_args(args.common_args_file, args.common_args)
    if not base_args:
        parser.error(
            "No base Megatron arguments provided. Use --common-args-file and/or --common-args."
        )

    configs = enumerate_configs(
        world_size=args.world_size,
        tp_values=args.tp_values,
        pp_values=args.pp_values,
        cp_values=args.cp_values,
        include_baseline=not args.skip_baseline,
    )
    if not configs:
        parser.error("No valid parallel configurations for the requested sweep.")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    results: List[RunResult] = []

    baseline_grad: float | None = None

    # Determine whether FSDP is enabled; if so, do NOT force CUDA_DEVICE_MAX_CONNECTIONS=1
    uses_fsdp = ("--use-torch-fsdp2" in base_args) or ("--use-custom-fsdp" in base_args)

    for idx, config in enumerate(configs):
        log_path = args.log_dir / f"{config.tag()}.jsonl"
        cmd = build_run_command(
            torchrun=args.torchrun,
            pretrain_script=args.pretrain_script,
            world_size=args.world_size,
            base_args=base_args,
            config=config,
            log_path=log_path,
            iteration=args.log_iteration,
        )

        print(f"\n=== [{idx+1}/{len(configs)}] Running {config.tag()} ===")
        print("Command:", " ".join(shlex.quote(part) for part in cmd))
        run_env = os.environ.copy()
        if not uses_fsdp:
            run_env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        status, runtime, error = run_configuration(
            cmd, log_path, args.timeout, env=run_env
        )
        grad_norm = extract_grad_norm(log_path)

        result = RunResult(
            config=config,
            grad_norm=grad_norm,
            status=status,
            runtime_sec=runtime,
            log_path=log_path,
            error=error,
        )

        if status != "ok":
            print(f"  -> status={status}, error={error}")
        elif grad_norm is None:
            result.status = "no_log"
            result.error = "Grad norm log missing"
            print("  -> grad norm not found in log")
        else:
            print(f"  -> grad_norm={grad_norm:.6e}")

        results.append(result)

        if baseline_grad is None and result.grad_norm is not None:
            baseline_grad = result.grad_norm

        if baseline_grad is not None and result.grad_norm is not None:
            result.abs_diff = result.grad_norm - baseline_grad
            denom = max(abs(baseline_grad), 1e-12)
            result.rel_diff = result.abs_diff / denom

    print("\n=== Summary ===")
    summarize(results)

    if args.output_csv:
        write_csv(results, args.output_csv)
        print(f"\nWrote CSV summary to {args.output_csv}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
