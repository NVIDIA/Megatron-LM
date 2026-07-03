# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Profile the four mHC high-priority stream modes with Nsight Systems.

This is an intentionally small, mock-data benchmark. It is meant to expose
stream placement and bubbles, not to represent a production performance
recipe. Use ``--dry-run`` to inspect the exact commands without GPUs or nsys.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

MODES = ("none", "post", "recompute", "all")

# These options determine wrapper-side rank selection, output labels, capture windows,
# or the schedule topology being profiled. Letting trailing training arguments override
# them would make the generated report misleading or break multi-node coordination.
CONTROLLED_TRAINING_ARGS = frozenset(
    {
        "--distributed-backend",
        "--enable-hyper-connections",
        "--expert-model-parallel-size",
        "--expert-tensor-parallel-size",
        "--high-priority-a2a-comm-stream",
        "--mhc-high-priority-stream-mode",
        "--mhc-recompute-layer-num",
        "--moe-flex-dispatcher-backend",
        "--moe-token-dispatcher-type",
        "--num-layers-per-virtual-pipeline-stage",
        "--num-residual-streams",
        "--overlap-moe-expert-parallel-comm",
        "--pipeline-model-parallel-size",
        "--profile",
        "--profile-ranks",
        "--profile-step-end",
        "--profile-step-start",
        "--recompute-granularity",
        "--recompute-modules",
        "--tensor-model-parallel-size",
        "--train-iters",
        "--use-pytorch-profiler",
    }
)


@dataclass(frozen=True)
class Preset:
    """Distributed topology and dispatcher defaults for a hardware target."""

    name: str
    nnodes: int
    nproc_per_node: int
    dispatcher: str
    flex_backend: str | None = None


PRESETS = {
    "h100-nccl": Preset("h100-nccl", nnodes=1, nproc_per_node=8, dispatcher="alltoall"),
    "gb200-flex": Preset(
        "gb200-flex", nnodes=2, nproc_per_node=4, dispatcher="flex", flex_backend="deepep"
    ),
}


@dataclass(frozen=True)
class RunCase:
    """One stream-placement configuration in the profiling matrix."""

    mode: str
    high_priority_comm: bool = False

    @property
    def label(self) -> str:
        suffix = "_comm_high" if self.high_priority_comm else ""
        return f"{self.mode}{suffix}"


def _environment_int(*names: str, default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return default


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse profiling matrix command-line options."""

    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="h100-nccl")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--include-comm-priority-case",
        action="store_true",
        help="also profile mode=all with the A2A communication stream at high priority",
    )
    parser.add_argument(
        "--flex-backend",
        choices=("deepep", "hybridep"),
        default=os.environ.get("MHC_FLEX_BACKEND", "deepep"),
        help="backend for gb200-flex (also read from MHC_FLEX_BACKEND)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=repository_root / "artifacts" / "mhc_overlap_nsys"
    )
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for torchrun")
    parser.add_argument("--nsys-bin", default="nsys")
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
    parser.add_argument(
        "--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "29500"))
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=_environment_int("NODE_RANK", "SLURM_NODEID", default=0),
        help="rank of this node for the two-node preset",
    )
    parser.add_argument(
        "--profile-ranks",
        nargs="+",
        type=int,
        help="global ranks that call cudaProfilerStart/Stop (default: rank 0)",
    )
    parser.add_argument(
        "--profile-node-leaders",
        action="store_true",
        help="profile the first global rank on every node (useful for gb200-flex)",
    )
    parser.add_argument("--profile-step-start", type=int, default=5)
    parser.add_argument("--profile-step-end", type=int, default=8)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--gpu-metrics-devices",
        help="optional nsys GPU metrics device list, for example '0' or 'all'",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="do not export cuda_gpu_kern_sum and nvtx_gpu_proj_sum CSV files",
    )
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help=(
            "extra model or batch arguments for pretrain_gpt.py after '--'; "
            "wrapper-controlled arguments are rejected"
        ),
    )
    args = parser.parse_args(argv)
    if args.training_args[:1] == ["--"]:
        args.training_args = args.training_args[1:]
    _validate_training_args(args.training_args)
    return args


def _validate_training_args(training_args: Sequence[str]) -> None:
    """Reject trailing arguments that would invalidate wrapper-side assumptions."""

    overridden = sorted(
        {
            argument.partition("=")[0]
            for argument in training_args
            if argument.startswith("--") and argument.partition("=")[0] in CONTROLLED_TRAINING_ARGS
        }
    )
    if overridden:
        raise ValueError(
            "The profiling wrapper controls these training arguments; use its named options "
            f"instead: {', '.join(overridden)}"
        )


def _profile_ranks(args: argparse.Namespace, preset: Preset) -> list[int]:
    if args.profile_ranks:
        ranks = sorted(set(args.profile_ranks))
    elif args.profile_node_leaders:
        ranks = [node * preset.nproc_per_node for node in range(preset.nnodes)]
    else:
        ranks = [0]
    world_size = preset.nnodes * preset.nproc_per_node
    invalid = [rank for rank in ranks if rank < 0 or rank >= world_size]
    if invalid:
        raise ValueError(f"profile ranks {invalid} are outside world size {world_size}")
    return ranks


def _torchrun_command(
    args: argparse.Namespace, preset: Preset, case: RunCase, profile_ranks: Sequence[int]
) -> list[str]:
    command = [args.python, "-m", "torch.distributed.run"]
    if preset.nnodes == 1:
        command.extend(("--standalone", f"--nproc-per-node={preset.nproc_per_node}"))
    else:
        command.extend(
            (
                f"--nnodes={preset.nnodes}",
                f"--nproc-per-node={preset.nproc_per_node}",
                f"--node-rank={args.node_rank}",
                f"--master-addr={args.master_addr}",
                f"--master-port={args.master_port}",
            )
        )

    command.extend(
        (str(Path(__file__).resolve().parents[1] / "pretrain_gpt.py"), *_model_args(args))
    )
    command.extend(_dispatcher_args(preset, args.flex_backend))
    command.extend(("--mhc-high-priority-stream-mode", case.mode))
    if case.high_priority_comm:
        command.append("--high-priority-a2a-comm-stream")
    command.extend(("--profile-ranks", *(str(rank) for rank in profile_ranks)))
    command.extend(args.training_args)
    return command


def _model_args(args: argparse.Namespace) -> list[str]:
    """A compact model that still exercises PP/VPP, EP A2A, mHC, and recompute."""

    return [
        "--num-layers",
        "8",
        "--hidden-size",
        "1024",
        "--ffn-hidden-size",
        "4096",
        "--num-attention-heads",
        "16",
        "--seq-length",
        "1024",
        "--max-position-embeddings",
        "1024",
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        "16",
        "--train-iters",
        str(args.train_iters),
        "--lr",
        "1.0e-4",
        "--lr-decay-style",
        "constant",
        "--min-lr",
        "1.0e-4",
        "--lr-warmup-fraction",
        "0.0",
        "--weight-decay",
        "0.0",
        "--clip-grad",
        "1.0",
        "--seed",
        str(args.seed),
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "2",
        "--num-layers-per-virtual-pipeline-stage",
        "2",
        "--expert-model-parallel-size",
        "4",
        "--expert-tensor-parallel-size",
        "1",
        "--num-experts",
        "8",
        "--moe-ffn-hidden-size",
        "2048",
        "--moe-router-load-balancing-type",
        "aux_loss",
        "--moe-router-topk",
        "2",
        "--moe-aux-loss-coeff",
        "1.0e-2",
        "--moe-grouped-gemm",
        "--overlap-moe-expert-parallel-comm",
        "--enable-hyper-connections",
        "--num-residual-streams",
        "4",
        "--mhc-sinkhorn-iterations",
        "20",
        "--recompute-granularity",
        "selective",
        "--recompute-modules",
        "mhc",
        "--mhc-recompute-layer-num",
        "2",
        "--mock-data",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "32000",
        "--num-workers",
        "0",
        "--split",
        "99,1,0",
        "--bf16",
        "--transformer-impl",
        "transformer_engine",
        "--normalization",
        "RMSNorm",
        "--swiglu",
        "--disable-bias-linear",
        "--position-embedding-type",
        "rope",
        "--untie-embeddings-and-output-weights",
        "--use-mcore-models",
        "--distributed-backend",
        "nccl",
        "--no-gradient-accumulation-fusion",
        "--attention-softmax-in-fp32",
        "--log-interval",
        "1",
        "--eval-iters",
        "0",
        "--profile",
        "--profile-step-start",
        str(args.profile_step_start),
        "--profile-step-end",
        str(args.profile_step_end),
        "--nvtx-ranges",
    ]


def _dispatcher_args(preset: Preset, flex_backend: str) -> list[str]:
    if preset.dispatcher == "alltoall":
        return ["--moe-token-dispatcher-type", "alltoall"]
    return ["--moe-token-dispatcher-type", "flex", "--moe-flex-dispatcher-backend", flex_backend]


def _nsys_command(
    nsys_bin: str,
    output_stem: Path,
    training_command: Sequence[str],
    gpu_metrics_devices: str | None,
) -> list[str]:
    command = [
        nsys_bin,
        "profile",
        "--sample=none",
        "--cpuctxsw=none",
        "--trace=cuda,nvtx",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--cuda-graph-trace=node",
        "--cuda-memory-usage=true",
        "--force-overwrite=true",
        "--output",
        str(output_stem),
    ]
    if gpu_metrics_devices:
        command.append(f"--gpu-metrics-devices={gpu_metrics_devices}")
    command.extend(training_command)
    return command


def _run_stats(nsys_bin: str, report: Path, output_stem: Path) -> None:
    for report_name in ("cuda_gpu_kern_sum", "nvtx_gpu_proj_sum"):
        command = [
            nsys_bin,
            "stats",
            "--report",
            report_name,
            "--format",
            "csv",
            "--output",
            str(output_stem),
            "--force-overwrite=true",
            str(report),
        ]
        print("+", shlex.join(command), flush=True)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            print(
                f"warning: nsys could not export {report_name} (exit {error.returncode})",
                file=sys.stderr,
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Run or print the requested Nsight Systems profiling matrix."""

    args = parse_args(argv)
    preset = PRESETS[args.preset]
    if not 0 <= args.node_rank < preset.nnodes:
        raise ValueError(f"node rank {args.node_rank} is invalid for {preset.nnodes} nodes")
    if args.profile_step_end <= args.profile_step_start:
        raise ValueError("--profile-step-end must be greater than --profile-step-start")
    if args.train_iters <= args.profile_step_end:
        raise ValueError("--train-iters must be greater than --profile-step-end")

    profile_ranks = _profile_ranks(args, preset)
    local_rank_start = args.node_rank * preset.nproc_per_node
    local_global_ranks = set(range(local_rank_start, local_rank_start + preset.nproc_per_node))
    profile_this_node = bool(local_global_ranks.intersection(profile_ranks))
    backend_label = args.flex_backend if preset.dispatcher == "flex" else "nccl"
    cases = [RunCase(mode) for mode in args.modes]
    if args.include_comm_priority_case:
        cases.append(RunCase("all", high_priority_comm=True))

    if not args.dry_run:
        if shutil.which(args.python) is None and not Path(args.python).exists():
            raise FileNotFoundError(f"Python interpreter not found: {args.python}")
        if profile_this_node and shutil.which(args.nsys_bin) is None:
            raise FileNotFoundError(f"Nsight Systems CLI not found: {args.nsys_bin}")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    # Fine-grained expert-parallel overlap requires enough CUDA work queues to
    # preserve the intended A2A/compute concurrency on both Hopper and Blackwell.
    environment["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
    environment.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    if preset.name == "h100-nccl":
        environment.setdefault("NCCL_ALGO", "Ring")

    for case in cases:
        output_name = f"{preset.name}_{backend_label}_{case.label}_node{args.node_rank}"
        output_stem = args.output_dir.resolve() / output_name
        training_command = _torchrun_command(args, preset, case, profile_ranks)
        command = (
            _nsys_command(args.nsys_bin, output_stem, training_command, args.gpu_metrics_devices)
            if profile_this_node
            else training_command
        )
        print("+", shlex.join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(
            command, check=True, cwd=Path(__file__).resolve().parents[1], env=environment
        )

        report = output_stem.with_suffix(".nsys-rep")
        if profile_this_node and not args.skip_stats:
            if report.exists():
                _run_stats(args.nsys_bin, report, output_stem)
            else:
                print(f"warning: expected report was not created: {report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
