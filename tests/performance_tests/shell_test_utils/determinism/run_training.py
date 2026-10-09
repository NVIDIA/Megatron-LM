# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared training launcher for unprofiled measurements and optional diagnosis."""

from __future__ import annotations

import argparse
import os
import sys


def training_command(
    recipe: str, mode: str, gpus: int, train_iters: int, log_dir: str, profile: bool = False
) -> list[str]:
    """Build identical model configurations for the two execution policies."""
    if recipe not in ("dense", "moe", "hybrid") or mode not in ("det", "default"):
        raise ValueError("Unknown benchmark recipe or mode")
    if gpus < 2 or gpus % 2:
        raise ValueError("These TP=2 recipes require a positive even GPU count")
    script = "pretrain_hybrid.py" if recipe == "hybrid" else "pretrain_gpt.py"
    model = (
        ["--hybrid-layer-pattern", "M-*-", "--mamba-num-groups", "8", "--mamba-state-dim", "128"]
        if recipe == "hybrid"
        else ["--num-layers", "4"]
    )
    if recipe == "hybrid":
        model += ["--spec", "megatron.core.models.hybrid.hybrid_layer_specs", "hybrid_stack_spec"]
    if recipe == "moe":
        if gpus % 4:
            raise ValueError("The TP=2, EP=2 MoE recipe requires a multiple of four GPUs")
        model += [
            "--num-experts",
            "8",
            "--moe-router-topk",
            "2",
            "--moe-grouped-gemm",
            "--expert-model-parallel-size",
            "2",
            "--moe-token-dispatcher-type",
            "alltoall",
            "--sequence-parallel",
        ]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--log-dir",
        str(log_dir),
        "--tee",
        f"0:3,{gpus - 1}:3",
        "--redirects",
        "3",
        "--nproc_per_node",
        str(gpus),
        script,
        *model,
        "--hidden-size",
        "1024",
        "--num-attention-heads",
        "16",
        "--seq-length",
        "256",
        "--max-position-embeddings",
        "256",
        "--micro-batch-size",
        "2",
        "--global-batch-size",
        "16",
        "--train-iters",
        str(train_iters),
        "--lr",
        "1e-4",
        "--lr-decay-style",
        "constant",
        "--lr-decay-iters",
        "100",
        "--min-lr",
        "1e-5",
        "--weight-decay",
        "0",
        "--clip-grad",
        "1.0",
        "--tensor-model-parallel-size",
        "2",
        "--pipeline-model-parallel-size",
        "1",
        "--distributed-backend",
        "nccl",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "256",
        "--mock-data",
        "--split",
        "1,0,0",
        "--transformer-impl",
        "transformer_engine",
        "--use-mcore-models",
        "--no-gradient-accumulation-fusion",
        "--bf16",
        "--seed",
        "1234",
        "--log-interval",
        "1",
        "--eval-iters",
        "0",
        "--eval-interval",
        "10000",
        "--no-load-optim",
        "--no-load-rng",
    ]
    if mode == "det":
        command.append("--deterministic-mode")
    if profile:
        command.extend(
            ["--profile", "--nvtx-ranges", "--profile-step-start", "5", "--profile-step-end", "7"]
        )
    return command


def main(argv: list[str] | None = None) -> None:
    """Replace this process with the training launch for one benchmark arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("dense", "moe", "hybrid"), default="dense")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--mode", choices=("det", "default"), required=True)
    parser.add_argument("--train-iters", type=int, default=70)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--profile", action="store_true", help="Add Nsight profiler ranges")
    args = parser.parse_args(argv)
    from benchmark import mode_environment

    environment = mode_environment(dict(os.environ), args.mode)
    command = training_command(
        args.recipe, args.mode, args.gpus, args.train_iters, args.log_dir, args.profile
    )
    os.execve(command[0], command, environment)


if __name__ == "__main__":
    main()
