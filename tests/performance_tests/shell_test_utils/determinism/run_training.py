# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared training launcher for unprofiled measurements and optional diagnosis."""

from __future__ import annotations

import os
import sys


def training_command(environment: dict[str, str]) -> list[str]:
    """Build identical model configurations for the two execution policies."""
    recipe = environment.get("DETERMINISM_PERF_RECIPE", "dense")
    mode = environment["DETERMINISM_PERF_MODE"]
    if recipe not in ("dense", "moe", "hybrid") or mode not in ("det", "default", "nondet"):
        raise ValueError("Unknown benchmark recipe or mode")
    gpus = int(environment.get("DETERMINISM_PERF_GPUS", "8"))
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
        environment["DETERMINISM_PERF_LOG_DIR"],
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
        environment.get("DETERMINISM_PERF_TRAIN_ITERS", "70"),
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
    if environment.get("DETERMINISM_PERF_PROFILE") == "1":
        command.extend(
            ["--profile", "--nvtx-ranges", "--profile-step-start", "5", "--profile-step-end", "7"]
        )
    return command


if __name__ == "__main__":
    from benchmark import mode_environment

    mode = os.environ["DETERMINISM_PERF_MODE"]
    environment = mode_environment(dict(os.environ), "default" if mode == "nondet" else mode)
    argv = training_command(environment)
    os.execve(argv[0], argv, environment)
