# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared arg vectors, env, parallel-layout flags and training schedule for the
reverse-converter end-to-end suite.

Kept **stdlib-only** (no ``torch`` / ``megatron`` import) so importing it during
pytest collection is cheap.
"""

import os
from pathlib import Path

# tests/integration_tests/tools/checkpoint/fsdp_dtensor_to_torch_dist/config.py
# -> repo root is five package levels up.
REPO_ROOT = Path(__file__).resolve().parents[5]
INSPECTOR = REPO_ROOT / "tools" / "checkpoint" / "checkpoint_inspector.py"

# --- tiny shared model config (identical across every family) ----------------
# 12L/512H bf16 deterministic mock-data run. Dropout is off: the classic resume
# loads with --no-load-rng, so its dropout masks would differ from the FSDP run's
# and add noise that hides small conversion errors. Per-family ``num_layers`` and
# ``arch`` are appended by the caller; caller-specific flags (--train-iters, --save,
# --save-interval, --load) are appended per stage, NOT here.
COMMON_ARGS = (
    "--hidden-size", "512", "--num-attention-heads", "8",
    "--seq-length", "1024", "--max-position-embeddings", "1024",
    "--micro-batch-size", "4", "--global-batch-size", "32",
    "--eval-interval", "1000", "--eval-iters", "5", "--split", "949,50,1",
    "--lr", "1.5e-4", "--min-lr", "1e-5", "--lr-decay-style", "cosine",
    "--lr-warmup-fraction", "0.01",
    "--weight-decay", "1e-2", "--clip-grad", "1.0", "--use-checkpoint-opt_param-scheduler",
    "--transformer-impl", "transformer_engine", "--bf16", "--deterministic-mode",
    "--no-gradient-accumulation-fusion", "--seed", "1234",
    "--hidden-dropout", "0.0", "--attention-dropout", "0.0",
    "--tokenizer-type", "NullTokenizer", "--vocab-size", "32000", "--mock-data",
    "--log-interval", "1",
    "--tensor-model-parallel-size", "1",
)  # fmt: skip

# Deterministic env for FSDP train + classic resume. Omitting
# NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 with --deterministic-mode + TE fails the model
# build. Do NOT add CUDA_DEVICE_MAX_CONNECTIONS=1 here — single-rank FSDP must not
# set it.
DETERMINISTIC_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "NCCL_ALGO": "Ring",
}

# Multi-GPU (2-rank) NCCL rules, applied on top of DETERMINISTIC_ENV:
#   * classic *load* (reshard) needs P2P disabled (pre-Blackwell) and one copy engine.
#   * FSDP *training* on >1 GPU wants P2P disabled but must NOT set the copy-engine
#     limit (Megatron-FSDP forbids CUDA_DEVICE_MAX_CONNECTIONS=1).
MULTI_GPU_LOAD_ENV = {"NCCL_P2P_DISABLE": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}
MULTI_GPU_TRAIN_ENV = {"NCCL_P2P_DISABLE": "1"}

# Produce the fsdp_dtensor source checkpoint.
FSDP_TRAIN_FLAGS = (
    "--use-megatron-fsdp", "--use-distributed-optimizer", "--ckpt-format", "fsdp_dtensor",
    "--data-parallel-sharding-strategy", "optim_grads_params",
)  # fmt: skip

# Resume a classic (non-FSDP) job from the converted torch_dist checkpoint.
# --dist-ckpt-optim-fully-reshardable is the only distributed-optimizer on-disk
# layout that is per-parameter and model-shaped (what the converter emits);
# log_all strictness drops the omitted TE _extra_state keys instead of erroring.
CLASSIC_LOAD_FLAGS = (
    "--use-distributed-optimizer", "--ckpt-format", "torch_dist",
    "--dist-ckpt-optim-fully-reshardable", "--dist-ckpt-strictness", "log_all",
    "--no-load-rng",
)  # fmt: skip


# --- training schedule --------------------------------------------------------
# Train to 100 saving every 20, then convert two interior saves: both mid-cosine
# decay (LR still moving), so a single lucky checkpoint cannot pass, and each has
# RESUME_EXTRA_ITERS FSDP reference iterations after it to compare the resume with.
TRAIN_ITERS = 100
SAVE_INTERVAL = 20
CONVERT_ITERS = (60, 80)
RESUME_EXTRA_ITERS = 3


# --- parallel-layout flags ---------------------------------------------------
def target_parallel_flags(layout: str):
    """Classic *load* layout for the reshard sweep."""
    return {
        "TP2": ("--tensor-model-parallel-size", "2"),
        "TP2SP": ("--tensor-model-parallel-size", "2", "--sequence-parallel"),
        "PP2": ("--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", "2"),
        "EP2": ("--tensor-model-parallel-size", "1", "--expert-model-parallel-size", "2"),
    }[layout]


def source_parallel_flags(layout: str):
    """FSDP *training* layout for the source-shard sweep.

    DP2 is the plain "trained on >1 GPU" case (TP1/PP1/EP1) — no extra flags.
    """
    return {
        "DP2": (),
        "TP2": ("--tensor-model-parallel-size", "2"),
        "PP2": ("--pipeline-model-parallel-size", "2"),
        "EP2": ("--expert-model-parallel-size", "2"),
    }[layout]


def results_root_override():
    """Honor the RESULTS_DIR override (keeps checkpoints + logs); else None -> tmp dir."""
    env = os.environ.get("RESULTS_DIR")
    return Path(env) if env else None
