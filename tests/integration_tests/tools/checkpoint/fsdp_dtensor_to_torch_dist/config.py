# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared arg vectors, env, and parallel-layout flags for the reverse-converter
end-to-end suite.

Holds the tiny-model training args, the deterministic env, the FSDP-train /
classic-resume flag sets, the reshard / source-shard parallel-flag maps, and the
training schedule (train iters, save interval, and which checkpoints to convert +
validate). The schedule is derived from the training length and is overridable via
environment variables (see below) — nothing about it is hard-coded into the tests.

Kept **stdlib-only** (no ``torch`` / ``megatron`` import) so importing it during
pytest collection is cheap.

Environment overrides (all optional):
  * ``MCORE_CHECKPOINT_E2E_TRAIN_ITERS``       — total FSDP training iters (default 100).
  * ``MCORE_CHECKPOINT_E2E_SAVE_INTERVAL``     — checkpoint save interval (default 20).
  * ``MCORE_CHECKPOINT_E2E_CONVERT_ITERS``     — comma-separated save iters to convert
                                                 (default: two interior, mid-decay saves).
  * ``MCORE_CHECKPOINT_E2E_RESUME_EXTRA_ITERS``— iters the classic resume runs past the
                                                 load point (default 3).
"""

import os
from pathlib import Path

# tests/integration_tests/tools/checkpoint/fsdp_dtensor_to_torch_dist/config.py
# -> repo root is five package levels up.
REPO_ROOT = Path(__file__).resolve().parents[5]
PRETRAIN_GPT = REPO_ROOT / "pretrain_gpt.py"
INSPECTOR = REPO_ROOT / "tools" / "checkpoint" / "checkpoint_inspector.py"

# --- tiny shared model config (identical across every family) ----------------
# 12L/512H bf16 deterministic mock-data run. Per-family ``num_layers`` and ``arch``
# are appended by the caller; caller-specific flags (--train-iters, --save,
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
    "--tokenizer-type", "NullTokenizer", "--vocab-size", "131073", "--mock-data",
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


# --- training schedule (derived, env-overridable) ----------------------------
def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


TRAIN_ITERS = _env_int("MCORE_CHECKPOINT_E2E_TRAIN_ITERS", 100)
SAVE_INTERVAL = _env_int("MCORE_CHECKPOINT_E2E_SAVE_INTERVAL", 20)
# How many iters the classic resume runs past its load point (>=1 so it produces a
# first post-load loss to compare).
RESUME_EXTRA_ITERS = _env_int("MCORE_CHECKPOINT_E2E_RESUME_EXTRA_ITERS", 3)


def save_iters(train_iters: int = None, save_interval: int = None):
    """The iterations at which the FSDP run writes checkpoints."""
    train_iters = TRAIN_ITERS if train_iters is None else train_iters
    save_interval = SAVE_INTERVAL if save_interval is None else save_interval
    return tuple(range(save_interval, train_iters + 1, save_interval))


def default_convert_iters(train_iters: int = None, save_interval: int = None):
    """Two interior, mid-decay save points to convert + validate.

    Drops the warmup-adjacent first save and the final save (whose next iter has no
    training reference to compare against), then takes the last two that remain, so
    the LR is still moving at each and a single lucky checkpoint can't pass the
    suite. Falls back gracefully for short schedules.
    """
    saves = list(save_iters(train_iters, save_interval))
    candidates = saves[1:-1] or saves[:-1] or saves
    return tuple(candidates[-2:])


def _parse_convert_iters(spec: str):
    return tuple(int(x) for x in spec.split(",") if x.strip())


_convert_spec = os.environ.get("MCORE_CHECKPOINT_E2E_CONVERT_ITERS")
# The checkpoints to convert + resume from. Each must be a real save point strictly
# inside the run (so its +1 iteration has an FSDP reference).
CONVERT_ITERS = _parse_convert_iters(_convert_spec) if _convert_spec else default_convert_iters()
for _it in CONVERT_ITERS:
    assert _it % SAVE_INTERVAL == 0 and 0 < _it < TRAIN_ITERS, (
        f"convert iter {_it} must be a save point (multiple of {SAVE_INTERVAL}) "
        f"strictly inside (0, {TRAIN_ITERS})"
    )


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
    """Honor the RESULTS_DIR override; else None -> use a tmp dir."""
    env = os.environ.get("RESULTS_DIR")
    return Path(env) if env else None
