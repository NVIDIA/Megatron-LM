# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Acceptance thresholds for MiniMax-M3 alignment (P0 decision 2026-09-10).

fp32 is the ONLY gate. bf16 numbers are recorded as sanity evidence, never used
to pass/fail structure or weight mapping. Rationale and measurements live in the
project notes (``PROGRESS.md``): HF bf16 is run-to-run deterministic (noise floor
0), while any kernel change flips 5-17% of MSA top-k rows in bf16 because the
indexer score margins have a heavy near-zero tail (7-13% of rows within one bf16
ulp). Cross-dtype (bf16 vs fp32) flips 59-66% of rows.

Usage::

    from thresholds import FP32, BF16, assert_fp32_env, check_discrete_sets

    assert_fp32_env()                          # TF32 off, or fp32 is ~1e-3 precision
    diffs = compare_dumps(mlite_dump, hf_dump)
    bad = [d for d in diffs if d.rel_to_max > FP32.module_rel]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Fp32Gate:
    """Hard gate. Applies to mlite-fp32 vs HF-fp32 (or vs fp64 ``msa_ref``)."""

    module_rel: float = 1e-5          # per-module max-abs / max|ref|
    grad_rel: float = 1e-5            # per-parameter gradient
    logits_logprob_delta: float = 5e-3
    logits_kl: float = 1e-4
    fp64_ref_abs: float = 1e-10       # msa_ref degenerate / gradcheck-grade
    # Discrete selections (block top-k, router top-k): sets must match, except rows whose
    # top-k margin is smaller than `margin_factor` x the measured score perturbation.
    discrete_margin_factor: float = 10.0
    discrete_unexplained_rows: int = 0


@dataclass(frozen=True)
class Bf16Sanity:
    """Recorded only. Same-dtype (mlite-bf16 vs HF-bf16) unless stated."""

    dense_module_rel: float = 2e-2         # ~2x HF bf16-vs-fp32 on dense layers (1e-3..1e-2 measured)
    kernel_vs_fp32_ref_rel: float = 1e-2   # cuDNN BSA / flex bf16 vs fp32 masked-dense, same indices (3e-3 measured)
    logits_logprob_delta: float = 2e-2     # HF eager-vs-sdpa: 1.3e-2
    logits_kl: float = 1e-3                # HF eager-vs-sdpa: 3e-4
    # Top-k set flip rate per MSA layer must not exceed HF eager-vs-sdpa (S=8192, Truncated-M3).
    topk_flip_rows_by_layer: tuple[float, ...] = (0.053, 0.111, 0.166)
    train_loss_rel: float = 1e-2           # 200-300 step curve, P4/P5


FP32 = Fp32Gate()
BF16 = Bf16Sanity()


def assert_fp32_env() -> None:
    """fp32 comparisons are meaningless with TF32 enabled anywhere.

    Transformer Engine's fp32 GEMMs go through cuBLASLt with TF32 by default and ignore the torch
    switches (measured: te.Linear fp32 rel err 2.8e-4 vs 6.6e-7 with ``NVIDIA_TF32_OVERRIDE=0``).
    The env var must be set before the process starts.
    """
    import os

    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("set NVIDIA_TF32_OVERRIDE=0 before launching fp32 alignment runs (TE uses TF32 otherwise)")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    if hasattr(torch.backends.cudnn, "conv"):
        try:
            torch.backends.cudnn.conv.fp32_precision = "ieee"
        except Exception:
            pass
    torch.set_float32_matmul_precision("highest")
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert torch.get_float32_matmul_precision() == "highest"


def check_discrete_sets(
    a_idx: torch.Tensor,
    b_idx: torch.Tensor,
    b_margin: torch.Tensor | None = None,
    score_perturbation: float | None = None,
    margin_factor: float = FP32.discrete_margin_factor,
) -> dict[str, float | int]:
    """Compare two top-k selections ``[..., k]`` as sets.

    ``b_margin`` (``[...]``, gap between the k-th kept and first rejected score of
    the reference) and ``score_perturbation`` (max |score_a - score_b| observed)
    let a mismatching row be *explained* when ``margin < margin_factor * perturbation``.
    Returns row mismatch fraction and the number of unexplained rows.
    """
    a_s, b_s = a_idx.sort(-1).values, b_idx.sort(-1).values
    mismatch = (a_s != b_s).any(-1)
    out: dict[str, float | int] = {
        "rows": int(mismatch.numel()),
        "mismatch_rows": int(mismatch.sum()),
        "mismatch_frac": mismatch.float().mean().item(),
    }
    if b_margin is not None and score_perturbation is not None:
        explained = b_margin < margin_factor * score_perturbation
        out["unexplained_rows"] = int((mismatch & ~explained).sum())
    return out
