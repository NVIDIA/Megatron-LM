# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Operand quantizers shared by every indexer top-k selector.

An indexer top-k selector scores ``sum_h w[h] * relu(q[h] . k)`` for every visible key and keeps
the ``topk`` best keys of each query row. The matched-precision reference selector (DeepGEMM
``fp8_fp4_mqa_logits`` plus an exact top-k) and the external LiteTopK kernels must consume
byte-identical operands; otherwise they disagree at near ties for reasons unrelated to the
selection. This module is the single implementation of those operands:

* :func:`quantize_indexer_fp8_rows`: E4M3 values with one float32 scale per row;
* :func:`fold_indexer_weights`: the float32 head weights that multiply the per-head scores.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["fold_indexer_weights", "quantize_indexer_fp8_rows"]

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)


def quantize_indexer_fp8_rows(value: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize the last dimension to E4M3 and return a positive float32 row scale.

    ``scale = amax(|value|) / 448`` over the last dimension, floored at the smallest normal
    float32 so that all-zero rows keep a finite scale, and
    ``data = clamp(value / scale, -448, 448)`` cast to E4M3. All arithmetic is float32 on the
    device of ``value`` with PyTorch's device semantics: CUDA divides by the Python scalar 448 by
    multiplying with its float32 reciprocal, so CPU and CUDA scales can differ in the last bit.
    Each device is deterministic.

    Args:
        value: Floating-point tensor ``[..., D]``, typically bfloat16 indexer queries or keys.

    Returns:
        ``(data, scale)``: contiguous ``float8_e4m3fn`` ``[..., D]`` and float32 ``[...]``.
    """
    value_fp32 = value.float()
    scale = value_fp32.abs().amax(dim=-1).div(_FP8_MAX)
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    quantized = value_fp32.div(scale.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX)
    return quantized.to(_FP8_DTYPE).contiguous(), scale.contiguous()


def fold_indexer_weights(
    weights: Tensor, *, softmax_scale: float, q_scale: Tensor | None
) -> Tensor:
    """Fold the score scales into the float32 per-head indexer weights.

    Computes ``weights.float()``, multiplies it by ``softmax_scale`` unless that is exactly 1.0,
    then by ``q_scale`` when one is given. The order of the two float32 multiplications is part
    of the contract: the reference selector and the LiteTopK plugin must receive bit-identical
    weights.

    The semantics equal those of upstream ``dsa_kernels.indexer_topk``, which also moves the
    indexer softmax scale onto the weights (``relu(c * x) = c * relu(x)`` for ``c > 0``). The bits
    do not: upstream rounds the scaled weights back to the weight dtype (bfloat16) before
    scoring, while the folded weights here stay float32.

    Args:
        weights: Per-head indexer weights ``[..., H]``.
        softmax_scale: Positive score scale, usually ``head_dim ** -0.5``; pass 1.0 when
            ``weights`` already include it.
        q_scale: Per-row query scales from :func:`quantize_indexer_fp8_rows` with the shape of
            ``weights``, or None to fold only ``softmax_scale``.

    Returns:
        Contiguous float32 tensor shaped like ``weights``. It may share storage with ``weights``
        when no multiplication is needed and ``weights`` is already contiguous float32.

    Raises:
        ValueError: If ``softmax_scale`` is not positive or ``q_scale`` has another shape.
    """
    if not softmax_scale > 0:
        raise ValueError(f"softmax_scale must be positive, got {softmax_scale}")
    if q_scale is not None and q_scale.shape != weights.shape:
        raise ValueError(
            f"q_scale shape {tuple(q_scale.shape)} must equal weights shape "
            f"{tuple(weights.shape)}"
        )
    folded = weights.float()
    if softmax_scale != 1.0:
        folded = folded.mul(float(softmax_scale))
    if q_scale is not None:
        folded = folded.mul(q_scale)
    return folded.contiguous()
