# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Operand quantizers shared by every indexer top-k selector.

An indexer top-k selector scores ``sum_h w[h] * relu(q[h] . k)`` for every visible key and keeps
the ``topk`` best keys of each query row. The matched-precision reference selector (DeepGEMM
``fp8_fp4_mqa_logits`` plus an exact top-k) and the external LiteTopK kernels must consume
byte-identical operands; otherwise they disagree at near ties for reasons unrelated to the
selection. This module is the single implementation of those operands:

* :func:`quantize_indexer_fp8_rows`: E4M3 values with one float32 scale per row;
* :func:`quantize_indexer_mxfp4_rows`: E2M1 values with four UE8M0 group scales per 128-value
  row, computed by a batched Triton kernel on CUDA and by
  :func:`quantize_indexer_mxfp4_rows_reference` on other devices;
* :func:`fold_indexer_weights`: the float32 head weights that multiply the per-head scores.

Why these quantizers are not in :mod:`megatron.lite.primitive.quantization`: that package holds
the weight-only QAT and checkpoint formats, and its
:func:`~megatron.lite.primitive.quantization.mxfp4.quantize_mxfp4` is pinned bit for bit to the
ModelOpt checkpoint quantizer. The indexer MXFP4 operand follows a different, externally pinned
contract: the activation quantizer of the SGLang indexer, whose output DeepGEMM and the LiteTopK
kernels read. It differs from ``quantize_mxfp4`` in three ways:

1. Scale floor and clamp. The block scale is ``2 ** ceil(log2(max(amax / 6, 1e-4)))``, taken
   from the float32 bit pattern of ``max(amax / 6, 1e-4)``, and its biased UE8M0 exponent is
   clamped to ``[1, 254]``. Every block with ``amax / 6 < 1e-4``, including an all-zero block,
   therefore gets the scale ``2 ** -13``. ``quantize_mxfp4`` floors the exponent at ``-127``
   (E8M0 code 0) and does not clamp.
2. Sign of zero codes. The sign bit is set only on nonzero codes, so ``-0.0`` and negative values
   that round to zero encode as ``+0`` (nibble ``0x0``). ``quantize_mxfp4`` copies the sign bit
   and encodes them as ``-0`` (nibble ``0x8``).
3. Scale layout. A row has exactly 128 values and its four UE8M0 exponents are packed
   little-endian into one int32 (group ``g`` in bits ``[8g, 8g + 8)``), which is the layout the
   kernels read. ``quantize_mxfp4`` returns one ``float8_e8m0fnu`` scale per 32-value block for
   any row length that is a multiple of 32.

Both use the same E2M1 grid and the same tie rule: a magnitude is compared with the midpoints
0.25, 0.75, 1.25, 1.75, 2.5, 3.5 and 5 using a strict ``>``, so ties round down.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only environments use the torch reference.
    triton = None
    tl = None

__all__ = [
    "fold_indexer_weights",
    "quantize_indexer_fp8_rows",
    "quantize_indexer_mxfp4_rows",
    "quantize_indexer_mxfp4_rows_reference",
]

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)
_MXFP4_ROW = 128
_MXFP4_PACKED_ROW = _MXFP4_ROW // 2
_MXFP4_SOURCE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_MXFP4_ROWS_PER_PROGRAM = 32


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


def _check_mxfp4_source(value: Tensor) -> None:
    if value.ndim == 0 or value.shape[-1] != _MXFP4_ROW:
        raise ValueError(
            f"indexer MXFP4 quantization requires rows of {_MXFP4_ROW} values, "
            f"got shape {tuple(value.shape)}"
        )
    if value.dtype not in _MXFP4_SOURCE_DTYPES:
        raise TypeError(
            "indexer MXFP4 quantization requires bfloat16, float16 or float32 input, "
            f"got {value.dtype}"
        )


def quantize_indexer_mxfp4_rows_reference(value: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize 128-value rows to indexer MXFP4 with torch operations on any device.

    This is the reference of :func:`quantize_indexer_mxfp4_rows`; both return the same bits for
    finite inputs. See the module docstring for how the format differs from the checkpoint
    quantizer.

    Args:
        value: bfloat16, float16 or float32 tensor ``[..., 128]``.

    Returns:
        ``(packed, scales)``: int8 ``[..., 64]`` holding two E2M1 codes per byte (even element in
        the low nibble) and int32 ``[...]`` holding the four UE8M0 group exponents of each row.

    Raises:
        ValueError: If the last dimension is not 128.
        TypeError: If ``value`` has another dtype.
    """
    _check_mxfp4_source(value)
    rows = value.contiguous().reshape(-1, _MXFP4_ROW).float()
    grouped = rows.reshape(-1, 4, 32)
    scale = grouped.abs().amax(dim=-1).div(6.0).clamp_min(1.0e-4).contiguous()
    scale_bits = scale.view(torch.int32)
    exponent = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(torch.int32)
    exponent = exponent.clamp_(1, 254)
    power_of_two_scale = (exponent << 23).contiguous().view(torch.float32)

    normalized = grouped / power_of_two_scale.unsqueeze(-1)
    absolute = normalized.abs().clamp_max(6.0)
    code = torch.zeros_like(absolute, dtype=torch.uint8)
    for threshold in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code.add_(absolute > threshold)
    code |= ((normalized < 0) & (code != 0)).to(torch.uint8) << 3
    code = code.reshape(-1, _MXFP4_ROW)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous().view(torch.int8)
    packed_scale = (
        exponent[:, 0] | (exponent[:, 1] << 8) | (exponent[:, 2] << 16) | (exponent[:, 3] << 24)
    ).contiguous()
    return packed.view(*value.shape[:-1], _MXFP4_PACKED_ROW), packed_scale.view(value.shape[:-1])


if triton is not None:

    @triton.jit
    def _ceil_ue8m0_exp(x):
        bits = x.to(tl.int32, bitcast=True)
        exponent = (bits >> 23) & 0xFF
        mantissa = bits & 0x7FFFFF
        exponent += mantissa != 0
        return tl.minimum(tl.maximum(exponent, 1), 254)

    @triton.jit
    def _fp4_e2m1_code(x):
        absolute = tl.minimum(tl.abs(x), 6.0)
        code = (absolute > 0.25).to(tl.uint8)
        code += (absolute > 0.75).to(tl.uint8)
        code += (absolute > 1.25).to(tl.uint8)
        code += (absolute > 1.75).to(tl.uint8)
        code += (absolute > 2.5).to(tl.uint8)
        code += (absolute > 3.5).to(tl.uint8)
        code += (absolute > 5.0).to(tl.uint8)
        sign = ((x < 0) & (code != 0)).to(tl.uint8)
        return code | (sign << 3)

    @triton.jit
    def _quantize_mxfp4_rows_kernel(
        source, packed_out, scale_out, num_rows, BLOCK_ROWS: tl.constexpr
    ):
        rows = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        columns = tl.arange(0, 128)
        values = tl.load(
            source + rows[:, None] * 128 + columns[None, :], rows[:, None] < num_rows, 0
        ).to(tl.float32)
        grouped = tl.reshape(values, (BLOCK_ROWS, 4, 32))
        amax = tl.max(tl.abs(grouped), 2)
        exponent = _ceil_ue8m0_exp(tl.maximum(amax / 6.0, 1e-4))
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        codes = _fp4_e2m1_code(grouped / scale[:, :, None])
        low, high = tl.split(tl.reshape(codes, (BLOCK_ROWS, 64, 2)))
        tl.store(
            packed_out + rows[:, None] * 64 + tl.arange(0, 64)[None, :],
            low | (high << 4),
            rows[:, None] < num_rows,
        )
        shifts = tl.arange(0, 4) * 8
        tl.store(scale_out + rows, tl.sum(exponent << shifts[None, :], 1), rows < num_rows)


def quantize_indexer_mxfp4_rows(value: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize 128-value rows to the indexer MXFP4 operand format.

    Each row is split into four groups of 32 values. A group's UE8M0 scale is the smallest power
    of two not below ``max(amax / 6, 1e-4)`` (biased exponent clamped to ``[1, 254]``), and each
    value becomes the E2M1 code of ``value / scale``: magnitudes are compared with the midpoints
    ``0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5`` using a strict ``>``, and the sign bit is set only on
    nonzero codes. The module docstring lists the differences from the checkpoint quantizer.

    CUDA tensors use a batched Triton kernel (32 rows per program); other devices use
    :func:`quantize_indexer_mxfp4_rows_reference`. Both produce the same bits for finite inputs.
    Results for NaN inputs are unspecified.

    Args:
        value: bfloat16, float16 or float32 tensor ``[..., 128]``.

    Returns:
        ``(packed, scales)``: int8 ``[..., 64]`` holding two E2M1 codes per byte (even element in
        the low nibble) and int32 ``[...]`` holding the four UE8M0 group exponents of each row
        (group ``g`` in bits ``[8g, 8g + 8)``).

    Raises:
        ValueError: If the last dimension is not 128.
        TypeError: If ``value`` has another dtype.
        RuntimeError: If ``value`` is a CUDA tensor and Triton is not installed.
    """
    _check_mxfp4_source(value)
    if not value.is_cuda:
        return quantize_indexer_mxfp4_rows_reference(value)
    if triton is None:
        raise RuntimeError(
            "quantize_indexer_mxfp4_rows needs Triton for CUDA tensors; install Triton or call "
            "quantize_indexer_mxfp4_rows_reference"
        )
    rows = value.contiguous().reshape(-1, _MXFP4_ROW)
    num_rows = rows.shape[0]
    packed = torch.empty((num_rows, _MXFP4_PACKED_ROW), dtype=torch.int8, device=value.device)
    scales = torch.empty((num_rows,), dtype=torch.int32, device=value.device)
    if num_rows:
        grid = (triton.cdiv(num_rows, _MXFP4_ROWS_PER_PROGRAM),)
        with torch.cuda.device(value.device):
            _quantize_mxfp4_rows_kernel[grid](
                rows, packed, scales, num_rows, BLOCK_ROWS=_MXFP4_ROWS_PER_PROGRAM
            )
    return packed.view(*value.shape[:-1], _MXFP4_PACKED_ROW), scales.view(value.shape[:-1])


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
            ``weights``, or None for operands whose scales are applied inside the score kernel
            (MXFP4).

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
