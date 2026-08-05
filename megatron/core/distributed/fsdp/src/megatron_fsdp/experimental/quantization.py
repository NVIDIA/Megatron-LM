# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MXFP8 block quantization kernels for the minimal Megatron-FSDP path.

Implements MXFP8 E4M3 block-scaled quantization with 32-element blocks and
one bf16 scale per block (scale = block amax / E4M3 max normal). The encode is
bit-exact E4M3: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits, with
round-half-to-even and saturation at +/-448. Values below 2**-6 encode as
subnormals in 2**-9 steps.

Two quantization geometries are provided for model weights, matching
Transformer Engine's MXFP8 primary weights:

- Row-wise: 1x32 blocks along the last dimension (forward GEMM weight).
- Column-wise: 32x1 blocks along the first dimension (backward GEMM weight).
  Column-wise scales are global: each rank computes a partial per-block amax
  over its own rows, the ranks reduce-max, and every rank quantizes its rows
  with the merged scale grid.

Both payloads are ``(rows, cols)`` row-major uint8 data (TE's
``_columnwise_data`` has the same shape as ``_rowwise_data``); only the block
direction and the scale grid differ.

The scale dtype is bf16 rather than E8M0 for simplicity; the E4M3 payload
format is identical to MXFP8, so the numerics match block-scaled FP8 training
to within scale precision.

The production path delegates quantization to TE's verified
``cast_master_weights_to_fp8`` (see ``te_cast_master_weights_to_fp8`` and
``allocate_quantize_temp``); the kernels in this module are the CPU-testable
reference for the protocol. ``set_*_payload`` / ``clear_payloads`` rebind raw
payloads through TE's verified ``fp8_set_raw_data``.
"""

import math

import torch
import torch.distributed as dist

from ..mixed_precision import fp8_set_raw_data

E4M3_BLOCK_SIZE = 32
_E4M3_MAX_NORMAL = 448.0
_E4M3_SUBNORMAL_THRESHOLD = 2**-6
_E4M3_SUBNORMAL_STEP = 2**-9


def _round_to_e4m3_bits(values: torch.Tensor) -> torch.Tensor:
    """Round float32 values to E4M3 and return their bit patterns as uint8."""
    sign = (values < 0).to(torch.int64)
    a = values.abs()

    # Normal encode: a = m * 2**exp with m in [0.5, 1), so E = exp + 6 and
    # M = round((2m - 1) * 8). M can round up to 8, which carries into E.
    mantissa, exponent = torch.frexp(a)
    exp = exponent + 6
    mant = torch.round((2 * mantissa - 1) * 8).to(torch.int64)

    carry = mant == 8
    exp = torch.where(carry, exp + 1, exp)
    mant = torch.where(carry, 0, mant)

    # Subnormals (including zero): E = 0, value = M * 2**-9.
    subnormal = a < _E4M3_SUBNORMAL_THRESHOLD
    exp = torch.where(subnormal, 0, exp)
    mant = torch.where(
        subnormal, torch.round(a / _E4M3_SUBNORMAL_STEP).to(torch.int64).clamp(0, 7), mant
    )

    # E4M3 has no reserved exponent: E=15 is a normal exponent and the max
    # normal is 448 (E=15, M=6). Only NaN and overflow saturate.
    saturate = (exp > 15) | a.isnan() | a.isinf()
    exp = torch.where(saturate, 15, exp)
    mant = torch.where(saturate, 6, mant)

    bits = (sign << 7) | (exp << 3) | mant
    return bits.to(torch.uint8)


def _decode_e4m3_bits(bits: torch.Tensor) -> torch.Tensor:
    """Interpret E4M3 bit patterns as float32 values."""
    b = bits.to(torch.int64)
    sign = (b >> 7) & 1
    exp = (b >> 3) & 0xF
    mant = b & 0x7

    subnormal = exp == 0
    value = torch.where(
        subnormal,
        mant.to(torch.float32) * _E4M3_SUBNORMAL_STEP,
        (1 + mant.to(torch.float32) / 8) * torch.exp2((exp - 7).to(torch.float32)),
    )
    return torch.where(sign == 1, -value, value)


def _num_blocks(numel: int) -> int:
    return (numel + E4M3_BLOCK_SIZE - 1) // E4M3_BLOCK_SIZE


def pad_scale_inv(
    scale_inv: torch.Tensor, height_multiple: int, width_multiple: int
) -> torch.Tensor:
    """Zero-pad a scale-inverse grid to multiples of (height, width) multiples.

    Transformer Engine's MXFP8 tensors expect rowwise scale-inverse grids
    padded to multiples of ``(128, 4)`` and columnwise grids to ``(4, 128)``.
    """
    height, width = scale_inv.shape
    pad_height = (height_multiple - height % height_multiple) % height_multiple
    pad_width = (width_multiple - width % width_multiple) % width_multiple
    if pad_height == 0 and pad_width == 0:
        return scale_inv
    return torch.nn.functional.pad(scale_inv, (0, pad_width, 0, pad_height))


def _block_scales(amax: torch.Tensor) -> torch.Tensor:
    """Convert per-block amax to per-block decode scales (amax/448)."""
    scale = amax / _E4M3_MAX_NORMAL
    return torch.where(scale == 0, torch.ones_like(scale), scale)


def quantize_rowwise_2d(
    tensor: torch.Tensor, *, out_payload: torch.Tensor, out_scales: torch.Tensor
) -> None:
    """Row-wise MXFP8 E4M3 quantization of a 2D tensor (rows, cols).

    Blocks are 1x32 along the last dimension, so ``cols`` must be a multiple of
    32. The payload is the flat ``(rows, cols)`` uint8 data and the scales are
    the ``(rows, cols // 32)`` bf16 grid. Every row is quantized independently
    of the others, so a row shard can be quantized locally and concatenated
    across ranks to form the full payload.

    Args:
        tensor: 2D bf16/float16/float32 tensor with ``cols % 32 == 0``.
        out_payload: uint8 buffer with ``rows * cols`` elements.
        out_scales: bf16 buffer with ``rows * cols // 32`` elements.
    """
    if tensor.dim() != 2:
        raise ValueError(f"quantize_rowwise_2d expects a 2D tensor, got {tensor.dim()}D.")
    rows, cols = tensor.shape
    if cols % E4M3_BLOCK_SIZE != 0:
        raise ValueError(f"Row-wise MXFP8 requires cols % {E4M3_BLOCK_SIZE} == 0, got cols={cols}.")
    if out_payload.numel() != rows * cols or out_payload.dtype != torch.uint8:
        raise ValueError(
            f"Expected out_payload with {rows * cols} uint8 elements, "
            f"got shape {out_payload.shape} dtype {out_payload.dtype}."
        )
    if out_scales.numel() != rows * cols // E4M3_BLOCK_SIZE or out_scales.dtype != torch.bfloat16:
        raise ValueError(
            f"Expected out_scales with {rows * cols // E4M3_BLOCK_SIZE} bf16 elements, "
            f"got shape {out_scales.shape} dtype {out_scales.dtype}."
        )

    values = tensor.float()
    blocks_2d = values.view(rows * (cols // E4M3_BLOCK_SIZE), E4M3_BLOCK_SIZE)
    amax = torch.nan_to_num(blocks_2d.abs(), nan=0.0, posinf=0.0, neginf=0.0).amax(dim=1)
    scale = _block_scales(amax)
    quantized = _round_to_e4m3_bits((blocks_2d / scale.unsqueeze(1)).reshape(-1))
    out_payload.copy_(quantized)
    out_scales.copy_(scale.to(torch.bfloat16))


def compute_colwise_amax(tensor: torch.Tensor, row_offset: int, height: int) -> torch.Tensor:
    """Return the partial column-wise amax grid for a local row shard.

    Column-wise blocks are 32x1 over the first dimension of the full ``(height,
    cols)`` tensor. A rank holding rows ``[row_offset, row_offset + rows)``
    computes the per-column amax of its rows for every block it touches and
    returns the full ``(ceil(height/32), cols)`` grid with zeros elsewhere.
    Ranks reduce-max these grids before quantizing.

    Args:
        tensor: 2D local row shard ``(rows, cols)``.
        row_offset: First row of ``tensor`` in the full tensor.
        height: First dimension of the full tensor.

    Returns:
        float32 ``(ceil(height/32), cols)`` partial amax grid.
    """
    if tensor.dim() != 2:
        raise ValueError(f"compute_colwise_amax expects a 2D tensor, got {tensor.dim()}D.")
    rows, cols = tensor.shape
    grid = torch.zeros(
        (math.ceil(height / E4M3_BLOCK_SIZE), cols), dtype=torch.float32, device=tensor.device
    )
    if rows == 0:
        return grid

    pad_front = row_offset % E4M3_BLOCK_SIZE
    pad_back = (E4M3_BLOCK_SIZE - (row_offset + rows) % E4M3_BLOCK_SIZE) % E4M3_BLOCK_SIZE
    padded = torch.nn.functional.pad(
        torch.nan_to_num(tensor.float().abs(), nan=0.0, posinf=0.0, neginf=0.0),
        (0, 0, pad_front, pad_back),
    )
    blocks = padded.view(-1, E4M3_BLOCK_SIZE, cols).amax(dim=1)
    start_block = row_offset // E4M3_BLOCK_SIZE
    grid[start_block : start_block + blocks.shape[0]] = blocks
    return grid


def quantize_colwise_with_scales(
    tensor: torch.Tensor, scales: torch.Tensor, row_offset: int, *, out_payload: torch.Tensor
) -> None:
    """Quantize a local row shard with the global column-wise scales.

    Column-wise blocks are 32x1 over the first dimension; block ``b`` covers
    rows ``[32b, 32b + 32)``. Each local row is scaled by its block's scale
    before the E4M3 encode.

    Args:
        tensor: 2D local row shard ``(rows, cols)``.
        scales: bf16/float32 ``(ceil(height/32), cols)`` global scale grid.
        row_offset: First row of ``tensor`` in the full tensor.
        out_payload: uint8 buffer with ``rows * cols`` elements, receiving the
            ``(rows, cols)`` payload (same layout as the row-wise payload;
            only the block direction differs), so concatenating rank chunks
            in group order reconstructs the full ``(height, cols)``
            column-wise payload.
    """
    if tensor.dim() != 2:
        raise ValueError(f"quantize_colwise_with_scales expects a 2D tensor, got {tensor.dim()}D.")
    rows, cols = tensor.shape
    if out_payload.numel() != cols * rows or out_payload.dtype != torch.uint8:
        raise ValueError(
            f"Expected out_payload with {cols * rows} uint8 elements, "
            f"got shape {out_payload.shape} dtype {out_payload.dtype}."
        )

    block_indices = (torch.arange(rows, device=tensor.device) + row_offset) // E4M3_BLOCK_SIZE
    scale = scales[block_indices].to(torch.float32)
    quantized = _round_to_e4m3_bits((tensor.float() / scale).reshape(-1))
    out_payload.copy_(quantized)


def dequantize_rowwise_2d(
    payload: torch.Tensor, scales: torch.Tensor, *, out: torch.Tensor
) -> None:
    """Dequantize a row-wise (rows, cols) payload with its (rows, cols/32) scales."""
    if payload.dim() != 2:
        raise ValueError(f"dequantize_rowwise_2d expects a 2D payload, got {payload.dim()}D.")
    rows, cols = payload.shape
    if cols % E4M3_BLOCK_SIZE != 0:
        raise ValueError(f"Row-wise payload cols must be a multiple of {E4M3_BLOCK_SIZE}.")
    if scales.shape != (rows, cols // E4M3_BLOCK_SIZE):
        raise ValueError(
            f"Expected scales of shape ({rows}, {cols // E4M3_BLOCK_SIZE}), " f"got {scales.shape}."
        )
    values = _decode_e4m3_bits(payload.view(-1, E4M3_BLOCK_SIZE))
    values = values * scales.to(torch.float32).view(-1, 1)
    out.copy_(values.view(rows, cols).to(out.dtype))


def dequantize_colwise_chunk(
    payload: torch.Tensor, scales: torch.Tensor, row_offset: int, *, out: torch.Tensor
) -> None:
    """Dequantize a ``(rows, cols)`` column-wise payload chunk into bf16 rows.

    Args:
        payload: uint8 ``(rows, cols)`` chunk for one rank's rows.
        scales: bf16/float32 ``(ceil(height/32), cols)`` global scale grid.
        row_offset: First row of the chunk in the full tensor.
        out: bf16/float32 ``(rows, cols)`` destination.
    """
    if payload.dim() != 2:
        raise ValueError(f"dequantize_colwise_chunk expects a 2D payload, got {payload.dim()}D.")
    rows, cols = payload.shape
    values = _decode_e4m3_bits(payload)
    block_indices = (torch.arange(rows, device=payload.device) + row_offset) // E4M3_BLOCK_SIZE
    values = values * scales[block_indices].to(torch.float32)
    out.copy_(values.to(out.dtype))


def quantize_block_e4m3(
    tensor: torch.Tensor,
    *,
    out_payload: torch.Tensor | None = None,
    out_scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-quantize a contiguous 1D tensor to MXFP8 E4M3.

    Blocks of 32 elements; the final block is zero-padded. The per-block scale
    is ``block_amax / 448`` stored in bf16.

    Args:
        tensor: Contiguous 1D bf16/float16/float32 tensor to quantize.
        out_payload: Optional uint8 buffer of ``ceil(numel/32)*32`` elements.
        out_scales: Optional bf16 buffer of ``ceil(numel/32)`` elements.

    Returns:
        The (payload, scales) tensors. If an ``out_*`` buffer is given, the
        result is copied into it and that buffer is returned.
    """
    if tensor.dim() != 1 or not tensor.is_contiguous():
        raise ValueError("quantize_block_e4m3 expects a contiguous 1D tensor.")

    numel = tensor.numel()
    blocks = _num_blocks(numel)
    padded_numel = blocks * E4M3_BLOCK_SIZE

    values = tensor.float()
    if numel != padded_numel:
        padded_values = torch.zeros(padded_numel, dtype=torch.float32, device=tensor.device)
        padded_values[:numel] = values
        values = padded_values

    blocks_2d = values.view(blocks, E4M3_BLOCK_SIZE)
    # NaN/Inf values are saturated inside _round_to_e4m3_bits; exclude them
    # from the block amax so they do not poison the scale.
    amax = torch.nan_to_num(blocks_2d.abs(), nan=0.0, posinf=0.0, neginf=0.0).amax(dim=1)
    scale = amax / _E4M3_MAX_NORMAL
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = _round_to_e4m3_bits((blocks_2d / scale.unsqueeze(1)).reshape(-1))

    if out_payload is None:
        payload = quantized
    else:
        if out_payload.numel() != padded_numel or out_payload.dtype != torch.uint8:
            raise ValueError(
                f"Expected out_payload with {padded_numel} uint8 elements, "
                f"got shape {out_payload.shape} dtype {out_payload.dtype}."
            )
        out_payload.copy_(quantized)
        payload = out_payload

    if out_scales is None:
        scales = scale.to(torch.bfloat16)
    else:
        if out_scales.numel() != blocks or out_scales.dtype != torch.bfloat16:
            raise ValueError(
                f"Expected out_scales with {blocks} bf16 elements, "
                f"got shape {out_scales.shape} dtype {out_scales.dtype}."
            )
        out_scales.copy_(scale.to(torch.bfloat16))
        scales = out_scales
    return payload, scales


def dequantize_block_e4m3(
    payload: torch.Tensor, scales: torch.Tensor, *, out: torch.Tensor
) -> None:
    """Dequantize a 1D block stream into ``out``.

    Args:
        payload: uint8 E4M3 payload; numel must be a multiple of 32.
        scales: bf16 per-block scales.
        out: Destination 1D contiguous tensor; its numel must not exceed the
            payload's element count (padding at the tail is dropped).
    """
    if payload.numel() % E4M3_BLOCK_SIZE != 0:
        raise ValueError("dequantize_block_e4m3 payload must be a multiple of 32 elements.")
    blocks = payload.numel() // E4M3_BLOCK_SIZE
    if scales.numel() != blocks:
        raise ValueError(
            f"Expected {blocks} scales for {payload.numel()} payload elements, "
            f"got {scales.numel()}."
        )
    if out.dim() != 1 or not out.is_contiguous():
        raise ValueError("dequantize_block_e4m3 expects a contiguous 1D out tensor.")
    if out.numel() > payload.numel():
        raise ValueError(
            f"Out tensor with {out.numel()} elements exceeds payload "
            f"capacity of {payload.numel()} elements."
        )

    values = _decode_e4m3_bits(payload.view(blocks, E4M3_BLOCK_SIZE))
    values = values * scales.to(torch.float32).unsqueeze(1)
    out.copy_(values.reshape(-1)[: out.numel()].to(out.dtype))


def set_rowwise_payload(tensor: torch.Tensor, data: torch.Tensor) -> None:
    """Bind the raw row-wise fp8 payload onto a TE quantized tensor."""
    fp8_set_raw_data(tensor, data, set_transpose=False)


def set_columnwise_payload(tensor: torch.Tensor, data: torch.Tensor) -> None:
    """Bind the raw column-wise (backward-GEMM) fp8 payload onto a TE quantized tensor."""
    fp8_set_raw_data(tensor, data, set_transpose=True)


def clear_payloads(tensor: torch.Tensor) -> None:
    """Detach a TE quantized tensor from its raw payload storage.

    The payloads are only read while the tensor is installed for compute
    (between unshard and reshard); between unshards they rest detached while
    the sharded payloads live in the group's rowwise/colwise DBuffers.
    """
    if hasattr(tensor, "_rowwise_data"):
        tensor._rowwise_data = None
    elif hasattr(tensor, "_data"):
        tensor._data = None
    if hasattr(tensor, "_columnwise_data"):
        tensor._columnwise_data = None


_TE_CAST_MASTER_WEIGHTS_TO_FP8: bool | None = None


def te_cast_master_weights_to_fp8():
    """Return TE's ``cast_master_weights_to_fp8`` when importable, else None."""
    global _TE_CAST_MASTER_WEIGHTS_TO_FP8
    if _TE_CAST_MASTER_WEIGHTS_TO_FP8 is None:
        try:
            from transformer_engine.pytorch.tensor.utils import cast_master_weights_to_fp8

            _TE_CAST_MASTER_WEIGHTS_TO_FP8 = cast_master_weights_to_fp8
        except ImportError:
            _TE_CAST_MASTER_WEIGHTS_TO_FP8 = False
    return _TE_CAST_MASTER_WEIGHTS_TO_FP8 or None


def allocate_quantize_temp(
    tensor: torch.Tensor, height: int, width: int, device: torch.device
) -> torch.Tensor:
    """Allocate a full-size temporary MXFP8Tensor for ``cast_master_weights_to_fp8``.

    TE quantizes into the temp's full-size row-wise and column-wise raw
    payloads, writing only the shard slices at the provided offsets; the
    caller copies those slices out and releases the temp. The temp's
    scale-inverse grids alias ``tensor``'s grids so TE fills them in place.
    ``height``/``width``/``device`` are the logical tensor geometry (TE's
    ``MXFP8Tensor.shape``/``device`` raise when the raw payloads are
    detached, so callers pass them in).
    """
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor

    return MXFP8Tensor(
        shape=(height, width),
        dtype=tensor.dtype,
        rowwise_data=torch.empty((height, width), dtype=torch.uint8, device=device),
        rowwise_scale_inv=tensor._rowwise_scale_inv,
        columnwise_data=torch.empty((height, width), dtype=torch.uint8, device=device),
        columnwise_scale_inv=tensor._columnwise_scale_inv,
        fp8_dtype=tensor._fp8_dtype,
        quantizer=MXFP8Quantizer(fp8_dtype=tensor._fp8_dtype),
        with_gemm_swizzled_scales=False,
        device=device,
    )
