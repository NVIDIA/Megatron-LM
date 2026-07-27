# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Quantization utilities for DSA indexer inputs.

The current SM100 path uses Transformer Engine for the expensive
BF16-to-MXFP8 conversion. Its
rowwise E8M0 scales are logical ``(flattened_rows, head_dim / 32)`` scales.
The compact cuDNN Indexer instead consumes the scales in its THD/GQA-aware
Blackwell 128x4 physical layout. A small Triton kernel performs only that
byte reordering and writes every padded byte into caller-owned storage.

Precision-specific helpers remain explicitly named so this module can also
host SM90 FP8 and future indexer quantization paths without ambiguity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    HAVE_TE_MXFP8 = True
except ImportError:
    tex = None
    MXFP8Quantizer = None
    HAVE_TE_MXFP8 = False

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def indexer_mxfp8_scale_shape(
    batch_size: int, max_seqlen: int, num_heads: int, head_dim: int, sf_vec_size: int = 32
) -> tuple[int, int, int]:
    """Return the physical BSHD E8M0 scale shape expected by the DSA kernel."""
    if min(batch_size, max_seqlen, num_heads, head_dim) <= 0:
        raise ValueError("MXFP8 indexer scale dimensions must all be positive")
    if sf_vec_size != 32:
        raise ValueError(f"MXFP8 indexer only supports sf_vec_size=32, got {sf_vec_size}")
    if head_dim % sf_vec_size != 0:
        raise ValueError(f"MXFP8 indexer head_dim ({head_dim}) must be divisible by {sf_vec_size}")

    scale_groups = head_dim // sf_vec_size
    packed_rows = _ceil_div(max_seqlen * num_heads, 128) * 128
    packed_groups = _ceil_div(scale_groups, 4) * 4
    return batch_size, packed_rows, packed_groups


def make_indexer_mxfp8_scale_cu_seqlens(cu_seqlens: Tensor, num_heads: int) -> Tensor:
    """Return compact THD scale prefixes with independently padded sequences.

    ``num_heads`` is the number of logical scale rows per token. Each returned
    sequence span is minimally padded so that its packed scale rows are a
    multiple of the Blackwell 128-row atom.
    """
    if (
        cu_seqlens.dtype != torch.int32
        or cu_seqlens.ndim != 1
        or cu_seqlens.numel() < 2
        or not cu_seqlens.is_contiguous()
    ):
        raise ValueError("cu_seqlens must be a contiguous int32 tensor with at least two elements")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")

    token_alignment = 128 // math.gcd(128, num_heads)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    padded_lengths = ((lengths + token_alignment - 1) // token_alignment) * token_alignment
    out = torch.zeros_like(cu_seqlens)
    torch.cumsum(padded_lengths, dim=0, out=out[1:])
    return out


def indexer_mxfp8_thd_scale_shape(
    padded_tokens: int, num_heads: int, head_dim: int, sf_vec_size: int = 32
) -> tuple[int, int, int]:
    """Return the compact THD E8M0 scale shape expected by the DSA kernel."""
    if min(padded_tokens, num_heads, head_dim) <= 0:
        raise ValueError("MXFP8 indexer scale dimensions must all be positive")
    if sf_vec_size != 32:
        raise ValueError(f"MXFP8 indexer only supports sf_vec_size=32, got {sf_vec_size}")
    if head_dim % sf_vec_size != 0:
        raise ValueError(f"MXFP8 indexer head_dim ({head_dim}) must be divisible by {sf_vec_size}")

    packed_rows = padded_tokens * num_heads
    if packed_rows % 128 != 0:
        raise ValueError("THD MXFP8 scale rows must be a multiple of 128")
    scale_groups = head_dim // sf_vec_size
    packed_groups = _ceil_div(scale_groups, 4) * 4
    return 1, packed_rows, packed_groups


@dataclass
class IndexerMXFP8QuantizationBuffers:
    """Preallocated TE destination and optional input padding for one tensor."""

    quantizer: Any
    quantized: Any
    data: Tensor
    logical_scale: Tensor
    padded_input: Tensor | None
    input_shape: tuple[int, ...]
    num_rows: int

    def matches(self, x: Tensor) -> bool:
        """Return whether these buffers can quantize ``x`` without allocation."""
        return all(
            (
                tuple(x.shape) == self.input_shape,
                x.device == self.data.device,
                x.dtype == torch.bfloat16,
                x.is_contiguous(),
                self.data.dtype == torch.float8_e4m3fn,
                tuple(self.data.shape) == self.input_shape,
                self.data.is_contiguous(),
                self.logical_scale.device == x.device,
                self.logical_scale.dtype == torch.uint8,
                self.logical_scale.is_contiguous(),
                self.padded_input is None
                or (
                    self.padded_input.device == x.device
                    and self.padded_input.dtype == x.dtype
                    and self.padded_input.is_contiguous()
                ),
            )
        )


def create_indexer_mxfp8_quantization_buffers(x: Tensor) -> IndexerMXFP8QuantizationBuffers:
    """Allocate an unswizzled TE MXFP8 destination outside CUDA graph capture."""
    if not HAVE_TE_MXFP8:
        raise RuntimeError("MXFP8 indexer quantization requires Transformer Engine MXFP8 support")
    if not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.ndim < 2:
        raise ValueError("MXFP8 indexer input must be a contiguous CUDA BF16 tensor with ndim >= 2")

    head_dim = x.shape[-1]
    if head_dim % 32 != 0:
        raise ValueError(f"MXFP8 indexer head_dim ({head_dim}) must be divisible by 32")
    num_rows = x.numel() // head_dim
    padded_rows = _ceil_div(num_rows, 32) * 32

    # TE requires the product of leading dimensions to be divisible by 32.
    # Avoid an input copy for the common aligned Q/K shapes. Only small,
    # ragged K tensors need the padded staging buffer.
    padded_input = None
    quantized_shape: tuple[int, ...]
    if padded_rows == num_rows:
        quantized_shape = tuple(x.shape)
    else:
        quantized_shape = (padded_rows, head_dim)
        padded_input = torch.zeros(quantized_shape, dtype=x.dtype, device=x.device)

    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    # The Indexer has its own THD/GQA packing. TE must expose logical scales.
    quantizer.optimize_for_gemm = False
    quantized = quantizer.make_empty(quantized_shape, dtype=x.dtype, device=x.device)
    data = (
        quantized._rowwise_data.view(torch.float8_e4m3fn)
        .reshape(padded_rows, head_dim)[:num_rows]
        .reshape(x.shape)
    )
    logical_scale = quantized._rowwise_scale_inv
    return IndexerMXFP8QuantizationBuffers(
        quantizer=quantizer,
        quantized=quantized,
        data=data,
        logical_scale=logical_scale,
        padded_input=padded_input,
        input_shape=tuple(x.shape),
        num_rows=num_rows,
    )


@triton.jit
def _pack_indexer_mxfp8_scale_bshd_kernel(
    out_ptr,
    logical_scale_ptr,
    seqlen,
    total_out_bytes,
    NUM_HEADS: tl.constexpr,
    REAL_GROUPS: tl.constexpr,
    LOGICAL_PADDED_GROUPS: tl.constexpr,
    PADDED_ROWS: tl.constexpr,
    PADDED_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pack logical BSHD TE scale bytes into the Indexer physical layout."""
    out_linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = out_linear < total_out_bytes
    bytes_per_batch = PADDED_ROWS * PADDED_GROUPS
    batch = out_linear // bytes_per_batch
    logical_linear = out_linear - batch * bytes_per_batch
    packed_row = logical_linear // PADDED_GROUPS
    scale_group = logical_linear - packed_row * PADDED_GROUPS
    local_token = packed_row // NUM_HEADS
    head = packed_row - local_token * NUM_HEADS

    global_token = batch * seqlen + local_token
    valid = in_bounds & (local_token < seqlen) & (scale_group < REAL_GROUPS)
    source_row = global_token * NUM_HEADS + head
    source_offset = source_row * LOGICAL_PADDED_GROUPS + scale_group
    value = tl.load(logical_scale_ptr + source_offset, mask=valid, other=0)

    # Map logical (row, scale_group) to NVIDIA's packed F8_128x4 layout.
    tile_idx = (packed_row // 128) * (PADDED_GROUPS // 4) + scale_group // 4
    physical_offset = (
        batch * bytes_per_batch
        + tile_idx * 512
        + (packed_row % 32) * 16
        + ((packed_row % 128) // 32) * 4
        + scale_group % 4
    )
    tl.store(out_ptr + physical_offset, value, mask=in_bounds)


@triton.jit
def _pack_indexer_mxfp8_scale_thd_kernel(
    out_ptr,
    logical_scale_ptr,
    cu_seqlens_ptr,
    cu_seqlens_scale_padded_ptr,
    NUM_HEADS: tl.constexpr,
    REAL_GROUPS: tl.constexpr,
    LOGICAL_PADDED_GROUPS: tl.constexpr,
    PADDED_GROUPS: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    """Pack logical THD TE scale bytes into concatenated padded scale spans."""
    tile_idx = tl.program_id(0)
    scale_tiles = PADDED_GROUPS // 4
    mn_tile = tile_idx // scale_tiles
    scale_tile = tile_idx - mn_tile * scale_tiles
    packed_row_start = mn_tile * 128

    batch_lo = 0
    batch_hi = BATCH_SIZE
    for _ in range(SEARCH_STEPS):
        batch_mid = (batch_lo + batch_hi) // 2
        next_packed_row = tl.load(cu_seqlens_scale_padded_ptr + batch_mid + 1) * NUM_HEADS
        in_lower_half = packed_row_start < next_packed_row
        batch_hi = tl.where(in_lower_half, batch_mid, batch_hi)
        batch_lo = tl.where(in_lower_half, batch_lo, batch_mid + 1)
    batch = tl.minimum(batch_lo, BATCH_SIZE - 1)

    logical = tl.arange(0, 512)
    row_in_tile = logical // 4
    packed_row = packed_row_start + row_in_tile
    scale_group = scale_tile * 4 + logical % 4

    scale_row_start = tl.load(cu_seqlens_scale_padded_ptr + batch) * NUM_HEADS
    local_row = packed_row - scale_row_start
    local_token = local_row // NUM_HEADS
    head = local_row - local_token * NUM_HEADS
    seq_start = tl.load(cu_seqlens_ptr + batch)
    seq_end = tl.load(cu_seqlens_ptr + batch + 1)
    seq_len = seq_end - seq_start

    valid = (local_token < seq_len) & (scale_group < REAL_GROUPS)
    source_row = (seq_start + local_token) * NUM_HEADS + head
    source_offset = source_row * LOGICAL_PADDED_GROUPS + scale_group
    value = tl.load(logical_scale_ptr + source_offset, mask=valid, other=0)

    physical_offset = (
        tile_idx * 512 + (row_in_tile % 32) * 16 + ((row_in_tile % 128) // 32) * 4 + logical % 4
    )
    tl.store(out_ptr + physical_offset, value)


def pack_indexer_mxfp8_scale(
    logical_scale: Tensor,
    out_scale: Tensor,
    *,
    num_heads: int,
    real_groups: int,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_scale_padded: Tensor | None = None,
    seqlen: int = 0,
) -> Tensor:
    """Pack TE logical E8M0 bytes into caller-owned Indexer scale storage."""
    if not HAVE_TRITON:
        raise RuntimeError("MXFP8 indexer scale packing requires Triton")
    if (
        logical_scale.dtype != torch.uint8
        or logical_scale.ndim != 2
        or not logical_scale.is_cuda
        or not logical_scale.is_contiguous()
    ):
        raise ValueError("logical_scale must be a contiguous CUDA uint8 matrix")
    if (
        out_scale.dtype != torch.float8_e8m0fnu
        or out_scale.ndim != 3
        or out_scale.device != logical_scale.device
        or not out_scale.is_contiguous()
    ):
        raise ValueError("out_scale must be contiguous CUDA E8M0 storage on logical_scale.device")
    if num_heads <= 0 or real_groups <= 0:
        raise ValueError("num_heads and real_groups must be positive")
    if out_scale.shape[1] % 128 != 0 or out_scale.shape[2] % 4 != 0:
        raise ValueError("out_scale must have Blackwell 128x4 padded dimensions")
    if real_groups > logical_scale.shape[1] or real_groups > out_scale.shape[2]:
        raise ValueError("scale group count exceeds logical or packed scale storage")

    is_thd = cu_seqlens is not None
    if is_thd:
        if (
            cu_seqlens.device != logical_scale.device
            or cu_seqlens.dtype != torch.int32
            or cu_seqlens.ndim != 1
            or cu_seqlens.numel() < 2
            or not cu_seqlens.is_contiguous()
        ):
            raise ValueError("cu_seqlens must be contiguous CUDA int32 storage")
        if (
            cu_seqlens_scale_padded is None
            or cu_seqlens_scale_padded.device != logical_scale.device
            or cu_seqlens_scale_padded.dtype != torch.int32
            or cu_seqlens_scale_padded.ndim != 1
            or cu_seqlens_scale_padded.numel() != cu_seqlens.numel()
            or not cu_seqlens_scale_padded.is_contiguous()
        ):
            raise ValueError(
                "cu_seqlens_scale_padded must be contiguous CUDA int32 storage "
                "matching cu_seqlens"
            )
        if out_scale.shape[0] != 1 or out_scale.shape[1] % num_heads != 0:
            raise ValueError("THD out_scale must have one L dimension and whole-token rows")
    elif seqlen <= 0:
        raise ValueError("BSHD scale packing requires a positive seqlen")
    elif cu_seqlens_scale_padded is not None:
        raise ValueError("cu_seqlens_scale_padded is only valid for THD scale packing")

    if is_thd:
        _pack_indexer_mxfp8_scale_thd_kernel[(out_scale.numel() // 512,)](
            out_scale.view(torch.uint8),
            logical_scale,
            cu_seqlens,
            cu_seqlens_scale_padded,
            NUM_HEADS=num_heads,
            REAL_GROUPS=real_groups,
            LOGICAL_PADDED_GROUPS=logical_scale.shape[1],
            PADDED_GROUPS=out_scale.shape[2],
            BATCH_SIZE=cu_seqlens.numel() - 1,
            SEARCH_STEPS=(cu_seqlens.numel() - 1).bit_length(),
        )
    else:
        total_out_bytes = out_scale.numel()
        block = 256
        _pack_indexer_mxfp8_scale_bshd_kernel[(triton.cdiv(total_out_bytes, block),)](
            out_scale.view(torch.uint8),
            logical_scale,
            seqlen,
            total_out_bytes,
            NUM_HEADS=num_heads,
            REAL_GROUPS=real_groups,
            LOGICAL_PADDED_GROUPS=logical_scale.shape[1],
            PADDED_ROWS=out_scale.shape[1],
            PADDED_GROUPS=out_scale.shape[2],
            BLOCK=block,
        )
    return out_scale


def quantize_indexer_mxfp8(
    x: Tensor,
    *,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_scale_padded: Tensor | None = None,
    buffers: IndexerMXFP8QuantizationBuffers | None = None,
    out_scale: Tensor | None = None,
    sf_vec_size: int = 32,
) -> tuple[Tensor, Tensor]:
    """Quantize BF16 BSHD/THD input with TE and pack scales for the Indexer.

    BSHD Q is ``(B, S, H, D)`` and K is ``(B, S, D)``. Packed THD Q is
    ``(T, H, D)`` and K is ``(T, D)``. Supplying ``buffers`` and
    ``out_scale`` makes the operation allocation-free and capture-safe.
    """
    if sf_vec_size != 32:
        raise ValueError(f"MXFP8 indexer only supports sf_vec_size=32, got {sf_vec_size}")
    if not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("MXFP8 indexer input must be a contiguous CUDA BF16 tensor")

    is_thd = cu_seqlens is not None
    if is_thd:
        if x.ndim == 3:
            _, num_heads, head_dim = x.shape
        elif x.ndim == 2:
            _, head_dim = x.shape
            num_heads = 1
        else:
            raise ValueError(f"Packed THD MXFP8 input must be 2D or 3D, got shape {x.shape}")
        if cu_seqlens_scale_padded is None:
            raise ValueError("Packed THD MXFP8 quantization requires padded scale cu_seqlens")
        seqlen = 0
    else:
        if x.ndim == 4:
            batch_size, seqlen, num_heads, head_dim = x.shape
        elif x.ndim == 3:
            batch_size, seqlen, head_dim = x.shape
            num_heads = 1
        else:
            raise ValueError(f"BSHD MXFP8 input must be 3D or 4D, got shape {x.shape}")
        if cu_seqlens_scale_padded is not None:
            raise ValueError("Padded scale cu_seqlens are only valid for packed THD input")

    if buffers is None:
        buffers = create_indexer_mxfp8_quantization_buffers(x)
    elif not buffers.matches(x):
        raise ValueError("MXFP8 quantization buffers do not match the input tensor")

    source = x
    if buffers.padded_input is not None:
        buffers.padded_input[: buffers.num_rows].copy_(x.reshape(buffers.num_rows, head_dim))
        source = buffers.padded_input
    buffers.quantizer.update_quantized(source, buffers.quantized)

    if is_thd:
        if out_scale is None and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("THD MXFP8 CUDA graph capture requires preallocated scale storage")
        expected_scale_shape = (
            indexer_mxfp8_thd_scale_shape(
                int(cu_seqlens_scale_padded[-1].item()), num_heads, head_dim, sf_vec_size
            )
            if out_scale is None
            else None
        )
    else:
        expected_scale_shape = indexer_mxfp8_scale_shape(
            batch_size, seqlen, num_heads, head_dim, sf_vec_size
        )
    if out_scale is None:
        assert expected_scale_shape is not None
        out_scale = torch.empty(expected_scale_shape, dtype=torch.float8_e8m0fnu, device=x.device)
    elif is_thd and (
        out_scale.device != x.device
        or out_scale.dtype != torch.float8_e8m0fnu
        or out_scale.ndim != 3
        or out_scale.shape[0] != 1
        or out_scale.shape[1] % 128 != 0
        or out_scale.shape[2] != _ceil_div(head_dim // sf_vec_size, 4) * 4
        or not out_scale.is_contiguous()
    ):
        raise ValueError(
            "THD out_scale must be contiguous E8M0 storage with shape "
            "(1, multiple_of_128, padded_scale_groups)"
        )
    elif not is_thd and (
        out_scale.device != x.device
        or out_scale.dtype != torch.float8_e8m0fnu
        or tuple(out_scale.shape) != expected_scale_shape
        or not out_scale.is_contiguous()
    ):
        raise ValueError(
            f"out_scale must be contiguous E8M0 storage with shape {expected_scale_shape}"
        )

    pack_indexer_mxfp8_scale(
        buffers.logical_scale,
        out_scale,
        num_heads=num_heads,
        real_groups=head_dim // sf_vec_size,
        cu_seqlens=cu_seqlens,
        cu_seqlens_scale_padded=cu_seqlens_scale_padded,
        seqlen=seqlen,
    )
    return buffers.data, out_scale


__all__ = [
    "HAVE_TE_MXFP8",
    "HAVE_TRITON",
    "IndexerMXFP8QuantizationBuffers",
    "create_indexer_mxfp8_quantization_buffers",
    "indexer_mxfp8_scale_shape",
    "indexer_mxfp8_thd_scale_shape",
    "make_indexer_mxfp8_scale_cu_seqlens",
    "pack_indexer_mxfp8_scale",
    "quantize_indexer_mxfp8",
]
