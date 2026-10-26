# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""TE MXFP8 indexer quantization with logical or packed THD/GQA scales.

CSA2 prepares both SBHD and THD projections as packed rows. The cuDNN scorer
consumes swizzled per-sequence scales; indexed Reindex gathers logical scales.
Quantization is used only for discrete selection, not indexer supervision.
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


def indexer_mxfp8_thd_scale_capacity(total_tokens: int, num_sequences: int, num_heads: int) -> int:
    """Return the maximum padded-token scale capacity for static THD geometry."""
    if min(total_tokens, num_sequences, num_heads) <= 0:
        raise ValueError("THD MXFP8 scale capacity dimensions must all be positive")

    token_alignment = 128 // math.gcd(128, num_heads)
    positive_sequences = min(total_tokens, num_sequences)
    return (
        (total_tokens + positive_sequences * (token_alignment - 1)) // token_alignment
    ) * token_alignment


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


def _update_indexer_mxfp8(x: Tensor, buffers: IndexerMXFP8QuantizationBuffers) -> None:
    """Write TE's rowwise data and logical scales, including ragged input padding."""
    source = x
    if buffers.padded_input is not None:
        buffers.padded_input[: buffers.num_rows].copy_(x.reshape(buffers.num_rows, x.shape[-1]))
        source = buffers.padded_input
    buffers.quantizer.update_quantized(source, buffers.quantized)


@torch.no_grad()
def quantize_indexer_mxfp8_logical(x: Tensor) -> tuple[Tensor, Tensor]:
    """Return TE FP8 data and logical uint8 E8M0 scales for indexed dot products.

    Scales have shape ``[flattened_rows, head_dim / 32]``. Unlike the cuDNN
    dense scorer, the candidate scorer gathers logical rows directly, so it
    needs neither the Blackwell physical scale permutation nor dequantized
    Q/K tensors. This quantizer is only for discrete index selection.
    """
    if x.numel() == 0:
        return (
            torch.empty_like(x, dtype=torch.float8_e4m3fn),
            torch.empty((0, x.shape[-1] // 32), device=x.device, dtype=torch.uint8),
        )
    x = x.contiguous()
    buffers = create_indexer_mxfp8_quantization_buffers(x)
    _update_indexer_mxfp8(x, buffers)
    groups = x.shape[-1] // 32
    return buffers.data, buffers.logical_scale[: buffers.num_rows, :groups]


@torch.no_grad()
def quantize_indexer_mxfp8(
    x: Tensor,
    *,
    cu_seqlens: Tensor,
    cu_seqlens_scale_padded: Tensor,
    out_scale: Tensor,
    buffers: IndexerMXFP8QuantizationBuffers | None = None,
) -> tuple[Tensor, Tensor]:
    """Quantize packed rows and write caller-sized cuDNN scales without host reads."""
    if x.ndim not in (2, 3):
        raise ValueError("Packed MXFP8 projections must be [T,D] or [T,H,D]")
    if buffers is None:
        buffers = create_indexer_mxfp8_quantization_buffers(x)
    elif not buffers.matches(x):
        raise ValueError("MXFP8 buffers do not match the input")
    heads = x.shape[-2] if x.ndim == 3 else 1
    if (
        out_scale.dtype != torch.float8_e8m0fnu
        or out_scale.device != x.device
        or out_scale.ndim != 3
        or out_scale.shape[0] != 1
        or out_scale.shape[1] % 128
        or out_scale.shape[2] != _ceil_div(x.shape[-1] // 32, 4) * 4
        or not out_scale.is_contiguous()
    ):
        raise ValueError("Packed MXFP8 scale storage must use the 128x4 layout")
    if (
        cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != x.device
        or not cu_seqlens.is_contiguous()
        or cu_seqlens_scale_padded.shape != cu_seqlens.shape
        or cu_seqlens_scale_padded.dtype != torch.int32
        or cu_seqlens_scale_padded.device != x.device
        or not cu_seqlens_scale_padded.is_contiguous()
    ):
        raise ValueError("Packed scale prefixes must be matching contiguous CUDA int32 tensors")
    _update_indexer_mxfp8(x, buffers)
    _pack_indexer_mxfp8_scale_thd_kernel[(out_scale.numel() // 512,)](
        out_scale.view(torch.uint8),
        buffers.logical_scale,
        cu_seqlens,
        cu_seqlens_scale_padded,
        NUM_HEADS=heads,
        REAL_GROUPS=x.shape[-1] // 32,
        LOGICAL_PADDED_GROUPS=buffers.logical_scale.shape[1],
        PADDED_GROUPS=out_scale.shape[2],
        BATCH_SIZE=cu_seqlens.numel() - 1,
        SEARCH_STEPS=(cu_seqlens.numel() - 1).bit_length(),
    )
    return buffers.data, out_scale
