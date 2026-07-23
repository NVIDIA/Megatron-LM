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
    """Return the physical E8M0 scale shape expected by the DSA kernel."""
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
def _pack_indexer_mxfp8_scale_kernel(
    out_ptr,
    logical_scale_ptr,
    cu_seqlens_ptr,
    seqlen,
    total_out_bytes,
    NUM_HEADS: tl.constexpr,
    REAL_GROUPS: tl.constexpr,
    LOGICAL_PADDED_GROUPS: tl.constexpr,
    PADDED_ROWS: tl.constexpr,
    PADDED_GROUPS: tl.constexpr,
    IS_THD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pack logical TE scale bytes directly into the Indexer physical layout."""
    out_linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = out_linear < total_out_bytes
    bytes_per_batch = PADDED_ROWS * PADDED_GROUPS
    batch = out_linear // bytes_per_batch
    logical_linear = out_linear - batch * bytes_per_batch
    packed_row = logical_linear // PADDED_GROUPS
    scale_group = logical_linear - packed_row * PADDED_GROUPS
    local_token = packed_row // NUM_HEADS
    head = packed_row - local_token * NUM_HEADS

    if IS_THD:
        seq_start = tl.load(cu_seqlens_ptr + batch, mask=in_bounds, other=0)
        seq_end = tl.load(cu_seqlens_ptr + batch + 1, mask=in_bounds, other=0)
        seq_len = seq_end - seq_start
        global_token = seq_start + local_token
    else:
        seq_len = seqlen
        global_token = batch * seqlen + local_token

    valid = in_bounds & (local_token < seq_len) & (scale_group < REAL_GROUPS)
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


def pack_indexer_mxfp8_scale(
    logical_scale: Tensor,
    out_scale: Tensor,
    *,
    num_heads: int,
    real_groups: int,
    cu_seqlens: Tensor | None = None,
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
            or cu_seqlens.numel() != out_scale.shape[0] + 1
            or not cu_seqlens.is_contiguous()
        ):
            raise ValueError(
                "cu_seqlens must be contiguous CUDA int32 storage for out_scale batches"
            )
    elif seqlen <= 0:
        raise ValueError("BSHD scale packing requires a positive seqlen")

    total_out_bytes = out_scale.numel()
    block = 256
    cu_seqlens_ptr = cu_seqlens if cu_seqlens is not None else logical_scale
    _pack_indexer_mxfp8_scale_kernel[(triton.cdiv(total_out_bytes, block),)](
        out_scale.view(torch.uint8),
        logical_scale,
        cu_seqlens_ptr,
        seqlen,
        total_out_bytes,
        NUM_HEADS=num_heads,
        REAL_GROUPS=real_groups,
        LOGICAL_PADDED_GROUPS=logical_scale.shape[1],
        PADDED_ROWS=out_scale.shape[1],
        PADDED_GROUPS=out_scale.shape[2],
        IS_THD=is_thd,
        BLOCK=block,
    )
    return out_scale


def quantize_indexer_mxfp8(
    x: Tensor,
    *,
    cu_seqlens: Tensor | None = None,
    max_seqlen: int | None = None,
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
        if max_seqlen is None:
            raise ValueError("Packed THD MXFP8 quantization requires max_seqlen")
        batch_size = cu_seqlens.numel() - 1
        seqlen = 0
    else:
        if x.ndim == 4:
            batch_size, seqlen, num_heads, head_dim = x.shape
        elif x.ndim == 3:
            batch_size, seqlen, head_dim = x.shape
            num_heads = 1
        else:
            raise ValueError(f"BSHD MXFP8 input must be 3D or 4D, got shape {x.shape}")
        max_seqlen = seqlen

    if buffers is None:
        buffers = create_indexer_mxfp8_quantization_buffers(x)
    elif not buffers.matches(x):
        raise ValueError("MXFP8 quantization buffers do not match the input tensor")

    source = x
    if buffers.padded_input is not None:
        buffers.padded_input[: buffers.num_rows].copy_(x.reshape(buffers.num_rows, head_dim))
        source = buffers.padded_input
    buffers.quantizer.update_quantized(source, buffers.quantized)

    expected_scale_shape = indexer_mxfp8_scale_shape(
        batch_size, int(max_seqlen), num_heads, head_dim, sf_vec_size
    )
    if out_scale is None:
        out_scale = torch.empty(expected_scale_shape, dtype=torch.float8_e8m0fnu, device=x.device)
    elif (
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
        seqlen=seqlen,
    )
    return buffers.data, out_scale


__all__ = [
    "HAVE_TE_MXFP8",
    "HAVE_TRITON",
    "IndexerMXFP8QuantizationBuffers",
    "create_indexer_mxfp8_quantization_buffers",
    "indexer_mxfp8_scale_shape",
    "pack_indexer_mxfp8_scale",
    "quantize_indexer_mxfp8",
]
