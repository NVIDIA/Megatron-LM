# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

try:
    from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False

try:
    from triton_kernels.numerics_details.mxfp import downcast_to_mxfp

    HAVE_TRITON_MXFP = True
except ImportError:
    HAVE_TRITON_MXFP = False


# --------------------------------------------------------------------------- #
# Triton kernel: swizzle MXFP8 scales to cuBLAS blocked layout
# --------------------------------------------------------------------------- #
def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def _swizzle_scales_kernel(
    input_ptr, output_ptr,
    rows, cols,
    padded_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Convert [rows, cols] scale matrix to cuBLAS 2D blocked layout in one pass.

    See: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_col_blocks = padded_cols // 4
    macro_tile_idx = offsets // 512
    within_tile = offsets % 512
    sub_row = within_tile // 16
    sub_col = within_tile % 16
    group = sub_col // 4
    col_in_group = sub_col % 4
    macro_row_block = macro_tile_idx // n_col_blocks
    macro_col_block = macro_tile_idx % n_col_blocks
    src_row = macro_row_block * 128 + group * 32 + sub_row
    src_col = macro_col_block * 4 + col_in_group
    in_bounds = (src_row < rows) & (src_col < cols)
    val = tl.load(input_ptr + src_row * cols + src_col, mask=in_bounds, other=0)
    n_row_blocks = tl.cdiv(rows, 128)
    tl.store(output_ptr + offsets, val, mask=offsets < n_row_blocks * n_col_blocks * 512)


def swizzle_scales(scale: torch.Tensor) -> torch.Tensor:
    """Swizzle a [rows, cols] e8m0 scale matrix to cuBLAS blocked layout.

    Args:
        scale: [M, K//32] tensor of dtype float8_e8m0fnu or uint8.

    Returns:
        1D tensor in cuBLAS blocked layout, same dtype as input.
    """
    rows, cols = scale.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    total_out = n_row_blocks * n_col_blocks * 512
    inp_u8 = scale.view(torch.uint8)
    out_u8 = torch.empty(total_out, dtype=torch.uint8, device=scale.device)
    BLOCK_SIZE = 1024
    _swizzle_scales_kernel[(_ceil_div(total_out, BLOCK_SIZE),)](
        inp_u8, out_u8, rows, cols, n_col_blocks * 4, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_u8.view(scale.dtype)


# --------------------------------------------------------------------------- #
# MXFP8Tensor
# --------------------------------------------------------------------------- #
@dataclass
class MXFP8Tensor:
    """MXFP8 tensor wrapper storing quantized fp8_e4m3 data and swizzled e8m0 scales."""

    data: torch.Tensor   # [M, K] fp8_e4m3fn
    scale: torch.Tensor  # 1D, swizzled cuBLAS blocked layout, e8m0

    def size(self, idx: Optional[int] = None):
        """Wrapper for calling self.data.size()"""
        return self.data.size(idx)

    @classmethod
    def from_bf16_flashinfer(cls, x: torch.Tensor, group_size: int = 32):
        """Quantize BF16 tensor to MXFP8 using FlashInfer (single fused CUDA kernel)."""
        assert HAVE_FLASHINFER, "FlashInfer not available"
        assert x.is_cuda and x.dim() == 2
        assert x.shape[-1] % group_size == 0
        return cls(*flashinfer_mxfp8_quantize(x))

    @classmethod
    def from_bf16_torch(cls, x: torch.Tensor, group_size: int = 32):
        """Quantize BF16 tensor to MXFP8 using Triton (downcast_to_mxfp + to_blocked)."""
        assert HAVE_TRITON_MXFP, "triton_kernels.numerics_details.mxfp not available"
        assert x.is_cuda and x.dim() == 2
        assert x.shape[-1] % group_size == 0
        xq, xs = downcast_to_mxfp(x, torch.float8_e4m3fn, 1)
        if xs.dtype == torch.uint8:
            xs = xs.view(torch.float8_e8m0fnu)
        return cls(data=xq, scale=swizzle_scales(xs))
