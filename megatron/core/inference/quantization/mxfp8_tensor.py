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

from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize as mcore_mxfp8_quantize


def _ceil_div(a, b):
    return (a + b - 1) // b


@dataclass
class MXFP8Tensor:
    """MXFP8 tensor wrapper storing quantized fp8_e4m3 data and swizzled e8m0 scales."""

    data: torch.Tensor   # [M, K] fp8_e4m3fn
    scale: torch.Tensor  # 1D, swizzled cuBLAS blocked layout, e8m0
    backend: Optional[str] = None  # quantization backend: 'flashinfer' or 'triton'

    def size(self, idx: Optional[int] = None):
        """Wrapper for calling self.data.size()"""
        return self.data.size(idx)

    def scale_2d(self, K: Optional[int] = None) -> torch.Tensor:
        """Reshape 1D swizzled scale to 2D for scaled_grouped_mm / scaled_mm.

        Swizzle pads rows to multiples of 128 and cols to multiples of 4.
        Returns (padded_M, padded_cols) where padded_cols = ceil(K//32, 4) * 4.
        """
        if self.scale.dim() == 2:
            return self.scale
        if K is None:
            K = self.data.shape[-1]
        n_col_blocks = _ceil_div(K // 32, 4)
        padded_cols = n_col_blocks * 4
        return self.scale.reshape(-1, padded_cols)

    @classmethod
    def from_bf16(cls, x: torch.Tensor, group_size: int = 32, backend: str = "flashinfer"):
        """Quantize BF16 tensor to MXFP8.

        Args:
            x: [M, K] BF16 tensor on CUDA.
            group_size: MXFP8 group size (default 32).
            backend: 'triton' (fused quantize + swizzle Triton kernel) or
                     'flashinfer' (single fused FlashInfer CUDA kernel).
        """
        assert x.is_cuda and x.dim() == 2
        assert x.shape[-1] % group_size == 0
        if backend == "flashinfer":
            assert HAVE_FLASHINFER, "FlashInfer not available"
            return cls(*flashinfer_mxfp8_quantize(x), backend=backend)
        elif backend == "triton":
            xq, xs = mcore_mxfp8_quantize(x)
            return cls(data=xq, scale=xs, backend=backend)
        else:
            raise ValueError(
                f"Unknown MXFP8 quantization backend: '{backend}'. "
                "Must be 'triton' or 'flashinfer'."
            )