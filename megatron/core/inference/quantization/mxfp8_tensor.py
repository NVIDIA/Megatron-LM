# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
import torch

try:
    from flashinfer import mxfp8_quantize

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = True


@dataclass
class MXFP8Tensor:
    data: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def from_bf16(cls, x: torch.Tensor, group_size: int = 32):
        """
        Quantize BF16 tensor to MXFP8 format using FlashInfer.
        Returns:
            x_fp8: Tensor of type float8_e4m3fn (M, K)
            x_scale: Tensor of type float8_e8m0fnu (M, K // 32)
        """
        assert x.is_cuda, "Input must be on CUDA"
        assert x.dim() == 2, "Input must be 2D [M, K]"
        M, K = x.shape
        assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"

        return cls(*mxfp8_quantize(x))
