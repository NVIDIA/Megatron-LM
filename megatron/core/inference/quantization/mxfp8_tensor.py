# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import Optional

import torch

try:
    from flashinfer import mxfp8_quantize

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = True


@dataclass
class MXFP8Tensor:
    """MXFP8 tensor wrapper class."""

    data: torch.Tensor
    scale: torch.Tensor

    def size(self, idx: Optional[int] = None):
        """Wrapper for calling self.data.size()"""
        return self.data.size(idx)

    @classmethod
    def from_bf16(cls, x: torch.Tensor, group_size: int = 32):
        """Quantize BF16 tensor to MXFP8 format using FlashInfer."""

        assert HAVE_FLASHINFER, "Need flashinfer for mxfp8 quantization"
        assert x.is_cuda, "Input must be on CUDA"
        assert x.dim() == 2, "Input must be 2D [M, K]"
        M, K = x.shape
        assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"

        return cls(*mxfp8_quantize(x))
