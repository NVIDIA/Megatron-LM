# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Reusable CuTe top-k launches with a pure Triton exact fallback."""

import torch
import triton
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import from_dlpack

from .histogram_topk import _compile_kernel
from .radix_fallback import fallback_if_needed


class CuteTopK:
    """Compile a bounded CuTe selection launch with preallocated exact fallback storage."""

    def __init__(
        self, scores: torch.Tensor, lengths: torch.Tensor, output: torch.Tensor, flags: torch.Tensor
    ) -> None:
        self.scores, self.lengths, self.output, self.flags = (scores, lengths, output, flags)
        self.arguments = tuple(from_dlpack(t) for t in (scores, lengths, output, flags))
        self.compiled = _compile_kernel(scores, lengths, output, flags)

    def __call__(self) -> None:
        """Run the prepared CuTe selection and conditional exact fallback."""
        self.compiled(*self.arguments, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        fallback_if_needed[(self.scores.shape[0],)](
            self.scores,
            self.lengths,
            self.output,
            self.flags,
            WIDTH=self.scores.shape[1],
            K=self.output.shape[1],
            BLOCK=triton.next_power_of_2(self.scores.shape[1]),
            num_warps=16,
            enable_fp_fusion=False,
        )
