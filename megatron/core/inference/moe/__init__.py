# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import enum
import torch
from .fused_moe import ActivationType, mcore_fused_moe


class InferenceGroupedGemmBackend(enum.Enum):
    """Resolved backend for grouped GEMM operations during inference."""

    FLASHINFER = "flashinfer"
    TORCH = "torch"
    TE = "te"


def resolve_inference_grouped_gemm_backend(
    backend: str,
    is_cuda_graphed: bool,
) -> InferenceGroupedGemmBackend:
    """Resolve the grouped GEMM backend to use for the current iteration.

    Prerequisites are validated at init time in MoELayer; this function
    simply maps (backend, is_cuda_graphed) to the concrete backend enum.

    Args:
        backend: One of 'auto', 'torch', 'te'.
        is_cuda_graphed: Whether this is a CUDA-graphed iteration.

    Returns:
        An InferenceGroupedGemmBackend enum value.
    """
    if backend == 'auto':
        if is_cuda_graphed:
            return InferenceGroupedGemmBackend.FLASHINFER
        else:
            if hasattr(torch.nn.functional, 'grouped_mm'):
                return InferenceGroupedGemmBackend.TORCH
            else:
                return InferenceGroupedGemmBackend.TE
    elif backend == 'torch':
        return InferenceGroupedGemmBackend.TORCH
    elif backend == 'te':
        return InferenceGroupedGemmBackend.TE
    else:
        raise ValueError(
            f"Unknown inference_grouped_gemm_backend: '{backend}'. "
            "Must be 'auto', 'torch', or 'te'."
        )
