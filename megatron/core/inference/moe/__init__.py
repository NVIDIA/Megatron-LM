# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import enum

from .fused_moe import ActivationType, mcore_fused_moe


class InferenceGroupedGemmBackend(enum.Enum):
    """Backend for grouped GEMM operations during inference.

    The string value matches the inference_grouped_gemm_backend config field so
    TransformerConfig.__post_init__ can convert via InferenceGroupedGemmBackend(str).
    """

    FLASHINFER = "flashinfer"
    TORCH = "torch"
    TE = "te"