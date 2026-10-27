# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import enum

from .fused_moe import ActivationType, mcore_fused_moe
from .vllm_fused_moe import vllm_fused_moe


class InferenceGroupedGemmBackend(enum.Enum):
    """Backend for grouped GEMM operations during inference.

    The string values match the ``inference_grouped_gemm_backend`` config field.
    """

    FLASHINFER = "flashinfer"
    TORCH = "torch"
    VLLM = "vllm"

    @classmethod
    def from_config(cls, value: str | InferenceGroupedGemmBackend) -> InferenceGroupedGemmBackend:
        """Convert a config value to a grouped-GEMM backend.

        Args:
            value: Backend name or an already-converted enum member.

        Returns:
            The corresponding grouped-GEMM backend.

        Raises:
            ValueError: If ``value`` is not a supported backend.
        """
        try:
            return cls(value)
        except ValueError as error:
            choices = ", ".join(repr(backend.value) for backend in cls)
            raise ValueError(
                f"inference_grouped_gemm_backend must be one of {choices}, got {value!r}"
            ) from error
