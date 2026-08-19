# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reference (pure PyTorch) normalization."""

from __future__ import annotations

from megatron.core.transformer.torch_norm import L2Norm, LayerNormBuilder, WrappedTorchNorm

__all__ = ["L2Norm", "Norm", "WrappedTorchNorm"]


class Norm:
    """Owns ``layer_norm`` using PyTorch's own LayerNorm and RMSNorm.

    Always available, so this is the fallback whenever a faster norm is not installed. It does
    not support sequence parallelism, persistent norm, zero-centered gamma, or the
    memory-efficient path; ``WrappedTorchNorm`` rejects those configurations itself.
    """

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """WrappedTorchNorm reads config.normalization, so one target covers every variant."""
        del rms_norm, for_qk, has_residual
        return WrappedTorchNorm
