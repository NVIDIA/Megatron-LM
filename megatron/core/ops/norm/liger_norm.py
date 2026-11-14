# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Liger's Megatron RMSNorm behind the normalization builder interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


class LigerRMSNorm:
    """Build ``liger_kernel.megatron.LigerMegatronRMSNorm``, importing Liger only when built.

    Liger keeps the ``weight`` checkpoint key, zero-centered gamma and the sequence-parallel
    gradient marker. Gamma is also tagged for ``--rmsnorm-weight-decay-mult``, as in
    ``WrappedTorchNorm`` and the Transformer Engine norms.
    """

    def __new__(
        cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5, **kwargs
    ) -> torch.nn.Module:
        from liger_kernel.megatron import LigerMegatronRMSNorm  # type: ignore[import-untyped]

        norm = LigerMegatronRMSNorm(config, hidden_size, eps, **kwargs)
        norm.weight.is_rmsnorm_weight = True
        norm.returns_residual = False
        return norm
