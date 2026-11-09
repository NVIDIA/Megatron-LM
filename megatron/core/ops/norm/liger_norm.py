# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Liger's Megatron RMSNorm adapted to the normalization interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from liger_kernel.megatron import LigerMegatronRMSNorm  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


class LigerRMSNorm(LigerMegatronRMSNorm):
    """RMSNorm with checkpoint-compatible weights and no fused residual output.

    The inherited implementation supports zero-centered gamma and marks its weight
    for sequence-parallel gradient reduction. Parameter placement follows PyTorch's
    default device and dtype, as with the other local normalization modules.
    The adapter also identifies gamma for RMSNorm-specific weight decay.
    """

    returns_residual: bool = False

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "RMSNorm",
    ):
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            eps=eps,
            persist_layer_norm=persist_layer_norm,
            zero_centered_gamma=zero_centered_gamma,
            normalization=normalization,
        )
        self.weight.is_rmsnorm_weight = True
