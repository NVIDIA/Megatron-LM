# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Liger's Megatron RMSNorm behind the normalization builder interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from megatron.core.ops._backends import require

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


def _normalize_in_weight_dtype(norm: torch.nn.Module, args: tuple) -> tuple:
    """Cast the input to gamma's dtype first, as Transformer Engine's RMSNorm does.

    Liger keeps the input dtype, so FP32 residuals (``fp32_residual_connection``) would
    otherwise reach the BF16 output layer as FP32.
    """
    return (args[0].to(norm.weight.dtype), *args[1:])


class LigerRMSNorm:
    """Build ``liger_kernel.megatron.LigerMegatronRMSNorm``, importing Liger only when built.

    Liger keeps the ``weight`` checkpoint key, zero-centered gamma and the sequence-parallel
    gradient marker. Gamma is created on Megatron's initialization device in
    ``params_dtype`` and tagged for ``--rmsnorm-weight-decay-mult``, as the other norms are.
    """

    def __new__(
        cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5, **kwargs
    ) -> torch.nn.Module:
        liger = require(
            "liger_kernel.megatron",
            "LigerMegatronRMSNorm",
            min_version="0.8.5",
            dist="liger-kernel",
            needed_by="Liger RMSNorm",
        )
        norm = liger.LigerMegatronRMSNorm(config, hidden_size, eps, **kwargs)
        cpu = getattr(config, "use_cpu_initialization", False)
        norm.to(
            device="cpu" if cpu else torch.cuda.current_device(),
            dtype=getattr(config, "params_dtype", torch.float32),
        )
        norm.weight.is_rmsnorm_weight = True
        norm.returns_residual = False
        norm.register_forward_pre_hook(_normalize_in_weight_dtype)
        return norm
