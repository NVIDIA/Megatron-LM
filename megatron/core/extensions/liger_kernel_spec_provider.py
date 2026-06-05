# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""BackendSpecProvider that uses Liger-Kernel Triton operations.

This is a thin extension over ``LocalSpecProvider`` that swaps Liger
implementations into select slots. Currently overrides only ``layer_norm``
(returns Liger's RMSNorm); other slots inherit the local defaults. The
provider is enabled via ``use_liger=True`` in ``get_gpt_layer_local_submodules``.
"""
from __future__ import annotations

from typing import Optional

try:
    from liger_kernel.megatron import LigerMegatronRMSNorm

    HAVE_LIGER = True
except ImportError:
    LigerMegatronRMSNorm = None
    HAVE_LIGER = False

from megatron.core.models.backends import LocalSpecProvider
from megatron.core.transformer.torch_norm import LayerNormBuilder


class LigerSpecProvider(LocalSpecProvider):
    """A spec provider that uses Liger Triton kernels where available.

    Inherits all defaults from ``LocalSpecProvider``; currently overrides only
    ``layer_norm`` to return ``LigerMegatronRMSNorm`` when ``rms_norm=True``.
    As more Liger kernels are integrated (RoPE, SwiGLU MLP, fused linear
    cross-entropy), this provider will gain the corresponding overrides.
    """

    def __init__(self) -> None:
        if not HAVE_LIGER:
            raise ImportError(
                "Liger-Kernel is required for LigerSpecProvider. "
                "Install with `pip install liger-kernel`."
            )

    def layer_norm(
        self,
        rms_norm: bool = False,
        for_qk: bool = False,
        has_residual: bool = False,
    ) -> LayerNormBuilder:
        """Which module to use for layer norm."""
        if rms_norm:
            return LigerMegatronRMSNorm
        return super().layer_norm(
            rms_norm=rms_norm, for_qk=for_qk, has_residual=has_residual
        )
