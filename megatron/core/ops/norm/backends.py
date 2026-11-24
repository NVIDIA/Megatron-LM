# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Which norm module a spec names for a norm backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.core.ops.norm.torch_norm import LayerNormBuilder


def get_norm_module_spec(backend: str = "te") -> LayerNormBuilder:
    """Return the norm module a spec should name for ``backend`` (a ``rmsnorm_impl`` value).

    ``"te"`` is Transformer Engine's ``TENorm``; ``"liger"`` is Liger's RMSNorm, which
    imports Liger only when a norm is built. Specs choose their norms with this, as they choose
    every other module.
    """
    if backend == "te":
        from megatron.core.extensions.transformer_engine import TENorm

        return TENorm
    if backend == "liger":
        from megatron.core.ops.norm.liger_norm import LigerRMSNorm

        return LigerRMSNorm
    raise ValueError(f"Unknown norm backend {backend!r}; expected 'te' or 'liger'.")
