# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Which RMSNorm kernel a config asks for."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from megatron.core.ops.norm.torch_norm import LayerNormBuilder
    from megatron.core.transformer.transformer_config import TransformerConfig


def select_rms_norm(config: TransformerConfig) -> Optional[LayerNormBuilder]:
    """Return the RMSNorm ``config.rmsnorm_impl`` selects, or ``None`` for the caller's own.

    ``TENorm`` and ``WrappedTorchNorm`` ask this when they build an RMSNorm, so the choice
    follows the config wherever a spec or provider names those norms, as cross entropy
    follows ``select_cross_entropy``. Liger is imported only when its norm is built.
    """
    if getattr(config, "rmsnorm_impl", "native") != "liger" or config.normalization != "RMSNorm":
        return None
    from megatron.core.ops.norm.liger_norm import LigerRMSNorm

    return LigerRMSNorm
