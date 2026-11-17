# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Which RMSNorm kernel a config asks for, shared by every provider and spec."""

from __future__ import annotations

from typing import TYPE_CHECKING

from megatron.core.ops._backends import require
from megatron.core.ops.norm.torch_norm import LayerNormBuilder

if TYPE_CHECKING:
    import torch

    from megatron.core.transformer.transformer_config import TransformerConfig


def select_rms_norm(native: LayerNormBuilder, *, use_liger: bool = False) -> LayerNormBuilder:
    """Return the RMSNorm the config selects, or ``native``, the caller's own choice.

    Like ``select_cross_entropy``, this depends on the config, not on which provider or
    spec builds the rest of the model. Liger is imported only when selected.
    """
    if not use_liger:
        return native
    require("liger_kernel.megatron", "LigerMegatronRMSNorm", needed_by="Liger RMSNorm")
    from megatron.core.ops.norm.liger_norm import LigerRMSNorm

    return LigerRMSNorm


class SelectableNorm:
    """Build ``native``, or the RMSNorm kernel the config selects instead.

    For specs fixed at import time, such as HybridModel's, which cannot ask a provider. The
    kernel is chosen from the config when the norm is built, as ``TENorm`` already chooses
    LayerNorm or RMSNorm. Bind ``native`` with ``functools.partial``.
    """

    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        *,
        native: LayerNormBuilder,
        **kwargs,
    ) -> torch.nn.Module:
        if config.normalization == "RMSNorm":
            native = select_rms_norm(native, use_liger=config.use_liger)
        return native(config=config, hidden_size=hidden_size, eps=eps, **kwargs)
