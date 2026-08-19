# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Transformer Engine normalization."""

from __future__ import annotations

from megatron.core.ops import _availability
from megatron.core.transformer.torch_norm import LayerNormBuilder

__all__ = ["TENormBackend", "TENormWithResidual", "te_norm"]

_BACKEND_NAME = "transformer_engine"


def te_norm() -> type:
    """Return Transformer Engine's norm target, importing it only when selected."""
    _availability.require("transformer_engine", backend=_BACKEND_NAME)
    from megatron.core.extensions.transformer_engine import TENorm

    return TENorm


class TENormWithResidual:
    """Class adapter for TENorm with residual fusion enabled.

    Defined at module scope on purpose: spec building compares and copies these targets, so a
    class created per provider would make two identically configured models compare unequal.
    """

    def __new__(cls, *args, **kwargs):
        return te_norm()(*args, has_residual=True, **kwargs)


class TENormBackend:
    """Owns ``layer_norm`` using Transformer Engine.

    Two backend-internal details stay here rather than at the call site: TE below 1.9 harms
    convergence for query/key norm, and residual fusion needs a distinct target.
    """

    def __init__(self, *, fuse_residual: bool = True) -> None:
        _availability.require("transformer_engine", backend=_BACKEND_NAME)
        self._fuse_residual = fuse_residual

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """TENorm, or Apex for query/key norm on TE versions that regress convergence."""
        del rms_norm  # TENorm reads config.normalization.
        from megatron.core.utils import is_te_min_version

        if for_qk and not is_te_min_version("1.9.0"):
            # TENorm significantly harms convergence when used for QKLayerNorm if
            # TE version < 1.9; we instead use the Apex implementation.
            from megatron.core.ops.norm.apex import apex_layer_norm

            return apex_layer_norm()
        if has_residual and self._fuse_residual:
            return TENormWithResidual
        return te_norm()
