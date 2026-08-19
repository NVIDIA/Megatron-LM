# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Every normalization backend, side by side. The contract they meet is in this package's ``__init__``.

Each class is named for the backend key that selects it, so ``--op-backend layer_norm=apex``
and ``NormApex`` are the same word. Targets are imported inside the method that returns them,
never at module scope, so importing this file pulls in no optional package.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from megatron.core.ops._availability import is_installed
from megatron.core.transformer.torch_norm import WrappedTorchNorm

if TYPE_CHECKING:
    from megatron.core.transformer.torch_norm import LayerNormBuilder

__all__ = ["NormApex", "NormLocal", "NormTE", "NormTorch", "TENormWithResidual"]


class TENormWithResidual:
    """Class adapter for TENorm with residual fusion enabled.

    Defined at module scope on purpose: spec building compares and copies these targets, so a
    class created per provider would make two identically configured models compare unequal.
    """

    def __new__(cls, *args, **kwargs):
        from megatron.core.extensions.transformer_engine import TENorm

        return TENorm(*args, has_residual=True, **kwargs)


class NormTorch:
    """PyTorch's own LayerNorm and RMSNorm.

    Always available, so this is the fallback whenever a faster norm is not installed. It does
    not support sequence parallelism, persistent norm, zero-centered gamma, or the
    memory-efficient path; ``WrappedTorchNorm`` rejects those configurations itself.
    """

    #: Torch's norm backward has not been audited here.
    DETERMINISM = "unknown"

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> "LayerNormBuilder":
        """WrappedTorchNorm reads config.normalization, so one target covers every variant."""
        del rms_norm, for_qk, has_residual
        return WrappedTorchNorm


class NormApex:
    """Apex's fused LayerNorm.

    Apex implements LayerNorm only, so RMSNorm falls through to the Torch target. This mirrors
    what Megatron has always done rather than failing a model that mixes the two.
    """

    #: Apex's fused backward has not been audited here.
    DETERMINISM = "unknown"

    REQUIRES = "apex"

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> "LayerNormBuilder":
        """Apex LayerNorm, or the Torch target when RMSNorm is requested."""
        del for_qk, has_residual
        if rms_norm:
            return WrappedTorchNorm
        from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

        return FusedLayerNorm


class NormTE:
    """Transformer Engine's norm.

    Two backend-internal details stay here rather than at the call site: TE below 1.9 harms
    convergence for query/key norm, and residual fusion needs a distinct target.
    """

    #: TE's norm backward has not been audited here.
    DETERMINISM = "unknown"

    REQUIRES = "transformer_engine"
    FUSES_RESIDUAL = True

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> "LayerNormBuilder":
        """TENorm, or Apex for query/key norm on TE versions that regress convergence."""
        del rms_norm  # TENorm reads config.normalization.
        from megatron.core.utils import is_te_min_version

        if for_qk and not is_te_min_version("1.9.0"):
            # TENorm significantly harms convergence when used for QKLayerNorm if
            # TE version < 1.9; we instead use the Apex implementation.
            from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

            return FusedLayerNorm
        if has_residual and self.FUSES_RESIDUAL:
            return TENormWithResidual
        from megatron.core.extensions.transformer_engine import TENorm

        return TENorm


class NormLocal:
    """Megatron Core's default: Apex's fused LayerNorm when installed, Torch otherwise.

    Torch always for RMSNorm, because Apex has none and its constructor rejects an RMSNorm
    config. This is the one backend that chooses by capability rather than failing.
    """

    #: Defers to Apex or Torch, neither audited here.
    DETERMINISM = "unknown"

    def __init__(self) -> None:
        self._have_apex = is_installed("apex")
        if not self._have_apex:
            warnings.warn("Apex is not installed. Falling back to Torch Norm")

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> "LayerNormBuilder":
        """Apex's fused LayerNorm when available; Torch otherwise, and always for RMSNorm."""
        del for_qk, has_residual
        if rms_norm or not self._have_apex:
            return WrappedTorchNorm
        from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

        return FusedLayerNorm
