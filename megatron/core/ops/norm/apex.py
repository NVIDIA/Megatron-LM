# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Apex fused normalization."""

from __future__ import annotations

from megatron.core.transformer.torch_norm import LayerNormBuilder

__all__ = ["Norm", "apex_layer_norm"]


def apex_layer_norm() -> type:
    """Return Apex's fused LayerNorm target, importing it only when selected."""
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    return FusedLayerNorm


class Norm:
    """Owns ``layer_norm`` using Apex's fused LayerNorm.

    Apex implements LayerNorm only, so RMSNorm falls through to the reference target. This
    mirrors what Megatron has always done rather than failing a model that mixes the two.
    """

    REQUIRES = "apex"

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Apex LayerNorm, or the reference target when RMSNorm is requested."""
        del for_qk, has_residual
        if rms_norm:
            from megatron.core.ops.norm.reference import WrappedTorchNorm

            return WrappedTorchNorm
        return apex_layer_norm()
