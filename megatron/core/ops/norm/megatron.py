# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron Core's default normalization policy."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from megatron.core.ops._availability import is_installed
from megatron.core.ops.norm.apex import apex_layer_norm
from megatron.core.ops.norm.reference import WrappedTorchNorm

if TYPE_CHECKING:
    from megatron.core.transformer.torch_norm import LayerNormBuilder

__all__ = ["Norm"]


class Norm:
    """Owns ``layer_norm`` with Megatron Core's long-standing choice.

    Apex's fused LayerNorm when it is installed, Torch otherwise, and Torch always for RMSNorm
    because Apex has none and its constructor rejects an RMSNorm config.
    """

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
        return apex_layer_norm()
