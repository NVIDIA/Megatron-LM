# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Every dense MLP backend, side by side. The contract they meet is in this package's
``__init__``."""

from __future__ import annotations

__all__ = ["MlpMegatron", "MlpTEOpFuser"]


class MlpMegatron:
    """Megatron Core's MLP block, built from whatever linear backend was selected."""

    #: Not audited; it is the linear and activation backends underneath that would decide.
    DETERMINISM = "unknown"

    def mlp_module(self, grouped: bool = False) -> type:
        """Megatron Core has one MLP block; grouped GEMM is a Transformer Engine feature."""
        del grouped
        from megatron.core.transformer.mlp import MLP

        return MLP


class MlpTEOpFuser:
    """Transformer Engine's operation-fused MLP.

    Selected by ``--use-transformer-engine-op-fuser``. The version it needs is declared rather
    than checked in code, so it is visible in the class header and enforced while arguments are
    parsed.
    """

    REQUIRES = "transformer_engine>=1.13.0"

    #: Not audited; the fused operations have not been checked for bit-exactness.
    DETERMINISM = "unknown"

    #: The block folds the linears it is handed into Transformer Engine operations and can
    #: only convert Transformer Engine ones -- TEFusedMLP raises on anything else. So this
    #: backend is only valid alongside LinearTE, which is why only TESpecProvider offers it.

    def mlp_module(self, grouped: bool = False) -> type:
        """The fused MLP, in its grouped-linear form when the dense MLP uses grouped GEMM."""
        from megatron.core.extensions.transformer_engine import (
            TEFusedMLP,
            TEFusedMLPWithGroupedLinear,
        )

        target = TEFusedMLPWithGroupedLinear if grouped else TEFusedMLP
        if target is None:
            raise ImportError(
                "Transformer Engine is installed but does not expose the operation-fused MLP."
            )
        return target
