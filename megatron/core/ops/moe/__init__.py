# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MoE implementations, grouped by backend. See :mod:`.contract` for the requirements."""

from megatron.core.ops.moe import inference, megatron, transformer_engine
from megatron.core.ops.moe.contract import (
    ACTIVATION_FUNC,
    FAMILY,
    GROUPED_MLP_MODULES,
    MOE_ROUTER,
    OPERATIONS,
    MoeSlots,
)

#: Backend name -> the class that owns this family's slots. Add a backend by adding its module
#: and one entry here; a backend that needs an optional package declares it as ``REQUIRES``.
BACKENDS = {
    "local": megatron.Moe,
    "transformer_engine": transformer_engine.Moe,
    "inference_optimized": inference.Moe,
}

#: Used when the selected preset has no entry above.
DEFAULT = "local"

__all__ = [
    "ACTIVATION_FUNC",
    "BACKENDS",
    "DEFAULT",
    "FAMILY",
    "GROUPED_MLP_MODULES",
    "MOE_ROUTER",
    "OPERATIONS",
    "MoeSlots",
]
