# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MoE implementations. :mod:`.contract` says what they meet, :mod:`.backends` has them."""

from megatron.core.inference.ops.backends import MoeInference
from megatron.core.ops.moe.backends import MoeLocal, MoeTE
from megatron.core.ops.moe.contract import (
    ACTIVATION_FUNC,
    FAMILY,
    GROUPED_MLP_MODULES,
    MOE_ROUTER,
    OPERATIONS,
    MoeSlots,
)

#: Backend name -> the class that owns this family's slots. Add a backend by adding its class
#: to backends.py and one entry here; one that needs an optional package declares ``REQUIRES``.
BACKENDS = {"local": MoeLocal, "transformer_engine": MoeTE, "inference_optimized": MoeInference}

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
