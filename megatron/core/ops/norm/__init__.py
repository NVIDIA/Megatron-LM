# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Norm implementations, grouped by backend. See :mod:`.contract` for the requirements."""

from megatron.core.ops.norm import apex, megatron, reference, transformer_engine
from megatron.core.ops.norm.contract import FAMILY, LAYER_NORM, OPERATIONS, NormSlots
from megatron.core.ops.norm.reference import L2Norm, WrappedTorchNorm

#: Backend name -> the class that owns this family's slots. Add a backend by adding its module
#: and one entry here; a backend that needs an optional package declares it as ``REQUIRES``.
BACKENDS = {
    "local": megatron.Norm,
    "transformer_engine": transformer_engine.Norm,
    "inference_optimized": transformer_engine.InferenceNorm,
    "torch": reference.Norm,
    "apex": apex.Norm,
}

#: Used when the selected preset has no entry above.
DEFAULT = "local"

__all__ = [
    "BACKENDS",
    "DEFAULT",
    "FAMILY",
    "LAYER_NORM",
    "OPERATIONS",
    "L2Norm",
    "NormSlots",
    "WrappedTorchNorm",
]
