# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Norm implementations. :mod:`.contract` says what they meet, :mod:`.backends` has them."""

from megatron.core.inference.ops.backends import NormInference
from megatron.core.ops.norm.backends import NormApex, NormLocal, NormTE, NormTorch
from megatron.core.ops.norm.contract import FAMILY, LAYER_NORM, OPERATIONS, NormSlots
from megatron.core.transformer.torch_norm import L2Norm, WrappedTorchNorm

#: Backend name -> the class that owns this family's slots. Add a backend by adding its class
#: to backends.py and one entry here; one that needs an optional package declares ``REQUIRES``.
BACKENDS = {
    "local": NormLocal,
    "transformer_engine": NormTE,
    "inference_optimized": NormInference,
    "torch": NormTorch,
    "apex": NormApex,
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
