# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Linear implementations. :mod:`.contract` says what they meet, :mod:`.backends` has them."""

from megatron.core.inference.ops.backends import LinearInference
from megatron.core.ops.linear.backends import LinearLocal, LinearTE
from megatron.core.ops.linear.contract import (
    COLUMN_PARALLEL_LAYER_NORM_LINEAR,
    COLUMN_PARALLEL_LINEAR,
    FAMILY,
    LINEAR,
    OPERATIONS,
    ROW_PARALLEL_LINEAR,
    LinearSlots,
)

#: Backend name -> the class that owns this family's slots. Add a backend by adding its class
#: to backends.py and one entry here; one that needs an optional package declares ``REQUIRES``.
BACKENDS = {
    "local": LinearLocal,
    "transformer_engine": LinearTE,
    "inference_optimized": LinearInference,
}

#: Used when the selected preset has no entry above.
DEFAULT = "local"

__all__ = [
    "BACKENDS",
    "COLUMN_PARALLEL_LAYER_NORM_LINEAR",
    "COLUMN_PARALLEL_LINEAR",
    "DEFAULT",
    "FAMILY",
    "LINEAR",
    "OPERATIONS",
    "ROW_PARALLEL_LINEAR",
    "LinearSlots",
]
