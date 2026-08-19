# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Linear implementations, grouped by backend. See :mod:`.contract` for the requirements."""

from megatron.core.ops.linear import inference, megatron, transformer_engine
from megatron.core.ops.linear.contract import (
    COLUMN_PARALLEL_LAYER_NORM_LINEAR,
    COLUMN_PARALLEL_LINEAR,
    FAMILY,
    LINEAR,
    OPERATIONS,
    ROW_PARALLEL_LINEAR,
    LinearSlots,
)

#: Backend name -> the class that owns this family's slots. Add a backend by adding its module
#: and one entry here; a backend that needs an optional package declares it as ``REQUIRES``.
BACKENDS = {
    "local": megatron.Linear,
    "transformer_engine": transformer_engine.Linear,
    "inference_optimized": inference.Linear,
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
