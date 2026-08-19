# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Attention implementations. :mod:`.contract` says what they meet, :mod:`.backends` has them."""

from megatron.core.ops.attention.backends import AttentionLocal, AttentionTE
from megatron.core.ops.attention.contract import CORE_ATTENTION, FAMILY, OPERATIONS, AttentionSlots

#: Backend name -> the class that owns this family's slots. Add a backend by adding its class
#: to backends.py and one entry here; one that needs an optional package declares ``REQUIRES``.
BACKENDS = {
    "local": AttentionLocal,
    "transformer_engine": AttentionTE,
    # Inference reuses TE attention; the inference gains are in the linear and MoE layers.
    "inference_optimized": AttentionTE,
}

#: Used when the selected preset has no entry above.
DEFAULT = "local"

__all__ = ["BACKENDS", "CORE_ATTENTION", "DEFAULT", "FAMILY", "OPERATIONS", "AttentionSlots"]
