# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Attention implementations, grouped by backend. See :mod:`.contract` for the requirements."""

from megatron.core.ops.attention import reference, transformer_engine
from megatron.core.ops.attention.contract import CORE_ATTENTION, FAMILY, OPERATIONS, AttentionSlots

#: Backend name -> the class that owns this family's slots. Add a backend by adding its module
#: and one entry here; a backend that needs an optional package declares it as ``REQUIRES``.
BACKENDS = {
    "local": reference.Attention,
    "transformer_engine": transformer_engine.Attention,
    # Inference reuses TE attention; the inference gains are in the linear and MoE layers.
    "inference_optimized": transformer_engine.Attention,
}

#: Used when the selected preset has no entry above.
DEFAULT = "local"

__all__ = ["BACKENDS", "CORE_ATTENTION", "DEFAULT", "FAMILY", "OPERATIONS", "AttentionSlots"]
