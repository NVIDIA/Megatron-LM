# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Core attention: what a backend must provide, and which ones exist.

The implementations are in :mod:`.backends`.

Contract
--------
* **Target**: a class the attention module constructs with Megatron's core-attention
  signature, taking the layer number, attention mask type, and CP communication type.
* **State**: none beyond what the target itself allocates. Q/K/V projections and the output
  projection stay with the attention module, not with this target.
* **Process groups**: the target owns context-parallel communication using the CP group and
  communication type the attention module passes in. Tensor-parallel splitting of heads is
  done by the projections, not here.
* **Modes**: backends differ on packed and variable-length input, decode, and quantization.
  Transformer Engine selects flash, fused, or unfused internally; Megatron does not add a
  second attention dispatcher on top of it.
"""

from __future__ import annotations

from megatron.core.ops.attention.backends import AttentionLocal, AttentionTE
from megatron.core.ops.operations import Operation, unowned

FAMILY = "attention"

CORE_ATTENTION = Operation(FAMILY, "core_attention")

OPERATIONS = (CORE_ATTENTION,)


class AttentionSlots:
    """The attention slots."""

    def core_attention(self) -> type:
        """Which module to use for attention."""
        raise unowned(CORE_ATTENTION)


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
