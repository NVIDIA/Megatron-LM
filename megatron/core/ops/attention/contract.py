# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""What a core-attention backend has to provide.

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

from megatron.core.ops.operations import Operation, unowned

FAMILY = "attention"

CORE_ATTENTION = Operation(FAMILY, "core_attention")

OPERATIONS = (CORE_ATTENTION,)

__all__ = ["CORE_ATTENTION", "FAMILY", "OPERATIONS", "AttentionSlots"]


class AttentionSlots:
    """The attention slots."""

    def core_attention(self) -> type:
        """Which module to use for attention."""
        raise unowned(CORE_ATTENTION)
