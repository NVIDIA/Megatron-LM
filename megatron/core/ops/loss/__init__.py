# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Loss implementations, grouped by backend. See :mod:`.contract` for the requirements."""

from megatron.core.ops.loss import megatron, megatron_fused, transformer_engine
from megatron.core.ops.loss.contract import (
    FAMILY,
    OPERATIONS,
    VOCAB_PARALLEL_CROSS_ENTROPY,
    LossSlots,
    VocabParallelCrossEntropy,
)
from megatron.core.ops.loss.megatron import vocab_parallel_cross_entropy
from megatron.core.ops.loss.megatron_fused import fused_vocab_parallel_cross_entropy

#: Backend name -> the class that owns this family's slots.
#:
#: Deliberately no "transformer_engine" key: a Transformer Engine run defaults to the reference
#: cross entropy, and only takes TE's fused kernel when asked for it. Naming the fused entry
#: "te_fused" keeps the --transformer-impl preset from silently claiming this slot.
BACKENDS = {
    "megatron": megatron.Loss,
    "megatron_fused": megatron_fused.Loss,
    "te_fused": transformer_engine.Loss,
}

#: Used when the selected preset has no entry above, which for this family is always.
DEFAULT = "megatron"

__all__ = [
    "BACKENDS",
    "DEFAULT",
    "FAMILY",
    "OPERATIONS",
    "VOCAB_PARALLEL_CROSS_ENTROPY",
    "LossSlots",
    "VocabParallelCrossEntropy",
    "fused_vocab_parallel_cross_entropy",
    "vocab_parallel_cross_entropy",
]
