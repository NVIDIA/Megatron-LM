# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Loss implementations. :mod:`.contract` says what they meet, :mod:`.backends` has them."""

from megatron.core.ops.loss.backends import (
    LossMegatron,
    LossMegatronFused,
    LossTEFused,
    fused_vocab_parallel_cross_entropy,
    vocab_parallel_cross_entropy,
)
from megatron.core.ops.loss.contract import (
    FAMILY,
    OPERATIONS,
    VOCAB_PARALLEL_CROSS_ENTROPY,
    LossSlots,
    VocabParallelCrossEntropy,
)

#: Backend name -> the class that owns this family's slots.
#:
#: Deliberately no "transformer_engine" key: a Transformer Engine run defaults to the reference
#: cross entropy, and only takes TE's fused kernel when asked for it. Naming the fused entry
#: "te_fused" keeps the --transformer-impl preset from silently claiming this slot.
BACKENDS = {"megatron": LossMegatron, "megatron_fused": LossMegatronFused, "te_fused": LossTEFused}

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
