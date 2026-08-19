# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Loss: what a backend must provide, and which ones exist.

The implementations are in :mod:`.backends`.

Contract for vocab-parallel cross entropy
-----------------------------------------
* **Target**: ``loss = target(logits, labels, tp_group)``.
* **Layout**: ``logits`` are ``[s, b, vocab / tp]`` and ``labels`` are ``[s, b]``, already
  transposed by the caller. The returned loss is ``[s, b]``.
* **State**: none. The target holds no parameters and no cache. It does, however, *consume*
  ``logits``: every target subtracts the row max and exponentiates in place to save memory,
  and the up-cast that would otherwise copy is a no-op when ``logits`` is already float32. A
  caller that needs ``logits`` afterwards has to pass a clone.
* **Process groups**: the target owns every reduction over ``tp_group``. The caller performs
  no collectives, so backends are free to fuse the reduction into their kernel. ``tp_group``
  may be ``None``, meaning the default tensor-parallel group; targets whose kernel cannot take
  ``None`` fill it in themselves rather than pushing that onto the caller.
* **Modes**: every backend supports training and backward. Only the Transformer Engine target
  supports capture under a full-iteration CUDA graph, and only from TE 2.7.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from megatron.core.ops.loss.backends import (
    LossMegatron,
    LossMegatronFused,
    LossTEFused,
    fused_vocab_parallel_cross_entropy,
    vocab_parallel_cross_entropy,
)
from megatron.core.ops.operations import Operation, unowned

FAMILY = "loss"

VOCAB_PARALLEL_CROSS_ENTROPY = Operation(FAMILY, "vocab_parallel_cross_entropy")

OPERATIONS = (VOCAB_PARALLEL_CROSS_ENTROPY,)

VocabParallelCrossEntropy = Callable[
    [torch.Tensor, torch.Tensor, Optional[torch.distributed.ProcessGroup]], torch.Tensor
]
"""The callable shape this family's targets have."""


class LossSlots:
    """The loss slots."""

    def vocab_parallel_cross_entropy(self) -> VocabParallelCrossEntropy:
        """Which vocab-parallel cross entropy to use."""
        raise unowned(VOCAB_PARALLEL_CROSS_ENTROPY)


#: Backend name -> the class that owns this family's slots.
#:
#: Deliberately no "transformer_engine" key: a Transformer Engine run defaults to the reference
#: cross entropy rather than having the --transformer-impl preset claim this slot.
BACKENDS = {
    "megatron": LossMegatron,
    "megatron_fused": LossMegatronFused,
    # "te_fused": LossTEFused is deliberately absent. megatron/training/arguments.py disables
    # Transformer Engine cross entropy outright for stability, and registering it here would
    # let --op-backend reach it and route around that. LossTEFused stays for when it is
    # re-enabled; adding this entry back is the whole change.
}

#: Used when the selected preset has no entry above, which for this family is always.
DEFAULT = "megatron"


def legacy_backends(options) -> dict:
    """Which backend the cross entropy flags that predate --op-backend select.

    ``--cross-entropy-loss-fusion`` and ``--cross-entropy-fusion-impl`` are older than
    ``--op-backend``, so their values are a second vocabulary for the same choice. The
    translation lives here, next to BACKENDS, so both vocabularies are visible together:

    =======================================  =========================
    flags                                    backend
    =======================================  =========================
    (no fusion)                              ``megatron``  (DEFAULT)
    ``--cross-entropy-fusion-impl native``   ``megatron_fused``
    ``--cross-entropy-fusion-impl te``       refused, see below
    =======================================  =========================
    """
    if not options.cross_entropy_loss_fusion:
        return {}
    impl = options.cross_entropy_fusion_impl
    if impl == "te":
        # The same refusal megatron/training/arguments.py gives; te_fused is not registered
        # above either, so neither path can reach it.
        raise ValueError(
            "Transformer Engine cross entropy loss fusion is disabled due to stability "
            "issues. Use --cross-entropy-fusion-impl native, or omit "
            "--cross-entropy-loss-fusion."
        )
    if impl != "native":
        raise ValueError(f"Unknown cross_entropy_fusion_impl='{impl}'. Valid choices: native, te")
    return {VOCAB_PARALLEL_CROSS_ENTROPY: "megatron_fused"}


__all__ = [
    "BACKENDS",
    "DEFAULT",
    "FAMILY",
    "OPERATIONS",
    "VOCAB_PARALLEL_CROSS_ENTROPY",
    "LossSlots",
    "VocabParallelCrossEntropy",
    "fused_vocab_parallel_cross_entropy",
    "legacy_backends",
    "vocab_parallel_cross_entropy",
]
