# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""What a loss backend has to provide.

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

from megatron.core.ops.operations import Operation, unowned

FAMILY = "loss"

VOCAB_PARALLEL_CROSS_ENTROPY = Operation(FAMILY, "vocab_parallel_cross_entropy")

OPERATIONS = (VOCAB_PARALLEL_CROSS_ENTROPY,)

VocabParallelCrossEntropy = Callable[
    [torch.Tensor, torch.Tensor, Optional[torch.distributed.ProcessGroup]], torch.Tensor
]
"""The callable shape this family's targets have."""

__all__ = [
    "FAMILY",
    "OPERATIONS",
    "VOCAB_PARALLEL_CROSS_ENTROPY",
    "LossSlots",
    "VocabParallelCrossEntropy",
]


class LossSlots:
    """The loss slots."""

    def vocab_parallel_cross_entropy(self) -> VocabParallelCrossEntropy:
        """Which vocab-parallel cross entropy to use."""
        raise unowned(VOCAB_PARALLEL_CROSS_ENTROPY)
