# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Loss: what a backend must provide, and which ones exist.

The implementations are in :mod:`.backends`; a provider in
``megatron.core.models.backends`` or ``megatron.core.extensions`` picks between them.

**Contract for vocab-parallel cross entropy**

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

from megatron.core.ops.loss.backends import (
    LossMegatron,
    LossMegatronFused,
    LossTEFused,
    fused_vocab_parallel_cross_entropy,
    vocab_parallel_cross_entropy,
)

__all__ = [
    "LossMegatron",
    "LossMegatronFused",
    "LossTEFused",
    "fused_vocab_parallel_cross_entropy",
    "vocab_parallel_cross_entropy",
]
