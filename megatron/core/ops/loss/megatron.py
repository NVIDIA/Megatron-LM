# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reference vocab-parallel cross entropy."""

from __future__ import annotations

import torch

from megatron.core.tensor_parallel.cross_entropy import (
    vocab_parallel_cross_entropy as _vocab_parallel_cross_entropy,
)

__all__ = ["Loss", "vocab_parallel_cross_entropy"]


def vocab_parallel_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Megatron's custom-autograd vocab-parallel cross entropy, in family contract order."""
    return _vocab_parallel_cross_entropy(logits, labels, tp_group=tp_group)


class Loss:
    """Owns ``vocab_parallel_cross_entropy`` using Megatron's unfused implementation."""

    def vocab_parallel_cross_entropy(self):
        """Return the reference target."""
        return vocab_parallel_cross_entropy
