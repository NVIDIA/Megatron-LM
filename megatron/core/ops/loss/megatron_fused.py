# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron's fused vocab-parallel cross entropy."""

from __future__ import annotations

import torch

from megatron.core.fusions.fused_cross_entropy import (
    fused_vocab_parallel_cross_entropy as _fused_vocab_parallel_cross_entropy,
)

__all__ = ["Loss", "fused_vocab_parallel_cross_entropy"]


def fused_vocab_parallel_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Megatron's compiled vocab-parallel cross entropy, in family contract order.

    The kernel reads rank and size straight off the group, so unlike the reference target it
    cannot take ``None``. Filling that in is this adapter's job, not the caller's.
    """
    if tp_group is None:
        from megatron.core.parallel_state import get_tensor_model_parallel_group

        tp_group = get_tensor_model_parallel_group()
    return _fused_vocab_parallel_cross_entropy(logits, labels, tp_group)


class Loss:
    """Owns ``vocab_parallel_cross_entropy`` using Megatron's compiled reductions."""

    def vocab_parallel_cross_entropy(self):
        """Return the fused target."""
        return fused_vocab_parallel_cross_entropy
