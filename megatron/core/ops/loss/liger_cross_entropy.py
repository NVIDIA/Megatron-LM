# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Liger's vocab-parallel cross entropy with optional original-logit predictions."""

from __future__ import annotations

from typing import Literal, overload

import torch
from liger_kernel.megatron import LigerMegatronCrossEntropy  # type: ignore[import-untyped]

_cross_entropy = LigerMegatronCrossEntropy(ignore_index=-100, reduction="none", label_smoothing=0.0)


@torch.no_grad()
def _original_argmax(
    logits: torch.Tensor, tp_group: torch.distributed.ProcessGroup | None
) -> torch.Tensor:
    """Find the smallest global index attaining the maximum, using only per-token buffers."""
    local_max, indices = logits.max(dim=-1)
    if tp_group is None or tp_group.size() == 1:
        return indices

    global_max = local_max.clone()
    torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    indices.add_(tp_group.rank() * logits.size(-1))
    indices.masked_fill_(local_max != global_max, torch.iinfo(indices.dtype).max)
    torch.distributed.all_reduce(indices, op=torch.distributed.ReduceOp.MIN, group=tp_group)
    return indices


@overload
def liger_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup | None = None,
    *,
    return_argmax: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def liger_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup | None = None,
    *,
    return_argmax: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def liger_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup | None = None,
    *,
    return_argmax: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


def liger_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup | None = None,
    *,
    return_argmax: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 per-token loss, optionally with original global-vocabulary argmax ids.

    ``logits`` is [sequence, batch, local_vocab] and ``labels`` contains global ids in
    [sequence, batch]. The supplied group owns all vocabulary reductions; ``None`` means
    local vocabulary even when distributed is initialized. Liger owns autograd, including
    per-token upstream gradients and conversion back to the incoming logits dtype.

    Like the other CE targets, this target may consume logits. Compute predictions before
    calling Liger: implementations that store gradients in logits during forward no longer
    preserve their argmax. Only [sequence, batch] buffers are needed, not a logits clone.
    """
    predictions = _original_argmax(logits, tp_group) if return_argmax else None
    loss = _cross_entropy(logits, labels, tp_group=tp_group).float()
    return (loss, predictions) if predictions is not None else loss
