# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Probability-first primitives for Megatron Lite phase 1."""

from __future__ import annotations

import torch
import torch.distributed as dist


def vocab_parallel_log_probs_from_logits(
    logits: torch.Tensor, labels: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute log-probabilities from already materialized logits.

    Phase 1 assumes logits are already full-vocab tensors on the local rank.
    If `labels` are provided, this returns token-selected log-probabilities with
    the same shape as `labels` (modulo an internal transpose when logits are
    `[S, B, V]` and labels are `[B, S]`).
    """

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    if labels is None:
        return log_probs

    aligned_labels, transposed = _align_labels_to_logits(logits, labels)
    gathered = log_probs.gather(dim=-1, index=aligned_labels.unsqueeze(-1)).squeeze(-1)
    return gathered.transpose(0, 1).contiguous() if transposed else gathered


def _all_reduce_if_needed(tensor: torch.Tensor, group=None, op=dist.ReduceOp.SUM) -> torch.Tensor:
    if group is not None and dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.all_reduce(tensor, op=op, group=group)
    return tensor


def vocab_parallel_entropy(logits: torch.Tensor, tp_group=None) -> torch.Tensor:
    """Compute per-token entropy from logits."""

    logits = logits.float()
    logits_max = logits.max(dim=-1).values
    _all_reduce_if_needed(logits_max, tp_group, op=dist.ReduceOp.MAX)

    shifted = logits - logits_max.unsqueeze(-1)
    exp_logits = torch.exp(shifted)
    sum_exp = exp_logits.sum(dim=-1)
    _all_reduce_if_needed(sum_exp, tp_group)

    weighted_logits = (exp_logits * logits).sum(dim=-1)
    _all_reduce_if_needed(weighted_logits, tp_group)
    expected_logits = weighted_logits / sum_exp
    return torch.log(sum_exp) + logits_max - expected_logits


def _align_labels_to_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, bool]:
    if logits.ndim != labels.ndim + 1:
        raise ValueError(
            f"logits rank must be labels rank + 1, got logits={logits.shape}, labels={labels.shape}."
        )

    if logits.shape[:-1] == labels.shape:
        return labels, False

    if (
        logits.ndim == 3
        and labels.ndim == 2
        and logits.shape[0] == labels.shape[1]
        and logits.shape[1] == labels.shape[0]
    ):
        return labels.transpose(0, 1).contiguous(), True

    raise ValueError(f"Could not align labels {labels.shape} with logits {logits.shape}.")
