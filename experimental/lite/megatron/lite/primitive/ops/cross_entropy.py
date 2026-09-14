# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Vocab-parallel cross entropy loss (copied from Megatron-Core).

Computes cross entropy when logits are split across TP ranks.
With TP=1 the all-reduce calls are no-ops and this degenerates to a
standard cross-entropy with in-place ops and a memory-efficient custom backward.
"""

from __future__ import annotations

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]

_DEFAULT_CHUNK_SIZE = 1024


def _vocab_range(partition_vocab_size: int, rank: int, world_size: int):
    start = rank * partition_vocab_size
    return start, start + partition_vocab_size


def _tp_info(tp_group):
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        return dist.get_rank(tp_group), dist.get_world_size(tp_group)
    return 0, 1


def _chunk_loss_and_softmax(
    logits, target, tp_group, vocab_start_index, vocab_end_index, *, return_softmax
):
    logits = logits.float()
    logits_max = logits.max(dim=-1).values
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
    logits -= logits_max.unsqueeze(-1)

    target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
    masked_target = target - vocab_start_index
    masked_target = masked_target.masked_fill(target_mask, 0)
    row_indices = torch.arange(logits.size(0), device=logits.device)
    predicted_logits = logits[row_indices, masked_target].clone()
    predicted_logits.masked_fill_(target_mask, 0.0)

    torch.exp(logits, out=logits)
    sum_exp_logits = logits.sum(dim=-1)
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=tp_group)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=tp_group)

    loss = torch.log(sum_exp_logits) - predicted_logits
    if not return_softmax:
        return loss, None, None, None

    logits.div_(sum_exp_logits.unsqueeze(-1))
    return loss, logits, target_mask, masked_target


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, tp_group, chunk_size):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if vocab_parallel_logits.shape[:-1] != target.shape:
            raise ValueError(
                "logits leading shape must match target shape, "
                f"got {vocab_parallel_logits.shape[:-1]} and {target.shape}"
            )

        partition_vocab_size = vocab_parallel_logits.size(-1)
        rank, world_size = _tp_info(tp_group)
        vocab_start_index, vocab_end_index = _vocab_range(partition_vocab_size, rank, world_size)

        logits_2d = vocab_parallel_logits.reshape(-1, partition_vocab_size)
        target_1d = target.reshape(-1)
        loss = torch.empty(target_1d.shape, dtype=torch.float32, device=target.device)
        for start in range(0, target_1d.numel(), chunk_size):
            end = min(start + chunk_size, target_1d.numel())
            loss[start:end], _, _, _ = _chunk_loss_and_softmax(
                logits_2d[start:end],
                target_1d[start:end],
                tp_group,
                vocab_start_index,
                vocab_end_index,
                return_softmax=False,
            )

        ctx.save_for_backward(vocab_parallel_logits, target)
        ctx.tp_group = tp_group
        ctx.chunk_size = chunk_size
        ctx.vocab_start_index = vocab_start_index
        ctx.vocab_end_index = vocab_end_index
        return loss.reshape_as(target)

    @staticmethod
    def backward(ctx, grad_output):
        vocab_parallel_logits, target = ctx.saved_tensors
        partition_vocab_size = vocab_parallel_logits.size(-1)
        logits_2d = vocab_parallel_logits.reshape(-1, partition_vocab_size)
        target_1d = target.reshape(-1)
        grad_output_1d = grad_output.reshape(-1)
        grad_input = torch.empty_like(vocab_parallel_logits)
        grad_input_2d = grad_input.reshape(-1, partition_vocab_size)

        for start in range(0, target_1d.numel(), ctx.chunk_size):
            end = min(start + ctx.chunk_size, target_1d.numel())
            _, softmax, target_mask, masked_target = _chunk_loss_and_softmax(
                logits_2d[start:end],
                target_1d[start:end],
                ctx.tp_group,
                ctx.vocab_start_index,
                ctx.vocab_end_index,
                return_softmax=True,
            )
            row_indices = torch.arange(end - start, device=softmax.device)
            softmax[row_indices, masked_target] -= 1.0 - target_mask.float()
            softmax.mul_(grad_output_1d[start:end].unsqueeze(-1))
            grad_input_2d[start:end].copy_(softmax)

        return grad_input, None, None, None


def vocab_parallel_cross_entropy(
    vocab_parallel_logits, target, tp_group=None, chunk_size=_DEFAULT_CHUNK_SIZE
):
    """Cross entropy loss for vocab-parallel logits.

    Args:
        vocab_parallel_logits: [S, B, V/tp] logits split across TP ranks.
        target: [S, B] integer target token ids.
        tp_group: TP process group (None or single-rank group → no communication).
        chunk_size: Maximum number of tokens materialized in float32 at once.

    Returns:
        Per-token loss tensor of shape [S, B].
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError(f"chunk_size must be an integer, got {chunk_size!r}")
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, tp_group, chunk_size)


__all__ = ["vocab_parallel_cross_entropy"]
