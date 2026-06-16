# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Vocab-parallel cross entropy loss (copied from Megatron-Core).

Computes cross entropy when logits are split across TP ranks.
With TP=1 the all-reduce calls are no-ops and this degenerates to a
standard cross-entropy with in-place ops and a memory-efficient custom backward.
"""

from __future__ import annotations

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]


def _vocab_range(partition_vocab_size: int, rank: int, world_size: int):
    start = rank * partition_vocab_size
    return start, start + partition_vocab_size


class _VocabParallelCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, tp_group):
        # Cast to float32 and compute max for numerical stability.
        vocab_parallel_logits = vocab_parallel_logits.float()
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]

        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        # In-place subtract max.
        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)

        # Partition info.
        partition_vocab_size = vocab_parallel_logits.size(-1)
        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            rank = dist.get_rank(tp_group)
            world_size = dist.get_world_size(tp_group)
        else:
            rank = 0
            world_size = 1
        vocab_start_index, vocab_end_index = _vocab_range(partition_vocab_size, rank, world_size)

        # Mask targets outside this partition's vocab range.
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted logits = logits[target].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(logits_2d.size(0), device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        # Sum of exp(logits).
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        # All-reduce predicted_logits and sum_exp_logits across TP.
        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=tp_group)
            dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Loss = log(sum(exp(logits))) - predicted_logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize exp_logits to get softmax (reused in backward).
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        # Save for backward.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # grad_input = softmax (copy is implicit since softmax is saved).
        grad_input = softmax
        partition_vocab_size = softmax.size(-1)
        grad_2d = grad_input.view(-1, partition_vocab_size)

        arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)
        softmax_update = 1.0 - target_mask.view(-1).float()

        grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Scale by upstream gradient.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group=None):
    """Cross entropy loss for vocab-parallel logits.

    Args:
        vocab_parallel_logits: [S, B, V/tp] logits split across TP ranks.
        target: [S, B] integer target token ids.
        tp_group: TP process group (None or single-rank group → no communication).

    Returns:
        Per-token loss tensor of shape [S, B].
    """
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, tp_group)


__all__ = ["vocab_parallel_cross_entropy"]
