# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Branch layout copies with one destination allocation in each backward.

Ordinary differentiable slice assignments build a CopySlices chain. Its backward
can repeatedly materialize the entire branch rectangle or packed input. These
operators keep the same forward copies and explicitly assemble each input
gradient once. They do not change convolution or recurrent scan arithmetic.
"""

import torch
from torch import Tensor


class _PackBranches(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, prefix_len, tail_len, lengths):
        """Pack the shared prompt tail and completions into a padded branch rectangle."""
        if projected.ndim != 3 or projected.shape[1] != 1:
            raise ValueError("Mamba branch packing requires [sequence, 1, channels]")
        if not lengths or min(lengths) < 0 or not 0 <= tail_len <= prefix_len:
            raise ValueError("Invalid Mamba prefix tail or completion lengths")
        if prefix_len + sum(lengths) != projected.shape[0]:
            raise ValueError("Mamba branches must cover the physical projected input")
        ctx.prefix_len = prefix_len
        ctx.tail_len = tail_len
        ctx.lengths = lengths
        ctx.input_shape = projected.shape
        branches = projected.new_zeros(tail_len + max(lengths), len(lengths), projected.shape[-1])
        start = prefix_len
        for branch, length in enumerate(lengths):
            if tail_len:
                branches[:tail_len, branch] = projected[prefix_len - tail_len : prefix_len, 0]
            branches[tail_len : tail_len + length, branch] = projected[start : start + length, 0]
            start += length
        return branches

    @staticmethod
    def backward(ctx, grad_branches):
        """Restore completion gradients and sum shared tail contributions in reverse order."""
        gradient = grad_branches.new_zeros(ctx.input_shape)
        end = ctx.input_shape[0]
        # Reverse branch order follows the original slice-assignment graph. Keep
        # accumulation in the input dtype, including BF16 rounding at each add.
        for branch in range(len(ctx.lengths) - 1, -1, -1):
            length = ctx.lengths[branch]
            start = end - length
            gradient[start:end, 0] = grad_branches[ctx.tail_len : ctx.tail_len + length, branch]
            if ctx.tail_len:
                gradient[ctx.prefix_len - ctx.tail_len : ctx.prefix_len, 0].add_(
                    grad_branches[: ctx.tail_len, branch]
                )
            end = start
        return gradient, None, None, None


class _MergeBranches(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prefix_head, branches, tail_len, lengths):
        """Join the unique prompt and unpadded completion rows in physical order."""
        if prefix_head.ndim != 3 or prefix_head.shape[1] != 1:
            raise ValueError("Mamba prefix head requires [sequence, 1, channels]")
        if (
            branches.ndim != 3
            or branches.shape[1] != len(lengths)
            or branches.shape[-1] != prefix_head.shape[-1]
            or not lengths
            or min(lengths) < 0
            or tail_len < 0
            or tail_len + max(lengths) != branches.shape[0]
        ):
            raise ValueError("Mamba branch rectangle does not match its layout")
        ctx.head_len = prefix_head.shape[0]
        ctx.branch_shape = branches.shape
        ctx.tail_len = tail_len
        ctx.lengths = lengths
        return torch.cat(
            [prefix_head, branches[:tail_len, :1]]
            + [
                branches[tail_len : tail_len + length, branch : branch + 1]
                for branch, length in enumerate(lengths)
            ],
            dim=0,
        )

    @staticmethod
    def backward(ctx, gradient):
        """Distribute gradients to the unique prompt head and selected branch rows."""
        grad_head = gradient[: ctx.head_len]
        grad_branches = gradient.new_zeros(ctx.branch_shape)
        start = ctx.head_len
        grad_branches[: ctx.tail_len, :1] = gradient[start : start + ctx.tail_len]
        start += ctx.tail_len
        for branch, length in enumerate(ctx.lengths):
            grad_branches[ctx.tail_len : ctx.tail_len + length, branch : branch + 1] = gradient[
                start : start + length
            ]
            start += length
        return grad_head, grad_branches, None, None


def pack_mamba_branches(
    projected: Tensor, *, prefix_len: int, tail_len: int, completion_lens: tuple[int, ...]
) -> Tensor:
    """Copy a shared tail and disjoint completions into a padded branch batch."""
    return _PackBranches.apply(projected, prefix_len, tail_len, completion_lens)


def merge_mamba_branches(
    prefix_head: Tensor, branches: Tensor, *, tail_len: int, completion_lens: tuple[int, ...]
) -> Tensor:
    """Restore one prefix and all completions, ignoring sibling tail duplicates."""
    return _MergeBranches.apply(prefix_head, branches, tail_len, completion_lens)
