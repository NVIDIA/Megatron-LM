# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Experimental shared-projection replay with one recurrent scan per forest."""

from functools import lru_cache

import torch
from torch import Tensor

from megatron.core.models.hybrid.shared_prefix_layout import (
    SharedPrefixForestLayout,
    SharedPrefixLayout,
)
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.ssm.mamba_sequence_packing import scan_mamba_packed_recurrence


@lru_cache(maxsize=32)
def forest_replay_indices(
    roots: tuple[tuple[int, tuple[int, ...]], ...], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, tuple[int, ...]]:
    """Cache expansion, output selection and ordered backward contribution maps."""
    if not roots or any(p < 1 or not cs or min(cs) < 1 for p, cs in roots):
        raise ValueError("Forest replay requires positive prefix and completion spans")
    physical = sum(p + sum(cs) for p, cs in roots)
    expanded = sum(len(cs) * p + sum(cs) for p, cs in roots)
    copies = max(len(cs) for _, cs in roots)
    dense_indices = []
    output_indices = []
    # The sentinel addresses the appended zero row in the gradient. Each column
    # contributes once to a physical prefix token, in reverse sibling order.
    contributors = [[expanded] * physical for _ in range(copies)]
    lengths = []
    source_offset = dense_offset = 0
    for prefix, completions in roots:
        output_indices.extend(range(dense_offset, dense_offset + prefix))
        completion_offset = source_offset + prefix
        for branch, length in enumerate(completions):
            dense_indices.extend(range(source_offset, source_offset + prefix))
            dense_indices.extend(range(completion_offset, completion_offset + length))
            output_indices.extend(range(dense_offset + prefix, dense_offset + prefix + length))
            column = len(completions) - 1 - branch
            contributors[column][source_offset : source_offset + prefix] = range(
                dense_offset, dense_offset + prefix
            )
            contributors[0][completion_offset : completion_offset + length] = range(
                dense_offset + prefix, dense_offset + prefix + length
            )
            lengths.append(prefix + length)
            completion_offset += length
            dense_offset += prefix + length
        source_offset = completion_offset
    assert source_offset == physical and dense_offset == expanded
    assert len(dense_indices) == expanded and len(output_indices) == physical
    return (
        torch.tensor(dense_indices, device=device, dtype=torch.long),
        torch.tensor(output_indices, device=device, dtype=torch.long),
        torch.tensor(contributors, device=device, dtype=torch.long),
        tuple(lengths),
    )


class _ExpandForest(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, dense_indices, contributors):
        """Gather logical dense branch copies from one physical forest."""
        ctx.save_for_backward(contributors)
        return projected.index_select(0, dense_indices)

    @staticmethod
    def backward(ctx, gradient):
        """Accumulate each shared row in a fixed floating-point addition order."""
        (contributors,) = ctx.saved_tensors
        # Accumulate shared-prefix contributions in a fixed FP32 order, then
        # round once to the model dtype. Avoid BF16 atomic scatter accumulation.
        padded = torch.cat([gradient, gradient.new_zeros((1, *gradient.shape[1:]))], dim=0)
        accumulation_dtype = torch.float64 if gradient.dtype == torch.float64 else torch.float32
        result = torch.zeros(
            (contributors.shape[1], *gradient.shape[1:]),
            device=gradient.device,
            dtype=accumulation_dtype,
        )
        for indices in contributors.unbind(0):
            result.add_(padded.index_select(0, indices).to(accumulation_dtype))
        return result.to(gradient.dtype), None, None


def scan_mamba_forest_replay(
    mixer: MambaMixer, projected: Tensor, layout: SharedPrefixLayout | SharedPrefixForestLayout
) -> Tensor:
    """Expand recurrence fields only; run all roots in a single packed scan.

    Prefix projection, gating, attention and MoE remain shared. The recurrent
    computation repeats prefixes so that every packed sequence starts from zero,
    enabling the ordinary chunk-aligned packed scan without varlen state-fork
    backward support. This trades arithmetic for fewer scan calls and removes
    the max-sibling rectangle padding used by the state-fork implementation.
    """
    if projected.shape[0] < layout.total_len:
        raise ValueError("Forest replay input is shorter than its layout")
    roots = []
    for offset, root in layout.iter_roots():
        lengths = list(root.completion_lens)
        if offset + root.total_len == layout.total_len:
            lengths[-1] += projected.shape[0] - layout.total_len
        roots.append((root.prefix_len, tuple(lengths)))
    dense_indices, output_indices, contributors, lengths = forest_replay_indices(
        tuple(roots), projected.device
    )
    expanded = _ExpandForest.apply(projected, dense_indices, contributors)
    output = scan_mamba_packed_recurrence(mixer, expanded, lengths)
    return output.index_select(0, output_indices)
