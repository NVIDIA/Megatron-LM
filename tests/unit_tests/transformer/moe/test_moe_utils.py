# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs


def _reference_sort_chunks(
    input_tensor: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor
) -> torch.Tensor:
    chunks = torch.split(input_tensor, split_sizes.tolist(), dim=0)
    return torch.cat([chunks[idx] for idx in sorted_idxs.tolist()], dim=0)


@pytest.mark.internal
def test_sort_chunks_by_idxs_unfused_matches_split_cat_and_backward():
    """Check the memory-saving unfused chunk reorder against split/cat semantics."""
    split_sizes = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    sorted_idxs = torch.tensor([2, 1, 0, 3], dtype=torch.long)

    input_tensor = torch.arange(24, dtype=torch.float32).view(6, 4).requires_grad_(True)
    probs = torch.arange(6, dtype=torch.float32).requires_grad_(True)
    ref_input = input_tensor.detach().clone().requires_grad_(True)
    ref_probs = probs.detach().clone().requires_grad_(True)

    output, permuted_probs = sort_chunks_by_idxs(
        input_tensor, split_sizes, sorted_idxs, probs=probs
    )
    ref_output = _reference_sort_chunks(ref_input, split_sizes, sorted_idxs)
    ref_permuted_probs = _reference_sort_chunks(ref_probs, split_sizes, sorted_idxs)

    torch.testing.assert_close(output, ref_output)
    torch.testing.assert_close(permuted_probs, ref_permuted_probs)

    output_grad = torch.arange(output.numel(), dtype=output.dtype).view_as(output)
    probs_grad = torch.arange(permuted_probs.numel(), dtype=permuted_probs.dtype)
    torch.autograd.backward((output, permuted_probs), (output_grad, probs_grad))
    torch.autograd.backward((ref_output, ref_permuted_probs), (output_grad, probs_grad))

    torch.testing.assert_close(input_tensor.grad, ref_input.grad)
    torch.testing.assert_close(probs.grad, ref_probs.grad)
