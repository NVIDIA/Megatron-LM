# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Reference semantics shared by the MoE megakernel tests."""

from typing import Tuple

import torch


def drop_padding_routes(
    routing_probs: torch.Tensor, routing_map: torch.Tensor, padding_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove every route of the padding tokens from a dense routing result.

    Reference semantics of the ``padding_mask`` that ``MoELayer`` passes to a megakernel backend
    (``MegakernelBackend.forward``). The router keeps padding tokens in its routes and excludes
    them only from its statistics; the backend gives them no routes, so their routed output and
    their probs gradient are zero. The shared expert is separate and not masked here. MOK drops
    these routes during route compaction (``routing_map_to_mok_inputs`` with ``padding_mask``)
    without materializing masked copies; the tests compare it with this function.

    Args:
        routing_probs (torch.Tensor): Tensor of shape [num_tokens, num_experts] containing the
            routing probabilities.
        routing_map (torch.Tensor): Boolean tensor of shape [num_tokens, num_experts]
            indicating which experts were selected for each token.
        padding_mask (torch.Tensor): Batch-first [bsz, seq_length] mask, True marks padding.
            Tokens are flattened from sequence-first [seq_length, bsz] hidden states.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The routing probabilities and routing map with the
            rows of padding tokens zeroed. New tensors; the inputs are not modified.
    """
    assert routing_map.dtype == torch.bool, "drop_padding_routes expects a boolean routing map"
    padding_rows = padding_mask.transpose(0, 1).reshape(-1, 1).bool()
    assert padding_rows.shape[0] == routing_map.shape[0], (
        f"padding_mask has {padding_rows.shape[0]} tokens, "
        f"routing_map has {routing_map.shape[0]}"
    )
    return routing_probs.masked_fill(padding_rows, 0), routing_map & ~padding_rows
