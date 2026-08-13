# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batch-invariant MoE permutation helpers."""

from typing import Optional

import torch

from megatron.core import parallel_state


def build_inverse_permutation_map(
    routing_map: torch.Tensor,
    flat_sorted: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_out_tokens: int,
) -> torch.Tensor:
    """Build token/top-k -> permuted-row and expert-id map for batch-invariant unpermute.

    The regular permutation map is row -> token. Batch-invariant unpermute needs
    the inverse ownership model so each output token can read its routed rows and
    add them in a fixed order.
    """
    num_tokens = routing_map.size(0)
    assert isinstance(
        num_out_tokens, int
    ), "batch-invariant graph unpermute requires static num_out_tokens"
    assert (
        num_out_tokens % num_tokens == 0
    ), "batch-invariant graph unpermute expects fixed top-k per token"

    topk = num_out_tokens // num_tokens
    row_ids = torch.arange(num_out_tokens, device=routing_map.device, dtype=torch.long)
    expert_ids = torch.div(flat_sorted, num_tokens, rounding_mode='floor').to(torch.long)
    token_ids = sorted_indices.to(torch.long)

    slots_by_token_expert = routing_map.bool().to(torch.long).cumsum(dim=1) - 1
    row_slots = slots_by_token_expert[token_ids, expert_ids]
    linear_slots = token_ids * topk + row_slots

    inverse_rows = torch.full((num_tokens, topk), -1, device=routing_map.device, dtype=torch.long)
    inverse_experts = torch.full(
        (num_tokens, topk), -1, device=routing_map.device, dtype=torch.long
    )
    inverse_rows.view(-1).scatter_(0, linear_slots, row_ids)
    inverse_experts.view(-1).scatter_(0, linear_slots, expert_ids)
    return torch.stack((inverse_rows, inverse_experts), dim=0)


def unpermute(
    permuted_tokens: torch.Tensor,
    restore_shape: torch.Size,
    *,
    probs: Optional[torch.Tensor],
    num_experts: int,
    inverse_map: torch.Tensor,
) -> torch.Tensor:
    """Batch-invariant MoE unpermute.

    Accumulation is token-owned. The AllToAll inverse map avoids data-dependent
    shapes and adds contributions by EP rank then top-k slot, matching the
    inference NVLS rank-ordered combine.
    """
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        get_batch_invariant_collective,
    )

    input_dtype = permuted_tokens.dtype
    # Mirror the inference engine's summation tree exactly:
    # - When probs is None the activations were probability-weighted upstream
    #   (gated/SwiGLU convention) and the engine's within-rank _moe_sum
    #   accumulates in fp64 with unit weights -> mirror in fp64, round each
    #   rank partial to fp32 (the engine stores fp32 partials into the RSV
    #   buffer before the collective).
    # - The cross-rank stage matches the configured collective: "multimem"
    #   returns the correctly-rounded exact fp32 sum of the partials (mirror =
    #   fp64 accumulate, single final rounding); "ordered" is an ascending
    #   rank-order fp32 chain.
    within_rank_fp64 = probs is None
    cross_rank_fp64 = get_batch_invariant_collective() == "multimem"
    acc_dtype = torch.float64 if cross_rank_fp64 else torch.float32
    output_tokens = torch.zeros(restore_shape, dtype=acc_dtype, device=permuted_tokens.device)
    ep_size = parallel_state.get_expert_model_parallel_world_size() or 1
    assert num_experts % ep_size == 0, "batch-invariant MoE expects contiguous EP shards"
    experts_per_rank = num_experts // ep_size
    inverse_rows = inverse_map[0]
    inverse_experts = inverse_map[1]
    topk = inverse_rows.size(1)

    partial_dtype = torch.float64 if within_rank_fp64 else torch.float32
    for ep_rank in range(ep_size):
        rank_partial = torch.zeros(
            restore_shape, dtype=partial_dtype, device=permuted_tokens.device
        )
        start_expert = ep_rank * experts_per_rank
        end_expert = start_expert + experts_per_rank

        for k in range(topk):
            row_ids = inverse_rows[:, k]
            expert_ids = inverse_experts[:, k]
            valid_mask = (row_ids >= 0) & (expert_ids >= start_expert) & (expert_ids < end_expert)

            safe_rows = row_ids.clamp_min(0)
            chunk = permuted_tokens.index_select(0, safe_rows).to(partial_dtype)
            if probs is not None:
                safe_experts = expert_ids.clamp_min(0)
                chunk = chunk * probs.gather(1, safe_experts.unsqueeze(1)).to(partial_dtype)
            chunk.masked_fill_(~valid_mask.unsqueeze(-1), 0.0)
            rank_partial += chunk

        # The engine stores each rank's partial as fp32 (the RSV buffer dtype)
        # before the cross-rank reduction; mirror that rounding point.
        output_tokens += rank_partial.to(torch.float32).to(acc_dtype)

    return output_tokens.to(torch.float32).to(dtype=input_dtype)
