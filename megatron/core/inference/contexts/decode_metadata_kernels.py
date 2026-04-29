# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU-side metadata preparation for steady-state dynamic decode rows."""

from __future__ import annotations

from typing import Sequence

import torch


def prepare_decode_metadata(
    gpu_view,
    *,
    decode_request_slots: Sequence[int],
    decode_input_destination_indices: Sequence[int],
    speculative_width: int,
    block_size_tokens: int,
) -> None:
    """Populate decode token metadata directly in a snapshot-bound GPU view."""
    decode_count = len(decode_input_destination_indices)
    if decode_count == 0:
        return
    if len(decode_request_slots) != decode_count:
        raise ValueError("decode request slots and destination indices must have equal length")

    tokens_per_request = int(speculative_width) + 1
    if tokens_per_request <= 0:
        raise ValueError(f"tokens_per_request must be positive, got {tokens_per_request}")

    device = gpu_view.token_to_pos_ids.device
    request_slots = torch.tensor(decode_request_slots, dtype=torch.long, device=device)
    destinations = torch.tensor(
        decode_input_destination_indices, dtype=torch.long, device=device
    )
    token_offsets = torch.arange(tokens_per_request, dtype=torch.long, device=device)
    token_indices = (destinations[:, None] + token_offsets[None, :]).reshape(-1)

    request_kv_offsets = gpu_view.request_kv_length_offsets[request_slots].to(torch.long)
    positions = (request_kv_offsets[:, None] + token_offsets[None, :]).reshape(-1)
    positions_i32 = positions.to(torch.int32)

    gpu_view.token_to_pos_ids.index_copy_(0, token_indices, positions)
    gpu_view.token_to_position_in_request.index_copy_(0, token_indices, positions_i32)
    gpu_view.token_to_local_position_within_kv_block.index_copy_(
        0, token_indices, torch.remainder(positions_i32, int(block_size_tokens))
    )
    gpu_view.token_to_request_idx.index_copy_(
        0,
        token_indices,
        request_slots.to(torch.int32).repeat_interleave(tokens_per_request),
    )

    block_columns = torch.div(
        positions, int(block_size_tokens), rounding_mode="floor"
    ).to(torch.long)
    request_slots_per_token = request_slots.repeat_interleave(tokens_per_request)
    block_ids = gpu_view.mha_block_table[request_slots_per_token, block_columns]
    gpu_view.token_to_block_idx.index_copy_(0, token_indices, block_ids)

    query_lengths = torch.full(
        (decode_count,), tokens_per_request, dtype=torch.int32, device=device
    )
    gpu_view.request_query_lengths.index_copy_(0, request_slots, query_lengths)
    gpu_view.mha_query_lengths.index_copy_(0, request_slots, query_lengths)
    kv_seq_lengths = gpu_view.request_kv_length_offsets[request_slots] + query_lengths
    gpu_view.mha_kv_seq_lengths.index_copy_(0, request_slots, kv_seq_lengths)

    cu_query = torch.arange(
        decode_count + 1, dtype=torch.int32, device=device
    ) * tokens_per_request
    gpu_view.mha_cu_query_seq_lengths[: decode_count + 1].copy_(cu_query)
    gpu_view.mha_cu_kv_seq_lengths[0].zero_()
    gpu_view.mha_cu_kv_seq_lengths[1 : decode_count + 1].copy_(
        torch.cumsum(kv_seq_lengths, dim=0)
    )
