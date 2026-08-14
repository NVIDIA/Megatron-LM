# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Direct decode admission for imported prefill state."""

from __future__ import annotations

import math

import torch


def additional_decode_blocks(prompt_length: int, input_token_count: int, block_size: int) -> int:
    """Return blocks needed beyond those containing the imported prompt."""

    prompt_blocks = math.ceil(prompt_length / block_size)
    blocks_after_input = math.ceil((prompt_length + input_token_count) / block_size)
    return blocks_after_input - prompt_blocks


def can_admit_prefilled_decode(context, input_token_count: int) -> bool:
    """Return whether an imported request can enter the live decode batch."""

    if context.num_prefill_requests != 0 or context.chunked_prefill_request_id != -1:
        raise RuntimeError("A decode-only engine cannot execute prefill requests")
    if context.total_request_count >= context.max_requests:
        return False
    if context.active_token_count + input_token_count > context.max_tokens:
        return False
    return True


def admit_prefilled_decode(
    context,
    request,
    prompt_block_ids: list[int],
    continuation_block_ids: list[int],
    input_tokens: list[int],
) -> None:
    """Admit imported KV state directly as an active decode request.

    The imported prompt blocks already carry one allocator reference. This
    function transfers that ownership to the live request and schedules the
    sampled first token (plus any MTP proposals) as the next decode query.
    """

    input_token_count = len(input_tokens)
    if not can_admit_prefilled_decode(context, input_token_count):
        raise RuntimeError("Imported prefill state cannot enter the current decode batch")

    prompt_length = len(request.prompt_tokens)
    expected_prompt_blocks = math.ceil(prompt_length / context.block_size_tokens)
    if len(prompt_block_ids) != expected_prompt_blocks:
        raise ValueError(
            f"Expected {expected_prompt_blocks} imported prompt blocks, "
            f"got {len(prompt_block_ids)}"
        )
    expected_decode_blocks = additional_decode_blocks(
        prompt_length, input_token_count, context.block_size_tokens
    )
    if len(continuation_block_ids) != expected_decode_blocks:
        raise ValueError(
            f"Expected {expected_decode_blocks} decode continuation blocks, "
            f"got {len(continuation_block_ids)}"
        )
    if input_token_count != context.num_speculative_tokens + 1:
        raise ValueError(
            "Imported decode inputs must contain one sampled token plus the configured "
            f"MTP proposals: expected {context.num_speculative_tokens + 1}, "
            f"got {input_token_count}"
        )
    if request.get_metadata_types() != context.request_metadata_types:
        raise ValueError("Imported request metadata does not match the decode context")

    current_id = context.total_request_count
    all_block_ids = prompt_block_ids + continuation_block_ids
    row = context.request_to_kv_block_ids[current_id]
    row.fill_(-1)
    row[: len(all_block_ids)] = torch.tensor(all_block_ids, dtype=row.dtype, device=row.device)

    context.request_ids[current_id] = request.request_id
    for metadata, metadata_type in zip(request.tracked_metadata, request.get_metadata_types()):
        label, _ = metadata_type
        value = metadata
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(
                value,
                device=context.request_metadata[label].device,
                dtype=context.request_metadata[label].dtype,
            )
        context.request_metadata[label][current_id] = value

    context.request_kv_length_offsets[current_id] = prompt_length
    context.request_query_lengths[current_id] = input_token_count
    context.request_output_lengths[current_id] = (
        prompt_length + request.sampling_params.num_tokens_to_generate
    )
    context.request_in_prefill_status_tensor[current_id] = 0
    context.request_kv_block_counts[current_id] = len(all_block_ids)
    context.request_last_kv_block_id[current_id] = all_block_ids[-1]
    context.request_last_kv_block_offset[current_id] = (
        prompt_length - 1 + input_token_count
    ) % context.block_size_tokens

    token_start = context.active_token_count
    token_end = token_start + input_token_count
    token_positions = torch.arange(
        prompt_length,
        prompt_length + input_token_count,
        dtype=context.token_to_pos_ids.dtype,
        device=context.token_to_pos_ids.device,
    )
    context.token_to_input_ids[token_start:token_end] = torch.tensor(
        input_tokens,
        dtype=context.token_to_input_ids.dtype,
        device=context.token_to_input_ids.device,
    )
    context.token_to_pos_ids[token_start:token_end] = token_positions
    context.token_to_request_idx[token_start:token_end] = current_id
    context.token_to_position_in_request[token_start:token_end] = token_positions
    context.token_to_local_position_within_kv_block[token_start:token_end] = (
        token_positions % context.block_size_tokens
    )
    block_ids_tensor = torch.tensor(
        all_block_ids,
        dtype=context.token_to_block_idx.dtype,
        device=context.token_to_block_idx.device,
    )
    context.token_to_block_idx[token_start:token_end] = block_ids_tensor[
        token_positions // context.block_size_tokens
    ]

    context.active_token_count = token_end
    context.total_request_count += 1

    request.remaining_prompt_tokens = request.remaining_prompt_tokens.new_empty(0)
    request.add_event_add_context()
