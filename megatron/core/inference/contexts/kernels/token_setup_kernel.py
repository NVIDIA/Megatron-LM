# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False


@triton.jit
def _token_setup_kernel(
    # --- Per-request tensors (int32, shape [max_requests]) --- read+write
    request_kv_length_offsets_ptr,
    request_query_lengths_ptr,
    request_last_kv_block_offset_ptr,
    # --- Per-request tensors (int32, shape [max_requests]) --- read only
    request_last_kv_block_id_ptr,
    # --- Token inputs ---
    next_tokens_ptr,  # int64, shape [total_request_count]
    new_speculative_tokens_ptr,  # int64, shape [num_spec * total_request_count] (or dummy)
    prev_last_block_ids_ptr,  # int32, shape [max_requests] (or dummy)
    # --- Per-token outputs (int64, shape [max_tokens]) ---
    token_to_input_ids_ptr,
    token_to_pos_ids_ptr,
    token_to_request_idx_ptr,
    token_to_position_in_request_ptr,
    token_to_local_position_within_kv_block_ptr,
    token_to_block_idx_ptr,
    # --- Scalar parameters ---
    paused_request_count: tl.int32,
    spec_stride: tl.int32,
    block_size_tokens: tl.int32,
    # --- Compile-time constants ---
    NUM_GENERATED_TOKENS: tl.constexpr,
    HAS_SPECULATIVE: tl.constexpr,
):
    """Fused token setup kernel: per-request bookkeeping updates + per-token tensor population.

    Grid: (active_request_count,)
    Each program handles one active request and writes NUM_GENERATED_TOKENS output entries.
    """
    active_idx = tl.program_id(0)
    req_idx = active_idx + paused_request_count

    # --- Load per-request inputs ---
    old_kv_offset = tl.load(request_kv_length_offsets_ptr + req_idx)
    old_query_len = tl.load(request_query_lengths_ptr + req_idx)
    old_block_offset = tl.load(request_last_kv_block_offset_ptr + req_idx)
    current_block_id = tl.load(request_last_kv_block_id_ptr + req_idx)

    # --- Request-level in-place updates ---
    new_kv_offset = old_kv_offset + old_query_len
    tl.store(request_kv_length_offsets_ptr + req_idx, new_kv_offset)
    tl.store(request_query_lengths_ptr + req_idx, NUM_GENERATED_TOKENS)
    new_block_offset = (old_block_offset + NUM_GENERATED_TOKENS) % block_size_tokens
    tl.store(request_last_kv_block_offset_ptr + req_idx, new_block_offset)

    # --- Preload speculative-only data (hoisted out of loop) ---
    if HAS_SPECULATIVE:
        prev_block_id = tl.load(prev_last_block_ids_ptr + req_idx)
        any_crosses = (old_block_offset + NUM_GENERATED_TOKENS) >= block_size_tokens

    # --- Per-token loop (compile-time unrolled) ---
    token_base = active_idx * NUM_GENERATED_TOKENS
    for t in tl.static_range(NUM_GENERATED_TOKENS):
        out_idx = token_base + t

        # token_to_input_ids: sampled token (t=0) or speculative token (t>0)
        if t == 0:
            token_id = tl.load(next_tokens_ptr + req_idx).to(tl.int64)
        else:
            if HAS_SPECULATIVE:
                spec_offset = (t - 1) * spec_stride + req_idx
                token_id = tl.load(new_speculative_tokens_ptr + spec_offset).to(tl.int64)
            else:
                token_id = tl.cast(0, tl.int64)
        tl.store(token_to_input_ids_ptr + out_idx, token_id)

        # token_to_pos_ids and token_to_position_in_request
        pos_id = (new_kv_offset + t).to(tl.int64)
        tl.store(token_to_pos_ids_ptr + out_idx, pos_id)
        tl.store(token_to_position_in_request_ptr + out_idx, pos_id)

        # token_to_request_idx
        tl.store(token_to_request_idx_ptr + out_idx, tl.cast(req_idx, tl.int64))

        # token_to_local_position_within_kv_block
        local_pos = pos_id % block_size_tokens
        tl.store(token_to_local_position_within_kv_block_ptr + out_idx, local_pos)

        # token_to_block_idx (with block boundary crossing logic for speculative)
        if HAS_SPECULATIVE:
            raw_pos = old_block_offset + 1 + t
            this_crosses = raw_pos >= block_size_tokens
            use_prev = any_crosses & (~this_crosses)
            if use_prev:
                block_id = prev_block_id.to(tl.int64)
            else:
                block_id = current_block_id.to(tl.int64)
        else:
            block_id = current_block_id.to(tl.int64)
        tl.store(token_to_block_idx_ptr + out_idx, block_id)


def triton_token_setup(
    request_kv_length_offsets: Tensor,
    request_query_lengths: Tensor,
    request_last_kv_block_offset: Tensor,
    request_last_kv_block_id: Tensor,
    next_tokens: Tensor,
    new_speculative_tokens: Optional[Tensor],
    prev_last_block_ids: Optional[Tensor],
    token_to_input_ids: Tensor,
    token_to_pos_ids: Tensor,
    token_to_request_idx: Tensor,
    token_to_position_in_request: Tensor,
    token_to_local_position_within_kv_block: Tensor,
    token_to_block_idx: Tensor,
    paused_request_count: int,
    total_request_count: int,
    block_size_tokens: int,
    num_speculative_tokens: int,
) -> int:
    """Fused token setup: update request bookkeeping and populate per-token tensors.

    Returns:
        active_token_count: Number of tokens written.
    """
    active_request_count = total_request_count - paused_request_count
    num_generated_tokens = 1 + num_speculative_tokens
    active_token_count = active_request_count * num_generated_tokens

    if active_request_count == 0:
        return 0

    has_speculative = num_speculative_tokens > 0

    if has_speculative:
        assert new_speculative_tokens is not None
        # Use the actual memory stride between rows, not shape[1]. The tensor
        # may be a non-contiguous slice (e.g. original[:, start:end]) where
        # stride(0) > shape[1].
        spec_stride = new_speculative_tokens.stride(0)
    else:
        spec_stride = 0

    # Triton requires valid pointers for all arguments even when HAS_SPECULATIVE=False.
    # Pass dummy tensors that are never dereferenced on that code path.
    spec_ptr = new_speculative_tokens if has_speculative else next_tokens
    prev_ptr = prev_last_block_ids if has_speculative else request_last_kv_block_id

    grid = (active_request_count,)

    _token_setup_kernel[grid](
        request_kv_length_offsets,
        request_query_lengths,
        request_last_kv_block_offset,
        request_last_kv_block_id,
        next_tokens,
        spec_ptr,
        prev_ptr,
        token_to_input_ids,
        token_to_pos_ids,
        token_to_request_idx,
        token_to_position_in_request,
        token_to_local_position_within_kv_block,
        token_to_block_idx,
        paused_request_count=paused_request_count,
        spec_stride=spec_stride,
        block_size_tokens=block_size_tokens,
        NUM_GENERATED_TOKENS=num_generated_tokens,
        HAS_SPECULATIVE=has_speculative,
    )

    return active_token_count


@triton.jit
def _find_chunked_prefill_kernel(
    request_ids_ptr,
    chunked_prefill_request_id: tl.int32,
    total_request_count: tl.int32,
    found_idx_ptr,
    is_in_bounds_ptr,
    swap_src_ptr,
    swap_dst_ptr,
    search_limit: tl.int32,
):
    """Find the chunked prefill request and compute swap indices. Grid: (1,)."""
    found = -1
    i = 0
    while i < search_limit:
        rid = tl.load(request_ids_ptr + i)
        if rid == chunked_prefill_request_id:
            found = i
            i = search_limit  # break
        else:
            i += 1

    tl.store(found_idx_ptr, found)

    if found >= 0:
        if found < total_request_count:
            # Path A: in-bounds → swap to total_request_count - 1
            tl.store(is_in_bounds_ptr, 1)
            tl.store(swap_src_ptr, found)
            tl.store(swap_dst_ptr, total_request_count - 1)
        else:
            # Path B: out-of-bounds → slide to total_request_count
            tl.store(is_in_bounds_ptr, 0)
            tl.store(swap_src_ptr, found)
            tl.store(swap_dst_ptr, total_request_count)
    else:
        tl.store(is_in_bounds_ptr, 0)


def triton_find_chunked_prefill(
    request_ids: Tensor,
    chunked_prefill_request_id: int,
    total_request_count: int,
    search_limit: int,
) -> tuple:
    """Find chunked prefill request index and compute swap indices.

    Returns:
        (found_idx, is_in_bounds, swap_src, swap_dst)
    """
    device = request_ids.device
    found_idx_buf = torch.tensor([-1], dtype=torch.int32, device=device)
    is_in_bounds_buf = torch.zeros(1, dtype=torch.int32, device=device)
    swap_src_buf = torch.zeros(1, dtype=torch.int32, device=device)
    swap_dst_buf = torch.zeros(1, dtype=torch.int32, device=device)

    _find_chunked_prefill_kernel[(1,)](
        request_ids,
        chunked_prefill_request_id,
        total_request_count,
        found_idx_buf,
        is_in_bounds_buf,
        swap_src_buf,
        swap_dst_buf,
        search_limit,
    )

    return (
        found_idx_buf.item(),
        is_in_bounds_buf.item(),
        swap_src_buf,
        swap_dst_buf,
    )
