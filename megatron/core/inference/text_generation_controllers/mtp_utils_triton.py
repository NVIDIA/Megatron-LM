# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import torch

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


# ---------------------------------------------------------------------------
# Kernel 1: KV-cache rewind for speculative decoding
# ---------------------------------------------------------------------------
@triton.jit
def _rewind_kv_cache_kernel(
    # Per-request input (read-only)
    ACCEPTED_COUNTS_PTR,
    PREFILL_STATUS_PTR,
    # Per-request state (read-write, updated in-place)
    LAST_KV_BLOCK_OFFSET_PTR,
    KV_LENGTH_OFFSETS_PTR,
    KV_BLOCK_COUNTS_PTR,
    LAST_KV_BLOCK_ID_PTR,
    # 2-D table [N, max_blocks] (read-write)
    KV_BLOCK_IDS_PTR,
    # Per-request outputs
    BLOCKS_TO_RELEASE_PTR,
    REMOVE_MASK_PTR,
    # Strides / limits
    kv_block_ids_stride,
    max_blocks_minus_1,
    num_active_requests,
    # Compile-time constants
    NUM_SPEC_TOKENS: tl.constexpr,
    BLOCK_SIZE_TOKENS: tl.constexpr,
):
    """Rewind KV-cache bookkeeping for one request after speculative verification.

    Grid: may be padded beyond active requests for CUDA-graph compatibility.
    Each program handles exactly one request.  Programs with
    `pid >= num_active_requests` are padding and produce safe no-op outputs.
    """
    pid = tl.program_id(0)

    # Padding programs: write safe defaults and skip all state mutation.
    if pid >= num_active_requests:
        tl.store(BLOCKS_TO_RELEASE_PTR + pid, 0)
        tl.store(REMOVE_MASK_PTR + pid, False)
        return

    # --- Load per-request scalars ---
    accepted = tl.load(ACCEPTED_COUNTS_PTR + pid)
    prefill = tl.load(PREFILL_STATUS_PTR + pid)
    last_offset = tl.load(LAST_KV_BLOCK_OFFSET_PTR + pid)
    kv_length = tl.load(KV_LENGTH_OFFSETS_PTR + pid)
    block_count = tl.load(KV_BLOCK_COUNTS_PTR + pid)
    last_block_id = tl.load(LAST_KV_BLOCK_ID_PTR + pid)

    # --- Compute rewind (zero for prefill requests) ---
    num_to_rewind = tl.where(prefill == 1, 0, NUM_SPEC_TOKENS - accepted)
    diff = last_offset - num_to_rewind
    remove = diff < 0

    # Python-style modulo: ((diff % M) + M) % M  to handle negative diff
    new_offset = ((diff % BLOCK_SIZE_TOKENS) + BLOCK_SIZE_TOKENS) % BLOCK_SIZE_TOKENS
    tl.store(LAST_KV_BLOCK_OFFSET_PTR + pid, new_offset)
    tl.store(KV_LENGTH_OFFSETS_PTR + pid, kv_length - num_to_rewind)

    # Save current last block id (will be released by caller if remove is True)
    tl.store(BLOCKS_TO_RELEASE_PTR + pid, last_block_id)

    # Decrement block count when a block boundary was crossed
    new_block_count = tl.where(remove, block_count - 1, block_count)
    tl.store(KV_BLOCK_COUNTS_PTR + pid, new_block_count)

    # Gather previous block id from the 2-D table
    kv_row_base = pid.to(tl.int64) * kv_block_ids_stride
    prev_idx = tl.maximum(new_block_count - 1, 0)
    prev_block_id = tl.load(KV_BLOCK_IDS_PTR + kv_row_base + prev_idx)

    # Conditionally update last block id
    tl.store(LAST_KV_BLOCK_ID_PTR + pid, tl.where(remove, prev_block_id, last_block_id))

    # Clear released block entry via scatter
    scatter_idx = tl.minimum(new_block_count, max_blocks_minus_1)
    current_val = tl.load(KV_BLOCK_IDS_PTR + kv_row_base + scatter_idx)
    tl.store(KV_BLOCK_IDS_PTR + kv_row_base + scatter_idx, tl.where(remove, -1, current_val))

    # Output remove mask for the caller (to release blocks outside this kernel)
    tl.store(REMOVE_MASK_PTR + pid, remove)


def rewind_kv_cache(
    accepted_counts,
    prefill_status,
    last_kv_block_offset,
    kv_length_offsets,
    kv_block_counts,
    last_kv_block_id,
    kv_block_ids,
    num_speculative_tokens,
    block_size_tokens,
    num_active_requests=None,
):
    """Launch the KV-cache rewind Triton kernel.

    Args:
        num_active_requests: Number of real (non-padding) requests. When the
            grid is padded beyond this count, the kernel skips padding
            programs so stale data in padding slots cannot corrupt
            bookkeeping.  Defaults to `accepted_counts.shape[0]` (no
            padding).

    Returns:
        (blocks_to_release, remove_mask) — same semantics as the original
        torch.compile'd `_rewind_kv_cache` (KV-cache portion only; Mamba
        state updates are handled separately by the caller).
    """
    N = accepted_counts.shape[0]
    if num_active_requests is None:
        num_active_requests = N
    if N == 0:
        return (
            torch.empty(0, device=accepted_counts.device, dtype=last_kv_block_id.dtype),
            torch.empty(0, device=accepted_counts.device, dtype=torch.bool),
        )

    blocks_to_release = torch.empty_like(last_kv_block_id)
    remove_mask = torch.empty(N, device=accepted_counts.device, dtype=torch.bool)

    _rewind_kv_cache_kernel[(N,)](
        accepted_counts,
        prefill_status,
        last_kv_block_offset,
        kv_length_offsets,
        kv_block_counts,
        last_kv_block_id,
        kv_block_ids,
        blocks_to_release,
        remove_mask,
        kv_block_ids_stride=kv_block_ids.stride(0),
        max_blocks_minus_1=kv_block_ids.shape[1] - 1,
        num_active_requests=num_active_requests,
        NUM_SPEC_TOKENS=num_speculative_tokens,
        BLOCK_SIZE_TOKENS=block_size_tokens,
    )
    return blocks_to_release, remove_mask


# ---------------------------------------------------------------------------
# Kernel 2: Verify speculative tokens
# ---------------------------------------------------------------------------
@triton.jit
def _verify_speculative_tokens_kernel(
    INPUT_TOKENS_PTR,
    OUTPUT_TOKENS_PTR,
    # Outputs
    ACCEPTED_MASK_PTR,
    LAST_ONE_INDICES_PTR,
    # Runtime scalars
    num_decode_requests,
    decode_len,
    # Compile-time constants
    STRIDE: tl.constexpr,  # num_speculative_tokens + 1
    BLOCK_SIZE: tl.constexpr,  # next_power_of_2(STRIDE)
):
    """Verify speculative tokens for one request.

    Grid: (active_request_count,)
    Programs 0..num_decode_requests-1 handle decode requests.
    Programs num_decode_requests..end handle prefill requests.
    """
    pid = tl.program_id(0)

    if pid < num_decode_requests:
        base = pid * STRIDE
        offsets = tl.arange(0, BLOCK_SIZE)
        valid = offsets < STRIDE

        input_toks = tl.load(INPUT_TOKENS_PTR + base + offsets, mask=valid, other=0)

        # Build shifted output: shifted[i] = output[i-1].
        # Position 0 uses a dummy load (always accepted regardless).
        safe_shifted = tl.where(offsets > 0, offsets - 1, 0)
        shifted_output = tl.load(OUTPUT_TOKENS_PTR + base + safe_shifted, mask=valid, other=0)

        # First token is always accepted; rest must match shifted output.
        match = tl.where(offsets == 0, 1, (input_toks == shifted_output).to(tl.int32))
        match = tl.where(valid, match, 0)

        # Consecutive acceptance via cumulative-sum trick:
        # accepted[i] iff cumsum(match)[i] == i + 1
        cumsum = tl.cumsum(match, axis=0)
        accepted = (cumsum == (offsets + 1)) & valid

        tl.store(ACCEPTED_MASK_PTR + base + offsets, accepted, mask=valid)

        accepted_count = tl.sum(accepted.to(tl.int32))
        tl.store(LAST_ONE_INDICES_PTR + pid, (base + accepted_count - 1).to(tl.int64))
    else:
        # Prefill request — single token, always accepted
        prefill_idx = decode_len + (pid - num_decode_requests)
        tl.store(ACCEPTED_MASK_PTR + prefill_idx, 1)
        tl.store(LAST_ONE_INDICES_PTR + pid, prefill_idx.to(tl.int64))


def verify_speculative_tokens(
    input_tokens, output_tokens, num_decode_requests, num_prefill_requests, num_speculative_tokens
):
    """Launch the speculative-token verification Triton kernel.

    Returns:
        (last_one_indices, accepted_tokens_mask, input_tokens)
        matching the original `_verify_speculative_tokens` signature.
    """
    if input_tokens.ndim == 2:
        input_tokens = input_tokens.squeeze(0)

    device = input_tokens.device
    active_request_count = num_decode_requests + num_prefill_requests
    stride = num_speculative_tokens + 1
    decode_len = num_decode_requests * stride

    accepted_tokens_mask = torch.zeros_like(input_tokens, dtype=torch.bool)
    last_one_indices = torch.full((active_request_count,), -1, device=device, dtype=torch.long)

    if active_request_count > 0:
        block_size = triton.next_power_of_2(stride)
        _verify_speculative_tokens_kernel[(active_request_count,)](
            input_tokens,
            output_tokens,
            accepted_tokens_mask,
            last_one_indices,
            num_decode_requests=num_decode_requests,
            decode_len=decode_len,
            STRIDE=stride,
            BLOCK_SIZE=block_size,
        )

    return last_one_indices, accepted_tokens_mask, input_tokens


# ---------------------------------------------------------------------------
# Kernel 3: Prepare speculative tokens for next forward pass
# ---------------------------------------------------------------------------
@triton.jit
def _prepare_next_forward_pass_kernel(
    OUTPUT_TOKENS_PTR,
    REQUIRED_LOGIT_INDICES_PTR,
    LAST_ONE_INDICES_PTR,
    INPUT_TOKENS_PTR,
    ACCEPTED_MASK_PTR,
    # Outputs
    SAMPLED_TOKENS_OUT_PTR,
    LAST_ACCEPTED_SEQ_OUT_PTR,
    ACCEPTED_TOKENS_OUT_PTR,
    ACCEPTED_COUNTS_OUT_PTR,
    # Strides
    accepted_tokens_out_stride,
    # Runtime scalars
    num_decode_requests,
    # Compile-time constants
    STRIDE: tl.constexpr,  # num_speculative_tokens + 1
    NUM_SPEC_TOKENS: tl.constexpr,
    SPEC_BLOCK_SIZE: tl.constexpr,  # next_power_of_2(NUM_SPEC_TOKENS)
):
    """Gather final tokens and extract accepted speculative tokens per request.

    Grid: (active_request_count,)
    """
    pid = tl.program_id(0)

    # --- Gather final sampled token and sequence index for every request ---
    idx = tl.load(LAST_ONE_INDICES_PTR + pid)
    tl.store(SAMPLED_TOKENS_OUT_PTR + pid, tl.load(OUTPUT_TOKENS_PTR + idx))
    tl.store(LAST_ACCEPTED_SEQ_OUT_PTR + pid, tl.load(REQUIRED_LOGIT_INDICES_PTR + idx))

    # --- For decode requests: extract accepted tokens and count ---
    if pid < num_decode_requests:
        base = pid * STRIDE
        spec_offsets = tl.arange(0, SPEC_BLOCK_SIZE)
        spec_valid = spec_offsets < NUM_SPEC_TOKENS
        token_positions = base + 1 + spec_offsets  # skip first (base) token

        tokens = tl.load(INPUT_TOKENS_PTR + token_positions, mask=spec_valid, other=0)
        mask_val = tl.load(ACCEPTED_MASK_PTR + token_positions, mask=spec_valid, other=0)
        accepted = mask_val != 0

        result = tl.where(accepted & spec_valid, tokens, -1)

        out_base = pid.to(tl.int64) * accepted_tokens_out_stride
        tl.store(ACCEPTED_TOKENS_OUT_PTR + out_base + spec_offsets, result, mask=spec_valid)

        count = tl.sum((accepted & spec_valid).to(tl.int64))
        tl.store(ACCEPTED_COUNTS_OUT_PTR + pid, count)


def prepare_next_forward_pass(
    num_decode_requests,
    output_tokens,
    required_logit_indices,
    last_one_indices,
    accepted_tokens_mask,
    input_tokens,
    sampled_tokens_buf,
    last_accepted_seq_buf,
    accepted_tokens_per_request,
    accepted_token_counts,
    num_speculative_tokens,
):
    """Launch the prepare-next-forward-pass Triton kernel.

    Writes results into the pre-allocated buffers provided by the caller.
    """
    active_request_count = last_one_indices.shape[0]
    if active_request_count == 0:
        return

    stride = num_speculative_tokens + 1
    spec_block_size = triton.next_power_of_2(num_speculative_tokens)

    _prepare_next_forward_pass_kernel[(active_request_count,)](
        output_tokens,
        required_logit_indices,
        last_one_indices,
        input_tokens,
        accepted_tokens_mask,
        sampled_tokens_buf,
        last_accepted_seq_buf,
        accepted_tokens_per_request,
        accepted_token_counts,
        accepted_tokens_out_stride=accepted_tokens_per_request.stride(0),
        num_decode_requests=num_decode_requests,
        STRIDE=stride,
        NUM_SPEC_TOKENS=num_speculative_tokens,
        SPEC_BLOCK_SIZE=spec_block_size,
    )


# ---------------------------------------------------------------------------
# Kernel 4: Mamba state selective copy (eliminates temporary allocations)
# ---------------------------------------------------------------------------
@triton.jit
def _mamba_state_selective_copy_kernel(
    # Source: intermediate states  [L, M, S+1, *state_shape]
    SRC_PTR,
    # Destination: current states  [L, M, *state_shape]
    DST_PTR,
    # Per-request index arrays
    PREFILL_STATUS_PTR,  # [N] 0=decode, 1=prefill
    STATE_IDX_PTR,  # [N] maps request → mamba state slot
    ACCEPTED_PTR,  # [N] accepted token index per request
    # Strides (in elements)
    src_stride_layer,
    src_stride_slot,
    src_stride_spec,
    dst_stride_layer,
    dst_stride_slot,
    # Data size
    STATE_SIZE,
    # Compile-time
    BLOCK_SIZE: tl.constexpr,
):
    """Copy intermediate Mamba state to current state for decode requests.

    Grid: (N, L, num_chunks)
      - dim 0: active request index
      - dim 1: mamba layer index
      - dim 2: chunk of the flattened state vector

    No-op for prefill requests.
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    # Skip prefill requests immediately.
    prefill = tl.load(PREFILL_STATUS_PTR + pid_req)
    if prefill == 1:
        return

    state_idx = tl.load(STATE_IDX_PTR + pid_req).to(tl.int64)
    accepted = tl.load(ACCEPTED_PTR + pid_req).to(tl.int64)

    chunk_start = pid_chunk * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    elem_offsets = chunk_start + offsets
    mask = elem_offsets < STATE_SIZE

    src_base = (
        pid_layer.to(tl.int64) * src_stride_layer
        + state_idx * src_stride_slot
        + accepted * src_stride_spec
    )
    dst_base = pid_layer.to(tl.int64) * dst_stride_layer + state_idx * dst_stride_slot

    data = tl.load(SRC_PTR + src_base + elem_offsets, mask=mask)
    tl.store(DST_PTR + dst_base + elem_offsets, data, mask=mask)


def mamba_state_selective_copy(
    intermediate_states, current_states, prefill_status, state_idx, accepted_counts, num_layers
):
    """Copy accepted intermediate Mamba states to current states in-place.

    For each decode request, copies
    `intermediate[layer, slot, accepted_count, ...]` →
    `current[layer, slot, ...]` for every Mamba layer.

    Args:
        intermediate_states: `(L, M, S+1, *state_shape)` — intermediate buffer.
        current_states: `(L, M, *state_shape)` — current state buffer (updated in-place).
        prefill_status: `(N,)` int tensor — 0 for decode, 1 for prefill.
        state_idx: `(N,)` int tensor — mamba state slot index per request.
        accepted_counts: `(N,)` int tensor — accepted token index per request.
        num_layers: number of Mamba layers (first dim of the state tensors).
    """
    N = prefill_status.shape[0]
    if N == 0:
        return

    # The state vector to copy per (layer, request) is the product of all
    # trailing dimensions after the speculative-token axis.
    # intermediate shape: (L, M, S+1, *state_shape) → state_size = prod(state_shape)
    state_size = math.prod(intermediate_states.shape[3:])

    BLOCK_SIZE = 1024
    num_chunks = triton.cdiv(state_size, BLOCK_SIZE)
    grid = (N, num_layers, num_chunks)

    _mamba_state_selective_copy_kernel[grid](
        intermediate_states,
        current_states,
        prefill_status,
        state_idx,
        accepted_counts,
        src_stride_layer=intermediate_states.stride(0),
        src_stride_slot=intermediate_states.stride(1),
        src_stride_spec=intermediate_states.stride(2),
        dst_stride_layer=current_states.stride(0),
        dst_stride_slot=current_states.stride(1),
        STATE_SIZE=state_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# ---------------------------------------------------------------------------
# Kernel 5: Per-depth attention metadata for the MTP draft loop
# ---------------------------------------------------------------------------
@triton.jit
def _mtp_draft_step_metadata_kernel(
    # Draft-loop state (read-only)
    OFFSETS_PTR,
    SRC_BLOCK_TABLE_PTR,
    # Per-token KV write maps
    TOKEN_TO_BLOCK_IDX_PTR,
    TOKEN_TO_LOCAL_POS_PTR,
    TOKEN_TO_REQUEST_IDX_PTR,
    TOKEN_TO_POS_IN_REQUEST_PTR,
    TOKEN_TO_POS_IDS_PTR,
    # Per-request length staging
    STAGED_QUERY_LENGTHS_PTR,
    STAGED_KV_LENGTHS_PTR,
    # Per-request MHA read metadata
    MHA_QUERY_LENGTHS_PTR,
    MHA_CU_QUERY_PTR,
    MHA_KV_LENGTHS_PTR,
    MHA_CU_KV_PTR,
    MHA_BLOCK_TABLE_PTR,
    # Strides / limits
    src_block_table_stride,
    mha_block_table_stride,
    num_block_columns,
    active_request_count,
    padded_count,
    dummy_block_idx,
    # Compile-time constants
    BLOCK_SIZE_TOKENS: tl.constexpr,
    BLOCK_R: tl.constexpr,  # next_power_of_2(padded_count)
    BLOCK_C: tl.constexpr,  # block-table columns copied per iteration
):
    """Write one draft depth's KV write maps and MHA read metadata.

    Grid: (1,). One program covers every request slot, since the cumulative lengths span
    requests. Each of the first `active_request_count` slots writes one token at its MTP
    position P (from OFFSETS) and attends over P + 1 keys. Slots in
    [active_request_count, padded_count) are padding: their tokens go to the dummy block and
    their requests get zero lengths and a -1 block table.
    """
    rows = tl.arange(0, BLOCK_R)
    real = rows < active_request_count
    padded = rows < padded_count

    # --- Per-token KV write maps (one token per request) ---
    positions = tl.load(OFFSETS_PTR + rows, mask=real, other=0)
    block_col = positions // BLOCK_SIZE_TOKENS
    destinations = tl.load(
        SRC_BLOCK_TABLE_PTR + rows.to(tl.int64) * src_block_table_stride + block_col,
        mask=real,
        other=0,
    )
    block_idx = tl.where(real, destinations.to(tl.int64), dummy_block_idx)
    tl.store(TOKEN_TO_BLOCK_IDX_PTR + rows, block_idx, mask=padded)
    local_pos = tl.where(real, positions % BLOCK_SIZE_TOKENS, 0)
    tl.store(TOKEN_TO_LOCAL_POS_PTR + rows, local_pos, mask=padded)
    tl.store(TOKEN_TO_REQUEST_IDX_PTR + rows, rows, mask=real)
    tl.store(TOKEN_TO_POS_IN_REQUEST_PTR + rows, positions, mask=real)
    tl.store(TOKEN_TO_POS_IDS_PTR + rows, positions.to(tl.int64), mask=real)

    # --- Per-request lengths: query_length = 1, kv_length = P + 1 (write-then-attend) ---
    query_lengths = tl.where(real, 1, 0)
    kv_lengths = tl.where(real, positions + 1, 0)
    tl.store(STAGED_QUERY_LENGTHS_PTR + rows, query_lengths, mask=real)
    tl.store(STAGED_KV_LENGTHS_PTR + rows, kv_lengths, mask=real)
    tl.store(MHA_QUERY_LENGTHS_PTR + rows, query_lengths, mask=padded)
    tl.store(MHA_KV_LENGTHS_PTR + rows, kv_lengths, mask=padded)

    # Cumulative lengths: entry 0 is 0, entry r + 1 sums slots 0..r. Padding slots add zero,
    # so they repeat the last real total.
    tl.store(MHA_CU_QUERY_PTR + rows, tl.zeros_like(rows), mask=rows == 0)
    tl.store(MHA_CU_KV_PTR + rows, tl.zeros_like(rows), mask=rows == 0)
    tl.store(MHA_CU_QUERY_PTR + 1 + rows, tl.cumsum(query_lengths, axis=0), mask=padded)
    tl.store(MHA_CU_KV_PTR + 1 + rows, tl.cumsum(kv_lengths, axis=0), mask=padded)

    # --- Block table: real rows copy the draft loop's table, padding rows get -1 ---
    real_2d = real[:, None]
    padded_2d = padded[:, None]
    src_rows = rows.to(tl.int64)[:, None] * src_block_table_stride
    dst_rows = rows.to(tl.int64)[:, None] * mha_block_table_stride
    for col_start in range(0, num_block_columns, BLOCK_C):
        cols = (col_start + tl.arange(0, BLOCK_C))[None, :]
        col_mask = cols < num_block_columns
        values = tl.load(SRC_BLOCK_TABLE_PTR + src_rows + cols, mask=real_2d & col_mask, other=0)
        values = tl.where(real_2d, values, -1)
        tl.store(
            MHA_BLOCK_TABLE_PTR + dst_rows + cols,
            values.to(MHA_BLOCK_TABLE_PTR.dtype.element_ty),
            mask=padded_2d & col_mask,
        )


def mtp_draft_step_metadata(
    gpu_view,
    offsets,
    block_table,
    staged_query_lengths,
    staged_kv_lengths,
    active_request_count,
    padded_count,
    block_size_tokens,
    dummy_block_idx,
):
    """Launch the per-depth MTP draft metadata kernel.

    Writes exactly what `MTPMetadata.stage_decode_lengths`, `write_token_maps` and
    `write_mha_metadata` write for one draft depth, in a single launch and without host
    syncs.

    Args:
        gpu_view: `ContextGPUView` whose token maps and `mha_*` buffers are written.
        offsets: `(>=active_request_count,)` int32 MTP write position per request. Must be
            non-negative.
        block_table: `(>=active_request_count, num_block_columns)` draft-loop block table.
        staged_query_lengths, staged_kv_lengths: `(>=active_request_count,)` int32 staging
            buffers, written with the per-request query and KV lengths.
        active_request_count: Requests that draft this depth.
        padded_count: Request (and token) slots the forward launches over, including padding.
        block_size_tokens: KV block size, in tokens.
        dummy_block_idx: Scratch block that padding tokens write to.
    """
    if padded_count == 0:
        return
    assert active_request_count <= padded_count
    num_block_columns = gpu_view.mha_block_table.shape[1]
    assert block_table.shape[1] == num_block_columns

    _mtp_draft_step_metadata_kernel[(1,)](
        offsets,
        block_table,
        gpu_view.token_to_block_idx,
        gpu_view.token_to_local_position_within_kv_block,
        gpu_view.token_to_request_idx,
        gpu_view.token_to_position_in_request,
        gpu_view.token_to_pos_ids,
        staged_query_lengths,
        staged_kv_lengths,
        gpu_view.mha_query_lengths,
        gpu_view.mha_cu_query_seq_lengths,
        gpu_view.mha_kv_seq_lengths,
        gpu_view.mha_cu_kv_seq_lengths,
        gpu_view.mha_block_table,
        src_block_table_stride=block_table.stride(0),
        mha_block_table_stride=gpu_view.mha_block_table.stride(0),
        num_block_columns=num_block_columns,
        active_request_count=active_request_count,
        padded_count=padded_count,
        dummy_block_idx=dummy_block_idx,
        BLOCK_SIZE_TOKENS=block_size_tokens,
        BLOCK_R=triton.next_power_of_2(padded_count),
        BLOCK_C=min(triton.next_power_of_2(num_block_columns), 64),
    )


# ---------------------------------------------------------------------------
# Kernel 6: Attention metadata for the CUDA-graphed MTP commit pass
# ---------------------------------------------------------------------------
@triton.jit
def _mtp_commit_step_metadata_kernel(
    # Per-request input (read-only)
    ACCEPTED_COUNTS_PTR,
    BASE_POSITIONS_PTR,
    SRC_BLOCK_TABLE_PTR,
    # Per-token KV write maps
    TOKEN_TO_BLOCK_IDX_PTR,
    TOKEN_TO_LOCAL_POS_PTR,
    TOKEN_TO_REQUEST_IDX_PTR,
    TOKEN_TO_POS_IN_REQUEST_PTR,
    TOKEN_TO_POS_IDS_PTR,
    # Per-request MHA read metadata
    MHA_QUERY_LENGTHS_PTR,
    MHA_CU_QUERY_PTR,
    MHA_KV_LENGTHS_PTR,
    MHA_CU_KV_PTR,
    MHA_BLOCK_TABLE_PTR,
    # Strides / limits
    src_block_table_stride,
    mha_block_table_stride,
    num_block_columns,
    active_request_count,
    padded_count,
    dummy_block_idx,
    # Compile-time constants
    SLOTS: tl.constexpr,  # num_speculative_tokens: commit rows per request
    BLOCK_SIZE_TOKENS: tl.constexpr,
    BLOCK_R: tl.constexpr,  # next_power_of_2(padded_count)
    BLOCK_S: tl.constexpr,  # next_power_of_2(SLOTS)
    BLOCK_C: tl.constexpr,  # block-table columns copied per iteration
):
    """Write the fixed-shape commit-pass metadata for a decode-only step.

    Grid: (1,). Request r owns token rows r*SLOTS .. r*SLOTS+SLOTS-1. Request r accepted a_r
    drafts, so its refreshed run starts at MTP position `start_r = base_r - 1 - a_r`; slot j
    writes position start_r + j when j < a_r and the dummy block otherwise. Every real request
    attends with SLOTS query rows over `start_r + SLOTS` keys, so causal (bottom-right aligned)
    masking lets slot j see keys up to start_r + j. Requests at or past active_request_count are
    padding: dummy-block writes, zero lengths, and a -1 block table.
    """
    rows = tl.arange(0, BLOCK_R)
    real = rows < active_request_count
    padded = rows < padded_count

    accepted = tl.load(ACCEPTED_COUNTS_PTR + rows, mask=real, other=0).to(tl.int32)
    base = tl.load(BASE_POSITIONS_PTR + rows, mask=real, other=1).to(tl.int32)
    starts = base - 1 - accepted

    # --- Per-token KV write maps ---
    slots = tl.arange(0, BLOCK_S)
    slot_in_range = (slots < SLOTS)[None, :]
    tokens = rows[:, None] * SLOTS + slots[None, :]
    token_mask = padded[:, None] & slot_in_range
    positions = starts[:, None] + slots[None, :]
    writes = real[:, None] & (slots[None, :] < accepted[:, None]) & slot_in_range
    destinations = tl.load(
        SRC_BLOCK_TABLE_PTR
        + rows.to(tl.int64)[:, None] * src_block_table_stride
        + positions // BLOCK_SIZE_TOKENS,
        mask=writes,
        other=0,
    )
    block_idx = tl.where(writes, destinations.to(tl.int64), dummy_block_idx)
    local_pos = tl.where(writes, positions % BLOCK_SIZE_TOKENS, 0)
    request_positions = tl.where(real[:, None], positions, 0)
    tl.store(TOKEN_TO_BLOCK_IDX_PTR + tokens, block_idx, mask=token_mask)
    tl.store(TOKEN_TO_LOCAL_POS_PTR + tokens, local_pos, mask=token_mask)
    tl.store(
        TOKEN_TO_REQUEST_IDX_PTR + tokens, rows[:, None] + tl.zeros_like(tokens), mask=token_mask
    )
    tl.store(TOKEN_TO_POS_IN_REQUEST_PTR + tokens, request_positions, mask=token_mask)
    tl.store(TOKEN_TO_POS_IDS_PTR + tokens, request_positions.to(tl.int64), mask=token_mask)

    # --- Per-request lengths and their cumulative sums ---
    query_lengths = tl.where(real, SLOTS, 0)
    kv_lengths = tl.where(real, starts + SLOTS, 0)
    tl.store(MHA_QUERY_LENGTHS_PTR + rows, query_lengths, mask=padded)
    tl.store(MHA_KV_LENGTHS_PTR + rows, kv_lengths, mask=padded)
    tl.store(MHA_CU_QUERY_PTR + rows, tl.zeros_like(rows), mask=rows == 0)
    tl.store(MHA_CU_KV_PTR + rows, tl.zeros_like(rows), mask=rows == 0)
    tl.store(MHA_CU_QUERY_PTR + 1 + rows, tl.cumsum(query_lengths, axis=0), mask=padded)
    tl.store(MHA_CU_KV_PTR + 1 + rows, tl.cumsum(kv_lengths, axis=0), mask=padded)

    # --- Block table: real rows copy the source table, padding rows get -1 ---
    real_2d = real[:, None]
    padded_2d = padded[:, None]
    src_rows = rows.to(tl.int64)[:, None] * src_block_table_stride
    dst_rows = rows.to(tl.int64)[:, None] * mha_block_table_stride
    for col_start in range(0, num_block_columns, BLOCK_C):
        cols = (col_start + tl.arange(0, BLOCK_C))[None, :]
        col_mask = cols < num_block_columns
        values = tl.load(SRC_BLOCK_TABLE_PTR + src_rows + cols, mask=real_2d & col_mask, other=0)
        values = tl.where(real_2d, values, -1)
        tl.store(
            MHA_BLOCK_TABLE_PTR + dst_rows + cols,
            values.to(MHA_BLOCK_TABLE_PTR.dtype.element_ty),
            mask=padded_2d & col_mask,
        )


def mtp_commit_step_metadata(
    gpu_view,
    accepted_counts,
    base_positions,
    block_table,
    active_request_count,
    padded_count,
    num_slots,
    block_size_tokens,
    dummy_block_idx,
):
    """Launch the graphed commit-pass metadata kernel.

    Lays the commit pass out as `num_slots` token rows per request (so its shape depends only on
    `padded_count`) and writes the token maps and MHA metadata for it, without host syncs.

    Args:
        gpu_view: `ContextGPUView` whose token maps and `mha_*` buffers are written.
        accepted_counts: `(>=active_request_count,)` drafts each request accepted this step.
        base_positions: `(>=active_request_count,)` position of each request's next draft
            (one past its last committed token).
        block_table: `(>=active_request_count, num_block_columns)` block ids per request.
        active_request_count: Real (decode) requests.
        padded_count: Request slots the forward launches over, including padding.
        num_slots: Commit rows per request (`num_speculative_tokens`).
        block_size_tokens: KV block size, in tokens.
        dummy_block_idx: Scratch block that unused and padding rows write to.
    """
    if padded_count == 0:
        return
    assert active_request_count <= padded_count
    num_block_columns = gpu_view.mha_block_table.shape[1]
    assert block_table.shape[1] == num_block_columns

    _mtp_commit_step_metadata_kernel[(1,)](
        accepted_counts,
        base_positions,
        block_table,
        gpu_view.token_to_block_idx,
        gpu_view.token_to_local_position_within_kv_block,
        gpu_view.token_to_request_idx,
        gpu_view.token_to_position_in_request,
        gpu_view.token_to_pos_ids,
        gpu_view.mha_query_lengths,
        gpu_view.mha_cu_query_seq_lengths,
        gpu_view.mha_kv_seq_lengths,
        gpu_view.mha_cu_kv_seq_lengths,
        gpu_view.mha_block_table,
        src_block_table_stride=block_table.stride(0),
        mha_block_table_stride=gpu_view.mha_block_table.stride(0),
        num_block_columns=num_block_columns,
        active_request_count=active_request_count,
        padded_count=padded_count,
        dummy_block_idx=dummy_block_idx,
        SLOTS=num_slots,
        BLOCK_SIZE_TOKENS=block_size_tokens,
        BLOCK_R=triton.next_power_of_2(padded_count),
        BLOCK_S=triton.next_power_of_2(num_slots),
        BLOCK_C=min(triton.next_power_of_2(num_block_columns), 64),
    )
