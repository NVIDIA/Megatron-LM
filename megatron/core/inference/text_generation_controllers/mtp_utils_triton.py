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
    IMMEDIATE_STATE_UPDATE: tl.constexpr,
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

    if IMMEDIATE_STATE_UPDATE:
        # The free token (speculative position 0) was already written to the live
        # state during the forward pass; intermediate index j holds the state
        # after drafted token j + 1. Roll the live state to
        # intermediate[accepted - 1], or skip entirely when no drafted token was
        # accepted (the live state already holds the correct free-token state).
        if accepted == 0:
            return
        spec_index = accepted - 1
    else:
        spec_index = accepted

    chunk_start = pid_chunk * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    elem_offsets = chunk_start + offsets
    mask = elem_offsets < STATE_SIZE

    src_base = (
        pid_layer.to(tl.int64) * src_stride_layer
        + state_idx * src_stride_slot
        + spec_index * src_stride_spec
    )
    dst_base = pid_layer.to(tl.int64) * dst_stride_layer + state_idx * dst_stride_slot

    data = tl.load(SRC_PTR + src_base + elem_offsets, mask=mask)
    tl.store(DST_PTR + dst_base + elem_offsets, data, mask=mask)


def mamba_state_selective_copy(
    intermediate_states,
    current_states,
    prefill_status,
    state_idx,
    accepted_counts,
    num_layers,
    immediate_state_update=False,
):
    """Copy accepted intermediate Mamba states to current states in-place.

    For each decode request, copies
    `intermediate[layer, slot, accepted_count, ...]` →
    `current[layer, slot, ...]` for every Mamba layer.

    Args:
        intermediate_states: `(L, M, S+1, *state_shape)` — intermediate buffer
            (or `(L, M, S, *state_shape)` when ``immediate_state_update`` is True).
        current_states: `(L, M, *state_shape)` — current state buffer (updated in-place).
        prefill_status: `(N,)` int tensor — 0 for decode, 1 for prefill.
        state_idx: `(N,)` int tensor — mamba state slot index per request.
        accepted_counts: `(N,)` int tensor — accepted token index per request.
        num_layers: number of Mamba layers (first dim of the state tensors).
        immediate_state_update: If True, the free token (speculative position 0)
            was already persisted to ``current_states`` during the forward pass and
            ``intermediate_states`` only holds the drafted tokens. The copy then
            targets ``intermediate[accepted_count - 1]`` and is skipped entirely
            when ``accepted_count == 0``.
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
        IMMEDIATE_STATE_UPDATE=immediate_state_update,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# ---------------------------------------------------------------------------
# Kernel 5: Mamba factorized state rollback (rank-1 reconstruction)
# ---------------------------------------------------------------------------
def _factorized_rollback_configs():
    """Autotune configs (GB200): vary warps/stages; block sizes are fixed by dims."""
    if not HAVE_TRITON:
        return []
    return [
        triton.Config({}, num_warps=w, num_stages=s) for w in (1, 2, 4, 8) for s in (1, 2, 3)
    ]


@triton.autotune(
    configs=_factorized_rollback_configs(),
    key=["HEADDIM", "DSTATE", "WINDOW"],
    # STATE_PTR is updated in place (S_new = ... * S_prev + ...). The autotuner runs
    # the kernel many times to benchmark configs, so without restoring S_prev between
    # trials the in-place update would compound and corrupt the state (and the first
    # real call, during which tuning happens, would be wrong). restore_value snapshots
    # and restores S_prev so every trial — and the final launch — starts from S_prev.
    restore_value=["STATE_PTR"],
)
@triton.jit
def _mamba_state_factorized_rollback_kernel(
    # Stored rank-1 factors
    DX_PTR,  # (L, M, W, nheads, headdim)
    B_PTR,  # (L, M, W, ngroups, dstate)
    ALPHA_PTR,  # (L, M, W, nheads)
    # Live SSM state (read S_prev, write S_new in place): (L, M, nheads, headdim, dstate)
    STATE_PTR,
    # Per-request index arrays
    PREFILL_STATUS_PTR,  # [N] 0=decode, 1=prefill
    STATE_IDX_PTR,  # [N] request -> mamba state slot
    ACCEPTED_PTR,  # [N] index of last accepted window position
    # Strides
    dx_stride_layer,
    dx_stride_slot,
    dx_stride_win,
    dx_stride_head,
    dx_stride_p,
    b_stride_layer,
    b_stride_slot,
    b_stride_win,
    b_stride_group,
    b_stride_n,
    a_stride_layer,
    a_stride_slot,
    a_stride_win,
    a_stride_head,
    st_stride_layer,
    st_stride_slot,
    st_stride_head,
    st_stride_p,
    st_stride_n,
    nheads_per_group,
    # Compile-time
    HEADDIM: tl.constexpr,
    DSTATE: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reconstruct the accepted SSM state from stored rank-1 factors.

    Grid: (N_requests, num_layers, nheads). Each program reconstructs the
    ``(headdim x dstate)`` state tile for one ``(request, layer, head)``:

        S_new = (prod_{i=0}^{m} alpha_i) * S_prev
                + sum_{t=0}^{m} (prod_{i=t+1}^{m} alpha_i) * (dx_t (outer) B_t)

    where ``m = accepted_counts[req]`` is the index of the last accepted window
    position (the free token at index 0 is always accepted). No-op for prefill.
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1)
    pid_head = tl.program_id(2)

    prefill = tl.load(PREFILL_STATUS_PTR + pid_req)
    if prefill == 1:
        return

    slot = tl.load(STATE_IDX_PTR + pid_req).to(tl.int64)
    if slot < 0:
        return
    m = tl.load(ACCEPTED_PTR + pid_req).to(tl.int32)
    group = (pid_head // nheads_per_group).to(tl.int64)

    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    mask_p = offs_p < HEADDIM
    mask_n = offs_n < DSTATE
    tile_mask = mask_p[:, None] & mask_n[None, :]

    dx_base = (
        pid_layer.to(tl.int64) * dx_stride_layer
        + slot * dx_stride_slot
        + pid_head.to(tl.int64) * dx_stride_head
    )
    b_base = pid_layer.to(tl.int64) * b_stride_layer + slot * b_stride_slot + group * b_stride_group
    a_base = (
        pid_layer.to(tl.int64) * a_stride_layer
        + slot * a_stride_slot
        + pid_head.to(tl.int64) * a_stride_head
    )

    acc = tl.zeros([BLOCK_P, BLOCK_N], dtype=tl.float32)
    c = 1.0
    # Backward over the window so the running product `c` accumulates the suffix
    # product of alpha; window positions beyond the accepted index `m` are masked
    # out. WINDOW is constexpr, so this loop is fully unrolled (CUDA-graph / autotune
    # friendly, no data-dependent loop bound).
    for t in range(WINDOW - 1, -1, -1):
        include = t <= m
        a_t = tl.load(ALPHA_PTR + a_base + t * a_stride_win).to(tl.float32)
        dx_t = tl.load(
            DX_PTR + dx_base + t * dx_stride_win + offs_p * dx_stride_p, mask=mask_p, other=0.0
        ).to(tl.float32)
        b_t = tl.load(
            B_PTR + b_base + t * b_stride_win + offs_n * b_stride_n, mask=mask_n, other=0.0
        ).to(tl.float32)
        contrib = dx_t[:, None] * b_t[None, :]
        acc += tl.where(include, c, 0.0) * contrib
        c = tl.where(include, c * a_t, c)
    # After the loop, c == prod_{i=0}^{m} alpha_i (the coefficient on S_prev).

    st_ptrs = (
        STATE_PTR
        + pid_layer.to(tl.int64) * st_stride_layer
        + slot * st_stride_slot
        + pid_head.to(tl.int64) * st_stride_head
        + offs_p[:, None] * st_stride_p
        + offs_n[None, :] * st_stride_n
    )
    h = tl.load(st_ptrs, mask=tile_mask, other=0.0).to(tl.float32)
    h_new = c * h + acc
    tl.store(st_ptrs, h_new.to(STATE_PTR.dtype.element_ty), mask=tile_mask)


def mamba_state_factorized_rollback(
    factor_dx,
    factor_B,
    factor_alpha,
    current_states,
    prefill_status,
    state_idx,
    accepted_counts,
    num_layers,
    nheads_per_group,
):
    """Fused rank-1 reconstruction of the accepted SSM state (Triton).

    For each decode request, reconstructs the live SSM state after the last
    accepted speculative token directly from the stored rank-1 update factors,
    in place, avoiding any full intermediate-state checkpoint:

        S_new = (prod_{i=0}^{m} alpha_i) * S_prev
                + sum_{t=0}^{m} (prod_{i=t+1}^{m} alpha_i) * (dx_t (outer) B_t)

    Args:
        factor_dx: ``(L, M, W, nheads, headdim)`` fp32 — ``dx_t = delta_t * x_t``.
        factor_B: ``(L, M, W, ngroups, dstate)`` fp32 — ``B_t`` (per group).
        factor_alpha: ``(L, M, W, nheads)`` fp32 — scalar per-head decay ``alpha_t``.
        current_states: ``(L, M, nheads, headdim, dstate)`` — live SSM state (in place).
        prefill_status: ``(N,)`` int — 0 for decode, 1 for prefill (skipped).
        state_idx: ``(N,)`` int — request -> Mamba state slot.
        accepted_counts: ``(N,)`` int — index of the last accepted window position.
        num_layers: number of Mamba layers.
        nheads_per_group: ``nheads // ngroups``.
    """
    N = prefill_status.shape[0]
    if N == 0:
        return
    nheads = current_states.shape[2]
    headdim = current_states.shape[3]
    dstate = current_states.shape[4]
    window = factor_dx.shape[2]

    block_p = triton.next_power_of_2(headdim)
    block_n = triton.next_power_of_2(dstate)
    grid = (N, num_layers, nheads)

    _mamba_state_factorized_rollback_kernel[grid](
        factor_dx,
        factor_B,
        factor_alpha,
        current_states,
        prefill_status,
        state_idx,
        accepted_counts,
        factor_dx.stride(0),
        factor_dx.stride(1),
        factor_dx.stride(2),
        factor_dx.stride(3),
        factor_dx.stride(4),
        factor_B.stride(0),
        factor_B.stride(1),
        factor_B.stride(2),
        factor_B.stride(3),
        factor_B.stride(4),
        factor_alpha.stride(0),
        factor_alpha.stride(1),
        factor_alpha.stride(2),
        factor_alpha.stride(3),
        current_states.stride(0),
        current_states.stride(1),
        current_states.stride(2),
        current_states.stride(3),
        current_states.stride(4),
        nheads_per_group,
        HEADDIM=headdim,
        DSTATE=dstate,
        WINDOW=window,
        BLOCK_P=block_p,
        BLOCK_N=block_n,
    )
