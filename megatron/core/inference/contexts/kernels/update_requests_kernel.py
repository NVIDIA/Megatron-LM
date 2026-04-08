# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

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
def _fused_counts_kernel(
    active_requests_mask_ptr,
    last_kv_block_offset_ptr,
    # Output GPU scalars
    active_count_ptr,
    finished_count_ptr,
    needs_new_block_count_ptr,
    # Scalars
    mask_len: tl.int32,
    paused_request_count: tl.int32,
    threshold: tl.int32,
):
    """Compute active_count, finished_count, and needs_new_block_count in one pass.

    Grid: (1,)
    Scans active_requests_mask (length mask_len) and last_kv_block_offset for
    active requests to produce all three counts in a single kernel launch.
    """
    active = 0
    finished = 0
    needs_block = 0

    i = 0
    while i < mask_len:
        mask_val = tl.load(active_requests_mask_ptr + i)
        if mask_val == 1:
            # Check if this active request needs a new block
            abs_idx = paused_request_count + i
            offset = tl.load(last_kv_block_offset_ptr + abs_idx)
            if offset >= threshold:
                needs_block += 1
            active += 1
        else:
            finished += 1
        i += 1

    tl.store(active_count_ptr, active)
    tl.store(finished_count_ptr, finished)
    tl.store(needs_new_block_count_ptr, needs_block)


def triton_fused_counts(
    active_requests_mask: Tensor,
    last_kv_block_offset: Tensor,
    paused_request_count: int,
    block_size_tokens: int,
    num_speculative_tokens: int,
    out_active: Tensor = None,
    out_finished: Tensor = None,
    out_needs_block: Tensor = None,
) -> tuple:
    """Compute active_count, finished_count, needs_new_block_count in one kernel.

    If out_* tensors are provided, writes to them (no sync).
    Otherwise allocates temporary buffers and returns Python ints (legacy path).
    """
    device = active_requests_mask.device
    mask_len = active_requests_mask.shape[0]
    threshold = block_size_tokens - 1 - num_speculative_tokens

    sync = out_active is None
    if sync:
        out_active = torch.zeros(1, dtype=torch.int32, device=device)
        out_finished = torch.zeros(1, dtype=torch.int32, device=device)
        out_needs_block = torch.zeros(1, dtype=torch.int32, device=device)

    if mask_len > 0:
        _fused_counts_kernel[(1,)](
            active_requests_mask,
            last_kv_block_offset,
            out_active,
            out_finished,
            out_needs_block,
            mask_len=mask_len,
            paused_request_count=paused_request_count,
            threshold=threshold,
        )

    if sync:
        return out_active.item(), out_finished.item(), out_needs_block.item()
    return None


# =============================================================================
# Scratchpad slot constants for sync-free kernel chaining
# =============================================================================
SLOT_ACTIVE_COUNT = 0
SLOT_FINISHED_COUNT = 1
SLOT_PAUSED_COUNT = 2
SLOT_TOTAL_COUNT = 3
SLOT_NUM_COMPACT = 4
SLOT_NUM_FINISHED = 5
SLOT_PAUSE_NEEDING = 6
SLOT_RESUME_COUNT_1 = 7
SLOT_EVICT_COUNT = 8
SLOT_SWAP_COUNT = 9
SLOT_RESUME_COUNT_2 = 10
SLOT_CHUNKED_FOUND_IDX = 11
SLOT_CHUNKED_IN_BOUNDS = 12
SLOT_ACTIVE_TOKEN_COUNT = 13
SLOT_DID_RESET = 14


@triton.jit
def _init_counters_kernel(
    scratch_ptr,
    paused_count: tl.int32,
):
    """Initialize scratchpad counters after fused_counts writes slots 0-1.

    Grid: (1,)
    Sets slot[2] = paused_count, slot[3] = slot[0] + paused_count, slot[14] = 0.
    """
    active = tl.load(scratch_ptr + 0)  # SLOT_ACTIVE_COUNT
    tl.store(scratch_ptr + 2, paused_count)  # SLOT_PAUSED_COUNT
    tl.store(scratch_ptr + 3, active + paused_count)  # SLOT_TOTAL_COUNT
    tl.store(scratch_ptr + 14, 0)  # SLOT_DID_RESET


@triton.jit
def _update_counts_after_pause_kernel(scratch_ptr):
    """After pause detection: paused += pause_needing, active -= pause_needing. Grid: (1,)."""
    pause_needing = tl.load(scratch_ptr + 6)  # SLOT_PAUSE_NEEDING
    if pause_needing > 0:
        old_paused = tl.load(scratch_ptr + 2)
        old_active = tl.load(scratch_ptr + 0)
        tl.store(scratch_ptr + 2, old_paused + pause_needing)
        tl.store(scratch_ptr + 0, old_active - pause_needing)


@triton.jit
def _update_counts_after_resume_kernel(scratch_ptr, resume_slot: tl.int32):
    """After resume: paused -= resume_count, active += resume_count. Grid: (1,)."""
    resume_count = tl.load(scratch_ptr + resume_slot)
    if resume_count > 0:
        old_paused = tl.load(scratch_ptr + 2)
        old_active = tl.load(scratch_ptr + 0)
        tl.store(scratch_ptr + 2, old_paused - resume_count)
        tl.store(scratch_ptr + 0, old_active + resume_count)


@triton.jit
def _update_counts_after_evict_kernel(scratch_ptr):
    """After eviction: paused -= evict_count, total = paused + active. Grid: (1,)."""
    evict_count = tl.load(scratch_ptr + 8)  # SLOT_EVICT_COUNT
    if evict_count > 0:
        old_paused = tl.load(scratch_ptr + 2)
        active = tl.load(scratch_ptr + 0)
        new_paused = old_paused - evict_count
        tl.store(scratch_ptr + 2, new_paused)
        tl.store(scratch_ptr + 3, new_paused + active)  # SLOT_TOTAL_COUNT


@triton.jit
def _update_counts_after_chunked_prefill_kernel(scratch_ptr):
    """After chunked prefill swap: if in_bounds, active -= 1, total -= 1. Grid: (1,)."""
    found = tl.load(scratch_ptr + 11)  # SLOT_CHUNKED_FOUND_IDX
    in_bounds = tl.load(scratch_ptr + 12)  # SLOT_CHUNKED_IN_BOUNDS
    if found >= 0 and in_bounds == 1:
        old_active = tl.load(scratch_ptr + 0)
        old_total = tl.load(scratch_ptr + 3)
        tl.store(scratch_ptr + 0, old_active - 1)
        tl.store(scratch_ptr + 3, old_total - 1)


@triton.jit
def _compute_active_avail_kernel(
    kv_block_counts_ptr,
    scratch_ptr,
    active_avail_out_ptr,
    active_count_budget: tl.int32,
):
    """Compute active_block_avail = active_count_budget - sum(kv_block_counts[paused:total]). Grid: (1,)."""
    paused = tl.load(scratch_ptr + 2)  # SLOT_PAUSED_COUNT
    total = tl.load(scratch_ptr + 3)  # SLOT_TOTAL_COUNT
    used = 0
    i = paused
    while i < total:
        used += tl.load(kv_block_counts_ptr + i)
        i += 1
    tl.store(active_avail_out_ptr, active_count_budget - used)


@triton.jit
def _reset_vacated_rows_kernel(
    num_compact_ptr,
    src_idxs_ptr,
    kv_block_ids_ptr,
    kv_stride: tl.int32,
    mamba_state_idx_ptr,
    max_kv: tl.int32,
    IS_HYBRID: tl.constexpr,
):
    """Reset kv_block_ids and mamba_state_idx for compacted source positions. Grid: (1,)."""
    num_compact = tl.load(num_compact_ptr)
    i = 0
    while i < num_compact:
        src = tl.load(src_idxs_ptr + i)
        row = src * kv_stride
        j = 0
        while j < max_kv:
            tl.store(kv_block_ids_ptr + row + j, tl.cast(-1, tl.int32))
            j += 1
        if IS_HYBRID:
            tl.store(mamba_state_idx_ptr + src, tl.cast(-1, tl.int32))
        i += 1


@triton.jit
def _mamba_cleanup_kernel(
    # Count pointer (GPU scalar)
    num_idxs_ptr,
    # Index buffer
    idxs_ptr,
    # Mamba state
    mamba_state_idx_ptr,
    mamba_free_slots_ptr,
    mamba_free_count_ptr,
    # Slot allocator tensors
    sa_counts_ptr, sa_offsets_ptr, sa_block_ids_ptr, sa_eos_ptr,
    # Compile-time
    HAS_MAMBA: tl.constexpr,
    HAS_SLOT_ALLOC: tl.constexpr,
):
    """Mamba cleanup for finished/evicted requests. Grid: (1,).

    Reads count from GPU scalar, frees Mamba slots, resets slot allocator fields.
    No-ops if count == 0.
    """
    num_idxs = tl.load(num_idxs_ptr)
    if num_idxs == 0:
        return

    if HAS_MAMBA:
        free_count = tl.load(mamba_free_count_ptr)
        i = 0
        while i < num_idxs:
            req_idx = tl.load(idxs_ptr + i)
            mamba_idx = tl.load(mamba_state_idx_ptr + req_idx)
            if mamba_idx != -1:
                tl.store(mamba_free_slots_ptr + free_count, mamba_idx)
                free_count += 1
            tl.store(mamba_state_idx_ptr + req_idx, tl.cast(-1, tl.int32))
            i += 1
        tl.store(mamba_free_count_ptr, free_count)

    if HAS_SLOT_ALLOC:
        j = 0
        while j < num_idxs:
            req_idx = tl.load(idxs_ptr + j)
            tl.store(sa_counts_ptr + req_idx, 0)
            tl.store(sa_offsets_ptr + req_idx, 0)
            tl.store(sa_block_ids_ptr + req_idx, tl.cast(-1, tl.int32))
            tl.store(sa_eos_ptr + req_idx, tl.cast(-1, tl.int32))
            j += 1
