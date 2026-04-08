# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional, Tuple

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
def _detect_pause_kernel(
    # Input
    last_kv_block_offset_ptr,
    # Output
    needs_new_block_ptr,
    num_needing_ptr,
    # Compaction indices output
    src_idxs_ptr,
    dst_idxs_ptr,
    num_compact_ptr,
    # Scalars
    active_request_count: tl.int32,
    paused_request_count: tl.int32,
    threshold: tl.int32,
):
    """Detect which active requests need a new block and compute compaction indices.

    Grid: (1,)

    Scans active requests, compares last_kv_block_offset against threshold.
    Produces a needs_new_block mask and compaction src/dst indices for the
    subsequent _move_book_keeping_tensors call.
    """
    # Pass 1: Compute needs_new_block and count
    total_needing = 0
    i = 0
    while i < active_request_count:
        abs_idx = i + paused_request_count
        offset = tl.load(last_kv_block_offset_ptr + abs_idx)
        needs = 1 if offset >= threshold else 0
        tl.store(needs_new_block_ptr + i, needs)
        total_needing += needs
        i += 1

    tl.store(num_needing_ptr, total_needing)

    # Pass 2: Compute compaction indices (only if some but not all need new block)
    # The compaction partitions at position total_needing:
    # - Active (non-pausing) requests in left partition (positions 0..total_needing-1)
    #   that are NOT needing new block → these are "active_on_left"
    # - Pausing requests in right partition (positions total_needing..end)
    #   that ARE needing new block → these are "paused_on_right"
    # We swap them: src = [paused_on_right, active_on_left], dst = [active_on_left, paused_on_right]
    if total_needing > 0 and total_needing < active_request_count:
        al_count = 0  # active (non-pausing) in left partition
        pr_count = 0  # pausing in right partition

        i = 0
        while i < active_request_count:
            abs_idx = i + paused_request_count
            needs = tl.load(needs_new_block_ptr + i)
            if i < total_needing and needs == 0:
                # Active (non-pausing) in left partition → needs to move right
                tl.store(dst_idxs_ptr + al_count, abs_idx)
                al_count += 1
            if i >= total_needing and needs == 1:
                # Pausing in right partition → needs to move left
                tl.store(src_idxs_ptr + pr_count, abs_idx)
                pr_count += 1
            i += 1

        # Now build full src/dst: src = [paused_on_right, active_on_left]
        #                          dst = [active_on_left, paused_on_right]
        # The src_idxs_ptr already has paused_on_right at [0..pr_count)
        # The dst_idxs_ptr already has active_on_left at [0..al_count)
        # We need to append: src += active_on_left, dst += paused_on_right
        # But we stored them in separate arrays. Copy cross:
        j = 0
        while j < al_count:
            val = tl.load(dst_idxs_ptr + j)
            tl.store(src_idxs_ptr + pr_count + j, val)
            j += 1
        j = 0
        while j < pr_count:
            val = tl.load(src_idxs_ptr + j)
            tl.store(dst_idxs_ptr + al_count + j, val)
            j += 1

        tl.store(num_compact_ptr, pr_count + al_count)
    else:
        tl.store(num_compact_ptr, 0)


@triton.jit
def _resume_and_allocate_kernel(
    # Request state
    last_kv_block_offset_ptr,
    kv_block_ids_ptr,
    kv_block_ids_stride: tl.int32,
    kv_block_counts_ptr,
    last_kv_block_id_ptr,
    # Allocator state
    block_bag_ptr,
    total_avail_ptr,
    # Output
    resume_count_ptr,
    # Scalars
    paused_request_count: tl.int32,
    active_block_avail: tl.int32,
    total_avail_limit: tl.int32,
    max_allowed_resume: tl.int32,
    threshold: tl.int32,
):
    """Determine resume count and allocate blocks for resumed requests.

    Grid: (1,)

    Scans paused requests in LIFO order (right to left), computes cumulative
    block requirement, determines how many can resume, then allocates blocks
    for those that need them.
    """
    if paused_request_count == 0:
        tl.store(resume_count_ptr, 0)
        return

    # LIFO scan: walk from rightmost paused request (index paused_count-1) backward
    cumsum = 0
    resume_count = 0
    avail = tl.load(total_avail_ptr)

    i = paused_request_count - 1
    while i >= 0:
        offset = tl.load(last_kv_block_offset_ptr + i)
        needs = 1 if offset >= threshold else 0
        block_count = tl.load(kv_block_counts_ptr + i)
        cumsum += block_count + needs

        if cumsum > active_block_avail:
            # Can't fit this request
            i = -1  # break
        else:
            candidate = resume_count + 1
            if candidate > total_avail_limit:
                i = -1  # break
            else:
                if candidate > max_allowed_resume:
                    i = -1  # break
                else:
                    resume_count = candidate
                    i -= 1

    tl.store(resume_count_ptr, resume_count)

    # Allocate blocks for resumed requests that need them
    if resume_count > 0:
        new_paused_count = paused_request_count - resume_count
        j = new_paused_count
        while j < paused_request_count:
            offset = tl.load(last_kv_block_offset_ptr + j)
            if offset >= threshold:
                # Allocate one block
                new_avail = avail - 1
                block_id = tl.load(block_bag_ptr + new_avail)
                avail = new_avail

                # Assign to request
                col = tl.load(kv_block_counts_ptr + j)
                tl.store(kv_block_ids_ptr + j * kv_block_ids_stride + col, block_id)
                tl.store(kv_block_counts_ptr + j, col + 1)
                tl.store(last_kv_block_id_ptr + j, block_id)
            j += 1

        tl.store(total_avail_ptr, avail)


def triton_detect_pause(
    last_kv_block_offset: Tensor,
    active_request_count: int,
    paused_request_count: int,
    block_size_tokens: int,
    num_speculative_tokens: int,
) -> Tuple[Tensor, int, Tensor, Tensor, int]:
    """Detect which active requests need new blocks and compute compaction indices.

    Returns:
        (needs_new_block, num_needing, src_idxs, dst_idxs, num_compact)
    """
    device = last_kv_block_offset.device
    threshold = block_size_tokens - 1 - num_speculative_tokens

    needs_new_block = torch.zeros(active_request_count, dtype=torch.int32, device=device)
    num_needing_buf = torch.zeros(1, dtype=torch.int32, device=device)
    src_idxs = torch.empty(active_request_count, dtype=torch.int32, device=device)
    dst_idxs = torch.empty(active_request_count, dtype=torch.int32, device=device)
    num_compact_buf = torch.zeros(1, dtype=torch.int32, device=device)

    if active_request_count > 0:
        _detect_pause_kernel[(1,)](
            last_kv_block_offset,
            needs_new_block,
            num_needing_buf,
            src_idxs,
            dst_idxs,
            num_compact_buf,
            active_request_count=active_request_count,
            paused_request_count=paused_request_count,
            threshold=threshold,
        )

    num_needing = num_needing_buf.item()
    num_compact = num_compact_buf.item()
    return needs_new_block, num_needing, src_idxs, dst_idxs, num_compact


def triton_resume_and_allocate(
    last_kv_block_offset: Tensor,
    request_to_kv_block_ids: Tensor,
    request_kv_block_counts: Tensor,
    request_last_kv_block_id: Tensor,
    block_bag: Tensor,
    total_avail_tensor: Tensor,
    paused_request_count: int,
    active_request_count: int,
    active_block_avail: int,
    block_size_tokens: int,
    num_speculative_tokens: int,
    max_requests: int,
    max_tokens: int,
) -> int:
    """Determine resume count and allocate blocks for resumed requests.

    Returns:
        resume_request_count
    """
    if paused_request_count == 0:
        return 0

    threshold = block_size_tokens - 1 - num_speculative_tokens
    total_avail_val = total_avail_tensor.item()
    max_allowed_active = min(max_requests, max_tokens // (num_speculative_tokens + 1))
    max_allowed_resume = max(0, max_allowed_active - active_request_count)

    resume_count_buf = torch.zeros(1, dtype=torch.int32, device=last_kv_block_offset.device)

    _resume_and_allocate_kernel[(1,)](
        last_kv_block_offset,
        request_to_kv_block_ids,
        request_to_kv_block_ids.stride(0),
        request_kv_block_counts,
        request_last_kv_block_id,
        block_bag,
        total_avail_tensor,
        resume_count_buf,
        paused_request_count=paused_request_count,
        active_block_avail=active_block_avail,
        total_avail_limit=total_avail_val,
        max_allowed_resume=max_allowed_resume,
        threshold=threshold,
    )

    return resume_count_buf.item()
