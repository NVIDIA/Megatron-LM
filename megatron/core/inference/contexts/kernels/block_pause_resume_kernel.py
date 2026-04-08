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
    request_ids_ptr,
    # Output
    needs_new_block_ptr,
    num_needing_ptr,
    newly_paused_ids_ptr,
    # Compaction indices output
    src_idxs_ptr,
    dst_idxs_ptr,
    num_compact_ptr,
    # Scalars
    active_request_count: tl.int32,
    paused_request_count: tl.int32,
    threshold: tl.int32,
    chunked_prefill_request_id: tl.int32,
    max_allowed_active: tl.int32,
):
    """Detect which active requests need a new block and compute compaction indices.

    Grid: (1,)

    Scans active requests, compares last_kv_block_offset against threshold.
    Handles chunked prefill exclusion and force-pause of excess requests.
    Produces: needs_new_block mask, num_needing count, newly_paused_request_ids,
    and compaction src/dst indices.
    """
    # Pass 1: Compute needs_new_block with chunked-prefill exclusion and force-pause
    total_needing = 0
    chunked_idx = -1

    # Find chunked prefill request index (if any)
    if chunked_prefill_request_id != -1:
        ci = paused_request_count
        while ci < paused_request_count + active_request_count:
            if tl.load(request_ids_ptr + ci) == chunked_prefill_request_id:
                chunked_idx = ci - paused_request_count
                ci = paused_request_count + active_request_count  # break
            else:
                ci += 1

    i = 0
    while i < active_request_count:
        abs_idx = i + paused_request_count
        offset = tl.load(last_kv_block_offset_ptr + abs_idx)
        needs = 1 if offset >= threshold else 0
        # Chunked prefill exclusion
        if i == chunked_idx:
            needs = 0
        # Force-pause excess when no chunked prefill
        if chunked_idx == -1 and i >= max_allowed_active:
            needs = 1
        tl.store(needs_new_block_ptr + i, needs)
        total_needing += needs
        i += 1

    tl.store(num_needing_ptr, total_needing)

    # Extract newly paused request IDs
    if total_needing > 0:
        pi = 0
        i = 0
        while i < active_request_count:
            if tl.load(needs_new_block_ptr + i) == 1:
                abs_idx = i + paused_request_count
                rid = tl.load(request_ids_ptr + abs_idx)
                tl.store(newly_paused_ids_ptr + pi, rid)
                pi += 1
            i += 1

    # Pass 2: Compute compaction indices (only if some but not all need new block)
    if total_needing > 0 and total_needing < active_request_count:
        al_count = 0
        pr_count = 0

        i = 0
        while i < active_request_count:
            abs_idx = i + paused_request_count
            needs = tl.load(needs_new_block_ptr + i)
            if i < total_needing and needs == 0:
                tl.store(dst_idxs_ptr + al_count, abs_idx)
                al_count += 1
            if i >= total_needing and needs == 1:
                tl.store(src_idxs_ptr + pr_count, abs_idx)
                pr_count += 1
            i += 1

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
    # Scratchpad
    scratch_ptr,
    # Output
    resume_count_ptr,
    # Scalars
    paused_request_count: tl.int32,
    active_block_avail: tl.int32,
    max_allowed_resume: tl.int32,
    threshold: tl.int32,
    max_allowed_active: tl.int32,
    # Compile-time
    USE_SCRATCH: tl.constexpr,
):
    """Determine resume count and allocate blocks for resumed requests.

    Grid: (1,)

    Scans paused requests in LIFO order (right to left), computes cumulative
    block requirement, determines how many can resume, then allocates blocks
    for those that need them.
    """
    if USE_SCRATCH:
        paused_request_count = tl.load(scratch_ptr + 2)  # SLOT_PAUSED_COUNT
        active_block_avail = tl.load(scratch_ptr + 15)  # spare slot: active_avail_buf
        active_count = tl.load(scratch_ptr + 0)  # SLOT_ACTIVE_COUNT
        max_allowed_resume = max_allowed_active - active_count
        if max_allowed_resume < 0:
            max_allowed_resume = 0

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
            i = -1  # break
        else:
            candidate = resume_count + 1
            if candidate > avail:
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
    request_ids: Tensor,
    active_request_count: int,
    paused_request_count: int,
    block_size_tokens: int,
    num_speculative_tokens: int,
    max_requests: int,
    max_tokens: int,
    chunked_prefill_request_id: int = -1,
    out_num_needing: Optional[Tensor] = None,
    out_num_compact: Optional[Tensor] = None,
    out_needs_new_block: Optional[Tensor] = None,
    out_newly_paused_ids: Optional[Tensor] = None,
    out_src_idxs: Optional[Tensor] = None,
    out_dst_idxs: Optional[Tensor] = None,
) -> Tuple:
    """Detect which active requests need new blocks and compute compaction indices.

    If out_* tensors are provided, writes to them (no sync).
    Otherwise allocates temporary buffers and returns with .item() (legacy path).
    """
    device = last_kv_block_offset.device
    threshold = block_size_tokens - 1 - num_speculative_tokens
    max_allowed_active = min(max_requests, max_tokens // (num_speculative_tokens + 1))

    sync = out_num_needing is None
    mr = max(active_request_count, 1)
    needs_new_block = out_needs_new_block if out_needs_new_block is not None else torch.zeros(mr, dtype=torch.int32, device=device)
    num_needing_buf = out_num_needing if out_num_needing is not None else torch.zeros(1, dtype=torch.int32, device=device)
    newly_paused_ids = out_newly_paused_ids if out_newly_paused_ids is not None else torch.empty(mr, dtype=torch.int32, device=device)
    src_idxs = out_src_idxs if out_src_idxs is not None else torch.empty(mr, dtype=torch.int32, device=device)
    dst_idxs = out_dst_idxs if out_dst_idxs is not None else torch.empty(mr, dtype=torch.int32, device=device)
    num_compact_buf = out_num_compact if out_num_compact is not None else torch.zeros(1, dtype=torch.int32, device=device)
    if out_num_needing is not None:
        out_num_needing.zero_()
    if out_num_compact is not None:
        out_num_compact.zero_()

    if active_request_count > 0:
        _detect_pause_kernel[(1,)](
            last_kv_block_offset,
            request_ids,
            needs_new_block,
            num_needing_buf,
            newly_paused_ids,
            src_idxs,
            dst_idxs,
            num_compact_buf,
            active_request_count=active_request_count,
            paused_request_count=paused_request_count,
            threshold=threshold,
            chunked_prefill_request_id=chunked_prefill_request_id,
            max_allowed_active=max_allowed_active,
        )

    if sync:
        num_needing = num_needing_buf.item()
        num_compact = num_compact_buf.item()
        return needs_new_block, num_needing, newly_paused_ids, src_idxs, dst_idxs, num_compact
    return needs_new_block, newly_paused_ids, src_idxs, dst_idxs


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
    out_resume_count: Optional[Tensor] = None,
    scratch: Optional[Tensor] = None,
) -> int:
    """Determine resume count and allocate blocks for resumed requests.

    If scratch is provided, reads paused_request_count from scratchpad (no sync).
    Returns resume_request_count (Python int), or -1 if scratch mode (no sync).
    """
    use_scratch = scratch is not None
    if not use_scratch and paused_request_count == 0:
        return 0

    threshold = block_size_tokens - 1 - num_speculative_tokens
    max_allowed_active = min(max_requests, max_tokens // (num_speculative_tokens + 1))
    max_allowed_resume = max(0, max_allowed_active - active_request_count)

    resume_count_buf = out_resume_count if out_resume_count is not None else torch.zeros(1, dtype=torch.int32, device=last_kv_block_offset.device)
    if out_resume_count is not None:
        out_resume_count.zero_()

    scratch_ptr = scratch if use_scratch else block_bag  # dummy

    _resume_and_allocate_kernel[(1,)](
        last_kv_block_offset,
        request_to_kv_block_ids,
        request_to_kv_block_ids.stride(0),
        request_kv_block_counts,
        request_last_kv_block_id,
        block_bag,
        total_avail_tensor,
        scratch_ptr,
        resume_count_buf,
        paused_request_count=paused_request_count,
        active_block_avail=active_block_avail,
        max_allowed_resume=max_allowed_resume,
        threshold=threshold,
        max_allowed_active=max_allowed_active,
        USE_SCRATCH=use_scratch,
    )

    if use_scratch:
        return -1  # caller reads from out_resume_count later
    return resume_count_buf.item()
