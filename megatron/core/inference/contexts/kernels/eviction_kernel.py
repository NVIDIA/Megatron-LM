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
def _evict_overflow_kernel(
    # Request state
    kv_block_counts_ptr,
    kv_block_ids_ptr,
    kv_block_ids_stride: tl.int32,
    request_ids_ptr,
    # Allocator state
    block_bag_ptr,
    total_avail_ptr,
    # Prefix cache
    block_ref_counts_ptr,
    block_hashes_ptr,
    # Outputs
    evict_count_ptr,
    evict_request_ids_ptr,
    evict_idxs_ptr,
    src_idxs_ptr,
    dst_idxs_ptr,
    swap_count_ptr,
    # Scalars
    paused_request_count: tl.int32,
    active_request_count: tl.int32,
    total_request_count: tl.int32,
    paused_block_limit: tl.int32,
    max_kv_blocks: tl.int32,
    # Compile-time
    HAS_PREFIX_CACHE: tl.constexpr,
):
    """Detect overflow, select eviction candidates, release blocks, compute swap indices.

    Grid: (1,)
    """
    # Step 1: Compute paused_used = sum of block counts for paused requests
    paused_used = 0
    i = 0
    while i < paused_request_count:
        paused_used += tl.load(kv_block_counts_ptr + i)
        i += 1

    if paused_used <= paused_block_limit:
        tl.store(evict_count_ptr, 0)
        tl.store(swap_count_ptr, 0)
        return

    # Step 2: Cumsum from left to find valid_paused_request_count
    cumsum = 0
    valid_count = 0
    i = 0
    while i < paused_request_count:
        cumsum += tl.load(kv_block_counts_ptr + i)
        if cumsum <= paused_block_limit:
            valid_count = i + 1
        i += 1

    overflow_count = paused_request_count - valid_count
    if overflow_count == 0:
        tl.store(evict_count_ptr, 0)
        tl.store(swap_count_ptr, 0)
        return

    # Step 3: Determine evict_request_count using rightmost overflow cumsum
    # Walk overflow requests from right (newest) to left, accumulating blocks.
    # For each candidate k (0-indexed from right), compute:
    #   cumsum_blocks[k] - (overflow_count - 1 - k) >= 0
    # The first k where this holds (plus 1) is the evict count.
    evict_count = overflow_count  # worst case: evict all overflow
    block_cumsum = 0
    k = 0
    while k < overflow_count:
        # The k-th request from the right is at index: paused_request_count - 1 - k
        req_idx = paused_request_count - 1 - k
        block_cumsum += tl.load(kv_block_counts_ptr + req_idx)
        remaining = overflow_count - 1 - k
        if block_cumsum - remaining >= 0:
            evict_count = k + 1
            k = overflow_count  # break
        else:
            k += 1

    tl.store(evict_count_ptr, evict_count)

    # Step 4: Evict requests at [paused_count - evict_count, paused_count)
    evict_start = paused_request_count - evict_count
    avail = tl.load(total_avail_ptr)

    j = 0
    while j < evict_count:
        req_idx = evict_start + j

        # Clone request ID
        rid = tl.load(request_ids_ptr + req_idx)
        tl.store(evict_request_ids_ptr + j, rid)
        tl.store(evict_idxs_ptr + j, req_idx)

        # Release KV blocks
        row_base = req_idx * kv_block_ids_stride
        b = 0
        while b < max_kv_blocks:
            block_id = tl.load(kv_block_ids_ptr + row_base + b)
            if block_id != -1:
                if HAS_PREFIX_CACHE:
                    rc = tl.load(block_ref_counts_ptr + block_id)
                    tl.store(block_ref_counts_ptr + block_id, rc - 1)
                    bh = tl.load(block_hashes_ptr + block_id)
                    if rc - 1 == 0 and bh == -1:
                        tl.store(block_bag_ptr + avail, block_id)
                        avail += 1
                else:
                    tl.store(block_bag_ptr + avail, block_id)
                    avail += 1
                tl.store(kv_block_ids_ptr + row_base + b, tl.cast(-1, tl.int32))
            b += 1
        j += 1

    tl.store(total_avail_ptr, avail)

    # Step 5: Compute swap indices
    if evict_count < active_request_count:
        # Pattern 1: swap evicted with rightmost active
        s = 0
        while s < evict_count:
            tl.store(src_idxs_ptr + s, paused_request_count - evict_count + s)
            tl.store(dst_idxs_ptr + s, total_request_count - evict_count + s)
            s += 1
        tl.store(swap_count_ptr, evict_count)
    else:
        # Pattern 2: swap active with leftmost evicted
        s = 0
        while s < active_request_count:
            tl.store(src_idxs_ptr + s, paused_request_count - evict_count + s)
            tl.store(dst_idxs_ptr + s, paused_request_count + s)
            s += 1
        tl.store(swap_count_ptr, active_request_count)


def triton_evict_overflow(
    kv_block_counts: Tensor,
    request_to_kv_block_ids: Tensor,
    request_ids: Tensor,
    block_bag: Tensor,
    total_avail_tensor: Tensor,
    block_ref_counts: Optional[Tensor],
    block_hashes: Optional[Tensor],
    paused_request_count: int,
    active_request_count: int,
    total_request_count: int,
    paused_block_limit: int,
    max_kv_block_count: int,
    has_prefix_cache: bool,
    out_evict_count: Optional[Tensor] = None,
    out_evict_request_ids: Optional[Tensor] = None,
    out_evict_idxs: Optional[Tensor] = None,
    out_src_idxs: Optional[Tensor] = None,
    out_dst_idxs: Optional[Tensor] = None,
    out_swap_count: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Detect overflow, evict, release blocks, compute swap indices.

    Returns:
        (evict_count_buf, evict_request_ids, evict_idxs, src_idxs, dst_idxs, swap_count_buf)
        All GPU tensors. Caller reads .item() when needed.
    """
    device = kv_block_counts.device
    max_out = max(paused_request_count, 1)

    evict_count_buf = out_evict_count if out_evict_count is not None else torch.zeros(1, dtype=torch.int32, device=device)
    evict_request_ids = out_evict_request_ids if out_evict_request_ids is not None else torch.empty(max_out, dtype=torch.int32, device=device)
    evict_idxs = out_evict_idxs if out_evict_idxs is not None else torch.empty(max_out, dtype=torch.int32, device=device)
    src_idxs = out_src_idxs if out_src_idxs is not None else torch.empty(max_out, dtype=torch.int32, device=device)
    dst_idxs = out_dst_idxs if out_dst_idxs is not None else torch.empty(max_out, dtype=torch.int32, device=device)
    swap_count_buf = out_swap_count if out_swap_count is not None else torch.zeros(1, dtype=torch.int32, device=device)
    if out_evict_count is not None:
        out_evict_count.zero_()
    if out_swap_count is not None:
        out_swap_count.zero_()

    dummy = block_bag
    ref_ptr = block_ref_counts if has_prefix_cache else dummy
    hash_ptr = block_hashes if has_prefix_cache else dummy

    _evict_overflow_kernel[(1,)](
        kv_block_counts,
        request_to_kv_block_ids,
        request_to_kv_block_ids.stride(0),
        request_ids,
        block_bag,
        total_avail_tensor,
        ref_ptr,
        hash_ptr,
        evict_count_buf,
        evict_request_ids,
        evict_idxs,
        src_idxs,
        dst_idxs,
        swap_count_buf,
        paused_request_count=paused_request_count,
        active_request_count=active_request_count,
        total_request_count=total_request_count,
        paused_block_limit=paused_block_limit,
        max_kv_blocks=max_kv_block_count,
        HAS_PREFIX_CACHE=has_prefix_cache,
    )

    return evict_count_buf, evict_request_ids, evict_idxs, src_idxs, dst_idxs, swap_count_buf
