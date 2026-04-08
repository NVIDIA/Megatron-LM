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
def _classify_and_release_kernel(
    # Input
    active_requests_mask_ptr,
    # KV block table
    kv_block_ids_ptr,
    kv_block_ids_stride: tl.int32,
    # Allocator state
    block_bag_ptr,
    total_avail_ptr,
    # Ref counts for prefix caching
    block_ref_counts_ptr,
    block_hashes_ptr,
    # Output: compaction indices
    finished_left_ptr,
    active_right_ptr,
    num_compact_ptr,
    finished_idxs_ptr,
    num_finished_ptr,
    # Scratchpad for sync-free mode (or dummy)
    scratch_ptr,
    # Scalars
    mask_len: tl.int32,
    active_request_count: tl.int32,
    paused_request_count: tl.int32,
    max_kv_blocks: tl.int32,
    # Compile-time: read counts from scratchpad instead of params
    USE_SCRATCH: tl.constexpr,
    # Compile-time
    HAS_PREFIX_CACHE: tl.constexpr,
):
    """Classify finished requests, release their blocks, compute compaction indices.

    Grid: (1,) — serial scan.

    The mask has length mask_len = active_request_count + finished_request_count.
    The compaction partition boundary is at position active_request_count:
    - "Left partition"  = mask positions [0, active_request_count)
    - "Right partition" = mask positions [active_request_count, mask_len)
    Finished requests in the left partition pair with active requests in the right
    partition for the move-compaction step.
    """
    if USE_SCRATCH:
        active_request_count = tl.load(scratch_ptr + 0)  # SLOT_ACTIVE_COUNT
        paused_request_count = tl.load(scratch_ptr + 2)  # SLOT_PAUSED_COUNT

    avail = tl.load(total_avail_ptr)
    fl_count = 0
    ar_count = 0
    finished_count = 0

    i = 0
    while i < mask_len:
        mask_val = tl.load(active_requests_mask_ptr + i)
        abs_idx = i + paused_request_count

        if mask_val == 0:
            # Finished request — record index and release blocks
            tl.store(finished_idxs_ptr + finished_count, abs_idx)
            finished_count += 1

            # Release KV blocks for this request
            row_base = abs_idx * kv_block_ids_stride
            j = 0
            while j < max_kv_blocks:
                block_id = tl.load(kv_block_ids_ptr + row_base + j)
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
                    tl.store(kv_block_ids_ptr + row_base + j, tl.cast(-1, tl.int32))
                j += 1

            # Compaction: finished in left partition
            if i < active_request_count:
                tl.store(finished_left_ptr + fl_count, abs_idx)
                fl_count += 1
        else:
            # Active request — compaction: active in right partition
            if i >= active_request_count:
                tl.store(active_right_ptr + ar_count, abs_idx)
                ar_count += 1
        i += 1

    tl.store(total_avail_ptr, avail)
    tl.store(num_compact_ptr, fl_count)
    tl.store(num_finished_ptr, finished_count)


def triton_classify_and_release(
    active_requests_mask: Tensor,
    request_to_kv_block_ids: Tensor,
    block_bag: Tensor,
    total_avail_tensor: Tensor,
    block_ref_counts: Optional[Tensor],
    block_hashes: Optional[Tensor],
    active_request_count: int,
    paused_request_count: int,
    max_kv_block_count: int,
    has_prefix_cache: bool,
    out_finished_left: Optional[Tensor] = None,
    out_active_right: Optional[Tensor] = None,
    out_num_compact: Optional[Tensor] = None,
    out_finished_idxs: Optional[Tensor] = None,
    out_num_finished: Optional[Tensor] = None,
    scratch: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Classify finished requests, release blocks, compute compaction indices.

    Returns:
        (finished_left, active_right, num_compact_buf, finished_idxs, num_finished_buf)
        All GPU tensors. Caller reads .item() when needed.
    """
    device = active_requests_mask.device
    mask_len = active_requests_mask.shape[0]

    finished_left = out_finished_left if out_finished_left is not None else torch.empty(mask_len, dtype=torch.int32, device=device)
    active_right = out_active_right if out_active_right is not None else torch.empty(mask_len, dtype=torch.int32, device=device)
    num_compact_buf = out_num_compact if out_num_compact is not None else torch.zeros(1, dtype=torch.int32, device=device)
    finished_idxs = out_finished_idxs if out_finished_idxs is not None else torch.empty(mask_len, dtype=torch.int32, device=device)
    num_finished_buf = out_num_finished if out_num_finished is not None else torch.zeros(1, dtype=torch.int32, device=device)
    # Reset scalar outputs (may be reused from previous step)
    if out_num_compact is not None:
        out_num_compact.zero_()
    if out_num_finished is not None:
        out_num_finished.zero_()

    dummy = block_bag
    ref_ptr = block_ref_counts if has_prefix_cache else dummy
    hash_ptr = block_hashes if has_prefix_cache else dummy

    use_scratch = scratch is not None
    scratch_ptr = scratch if use_scratch else block_bag  # dummy when not used

    _classify_and_release_kernel[(1,)](
        active_requests_mask,
        request_to_kv_block_ids,
        request_to_kv_block_ids.stride(0),
        block_bag,
        total_avail_tensor,
        ref_ptr,
        hash_ptr,
        finished_left,
        active_right,
        num_compact_buf,
        finished_idxs,
        num_finished_buf,
        scratch_ptr,
        mask_len=mask_len,
        active_request_count=active_request_count,
        paused_request_count=paused_request_count,
        max_kv_blocks=max_kv_block_count,
        HAS_PREFIX_CACHE=has_prefix_cache,
        USE_SCRATCH=use_scratch,
    )

    return finished_left, active_right, num_compact_buf, finished_idxs, num_finished_buf
