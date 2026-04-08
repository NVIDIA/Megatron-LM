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
) -> tuple:
    """Compute active_count, finished_count, needs_new_block_count in one kernel.

    Returns:
        (active_count, finished_count, needs_new_block_count) as Python ints.
    """
    device = active_requests_mask.device
    mask_len = active_requests_mask.shape[0]
    threshold = block_size_tokens - 1 - num_speculative_tokens

    active_buf = torch.zeros(1, dtype=torch.int32, device=device)
    finished_buf = torch.zeros(1, dtype=torch.int32, device=device)
    needs_block_buf = torch.zeros(1, dtype=torch.int32, device=device)

    if mask_len > 0:
        _fused_counts_kernel[(1,)](
            active_requests_mask,
            last_kv_block_offset,
            active_buf,
            finished_buf,
            needs_block_buf,
            mask_len=mask_len,
            paused_request_count=paused_request_count,
            threshold=threshold,
        )

    # Single batch sync: read all 3 values
    return active_buf.item(), finished_buf.item(), needs_block_buf.item()
