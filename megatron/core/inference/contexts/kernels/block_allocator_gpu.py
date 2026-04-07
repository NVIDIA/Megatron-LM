# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

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
def _allocate_blocks_kernel(
    block_bag_ptr,
    total_avail_ptr,
    output_ptr,
    ref_counts_ptr,
    num_blocks: tl.int32,
    HAS_PREFIX_CACHE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Allocate num_blocks from the free-list stack on GPU.

    Grid: (1,) — single program, serial allocation.
    Reads block IDs from block_bag[new_avail:avail] and writes to output.
    """
    avail = tl.load(total_avail_ptr)
    new_avail = avail - num_blocks

    for offset in tl.static_range(0, 1024, BLOCK_SIZE):
        idx = tl.arange(0, BLOCK_SIZE)
        mask = (offset + idx) < num_blocks
        block_ids = tl.load(block_bag_ptr + new_avail + offset + idx, mask=mask)
        tl.store(output_ptr + offset + idx, block_ids, mask=mask)
        if HAS_PREFIX_CACHE:
            tl.store(ref_counts_ptr + block_ids, 1, mask=mask)

    tl.store(total_avail_ptr, new_avail)


@triton.jit
def _release_blocks_kernel(
    block_bag_ptr,
    total_avail_ptr,
    blocks_ptr,
    num_blocks: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """Release num_blocks back to the free-list stack on GPU.

    Grid: (1,) — single program, serial release.
    Writes block IDs from blocks into block_bag[avail:avail+num_blocks].
    """
    avail = tl.load(total_avail_ptr)

    for offset in tl.static_range(0, 1024, BLOCK_SIZE):
        idx = tl.arange(0, BLOCK_SIZE)
        mask = (offset + idx) < num_blocks
        block_ids = tl.load(blocks_ptr + offset + idx, mask=mask)
        tl.store(block_bag_ptr + avail + offset + idx, block_ids, mask=mask)

    tl.store(total_avail_ptr, avail + num_blocks)


def triton_allocate_blocks(
    block_bag: Tensor,
    total_avail_tensor: Tensor,
    output: Tensor,
    num_blocks: int,
    ref_counts: Tensor = None,
    has_prefix_cache: bool = False,
) -> None:
    """Allocate blocks from GPU free-list stack.

    Args:
        block_bag: Free-list stack tensor.
        total_avail_tensor: GPU scalar [1] tracking stack pointer.
        output: Pre-allocated output tensor to write block IDs into.
        num_blocks: Number of blocks to allocate.
        ref_counts: Prefix cache ref count tensor (or None).
        has_prefix_cache: Whether to set ref_counts=1 for allocated blocks.
    """
    if num_blocks == 0:
        return

    dummy = block_bag
    ref_ptr = ref_counts if has_prefix_cache else dummy

    block_size = min(triton.next_power_of_2(num_blocks), 1024)

    _allocate_blocks_kernel[(1,)](
        block_bag,
        total_avail_tensor,
        output,
        ref_ptr,
        num_blocks=num_blocks,
        HAS_PREFIX_CACHE=has_prefix_cache,
        BLOCK_SIZE=block_size,
    )


def triton_release_blocks(
    block_bag: Tensor,
    total_avail_tensor: Tensor,
    blocks: Tensor,
    num_blocks: int,
) -> None:
    """Release blocks back to GPU free-list stack.

    Args:
        block_bag: Free-list stack tensor.
        total_avail_tensor: GPU scalar [1] tracking stack pointer.
        blocks: Block IDs to release.
        num_blocks: Number of blocks to release.
    """
    if num_blocks == 0:
        return

    block_size = min(triton.next_power_of_2(num_blocks), 1024)

    _release_blocks_kernel[(1,)](
        block_bag,
        total_avail_tensor,
        blocks,
        num_blocks=num_blocks,
        BLOCK_SIZE=block_size,
    )
