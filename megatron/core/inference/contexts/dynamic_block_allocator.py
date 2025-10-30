# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch
from torch import Tensor


class BlockAllocator:
    """Allocator that manages blocks of memory for the KV cache.

    This allocator is responsible for:
    - Initializing a pool of block IDs
    - Allocating blocks from the pool
    - Releasing blocks back to the pool
    - Managing the guaranteed block count for active requests

    Args:
        block_count_total (int): Total number of blocks available in the buffer.
        gtd_block_count (int): Number of blocks reserved for guaranteed requests.
    """

    def __init__(self, block_count_total: int, gtd_block_count: int):
        self.block_count_total = block_count_total
        self.gtd_block_count = gtd_block_count

        # Reserve last block ID as dummy block for decode-only inference steps
        self.block_count_avail = self.block_count_total - 1
        self.dummy_block_idx = self.block_count_total - 1

        # Initialize block pool as a "stack" data structure
        self.block_bag = torch.arange(
            self.block_count_total, dtype=torch.int32, device=torch.cuda.current_device()
        )

    def is_memory_available(self, num_blocks: int, safe: bool = False) -> bool:
        """Check if memory blocks are available.

        Use 'safe' to avoid all requests being deadlocked. A fraction of the KV cache
        memory buffer is reserved to guarantee that a minimum number of active
        requests can run on any given step.

        Args:
            num_blocks (int): Number of blocks to check.
            safe (bool): Include extra space for guaranteeing ability to run
                requests to completion.

        Return:
            (bool) Is memory available?
        """
        if safe:
            return self.block_count_avail >= num_blocks + self.gtd_block_count
        else:
            return self.block_count_avail >= num_blocks

    def allocate_memory_blocks(self, num_blocks: int = 1, safe: bool = False) -> Optional[Tensor]:
        """Allocate memory blocks if available, else return None.

        Args:
            num_blocks (int): Number of blocks to allocate.
            safe (bool): Include extra space for guaranteeing ability to run
                requests to completion.

        Return:
            (Optional[Tensor]) Allocated block IDs.
        """
        if self.is_memory_available(num_blocks, safe):
            self.block_count_avail -= num_blocks
            return self.block_bag[self.block_count_avail : (self.block_count_avail + num_blocks)]
        else:
            return None

    def release_memory_blocks(self, blocks: Tensor) -> None:
        """Release memory blocks.

        Args:
            blocks (Tensor): Block IDs to release.

        Return:
            None
        """
        num_blocks = blocks.size(dim=0)
        self.block_bag[self.block_count_avail : (self.block_count_avail + num_blocks)] = blocks
        self.block_count_avail += num_blocks

    def reset(self) -> None:
        """Reset the allocator to initial state.

        This resets the available block count to the entire memory pool
        (except for the dummy block).
        """
        self.block_count_avail = self.block_count_total - 1
