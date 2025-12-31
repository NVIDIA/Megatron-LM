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

    Args:
        context (DynamicInferenceContext): Dynamic inference context.
        active_count (int): Total number of active blocks available in the buffer.
            The full buffer size is 2*active_count, to accommodate an equal-size
            space for paused requests that live on the CPU.
    """

    def __init__(self, context: "DynamicInferenceContext", total_count: int):

        self.context = context

        active_count = (total_count - 1) // 2  # -1 for dummy_block_idx (see below)
        active_count = max(1, active_count)  # need at least one block
        self.total_count = 2 * active_count + 1  # +1 for dummy_block_idx
        self.total_avail = self.total_count - 1  # -1 for dummy_block_idx
        self.active_count = active_count
        self.paused_count = self.total_count - self.active_count - 1  # -1 for dummy_block_idx
        self.dummy_block_idx = self.total_count - 1

        # Initialize block pool as a "stack" data structure
        self.block_bag = torch.arange(
            self.total_count, dtype=torch.int32, device=torch.cuda.current_device()
        )

    def __str__(self):
        return (
            f"total avail {self.total_avail} / {self.total_count - 1}"
            f"; active {self.active_count}"
        )

    def get_active_used(self):
        """Compute number of active blocks used."""
        return (
            self.context.request_kv_block_counts[
                self.context.paused_request_count : self.context.total_request_count
            ]
            .sum()
            .item()
        )

    def get_paused_used(self):
        """Compute number of paused blocks used."""
        return (
            self.context.request_kv_block_counts[: self.context.paused_request_count].sum().item()
        )

    def get_active_avail(self):
        """Compute number of active blocks available."""
        return self.active_count - self.get_active_used()

    def get_paused_avail(self):
        """Compute number of paused blocks available."""
        return self.paused_count - self.get_paused_used()

    def is_memory_available(self, num_blocks: int) -> bool:
        """Check if memory blocks are available.

        Args:
            num_blocks (int): Number of blocks to check.

        Return:
            (bool) Is memory available?
        """
        return self.get_active_avail() >= num_blocks

    def allocate_memory_blocks(self, num_blocks: int) -> Optional[Tensor]:
        """Allocate memory blocks if available, else return None.

        Args:
            num_blocks (int): Number of blocks to allocate.

        Return:
            (Optional[Tensor]) Allocated block IDs.
        """
        if self.is_memory_available(num_blocks):
            self.total_avail -= num_blocks
            block_ids = self.block_bag[self.total_avail : (self.total_avail + num_blocks)]
            assert num_blocks == block_ids.numel()
            return block_ids
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
        self.block_bag[self.total_avail : (self.total_avail + num_blocks)] = blocks
        self.total_avail += num_blocks

    def reset(self) -> None:
        """Reset the allocator to initial state.

        This resets the available block count to the entire memory pool
        (except for the dummy block).
        """

        # Reset block bag to so we start consuming from the beginning of the pool
        # for UVM performance.
        # *Note*: Resetting the block bag is essential because if engine has been
        # suspended, then the block bag contains non-unique IDs since the
        # right-most IDs have been 'popped' off and are owned by the context.
        # Without resetting the block bag, context request memory will clash and
        # requests will point to each other's memory blocks, resulting in faulty
        # generations.
        self.block_bag = torch.arange(
            self.total_count, dtype=torch.int32, device=torch.cuda.current_device()
        )

        self.total_avail = self.total_count - 1
