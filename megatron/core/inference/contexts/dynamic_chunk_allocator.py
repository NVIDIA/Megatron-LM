# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch
from torch import Tensor


class ChunkAllocator:
    """Allocator that manages chunks of memory for the KV cache.

    This allocator is responsible for:
    - Initializing a pool of chunk IDs
    - Allocating chunks from the pool
    - Releasing chunks back to the pool

    Args:
        active_count (int): Total number of active chunks available in the buffer.
            The full buffer size is 2*active_count, to accommodate an equal-size
            space for paused requests that live on the CPU.
    """

    # def __init__(self, active_count: int):

    #     active_count -= 1 # subtract 1 for dummy_chunk_idx (see below)
    #     self.total_count = 2 * active_count
    #     self.active_count = active_count
    #     self.paused_count = active_count
    #     # >>>
    #     # self.active_avail = active_count - 1 # reserve last chunk for decode-only steps
    #     self.active_avail = active_count
    #     # <<<
    #     self.paused_avail = active_count
    #     self.dummy_chunk_idx = self.total_count - 1

    #     # Initialize chunk pool as a "stack" data structure
    #     self.chunk_bag = torch.arange(
    #         self.total_count,
    #         dtype=torch.int32,
    #         device=torch.cuda.current_device(),
    #     )
    def __init__(self, active_count: int):

        active_count -= 1 # -1 for dummy_chunk_idx (see below)
        self.total_count = 2 * active_count + 1
        self.total_avail = self.total_count - 1 # -1 for dummy_chunk_idx
        self.active_count = active_count
        self.dummy_chunk_idx = self.total_count - 1

        # Initialize chunk pool as a "stack" data structure
        self.chunk_bag = torch.arange(
            self.total_count,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )

    def is_memory_available(self, num_chunks: int) -> bool:
        """Check if memory chunks are available.

        Args:
            num_chunks (int): Number of chunks to check.

        Return:
            (bool) Is memory available?
        """
        return self.total_avail >= num_chunks

    def allocate_memory_chunks(self, num_chunks: int = 1) -> Optional[Tensor]:
        """Allocate memory chunks if available, else return None.

        Args:
            num_chunks (int): Number of chunks to allocate.

        Return:
            (Optional[Tensor]) Allocated chunk IDs.
        """
        if self.is_memory_available(num_chunks):
            self.total_avail -= num_chunks
            return self.chunk_bag[self.total_avail : (self.total_avail + num_chunks)]
        else:
            return None

    def release_memory_chunks(self, chunks: Tensor) -> None:
        """Release memory chunks.

        Args:
            chunks (Tensor): Chunk IDs to release.

        Return:
            None
        """
        num_chunks = chunks.size(dim=0)
        self.chunk_bag[self.total_avail : (self.total_avail + num_chunks)] = chunks
        self.total_avail += num_chunks

    def reset(self) -> None:
        """Reset the allocator to initial state.

        This resets the available chunk count to the entire memory pool
        (except for the dummy chunk).
        """
        self.total_avail = self.total_count - 1
