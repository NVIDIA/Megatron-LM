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
    - Managing the guaranteed chunk count for active requests

    Args:
        chunk_count_total (int): Total number of chunks available in the buffer.
        gtd_chunk_count (int): Number of chunks reserved for guaranteed requests.
    """

    def __init__(self, chunk_count_total: int, gtd_chunk_count: int):
        self.chunk_count_total = chunk_count_total
        self.gtd_chunk_count = gtd_chunk_count

        # Reserve last chunk ID as dummy chunk for decode-only inference steps
        self.chunk_count_avail = self.chunk_count_total - 1
        self.dummy_chunk_idx = self.chunk_count_total - 1

        # Initialize chunk pool as a "stack" data structure
        self.chunk_bag = torch.arange(
            self.chunk_count_total, dtype=torch.int32, device=torch.cuda.current_device()
        )

    def is_memory_available(self, num_chunks: int, safe: bool = False) -> bool:
        """Check if memory chunks are available.

        Use 'safe' to avoid all requests being blocked. A fraction of the KV cache
        memory buffer is reserved to guarantee that a minimum number of active
        requests can run on any given step.

        Args:
            num_chunks (int): Number of chunks to check.
            safe (bool): Include extra space for guaranteeing ability to run
                requests to completion.

        Return:
            (bool) Is memory available?
        """
        if safe:
            return self.chunk_count_avail >= num_chunks + self.gtd_chunk_count
        else:
            return self.chunk_count_avail >= num_chunks

    def allocate_memory_chunks(self, num_chunks: int = 1, safe: bool = False) -> Optional[Tensor]:
        """Allocate memory chunks if available, else return None.

        Args:
            num_chunks (int): Number of chunks to allocate.
            safe (bool): Include extra space for guaranteeing ability to run
                requests to completion.

        Return:
            (Optional[Tensor]) Allocated chunk IDs.
        """
        if self.is_memory_available(num_chunks, safe):
            self.chunk_count_avail -= num_chunks
            return self.chunk_bag[self.chunk_count_avail : (self.chunk_count_avail + num_chunks)]
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
        self.chunk_bag[self.chunk_count_avail : (self.chunk_count_avail + num_chunks)] = chunks
        self.chunk_count_avail += num_chunks

    def reset(self) -> None:
        """Reset the allocator to initial state.

        This resets the available chunk count to the entire memory pool
        (except for the dummy chunk).
        """
        self.chunk_count_avail = self.chunk_count_total - 1
