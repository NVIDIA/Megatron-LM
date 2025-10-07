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
        context (DynamicInferenceContext): Dynamic inference context.
        active_count (int): Total number of active chunks available in the buffer.
            The full buffer size is 2*active_count, to accommodate an equal-size
            space for paused requests that live on the CPU.
    """

    def __init__(self, context: "DynamicInferenceContext", active_count: int):

        self.context = context

        active_count -= 1 # -1 for dummy_chunk_idx (see below)
        self.total_count = 2 * active_count + 1 # +1 for dummy_chunk_idx
        self.total_avail = self.total_count - 1 # -1 for dummy_chunk_idx
        self.active_count = active_count
        self.paused_count = self.total_count - self.active_count - 1 # -1 for dummy_chunk_idx
        self.dummy_chunk_idx = self.total_count - 1

        # Initialize chunk pool as a "stack" data structure
        self.chunk_bag = torch.arange(
            self.total_count,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )

    def __str__(self):
        return (
            f"total avail {self.total_avail} / {self.total_count - 1}"
            f"; active {self.active_count}"
        )

    def get_active_used(self):
        """Compute number of active chunks used."""
        return self.context.request_kv_chunk_counts[
            self.context.paused_request_count:self.context.total_request_count
        ].sum().item()

    def get_paused_used(self):
        """Compute number of paused chunks used."""
        return self.context.request_kv_chunk_counts[
            :self.context.paused_request_count
        ].sum().item()

    def get_active_avail(self):
        """Compute number of active chunks available."""
        return self.active_count - self.get_active_used()

    def get_paused_avail(self):
        """Compute number of paused chunks available."""
        return self.paused_count - self.get_paused_used()

    def is_memory_available(self, num_chunks: int) -> bool:
        """Check if memory chunks are available.

        Args:
            num_chunks (int): Number of chunks to check.

        Return:
            (bool) Is memory available?
        """
        return self.get_active_avail() >= num_chunks

    def allocate_memory_chunks(self, num_chunks: int) -> Optional[Tensor]:
        """Allocate memory chunks if available, else return None.

        Args:
            num_chunks (int): Number of chunks to allocate.

        Return:
            (Optional[Tensor]) Allocated chunk IDs.
        """
        if self.is_memory_available(num_chunks):
            self.total_avail -= num_chunks
            chunk_ids = self.chunk_bag[self.total_avail : (self.total_avail + num_chunks)]
            assert num_chunks == chunk_ids.numel()
            return chunk_ids
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
