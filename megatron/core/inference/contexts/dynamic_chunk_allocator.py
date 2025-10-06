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

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def get_active_used(self, context: "DynamicInferenceContext"):
        """Compute number of active chunks used.

        Args:
            context (DynamicInferenceContext): Dynamic inference context.
        """
        return context.request_kv_chunk_counts[
            context.paused_request_count:context.total_request_count
        ].sum().item()

    # def get_paused_used(self, context: "DynamicInferenceContext"):
    #     """Compute number of paused chunks used.

    #     Args:
    #         context (DynamicInferenceContext): Dynamic inference context.
    #     """
    #     return context.request_kv_chunk_counts[
    #         :context.paused_request_count
    #     ].sum().item()

    def get_active_avail(self, context: "DynamicInferenceContext"):
        """Compute number of active chunks available.

        Args:
            context (DynamicInferenceContext): Dynamic inference context.
        """
        return self.active_count - self.get_active_used(context)

    # def get_paused_avail(self, context: "DynamicInferenceContext"):
    #     """Compute number of paused chunks available.

    #     Args:
    #         context (DynamicInferenceContext): Dynamic inference context.
    #     """
    #     return self.paused_count - self.get_paused_used(context)

    # def get_active_needed_addl(self, context: "DynamicInferenceContext"):
    #     """Compute max number of additional chunks needed to complete currently
    #     active requests.

    #     Args:
    #         context (DynamicInferenceContext): Dynamic inference context.
    #     """

    #     active_request_count = context.total_request_count - context.paused_request_count
    #     active_chunks_needed_total = active_request_count * context.max_kv_chunk_count
    #     active_chunks_used = self.get_active_used(context)
    #     active_chunks_needed_addl = active_chunks_needed_total - active_chunks_used
    #     # pax("active_request_count, active_chunks_used",
    #     #     "active_chunks_needed_total, active_chunks_needed_addl")
    #     return active_chunks_needed_addl
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # def is_memory_available(self, num_chunks: int) -> bool:
    #     """Check if memory chunks are available.

    #     Args:
    #         num_chunks (int): Number of chunks to check.

    #     Return:
    #         (bool) Is memory available?
    #     """
    #     return self.total_avail >= num_chunks
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # def is_memory_available(
    #     self,
    #     context: "DynamicInferenceContext",
    #     num_chunks: int,
    # ) -> bool:
    #     """Check if memory chunks are available.

    #     Args:
    #         context (DynamicInferenceContext): Dynamic inference context.
    #         num_chunks (int): Number of chunks to check.

    #     Return:
    #         (bool) Is memory available?
    #     """
    #     # >>>
    #     # paused_avail = self.get_paused_avail(context)
    #     active_avail = self.get_active_avail(context)
    #     active_needed_addl = self.get_active_needed_addl(context)
    #     # if active_avail >= num_chunks and self.total_avail == 0:
    #     #     pax("self, num_chunks", {
    #     #         "paused_used" : "%d / %d" % (self.get_paused_used(context), self.paused_count),
    #     #         "active_used" : "%d / %d" % (self.get_active_used(context), self.active_count),
    #     #         "total_request_count" : context.total_request_count,
    #     #         "paused_request_count" : context.paused_request_count,
    #     #     }, "active_addl")
    #     # <<<
    #     # return self.get_active_avail(context) >= num_chunks
    #     return active_avail >= num_chunks and active_needed_addl <= self.total_avail
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_memory_available(
        self,
        context: "DynamicInferenceContext",
        num_chunks: int,
    ) -> bool:
        """Check if memory chunks are available.

        Args:
            context (DynamicInferenceContext): Dynamic inference context.
            num_chunks (int): Number of chunks to check.

        Return:
            (bool) Is memory available?
        """
        # >>>
        # if context.paused_request_count != 0:
        #     return False
        # <<<
        return self.get_active_avail(context) >= num_chunks
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # def allocate_memory_chunks(self, num_chunks: int = 1) -> Optional[Tensor]:
    #     """Allocate memory chunks if available, else return None.

    #     Args:
    #         num_chunks (int): Number of chunks to allocate.

    #     Return:
    #         (Optional[Tensor]) Allocated chunk IDs.
    #     """
    #     if self.is_memory_available(num_chunks):
    #         self.total_avail -= num_chunks
    #         return self.chunk_bag[self.total_avail : (self.total_avail + num_chunks)]
    #     else:
    #         return None
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def allocate_memory_chunks(
        self,
        context: "DynamicInferenceContext",
        num_chunks: int,
    ) -> Optional[Tensor]:
        """Allocate memory chunks if available, else return None.

        Args:
            context (DynamicInferenceContext): Dynamic inference context.
            num_chunks (int): Number of chunks to allocate.

        Return:
            (Optional[Tensor]) Allocated chunk IDs.
        """
        if self.is_memory_available(context, num_chunks):
            self.total_avail -= num_chunks
            chunk_ids = self.chunk_bag[self.total_avail : (self.total_avail + num_chunks)]
            # >>>
            # assert num_chunks == chunk_ids.numel()
            if num_chunks != chunk_ids.numel():
                pax("self, num_chunks")
            # <<<
            return chunk_ids
        else:
            return None
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
