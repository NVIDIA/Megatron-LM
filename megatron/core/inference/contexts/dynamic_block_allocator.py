# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, Optional

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
        total_count (int): Total number of blocks in the buffer.
        paused_count (int): Number of paused blocks in the buffer. Must be less
            than `total_count`.
    """

    def __init__(self, context: "DynamicInferenceContext", total_count: int, paused_count: int):

        self.context = context

        self.total_count = total_count
        self.total_avail = total_count - 1  # -1 for dummy_block_idx (see below)
        self.paused_count = paused_count
        self.active_count = total_count - paused_count - 1  # -1 for dummy_block_idx
        assert self.active_count >= 1  # ensures paused_count < total_count - 1
        self.dummy_block_idx = self.total_count - 1

        # Initialize block pool as a "stack" data structure
        self.block_bag = torch.arange(
            self.total_count, dtype=torch.int32, device=torch.cuda.current_device()
        )

        # Block hash tracking for prefix caching: -1 = uncomputed, positive = valid hash
        self.block_hashes = torch.full(
            (self.total_count,), -1, dtype=torch.int64, device=torch.cuda.current_device()
        )

        # Prefix caching data structures
        # Hash-to-block mapping for O(1) prefix lookup
        self.hash_to_block_id: Dict[int, int] = {}

        # Reference count per block: 0 = cached (evictable), >0 = actively used
        self.block_ref_counts = torch.zeros(
            (self.total_count,), dtype=torch.int32, device=torch.cuda.current_device()
        )

        # LRU timestamps for eviction ordering (higher = more recently used)
        self.block_timestamps = torch.zeros(
            (self.total_count,), dtype=torch.int64, device=torch.cuda.current_device()
        )
        self.global_timestamp = 0

        # Pending block hashes for prefix caching coordination
        # Maps block_id -> hash for blocks registered but not yet computed
        self._pending_block_hashes: Dict[int, int] = {}

    def __str__(self):
        return (
            f"using: total {self.get_total_used()}/{self.total_count - 1}"
            f"; active {self.get_active_used()}/{self.active_count}"
            f"; paused {self.get_paused_used()}/{self.paused_count}"
        )

    def get_total_used(self):
        """Compute number of total blocks used."""
        return self.total_count - self.total_avail - 1

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

        Includes both free pool blocks and evictable cached blocks (ref_count == 0).

        Args:
            num_blocks (int): Number of blocks to check.

        Return:
            (bool) Is memory available?
        """
        # Fast path: avoid expensive evictable count computation when free pool suffices
        if self.total_avail >= num_blocks:
            return True
        # Also count evictable cached blocks
        evictable_count = self.get_evictable_block_count()
        return (self.total_avail + evictable_count) >= num_blocks

    def allocate_memory_blocks(self, num_blocks: int) -> Optional[Tensor]:
        """Allocate memory blocks if available, else return None.

        Will attempt LRU eviction of cached blocks if the free pool is insufficient.

        Args:
            num_blocks (int): Number of blocks to allocate.

        Return:
            (Optional[Tensor]) Allocated block IDs.
        """
        # Try to evict cached blocks if free pool is insufficient
        if self.total_avail < num_blocks:
            blocks_needed_from_eviction = num_blocks - self.total_avail
            if not self.evict_lru_blocks(blocks_needed_from_eviction):
                return None  # Not enough blocks even after eviction

        # Now allocate from the free pool
        self.total_avail -= num_blocks
        block_ids = self.block_bag[self.total_avail : (self.total_avail + num_blocks)]
        assert num_blocks == block_ids.numel()

        # Initialize ref counts and timestamps for newly allocated blocks
        self.block_ref_counts[block_ids] = 1
        self.update_timestamps(block_ids)

        return block_ids

    def release_memory_blocks(self, blocks: Tensor) -> None:
        """Release memory blocks by decrementing reference counts.

        Blocks with ref_count == 0 remain cached (in hash map) for potential reuse.
        They will be evicted via LRU when space is needed.

        Args:
            blocks (Tensor): Block IDs to release.

        Return:
            None
        """
        if blocks.numel() == 0:
            return

        # Decrement reference counts - blocks stay cached for prefix reuse
        self.decrement_ref_count(blocks)

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

        # Reset all block hashes
        self.block_hashes.fill_(-1)

        # Reset prefix caching state
        self.hash_to_block_id.clear()
        self._pending_block_hashes.clear()
        self.block_ref_counts.fill_(0)
        self.block_timestamps.fill_(0)
        self.global_timestamp = 0

    def set_block_hash(self, block_id: int, hash_value: int) -> None:
        """Set the hash for a specific block.

        Args:
            block_id: The block ID to set hash for.
            hash_value: The hash value to store.
        """
        self.block_hashes[block_id] = hash_value

    def get_block_hash(self, block_id: int) -> int:
        """Get the hash for a block.

        Args:
            block_id: The block ID to get hash for.

        Returns:
            Hash value (-1 if not computed).
        """
        return self.block_hashes[block_id].item()

    # =========================================================================
    # Prefix caching methods
    # =========================================================================

    def lookup_block_by_hash(self, block_hash: int) -> Optional[int]:
        """Look up a cached block by its hash.

        Args:
            block_hash: The hash value to look up.

        Returns:
            Block ID if found, None otherwise.
        """
        return self.hash_to_block_id.get(block_hash)

    def register_block_hash(self, block_id: int, block_hash: int) -> None:
        """Register a block in the hash-to-block mapping for discovery.

        NOTE: Does NOT mark block as computed. Call mark_block_computed() after
        KV is computed. This two-phase approach enables prefix caching coordination
        where subsequent requests wait for blocks to be computed before reusing.

        Args:
            block_id: The block ID.
            block_hash: The computed hash value.
        """
        # Store hash for later use, but block_hashes stays -1 until computed
        self._pending_block_hashes[block_id] = block_hash
        self.hash_to_block_id[block_hash] = block_id

    def mark_block_computed(self, block_id: int) -> None:
        """Mark a block as having its KV computed.

        Called after prefill completes for blocks that were registered.
        This sets block_hashes[block_id] to the actual hash value,
        signaling that the KV cache for this block is ready for reuse.

        Args:
            block_id: The block ID to mark as computed.
        """
        if block_id in self._pending_block_hashes:
            hash_value = self._pending_block_hashes.pop(block_id)
            self.set_block_hash(block_id, hash_value)

    def increment_ref_count(self, block_ids: Tensor) -> None:
        """Increment reference count for shared blocks.

        Called when a request starts using (sharing) existing cached blocks.

        Args:
            block_ids: Tensor of block IDs to increment.
        """
        if block_ids.numel() == 0:
            return
        self.block_ref_counts[block_ids] += 1

    def decrement_ref_count(self, block_ids: Tensor) -> None:
        """Decrement reference count when a request releases blocks.

        Blocks with ref_count == 0 become cached (evictable but still in hash map).

        Args:
            block_ids: Tensor of block IDs to decrement.
        """
        if block_ids.numel() == 0:
            return
        self.block_ref_counts[block_ids] -= 1

    def update_timestamps(self, block_ids: Tensor) -> None:
        """Update LRU timestamps for accessed blocks.

        Args:
            block_ids: Tensor of block IDs that were accessed.
        """
        if block_ids.numel() == 0:
            return
        self.global_timestamp += 1
        self.block_timestamps[block_ids] = self.global_timestamp

    def get_evictable_block_count(self) -> int:
        """Get count of cached blocks that can be evicted (ref_count == 0, hash set).

        Returns:
            Number of evictable cached blocks.
        """
        cached_mask = (self.block_ref_counts == 0) & (self.block_hashes != -1)
        return cached_mask.sum().item()

    def evict_lru_blocks(self, num_blocks_needed: int) -> bool:
        """Evict LRU cached blocks to free up space in the pool.

        Evicts blocks with ref_count == 0, starting with oldest timestamps.

        Args:
            num_blocks_needed: Number of blocks to evict.

        Returns:
            True if enough blocks were evicted, False otherwise.
        """
        # Find all cached blocks (ref_count == 0, hash != -1)
        cached_mask = (self.block_ref_counts == 0) & (self.block_hashes != -1)
        cached_block_ids = torch.nonzero(cached_mask, as_tuple=True)[0]

        if cached_block_ids.numel() < num_blocks_needed:
            return False  # Not enough cached blocks to evict

        # Sort by timestamp (ascending = oldest first)
        cached_timestamps = self.block_timestamps[cached_block_ids]
        sorted_indices = torch.argsort(cached_timestamps)
        blocks_to_evict = cached_block_ids[sorted_indices[:num_blocks_needed]]

        # Remove from hash mapping
        for block_id in blocks_to_evict:
            block_id_int = block_id.item()

            # Clean up pending hash if block was pending computation
            if block_id_int in self._pending_block_hashes:
                pending_hash = self._pending_block_hashes.pop(block_id_int)
                if pending_hash in self.hash_to_block_id:
                    del self.hash_to_block_id[pending_hash]

            # Clean up computed hash
            block_hash = self.block_hashes[block_id_int].item()
            if block_hash in self.hash_to_block_id:
                del self.hash_to_block_id[block_hash]

        # Reset block state
        self.block_hashes[blocks_to_evict] = -1
        self.block_ref_counts[blocks_to_evict] = 0
        self.block_timestamps[blocks_to_evict] = 0

        # Add back to free pool
        self.block_bag[self.total_avail : self.total_avail + num_blocks_needed] = blocks_to_evict
        self.total_avail += num_blocks_needed

        return True
