# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from collections import deque
from typing import Callable, Dict, Optional

import torch
from torch import Tensor

from megatron.core.inference.config import PrefixCachingEvictionPolicy

from .step_journal import ResourceReservation


class KVBlockAllocator:
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

    def __init__(
        self,
        context: "DynamicInferenceContext",
        total_count: int,
        paused_count: int,
        enable_prefix_caching: bool = False,
        prefix_caching_eviction_policy: PrefixCachingEvictionPolicy = (
            PrefixCachingEvictionPolicy.REF_ZERO
        ),
    ):

        self.context = context
        self.enable_prefix_caching = enable_prefix_caching
        self.prefix_caching_eviction_policy = prefix_caching_eviction_policy
        self.on_blocks_deregistered: Optional[Callable] = None
        self._next_reservation_id = 0
        self._open_reservations: Dict[int, ResourceReservation] = {}
        self._committed_reservations: Dict[int, ResourceReservation] = {}
        self._rolled_back_reservations: Dict[int, ResourceReservation] = {}
        self._deferred_snapshot_block_releases: Dict[int, list[Tensor]] = {}

        self.total_count = total_count
        self.total_avail = total_count - 1  # -1 for dummy_block_idx (see below)
        self.paused_count = paused_count
        self.active_count = total_count - paused_count - 1  # -1 for dummy_block_idx
        assert self.active_count >= 1  # ensures paused_count < total_count - 1
        self.dummy_block_idx = self.total_count - 1

        # Initialize block pool as a "stack" data structure (CPU for bookkeeping).
        self.block_bag = torch.arange(
            self.total_count, dtype=torch.int32, device='cpu',
        )

        if self.enable_prefix_caching:
            # Block hash tracking for prefix caching: -1 = uncomputed, positive = valid hash
            self.block_hashes = torch.full(
                (self.total_count,), -1, dtype=torch.int64, device='cpu',
            )

            # Hash-to-block mapping for O(1) prefix lookup
            self.kv_hash_to_block_id: Dict[int, int] = {}

            # Reference count per block: 0 = cached (evictable), >0 = actively used
            self.block_ref_counts = torch.zeros(
                (self.total_count,), dtype=torch.int32, device='cpu',
            )

            # LRU timestamps for eviction ordering (higher = more recently used)
            # Only needed in LRU mode; RZ mode evicts immediately on ref_count==0
            if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
                self.block_timestamps = torch.zeros(
                    (self.total_count,), dtype=torch.int64, device='cpu',
                )

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
        if not self.enable_prefix_caching:
            return (
                self.context.request_kv_block_counts[
                    self.context.paused_request_count : self.context.total_request_count
                ]
                .sum()
                .item()
            )

        active_start = self.context.paused_request_count
        active_end = self.context.total_request_count
        if active_end > active_start:
            active_rows = self.context.request_to_kv_block_ids[active_start:active_end]
            valid_ids = active_rows[active_rows >= 0]
            if valid_ids.numel() > 0:
                return int(torch.unique(valid_ids).numel())
        return 0

    def get_paused_used(self):
        """Compute number of paused blocks used."""
        if not self.enable_prefix_caching:
            return (
                self.context.request_kv_block_counts[: self.context.paused_request_count]
                .sum()
                .item()
            )

        if self.context.paused_request_count > 0:
            paused_rows = self.context.request_to_kv_block_ids[: self.context.paused_request_count]
            valid_ids = paused_rows[paused_rows >= 0]
            if valid_ids.numel() > 0:
                return int(torch.unique(valid_ids).numel())
        return 0

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
        if not self.enable_prefix_caching:
            return False
        if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO:
            return False  # RZ: no cached blocks to evict
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
            if (
                not self.enable_prefix_caching
                or self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO
            ):
                return None  # RZ: no eviction path; disabled: no cached blocks
            blocks_needed_from_eviction = num_blocks - self.total_avail
            if not self.evict_lru_blocks(blocks_needed_from_eviction):
                return None  # Not enough blocks even after eviction

        # Now allocate from the free pool
        self.total_avail -= num_blocks
        block_ids = self.block_bag[self.total_avail : (self.total_avail + num_blocks)]
        assert num_blocks == block_ids.numel()

        if self.enable_prefix_caching:
            # Initialize ref counts for newly allocated blocks
            self.block_ref_counts[block_ids] = 1
            if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
                self.update_timestamps(block_ids)

        return block_ids

    def reserve_blocks(
        self, request_slot: int, count: int, step_id: int, snapshot_slot_id: int = 0
    ) -> Optional[ResourceReservation]:
        """Reserve KV blocks for a scheduled-but-not-retired step."""
        if count == 0:
            block_ids = torch.empty(0, dtype=torch.int32, device='cpu')
        else:
            block_ids = self.allocate_memory_blocks(count)
            if block_ids is None:
                return None
        reservation = ResourceReservation(
            step_id=int(step_id),
            request_slot=int(request_slot),
            snapshot_slot_id=int(snapshot_slot_id),
            reservation_id=self._next_reservation_id,
            kv_block_ids=tuple(int(block_id) for block_id in block_ids.tolist()),
        )
        self._next_reservation_id += 1
        self._open_reservations[reservation.reservation_id] = reservation
        return reservation

    def commit_reservation(self, reservation: ResourceReservation) -> None:
        """Commit a KV reservation into durable request ownership."""
        open_reservation = self._open_reservations.pop(reservation.reservation_id, None)
        if open_reservation is None:
            if reservation.reservation_id in self._committed_reservations:
                return
            raise RuntimeError(f"KV reservation {reservation.reservation_id} is not open")
        self._committed_reservations[reservation.reservation_id] = open_reservation
        if hasattr(self.context, "async_overlap_debug_counters"):
            self.context.async_overlap_debug_counters["reservation_commits"] += 1

    def rollback_reservation(self, reservation: ResourceReservation) -> None:
        """Roll back a KV reservation and return its blocks to the allocator."""
        open_reservation = self._open_reservations.pop(reservation.reservation_id, None)
        if open_reservation is None:
            if reservation.reservation_id in self._rolled_back_reservations:
                return
            raise RuntimeError(f"KV reservation {reservation.reservation_id} is not open")
        block_ids = torch.tensor(open_reservation.kv_block_ids, dtype=torch.int32, device='cpu')
        self.release_memory_blocks(block_ids)
        self._rolled_back_reservations[reservation.reservation_id] = open_reservation
        if hasattr(self.context, "async_overlap_debug_counters"):
            self.context.async_overlap_debug_counters["reservation_rollbacks"] += 1

    def defer_release_until_snapshot_retired(
        self, block_ids: Tensor, snapshot_slot_id: int
    ) -> None:
        """Defer block release until the owning snapshot can no longer read them."""
        if block_ids.numel() == 0:
            return
        self._deferred_snapshot_block_releases.setdefault(int(snapshot_slot_id), []).append(
            block_ids.to(dtype=torch.int32, device='cpu').clone()
        )

    def release_deferred_blocks_for_snapshot(self, snapshot_slot_id: int) -> None:
        """Release blocks whose snapshot-read dependency has retired."""
        deferred = self._deferred_snapshot_block_releases.pop(int(snapshot_slot_id), [])
        for block_ids in deferred:
            self.release_memory_blocks(block_ids)

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

        if self.enable_prefix_caching:
            self.block_ref_counts[blocks] -= 1
            if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO:
                zero_mask = self.block_ref_counts[blocks] == 0
                if zero_mask.any():
                    self._deregister_blocks(blocks[zero_mask])
            elif self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
                # Unregistered blocks (hash == -1, ref_count == 0) have no hash
                # entry to preserve for reuse (e.g., partial blocks at the end of
                # a request). Return them directly to the free pool so they are not
                # leaked.
                unreg_mask = (self.block_ref_counts[blocks] == 0) & (
                    self.block_hashes[blocks] == -1
                )
                if unreg_mask.any():
                    unreg_blocks = blocks[unreg_mask]
                    num_unreg = unreg_blocks.numel()
                    self.block_bag[self.total_avail : self.total_avail + num_unreg] = unreg_blocks
                    self.total_avail += num_unreg
        else:
            num_blocks = blocks.numel()
            self.block_bag[self.total_avail : self.total_avail + num_blocks] = blocks
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
            self.total_count, dtype=torch.int32, device='cpu',
        )

        self.total_avail = self.total_count - 1
        self._open_reservations.clear()
        self._committed_reservations.clear()
        self._rolled_back_reservations.clear()
        self._deferred_snapshot_block_releases.clear()
        self._next_reservation_id = 0

        if self.enable_prefix_caching:
            # Reset all block hashes
            self.block_hashes.fill_(-1)

            # Reset prefix caching state
            self.kv_hash_to_block_id.clear()
            self.block_ref_counts.fill_(0)
            if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
                self.block_timestamps.fill_(0)

    # =========================================================================
    # Prefix caching methods
    # =========================================================================

    def register_kv_block_hashes(self, block_ids: list[int], block_hashes: list[int]) -> None:
        """Register blocks in the hash-to-block mapping for discovery (batch).

        Args:
            block_ids: List of block IDs.
            block_hashes: List of computed hash values (same length as block_ids).
        """
        if not block_ids:
            return
        id_tensor = torch.tensor(block_ids, dtype=torch.int64, device=self.block_hashes.device)
        hash_tensor = torch.tensor(block_hashes, dtype=torch.int64, device=self.block_hashes.device)
        self.block_hashes[id_tensor] = hash_tensor
        self.kv_hash_to_block_id.update(zip(block_hashes, block_ids))

    def _deregister_blocks(self, block_ids: Tensor) -> None:
        """Remove blocks from prefix caching state and return to free pool.

        Shared cleanup logic for both LRU eviction and RZ proactive eviction.

        Args:
            block_ids: Tensor of block IDs to deregister.
        """
        num_blocks = block_ids.numel()
        if num_blocks == 0:
            return

        # Gather hashes via batched tensor indexing
        block_ids_i64 = block_ids.to(torch.int64)
        hashes = self.block_hashes[block_ids_i64].tolist()

        # Remove from kv_hash_to_block_id dict (set ops + C-level map, no Python loop)
        keys_to_delete = set(hashes) - {-1}
        deque(
            map(self.kv_hash_to_block_id.pop, keys_to_delete & self.kv_hash_to_block_id.keys()),
            maxlen=0,
        )

        # Notify Mamba slot allocator (if wired) to clean up its state
        if self.on_blocks_deregistered is not None:
            self.on_blocks_deregistered(block_ids.tolist(), keys_to_delete)

        # Reset block state (batched tensor ops)
        self.block_hashes[block_ids] = -1
        self.block_ref_counts[block_ids] = 0
        if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
            self.block_timestamps[block_ids] = 0

        # Return blocks to free pool
        self.block_bag[self.total_avail : self.total_avail + num_blocks] = block_ids
        self.total_avail += num_blocks

    def update_timestamps(self, block_ids: Tensor) -> None:
        """Update LRU timestamps for accessed blocks. No-op in RZ mode.

        Args:
            block_ids: Tensor of block IDs that were accessed.
        """
        if (
            self.prefix_caching_eviction_policy != PrefixCachingEvictionPolicy.LRU
            or block_ids.numel() == 0
        ):
            return
        self.block_timestamps[block_ids] = self.context.prefix_cache_lru_clock

    def get_evictable_block_count(self) -> Tensor:
        """Get count of cached blocks that can be evicted (ref_count == 0, hash set).

        Returns:
            Scalar tensor with the number of evictable cached blocks.
        """
        cached_mask = (self.block_ref_counts == 0) & (self.block_hashes != -1)
        return cached_mask.sum()

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

        self._deregister_blocks(blocks_to_evict)

        return True
