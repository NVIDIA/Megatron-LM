# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import heapq
from collections import deque
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from megatron.core.inference.config import PrefixCachingEvictionPolicy


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

        self.total_count = total_count
        self.total_avail = total_count - 1  # -1 for dummy_block_idx (see below)
        self.paused_count = paused_count
        self.active_count = total_count - paused_count - 1  # -1 for dummy_block_idx
        assert self.active_count >= 1  # ensures paused_count < total_count - 1
        self.dummy_block_idx = self.total_count - 1

        # Initialize block pool as a "stack" data structure (CPU for bookkeeping).
        self.block_bag = torch.arange(self.total_count, dtype=torch.int32, device='cpu')

        if self.enable_prefix_caching:
            # Block hash tracking for prefix caching: -1 = uncomputed, positive = valid hash
            self.block_hashes = torch.full((self.total_count,), -1, dtype=torch.int64, device='cpu')

            # Hash-to-block mapping for O(1) prefix lookup
            self.kv_hash_to_block_id: Dict[int, int] = {}

            # Parent hash per block for prefix-chain bookkeeping: 0 = no parent
            # (root block or unregistered). Block hashes are parent-chained, so a
            # cached block whose hash is some other cached block's parent must not
            # be evicted before its child (see evict_lru_blocks). Valid hashes are
            # in [1, 2^63-1], so 0 is a safe "no parent" sentinel.
            self.block_parent_hashes = torch.zeros(
                (self.total_count,), dtype=torch.int64, device='cpu'
            )

            # Reference count per block: 0 = cached (evictable), >0 = actively used
            self.block_ref_counts = torch.zeros(
                (self.total_count,), dtype=torch.int32, device='cpu'
            )

            # LRU timestamps for eviction ordering (higher = more recently used)
            # Only needed in LRU mode; RZ mode evicts immediately on ref_count==0
            if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
                self.block_timestamps = torch.zeros(
                    (self.total_count,), dtype=torch.int64, device='cpu'
                )

        # Per-block MoE routing storage (populated when routing replay is enabled)
        self.block_routing: Dict[int, np.ndarray] = {}

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

    def is_memory_available(self, num_blocks: int, num_evictable_to_exclude: int = 0) -> bool:
        """Check if memory blocks are available.

        Includes both free pool blocks and evictable cached blocks (ref_count == 0).

        Args:
            num_blocks (int): Number of blocks to check.
            num_evictable_to_exclude (int): Number of currently-evictable cached
                blocks to subtract from the evictable count because the caller
                will pin them before allocating (e.g. prefix-matched blocks that
                get their ref counts bumped in add_request). These blocks are
                ref_count == 0 now, so they are included in the evictable count,
                but they will be protected from eviction, so they cannot supply
                the requested ``num_blocks``.

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
        # Also count evictable cached blocks, excluding those the caller will pin.
        evictable_count = int(self.get_evictable_block_count()) - num_evictable_to_exclude
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

        # Clear stale routing data for re-allocated blocks
        for bid in block_ids.tolist():
            self.block_routing.pop(bid, None)

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
        self.block_bag = torch.arange(self.total_count, dtype=torch.int32, device='cpu')

        self.total_avail = self.total_count - 1

        if self.enable_prefix_caching:
            # Reset all block hashes
            self.block_hashes.fill_(-1)

            # Reset prefix caching state
            self.kv_hash_to_block_id.clear()
            self.block_parent_hashes.fill_(0)
            self.block_ref_counts.fill_(0)
            if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
                self.block_timestamps.fill_(0)

        # Clear per-block routing storage
        self.block_routing.clear()

    # =========================================================================
    # Prefix caching methods
    # =========================================================================

    def register_kv_block_hashes(
        self,
        block_ids: list[int],
        block_hashes: list[int],
        parent_hashes: Optional[list[int]] = None,
    ) -> None:
        """Register blocks in the hash-to-block mapping for discovery (batch).

        Args:
            block_ids: List of block IDs.
            block_hashes: List of computed hash values (same length as block_ids).
            parent_hashes: Parent hash for each block in the prefix chain (same
                length as block_ids); 0 marks a root block with no parent. Used
                by LRU eviction to avoid evicting a parent before its children.
                If None, parents default to 0.
        """
        if not block_ids:
            return
        id_tensor = torch.tensor(block_ids, dtype=torch.int64, device=self.block_hashes.device)
        hash_tensor = torch.tensor(block_hashes, dtype=torch.int64, device=self.block_hashes.device)
        self.block_hashes[id_tensor] = hash_tensor
        if parent_hashes is not None:
            assert len(parent_hashes) == len(block_ids)
            self.block_parent_hashes[id_tensor] = torch.tensor(
                parent_hashes, dtype=torch.int64, device=self.block_hashes.device
            )
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
        self.block_parent_hashes[block_ids] = 0
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

        Evicts blocks with ref_count == 0, least-recently-used first, while never
        evicting a parent before its children. Block hashes are parent-chained,
        and ``_find_kv_match_count`` relies on the invariant that a cached child
        block always has all of its ancestors cached too. A naive oldest-first
        eviction breaks this: with chunked prefill, earlier chunks are allocated
        first (older timestamps) yet are ancestors of later chunks (newer
        timestamps), so once the request finishes and its blocks are cached, an
        ancestor can be older than its descendant and get evicted first, leaving a
        dangling child.

        To preserve the invariant while staying optimal we peel the cached forest
        from its leaves inward with a min-heap: only a leaf (a cached block with
        no cached children) is ever evictable, and among the currently-evictable
        leaves we always take the one with the oldest *own* timestamp. Evicting a
        leaf can turn its parent into a leaf, which is then pushed onto the heap.
        Repeating ``num_blocks_needed`` times gives, at each step, the globally
        least-recently-used block that can be removed without orphaning a child —
        the natural generalization of LRU to the parent-chain constraint. Keying
        each block by its *own* recency (and only reconsidering a parent once its
        children are gone) is what makes this optimal: a block is retained purely
        because it is recently used, never because a hot descendant props it up,
        so a colder evictable block is always evicted before a hotter one.

        Worked example, evicting 3 from::

            A(ts 1) -> B(ts 2) -> C(ts 5)   (C, F are leaves under B)
                              \-> F(ts 3)
                    \-> D(ts 3) -> E(ts 5)   (E is a leaf under D)

        Leaf-peel evicts F(3), then C(5); B is now childless so it joins the
        leaves with its own ts=2 and is evicted next -> retains {A, D, E}, keeping
        the hottest block E(5) rather than the colder interior block B(2).

        Note: because a request holds a contiguous block prefix [0..k], any in-use
        (ref_count > 0) block keeps all of its ancestors in use too. Hence a cached
        (ref_count == 0) block can only have cached children, and considering the
        cached set alone is sufficient to avoid dangling children.

        The parent graph is assumed acyclic (a forest), which holds for any hashes
        produced by the prefix-chain builder; an assertion guards against a
        pathological hash collision wedging the peel.

        Args:
            num_blocks_needed: Number of blocks to evict.

        Returns:
            True if enough blocks were evicted, False otherwise.
        """
        # Find all cached blocks (ref_count == 0, hash != -1)
        cached_mask = (self.block_ref_counts == 0) & (self.block_hashes != -1)
        cached_block_ids = torch.nonzero(cached_mask, as_tuple=True)[0]

        num_cached = cached_block_ids.numel()
        if num_cached < num_blocks_needed:
            return False  # Not enough cached blocks to evict
        if num_blocks_needed <= 0:
            return True

        own_ts = self.block_timestamps[cached_block_ids]
        hashes = self.block_hashes[cached_block_ids]
        parents = self.block_parent_hashes[cached_block_ids]

        # Resolve each block's parent hash to the local index of the parent block,
        # or -1 when the parent is not cached (root block with parent hash 0, or a
        # parent still in use). Hashes are unique per cached block, so a single
        # sorted lookup suffices. This O(num_cached log num_cached) setup stays
        # vectorized; only the inherently-sequential peel below is per-element.
        sorted_hashes, sort_order = torch.sort(hashes)
        pos = torch.searchsorted(sorted_hashes, parents).clamp(max=num_cached - 1)
        found = sorted_hashes[pos] == parents
        parent_idx = torch.where(found, sort_order[pos], torch.full_like(pos, -1))
        has_parent = parent_idx >= 0

        # Number of cached children per block; a block is an evictable leaf once
        # this reaches 0.
        child_count = torch.zeros(num_cached, dtype=torch.int64)
        child_count.scatter_add_(
            0, parent_idx[has_parent], torch.ones(int(has_parent.sum()), dtype=torch.int64)
        )

        parent_local = parent_idx.tolist()
        child_count = child_count.tolist()
        ts = own_ts.tolist()
        bid = cached_block_ids.tolist()

        # Min-heap of currently-evictable leaves keyed by (own timestamp, block
        # id). Block ids are unique, so the tie-break is total and deterministic.
        heap = [(ts[i], bid[i], i) for i in range(num_cached) if child_count[i] == 0]
        heapq.heapify(heap)

        evicted_local = []
        while heap and len(evicted_local) < num_blocks_needed:
            _, _, i = heapq.heappop(heap)
            evicted_local.append(i)
            p = parent_local[i]
            if p >= 0:
                child_count[p] -= 1
                if child_count[p] == 0:
                    heapq.heappush(heap, (ts[p], bid[p], p))

        # A forest is always fully peelable, so the heap always exposes enough
        # leaves to collect num_blocks_needed (guaranteed by the num_cached >=
        # num_blocks_needed check above). Falling short means the parent graph is
        # cyclic — only possible under a hash collision, which we treat as a bug.
        assert len(evicted_local) == num_blocks_needed, (
            f"leaf peel evicted {len(evicted_local)} of {num_blocks_needed} "
            f"requested from {num_cached} cached blocks; parent graph is not a "
            f"forest (likely a block-hash collision)"
        )

        blocks_to_evict = cached_block_ids[torch.tensor(evicted_local, dtype=torch.int64)]
        self._deregister_blocks(blocks_to_evict)

        return True

    # =========================================================================
    # Per-block routing storage methods (for MoE routing replay)
    # =========================================================================

    def store_routing_per_block(self, flat_routing: Optional[np.ndarray]) -> None:
        """Scatter flat routing indices into per-block storage.

        Uses the context's token-to-block mapping to distribute each token's
        routing data into the appropriate block. Matched (prefix-cached) blocks
        already have routing from the original request and are not overwritten
        here since their tokens are not in the active token layout.

        Args:
            flat_routing: ndarray of shape [active_token_count, num_layers, topk]
                aligned with the context's active-token layout, or None.
        """
        if flat_routing is None:
            return

        context = self.context
        token_count = context.active_token_count
        if token_count == 0:
            return

        assert (
            flat_routing.shape[0] == token_count
        ), f"Routing token count {flat_routing.shape[0]} != active token count {token_count}"

        # Token-to-block mapping for all active tokens
        block_ids_np = context.token_to_block_idx[:token_count].cpu().numpy()
        positions_np = context.token_to_local_position_within_kv_block[:token_count].cpu().numpy()

        dummy = self.dummy_block_idx

        # Group tokens by block_id using sort for efficient scatter
        unique_blocks, inverse, counts = np.unique(
            block_ids_np, return_inverse=True, return_counts=True
        )
        sorted_indices = np.argsort(inverse, kind='stable')
        sorted_positions = positions_np[sorted_indices]
        sorted_routing = flat_routing[sorted_indices]

        offset = 0
        for bid, count in zip(unique_blocks, counts):
            bid = int(bid)
            count = int(count)
            if bid == dummy:
                offset += count
                continue
            block_pos = sorted_positions[offset : offset + count]
            block_rout = sorted_routing[offset : offset + count]
            self.store_block_routing(bid, block_pos, block_rout)
            offset += count

    def reconstruct_routing_from_blocks(
        self, block_ids: list[int], total_routing_tokens: int
    ) -> Optional[np.ndarray]:
        """Reconstruct routing indices from per-block storage.

        Concatenates per-block routing ndarrays in block order, trimming the
        last block to exactly ``total_routing_tokens`` entries.

        Args:
            block_ids: Ordered list of block IDs for the request.
            total_routing_tokens: Expected number of routing tokens
                (total_tokens - 1, since the last generated token has no
                forward-pass routing).

        Returns:
            ndarray [total_routing_tokens, num_layers, topk] or None if any
            block is missing routing data.
        """
        block_size = self.context.block_size_tokens
        routing_parts = []
        tokens_collected = 0

        for bid in block_ids:
            routing = self.get_block_routing(bid)
            if routing is None:
                return None  # Missing routing data for this block
            remaining = total_routing_tokens - tokens_collected
            if remaining <= 0:
                break
            take = min(block_size, remaining)
            routing_parts.append(routing[:take])
            tokens_collected += take

        if not routing_parts or tokens_collected != total_routing_tokens:
            return None

        return np.concatenate(routing_parts, axis=0)

    def store_block_routing(
        self, block_id: int, positions: np.ndarray, routing: np.ndarray
    ) -> None:
        """Store routing indices for specific token positions in a block.

        Args:
            block_id: The block ID.
            positions: ndarray of token positions within the block (1D, int).
            routing: ndarray of routing data [num_positions, num_layers, topk].
        """
        if block_id not in self.block_routing:
            self.block_routing[block_id] = np.zeros(
                (self.context.block_size_tokens, routing.shape[-2], routing.shape[-1]),
                dtype=routing.dtype,
            )
        self.block_routing[block_id][positions] = routing

    def get_block_routing(self, block_id: int) -> Optional[np.ndarray]:
        """Get routing indices for a block.

        Args:
            block_id: The block ID.

        Returns:
            ndarray [block_size_tokens, num_layers, topk] or None if not stored.
        """
        return self.block_routing.get(block_id)
