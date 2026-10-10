# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from .prefix_cache_block_state import PrefixCacheBlockState
from .prefix_cache_registry import PrefixCacheRegistry

# Block deregistration observers are currently registered only by DynamoHelper.
BlocksDeregisteredObserver = Callable[[list[int], set[int]], None]


class KVBlockAllocator:
    """Allocator that manages blocks of memory for the KV cache.

    This allocator owns:

    - The free-pool stack (`block_bag`, `pool_avail`).
    - Allocation, release, retain, and reset orchestration.
    - The MoE routing-replay per-block storage.

    Args:
        context (DynamicInferenceContext): Dynamic inference context.
        pool_size (int): Number of blocks in the pool, including the dummy block.
        paused_limit (int): Paused-request block retention limit. Must leave at
            least one non-dummy block outside the limit.
        pc_state (Optional[PrefixCacheBlockState]): Per-block prefix-caching state.
            `None` disables prefix caching entirely on this allocator.
        prefix_cache_registry (Optional[PrefixCacheRegistry]): Host hash registry shared
            with the Mamba allocator. Given exactly when `pc_state` is given.
    """

    def __init__(
        self,
        context: "DynamicInferenceContext",
        pool_size: int,
        paused_limit: int,
        pc_state: Optional[PrefixCacheBlockState] = None,
        prefix_cache_registry: Optional[PrefixCacheRegistry] = None,
    ):

        self.context = context
        assert (pc_state is None) == (
            prefix_cache_registry is None
        ), "pc_state and prefix_cache_registry must be given together"
        if pc_state is not None:
            assert pc_state.pool_size == pool_size, "pc_state must span the whole block pool"
        self.pc_state = pc_state
        self.registry = prefix_cache_registry
        self._blocks_deregistered_observers: list[BlocksDeregisteredObserver] = []

        # Handoff blocks remain pinned until decode finishes pulling them.
        # Pinning at request finish only happens on engines with KV transfer
        # configured (setup_kv_transfer flips this on); other engines have no
        # release path for the pins.
        self.enable_handoff_pinning = False

        assert (
            0 <= paused_limit <= pool_size - 2
        ), "paused block limit must leave at least one usable block outside the limit"

        self.pool_size = pool_size
        self.pool_avail = pool_size - 1  # Raw free-pool count; -1 for dummy_block_idx.
        self.paused_limit = paused_limit
        self.dummy_block_idx = self.pool_size - 1

        # Initialize block pool as a "stack" data structure (CPU for bookkeeping).
        self.block_bag = torch.arange(self.pool_size, dtype=torch.int32, device='cpu')

        # Per-block MoE routing storage (populated when routing replay is enabled)
        self.block_routing: Dict[int, np.ndarray] = {}

    @property
    def enable_prefix_caching(self) -> bool:
        """True when this allocator carries prefix-caching state."""
        return self.pc_state is not None

    def __str__(self):
        return (
            f"blocks: occupied {self.get_total_used()}/{self.pool_size - 1}"
            f"; allocatable {self.get_allocatable_count()}"
            f"; active-used {self.get_active_used()}"
            f"; paused-used {self.get_paused_used()}/{self.paused_limit}"
        )

    def get_total_used(self):
        """Compute number of physical blocks outside the free pool."""
        return self.pool_size - self.pool_avail - 1

    def get_active_used(self):
        """Compute number of active blocks used."""
        if self.pc_state is None:
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
        if self.pc_state is None:
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

    def is_memory_available(self, num_blocks: int, potential_matched_count: int = 0) -> bool:
        """Check if memory blocks are available.

        Includes both free pool blocks and registered, evictable cached blocks.

        Args:
            num_blocks (int): Number of blocks to check.
            potential_matched_count (int): Number of currently-evictable cached
                blocks to subtract from the evictable count because the caller
                will pin them before allocating (e.g. prefix-matched blocks that
                get their ref counts bumped in add_request). These blocks are
                ref_count == 0 now, so they are included in the evictable count,
                but they will be protected from eviction, so they cannot supply
                the requested ``num_blocks``.

        Return:
            (bool) Is memory available?
        """
        # Fast path: avoid computing the evictable count when the free pool
        # suffices. Soon-to-be-pinned matches do not affect raw free capacity.
        if self.pool_avail >= num_blocks:
            return True
        return self.get_allocatable_count() - potential_matched_count >= num_blocks

    def allocate_memory_blocks(self, num_blocks: int) -> Optional[Tensor]:
        """Allocate memory blocks if available, else return None.

        Under LRU prefix caching, falls back to evicting cached blocks when the free pool is short.
        Returns `None` when even eviction cannot satisfy the request.

        Args:
            num_blocks (int): Number of blocks to allocate.

        Return:
            (Optional[Tensor]) Allocated block IDs.
        """
        # Try to evict cached blocks if free pool is insufficient.
        if self.pool_avail < num_blocks:
            if not self.evict_lru_blocks(num_blocks - self.pool_avail):
                return None  # RZ / disabled: no eviction path; LRU: not enough cached blocks.

        # Now allocate from the free pool
        self.pool_avail -= num_blocks
        block_ids = self.block_bag[self.pool_avail : (self.pool_avail + num_blocks)]
        assert num_blocks == block_ids.numel()

        if self.pc_state is not None:
            self.pc_state.on_allocate(block_ids, self.context.prefix_cache_lru_clock)

        # Clear stale routing data for re-allocated blocks
        for bid in block_ids.tolist():
            self.block_routing.pop(bid, None)

        return block_ids

    def release_memory_blocks(self, blocks: Tensor) -> None:
        """Release memory blocks.

        Without prefix caching: blocks return directly to the free pool.
        With prefix caching: one reference per occurrence is dropped, and blocks that reach
        zero are released according to the eviction policy (REF_ZERO deregisters them at
        once; LRU keeps registered blocks cached and returns only unregistered ones).

        Args:
            blocks (Tensor): Block IDs to release.
        """
        if blocks.numel() == 0:
            return

        if self.pc_state is None:
            self._push_to_pool(blocks)
            return

        pool_returns, hashes_to_drop = self.pc_state.on_release_compute_pool_returns(blocks)
        self._push_to_pool(pool_returns)
        if hashes_to_drop:
            self._notify_deregistered(pool_returns, hashes_to_drop)

    def retain_memory_blocks(self, block_ids: list[int]) -> None:
        """Add one prefix-cache reference to each block.

        Args:
            block_ids: Blocks retained by a new owner.
        """
        assert self.pc_state is not None, "retaining KV blocks requires prefix caching"
        if block_ids:
            blocks = torch.tensor(block_ids, dtype=torch.int32, device='cpu')
            self.pc_state.retain(blocks, self.context.prefix_cache_lru_clock)

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
        # Refill the existing buffer so it remains mutable when reset runs under
        # torch.inference_mode(), such as during CUDA graph setup.
        torch.arange(self.pool_size, out=self.block_bag)

        self.pool_avail = self.pool_size - 1

        if self.pc_state is not None:
            self.pc_state.reset()
            self.registry.clear_kv()

        # Clear per-block routing storage
        self.block_routing.clear()

    def _push_to_pool(self, blocks: Tensor) -> None:
        """Push blocks back onto the free-pool stack."""
        num_blocks = blocks.numel()
        if num_blocks == 0:
            return
        self.block_bag[self.pool_avail : self.pool_avail + num_blocks] = blocks
        self.pool_avail += num_blocks

    def _notify_deregistered(self, block_ids: Tensor, hashes: list[int]) -> None:
        """Drop deregistered hashes from the registry, then notify the observers."""
        keys_to_delete = set(hashes) - {-1}
        self.registry.evict_kv(keys_to_delete)
        block_ids_list = block_ids.tolist()
        for observer in tuple(self._blocks_deregistered_observers):
            observer(block_ids_list, keys_to_delete)

    # =========================================================================
    # Prefix caching methods
    # =========================================================================

    def register_kv_block_hashes(
        self,
        block_ids: list[int],
        block_hashes: list[int],
        parent_hashes: Optional[list[int]] = None,
    ) -> list[int]:
        """Register blocks in the hash-to-block mapping for discovery (batch).

        Registration is idempotent: a block that already carries the hash being
        registered is skipped. Callers may legitimately re-offer an already
        registered block (a cache-matched block whose block-table slot a later
        prefill chunk also spans), and the bookkeeping below is one-shot per
        block -- applying it twice adds a second child entry to the block's
        parent that no deregistration can ever cancel, leaving that parent
        permanently short of ``child_count == 0`` and therefore never an
        evictable leaf (see ``PrefixCacheBlockState.find_lru_evictable``).

        Re-registering a live block under a *different* hash would instead
        overwrite its recorded parent while leaving the previous parent's child
        count raised, so that case is rejected rather than absorbed.

        This method never touches reference counts. New blocks are pinned at
        ``ref_count == 1`` by ``allocate_memory_blocks``, and additional owners
        of an already registered block are pinned by the caller that matched it.

        Args:
            block_ids: List of block IDs.
            block_hashes: List of computed hash values (same length as block_ids).
            parent_hashes: Parent hash for each block in the prefix chain (same
                length as block_ids); 0 marks a root block with no parent. Used
                by LRU eviction to avoid evicting a parent before its children.
                If None, parents default to 0.

        Returns:
            Newly registered block IDs, in input order. Already registered blocks
            are excluded so callers can preserve their existing metadata.
        """
        if not block_ids:
            return []
        if parent_hashes is not None:
            assert len(parent_hashes) == len(block_ids)

        # Stamp the per-block shadow;
        # this filters already-registered blocks and rejects hash changes on live blocks.
        keep = self.pc_state.stamp_block_hashes(block_ids, block_hashes)
        if len(keep) != len(block_ids):
            if not keep:
                return []
            block_ids = [block_ids[i] for i in keep]
            block_hashes = [block_hashes[i] for i in keep]
            if parent_hashes is not None:
                parent_hashes = [parent_hashes[i] for i in keep]

        # Add the new blocks to the hash map first so that a block whose parent is
        # elsewhere in this same batch (block k's parent is block k-1) resolves.
        # Skipped blocks are already in the map, so they resolve as parents too.
        self.registry.register_kv(block_ids, block_hashes)

        if self.pc_state.is_lru:
            # Persist the resolved parent block id and bump each parent's child count.
            # Parents are earlier in the prefix chain and already registered
            # (a matched block or a prior chunk / earlier entry in this batch),
            # so a valid parent hash resolves; 0 marks a root and an unknown hash
            # falls back to -1.
            if parent_hashes is None:
                parent_hashes = [0] * len(block_ids)
            kv_map = self.registry.kv_hash_to_block_id
            parent_ids = [kv_map.get(ph, -1) if ph != 0 else -1 for ph in parent_hashes]
            self.pc_state.record_parent_chain(block_ids, parent_ids)
        return block_ids

    def add_blocks_deregistered_observer(self, observer: BlocksDeregisteredObserver) -> None:
        """Register a callback invoked when cached blocks are deregistered.

        Currently used only by DynamoHelper.
        """
        self._blocks_deregistered_observers.append(observer)

    def get_evictable_block_count(self) -> Tensor:
        """Get count of cached blocks that can be evicted (ref_count == 0, hash set).

        Returns:
            Scalar tensor with the number of evictable cached blocks.
        """
        return self.pc_state.get_evictable_block_count()

    def get_allocatable_count(self) -> int:
        """Compute the number of blocks available for allocation.

        Includes both blocks in the free pool and, under LRU prefix caching,
        registered ref-zero blocks that can be evicted.

        Returns:
            Number of blocks that can currently be allocated.
        """
        if self.pc_state is None:
            return self.pool_avail
        return self.pool_avail + self.pc_state.extra_blocks_available()

    def evict_lru_blocks(self, num_blocks_needed: int) -> bool:
        """Evict LRU cached blocks to free up space in the pool.

        Args:
            num_blocks_needed: Number of blocks to evict.

        Returns:
            True if enough blocks were evicted, False otherwise (also False without
            prefix caching or under REF_ZERO, which keep no evictable reservoir).
        """
        if self.pc_state is None:
            return False
        result = self.pc_state.try_lru_evict_for_pool(num_blocks_needed)
        if result is None:
            return False
        victims, hashes = result
        self._push_to_pool(victims)
        self._notify_deregistered(victims, hashes)
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
