# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional

import torch
from torch import Tensor

from megatron.core.inference.config import PrefixCachingEvictionPolicy

from .prefix_cache_registry import PrefixCacheRegistry


class PrefixCacheBlockState:
    """Per-block CPU shadow state for prefix caching.

    Owned by :class:`KVBlockAllocator` only when prefix caching is enabled
    (``allocator.pc_state`` is ``None`` otherwise). All per-block state
    that is meaningful only under prefix caching lives here:

    - ``block_hashes`` ``(total_count,)`` int64 — ``-1`` = uncomputed,
      otherwise the registered hash for the block.
    - ``block_ref_counts`` ``(total_count,)`` int32 — ``0`` = cached and
      evictable; ``> 0`` = actively held by at least one request.
    - ``block_timestamps`` ``(total_count,)`` int64 — LRU policy only;
      ``None`` for REF_ZERO.

    Coordinates with :class:`PrefixCacheRegistry` for the host
    ``hash -> block_id`` dict on register / evict. Does **not** own the
    free-pool stack pointer or the block bag; the allocator owns those.
    The split keeps ``KVBlockAllocator`` free of any prefix-caching
    state, so a non-prefix-caching deployment never instantiates this
    class.
    """

    def __init__(
        self,
        total_count: int,
        eviction_policy: PrefixCachingEvictionPolicy,
        registry: PrefixCacheRegistry,
    ):
        self.total_count = total_count
        self.eviction_policy = eviction_policy
        self.registry = registry

        # ``-1`` = uncomputed; positive = registered hash.
        self.block_hashes = torch.full((total_count,), -1, dtype=torch.int64, device='cpu')

        # ``0`` = cached/evictable; ``>0`` = actively held.
        self.block_ref_counts = torch.zeros((total_count,), dtype=torch.int32, device='cpu')

        # LRU only; REF_ZERO evicts immediately on ``ref_count == 0`` and
        # has no use for per-block timestamps.
        if eviction_policy == PrefixCachingEvictionPolicy.LRU:
            self.block_timestamps = torch.zeros((total_count,), dtype=torch.int64, device='cpu')
        else:
            self.block_timestamps = None

    @property
    def is_lru(self) -> bool:
        """True when the eviction policy is LRU (``block_timestamps`` is allocated)."""
        return self.eviction_policy == PrefixCachingEvictionPolicy.LRU

    def reset(self) -> None:
        """Reset all per-block state and clear the host KV registry."""
        self.block_hashes.fill_(-1)
        self.block_ref_counts.fill_(0)
        if self.block_timestamps is not None:
            self.block_timestamps.zero_()
        self.registry.clear_kv()

    # =========================================================================
    # Hash registration
    # =========================================================================

    def register_kv_block_hashes(
        self, block_ids: List[int], block_hashes: List[int]
    ) -> None:
        """Register blocks in the hash-to-block mapping for discovery (batch)."""
        if not block_ids:
            return
        id_tensor = torch.tensor(block_ids, dtype=torch.int64, device=self.block_hashes.device)
        hash_tensor = torch.tensor(
            block_hashes, dtype=torch.int64, device=self.block_hashes.device
        )
        self.block_hashes[id_tensor] = hash_tensor
        self.registry.register_kv(block_ids, block_hashes)

    # =========================================================================
    # LRU bookkeeping
    # =========================================================================

    def update_timestamps(self, block_ids: Tensor, lru_clock: int) -> None:
        """Stamp ``block_timestamps[block_ids] = lru_clock``. No-op in REF_ZERO mode."""
        if not self.is_lru or block_ids.numel() == 0:
            return
        self.block_timestamps[block_ids] = lru_clock

    def on_allocate(self, block_ids: Tensor, lru_clock: int) -> None:
        """Initialize ref counts (and timestamps under LRU) for newly-allocated blocks.

        Called by the allocator immediately after popping ``block_ids`` from
        the free pool.
        """
        self.block_ref_counts[block_ids] = 1
        self.update_timestamps(block_ids, lru_clock)

    # =========================================================================
    # Eviction-time helpers
    # =========================================================================

    def get_evictable_block_count(self) -> Tensor:
        """Count of blocks that are cached (``ref_count == 0``) and have a registered
        hash. Returned as a 0-d tensor."""
        cached_mask = (self.block_ref_counts == 0) & (self.block_hashes != -1)
        return cached_mask.sum()

    def find_lru_evictable(self, num_blocks_needed: int) -> Optional[Tensor]:
        """Return up to ``num_blocks_needed`` LRU-oldest evictable block IDs.

        ``None`` if not enough evictable blocks exist. The returned tensor is
        always exactly ``num_blocks_needed`` entries on success. Callers are
        responsible for pushing the evicted blocks back onto the free pool
        via the allocator after :meth:`deregister_blocks` clears their state.
        """
        assert self.is_lru, "find_lru_evictable requires LRU eviction policy"
        cached_mask = (self.block_ref_counts == 0) & (self.block_hashes != -1)
        cached_block_ids = torch.nonzero(cached_mask, as_tuple=True)[0]
        if cached_block_ids.numel() < num_blocks_needed:
            return None
        cached_timestamps = self.block_timestamps[cached_block_ids]
        sorted_indices = torch.argsort(cached_timestamps)
        return cached_block_ids[sorted_indices[:num_blocks_needed]]

    def deregister_blocks(self, block_ids: Tensor) -> None:
        """Drop hashes from the registry and reset per-block state.

        Does NOT push the blocks back to the free pool — that is the
        allocator's responsibility, called immediately after this returns.

        Used by both the LRU eviction path (after :meth:`find_lru_evictable`
        picks the victims) and the REF_ZERO release path (after
        :meth:`on_release_compute_pool_returns` detects newly-zero ref
        counts).
        """
        if block_ids.numel() == 0:
            return
        block_ids_i64 = block_ids.to(torch.int64)
        hashes = self.block_hashes[block_ids_i64].tolist()

        # Drop hashes from the registry. The cascade fires the Mamba
        # evict callback, which clears the corresponding Mamba GPU slots.
        self.registry.evict_kv(hashes)

        # Reset per-block state (batched tensor ops).
        self.block_hashes[block_ids] = -1
        self.block_ref_counts[block_ids] = 0
        if self.block_timestamps is not None:
            self.block_timestamps[block_ids] = 0

    # =========================================================================
    # Release path
    # =========================================================================

    def on_release_compute_pool_returns(self, blocks: Tensor) -> Tensor:
        """Decrement ref counts and return the subset that should go back to the pool.

        Policy-specific:

        - **REF_ZERO**: every block whose ref count just hit zero is
          deregistered (hash cleared, registry evicted, callback fired)
          and returned for pool insertion.
        - **LRU**: only unregistered (``hash == -1``) zero-ref blocks are
          returned. Blocks with a registered hash stay cached for later
          reuse and are evicted via :meth:`find_lru_evictable` on demand.

        The allocator pushes the returned tensor onto the block bag.
        """
        if blocks.numel() == 0:
            return blocks

        self.block_ref_counts[blocks] -= 1

        if self.eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO:
            zero_mask = self.block_ref_counts[blocks] == 0
            if not zero_mask.any():
                return blocks[:0]  # empty slice, preserves dtype/device
            zero_blocks = blocks[zero_mask]
            self.deregister_blocks(zero_blocks)
            return zero_blocks

        # LRU: return only unregistered (no-hash) zero-ref blocks; cached
        # blocks remain in the pool's eviction-eligible reservoir until
        # ``find_lru_evictable`` claims them.
        unreg_mask = (self.block_ref_counts[blocks] == 0) & (self.block_hashes[blocks] == -1)
        if not unreg_mask.any():
            return blocks[:0]
        return blocks[unreg_mask]
