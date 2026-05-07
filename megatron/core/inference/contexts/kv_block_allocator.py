# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, Optional

import torch
from torch import Tensor

from megatron.core.inference.config import PrefixCachingEvictionPolicy

from .moe_routing_replay import MoERoutingReplayCache
from .prefix_cache_block_state import PrefixCacheBlockState
from .prefix_cache_registry import PrefixCacheRegistry


class KVBlockAllocator:
    """Allocator that manages blocks of memory for the KV cache.

    This allocator owns:

    - The free-pool stack (``block_bag``, ``total_avail``).
    - Allocation, release, and reset orchestration.
    - The MoE routing-replay per-block storage (separate concern; lives
      here for now until the routing-replay refactor lands).

    Prefix-caching state — block hashes, ref counts, LRU timestamps, the
    register / deregister / evict primitives — is **not** in this class.
    When ``enable_prefix_caching=True``, the allocator constructs a
    :class:`PrefixCacheBlockState` and exposes it as ``self.pc_state``;
    otherwise ``self.pc_state`` is ``None`` and no prefix-caching state
    is allocated.

    Args:
        context (DynamicInferenceContext): Dynamic inference context.
        total_count (int): Total number of blocks in the buffer.
        paused_count (int): Number of paused blocks in the buffer. Must be less
            than ``total_count``.
        enable_prefix_caching (bool): When True, ``self.pc_state`` holds a
            :class:`PrefixCacheBlockState`; the prefix-caching code paths
            in this class delegate there.
        prefix_caching_eviction_policy (PrefixCachingEvictionPolicy):
            Eviction policy passed to :class:`PrefixCacheBlockState`.
        prefix_cache_registry (Optional[PrefixCacheRegistry]): Host-side
            ``hash -> block_id`` dict, shared with the Mamba slot
            allocator. Required when prefix caching is enabled.
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
        prefix_cache_registry: Optional[PrefixCacheRegistry] = None,
    ):

        self.context = context
        self.enable_prefix_caching = enable_prefix_caching
        self.prefix_caching_eviction_policy = prefix_caching_eviction_policy

        self.total_count = total_count
        self.total_avail = total_count - 1  # -1 for dummy_block_idx (see below)
        self.paused_count = paused_count
        self.active_count = total_count - paused_count - 1  # -1 for dummy_block_idx
        assert self.active_count >= 1  # ensures paused_count < total_count - 1
        self.dummy_block_idx = self.total_count - 1

        # Initialize block pool as a "stack" data structure (CPU for bookkeeping).
        self.block_bag = torch.arange(self.total_count, dtype=torch.int32, device='cpu')

        # Per-block prefix-caching state, only when enabled.
        self.pc_state: Optional[PrefixCacheBlockState]
        if self.enable_prefix_caching:
            assert (
                prefix_cache_registry is not None
            ), "enable_prefix_caching=True requires a PrefixCacheRegistry"
            self.pc_state = PrefixCacheBlockState(
                total_count=self.total_count,
                eviction_policy=self.prefix_caching_eviction_policy,
                registry=prefix_cache_registry,
            )
        else:
            self.pc_state = None

        # MoE routing-replay storage. Held as an attached object so the
        # allocator stays free of routing-replay state; methods on the
        # cache are the only entry points for store / reconstruct / clear.
        # `block_timestamps` no longer lives on the allocator — it moved
        # into ``PrefixCacheBlockState`` along with the other prefix-cache
        # bookkeeping fields.
        self.routing_replay = MoERoutingReplayCache(
            context=context,
            dummy_block_idx=self.dummy_block_idx,
            block_size_tokens=context.block_size_tokens,
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
        active_count = self.context.total_request_count - self.context.paused_request_count
        if self.pc_state is None:
            return self.context.request_kv_block_counts[:active_count].sum().item()

        if active_count > 0:
            active_rows = self.context.request_to_kv_block_ids[:active_count]
            valid_ids = active_rows[active_rows >= 0]
            if valid_ids.numel() > 0:
                return int(torch.unique(valid_ids).numel())
        return 0

    def get_paused_used(self):
        """Compute number of paused blocks used."""
        active_count = self.context.total_request_count - self.context.paused_request_count
        paused_end = self.context.total_request_count
        if self.pc_state is None:
            return self.context.request_kv_block_counts[active_count:paused_end].sum().item()

        if self.context.paused_request_count > 0:
            paused_rows = self.context.request_to_kv_block_ids[active_count:paused_end]
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
        if self.pc_state is None:
            return False
        if self.pc_state.eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO:
            return False  # RZ: no cached blocks to evict
        # Also count evictable cached blocks
        evictable_count = self.pc_state.get_evictable_block_count()
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
                self.pc_state is None
                or self.pc_state.eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO
            ):
                return None  # RZ: no eviction path; disabled: no cached blocks
            blocks_needed_from_eviction = num_blocks - self.total_avail
            if not self._evict_lru_for_pool(blocks_needed_from_eviction):
                return None  # Not enough blocks even after eviction

        # Now allocate from the free pool
        self.total_avail -= num_blocks
        block_ids = self.block_bag[self.total_avail : (self.total_avail + num_blocks)]
        assert num_blocks == block_ids.numel()

        if self.pc_state is not None:
            self.pc_state.on_allocate(block_ids, self.context.prefix_cache_lru_clock)

        # Clear stale routing data for re-allocated blocks. No-op when
        # routing replay is not in use (empty-dict short-circuit).
        self.routing_replay.clear_for_reallocated(block_ids.tolist())

        return block_ids

    def release_memory_blocks(self, blocks: Tensor) -> None:
        """Release memory blocks.

        Without prefix caching: blocks return directly to the free pool.

        With prefix caching: ref counts are decremented and the subset of
        blocks to actually return to the pool is policy-specific:

        - REF_ZERO: blocks whose ref count just hit zero are
          deregistered (hash cleared, registry evicted) and returned to
          the pool.
        - LRU: only unregistered (``hash == -1``) zero-ref blocks return
          to the pool. Hashed zero-ref blocks stay cached for reuse and
          are evicted via :meth:`_evict_lru_for_pool` on demand.

        Args:
            blocks (Tensor): Block IDs to release.
        """
        if blocks.numel() == 0:
            return

        if self.pc_state is None:
            self._push_to_pool(blocks)
            return

        pool_returns = self.pc_state.on_release_compute_pool_returns(blocks)
        if pool_returns.numel() > 0:
            self._push_to_pool(pool_returns)

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

        if self.pc_state is not None:
            self.pc_state.reset()

        # Reset routing-replay storage.
        self.routing_replay.reset()

    # =========================================================================
    # Pool management (private)
    # =========================================================================

    def _push_to_pool(self, blocks: Tensor) -> None:
        """Push blocks back onto the free-pool stack."""
        num_blocks = blocks.numel()
        if num_blocks == 0:
            return
        self.block_bag[self.total_avail : self.total_avail + num_blocks] = blocks
        self.total_avail += num_blocks

    def _evict_lru_for_pool(self, num_blocks_needed: int) -> bool:
        """Evict ``num_blocks_needed`` LRU blocks back to the free pool.

        Picks the oldest cached blocks via :meth:`PrefixCacheBlockState.find_lru_evictable`,
        deregisters them, and pushes them onto the pool. Returns False if
        not enough evictable blocks exist.
        """
        assert self.pc_state is not None
        blocks_to_evict = self.pc_state.find_lru_evictable(num_blocks_needed)
        if blocks_to_evict is None:
            return False
        self.pc_state.deregister_blocks(blocks_to_evict)
        self._push_to_pool(blocks_to_evict)
        return True

