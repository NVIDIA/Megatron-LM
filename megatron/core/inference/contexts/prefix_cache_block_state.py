# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import heapq
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from megatron.core.inference.config import PrefixCachingEvictionPolicy


class PrefixCacheBlockState:
    """Per-block CPU shadow state for prefix caching.

    Owns the tensor-side bookkeeping (hashes, ref counts, MTP successor tokens, LRU timestamps
    and the LRU parent chain) and exposes mutation primitives that return the hashes a caller
    must forward to `PrefixCacheRegistry`. The state class itself is registry-free so that the
    registry boundary stays the only seam between in-process and remote prefix-cache deployments.
    """

    def __init__(self, pool_size: int, eviction_policy: PrefixCachingEvictionPolicy):
        self.pool_size = pool_size
        self.eviction_policy = eviction_policy

        # `-1` = uncomputed; positive = registered hash.
        self.block_hashes = torch.full((pool_size,), -1, dtype=torch.int64, device='cpu')

        # `0` = cached/evictable; `>0` = actively held.
        self.block_ref_counts = torch.zeros((pool_size,), dtype=torch.int32, device='cpu')

        # Token the block's FINAL MTP draft slot was computed against, or -1 when that slot
        # holds no draft KV. A block's last draft entry pairs its last hidden with the first
        # token of the NEXT block, so it is reusable only by a consumer whose next token
        # matches; the hash alone does not determine it. -1 is the safe default: only the
        # prefill path that knows the producer's next token records one, so blocks
        # registered by any other route (a disaggregated import, say) stay uninheritable.
        self.block_mtp_next_token = torch.full((pool_size,), -1, dtype=torch.int64, device='cpu')

        # LRU only; REF_ZERO evicts immediately on `ref_count == 0`.
        if eviction_policy == PrefixCachingEvictionPolicy.LRU:
            self.block_timestamps = torch.zeros((pool_size,), dtype=torch.int64, device='cpu')

            # Persisted prefix-chain bookkeeping for LRU eviction, maintained
            # incrementally on register/deregister. Block hashes are parent-chained:
            # a cached block that is another cached block's parent must not be
            # evicted before its child (see `find_lru_evictable`).
            #
            # block_parent_id[b] = block id of b's parent in the prefix chain, or
            #   -1 when b is a root block or its parent is not registered.
            self.block_parent_id = torch.full((pool_size,), -1, dtype=torch.int64, device='cpu')
            # block_child_count[b] = number of currently-registered children of b.
            # For a cached block all of its children are cached too, so this equals
            # its cached-child count and b is an evictable leaf exactly when it hits 0.
            self.block_child_count = torch.zeros((pool_size,), dtype=torch.int64, device='cpu')
        else:
            self.block_timestamps = None
            self.block_parent_id = None
            self.block_child_count = None

    @property
    def is_lru(self) -> bool:
        """True when the eviction policy is LRU (`block_timestamps` is allocated)."""
        return self.eviction_policy == PrefixCachingEvictionPolicy.LRU

    def reset(self) -> None:
        """Reset all per-block state. Caller is responsible for clearing the registry."""
        self.block_hashes.fill_(-1)
        self.block_ref_counts.fill_(0)
        self.block_mtp_next_token.fill_(-1)
        if self.block_timestamps is not None:
            self.block_timestamps.zero_()
            self.block_parent_id.fill_(-1)
            self.block_child_count.zero_()

    # =========================================================================
    # Registration
    # =========================================================================

    def stamp_block_hashes(self, block_ids: List[int], block_hashes: List[int]) -> List[int]:
        """Write `block_hashes` into the per-block shadow tensor.

        Returns:
            Positions into the inputs of the blocks that were newly stamped, in input order.
        """
        if not block_ids:
            return []
        device = self.block_hashes.device
        id_tensor = torch.tensor(block_ids, dtype=torch.int64, device=device)
        hash_tensor = torch.tensor(block_hashes, dtype=torch.int64, device=device)

        # Hash each block holds right now; -1 means it is not registered.
        current_hashes = self.block_hashes[id_tensor]
        # Per-entry: this exact (block, hash) pair is already registered -> skip it.
        already_registered = current_hashes == hash_tensor
        # Per-entry: block is registered, but under some other hash -> illegal.
        conflict_mask = (current_hashes != -1) & ~already_registered
        conflicting = torch.nonzero(conflict_mask, as_tuple=True)[0].tolist()
        assert not conflicting, "block re-registered under a different hash: " + ", ".join(
            f"block {block_ids[i]} holds {int(current_hashes[i])}, given {block_hashes[i]}"
            for i in conflicting
        )

        keep = torch.nonzero(~already_registered, as_tuple=True)[0]
        if keep.numel() == 0:
            return []
        self.block_hashes[id_tensor[keep]] = hash_tensor[keep]
        return keep.tolist()

    def record_parent_chain(self, block_ids: List[int], parent_ids: List[int]) -> None:
        """Persist the LRU-eviction parent chain for freshly-registered blocks.

        `parent_ids[k]` is the block id of `block_ids[k]`'s parent (resolved from
        the parent hash by the caller via the registry), or -1 for a root or
        not-yet-registered parent. Bumps each resolved parent's child count so a
        parent only becomes an evictable leaf once its last child is gone. No-op
        unless LRU; see `find_lru_evictable` for how the chain constrains eviction.
        """
        if not self.is_lru or not block_ids:
            return
        id_tensor = torch.tensor(block_ids, dtype=torch.int64, device=self.block_hashes.device)
        parent_id_tensor = torch.tensor(parent_ids, dtype=torch.int64, device=id_tensor.device)
        self.block_parent_id[id_tensor] = parent_id_tensor
        has_parent = parent_id_tensor >= 0
        if has_parent.any():
            self.block_child_count.scatter_add_(
                0,
                parent_id_tensor[has_parent],
                torch.ones(int(has_parent.sum()), dtype=torch.int64),
            )

    # =========================================================================
    # Reference counting
    # =========================================================================

    def update_timestamps(self, block_ids: Tensor, lru_clock: int) -> None:
        """Stamp `block_timestamps[block_ids] = lru_clock`. No-op in REF_ZERO mode."""
        if not self.is_lru or block_ids.numel() == 0:
            return
        self.block_timestamps[block_ids] = lru_clock

    def on_allocate(self, block_ids: Tensor, lru_clock: int) -> None:
        """Initialize ref counts (and timestamps under LRU) for newly-allocated blocks.

        Called by the allocator immediately after popping `block_ids` from the free pool.
        """
        self.block_ref_counts[block_ids] = 1
        self.update_timestamps(block_ids, lru_clock)

    def retain(self, block_ids: Tensor, lru_clock: int) -> None:
        """Add one reference per occurrence in `block_ids` (and refresh LRU timestamps)."""
        if block_ids.numel() == 0:
            return
        unique_blocks, retain_counts = torch.unique(block_ids, return_counts=True)
        self.block_ref_counts[unique_blocks] += retain_counts.to(dtype=self.block_ref_counts.dtype)
        self.update_timestamps(unique_blocks, lru_clock)

    def get_evictable_block_count(self) -> Tensor:
        """Count of blocks that are cached (`ref_count == 0`) and have a registered hash."""
        cached_mask = (self.block_ref_counts == 0) & (self.block_hashes != -1)
        return cached_mask.sum()

    def extra_blocks_available(self) -> int:
        """How many cached blocks could be made available via eviction.

        Zero under REF_ZERO: in that policy a block whose ref count hits zero is immediately
        deregistered and pushed to the pool; the cached reservoir is always empty by construction.
        Equals the evictable count under LRU.
        """
        if not self.is_lru:
            return 0
        return int(self.get_evictable_block_count().item())

    # =========================================================================
    # Eviction / deregistration
    # =========================================================================

    def try_lru_evict_for_pool(self, num_blocks_needed: int) -> Optional[Tuple[Tensor, List[int]]]:
        """Pick + reset the LRU-oldest victims; return them plus their hashes.

        Returns:
            - `None` under REF_ZERO (no cached reservoir to evict from).
            - `None` under LRU when fewer than `num_blocks_needed` evictable blocks exist.
            - `(victims, hashes)` on success: `victims` is a tensor of exactly
              `num_blocks_needed` block IDs, `hashes` is the list of hashes
              the caller must drop from the registry. The caller (the allocator) is
              also responsible for pushing `victims` onto the free pool.
        """
        if not self.is_lru:
            return None
        victims = self.find_lru_evictable(num_blocks_needed)
        if victims is None:
            return None
        hashes = self.deregister_blocks(victims)
        return victims, hashes

    def find_lru_evictable(self, num_blocks_needed: int) -> Optional[Tensor]:
        """Return exactly `num_blocks_needed` evictable block IDs, or `None`.

        Least-recently-used first, but never a parent before its children. Block
        hashes are parent-chained and prefix matching relies on the invariant that
        a cached child always has all of its ancestors cached too; a naive
        oldest-first eviction breaks it (with chunked prefill an ancestor chunk can
        be allocated first, hence older, yet outlive a younger descendant). We peel
        the cached forest from its leaves inward with a min-heap: only a leaf (a
        cached block with no cached children) is evictable, and among evictable
        leaves we take the one with the oldest own timestamp; evicting a leaf can
        turn its parent into a leaf, which is then pushed onto the heap. Keying each
        block by its own recency (and reconsidering a parent only once its children
        are gone) is what makes this the optimal LRU generalization.

        `None` if fewer than `num_blocks_needed` evictable blocks exist. Callers
        must push the returned blocks back onto the free pool via the allocator
        after `deregister_blocks` clears their state.
        """
        assert self.is_lru, "find_lru_evictable requires LRU eviction policy"
        cached_mask = (self.block_ref_counts == 0) & (self.block_hashes != -1)
        cached_block_ids = torch.nonzero(cached_mask, as_tuple=True)[0]
        num_cached = cached_block_ids.numel()
        if num_cached < num_blocks_needed:
            return None
        if num_blocks_needed <= 0:
            return cached_block_ids[:0]

        ts = self.block_timestamps[cached_block_ids].tolist()
        bid = cached_block_ids.tolist()
        parent_global = self.block_parent_id[cached_block_ids].tolist()
        child_count = self.block_child_count[cached_block_ids].tolist()

        # Map a cached block's global id to its local index so the peel can find a
        # parent's slot to decrement. Parents that are not cached (root, or a
        # parent still in use) are absent and are simply treated as peel roots.
        global_to_local = {bid[i]: i for i in range(num_cached)}
        parent_local = [global_to_local.get(p, -1) for p in parent_global]

        # Min-heap of currently-evictable leaves keyed by (own timestamp, block
        # id). Block ids are unique, so the tie-break is total and deterministic.
        heap = [(ts[i], bid[i], i) for i in range(num_cached) if child_count[i] == 0]
        heapq.heapify(heap)

        evicted_local: List[int] = []
        while heap and len(evicted_local) < num_blocks_needed:
            _, _, i = heapq.heappop(heap)
            evicted_local.append(i)
            p = parent_local[i]
            if p >= 0:
                child_count[p] -= 1
                if child_count[p] == 0:
                    heapq.heappush(heap, (ts[p], bid[p], p))

        # A forest is always fully peelable, so the heap always exposes enough
        # leaves to collect num_blocks_needed (guaranteed by the num_cached check
        # above). Falling short means the parent graph is cyclic -- only possible
        # under a block-hash collision, which we treat as a bug.
        assert len(evicted_local) == num_blocks_needed, (
            f"leaf peel evicted {len(evicted_local)} of {num_blocks_needed} "
            f"requested from {num_cached} cached blocks; parent graph is not a "
            f"forest (likely a block-hash collision)"
        )
        return cached_block_ids[torch.tensor(evicted_local, dtype=torch.int64)]

    def deregister_blocks(self, block_ids: Tensor) -> List[int]:
        """Reset per-block state for `block_ids`; return their hashes.

        Does NOT touch the host hash registry and does NOT push the blocks
        back to the free pool -- both are the caller's responsibility.

        Used by both the LRU eviction path and the REF_ZERO release path.
        """
        if block_ids.numel() == 0:
            return []
        block_ids_i64 = block_ids.to(torch.int64)
        hashes = self.block_hashes[block_ids_i64].tolist()

        if self.block_timestamps is not None:
            # Drop these blocks from their parents' child counts before clearing
            # their own bookkeeping, so a parent becomes an evictable leaf once its
            # last child is deregistered (see `find_lru_evictable`).
            parent_ids = self.block_parent_id[block_ids_i64]
            has_parent = parent_ids >= 0
            if has_parent.any():
                self.block_child_count.scatter_add_(
                    0,
                    parent_ids[has_parent],
                    torch.full((int(has_parent.sum()),), -1, dtype=torch.int64),
                )
            self.block_parent_id[block_ids] = -1
            self.block_child_count[block_ids] = 0
            self.block_timestamps[block_ids] = 0

        # Reset per-block state (batched tensor ops).
        self.block_hashes[block_ids] = -1
        self.block_ref_counts[block_ids] = 0
        self.block_mtp_next_token[block_ids] = -1

        return hashes

    def on_release_compute_pool_returns(self, blocks: Tensor) -> Tuple[Tensor, List[int]]:
        """Drop one reference per occurrence; return `(blocks_for_pool, hashes_to_drop)`.

        Duplicate ids (a shared block released by several finishing owners in one batch)
        each drop one reference, but a block reaching zero is returned at most once.

        Policy-specific:

        - REF_ZERO: every block whose ref count reaches zero is reset; both its block ID
          and its hash are returned for the caller to push to the pool and drop
          from the registry, respectively.
        - LRU: only unregistered (`hash == -1`) zero-ref blocks are returned for
          the pool; their hashes are empty (already -1). Blocks with a registered
          hash stay cached for later reuse and are evicted on demand.

        The allocator pushes `blocks_for_pool` onto the block bag
        and forwards `hashes_to_drop` to the registry.
        """
        if blocks.numel() == 0:
            return blocks, []

        unique_blocks, release_counts = torch.unique(blocks, return_counts=True)
        remaining_ref_counts = self.block_ref_counts[unique_blocks] - release_counts.to(
            dtype=self.block_ref_counts.dtype
        )
        assert torch.all(
            remaining_ref_counts >= 0
        ), "released more KV block references than the allocator owns"
        self.block_ref_counts[unique_blocks] = remaining_ref_counts

        if self.eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO:
            zero_mask = remaining_ref_counts == 0
            if not zero_mask.any():
                return blocks[:0], []  # empty slice, preserves dtype/device
            zero_blocks = unique_blocks[zero_mask]
            hashes = self.deregister_blocks(zero_blocks)
            return zero_blocks, hashes

        # LRU: return only unregistered (no-hash) zero-ref blocks; cached blocks remain
        # in the pool's eviction-eligible reservoir until `find_lru_evictable` claims them.
        unreg_mask = (remaining_ref_counts == 0) & (self.block_hashes[unique_blocks] == -1)
        if not unreg_mask.any():
            return blocks[:0], []
        return unique_blocks[unreg_mask], []
