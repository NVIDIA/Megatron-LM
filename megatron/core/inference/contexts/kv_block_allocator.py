# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

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
        max_kv_block_count: Optional[int] = None,
        max_requests: Optional[int] = None,
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
        self.total_avail = total_count
        self.paused_count = paused_count
        self.active_count = total_count - paused_count
        assert self.active_count >= 1  # ensures paused_count < total_count
        # Dummy sentinel ID. Reserved one past the real block range so the
        # graphed scatter paths can route invalid lanes to a slot that
        # never collides with a real block ID. The per-block tracking
        # tensors (block_hashes / block_ref_counts / block_timestamps,
        # plus the graphed _pc_released_bitmap) are sized total_count + 1
        # so this index is always valid.
        self.dummy_block_idx = self.total_count

        # Fall back to context when not explicitly passed (backwards compat).
        if max_requests is None:
            max_requests = context.max_requests
        if max_kv_block_count is None:
            max_kv_block_count = context.max_kv_block_count

        self.max_requests = max_requests
        self.max_release_per_step = max_requests * max_kv_block_count

        device = torch.cuda.current_device()

        # Initialize block pool as a stack data structure.
        # Pre-sized to hold the graphed release path's write footprint.
        self.block_bag = torch.empty(
            total_count + self.max_release_per_step, dtype=torch.int32, device=device
        )
        self.block_bag[:total_count] = torch.arange(
            total_count, dtype=torch.int32, device=device
        )
        self.block_bag[total_count:] = self.dummy_block_idx

        # GPU mirror of the stack pointer.
        self.total_avail_gpu = torch.tensor(
            [self.total_avail], dtype=torch.int32, device=device
        )

        # Static aranges for graphed release and allocate paths.
        self._release_arange_i32 = torch.arange(
            self.max_release_per_step, dtype=torch.int32, device=device
        )
        self._release_arange_i64 = torch.arange(
            self.max_release_per_step, dtype=torch.int64, device=device
        )
        self._alloc_arange = torch.arange(max_requests, dtype=torch.int32, device=device)
        self._alloc_arange_i64 = torch.arange(max_requests, dtype=torch.int64, device=device)

        if self.enable_prefix_caching:
            # Block hash tracking for prefix caching: -1 = uncomputed, positive = valid hash.
            # Sized total_count + 1 so the slot at ``dummy_block_idx`` is a no-op
            # target for the graphed scatter paths.
            self.block_hashes = torch.full(
                (self.total_count + 1,),
                -1,
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )

            # Hash-to-block mapping for O(1) prefix lookup
            self.kv_hash_to_block_id: Dict[int, int] = {}

            # Reference count per block: 0 = cached (evictable), >0 = actively used
            self.block_ref_counts = torch.zeros(
                (self.total_count + 1,),
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )

            # LRU timestamps for eviction ordering (higher = more recently used)
            # Only needed in LRU mode; RZ mode evicts immediately on ref_count==0
            if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
                self.block_timestamps = torch.zeros(
                    (self.total_count + 1,),
                    dtype=torch.int64,
                    device=torch.cuda.current_device(),
                )

            # ── Graphed-path scaffolding ──
            # Dereg queue: the graphed release path writes (block_id, old_hash)
            # pairs for blocks whose ref count drops to zero. Sized to hold
            # events from both graph bodies back-to-back without an intermediate
            # drain: `2 * max_release_per_step + 2 * max_requests + 1` sink slot.
            dereg_capacity = 2 * self.max_release_per_step + 2 * max_requests + 1
            self._pc_dereg_sink_idx = dereg_capacity - 1
            self._pc_pending_dereg_ids = torch.full(
                (dereg_capacity,), -1, dtype=torch.int32, device=device
            )
            self._pc_pending_dereg_hashes = torch.full(
                (dereg_capacity,), -1, dtype=torch.int64, device=device
            )
            self._pc_pending_dereg_count = torch.zeros((), dtype=torch.int64, device=device)
            self._pc_dereg_may_have_events: bool = False

            # Per-block workspace for deduplicating released block IDs.
            self._pc_released_bitmap = torch.zeros(
                (self.total_count,), dtype=torch.bool, device=device
            )
            self._pc_block_arange = torch.arange(
                self.total_count, dtype=torch.int32, device=device
            )

        # Per-block MoE routing storage (populated when routing replay is enabled)
        self.block_routing: Dict[int, np.ndarray] = {}

    def __str__(self):
        return (
            f"using: total {self.get_total_used()}/{self.total_count}"
            f"; active {self.get_active_used()}/{self.active_count}"
            f"; paused {self.get_paused_used()}/{self.paused_count}"
        )

    def get_total_used(self):
        """Compute number of total blocks used."""
        return self.total_count - self.total_avail

    def get_active_used(self):
        """Compute number of active blocks used."""
        active_count = self.context.total_request_count - self.context.paused_request_count
        if not self.enable_prefix_caching:
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
        if not self.enable_prefix_caching:
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
        self.total_avail_gpu.fill_(self.total_avail)

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
        self.total_avail_gpu.fill_(self.total_avail)

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
        # Fill in place to preserve the tensor's memory address (required for
        # static-address CUDA graph capture against the pre-sized block_bag).
        self.block_bag[: self.total_count].copy_(
            torch.arange(self.total_count, dtype=torch.int32, device=torch.cuda.current_device())
        )
        self.block_bag[self.total_count :].fill_(self.dummy_block_idx)

        self.total_avail = self.total_count
        self.total_avail_gpu.fill_(self.total_avail)

        if self.enable_prefix_caching:
            # Reset all block hashes
            self.block_hashes.fill_(-1)

            # Reset prefix caching state
            self.kv_hash_to_block_id.clear()
            self.block_ref_counts.fill_(0)
            if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
                self.block_timestamps.fill_(0)

            # Reset graphed-path scaffolding state.
            self._pc_pending_dereg_ids.fill_(-1)
            self._pc_pending_dereg_hashes.fill_(-1)
            self._pc_pending_dereg_count.zero_()
            self._pc_dereg_may_have_events = False

        # Clear per-block routing storage
        self.block_routing.clear()

    def drain_pending_dereg(self) -> None:
        """Apply queued dereg events to the host-side dict and callback.

        The graphed release path mutates all GPU-resident prefix caching
        state directly, but ``kv_hash_to_block_id`` is a Python dict and
        ``on_blocks_deregistered`` is a host callback — both inherently
        host-only. Instead of syncing per release, the graphed body queues
        (block_id, old_hash) pairs in ``_pc_pending_dereg_ids`` and
        ``_pc_pending_dereg_hashes``; this method drains the queue in one
        host pass.

        Call from ``update_requests`` after the final graph body finishes
        and before returning, so ``add_request`` (the only reader of the
        dict) sees a consistent view between steps.

        Idempotent. Skips the GPU sync entirely when no prefix-aware
        release has run since the last drain (via the host-side
        ``_pc_dereg_may_have_events`` flag).
        """
        if not self.enable_prefix_caching:
            return
        if not self._pc_dereg_may_have_events:
            return

        self._pc_dereg_may_have_events = False
        count = int(self._pc_pending_dereg_count.item())
        if count == 0:
            return

        # Pack ids (int32) and hashes (int64) into one int64 buffer and
        # do a single .cpu() to avoid two separate D2H syncs (was 3 in
        # the slow path: count.item() + ids.tolist() + hashes.tolist()).
        device = self._pc_pending_dereg_count.device
        combined = torch.empty(2 * count, dtype=torch.int64, device=device)
        combined[:count] = self._pc_pending_dereg_ids[:count].to(torch.int64)
        combined[count:] = self._pc_pending_dereg_hashes[:count]
        flat = combined.cpu().tolist()
        ids = flat[:count]
        hashes = flat[count:]

        keys_to_delete = set(hashes) - {-1}
        for h in keys_to_delete:
            self.kv_hash_to_block_id.pop(h, None)

        if self.on_blocks_deregistered is not None:
            self.on_blocks_deregistered(ids, keys_to_delete)

        self._pc_pending_dereg_count.zero_()

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
        self.total_avail_gpu.fill_(self.total_avail)

    def sync_to_cpu(self) -> None:
        """Mirror ``total_avail_gpu`` back into the Python int.

        The graphed allocate/release paths mutate ``total_avail_gpu``
        in-place; eager callers (``add_request``, etc.) read the Python
        ``total_avail``. Call this once per step (after the graphed
        bodies finish) so the host-visible counter stays consistent.
        """
        self.total_avail = int(self.total_avail_gpu.item())

    def allocate_memory_blocks_gpu(self, count_gpu: Tensor) -> Tensor:
        """Shape-stable allocate with GPU-resident state.

        Decrements ``total_avail_gpu`` by ``count_gpu`` and returns a fixed-
        size ``(max_requests,)`` tensor of block IDs read from ``block_bag``
        starting at the new stack pointer. Only the first ``count_gpu``
        entries are newly allocated; the remaining ``max_requests - count_gpu``
        entries are stale blocks above the new stack pointer and must not
        be used by the caller.

        Caller is responsible for ensuring ``count_gpu <= total_avail_gpu``;
        callers in ``update_requests`` already clamp via ``resume_count_gpu
        = min(fits_gpu, total_avail_gpu)`` before reaching this point.

        Args:
            count_gpu: 0-d or 1-element int tensor holding the number of
                blocks to allocate.

        Returns:
            Tensor of shape ``(max_requests,)`` with the allocated block
            IDs in the first ``count_gpu`` positions.
        """
        # Decrement the stack pointer in-place.
        self.total_avail_gpu.sub_(count_gpu.to(torch.int32).view(1))
        # Gather max_requests entries starting at the new pointer.
        indices = self.total_avail_gpu + self._alloc_arange
        return self.block_bag[indices.long()]

    def release_memory_blocks_gpu(self, packed_blocks: Tensor, num_valid_gpu: Tensor) -> None:
        """Shape-stable release that writes a pre-packed buffer onto the stack.

        The caller is responsible for packing valid block ids into the first
        ``num_valid_gpu`` entries of ``packed_blocks``; the remaining entries
        may contain any value and will be overwritten by future releases or
        left above the new stack pointer where they are never read.

        Args:
            packed_blocks: Fixed-size tensor of length ``max_release_per_step``,
                with valid block IDs packed at [0, num_valid_gpu).
            num_valid_gpu: 0-d or 1-element int tensor holding the valid count.
        """
        assert packed_blocks.numel() == self.max_release_per_step

        # Blind copy the whole packed buffer into block_bag at the current
        # stack pointer. Entries past the new stack top are ignored.
        bag_indices = self.total_avail_gpu + self._release_arange_i32
        self.block_bag[bag_indices.long()] = packed_blocks
        self.total_avail_gpu.add_(num_valid_gpu.to(torch.int32).view(1))

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
