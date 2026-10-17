# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from megatron.core.ssm.ops.gdp.common import CHUNK_SIZE as GDP_CHUNK_SIZE

from .attention_context.mamba_metadata import PrefixCachedMambaMetadata
from .prefix_cache_registry import PrefixCacheRegistry

if TYPE_CHECKING:
    from .dynamic_context import DynamicInferenceContext


class MambaSlotCapacityError(RuntimeError):
    """Raised when the durable Mamba cache cannot satisfy an allocation."""

    def __init__(self, required: int, available: int):
        self.required = required
        self.available = available
        super().__init__(
            f"Mamba cache requires {required} new durable slots, but only "
            f"{available} free or evictable slots are available"
        )


class MambaSlotAllocator:
    """Owns the Mamba prefix cache pool: cached conv/ssm state plus the block-to-slot map.

    Constructed only when prefix caching is enabled and `prefix_caching_mamba_gb > 0`.
    The host `hash -> block_id` map lives on the shared `PrefixCacheRegistry`;
    the intermediate-state extraction buffers and per-request commit bookkeeping live on
    `PrefixCachedMambaMetadata`.
    This allocator owns just the cache pool resources and the commit orchestration.

    Args:
        context: The DynamicInferenceContext that owns this allocator.
        max_slots: Maximum number of cache slots.
        num_mamba_layers: Number of Mamba layers in the model.
        conv_states_shape: Shape of per-slot conv state (excluding layer/slot dims).
        ssm_states_shape: Shape of per-slot SSM state (excluding layer/slot dims).
        conv_states_dtype: Dtype for conv state tensors.
        ssm_states_dtype: Dtype for SSM state tensors.
        prefix_cache_registry: Host hash registry; the Mamba evict callback is wired here.
    """

    def __init__(
        self,
        context: "DynamicInferenceContext",
        max_slots: int,
        num_mamba_layers: int,
        conv_states_shape: tuple,
        ssm_states_shape: tuple,
        conv_states_dtype: torch.dtype,
        ssm_states_dtype: torch.dtype,
        prefix_cache_registry: PrefixCacheRegistry,
    ):
        self.context = context
        self.max_slots = max_slots
        self.num_mamba_layers = num_mamba_layers
        self.registry = prefix_cache_registry
        self.registry.clear_mamba()
        self.registry.set_mamba_evict_callback(self._on_mamba_evicted)

        # The commit path reads the per-request records and scratch buffers that
        # only the prefix-cached metadata carries, so make that coupling explicit.
        assert isinstance(context.mamba_metadata, PrefixCachedMambaMetadata), (
            "MambaSlotAllocator requires context.mamba_metadata to be a "
            f"PrefixCachedMambaMetadata; got {type(context.mamba_metadata).__name__}."
        )
        assert context.mamba_metadata.intermediate_ssm_out is not None, (
            "PrefixCachedMambaMetadata.allocate_scratch_buffers() must run before "
            "constructing MambaSlotAllocator."
        )

        # PrefixCachedMambaMetadata.compute_and_store_offsets() records extraction
        # offsets on the model-wide SSM chunk quantum, and each mixer converts those
        # offsets to a row of its own per-chunk states using its own chunk size. That
        # conversion is exact only if the quantum is a multiple of that chunk size.
        # Which chunk size to check against follows from the Householder count:
        # ssm_chunking() asserts a homogeneous SSM stack, so a nonzero count means
        # every SSM layer is Gated Delta Product, whose prefill kernels chunk at their
        # own fixed size and for which mamba_chunk_size is an unused default.
        if context.gdp_num_householder > 0:
            assert context.ssm_chunk_alignment % GDP_CHUNK_SIZE == 0, (
                f"SSM chunk alignment must be a multiple of the GDP chunk size "
                f"({GDP_CHUNK_SIZE}); got {context.ssm_chunk_alignment}."
            )
        else:
            assert context.ssm_chunk_alignment % context.mamba_chunk_size == 0, (
                "SSM chunk alignment must be a multiple of mamba_chunk_size "
                f"({context.mamba_chunk_size}); got {context.ssm_chunk_alignment}."
            )
        gpu_device = torch.cuda.current_device()
        num_blocks = context.kv_block_allocator.pool_size

        # Block <-> slot mappings (CPU for bookkeeping).
        self.block_to_slot = torch.full((num_blocks,), -1, dtype=torch.int32, device='cpu')
        self.slot_to_block = torch.full((max_slots,), -1, dtype=torch.int32, device='cpu')

        # Free slot pool (stack, CPU).
        self.free_slots = torch.arange(max_slots, dtype=torch.int32, device='cpu')
        self.free_count = max_slots

        # Durable cache state tensors (GPU - accessed by Mamba CUDA kernels):
        # one slot per cached block boundary, reused across requests. Sized to
        # `max_slots`; the budget accounting in DynamicInferenceContext refers to
        # these as the "durable" buffers.
        self.conv_states = torch.zeros(
            (num_mamba_layers, max_slots) + conv_states_shape,
            dtype=conv_states_dtype,
            device=gpu_device,
        )
        self.ssm_states = torch.zeros(
            (num_mamba_layers, max_slots) + ssm_states_shape,
            dtype=ssm_states_dtype,
            device=gpu_device,
        )

        # The host hash -> block_id map for Mamba-cached blocks lives on
        # `self.registry.mamba_hash_to_block_id`. The intermediate extraction buffers and
        # per-request commit bookkeeping live on `context.mamba_metadata`.

    # =========================================================================
    # Slot allocation
    # =========================================================================

    def allocate_slots_batch(self, block_ids: list) -> list:
        """Get free Mamba cache slots for multiple blocks, evicting if necessary.

        Handles deduplication: if the same block_id appears multiple times,
        only one slot is allocated and all occurrences get the same slot.

        Args:
            block_ids: List of KV block IDs to associate with slots.

        Returns:
            List of allocated slot indices (same length as block_ids).
        """
        if not block_ids:
            return []

        device = self.block_to_slot.device
        bid_tensor = torch.tensor(block_ids, dtype=torch.int64, device=device)

        # Phase 1: Batch lookup existing slots (1 GPU sync)
        existing_slots = self.block_to_slot[bid_tensor].tolist()

        # Phase 2: Identify new blocks needing allocation (deduplicated)
        # seen_new maps block_id -> index in new_bids list
        seen_new = {}
        new_bids = []
        for i, (bid, slot) in enumerate(zip(block_ids, existing_slots)):
            if slot < 0 and bid not in seen_new:
                seen_new[bid] = len(new_bids)
                new_bids.append(bid)

        num_new = len(new_bids)
        if num_new == 0:
            return existing_slots

        # Reserve the full batch atomically. A failed eviction must not consume
        # the free portion of the request.
        need_evict = max(0, num_new - self.free_count)
        evictable_block_ids = (
            self._evictable_block_ids()
            if need_evict > 0
            else torch.empty(0, dtype=torch.int64, device=device)
        )
        available = self.free_count + evictable_block_ids.numel()
        if available < num_new:
            raise MambaSlotCapacityError(required=num_new, available=available)

        # Phase 3: Get slots from free pool, evicting if necessary
        from_free = min(num_new, self.free_count)
        new_slots = []
        if from_free > 0:
            start = self.free_count - from_free
            new_slots = self.free_slots[start : self.free_count].tolist()  # 1 GPU sync
            self.free_count = start

        need_evict = num_new - from_free
        if need_evict > 0:
            new_slots.extend(self._evict_lru_slots_batch(need_evict, evictable_block_ids))

        # Phase 4: Batch GPU writes for new mappings
        new_bid_tensor = torch.tensor(new_bids, dtype=torch.int64, device=device)
        new_slot_tensor = torch.tensor(new_slots, dtype=torch.int64, device=device)
        self.block_to_slot[new_bid_tensor] = new_slot_tensor.to(torch.int32)
        self.slot_to_block[new_slot_tensor] = new_bid_tensor.to(torch.int32)

        # Phase 5: Build result mapping
        # Map new block_ids to their allocated slots
        alloc_bid_to_slot = {bid: slot for bid, slot in zip(new_bids, new_slots)}
        result = []
        for bid, existing in zip(block_ids, existing_slots):
            if existing >= 0:
                result.append(existing)
            else:
                result.append(alloc_bid_to_slot[bid])
        return result

    def _evictable_block_ids(self) -> Tensor:
        """Return blocks whose durable Mamba slots have no live KV owner."""

        kv_alloc = self.context.kv_block_allocator
        pc_state = kv_alloc.pc_state
        has_slot_mask = self.block_to_slot[: kv_alloc.pool_size] >= 0
        ref_zero_mask = pc_state.block_ref_counts[: kv_alloc.pool_size] == 0
        return torch.nonzero(has_slot_mask & ref_zero_mask, as_tuple=True)[0]

    def _evict_lru_slots_batch(self, num_needed: int, candidate_ids: Tensor) -> list:
        """Evict the least recently used Mamba cache slots.

        Does NOT return slots to the free pool — caller takes ownership.

        Args:
            num_needed: Number of slots to evict.
            candidate_ids: Blocks confirmed to be evictable for this allocation.

        Returns:
            List of freed slot indices.
        """
        pc_state = self.context.kv_block_allocator.pc_state
        assert candidate_ids.numel() >= num_needed

        # Pick oldest blocks by timestamp (LRU) or first N (REF_ZERO)
        if pc_state.is_lru:
            timestamps = pc_state.block_timestamps[candidate_ids]
            _, oldest_indices = torch.topk(timestamps, k=num_needed, largest=False, sorted=False)
            evict_ids = candidate_ids[oldest_indices]
        else:
            evict_ids = candidate_ids[:num_needed]

        # Batch gather slots + hashes (2 GPU syncs)
        slots = self.block_to_slot[evict_ids].tolist()
        hashes = pc_state.block_hashes[evict_ids].tolist()

        # Clear block <-> slot mappings up front. The registry's evict callback then
        # runs `_on_mamba_evicted -> _invalidate_blocks_batch`, which short-circuits
        # because `block_to_slot[bid] < 0` for all evicted blocks: the slots are NOT
        # returned to the free pool (the caller takes ownership of them).
        self.block_to_slot[evict_ids] = -1
        slot_tensor = torch.tensor(slots, dtype=torch.int64, device=self.block_to_slot.device)
        self.slot_to_block[slot_tensor] = -1

        self.registry.evict_mamba(hashes)

        return slots

    def get_slot(self, block_id: int) -> int:
        """Return the cache slot for a block, or -1 if none.

        Args:
            block_id: The KV block ID.

        Returns:
            Slot index or -1.
        """
        return self.block_to_slot[block_id].item()

    def has_state(self, block_id: int) -> bool:
        """Check if a block has cached Mamba state."""
        return self.block_to_slot[block_id].item() >= 0

    # =========================================================================
    # Slot invalidation and deregistration
    # =========================================================================

    def invalidate_block(self, block_id: int) -> None:
        """Free cache slot and clear mappings for a block.

        Args:
            block_id: The KV block ID.
        """
        slot = self.block_to_slot[block_id].item()
        if slot < 0:
            return
        self.block_to_slot[block_id] = -1
        self.slot_to_block[slot] = -1
        # Return slot to free pool
        self.free_slots[self.free_count] = slot
        self.free_count += 1

    def _invalidate_blocks_batch(self, block_ids_list: list) -> None:
        """Free cache slots and clear mappings for multiple blocks at once.

        Vectorized version of invalidate_block that avoids per-block .item()
        GPU syncs. Used by ``_on_mamba_evicted`` for bulk slot release.

        Args:
            block_ids_list: List of block IDs to invalidate.
        """
        if not block_ids_list:
            return
        bid_t = torch.tensor(block_ids_list, dtype=torch.int64, device=self.block_to_slot.device)
        slots = self.block_to_slot[bid_t].to(torch.int64)
        valid_mask = slots >= 0
        if not valid_mask.any():
            return
        valid_slots = slots[valid_mask]
        valid_bids = bid_t[valid_mask]
        self.block_to_slot[valid_bids] = -1
        self.slot_to_block[valid_slots] = -1
        n = valid_slots.numel()
        self.free_slots[self.free_count : self.free_count + n] = valid_slots.to(torch.int32)
        self.free_count += n

    def _on_mamba_evicted(self, block_ids: list) -> None:
        """Registry callback: free GPU slots for the given block IDs.

        Fired by `PrefixCacheRegistry` after the host Mamba dict drops entries: either as a
        cascade from KV eviction (KV blocks gone means their cached Mamba state is also gone)
        or from an explicit `registry.evict_mamba` call.

        `_invalidate_blocks_batch` no-ops on blocks whose `block_to_slot` is already `-1`,
        which lets `_evict_lru_slots_batch` suppress the free-pool return by clearing
        mappings before firing the registry call.
        """
        self._invalidate_blocks_batch(block_ids)

    # =========================================================================
    # State store/restore
    # =========================================================================

    def store_from_tensors(
        self, block_id: int, layer_idx: int, ssm_state: Tensor, conv_state: Tensor
    ) -> None:
        """Write provided state tensors to a cache slot for a specific layer.

        Args:
            block_id: The KV block ID.
            layer_idx: The Mamba layer index.
            ssm_state: SSM state tensor to store.
            conv_state: Conv state tensor to store.
        """
        slot = self.block_to_slot[block_id].item()
        assert slot >= 0, f"Block {block_id} has no Mamba cache slot"
        self.ssm_states[layer_idx, slot].copy_(ssm_state)
        self.conv_states[layer_idx, slot].copy_(conv_state)

    def store_from_live_batch(self, slots: list, request_indices: list) -> None:
        """Copy all layers from live per-request buffers to cache slots.

        Args:
            slots: List of cache slot indices.
            request_indices: List of context request indices.
        """
        if not slots:
            return
        device = self.conv_states.device
        slot_tensor = torch.tensor(slots, dtype=torch.int64, device=device)
        # Lookup mamba indices from CPU bookkeeping, then move to GPU for state copy.
        req_tensor_cpu = torch.tensor(request_indices, dtype=torch.int64)
        mamba_indices = self.context.mamba_metadata.request_to_mamba_state_idx[
            req_tensor_cpu
        ].tolist()
        mamba_idx_tensor = torch.tensor(mamba_indices, dtype=torch.int64, device=device)
        # Fancy-indexed copy (2 kernel launches instead of 2E)
        self.conv_states[:, slot_tensor] = self.context.mamba_conv_states[:, mamba_idx_tensor]
        self.ssm_states[:, slot_tensor] = self.context.mamba_ssm_states[:, mamba_idx_tensor]

    def restore_to_live(self, request_idx: int, block_id: int) -> bool:
        """Copy all layers from cache slot to live request state.

        Args:
            request_idx: The context request index.
            block_id: The KV block ID.

        Returns:
            True if state was restored, False if block has no cached state.
        """
        slot = self.block_to_slot[block_id].item()
        if slot < 0:
            return False
        mamba_idx = self.context.mamba_metadata.request_to_mamba_state_idx[request_idx].item()
        self.context.mamba_conv_states[:, mamba_idx].copy_(self.conv_states[:, slot])
        self.context.mamba_ssm_states[:, mamba_idx].copy_(self.ssm_states[:, slot])
        return True

    # =========================================================================
    # Hash registration
    # =========================================================================

    def register_block_hashes_batch(self, block_ids: list, hashes: list) -> None:
        """Register multiple blocks as having cached Mamba state.

        Only registers entries where hash > 0.

        Args:
            block_ids: List of block IDs.
            hashes: List of hash values (same length as block_ids).
        """
        self.registry.register_mamba(block_ids, hashes)

    # =========================================================================
    # Intermediate state commit (copies intermediate buffers + live EOS state into cache slots)
    # =========================================================================

    def commit_intermediate_states(self) -> None:
        """Move intermediate states recorded this step into the cache pool.

        Called after the forward pass (including CUDA graph replay) completes. Pulls
        per-request commit data from `context.mamba_metadata`, allocates slots for the new
        entries, copies the GPU intermediates into the cache pool, copies live EOS state
        where applicable, and registers Mamba hashes.
        """
        collected = self._collect_commit_data()
        if collected is None:
            return
        intermediate_bids, src_offsets, eos_bids, eos_ctx_indices, all_hashes = collected

        # These snapshots only improve future cache hits; the active requests
        # continue from their live Mamba state if durable capacity is exhausted.
        all_bids = intermediate_bids + eos_bids
        n_intermediate = len(intermediate_bids)
        try:
            all_slots = self.allocate_slots_batch(all_bids)
        except MambaSlotCapacityError as error:
            existing_slots = self.block_to_slot[all_bids].tolist()
            kept_indices = []
            kept_new_bids = set()
            for index, (block_id, slot) in enumerate(zip(all_bids, existing_slots)):
                if slot >= 0 or block_id in kept_new_bids:
                    kept_indices.append(index)
                elif len(kept_new_bids) < error.available:
                    kept_new_bids.add(block_id)
                    kept_indices.append(index)

            if not kept_indices:
                self.context.mamba_metadata.clear_intermediate_state(self.context)
                return

            all_bids = [all_bids[index] for index in kept_indices]
            all_hashes = [all_hashes[index] for index in kept_indices]
            src_offsets = [src_offsets[index] for index in kept_indices if index < n_intermediate]
            eos_ctx_indices = [
                eos_ctx_indices[index - n_intermediate]
                for index in kept_indices
                if index >= n_intermediate
            ]
            all_slots = self.allocate_slots_batch(all_bids)
            n_intermediate = len(src_offsets)

        # Copy intermediate states from output buffers to cache
        self._copy_intermediate_to_cache(src_offsets, all_slots[:n_intermediate])

        # Copy EOS states from live buffers to cache
        self.store_from_live_batch(all_slots[n_intermediate:], eos_ctx_indices)

        # Register hashes for all committed blocks
        self.register_block_hashes_batch(all_bids, all_hashes)

        self.context.mamba_metadata.clear_intermediate_state(self.context)

    def _collect_commit_data(self):
        """Pull the metadata's commit records and the KV-side block hashes.

        Returns:
            Tuple of (intermediate_bids, src_offsets, eos_bids, eos_ctx_indices,
            all_hashes) or None if nothing to commit. all_hashes covers
            intermediate_bids + eos_bids in that order.
        """
        ctx = self.context
        metadata = ctx.mamba_metadata
        intermediate_bids, src_offsets, eos_bids, eos_ctx_indices = metadata.collect_commit_records(
            ctx
        )
        if not intermediate_bids and not eos_bids:
            metadata.clear_intermediate_state(ctx)
            return None

        # Single batch hash fetch for all block IDs (1 GPU sync). pc_state is always
        # present: this allocator exists only with prefix caching.
        all_bids_for_hash = intermediate_bids + eos_bids
        pc_state = ctx.kv_block_allocator.pc_state
        device = pc_state.block_hashes.device
        bid_tensor = torch.tensor(all_bids_for_hash, dtype=torch.int64, device=device)
        all_hashes = pc_state.block_hashes[bid_tensor].tolist()

        return intermediate_bids, src_offsets, eos_bids, eos_ctx_indices, all_hashes

    def _copy_intermediate_to_cache(self, src_offsets: list, slots: list) -> None:
        """Copy intermediate states from metadata's GPU buffers to cache slots.

        Uses fancy-indexed GPU D2D copy (2 kernel launches instead of 2N).

        Args:
            src_offsets: Source indices into metadata's intermediate buffers.
            slots: Destination cache slot indices.
        """
        if not src_offsets:
            return
        metadata = self.context.mamba_metadata
        device = self.ssm_states.device
        src_idx = torch.tensor(src_offsets, dtype=torch.int64, device=device)
        dst_idx = torch.tensor(slots, dtype=torch.int64, device=device)
        self.ssm_states[:, dst_idx] = metadata.intermediate_ssm_out[:, src_idx]
        self.conv_states[:, dst_idx] = metadata.intermediate_conv_out[:, src_idx]

    # =========================================================================
    # Reset
    # =========================================================================

    def reset(self) -> None:
        """Reset slot pool, cache state, the Mamba host registry, and the metadata's buffers."""
        self.block_to_slot.fill_(-1)
        self.slot_to_block.fill_(-1)
        torch.arange(self.max_slots, out=self.free_slots)
        self.free_count = self.max_slots
        self.registry.clear_mamba()
        # Intermediate buffers + CPU bookkeeping live on the metadata.
        self.context.mamba_metadata.reset_intermediate_state()
