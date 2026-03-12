# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from torch import Tensor

from megatron.core.inference.config import PrefixCachingEvictionPolicy

if TYPE_CHECKING:
    from .dynamic_context import DynamicInferenceContext


class MambaSlotAllocator:
    """Manages Mamba state caching for prefix caching in hybrid models.

    Owns the Mamba cache slot pool, block-to-slot mappings, hash-to-block
    mapping, and intermediate state tracking. Accesses KV allocator state
    (ref counts, timestamps, block hashes) via the parent context.

    Args:
        context: The DynamicInferenceContext that owns this allocator.
        max_slots: Maximum number of cache slots.
        num_mamba_layers: Number of Mamba layers in the model.
        conv_states_shape: Shape of per-slot conv state (excluding layer/slot dims).
        ssm_states_shape: Shape of per-slot SSM state (excluding layer/slot dims).
        conv_states_dtype: Dtype for conv state tensors.
        ssm_states_dtype: Dtype for SSM state tensors.
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
    ):
        self.context = context
        self.max_slots = max_slots
        self.num_mamba_layers = num_mamba_layers

        device = torch.cuda.current_device()
        num_blocks = context.kv_block_allocator.total_count

        # Block <-> slot mappings
        self.block_to_slot = torch.full((num_blocks,), -1, dtype=torch.int32, device=device)
        self.slot_to_block = torch.full((max_slots,), -1, dtype=torch.int32, device=device)

        # Free slot pool (stack)
        self.free_slots = torch.arange(max_slots, dtype=torch.int32, device=device)
        self.free_count = max_slots

        # State tensors
        self.conv_states = torch.zeros(
            (num_mamba_layers, max_slots) + conv_states_shape,
            dtype=conv_states_dtype,
            device=device,
        )
        self.ssm_states = torch.zeros(
            (num_mamba_layers, max_slots) + ssm_states_shape,
            dtype=ssm_states_dtype,
            device=device,
        )

        # Hash-to-block mapping: only blocks with cached Mamba state
        self.hash_to_block_id: Dict[int, int] = {}

        # Per-request intermediate state storage
        self._intermediate_offsets: list = [None] * context.max_requests
        self._intermediate_block_ids: list = [None] * context.max_requests
        self._eos_cache_block_id: list = [None] * context.max_requests
        self._intermediate_buffer: dict = {}

    # =========================================================================
    # Slot management
    # =========================================================================

    def allocate_slot(self, block_id: int) -> int:
        """Get a free Mamba cache slot for a block, evicting if necessary.

        Args:
            block_id: The KV block ID to associate with this slot.

        Returns:
            The allocated slot index.
        """
        # Check if block already has a slot
        existing = self.block_to_slot[block_id].item()
        if existing >= 0:
            return existing

        # Try free pool
        if self.free_count > 0:
            self.free_count -= 1
            slot = self.free_slots[self.free_count].item()
        else:
            slot = self._evict_lru_slot()

        self.block_to_slot[block_id] = slot
        self.slot_to_block[slot] = block_id
        return slot

    def _evict_lru_slot(self) -> int:
        """Evict the least recently used Mamba cache slot.

        Returns:
            The freed slot index.
        """
        kv_alloc = self.context.kv_block_allocator
        # Find blocks that have mamba slots and ref_count == 0
        has_slot_mask = self.block_to_slot[: kv_alloc.total_count] >= 0
        ref_zero_mask = kv_alloc.block_ref_counts[: kv_alloc.total_count] == 0
        candidates = has_slot_mask & ref_zero_mask
        candidate_ids = torch.nonzero(candidates, as_tuple=True)[0]

        if candidate_ids.numel() == 0:
            raise RuntimeError("No evictable Mamba cache slots available")

        # Pick block with oldest timestamp if LRU, otherwise just pick first
        if self.context.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
            timestamps = kv_alloc.block_timestamps[candidate_ids]
            evict_idx = candidate_ids[torch.argmin(timestamps)].item()
        else:
            evict_idx = candidate_ids[0].item()

        slot = self.block_to_slot[evict_idx].item()
        block_hash = kv_alloc.block_hashes[evict_idx].item()

        # Clean up mappings
        self.block_to_slot[evict_idx] = -1
        self.slot_to_block[slot] = -1
        if block_hash > 0 and block_hash in self.hash_to_block_id:
            del self.hash_to_block_id[block_hash]

        return slot

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

    def invalidate_block(self, block_id: int) -> None:
        """Free cache slot and clear mappings for a block.

        Called when KV blocks are evicted/deregistered.

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

    def store_from_live(self, block_id: int, request_idx: int) -> None:
        """Copy all layers from live per-request buffer to cache slot.

        Used for block-aligned EOS case where the final kernel state
        is in the live buffer.

        Args:
            block_id: The KV block ID.
            request_idx: The context request index.
        """
        slot = self.block_to_slot[block_id].item()
        assert slot >= 0, f"Block {block_id} has no Mamba cache slot"
        mamba_idx = self.context.mamba_metadata.request_to_mamba_state_idx[request_idx].item()
        self.conv_states[:, slot].copy_(self.context.mamba_conv_states[:, mamba_idx])
        self.ssm_states[:, slot].copy_(self.context.mamba_ssm_states[:, mamba_idx])

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

    def register_block_hash(self, block_id: int, block_hash: int) -> None:
        """Register a block as having cached Mamba state.

        Args:
            block_id: The block ID.
            block_hash: The block's hash value.
        """
        self.hash_to_block_id[block_hash] = block_id

    # =========================================================================
    # Deregistration callback
    # =========================================================================

    def on_kv_blocks_deregistered(self, block_ids_list: list, hashes_to_delete: set) -> None:
        """Handle KV block deregistration by cleaning up Mamba state.

        Called by KVBlockAllocator._deregister_blocks via callback.

        Args:
            block_ids_list: List of deregistered block IDs.
            hashes_to_delete: Set of hashes being deregistered (excludes -1).
        """
        if self.hash_to_block_id:
            mamba_keys = hashes_to_delete & self.hash_to_block_id.keys()
            if mamba_keys:
                from collections import deque

                deque(map(self.hash_to_block_id.pop, mamba_keys), maxlen=0)
                for bid in block_ids_list:
                    self.invalidate_block(bid)

    # =========================================================================
    # Intermediate offset tracking
    # =========================================================================

    def compute_and_store_offsets(
        self,
        req,
        current_id: int,
        skip_tokens: int,
        prefill_chunk_length: int,
        num_matched_blocks: int,
        matched_block_ids: list,
        overall_required_blocks: int,
    ) -> None:
        """Compute intermediate state extraction offsets and store per-request.

        Args:
            req: The inference request.
            current_id: Context request index.
            skip_tokens: Number of tokens being skipped (mamba match).
            prefill_chunk_length: Total prefill chunk length before skipping.
            num_matched_blocks: Number of KV-matched blocks.
            matched_block_ids: List of matched KV block IDs.
            overall_required_blocks: Total blocks needed for this request.
        """
        ctx = self.context
        prompt_len = len(req.prompt_tokens)
        num_kv_matched = num_matched_blocks
        kv_div_abs = num_kv_matched * ctx.block_size_tokens
        last_aligned_abs = (prompt_len // ctx.block_size_tokens) * ctx.block_size_tokens
        seq_len = prefill_chunk_length - skip_tokens  # effective prefill length

        # Compute relative offsets (relative to prefill start after skip)
        kv_div_rel = kv_div_abs - skip_tokens
        last_aligned_rel = last_aligned_abs - skip_tokens
        penultimate_abs = (overall_required_blocks - 1) * ctx.block_size_tokens
        penultimate_rel = penultimate_abs - skip_tokens

        # Determine mamba_chunk_size from mamba config (128 is the standard SSM kernel chunk size)
        mamba_chunk_size = 128

        # Build offset list: include if > 0, < seq_len, and % mamba_chunk_size == 0
        offsets_set = set()
        for offset in [kv_div_rel, last_aligned_rel, penultimate_rel]:
            if offset > 0 and offset < seq_len and offset % mamba_chunk_size == 0:
                offsets_set.add(offset)

        offsets = sorted(offsets_set)

        # Map each offset back to block index and block ID
        block_ids_for_offsets = []
        for offset in offsets:
            abs_token = skip_tokens + offset
            block_idx = abs_token // ctx.block_size_tokens - 1
            bid = ctx.request_to_kv_block_ids[current_id][block_idx].item()
            block_ids_for_offsets.append(bid)

        self._intermediate_offsets[current_id] = offsets if offsets else None
        self._intermediate_block_ids[current_id] = (
            block_ids_for_offsets if block_ids_for_offsets else None
        )

        # Block-aligned EOS: prompt_len is exactly block-aligned
        if last_aligned_abs == prompt_len and prompt_len > 0:
            last_block_idx = prompt_len // ctx.block_size_tokens - 1
            if last_block_idx >= 0:
                eos_bid = ctx.request_to_kv_block_ids[current_id][last_block_idx].item()
                self._eos_cache_block_id[current_id] = eos_bid
            else:
                self._eos_cache_block_id[current_id] = None
        else:
            self._eos_cache_block_id[current_id] = None

    def get_intermediate_offsets(self) -> Optional[List[List[int]]]:
        """Get intermediate token offsets for all prefill requests in the current batch.

        Returns:
            List of offset lists (one per prefill request), or None if no
            request has intermediate offsets.
        """
        ctx = self.context
        prefill_count = ctx.batch_dimensions.prefill_req_count
        if prefill_count == 0:
            return None

        # Prefill requests are the last `prefill_count` active requests
        active_start = ctx.paused_request_count
        decode_count = ctx.batch_dimensions.decode_req_count
        prefill_start = active_start + decode_count

        result = []
        has_any = False
        for i in range(prefill_start, prefill_start + prefill_count):
            offsets = self._intermediate_offsets[i]
            if offsets is not None:
                has_any = True
                result.append(offsets)
            else:
                result.append([])

        return result if has_any else None

    def buffer_intermediate_states(
        self, mamba_layer_idx: int, intermediate_states_per_request: list
    ) -> None:
        """Buffer intermediate states from a single Mamba layer's forward pass.

        Args:
            mamba_layer_idx: The Mamba layer index.
            intermediate_states_per_request: Per-request list of
                (ssm_states, conv_states) tuples or None.
        """
        self._intermediate_buffer[mamba_layer_idx] = intermediate_states_per_request

    def commit_intermediate_states(self) -> None:
        """Commit buffered intermediate states to the Mamba cache.

        Called after the forward pass completes. For each prefill request:
        - Intermediate states at kv_divergence/last_aligned: allocate cache slot,
          write state, register hash in hash_to_block_id.
        - Block-aligned EOS: copy final state from live buffer to cache slot.
        """
        ctx = self.context
        prefill_count = ctx.batch_dimensions.prefill_req_count
        if prefill_count == 0:
            self._clear_intermediate_state()
            return

        active_start = ctx.paused_request_count
        decode_count = ctx.batch_dimensions.decode_req_count
        prefill_start = active_start + decode_count
        has_buffer = bool(self._intermediate_buffer)

        for req_batch_idx in range(prefill_count):
            ctx_idx = prefill_start + req_batch_idx
            offsets = self._intermediate_offsets[ctx_idx]
            block_ids = self._intermediate_block_ids[ctx_idx]

            # Commit intermediate states from forward pass
            if offsets is not None and block_ids is not None and has_buffer:
                for offset_idx in range(len(offsets)):
                    bid = block_ids[offset_idx]
                    slot = self.allocate_slot(bid)

                    # Write states from each mamba layer
                    for layer_idx, states_list in self._intermediate_buffer.items():
                        if states_list[req_batch_idx] is not None:
                            ssm_states, conv_states = states_list[req_batch_idx]
                            self.ssm_states[layer_idx, slot].copy_(ssm_states[offset_idx])
                            self.conv_states[layer_idx, slot].copy_(conv_states[offset_idx])

                    # Register in mamba hash map
                    block_hash = ctx.kv_block_allocator.block_hashes[bid].item()
                    if block_hash > 0:
                        self.register_block_hash(bid, block_hash)

            # Handle block-aligned EOS: copy final state from live buffer
            eos_bid = self._eos_cache_block_id[ctx_idx]
            if eos_bid is not None:
                slot = self.allocate_slot(eos_bid)
                self.store_from_live(eos_bid, ctx_idx)
                block_hash = ctx.kv_block_allocator.block_hashes[eos_bid].item()
                if block_hash > 0:
                    self.register_block_hash(eos_bid, block_hash)

        self._clear_intermediate_state()

    def _clear_intermediate_state(self) -> None:
        """Clear all per-request intermediate state tracking."""
        self._intermediate_buffer.clear()
        ctx = self.context
        prefill_count = ctx.batch_dimensions.prefill_req_count
        if prefill_count > 0:
            active_start = ctx.paused_request_count
            decode_count = ctx.batch_dimensions.decode_req_count
            prefill_start = active_start + decode_count
            for i in range(prefill_start, prefill_start + prefill_count):
                self._intermediate_offsets[i] = None
                self._intermediate_block_ids[i] = None
                self._eos_cache_block_id[i] = None

    # =========================================================================
    # Reset
    # =========================================================================

    def reset(self) -> None:
        """Reset all state (mappings, free pool, cache, intermediate tracking)."""
        self.block_to_slot.fill_(-1)
        self.slot_to_block.fill_(-1)
        self.free_slots = torch.arange(
            self.max_slots, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.free_count = self.max_slots
        self.hash_to_block_id.clear()
        self._intermediate_buffer.clear()
        for i in range(self.context.max_requests):
            self._intermediate_offsets[i] = None
            self._intermediate_block_ids[i] = None
            self._eos_cache_block_id[i] = None
