# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import TYPE_CHECKING, Dict

import torch
from torch import Tensor

from megatron.core.inference.config import PrefixCachingEvictionPolicy

if TYPE_CHECKING:
    from .dynamic_context import DynamicInferenceContext

# Maximum intermediate state extraction offsets per request. The 3 candidates
# are: KV divergence boundary, last block-aligned boundary, and penultimate
# block boundary (see compute_and_store_offsets for details).
MAX_INTERMEDIATE_OFFSETS_PER_REQUEST = 3


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
            (num_mamba_layers, max_slots) + ssm_states_shape, dtype=ssm_states_dtype, device=device
        )

        # Hash-to-block mapping: only blocks with cached Mamba state
        self.hash_to_block_id: Dict[int, int] = {}

        # Per-request intermediate state storage (GPU tensors, fixed-size per request)
        # 0 = no offset, -1 = no block
        k = MAX_INTERMEDIATE_OFFSETS_PER_REQUEST
        self._intermediate_offsets_gpu = torch.zeros(
            (context.max_requests, k), dtype=torch.int32, device=device
        )
        self._intermediate_block_ids_gpu = torch.full(
            (context.max_requests, k), -1, dtype=torch.int32, device=device
        )
        self._intermediate_counts_gpu = torch.zeros(
            context.max_requests, dtype=torch.int32, device=device
        )
        self._eos_cache_block_id_gpu = torch.full(
            (context.max_requests,), -1, dtype=torch.int32, device=device
        )
        # CPU flag to skip GPU sync when no intermediates exist
        self._has_intermediates = False

        # Pre-allocated output buffers for CUDA graph compatible extraction
        self.max_intermediate_count = MAX_INTERMEDIATE_OFFSETS_PER_REQUEST * context.max_requests
        self.intermediate_ssm_out = torch.zeros(
            (num_mamba_layers, self.max_intermediate_count) + ssm_states_shape,
            dtype=ssm_states_dtype,
            device=device,
        )
        self.intermediate_conv_out = torch.zeros(
            (num_mamba_layers, self.max_intermediate_count) + conv_states_shape,
            dtype=conv_states_dtype,
            device=device,
        )

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

    def _invalidate_blocks_batch(self, block_ids_list: list) -> None:
        """Free cache slots and clear mappings for multiple blocks at once.

        Vectorized version of invalidate_block that avoids per-block .item()
        GPU syncs. Used by on_kv_blocks_deregistered for bulk eviction.

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
                self._invalidate_blocks_batch(block_ids_list)

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
        count = len(offsets)

        # Vectorized block ID lookup: GPU gather avoids per-block .item() syncs
        if count > 0:
            device = self._intermediate_offsets_gpu.device
            abs_tokens = torch.tensor(
                [skip_tokens + o for o in offsets], dtype=torch.int64, device=device
            )
            block_indices = abs_tokens // ctx.block_size_tokens - 1
            bids = ctx.request_to_kv_block_ids[current_id][block_indices]

            self._intermediate_offsets_gpu[current_id, :count] = torch.tensor(
                offsets, dtype=torch.int32, device=device
            )
            self._intermediate_block_ids_gpu[current_id, :count] = bids.to(torch.int32)
            self._has_intermediates = True
        self._intermediate_counts_gpu[current_id] = count

        # Block-aligned EOS: prompt_len is exactly block-aligned
        if last_aligned_abs == prompt_len and prompt_len > 0:
            last_block_idx = prompt_len // ctx.block_size_tokens - 1
            if last_block_idx >= 0:
                self._eos_cache_block_id_gpu[current_id] = ctx.request_to_kv_block_ids[current_id][
                    last_block_idx
                ]
                self._has_intermediates = True
            else:
                self._eos_cache_block_id_gpu[current_id] = -1
        else:
            self._eos_cache_block_id_gpu[current_id] = -1

    def get_intermediate_gpu_data(self):
        """Get intermediate offsets and counts as GPU tensor slices for current prefill batch.

        Returns:
            Tuple of (offsets_gpu, counts_gpu) where:
                offsets_gpu: [prefill_count, 3] int32 GPU tensor
                counts_gpu: [prefill_count] int32 GPU tensor
            Returns (None, None) if no prefill requests or no intermediates.
        """
        if not self._has_intermediates:
            return None, None

        ctx = self.context
        prefill_count = ctx.batch_dimensions.prefill_req_count
        if prefill_count == 0:
            return None, None

        active_start = ctx.paused_request_count
        decode_count = ctx.batch_dimensions.decode_req_count
        prefill_start = active_start + decode_count

        offsets = self._intermediate_offsets_gpu[prefill_start : prefill_start + prefill_count]
        counts = self._intermediate_counts_gpu[prefill_start : prefill_start + prefill_count]
        return offsets, counts

    def commit_intermediate_states(self) -> None:
        """Commit intermediate states from pre-allocated output buffers to cache.

        Called after the forward pass (including CUDA graph replay) completes.
        Reads SSM states from intermediate_ssm_out and conv states from
        intermediate_conv_out, which were written by GPU ops inside the graph.

        For each prefill request:
        - Intermediate states at extraction offsets: allocate cache slot,
          copy from output buffers, register hash.
        - Block-aligned EOS: copy final state from live buffer to cache slot.
        """
        ctx = self.context
        metadata = ctx.mamba_metadata
        prefill_count = ctx.batch_dimensions.prefill_req_count
        if prefill_count == 0:
            self._clear_intermediate_state()
            return

        active_start = ctx.paused_request_count
        decode_count = ctx.batch_dimensions.decode_req_count
        prefill_start = active_start + decode_count

        # Batch-transfer block IDs and EOS block IDs from GPU in bulk
        intermediate_count = metadata.intermediate_count
        per_request_counts = metadata.per_request_intermediate_counts

        all_block_ids_cpu = self._intermediate_block_ids_gpu[
            prefill_start : prefill_start + prefill_count
        ].tolist()
        eos_bids_cpu = self._eos_cache_block_id_gpu[
            prefill_start : prefill_start + prefill_count
        ].tolist()

        # Collect all block IDs that need hash lookups (intermediate + EOS)
        all_bids_for_hash = []
        if intermediate_count > 0:
            for req_idx, count in enumerate(per_request_counts):
                for j in range(count):
                    all_bids_for_hash.append(all_block_ids_cpu[req_idx][j])
        intermediate_hash_count = len(all_bids_for_hash)
        for eos_bid in eos_bids_cpu:
            if eos_bid >= 0:
                all_bids_for_hash.append(eos_bid)

        # Single batch hash fetch for all block IDs
        if all_bids_for_hash:
            device = ctx.kv_block_allocator.block_hashes.device
            bid_tensor = torch.tensor(all_bids_for_hash, dtype=torch.int64, device=device)
            all_hashes = ctx.kv_block_allocator.block_hashes[bid_tensor].tolist()
        else:
            all_hashes = []

        # Commit intermediate states from output buffers
        if intermediate_count > 0:
            ssm_offset = 0
            hash_offset = 0
            for req_idx, count in enumerate(per_request_counts):
                if count > 0:
                    for j in range(count):
                        bid = all_block_ids_cpu[req_idx][j]
                        slot = self.allocate_slot(bid)

                        self.ssm_states[:, slot].copy_(self.intermediate_ssm_out[:, ssm_offset + j])
                        self.conv_states[:, slot].copy_(
                            self.intermediate_conv_out[:, ssm_offset + j]
                        )

                        block_hash = all_hashes[hash_offset + j]
                        if block_hash > 0:
                            self.register_block_hash(bid, block_hash)
                    hash_offset += count
                ssm_offset += count

        # Handle block-aligned EOS: copy final state from live buffer
        eos_hash_idx = intermediate_hash_count
        for req_batch_idx in range(prefill_count):
            ctx_idx = prefill_start + req_batch_idx
            eos_bid = eos_bids_cpu[req_batch_idx]
            if eos_bid >= 0:
                slot = self.allocate_slot(eos_bid)
                self.store_from_live(eos_bid, ctx_idx)
                block_hash = all_hashes[eos_hash_idx]
                eos_hash_idx += 1
                if block_hash > 0:
                    self.register_block_hash(eos_bid, block_hash)

        self._clear_intermediate_state()

    def _clear_intermediate_state(self) -> None:
        """Clear all per-request intermediate state tracking."""
        ctx = self.context
        prefill_count = ctx.batch_dimensions.prefill_req_count
        if prefill_count > 0:
            active_start = ctx.paused_request_count
            decode_count = ctx.batch_dimensions.decode_req_count
            prefill_start = active_start + decode_count
            end = prefill_start + prefill_count
            self._intermediate_counts_gpu[prefill_start:end].fill_(0)
            self._intermediate_offsets_gpu[prefill_start:end].fill_(0)
            self._intermediate_block_ids_gpu[prefill_start:end].fill_(-1)
            self._eos_cache_block_id_gpu[prefill_start:end].fill_(-1)
        self._has_intermediates = False

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
        self.intermediate_ssm_out.zero_()
        self.intermediate_conv_out.zero_()
        self._intermediate_offsets_gpu.fill_(0)
        self._intermediate_block_ids_gpu.fill_(-1)
        self._intermediate_counts_gpu.fill_(0)
        self._eos_cache_block_id_gpu.fill_(-1)
        self._has_intermediates = False
