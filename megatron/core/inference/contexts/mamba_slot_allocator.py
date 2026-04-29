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

        gpu_device = torch.cuda.current_device()
        num_blocks = context.kv_block_allocator.total_count

        # Block <-> slot mappings (CPU for bookkeeping).
        self.block_to_slot = torch.full((num_blocks,), -1, dtype=torch.int32, device='cpu')
        self.slot_to_block = torch.full((max_slots,), -1, dtype=torch.int32, device='cpu')

        # Free slot pool (stack, CPU).
        self.free_slots = torch.arange(max_slots, dtype=torch.int32, device='cpu')
        self.free_count = max_slots

        # State tensors (GPU - accessed by Mamba CUDA kernels).
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

        # Hash-to-block mapping: only blocks with cached Mamba state
        self.hash_to_block_id: Dict[int, int] = {}

        # Per-request intermediate state storage.
        # offsets_cpu and counts_cpu: CPU source of truth.  GPU copies are
        # populated by transfer_bookkeeping_to_gpu() since Triton kernels read them.
        # block_ids and eos_cache_block_id: CPU only (consumed by CPU code).
        k = MAX_INTERMEDIATE_OFFSETS_PER_REQUEST
        self._intermediate_offsets_cpu = torch.zeros(
            (context.max_requests, k), dtype=torch.int32, device='cpu'
        )
        self._intermediate_counts_cpu = torch.zeros(
            context.max_requests, dtype=torch.int32, device='cpu'
        )
        self._intermediate_offsets_gpu = torch.zeros(
            (context.max_requests, k), dtype=torch.int32, device=gpu_device
        )
        self._intermediate_counts_gpu = torch.zeros(
            context.max_requests, dtype=torch.int32, device=gpu_device
        )
        # CPU-only: consumed by _collect_commit_data() which needs .tolist() anyway.
        self._intermediate_block_ids_cpu = torch.full(
            (context.max_requests, k), -1, dtype=torch.int32, device='cpu'
        )
        self._eos_cache_block_id_cpu = torch.full(
            (context.max_requests,), -1, dtype=torch.int32, device='cpu'
        )
        # CPU flag to skip GPU sync when no intermediates exist
        self._has_intermediates = False

        # Pre-allocated output buffers for CUDA graph compatible extraction (GPU).
        self.max_intermediate_count = MAX_INTERMEDIATE_OFFSETS_PER_REQUEST * context.max_requests
        self.intermediate_ssm_out = torch.zeros(
            (num_mamba_layers, self.max_intermediate_count) + ssm_states_shape,
            dtype=ssm_states_dtype,
            device=gpu_device,
        )
        self.intermediate_conv_out = torch.zeros(
            (num_mamba_layers, self.max_intermediate_count) + conv_states_shape,
            dtype=conv_states_dtype,
            device=gpu_device,
        )

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

        # Phase 3: Get slots from free pool, evicting if necessary
        from_free = min(num_new, self.free_count)
        new_slots = []
        if from_free > 0:
            start = self.free_count - from_free
            new_slots = self.free_slots[start : self.free_count].tolist()  # 1 GPU sync
            self.free_count = start

        need_evict = num_new - from_free
        if need_evict > 0:
            new_slots.extend(self._evict_lru_slots_batch(need_evict))

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

    def _evict_lru_slots_batch(self, num_needed: int) -> list:
        """Evict the least recently used Mamba cache slots.

        Does NOT return slots to the free pool — caller takes ownership.

        Args:
            num_needed: Number of slots to evict.

        Returns:
            List of freed slot indices.
        """
        kv_alloc = self.context.kv_block_allocator
        # Find blocks that have mamba slots and ref_count == 0
        has_slot_mask = self.block_to_slot[: kv_alloc.total_count] >= 0
        ref_zero_mask = kv_alloc.block_ref_counts[: kv_alloc.total_count] == 0
        candidates = has_slot_mask & ref_zero_mask
        candidate_ids = torch.nonzero(candidates, as_tuple=True)[0]

        if candidate_ids.numel() < num_needed:
            raise RuntimeError("No evictable Mamba cache slots available")

        # Pick oldest blocks by timestamp (LRU) or first N (REF_ZERO)
        if self.context.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
            timestamps = kv_alloc.block_timestamps[candidate_ids]
            sorted_indices = torch.argsort(timestamps)[:num_needed]
            evict_ids = candidate_ids[sorted_indices]
        else:
            evict_ids = candidate_ids[:num_needed]

        # Batch gather slots + hashes (2 GPU syncs)
        slots = self.block_to_slot[evict_ids].tolist()
        hashes = kv_alloc.block_hashes[evict_ids].tolist()

        # Batch cleanup GPU mappings
        self.block_to_slot[evict_ids] = -1
        slot_tensor = torch.tensor(slots, dtype=torch.int64, device=self.block_to_slot.device)
        self.slot_to_block[slot_tensor] = -1

        # Clean up hash dict (CPU loop)
        for h in hashes:
            if h > 0 and h in self.hash_to_block_id:
                del self.hash_to_block_id[h]

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
        updates = {h: bid for bid, h in zip(block_ids, hashes) if h > 0}
        if updates:
            self.hash_to_block_id.update(updates)

    # =========================================================================
    # Intermediate state tracking
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

        # CPU bookkeeping writes (no GPU kernel launches).
        if count > 0:
            abs_tokens_cpu = torch.tensor([skip_tokens + o for o in offsets], dtype=torch.int64)
            block_indices_cpu = abs_tokens_cpu // ctx.block_size_tokens - 1
            bids_cpu = ctx.request_to_kv_block_ids[current_id][block_indices_cpu]

            self._intermediate_offsets_cpu[current_id, :count] = torch.tensor(
                offsets, dtype=torch.int32
            )
            self._intermediate_block_ids_cpu[current_id, :count] = bids_cpu.to(torch.int32)
            self._has_intermediates = True
        self._intermediate_counts_cpu[current_id] = count

        # Block-aligned EOS: prompt_len is exactly block-aligned
        if last_aligned_abs == prompt_len and prompt_len > 0:
            last_block_idx = prompt_len // ctx.block_size_tokens - 1
            if last_block_idx >= 0:
                self._eos_cache_block_id_cpu[current_id] = ctx.request_to_kv_block_ids[current_id][
                    last_block_idx
                ]
                self._has_intermediates = True
            else:
                self._eos_cache_block_id_cpu[current_id] = -1
        else:
            self._eos_cache_block_id_cpu[current_id] = -1

    def get_intermediate_cpu_data(self):
        """Get intermediate offsets and counts as CPU tensor slices for current prefill batch.

        Returns:
            Tuple of (offsets_cpu, counts_cpu) where:
                offsets_cpu: [prefill_count, 3] int32 CPU tensor
                counts_cpu: [prefill_count] int32 CPU tensor
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

        offsets = self._intermediate_offsets_cpu[prefill_start : prefill_start + prefill_count]
        counts = self._intermediate_counts_cpu[prefill_start : prefill_start + prefill_count]
        return offsets, counts

    def transfer_intermediate_to_gpu(self, prefill_start: int, prefill_count: int):
        """Copy intermediate offsets/counts slice from CPU to GPU for Mamba kernels.

        Returns the GPU tensor views for the forward-pass kernels to consume.
        """
        if prefill_count == 0:
            return None, None
        offsets_cpu = self._intermediate_offsets_cpu[prefill_start : prefill_start + prefill_count]
        counts_cpu = self._intermediate_counts_cpu[prefill_start : prefill_start + prefill_count]
        offsets_gpu = self._intermediate_offsets_gpu[prefill_start : prefill_start + prefill_count]
        counts_gpu = self._intermediate_counts_gpu[prefill_start : prefill_start + prefill_count]
        offsets_gpu.copy_(offsets_cpu, non_blocking=True)
        counts_gpu.copy_(counts_cpu, non_blocking=True)
        return offsets_gpu, counts_gpu

    # =========================================================================
    # Intermediate state commit
    # =========================================================================

    def commit_intermediate_states(self) -> None:
        """Commit intermediate states from pre-allocated output buffers to cache.

        Called after the forward pass (including CUDA graph replay) completes.
        Batched pipeline: collect data, allocate slots, copy states, register hashes.
        """
        collected = self._collect_commit_data()
        if collected is None:
            return
        intermediate_bids, src_offsets, eos_bids, eos_ctx_indices, all_hashes = collected

        # Allocate all slots in one batch (intermediates + EOS)
        all_bids = intermediate_bids + eos_bids
        all_slots = self.allocate_slots_batch(all_bids)

        # Copy intermediate states from output buffers to cache
        n_intermediate = len(intermediate_bids)
        self._copy_intermediate_to_cache(src_offsets, all_slots[:n_intermediate])

        # Copy EOS states from live buffers to cache
        self.store_from_live_batch(all_slots[n_intermediate:], eos_ctx_indices)

        # Register hashes for all committed blocks
        self.register_block_hashes_batch(all_bids, all_hashes)

        self._clear_intermediate_state()

    def _collect_commit_data(self):
        """Extract commit data from GPU intermediate state tracking.

        Returns:
            Tuple of (intermediate_bids, src_offsets, eos_bids, eos_ctx_indices,
            all_hashes) or None if nothing to commit. all_hashes covers
            intermediate_bids + eos_bids in that order.
        """
        ctx = self.context
        metadata = ctx.mamba_metadata
        prefill_count = ctx.batch_dimensions.prefill_req_count
        if prefill_count == 0:
            self._clear_intermediate_state()
            return None

        active_start = ctx.paused_request_count
        decode_count = ctx.batch_dimensions.decode_req_count
        prefill_start = active_start + decode_count

        # Block IDs and EOS block IDs live on CPU (no GPU sync needed).
        intermediate_count = metadata.intermediate_count
        per_request_counts = metadata.per_request_intermediate_counts

        all_block_ids_cpu = self._intermediate_block_ids_cpu[
            prefill_start : prefill_start + prefill_count
        ].tolist()
        eos_bids_cpu = self._eos_cache_block_id_cpu[
            prefill_start : prefill_start + prefill_count
        ].tolist()

        # Flatten intermediate block IDs and source offsets
        intermediate_bids = []
        src_offsets = []
        if intermediate_count > 0:
            ssm_offset = 0
            for req_idx, count in enumerate(per_request_counts):
                for j in range(count):
                    intermediate_bids.append(all_block_ids_cpu[req_idx][j])
                    src_offsets.append(ssm_offset + j)
                ssm_offset += count

        # Collect EOS block IDs and their context indices
        eos_bids = []
        eos_ctx_indices = []
        for req_batch_idx in range(prefill_count):
            eos_bid = eos_bids_cpu[req_batch_idx]
            if eos_bid >= 0:
                eos_bids.append(eos_bid)
                eos_ctx_indices.append(prefill_start + req_batch_idx)

        if not intermediate_bids and not eos_bids:
            self._clear_intermediate_state()
            return None

        # Single batch hash fetch for all block IDs (1 GPU sync)
        all_bids_for_hash = intermediate_bids + eos_bids
        device = ctx.kv_block_allocator.block_hashes.device
        bid_tensor = torch.tensor(all_bids_for_hash, dtype=torch.int64, device=device)
        all_hashes = ctx.kv_block_allocator.block_hashes[bid_tensor].tolist()

        return intermediate_bids, src_offsets, eos_bids, eos_ctx_indices, all_hashes

    def _copy_intermediate_to_cache(self, src_offsets: list, slots: list) -> None:
        """Copy intermediate states from output buffers to cache slots.

        Uses fancy-indexed GPU D2D copy (2 kernel launches instead of 2N).

        Args:
            src_offsets: Source indices into intermediate_ssm_out/intermediate_conv_out.
            slots: Destination cache slot indices.
        """
        if not src_offsets:
            return
        device = self.ssm_states.device
        src_idx = torch.tensor(src_offsets, dtype=torch.int64, device=device)
        dst_idx = torch.tensor(slots, dtype=torch.int64, device=device)
        self.ssm_states[:, dst_idx] = self.intermediate_ssm_out[:, src_idx]
        self.conv_states[:, dst_idx] = self.intermediate_conv_out[:, src_idx]

    def _clear_intermediate_state(self) -> None:
        """Clear all per-request intermediate state tracking."""
        ctx = self.context
        prefill_count = ctx.batch_dimensions.prefill_req_count
        if prefill_count > 0:
            active_start = ctx.paused_request_count
            decode_count = ctx.batch_dimensions.decode_req_count
            prefill_start = active_start + decode_count
            end = prefill_start + prefill_count
            self._intermediate_counts_cpu[prefill_start:end].fill_(0)
            self._intermediate_offsets_cpu[prefill_start:end].fill_(0)
            self._intermediate_block_ids_cpu[prefill_start:end].fill_(-1)
            self._eos_cache_block_id_cpu[prefill_start:end].fill_(-1)
        self._has_intermediates = False

    # =========================================================================
    # Reset
    # =========================================================================

    def reset(self) -> None:
        """Reset all state (mappings, free pool, cache, intermediate tracking)."""
        self.block_to_slot.fill_(-1)
        self.slot_to_block.fill_(-1)
        self.free_slots = torch.arange(self.max_slots, dtype=torch.int32, device='cpu')
        self.free_count = self.max_slots
        self.hash_to_block_id.clear()
        self.intermediate_ssm_out.zero_()
        self.intermediate_conv_out.zero_()
        self._intermediate_offsets_cpu.fill_(0)
        self._intermediate_counts_cpu.fill_(0)
        self._intermediate_block_ids_cpu.fill_(-1)
        self._eos_cache_block_id_cpu.fill_(-1)
        self._has_intermediates = False
