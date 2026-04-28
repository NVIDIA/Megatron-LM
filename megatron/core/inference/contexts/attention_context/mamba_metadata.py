# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.contexts.mamba_slot_allocator import (
    MAX_INTERMEDIATE_OFFSETS_PER_REQUEST,
)


class MambaMetadata:
    """Manages the metadata tensors required for Mamba layers during inference.

    All per-step compute is GPU-only and uses fixed-shape ops keyed on the
    padded batch dimensions (with ``torch.where`` masks against per-step GPU
    scalars), so :meth:`update` is CUDA-graph capturable as-is.
    """

    def __init__(
        self,
        max_requests: int,
        max_tokens: int,
        mamba_chunk_size: int = 128,
        d_conv: int = 0,
        request_to_mamba_state_idx_buf: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the Mamba metadata buffers.

        Args:
            max_requests (int): The maximum number of concurrent requests.
            max_tokens (int): The maximum number of tokens.
            mamba_chunk_size (int): The chunk size used by the Mamba SSM Triton kernels.
            d_conv (int): Convolution window size (from mamba_conv_states_shape[-1]).
                Used for vectorized conv state extraction at intermediate offsets.
            request_to_mamba_state_idx_buf (Optional[Tensor]): Pre-allocated CPU
                ``(max_requests,)`` int32 buffer for the request-to-slot mapping,
                aliased into the context's coalesced pinned bookkeeping buffer
                so writes here ride the per-step H2D into
                ``gpu_view.request_to_mamba_state_idx``. When ``None`` (e.g. unit
                tests constructing :class:`MambaMetadata` standalone) a fresh
                CPU tensor is allocated.
        """
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.mamba_chunk_size = mamba_chunk_size
        self.d_conv = d_conv
        self.device = torch.cuda.current_device()

        # Maximum possible chunks across all batch configurations
        self.max_chunks = max_tokens // mamba_chunk_size + max_requests

        # Map from requests to slots in the static Mamba state buffer.
        # Aliased into the context's coalesced pinned CPU bookkeeping buffer
        # when constructed with one (production); otherwise standalone.
        if request_to_mamba_state_idx_buf is not None:
            assert request_to_mamba_state_idx_buf.shape == (max_requests,)
            assert request_to_mamba_state_idx_buf.dtype == torch.int32
            self.request_to_mamba_state_idx = request_to_mamba_state_idx_buf
        else:
            self.request_to_mamba_state_idx = torch.full(
                (max_requests,), -1, dtype=torch.int32, device='cpu'
            )

        # Map from requests to slots in the static Mamba state buffer for active decode requests.
        # int64 so selective_state_update can index directly without a per-layer upcast kernel.
        self._batch_indices_decode_buffer = torch.full(
            (self.max_requests,), -1, dtype=torch.int64, device=self.device
        )

        # Map from requests to slots in the static Mamba state buffer for active prefill requests.
        self._batch_indices_prefill_buffer = torch.full(
            (self.max_requests,), -1, dtype=torch.int32, device=self.device
        )

        # Map from token id to request id for active prefill requests.
        self._seq_idx_buffer = torch.full(
            (1, self.max_tokens), -1, dtype=torch.int32, device=self.device
        )

        # Cumulative sequence lengths for active prefill requests.
        self._cu_seqlens_buffer = torch.zeros(
            (self.max_requests + 1,), dtype=torch.int32, device=self.device
        )

        # Tuple of (active decode token count, active prefill token count).
        self._device_decode_prefill_buffer = torch.zeros(
            (2,), dtype=torch.int32, device=self.device
        )

        # SSM chunk boundaries for the varlen kernel.
        self._cu_chunk_seqlens_buffer = torch.zeros(
            self.max_chunks + 1, dtype=torch.int32, device=self.device
        )

        # Index of the last chunk per sequence.
        self._last_chunk_indices_buffer = torch.zeros(
            max_requests, dtype=torch.int32, device=self.device
        )

        # Request ID per chunk.
        self._seq_idx_for_varlen_buffer = torch.zeros(
            self.max_chunks, dtype=torch.int32, device=self.device
        )

        # Conv1d per-token metadata (request ID and request start position).
        self._conv_seq_idx_buffer = torch.zeros(max_tokens, dtype=torch.int32, device=self.device)
        self._conv_seq_start_buffer = torch.zeros(max_tokens, dtype=torch.int32, device=self.device)

        # Allocator for Mamba state slots.
        # Pre-sized to `2 * max_requests` for shape-stability; this acts as a stack.
        self.mamba_state_free_slots = torch.empty(
            2 * self.max_requests, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.mamba_state_free_slots[: self.max_requests] = torch.arange(
            self.max_requests, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.mamba_state_free_slots[self.max_requests :] = 0
        self.mamba_state_free_slot_count = self.max_requests

        # GPU mirror of the stack pointer and static arange for the graphed release path.
        self._free_slot_count_gpu = torch.tensor(
            [self.mamba_state_free_slot_count], dtype=torch.int32, device=torch.cuda.current_device()
        )
        self._free_arange = torch.arange(
            self.max_requests, dtype=torch.int32, device=torch.cuda.current_device()
        )

        # Scratch buffer for cumulative chunk counts.
        self._cum_chunks_buffer = torch.zeros(
            self.max_requests + 1, dtype=torch.int64, device=self.device
        )

        # Pre-allocated index tensor for graphable chunk ops.
        self._arange_chunks = torch.arange(
            self.max_chunks, dtype=torch.int64, device=self.device
        )

        # Fields populated only by `PrefixCachedMambaMetadata`. Defined here so
        # callers (e.g. mamba_mixer) can read these uniformly regardless of
        # which subclass is in use.
        self.intermediate_chunk_indices: Optional[torch.Tensor] = None
        self.intermediate_abs_positions: Optional[torch.Tensor] = None
        self.conv_gather_offsets: Optional[torch.Tensor] = None

        self.reset_varlen_metadata()

    def reset(self) -> None:
        """Reset all Mamba state and free all allocated slots."""
        self.request_to_mamba_state_idx.fill_(-1)
        self.reset_varlen_metadata()

        # Re-initialize the free slot pool
        self.mamba_state_free_slots[: self.max_requests].copy_(
            torch.arange(self.max_requests, dtype=torch.int32, device=torch.cuda.current_device())
        )
        self.mamba_state_free_slot_count = self.max_requests
        self._free_slot_count_gpu.fill_(self.max_requests)

    def reset_varlen_metadata(self) -> None:
        """Reset varlen metadata."""
        self.batch_indices_decode = None
        self.batch_indices_prefill = None
        self.cu_seqlens = None
        self.cum_chunks = None
        self.seq_idx = None
        self.device_decode_prefill = self._device_decode_prefill_buffer

        # SSM/conv1d precomputed views.
        self.cu_chunk_seqlens = None
        self.last_chunk_indices = None
        self.seq_idx_for_varlen = None
        self.conv_seq_idx = None
        self.conv_seq_start = None

    def update(
        self,
        active_mamba_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        real_decode_count_gpu: torch.Tensor,
        real_prefill_count_gpu: torch.Tensor,
        arange_buf: torch.Tensor,
        padded_batch_dimensions: InferenceBatchDimensions,
        intermediate_offsets_gpu: Optional[torch.Tensor] = None,
        intermediate_counts_gpu: Optional[torch.Tensor] = None,
    ) -> None:
        """Updates the dedicated mapping tensors with the indices of currently active requests.

        All copy+pad ops use fixed-shape slices (keyed on padded dimensions) with GPU
        scalars and pre-allocated index tensors for the real/padding boundary, so
        the body is CUDA-graph capturable.

        Args:
            active_mamba_indices (Tensor): GPU mamba slot indices for active
                requests, sized at least ``[padded_decode + padded_prefill]``.
                Indexed up to ``padded_decode_count`` for decode and gathered
                with a GPU-scalar offset for prefill.
            cu_seqlens (Tensor): Cumulative sequence lengths across all active
                requests (the context's ``cu_active_request_query_lengths``).
                Gathered at ``real_decode_count_gpu`` to locate the prefill region.
            real_decode_count_gpu (Tensor): 1-element GPU scalar holding the
                real (unpadded) decode request count. Used as a mask boundary
                and as a GPU gather offset.
            real_prefill_count_gpu (Tensor): 1-element GPU scalar holding the
                real (unpadded) prefill request count. Used as a mask boundary
                for the padded prefill region.
            arange_buf (Tensor): Pre-allocated ``[0..N)`` index tensor, at
                least as long as the largest padded request count. Paired with
                the GPU scalars to produce padding masks without a
                data-dependent shape.
            padded_batch_dimensions (InferenceBatchDimensions): Padded
                decode / prefill / token counts (Python ints).
            intermediate_offsets_gpu (Optional[Tensor]): Slot-allocator buffer
                of intermediate-state offsets (full ``[max_requests, K]``).
                Forwarded to ``_update_intermediate_metadata``, which is a
                no-op on this base class. ``PrefixCachedMambaMetadata`` consumes it.
            intermediate_counts_gpu (Optional[Tensor]): Companion to
                ``intermediate_offsets_gpu`` (full ``[max_requests]``).
        """
        padded_decode_count = padded_batch_dimensions.decode_req_count
        padded_prefill_count = padded_batch_dimensions.prefill_req_count
        padded_token_count = padded_batch_dimensions.token_count

        if padded_decode_count > 0:
            # Decode indices: copy real, pad rest with -1.
            mask = arange_buf[:padded_decode_count] < real_decode_count_gpu
            self._batch_indices_decode_buffer[:padded_decode_count] = torch.where(
                mask, active_mamba_indices[:padded_decode_count], -1
            )
            self.batch_indices_decode = self._batch_indices_decode_buffer[:padded_decode_count]

        if padded_prefill_count > 0:
            # Prefill indices: source at [decode_count, ...); gather via GPU scalar offset.
            prefill_mask = arange_buf[:padded_prefill_count] < real_prefill_count_gpu
            gather_idx = (
                arange_buf[:padded_prefill_count].to(torch.int64) + real_decode_count_gpu
            )
            self._batch_indices_prefill_buffer[:padded_prefill_count] = torch.where(
                prefill_mask, active_mamba_indices[gather_idx], -1
            )
            self.batch_indices_prefill = self._batch_indices_prefill_buffer[:padded_prefill_count]

            # cu_seqlens: gather from source with GPU scalar offset, subtract base.
            cu_gather_idx = (
                arange_buf[:padded_prefill_count].to(torch.int64) + real_decode_count_gpu + 1
            )
            cu_base = cu_seqlens[real_decode_count_gpu]  # GPU gather, 0-d tensor
            self._cu_seqlens_buffer[1 : padded_prefill_count + 1] = (
                cu_seqlens[cu_gather_idx] - cu_base
            ).to(self._cu_seqlens_buffer.dtype)
            # Pad: positions beyond real_prefill get the last real cumulative value.
            last_val = self._cu_seqlens_buffer[real_prefill_count_gpu]  # GPU gather
            self._cu_seqlens_buffer[1 : padded_prefill_count + 1] = torch.where(
                prefill_mask, self._cu_seqlens_buffer[1 : padded_prefill_count + 1], last_val
            )
            self.cu_seqlens = self._cu_seqlens_buffer[: padded_prefill_count + 1]

            # seq_idx via repeat_interleave on padded cu_seqlens (zero-length
            # padding sequences contribute nothing). The trailing -1 + pad_tokens
            # ensures any unused tail of [padded_token_count] reads as -1.
            cu = self._cu_seqlens_buffer[: padded_prefill_count + 1]
            lengths = (cu[1:] - cu[:-1]).to(torch.int64)
            seq_indices = arange_buf[:padded_prefill_count]
            pad_tokens = (
                (padded_token_count - cu[padded_prefill_count]).to(torch.int64).unsqueeze(0)
            )
            self._seq_idx_buffer[0, :padded_token_count] = torch.repeat_interleave(
                torch.cat([seq_indices, seq_indices.new_full((1,), -1)]),
                torch.cat([lengths, pad_tokens]),
                output_size=padded_token_count,
            )
            self.seq_idx = self._seq_idx_buffer[:, :padded_token_count]

            # SSM and conv1d metadata.
            chunk_size = self.mamba_chunk_size
            padded_max_chunks = padded_token_count // chunk_size + padded_prefill_count

            # Per-sequence chunk counts and cumulative offsets.
            seq_lens = cu[1:] - cu[:-1]
            n_chunks = torch.clamp((seq_lens + chunk_size - 1) // chunk_size, min=1)

            torch.cumsum(
                n_chunks, dim=0, out=self._cum_chunks_buffer[1 : padded_prefill_count + 1]
            )
            self.cum_chunks = self._cum_chunks_buffer[: padded_prefill_count + 1]
            cum_chunks = self.cum_chunks

            self._last_chunk_indices_buffer[:padded_prefill_count] = cum_chunks[1:] - 1

            # seq_idx_for_varlen: repeat each seq index by its n_chunks, padded.
            total_real_chunks = cum_chunks[padded_prefill_count]
            pad_chunks = (padded_max_chunks - total_real_chunks).unsqueeze(0)
            self._seq_idx_for_varlen_buffer[:padded_max_chunks] = torch.repeat_interleave(
                torch.cat([seq_indices, seq_indices.new_zeros(1)]),
                torch.cat([n_chunks.to(torch.int64), pad_chunks]),
                output_size=padded_max_chunks,
            )

            # cu_chunk_seqlens: vectorized boundary computation.
            global_idx = self._arange_chunks[:padded_max_chunks]
            seq_of_chunk = self._seq_idx_for_varlen_buffer[:padded_max_chunks].to(torch.int64)
            local_idx = global_idx - cum_chunks[seq_of_chunk]
            starts_per_chunk = cu[seq_of_chunk].to(torch.int64)
            ends_per_chunk = cu[seq_of_chunk + 1].to(torch.int64)

            boundaries = torch.minimum(
                starts_per_chunk + (local_idx + 1) * chunk_size, ends_per_chunk
            )
            boundaries = torch.where(
                global_idx < total_real_chunks, boundaries, boundaries.new_zeros(1)
            )

            self._cu_chunk_seqlens_buffer[1 : padded_max_chunks + 1] = boundaries.to(torch.int32)

            n_cu = padded_max_chunks + 1
            self.cu_chunk_seqlens = self._cu_chunk_seqlens_buffer[:n_cu]
            self.last_chunk_indices = self._last_chunk_indices_buffer[:padded_prefill_count]
            self.seq_idx_for_varlen = self._seq_idx_for_varlen_buffer[:padded_max_chunks]

            # Conv metadata: use padded cu_seqlens. Padding sequences have
            # length 0, so repeat_interleave produces nothing for them.
            seq_starts = cu[:padded_prefill_count].to(torch.int32)
            pad_tokens_conv = (
                (padded_token_count - cu[padded_prefill_count]).to(torch.int64).unsqueeze(0)
            )
            padded_lengths = torch.cat([lengths, pad_tokens_conv])

            self._conv_seq_idx_buffer[:padded_token_count] = torch.repeat_interleave(
                torch.cat([seq_indices, seq_indices.new_zeros(1)]),
                padded_lengths,
                output_size=padded_token_count,
            )
            self._conv_seq_start_buffer[:padded_token_count] = torch.repeat_interleave(
                torch.cat([seq_starts, seq_starts.new_zeros(1)]),
                padded_lengths,
                output_size=padded_token_count,
            )

            self.conv_seq_idx = self._conv_seq_idx_buffer[:padded_token_count]
            self.conv_seq_start = self._conv_seq_start_buffer[:padded_token_count]

            self._update_intermediate_metadata(
                intermediate_offsets_gpu=intermediate_offsets_gpu,
                intermediate_counts_gpu=intermediate_counts_gpu,
                real_decode_count_gpu=real_decode_count_gpu,
                real_prefill_count_gpu=real_prefill_count_gpu,
                arange_buf=arange_buf,
                padded_prefill_count=padded_prefill_count,
            )

        if padded_decode_count > 0 and padded_prefill_count > 0:
            self._device_decode_prefill_buffer[0] = cu_seqlens[real_decode_count_gpu]
            self._device_decode_prefill_buffer[1] = (
                cu_seqlens[real_decode_count_gpu + real_prefill_count_gpu]
                - cu_seqlens[real_decode_count_gpu]
            )
            self.device_decode_prefill = self._device_decode_prefill_buffer

    def _update_intermediate_metadata(
        self,
        intermediate_offsets_gpu: Optional[torch.Tensor],
        intermediate_counts_gpu: Optional[torch.Tensor],
        real_decode_count_gpu: torch.Tensor,
        real_prefill_count_gpu: torch.Tensor,
        arange_buf: torch.Tensor,
        padded_prefill_count: int,
    ) -> None:
        """Hook called from :meth:`update` after base prefill metadata is built.

        No-op on the base class. :class:`PrefixCachedMambaMetadata` overrides this.
        """

    def allocate_slot(self) -> Optional[int]:
        """Allocate a new slot for a request in the Mamba state buffers.

        Returns:
            int: The index of the allocated slot, or ``None`` if exhausted.
        """
        if self.mamba_state_free_slot_count == 0:
            return None
        self.mamba_state_free_slot_count -= 1
        mamba_idx = self.mamba_state_free_slots[self.mamba_state_free_slot_count]
        self._free_slot_count_gpu.fill_(self.mamba_state_free_slot_count)

        return mamba_idx

    def batch_allocate_slots(self, num_slots: int) -> Optional[torch.Tensor]:
        """Allocate ``num_slots`` slots, or ``None`` if not enough are free."""
        if self.mamba_state_free_slot_count < num_slots:
            return None
        self.mamba_state_free_slot_count -= num_slots
        return self.mamba_state_free_slots[
            self.mamba_state_free_slot_count : self.mamba_state_free_slot_count + num_slots
        ]
        self._free_slot_count_gpu.fill_(self.mamba_state_free_slot_count)

    def free_slots(self, request_indices: torch.Tensor) -> None:
        """Free the Mamba state slots associated with the given request indices."""
        mamba_indices_to_free = self.request_to_mamba_state_idx[request_indices]
        mamba_indices_to_free = mamba_indices_to_free[mamba_indices_to_free != -1]
        num_to_free = len(mamba_indices_to_free)

        if num_to_free > 0:
            start_idx = self.mamba_state_free_slot_count
            end_idx = start_idx + num_to_free
            self.mamba_state_free_slots[start_idx:end_idx] = mamba_indices_to_free
            self.mamba_state_free_slot_count = end_idx
            self._free_slot_count_gpu.fill_(self.mamba_state_free_slot_count)

        self.request_to_mamba_state_idx[request_indices] = -1

    def free_slots_gpu(self, packed_slots: torch.Tensor, num_valid: torch.Tensor) -> None:
        """Shape-stable mamba free that writes a pre-packed buffer onto the stack.

        Args:
            packed_slots: (`max_requests`,) tensor with mamba slot IDs packed at `[0, num_valid)`.
            num_valid: 0-d or 1-element int tensor holding the valid count.
        """
        assert packed_slots.numel() == self.max_requests
        free_indices = self._free_slot_count_gpu + self._free_arange
        self.mamba_state_free_slots[free_indices.long()] = packed_slots
        self._free_slot_count_gpu.add_(num_valid.to(torch.int32).view(1))

    def sync_to_cpu(self) -> None:
        """Mirror `_free_slot_count_gpu` back into the Python int."""
        self.mamba_state_free_slot_count = int(self._free_slot_count_gpu.item())


class PrefixCachedMambaMetadata(MambaMetadata):
    """:class:`MambaMetadata` plus intermediate-state extraction for prefix caching.

    The base class produces everything the SSM/conv kernels need to run a
    prefill batch. Prefix caching additionally needs to extract per-block
    intermediate states (one entry per cached block, up to
    :data:`MAX_INTERMEDIATE_OFFSETS_PER_REQUEST` per request) from the
    per-step prefill output, so that subsequent requests with matching
    prefixes can resume from those checkpoints. That extraction is the only
    Mamba-side cost paid by enabling prefix caching, so it lives here in
    its own subclass.
    """

    def __init__(
        self,
        max_requests: int,
        max_tokens: int,
        mamba_chunk_size: int = 128,
        d_conv: int = 0,
        request_to_mamba_state_idx_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            max_requests,
            max_tokens,
            mamba_chunk_size,
            d_conv,
            request_to_mamba_state_idx_buf=request_to_mamba_state_idx_buf,
        )

        # Each prefill request can produce up to MAX_INTERMEDIATE_OFFSETS_PER_REQUEST offsets.
        self.max_intermediate_count = MAX_INTERMEDIATE_OFFSETS_PER_REQUEST * max_requests
        self._intermediate_chunk_indices_buffer = torch.zeros(
            self.max_intermediate_count, dtype=torch.int64, device=self.device
        )
        self._intermediate_abs_positions_buffer = torch.full(
            (self.max_intermediate_count,), d_conv, dtype=torch.int32, device=self.device
        )

        # Constant gather offsets for conv state extraction: [-d_conv, ..., -1].
        if d_conv > 0:
            self.conv_gather_offsets = torch.arange(
                -d_conv, 0, dtype=torch.int32, device=self.device
            )
        else:
            self.conv_gather_offsets = None

        # Pre-allocated index tensor for graphable intermediate ops.
        self._arange_intermediate = torch.arange(
            MAX_INTERMEDIATE_OFFSETS_PER_REQUEST, dtype=torch.int64, device=self.device
        )

        self._reset_intermediate_state()

    def _reset_intermediate_state(self) -> None:
        """Reset per-step intermediate-extraction views/counts to neutral values."""
        self.intermediate_chunk_indices = None
        self.intermediate_abs_positions = None
        self._pending_intermediate_counts_gpu: Optional[torch.Tensor] = None

    def reset_varlen_metadata(self) -> None:
        super().reset_varlen_metadata()
        self._reset_intermediate_state()

    def _update_intermediate_metadata(
        self,
        intermediate_offsets_gpu: Optional[torch.Tensor],
        intermediate_counts_gpu: Optional[torch.Tensor],
        real_decode_count_gpu: torch.Tensor,
        real_prefill_count_gpu: torch.Tensor,
        arange_buf: torch.Tensor,
        padded_prefill_count: int,
    ) -> None:
        """Precompute intermediate extraction metadata for CUDA-graph compatibility.

        Uses GPU scalar offsets and fixed-shape ops. Reuses ``self.cu_seqlens``
        and ``self.cum_chunks`` produced by the base class. When no slot
        allocator exists (buffers are ``None``), fills with safe defaults.

        Args:
            intermediate_offsets_gpu: Full ``[max_requests, K]`` GPU buffer, or None.
            intermediate_counts_gpu: Full ``[max_requests]`` GPU buffer, or None.
            real_decode_count_gpu: 1-element GPU scalar — gather offset.
            real_prefill_count_gpu: 1-element GPU scalar — mask boundary.
            arange_buf: Pre-allocated ``[0..N)`` index tensor.
            padded_prefill_count: Padded prefill request count (bounded).
        """
        max_count = self.max_intermediate_count
        chunk_size = self.mamba_chunk_size

        if intermediate_offsets_gpu is not None:
            # Gather the prefill portion from the full buffers using a GPU scalar offset.
            gather_idx = (
                arange_buf[:padded_prefill_count].to(torch.int64) + real_decode_count_gpu
            )
            offsets_2d = intermediate_offsets_gpu[gather_idx].to(torch.int64)  # [ppc, K]
            counts_1d = intermediate_counts_gpu[gather_idx]  # [ppc]

            # Zero out padding entries (beyond real prefill count).
            req_mask = arange_buf[:padded_prefill_count] < real_prefill_count_gpu
            counts_1d = torch.where(req_mask, counts_1d, 0)

            seq_starts = self.cu_seqlens[:padded_prefill_count].to(torch.int64)
            cum_chunks_exp = (
                self.cum_chunks[:padded_prefill_count].unsqueeze(1).expand_as(offsets_2d)
            )
            seq_starts_exp = seq_starts.unsqueeze(1).expand_as(offsets_2d)

            chunk_indices_2d = cum_chunks_exp + offsets_2d // chunk_size - 1
            abs_positions_2d = seq_starts_exp + offsets_2d

            j_indices = self._arange_intermediate.unsqueeze(0)
            valid_mask = j_indices < counts_1d.unsqueeze(1)

            safe_chunk_indices = torch.where(
                valid_mask, chunk_indices_2d, torch.zeros_like(chunk_indices_2d)
            )
            safe_abs_positions = torch.where(
                valid_mask, abs_positions_2d, torch.full_like(abs_positions_2d, self.d_conv)
            )

            total_entries = padded_prefill_count * MAX_INTERMEDIATE_OFFSETS_PER_REQUEST
            self._intermediate_chunk_indices_buffer[:total_entries] = safe_chunk_indices.flatten()
            self._intermediate_abs_positions_buffer[:total_entries] = (
                safe_abs_positions.flatten().to(torch.int32)
            )
            if total_entries < max_count:
                self._intermediate_chunk_indices_buffer[total_entries:].fill_(0)
                self._intermediate_abs_positions_buffer[total_entries:].fill_(self.d_conv)

            # Defer .tolist() to commit_intermediate_states (post-forward); the
            # GPU sync there overlaps with the forward pass.
            self._pending_intermediate_counts_gpu = counts_1d
        else:
            # No slot allocator: fill with safe defaults unconditionally.
            self._intermediate_chunk_indices_buffer.fill_(0)
            self._intermediate_abs_positions_buffer.fill_(self.d_conv)
            self._pending_intermediate_counts_gpu = None

        self.intermediate_chunk_indices = self._intermediate_chunk_indices_buffer[:max_count]
        self.intermediate_abs_positions = self._intermediate_abs_positions_buffer[:max_count]
