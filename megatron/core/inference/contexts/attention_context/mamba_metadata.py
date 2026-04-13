# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.contexts.mamba_slot_allocator import (
    MAX_INTERMEDIATE_OFFSETS_PER_REQUEST,
)

from .triton.fused_mamba_chunk_metadata import (
    HAVE_TRITON,
    fused_conv_metadata,
    fused_mamba_chunk_metadata,
)


class MambaMetadata:
    """Manages the metadata tensors required for Mamba layers during inference."""

    def __init__(
        self, max_requests: int, max_tokens: int, mamba_chunk_size: int = 128, d_conv: int = 0
    ):
        """
        Initializes the Mamba slot allocator.

        Args:
            max_requests (int): The maximum number of concurrent requests.
            max_tokens (int): The maximum number of tokens.
            mamba_chunk_size (int): The chunk size used by the Mamba SSM Triton kernels.
            d_conv (int): Convolution window size (from mamba_conv_states_shape[-1]).
                Used for vectorized conv state extraction at intermediate offsets.
        """
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.mamba_chunk_size = mamba_chunk_size
        self.d_conv = d_conv
        self.device = torch.cuda.current_device()

        # Maximum possible chunks across all batch configurations
        self.max_chunks = max_tokens // mamba_chunk_size + max_requests

        # Map from requests to slots in the static Mamba state buffer
        self.request_to_mamba_state_idx = torch.full(
            (self.max_requests,), -1, dtype=torch.int32, device=torch.cuda.current_device()
        )

        # Map from requests to slots in the static Mamba state buffer for active decode requests
        self._batch_indices_decode_buffer = torch.full(
            (self.max_requests,), -1, dtype=torch.int32, device=self.device
        )

        # Map from requests to slots in the static Mamba state buffer for active prefill requests
        self._batch_indices_prefill_buffer = torch.full(
            (self.max_requests,), -1, dtype=torch.int32, device=self.device
        )

        # Map from token id to request id for active prefill requests
        self._seq_idx_buffer = torch.full(
            (1, self.max_tokens), -1, dtype=torch.int32, device=self.device
        )

        # Cumulative sequence lengths for active prefill requests
        self._cu_seqlens_buffer = torch.zeros(
            (self.max_requests + 1,), dtype=torch.int32, device=self.device
        )

        # Tuple of (active decode request count, active prefill request count)
        self._device_decode_prefill_buffer = torch.zeros(
            (2,), dtype=torch.int32, device=self.device
        )

        # SSM chunk boundaries for varlen kernel
        self._cu_chunk_seqlens_buffer = torch.zeros(
            self.max_chunks + 1, dtype=torch.int32, device=self.device
        )

        # Index of the last chunk per sequence
        self._last_chunk_indices_buffer = torch.zeros(
            max_requests, dtype=torch.int32, device=self.device
        )

        # Request ID per chunk
        self._seq_idx_for_varlen_buffer = torch.zeros(
            self.max_chunks, dtype=torch.int32, device=self.device
        )

        # Conv1d per-token metadata (request ID and request start position)
        self._conv_seq_idx_buffer = torch.zeros(max_tokens, dtype=torch.int32, device=self.device)
        self._conv_seq_start_buffer = torch.zeros(max_tokens, dtype=torch.int32, device=self.device)

        # Allocator for Mamba state slots
        self.mamba_state_free_slots = torch.arange(
            self.max_requests, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.mamba_state_free_slot_count = self.max_requests

        # Intermediate state extraction buffers (CUDA graph compatible)
        # Each prefill request can produce up to 3 intermediate offsets
        self.max_intermediate_count = MAX_INTERMEDIATE_OFFSETS_PER_REQUEST * max_requests
        self._intermediate_chunk_indices_buffer = torch.zeros(
            self.max_intermediate_count, dtype=torch.int64, device=self.device
        )
        self._intermediate_abs_positions_buffer = torch.full(
            (self.max_intermediate_count,), d_conv, dtype=torch.int32, device=self.device
        )
        # Constant gather offsets for conv state extraction: [-d_conv, ..., -1]
        if d_conv > 0:
            self.conv_gather_offsets = torch.arange(
                -d_conv, 0, dtype=torch.int32, device=self.device
            )
        else:
            self.conv_gather_offsets = None

        # Scratch buffer shared between the two fused chunk metadata Triton kernels.
        # Kernel 1 writes cumulative chunk counts per sequence (e.g. [0, 2, 5, 6]),
        # kernel 2 reads them to know where each sequence's chunk boundaries go in
        # the output arrays. Pre-allocated to avoid per-step allocation.
        self._cum_chunks_buffer = torch.zeros(
            self.max_requests + 1, dtype=torch.int32, device=self.device
        )

        self.reset_varlen_metadata()

    def reset(self) -> None:
        """
        Resets all Mamba states and frees all allocated slots.
        """
        self.request_to_mamba_state_idx.fill_(-1)

        self.reset_varlen_metadata()

        # Re-initialize the free slot pool
        self.mamba_state_free_slots = torch.arange(
            self.max_requests, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.mamba_state_free_slot_count = self.max_requests

    def reset_varlen_metadata(self) -> None:
        """Resets varlen metadata."""
        self.batch_indices_decode = None
        self.batch_indices_prefill = None
        self.cu_seqlens = None
        self.seq_idx = None
        self.device_decode_prefill = None

        # SSM/conv1d precomputed views
        self.cu_chunk_seqlens = None
        self.last_chunk_indices = None
        self.seq_idx_for_varlen = None
        self.conv_seq_idx = None
        self.conv_seq_start = None

        # Intermediate state extraction views
        self.intermediate_chunk_indices = None
        self.intermediate_abs_positions = None
        self.intermediate_count = 0
        self.per_request_intermediate_counts = []

    def update(
        self,
        active_mamba_indices: torch.Tensor,
        token_to_request_idx: torch.Tensor,
        cu_seqlens: torch.Tensor,
        batch_dimensions: InferenceBatchDimensions,
        padded_batch_dimensions: InferenceBatchDimensions,
        enable_chunked_prefill: bool,
        intermediate_offsets_gpu: Optional[torch.Tensor] = None,
        intermediate_counts_gpu: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Updates the dedicated CUDA graph mapping tensor with the indices
        of currently active requests.

        Args:
            active_mamba_indices (Tensor): Tensor containing the Mamba slot indices
                                           for active requests.
            token_to_request_idx (Tensor): Map from token index to request index.
            cu_seqlens (Tensor): Cumulative sequence lengths.
            batch_dimensions (InferenceBatchDimensions): Dimensions of the current batch.
            padded_batch_dimensions (InferenceBatchDimensions): Dimensions of the padded batch.
            intermediate_offsets_gpu (Tensor): [prefill_count, 3] int32 GPU tensor of
                per-request intermediate token offsets, or None.
            intermediate_counts_gpu (Tensor): [prefill_count] int32 GPU tensor of
                per-request intermediate offset counts, or None.
        """
        real_decode_count = batch_dimensions.decode_req_count
        real_prefill_count = batch_dimensions.prefill_req_count

        padded_decode_count = padded_batch_dimensions.decode_req_count
        padded_prefill_count = padded_batch_dimensions.prefill_req_count
        padded_token_count = padded_batch_dimensions.token_count

        if padded_decode_count > 0:
            # Update decode indices
            self._batch_indices_decode_buffer[:real_decode_count].copy_(
                active_mamba_indices[:real_decode_count]
            )
            if padded_decode_count > real_decode_count:
                self._batch_indices_decode_buffer[real_decode_count:padded_decode_count] = -1
            self.batch_indices_decode = self._batch_indices_decode_buffer[:padded_decode_count]

        if padded_prefill_count > 0:
            # Update prefill indices (all prefill requests go through varlen)
            if real_prefill_count > 0:
                prefill_start_idx = real_decode_count
                self._batch_indices_prefill_buffer[:real_prefill_count].copy_(
                    active_mamba_indices[prefill_start_idx : prefill_start_idx + real_prefill_count]
                )

            if padded_prefill_count > real_prefill_count:
                self._batch_indices_prefill_buffer[real_prefill_count:padded_prefill_count] = -1

            self.batch_indices_prefill = self._batch_indices_prefill_buffer[:padded_prefill_count]

            # Update seq_idx for all prefill requests
            prefill_start_req_idx = real_decode_count
            end_prefill_req_idx = real_decode_count + real_prefill_count

            start_prefill_token_idx = cu_seqlens[prefill_start_req_idx]
            end_prefill_token_idx = cu_seqlens[end_prefill_req_idx]

            seq_len = end_prefill_token_idx - start_prefill_token_idx

            if seq_len > 0:
                # Normalize request IDs to 0-based relative to prefill requests
                self._seq_idx_buffer[:, :seq_len].copy_(
                    token_to_request_idx[start_prefill_token_idx:end_prefill_token_idx]
                    - token_to_request_idx[start_prefill_token_idx]
                )

            if padded_token_count > seq_len:
                self._seq_idx_buffer[:, seq_len:padded_token_count] = -1
            self.seq_idx = self._seq_idx_buffer[:, :padded_token_count]

            # Update cu_seqlens for all prefill requests
            self._cu_seqlens_buffer[0] = 0
            if real_prefill_count > 0:
                self._cu_seqlens_buffer[1 : real_prefill_count + 1].copy_(
                    cu_seqlens[prefill_start_req_idx + 1 : end_prefill_req_idx + 1]
                    - cu_seqlens[prefill_start_req_idx]
                )

            # Pad the rest with the last value (effectively length 0 segments)
            last_val = self._cu_seqlens_buffer[real_prefill_count]
            self._cu_seqlens_buffer[real_prefill_count + 1 : padded_prefill_count + 1].fill_(
                last_val
            )
            self.cu_seqlens = self._cu_seqlens_buffer[: padded_prefill_count + 1]

            # --- Precompute SSM and conv1d metadata for CUDA graph compatibility ---
            # All values the forward pass needs are computed here (before CUDA graph
            # capture/replay) so that the forward pass has no .item() calls or
            # data-dependent control flow.

            chunk_size = self.mamba_chunk_size
            padded_max_chunks = padded_token_count // chunk_size + padded_prefill_count

            if HAVE_TRITON:
                fused_mamba_chunk_metadata(
                    cu_seqlens_buf=self._cu_seqlens_buffer,
                    cum_chunks_buf=self._cum_chunks_buffer,
                    cu_chunk_seqlens_buf=self._cu_chunk_seqlens_buffer,
                    last_chunk_indices_buf=self._last_chunk_indices_buffer,
                    seq_idx_for_varlen_buf=self._seq_idx_for_varlen_buffer,
                    padded_prefill_count=padded_prefill_count,
                    chunk_size=chunk_size,
                    padded_max_chunks=padded_max_chunks,
                    padded_token_count=padded_token_count,
                    max_requests=self.max_requests,
                    max_tokens=self.max_tokens,
                )
            else:
                self._build_chunk_metadata_cpu(padded_prefill_count, chunk_size, padded_max_chunks)

            n_cu = padded_max_chunks + 1
            self.cu_chunk_seqlens = self._cu_chunk_seqlens_buffer[:n_cu]
            self.last_chunk_indices = self._last_chunk_indices_buffer[:padded_prefill_count]
            self.seq_idx_for_varlen = self._seq_idx_for_varlen_buffer[:padded_max_chunks]

            if HAVE_TRITON:
                fused_conv_metadata(
                    cu_seqlens_buf=self._cu_seqlens_buffer,
                    conv_seq_idx_buf=self._conv_seq_idx_buffer,
                    conv_seq_start_buf=self._conv_seq_start_buffer,
                    real_prefill_count=real_prefill_count,
                    padded_token_count=padded_token_count,
                    max_tokens=self.max_tokens,
                )
            else:
                self._build_conv_metadata_cpu(real_prefill_count, padded_token_count)

            self.conv_seq_idx = self._conv_seq_idx_buffer[:padded_token_count]
            self.conv_seq_start = self._conv_seq_start_buffer[:padded_token_count]

            # --- Precompute intermediate state extraction metadata ---
            # This converts per-request token offsets to chunk indices and
            # absolute positions, padded to fixed size for CUDA graph compat.
            self._update_intermediate_metadata(
                intermediate_offsets_gpu, intermediate_counts_gpu, real_prefill_count
            )

        if padded_decode_count > 0 and padded_prefill_count > 0:
            self._device_decode_prefill_buffer[0] = cu_seqlens[real_decode_count]
            self._device_decode_prefill_buffer[1] = (
                cu_seqlens[real_decode_count + real_prefill_count] - cu_seqlens[real_decode_count]
            )
            self.device_decode_prefill = self._device_decode_prefill_buffer

    def _update_intermediate_metadata(
        self,
        intermediate_offsets_gpu: Optional[torch.Tensor],
        intermediate_counts_gpu: Optional[torch.Tensor],
        real_prefill_count: int,
    ) -> None:
        """Precompute intermediate extraction metadata for CUDA graph compatibility.

        Converts per-request token offsets to chunk indices and absolute
        positions using vectorized GPU operations, padding unused entries
        to fixed buffer size.

        Args:
            intermediate_offsets_gpu: [real_prefill_count, 3] int32 GPU tensor
                of per-request token offsets, or None if no extraction needed.
            intermediate_counts_gpu: [real_prefill_count] int32 GPU tensor of
                per-request offset counts (0-3), or None.
            real_prefill_count: Number of real (non-padding) prefill requests.
        """
        chunk_size = self.mamba_chunk_size
        max_count = self.max_intermediate_count

        if intermediate_offsets_gpu is not None and real_prefill_count > 0:
            total_gpu = intermediate_counts_gpu.sum()

            # Compute cumulative chunk counts from cu_seqlens
            cu = self._cu_seqlens_buffer[: real_prefill_count + 1]
            seq_lens = (cu[1 : real_prefill_count + 1] - cu[:real_prefill_count]).to(torch.int64)
            num_chunks = torch.clamp((seq_lens + chunk_size - 1) // chunk_size, min=1)
            cum_chunks = torch.zeros(real_prefill_count + 1, dtype=torch.int64, device=self.device)
            torch.cumsum(num_chunks, dim=0, out=cum_chunks[1:])

            seq_starts = cu[:real_prefill_count].to(torch.int64)
            offsets = intermediate_offsets_gpu.to(torch.int64)

            # Expand per-request values to [real_prefill_count, 3]
            cum_chunks_exp = cum_chunks[:real_prefill_count].unsqueeze(1).expand_as(offsets)
            seq_starts_exp = seq_starts.unsqueeze(1).expand_as(offsets)

            # Vectorized computation of chunk indices and absolute positions
            chunk_indices_2d = cum_chunks_exp + offsets // chunk_size - 1
            abs_positions_2d = seq_starts_exp + offsets

            # Validity mask: j < count[i] for each request
            j_indices = torch.arange(
                MAX_INTERMEDIATE_OFFSETS_PER_REQUEST, device=self.device
            ).unsqueeze(0)
            valid_mask = j_indices < intermediate_counts_gpu.unsqueeze(1)

            # Flatten valid entries into output buffers
            valid_chunk_indices = chunk_indices_2d[valid_mask]
            valid_abs_positions = abs_positions_2d[valid_mask]

            # Now sync for the single scalar — all GPU work above is already queued
            total = total_gpu.item()

            if total > 0:
                real_count = valid_chunk_indices.numel()
                self._intermediate_chunk_indices_buffer[:real_count] = valid_chunk_indices
                self._intermediate_abs_positions_buffer[:real_count] = valid_abs_positions.to(
                    torch.int32
                )

                # Pad unused slots with safe defaults for CUDA graph replay:
                # - chunk_indices=0: reads from chunk 0 (always exists), output ignored
                # - abs_positions=d_conv: conv gather reads tokens [0..d_conv-1],
                #   which are within bounds and produce a valid but unused state
                if real_count < max_count:
                    self._intermediate_chunk_indices_buffer[real_count:].fill_(0)
                    self._intermediate_abs_positions_buffer[real_count:].fill_(self.d_conv)

                self.intermediate_count = real_count
            else:
                # All counts are 0
                self._intermediate_chunk_indices_buffer.fill_(0)
                self._intermediate_abs_positions_buffer.fill_(self.d_conv)
                self.intermediate_count = 0

            self.per_request_intermediate_counts = intermediate_counts_gpu.tolist()

            self.intermediate_chunk_indices = self._intermediate_chunk_indices_buffer[:max_count]
            self.intermediate_abs_positions = self._intermediate_abs_positions_buffer[:max_count]
        else:
            # No extraction: fill with safe defaults for CUDA graph warmup
            # (same rationale as padding comment above)
            self._intermediate_chunk_indices_buffer.fill_(0)
            self._intermediate_abs_positions_buffer.fill_(self.d_conv)
            self.intermediate_count = 0
            self.per_request_intermediate_counts = []
            self.intermediate_chunk_indices = self._intermediate_chunk_indices_buffer[:max_count]
            self.intermediate_abs_positions = self._intermediate_abs_positions_buffer[:max_count]

    def _build_chunk_metadata_cpu(
        self, padded_prefill_count: int, chunk_size: int, padded_max_chunks: int
    ) -> None:
        """Fallback: build chunk metadata on CPU via .tolist() + Python loop."""
        cu_seqlens_all = self._cu_seqlens_buffer[: padded_prefill_count + 1].tolist()
        chunk_boundaries = [0]
        last_chunk_idx_list = []
        chunk_to_seq_list = []

        for i in range(padded_prefill_count):
            start = cu_seqlens_all[i]
            end = cu_seqlens_all[i + 1]
            seq_len = end - start
            n_chunks = max(1, (seq_len + chunk_size - 1) // chunk_size)
            boundaries = [min(start + (k + 1) * chunk_size, end) for k in range(n_chunks)]
            chunk_boundaries.extend(boundaries)
            chunk_to_seq_list.extend([i] * n_chunks)
            last_chunk_idx_list.append(len(chunk_boundaries) - 2)

        last_boundary = chunk_boundaries[-1]
        pad_b = padded_max_chunks + 1 - len(chunk_boundaries)
        if pad_b > 0:
            chunk_boundaries.extend([last_boundary] * pad_b)
        pad_s = padded_max_chunks - len(chunk_to_seq_list)
        if pad_s > 0:
            chunk_to_seq_list.extend([0] * pad_s)

        n_cu = padded_max_chunks + 1
        self._cu_chunk_seqlens_buffer[:n_cu].copy_(
            torch.tensor(chunk_boundaries[:n_cu], dtype=torch.int32)
        )
        self._last_chunk_indices_buffer[:padded_prefill_count].copy_(
            torch.tensor(last_chunk_idx_list, dtype=torch.int32)
        )
        self._seq_idx_for_varlen_buffer[:padded_max_chunks].copy_(
            torch.tensor(chunk_to_seq_list[:padded_max_chunks], dtype=torch.int32)
        )

    def _build_conv_metadata_cpu(self, real_prefill_count: int, padded_token_count: int) -> None:
        """Fallback: build conv1d per-token metadata via repeat_interleave."""
        real_tokens_gpu = self._cu_seqlens_buffer[real_prefill_count]
        if real_prefill_count > 0:
            cu = self._cu_seqlens_buffer[: real_prefill_count + 1]
            lengths = (cu[1:] - cu[:-1]).to(torch.int64)
            seq_indices = torch.arange(real_prefill_count, dtype=torch.int32, device=self.device)
            seq_starts = cu[:real_prefill_count].to(torch.int32)
            self._conv_seq_idx_buffer[:real_tokens_gpu] = torch.repeat_interleave(
                seq_indices, lengths
            )
            self._conv_seq_start_buffer[:real_tokens_gpu] = torch.repeat_interleave(
                seq_starts, lengths
            )
            self._conv_seq_idx_buffer[real_tokens_gpu:padded_token_count] = 0
            self._conv_seq_start_buffer[real_tokens_gpu:padded_token_count] = 0
        else:
            self._conv_seq_idx_buffer[:padded_token_count] = 0
            self._conv_seq_start_buffer[:padded_token_count] = 0

    def allocate_slot(self) -> Optional[int]:
        """
        Allocates a new slot for a request in the Mamba state buffers.

        Returns:
            int: The index of the allocated slot.
            Returns None if no slots are available.
        """
        if self.mamba_state_free_slot_count == 0:
            return None

        # Get a free slot
        self.mamba_state_free_slot_count -= 1
        mamba_idx = self.mamba_state_free_slots[self.mamba_state_free_slot_count]

        return mamba_idx

    def batch_allocate_slots(self, num_slots: int) -> Optional[torch.Tensor]:
        """
        Allocates new slots for the given number of requests in the Mamba state buffers.

        Returns:
            torch.Tensor: The indices of the allocated slots.
            Returns None if not enough slots are available.
        """
        if self.mamba_state_free_slot_count < num_slots:
            return None

        # Get free slots
        self.mamba_state_free_slot_count -= num_slots
        mamba_idx = self.mamba_state_free_slots[
            self.mamba_state_free_slot_count : self.mamba_state_free_slot_count + num_slots
        ]

        return mamba_idx

    def free_slots(self, request_indices: torch.Tensor) -> None:
        """
        Frees the Mamba state slots associated with the given request indices.

        Args:
            request_indices (Tensor): A 1D tensor of request indices to free.
        """
        # Get the Mamba state indices for finished requests
        mamba_indices_to_free = self.request_to_mamba_state_idx[request_indices]

        # Filter out any invalid indices (e.g., -1)
        mamba_indices_to_free = mamba_indices_to_free[mamba_indices_to_free != -1]
        num_to_free = len(mamba_indices_to_free)

        if num_to_free > 0:
            # Add the freed indices back to the free slot pool
            start_idx = self.mamba_state_free_slot_count
            end_idx = start_idx + num_to_free
            self.mamba_state_free_slots[start_idx:end_idx] = mamba_indices_to_free
            self.mamba_state_free_slot_count = end_idx

        # Invalidate the Mamba state index for the finished requests
        self.request_to_mamba_state_idx[request_indices] = -1
