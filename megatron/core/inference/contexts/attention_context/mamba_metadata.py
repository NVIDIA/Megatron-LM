# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions


class MambaMetadata:
    """Manages the metadata tensors required for Mamba layers during inference."""

    def __init__(self, max_requests: int, max_tokens: int, mamba_chunk_size: int = 128):
        """
        Initializes the Mamba slot allocator.

        Args:
            max_requests (int): The maximum number of concurrent requests.
            max_tokens (int): The maximum number of tokens.
            mamba_chunk_size (int): The chunk size used by the Mamba SSM Triton kernels.
        """
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.mamba_chunk_size = mamba_chunk_size
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

        # Python-side precomputed values
        self.real_prefill_token_count = 0
        self.cu_seqlens_list = [0]

    def update(
        self,
        active_mamba_indices: torch.Tensor,
        token_to_request_idx: torch.Tensor,
        cu_seqlens: torch.Tensor,
        batch_dimensions: InferenceBatchDimensions,
        padded_batch_dimensions: InferenceBatchDimensions,
        enable_chunked_prefill: bool,
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
                    - prefill_start_req_idx
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

            # Transfer cu_seqlens to CPU for Python-side precomputation
            cu_seqlens_real = self._cu_seqlens_buffer[: real_prefill_count + 1].tolist()
            self.cu_seqlens_list = cu_seqlens_real
            self.real_prefill_token_count = (
                cu_seqlens_real[real_prefill_count] if real_prefill_count > 0 else 0
            )

            # Build cu_chunk_seqlens, last_chunk_indices, seq_idx_for_varlen.
            # Covers all padded sequences (real + padding). Each sequence is
            # subdivided into chunks of at most mamba_chunk_size tokens. Zero-length
            # sequences get a single zero-length chunk.
            cu_seqlens_all = self._cu_seqlens_buffer[: padded_prefill_count + 1].tolist()
            chunk_size = self.mamba_chunk_size
            chunk_boundaries = [0]
            last_chunk_idx_list = []
            chunk_to_seq_list = []

            for i in range(padded_prefill_count):
                start = cu_seqlens_all[i]
                end = cu_seqlens_all[i + 1]
                pos = start + chunk_size
                while pos < end:
                    chunk_boundaries.append(pos)
                    chunk_to_seq_list.append(i)
                    pos += chunk_size
                chunk_boundaries.append(end)
                chunk_to_seq_list.append(i)
                last_chunk_idx_list.append(len(chunk_boundaries) - 2)

            # Pad to fixed size for CUDA graph compatibility
            padded_max_chunks = padded_token_count // chunk_size + padded_prefill_count
            last_boundary = chunk_boundaries[-1]
            while len(chunk_boundaries) < padded_max_chunks + 1:
                chunk_boundaries.append(last_boundary)
            while len(chunk_to_seq_list) < padded_max_chunks:
                chunk_to_seq_list.append(0)

            # Fill GPU buffers
            n_cu = padded_max_chunks + 1
            self._cu_chunk_seqlens_buffer[:n_cu].copy_(
                torch.tensor(chunk_boundaries[:n_cu], dtype=torch.int32)
            )
            self.cu_chunk_seqlens = self._cu_chunk_seqlens_buffer[:n_cu]

            self._last_chunk_indices_buffer[:padded_prefill_count].copy_(
                torch.tensor(last_chunk_idx_list, dtype=torch.int32)
            )
            self.last_chunk_indices = self._last_chunk_indices_buffer[:padded_prefill_count]

            self._seq_idx_for_varlen_buffer[:padded_max_chunks].copy_(
                torch.tensor(chunk_to_seq_list[:padded_max_chunks], dtype=torch.int32)
            )
            self.seq_idx_for_varlen = self._seq_idx_for_varlen_buffer[:padded_max_chunks]

            # Build conv1d per-token metadata (request ID and request start position)
            real_tokens = self.real_prefill_token_count
            if real_tokens > 0:
                conv_seq_idx_list = []
                conv_seq_start_list = []
                for i in range(real_prefill_count):
                    start = cu_seqlens_real[i]
                    length = cu_seqlens_real[i + 1] - start
                    conv_seq_idx_list.extend([i] * length)
                    conv_seq_start_list.extend([start] * length)
                self._conv_seq_idx_buffer[:real_tokens].copy_(
                    torch.tensor(conv_seq_idx_list, dtype=torch.int32)
                )
                self._conv_seq_start_buffer[:real_tokens].copy_(
                    torch.tensor(conv_seq_start_list, dtype=torch.int32)
                )
            if padded_token_count > real_tokens:
                self._conv_seq_idx_buffer[real_tokens:padded_token_count] = 0
                self._conv_seq_start_buffer[real_tokens:padded_token_count] = 0

            self.conv_seq_idx = self._conv_seq_idx_buffer[:padded_token_count]
            self.conv_seq_start = self._conv_seq_start_buffer[:padded_token_count]

        if padded_decode_count > 0 and padded_prefill_count > 0:
            self._device_decode_prefill_buffer[0] = cu_seqlens[real_decode_count]
            self._device_decode_prefill_buffer[1] = (
                cu_seqlens[real_decode_count + real_prefill_count]
                - cu_seqlens[real_decode_count]
            )
            self.device_decode_prefill = self._device_decode_prefill_buffer

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
