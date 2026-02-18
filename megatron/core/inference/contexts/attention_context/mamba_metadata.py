# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions


class MambaMetadata:
    """Manages the metadata tensors required for Mamba layers during inference."""

    def __init__(self, max_requests: int, max_tokens: int):
        """
        Initializes the Mamba slot allocator.

        Args:
            max_requests (int): The maximum number of concurrent requests.
        """
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.device = torch.cuda.current_device()

        # Map from requests to slots in the static Mamba state buffer
        self.request_to_mamba_state_idx = torch.full(
            (self.max_requests,), -1, dtype=torch.int32, device=torch.cuda.current_device()
        )

        # Map from requests to slots in the static Mamba state buffer for active decode requests
        self._batch_indices_decode_buffer = torch.full(
            (self.max_requests,), -1, dtype=torch.int32, device=self.device
        )

        # Map from requests to slots in the static Mamba state buffer for active prefill requests
        # (varlen kernel requests only — batch-kernel requests use separate buffer)
        self._batch_indices_prefill_buffer = torch.full(
            (self.max_requests,), -1, dtype=torch.int32, device=self.device
        )

        # Per-batch-kernel-request: mamba slot index
        self._batch_kernel_indices_buffer = torch.full(
            (self.max_requests,), -1, dtype=torch.int32, device=self.device
        )

        # Per-batch-kernel-request: cumulative token offsets
        self._batch_kernel_cu_seqlens_buffer = torch.zeros(
            (self.max_requests + 1,), dtype=torch.int32, device=self.device
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

        # Tuple of (
        #   total batch-kernel prefill token count,
        #   total varlen prefill token count
        # )
        self._device_chunked_prefill_buffer = torch.zeros(
            (2,), dtype=torch.int32, device=self.device
        )

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
        self.device_chunked_prefill = None
        self.num_batch_kernel_prefills = 0
        self.batch_kernel_batch_indices = None
        self.batch_kernel_cu_seqlens = None

    def update(
        self,
        active_mamba_indices: torch.Tensor,
        token_to_request_idx: torch.Tensor,
        cu_seqlens: torch.Tensor,
        batch_dimensions: InferenceBatchDimensions,
        padded_batch_dimensions: InferenceBatchDimensions,
        enable_chunked_prefill: bool,
        prefill_has_initial_states: Optional[torch.Tensor] = None,
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
            enable_chunked_prefill (bool): Whether chunked prefill is enabled.
            prefill_has_initial_states (Optional[Tensor]): Boolean tensor indicating which
                active requests have initial Mamba states (from restored prefix cache or
                continuing chunked prefill). Shape: (num_active_requests,).
        """
        real_decode_count = batch_dimensions.decode_req_count
        real_prefill_count = batch_dimensions.prefill_req_count

        padded_decode_count = padded_batch_dimensions.decode_req_count
        padded_prefill_count = padded_batch_dimensions.prefill_req_count
        padded_token_count = padded_batch_dimensions.token_count

        # Determine how many prefill requests should use the batch kernel
        # (those with initial Mamba states: restored from cache or continuing chunked prefill).
        # The scheduler guarantees these come first among prefill requests.
        num_batch_kernel = 0
        if enable_chunked_prefill and real_prefill_count > 0:
            if prefill_has_initial_states is not None:
                prefill_flags = prefill_has_initial_states[
                    real_decode_count : real_decode_count + real_prefill_count
                ]
                num_batch_kernel = prefill_flags.sum().item()
            else:
                # Legacy fallback: treat first prefill as batch-kernel (old behavior)
                num_batch_kernel = 1

        self.num_batch_kernel_prefills = num_batch_kernel
        varlen_prefill_count = real_prefill_count - num_batch_kernel

        if padded_decode_count > 0:
            # Update decode indices
            self._batch_indices_decode_buffer[:real_decode_count].copy_(
                active_mamba_indices[:real_decode_count]
            )
            if padded_decode_count > real_decode_count:
                self._batch_indices_decode_buffer[real_decode_count:padded_decode_count] = -1
            self.batch_indices_decode = self._batch_indices_decode_buffer[:padded_decode_count]

        # Populate batch-kernel metadata
        if num_batch_kernel > 0:
            batch_start = real_decode_count
            self._batch_kernel_indices_buffer[:num_batch_kernel].copy_(
                active_mamba_indices[batch_start : batch_start + num_batch_kernel]
            )
            self.batch_kernel_batch_indices = self._batch_kernel_indices_buffer[:num_batch_kernel]

            # Cu seqlens for batch kernel requests (relative to first batch kernel token)
            batch_start_token = cu_seqlens[real_decode_count]
            self._batch_kernel_cu_seqlens_buffer[0] = 0
            self._batch_kernel_cu_seqlens_buffer[1 : num_batch_kernel + 1].copy_(
                cu_seqlens[real_decode_count + 1 : real_decode_count + num_batch_kernel + 1]
                - batch_start_token
            )
            self.batch_kernel_cu_seqlens = self._batch_kernel_cu_seqlens_buffer[
                : num_batch_kernel + 1
            ]
        else:
            self.batch_kernel_batch_indices = None
            self.batch_kernel_cu_seqlens = None

        if padded_prefill_count > 0:
            # Update varlen prefill indices
            if varlen_prefill_count > 0:
                varlen_start_idx = real_decode_count + num_batch_kernel
                self._batch_indices_prefill_buffer[:varlen_prefill_count].copy_(
                    active_mamba_indices[varlen_start_idx : varlen_start_idx + varlen_prefill_count]
                )

            if padded_prefill_count > varlen_prefill_count:
                self._batch_indices_prefill_buffer[
                    varlen_prefill_count:padded_prefill_count
                ] = -1

            self.batch_indices_prefill = self._batch_indices_prefill_buffer[:padded_prefill_count]

            # Update seq_idx for varlen prefills
            # Varlen requests start after batch-kernel requests in the prefill token stream
            varlen_start_req_idx = real_decode_count + num_batch_kernel
            end_varlen_req_idx = real_decode_count + real_prefill_count

            start_varlen_token_idx = cu_seqlens[varlen_start_req_idx]
            end_varlen_token_idx = cu_seqlens[end_varlen_req_idx]

            seq_len = end_varlen_token_idx - start_varlen_token_idx

            if seq_len > 0:
                # Normalize request IDs to 0-based relative to varlen requests
                self._seq_idx_buffer[:, :seq_len].copy_(
                    token_to_request_idx[start_varlen_token_idx:end_varlen_token_idx]
                    - varlen_start_req_idx
                )

            if padded_token_count > seq_len:
                self._seq_idx_buffer[:, seq_len:padded_token_count] = -1
            self.seq_idx = self._seq_idx_buffer[:, :padded_token_count]

            # Update cu_seqlens for varlen prefill requests
            self._cu_seqlens_buffer[0] = 0
            if varlen_prefill_count > 0:
                self._cu_seqlens_buffer[1 : varlen_prefill_count + 1].copy_(
                    cu_seqlens[varlen_start_req_idx + 1 : end_varlen_req_idx + 1]
                    - cu_seqlens[varlen_start_req_idx]
                )

            # Pad the rest with the last value (effectively length 0 segments)
            last_val = self._cu_seqlens_buffer[varlen_prefill_count]
            self._cu_seqlens_buffer[varlen_prefill_count + 1 : padded_prefill_count + 1].fill_(
                last_val
            )
            self.cu_seqlens = self._cu_seqlens_buffer[: padded_prefill_count + 1]

        if padded_decode_count > 0 and padded_prefill_count > 0:
            self._device_decode_prefill_buffer[0] = real_decode_count
            self._device_decode_prefill_buffer[1] = real_prefill_count
            self.device_decode_prefill = self._device_decode_prefill_buffer

        # Store the batch-kernel vs varlen token split
        if num_batch_kernel > 0:
            batch_kernel_total_tokens = (
                cu_seqlens[real_decode_count + num_batch_kernel] - cu_seqlens[real_decode_count]
            )
            varlen_total_tokens = 0
            if varlen_prefill_count > 0:
                varlen_total_tokens = (
                    cu_seqlens[real_decode_count + real_prefill_count]
                    - cu_seqlens[real_decode_count + num_batch_kernel]
                )

            self._device_chunked_prefill_buffer[0] = batch_kernel_total_tokens
            self._device_chunked_prefill_buffer[1] = varlen_total_tokens
            self.device_chunked_prefill = self._device_chunked_prefill_buffer

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
