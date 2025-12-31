# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions


@dataclass
class MambaInferenceStateConfig:
    """
    Config for initializing Mamba model inference state tensors.

    Note that we maintain separate metadata for decode, regular prefill, and
    chunked prefill requests because the Mamba kernels do not yet support mixing
    these. Once the kernels have been updated we can simplify this code.
    """

    layer_type_list: List[str]
    """
    A list of strings that indicates the layer type (Mamba / Attention / MLP) for each layer.
    See `megatron/core/ssm/mamba_hybrid_layer_allocation.py` for the list of symbols.
    """

    mamba_conv_states_shape: Tuple[int]
    """Mamba conv states shape per request."""

    mamba_ssm_states_shape: Tuple[int]
    """Mamba ssm states shape per request."""


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
        self._batch_indices_prefill_buffer = torch.full(
            (self.max_requests,), -1, dtype=torch.int32, device=self.device
        )

        # Map from the active chunked prefill request to its slot in the static Mamba state buffer
        self._batch_indices_chunked_prefill_buffer = torch.full(
            (1,), -1, dtype=torch.int32, device=self.device
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
        #   total prefill sequence length excluding chunked prefill,
        #   chunked prefill sequence length
        # )
        self._device_chunked_prefill_buffer = torch.zeros(
            (2,), dtype=torch.int32, device=self.device
        )

        # Allocator for Mamba state slots
        self.mamba_state_free_slots = torch.arange(
            self.max_requests, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.mamba_state_free_slot_count = self.max_requests

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
        self.batch_indices_chunked_prefill = None
        self.cu_seqlens = None
        self.seq_idx = None
        self.device_decode_prefill = None
        self.device_chunked_prefill = None

    def update(
        self,
        active_mamba_indices: torch.Tensor,
        token_to_request_idx: torch.Tensor,
        cu_seqlens: torch.Tensor,
        batch_dimensions: InferenceBatchDimensions,
        padded_batch_dimensions: InferenceBatchDimensions,
    ) -> None:
        """
        Updates the dedicated CUDA graph mapping tensor with the indices
        of currently active requests.

        Args:
            active_mamba_indices (Tensor): Tensor containing the Mamba slot indices
                                           for active requests.
            num_active_requests (int): The number of active requests.
        """
        real_decode_count = batch_dimensions.decode_req_count
        real_prefill_count = batch_dimensions.prefill_req_count
        real_token_count = batch_dimensions.token_count
        has_explicit_chunked_prefill_req = batch_dimensions.has_explicit_chunked_prefill_req

        padded_decode_count = padded_batch_dimensions.decode_req_count
        padded_prefill_count = padded_batch_dimensions.prefill_req_count
        padded_token_count = padded_batch_dimensions.token_count
        assert (
            has_explicit_chunked_prefill_req
            == padded_batch_dimensions.has_explicit_chunked_prefill_req
        )

        if padded_decode_count > 0:
            # Update decode indices
            self._batch_indices_decode_buffer[:real_decode_count].copy_(
                active_mamba_indices[:real_decode_count]
            )
            if padded_decode_count > real_decode_count:
                self._batch_indices_decode_buffer[real_decode_count:padded_decode_count].fill_(-1)
            self.batch_indices_decode = self._batch_indices_decode_buffer[:padded_decode_count]

        # Determine if we have a chunked prefill request and adjust counts for regular prefill
        regular_prefill_count = real_prefill_count
        if has_explicit_chunked_prefill_req:
            # The last prefill request is the chunked one
            regular_prefill_count -= 1
            chunked_req_idx = real_decode_count + regular_prefill_count

            # Update chunked prefill indices
            self._batch_indices_chunked_prefill_buffer[0] = active_mamba_indices[chunked_req_idx]
            self.batch_indices_chunked_prefill = self._batch_indices_chunked_prefill_buffer
        else:
            self.batch_indices_chunked_prefill = None

        if padded_prefill_count > 0:
            # Update prefill indices (excluding chunked prefill from regular prefill buffer)
            if regular_prefill_count > 0:
                self._batch_indices_prefill_buffer[:regular_prefill_count].copy_(
                    active_mamba_indices[
                        real_decode_count : real_decode_count + regular_prefill_count
                    ]
                )

            if padded_prefill_count > regular_prefill_count:
                self._batch_indices_prefill_buffer[
                    regular_prefill_count:padded_prefill_count
                ].fill_(-1)

            self.batch_indices_prefill = self._batch_indices_prefill_buffer[:padded_prefill_count]

            # Update seq_idx
            end_regular_prefill_token_idx = cu_seqlens[real_decode_count + regular_prefill_count]

            # The length of tokens belonging to regular prefill requests (excluding decode tokens)
            seq_len = end_regular_prefill_token_idx - real_decode_count

            if seq_len > 0:
                self._seq_idx_buffer[:, :seq_len].copy_(
                    token_to_request_idx[real_decode_count:end_regular_prefill_token_idx]
                    - real_decode_count
                )

            if padded_token_count > seq_len:
                self._seq_idx_buffer[:, seq_len:padded_token_count].fill_(-1)
            self.seq_idx = self._seq_idx_buffer[:, :padded_token_count]

            # Update cu_seqlens
            self._cu_seqlens_buffer[0] = 0
            if regular_prefill_count > 0:
                self._cu_seqlens_buffer[1 : regular_prefill_count + 1].copy_(
                    cu_seqlens[
                        real_decode_count + 1 : real_decode_count + regular_prefill_count + 1
                    ]
                    - real_decode_count
                )

            # Pad the rest with the last value (effectively length 0 segments)
            last_val = self._cu_seqlens_buffer[regular_prefill_count]
            self._cu_seqlens_buffer[regular_prefill_count + 1 : padded_prefill_count + 1].fill_(
                last_val
            )
            self.cu_seqlens = self._cu_seqlens_buffer[: padded_prefill_count + 1]

        if padded_decode_count > 0 and padded_prefill_count > 0:
            self._device_decode_prefill_buffer[0] = real_decode_count
            self._device_decode_prefill_buffer[1] = regular_prefill_count
            self.device_decode_prefill = self._device_decode_prefill_buffer

        # If using chunked prefill for this batch, store the number of regular prefill tokens
        # and the number of tokens in the chunked prefill request
        if has_explicit_chunked_prefill_req:
            chunked_prefill_token_count = (
                cu_seqlens[real_decode_count + real_prefill_count]
                - cu_seqlens[real_decode_count + real_prefill_count - 1]
            )
            assert self.cu_seqlens is not None
            self._device_chunked_prefill_buffer[0] = self.cu_seqlens[regular_prefill_count]
            self._device_chunked_prefill_buffer[1] = chunked_prefill_token_count
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
