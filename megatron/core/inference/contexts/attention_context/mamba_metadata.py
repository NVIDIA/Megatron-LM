# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions


@dataclass
class MambaInferenceStateConfig:
    """Config for initializing Mamba model inference state tensors."""

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

    def __init__(self, max_requests: int):
        """
        Initializes the Mamba slot allocator.

        Args:
            max_requests (int): The maximum number of concurrent requests.
        """
        self.max_requests = max_requests

        # Metadata for mapping requests to slots in the static Mamba state buffer
        self.request_to_mamba_state_idx = torch.full(
            (self.max_requests,), -1, dtype=torch.int32, device=torch.cuda.current_device()
        )
        # Copy of the request to slot mapping to use fo the step. Includes padding tokens.
        self.request_to_mamba_state_idx_for_step = torch.full(
            (self.max_requests,), -1, dtype=torch.int32, device=torch.cuda.current_device()
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
        self.request_to_mamba_state_idx_for_step.fill_(-1)

        # Re-initialize the free slot pool
        self.mamba_state_free_slots = torch.arange(
            self.max_requests, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.mamba_state_free_slot_count = self.max_requests

    def update(
        self,
        active_mamba_indices: torch.Tensor,
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
        # TODO: Support chunked prefill in InferenceBatchDimensions

        real_decode_count = batch_dimensions.decode_req_count
        real_prefill_count = batch_dimensions.prefill_req_count

        padded_decode_count = padded_batch_dimensions.decode_req_count

        # Copy real decode indices to the start of the buffer.
        if real_decode_count > 0:
            self.request_to_mamba_state_idx_for_step[:real_decode_count] = active_mamba_indices[
                :real_decode_count
            ]

        # Pad the rest of the decode section with -1
        if padded_decode_count > real_decode_count:
            self.request_to_mamba_state_idx_for_step[real_decode_count:padded_decode_count].fill_(
                -1
            )

        # Copy real prefill indices after the decode indices.
        if real_prefill_count > 0:
            self.request_to_mamba_state_idx_for_step[
                padded_decode_count : padded_decode_count + real_prefill_count
            ] = active_mamba_indices[real_decode_count : real_decode_count + real_prefill_count]

        # Pad the rest of the prefill section (if any remaining space in buffer)
        total_used = padded_decode_count + real_prefill_count
        if total_used < self.max_requests:
            self.request_to_mamba_state_idx_for_step[total_used:].fill_(-1)

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
