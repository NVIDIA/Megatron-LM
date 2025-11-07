# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Tuple

import torch
from torch import Tensor


class MambaMetadata:
    """
    Manages the state tensors required for Mamba layers during inference.

    This class allocates and manages the buffers for Mamba's convolution (conv)
    and state space model (SSM) states. It also handles the allocation of
    "slots" for each request, as Mamba states are persistent per-request,
    unlike the per-token KV cache in attention.
    """

    def __init__(
        self,
        *,
        num_mamba_layers: int,
        max_requests: int,
        mamba_conv_states_shape: Tuple[int],
        mamba_ssm_states_shape: Tuple[int],
        params_dtype: torch.dtype,
        device: torch.device,
        ctx_manager,
    ):
        """
        Initializes the Mamba state buffers and slot allocator.

        Args:
            num_mamba_layers (int): Number of Mamba layers in the model.
            max_requests (int): The maximum number of concurrent requests.
            mamba_conv_states_shape (Tuple[int]): Shape of the conv state for one layer, one request.
            mamba_ssm_states_shape (Tuple[int]): Shape of the ssm state for one layer, one request.
            params_dtype (torch.dtype): Data type for the state tensors.
            device (torch.device): The CUDA device to allocate tensors on.
            ctx_manager: Context manager for tensor allocation (e.g., for unified memory).
        """
        self.max_requests = max_requests
        self.device = device

        with ctx_manager:
            # Main state buffers
            self.mamba_conv_states = torch.zeros(
                (num_mamba_layers, max_requests) + mamba_conv_states_shape,
                dtype=params_dtype,
                device=device,
            )
            self.mamba_ssm_states = torch.zeros(
                (num_mamba_layers, max_requests) + mamba_ssm_states_shape,
                dtype=params_dtype,
                device=device,
            )

            # Metadata for mapping requests to slots in the static Mamba state buffer
            self.request_to_mamba_state_idx = torch.full(
                (max_requests,), -1, dtype=torch.int32, device=device
            )

            # Separate mapping used only for CUDA graph compatibility
            self.request_to_mamba_state_idx_cudagraph_only = torch.full(
                (max_requests,), -1, dtype=torch.int32, device=device
            )

            # Allocator for Mamba state slots
            self.mamba_state_free_slots = torch.arange(
                max_requests, dtype=torch.int32, device=device
            )
            self.mamba_state_free_slot_count = max_requests

    def reset(self) -> None:
        """
        Resets all Mamba states and frees all allocated slots.
        """
        self.mamba_conv_states.fill_(0)
        self.mamba_ssm_states.fill_(0)
        self.request_to_mamba_state_idx.fill_(-1)
        self.request_to_mamba_state_idx_cudagraph_only.fill_(-1)

        # Re-initialize the free slot pool
        self.mamba_state_free_slots = torch.arange(
            self.max_requests, dtype=torch.int32, device=self.device
        )
        self.mamba_state_free_slot_count = self.max_requests

    def reset_cudagraph_mapping(self) -> None:
        """
        Resets only the CUDA graph mapping tensor.
        """
        self.request_to_mamba_state_idx_cudagraph_only.fill_(-1)

    def update_cudagraph_mapping(
        self, active_mamba_indices: Tensor, num_active_requests: int
    ) -> None:
        """
        Updates the dedicated CUDA graph mapping tensor with the indices
        of currently active requests.

        Args:
            active_mamba_indices (Tensor): Tensor containing the Mamba slot indices
                                           for active requests.
            num_active_requests (int): The number of active requests.
        """
        self.request_to_mamba_state_idx_cudagraph_only[
            0:num_active_requests
        ] = active_mamba_indices

    def allocate_slot(self) -> int:
        """
        Allocates a new slot for a request in the Mamba state buffers.

        Also zeroes out the state for the allocated slot.

        Returns:
            int: The index of the allocated slot.
            Returns None if no slots are available.
        """
        if self.mamba_state_free_slot_count == 0:
            return None

        # Get a free slot
        self.mamba_state_free_slot_count -= 1
        mamba_idx = self.mamba_state_free_slots[self.mamba_state_free_slot_count]

        # Initialize the allocated Mamba state
        self.mamba_conv_states[:, mamba_idx] = 0.0
        self.mamba_ssm_states[:, mamba_idx] = 0.0

        return mamba_idx

    def free_slots(self, request_indices: Tensor) -> None:
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
