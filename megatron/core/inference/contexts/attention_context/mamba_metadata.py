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
        #   chunked prefill sequence length,
        #   total regular prefill sequence length
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
                enable_chunked_prefill (bool): Whether chunked prefill is enabled
        """
        real_decode_count = batch_dimensions.decode_req_count
        real_prefill_count = batch_dimensions.prefill_req_count

        padded_decode_count = padded_batch_dimensions.decode_req_count
        padded_prefill_count = padded_batch_dimensions.prefill_req_count
        padded_token_count = padded_batch_dimensions.token_count

        has_chunked_prefill_req = enable_chunked_prefill and real_prefill_count > 0

        # Although the context ensures that the last request is always the designated
        # chunked prefill request, what we actually care about is ensuring that any
        # prefill request with non-zero initial states is executed through the
        # chunked prefill path.
        #
        # In the batch arrangement passed to this update function, the logic assumes
        # the *first* prefill request is the one carrying states.
        #
        # There are three scenarios:
        #
        # Scenario A: No prefill request has initial states yet, but the last request
        #             is the designated chunked prefill request (starting a new chunk).
        #
        #   [ ... Decode Requests ... ] [ Prefill (start) ]
        #                               ^
        #                               |--- First prefill request
        #                                    Treated as having states.
        #                                    Harmless because actual initial states are 0.
        #
        # Scenario B: There is exactly 1 prefill request which is a continuing
        #             chunked prefill request with non-zero initial states.
        #
        #   [ ... Decode Requests ... ] [ Prefill (cont)  ]
        #                               ^
        #                               |--- First prefill request
        #                                    Has non-zero initial states.
        #
        # Scenario C: There is a leftover chunked prefill request that is executing
        #             its last chunk, followed by additional prefill requests.
        #
        #   [ ... Decode Requests ... ] [ Prefill (end)   ] [ Prefill (new) ] ...
        #                               ^
        #                               |--- First prefill request
        #                                    Has non-zero initial states.
        #
        # The implementation generalizes to Scenario A as well, where the first prefill
        # request is treated as if it has non-zero initial states, which is safe.
        # While this results in a minor inefficiency f there is no continuing chunked prefill
        # request in a given batch, this case is infrequent.

        if padded_decode_count > 0:
            # Update decode indices
            self._batch_indices_decode_buffer[:real_decode_count].copy_(
                active_mamba_indices[:real_decode_count]
            )
            if padded_decode_count > real_decode_count:
                self._batch_indices_decode_buffer[real_decode_count:padded_decode_count] = -1
            self.batch_indices_decode = self._batch_indices_decode_buffer[:padded_decode_count]

        # Determine if we have a chunked prefill request and adjust counts for regular prefill
        regular_prefill_count = real_prefill_count
        chunked_req_idx = -1

        if has_chunked_prefill_req:
            # The first prefill request is the chunked one
            regular_prefill_count -= 1
            chunked_req_idx = real_decode_count

            # Update chunked prefill indices
            self._batch_indices_chunked_prefill_buffer[0] = active_mamba_indices[chunked_req_idx]
            self.batch_indices_chunked_prefill = self._batch_indices_chunked_prefill_buffer
        else:
            self.batch_indices_chunked_prefill = None

        if padded_prefill_count > 0:
            # Update prefill indices (excluding chunked prefill from regular prefill buffer)
            if regular_prefill_count > 0:
                # If chunked prefill exists, regular prefills start after it.
                # If no chunked prefill, regular prefills start at real_decode_count.
                start_idx = real_decode_count + (1 if has_chunked_prefill_req else 0)

                self._batch_indices_prefill_buffer[:regular_prefill_count].copy_(
                    active_mamba_indices[start_idx : start_idx + regular_prefill_count]
                )

            if padded_prefill_count > regular_prefill_count:
                self._batch_indices_prefill_buffer[regular_prefill_count:padded_prefill_count] = -1

            self.batch_indices_prefill = self._batch_indices_prefill_buffer[:padded_prefill_count]

            # Update seq_idx for regular prefills
            # If chunked prefill exists, we need to skip its tokens in seq_idx

            # Index where regular prefills end in the batch (decode + chunked + regular)
            end_regular_prefill_req_idx = (
                real_decode_count + regular_prefill_count + (1 if has_chunked_prefill_req else 0)
            )
            end_regular_prefill_token_idx = cu_seqlens[end_regular_prefill_req_idx]

            # Index where regular prefills start
            start_regular_prefill_req_idx = real_decode_count + (
                1 if has_chunked_prefill_req else 0
            )
            start_regular_prefill_token_idx = cu_seqlens[start_regular_prefill_req_idx]

            # The length of tokens belonging to regular prefill requests
            seq_len = end_regular_prefill_token_idx - start_regular_prefill_token_idx

            if seq_len > 0:
                # We subtract start_regular_prefill_req_idx to normalize request IDs to
                # 0-based relative to this buffer
                self._seq_idx_buffer[:, :seq_len].copy_(
                    token_to_request_idx[
                        start_regular_prefill_token_idx:end_regular_prefill_token_idx
                    ]
                    - start_regular_prefill_req_idx
                )

            if padded_token_count > seq_len:
                self._seq_idx_buffer[:, seq_len:padded_token_count] = -1
            self.seq_idx = self._seq_idx_buffer[:, :padded_token_count]

            # Update cu_seqlens for regular prefill requests
            self._cu_seqlens_buffer[0] = 0
            if regular_prefill_count > 0:
                # Copy cu_seqlens for regular prefill requests and normalize by
                # subtracting the start token index
                start_req_idx = real_decode_count + (1 if has_chunked_prefill_req else 0)
                end_req_idx = start_req_idx + regular_prefill_count

                self._cu_seqlens_buffer[1 : regular_prefill_count + 1].copy_(
                    cu_seqlens[start_req_idx + 1 : end_req_idx + 1] - cu_seqlens[start_req_idx]
                )

            # Pad the rest with the last value (effectively length 0 segments)
            last_val = self._cu_seqlens_buffer[regular_prefill_count]
            self._cu_seqlens_buffer[regular_prefill_count + 1 : padded_prefill_count + 1].fill_(
                last_val
            )
            self.cu_seqlens = self._cu_seqlens_buffer[: padded_prefill_count + 1]

        if padded_decode_count > 0 and padded_prefill_count > 0:
            self._device_decode_prefill_buffer[0] = real_decode_count
            # This describes the number of items in the prefill tensor relative to the
            # decode tensor. If chunked prefill is present, it is included in the
            # "prefill" part of the main split.
            self._device_decode_prefill_buffer[1] = regular_prefill_count + (
                1 if has_chunked_prefill_req else 0
            )
            self.device_decode_prefill = self._device_decode_prefill_buffer

        # If using chunked prefill for this batch, store the number of chunked tokens
        # and the number of regular prefill tokens
        if has_chunked_prefill_req:
            # Chunked request is the first prefill request (index real_decode_count)
            chunked_prefill_token_count = (
                cu_seqlens[real_decode_count + 1] - cu_seqlens[real_decode_count]
            )

            # Regular prefill tokens are everything after the chunked request tokens
            regular_prefill_token_count = 0
            if regular_prefill_count > 0:
                regular_prefill_token_count = (
                    cu_seqlens[real_decode_count + 1 + regular_prefill_count]
                    - cu_seqlens[real_decode_count + 1]
                )

            self._device_chunked_prefill_buffer[0] = chunked_prefill_token_count
            self._device_chunked_prefill_buffer[1] = regular_prefill_token_count
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
