# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.contexts.mamba_slot_allocator import (
    MAX_INTERMEDIATE_OFFSETS_PER_REQUEST,
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

        # Map from requests to slots in the static Mamba state buffer (CPU for bookkeeping).
        self.request_to_mamba_state_idx = torch.full(
            (self.max_requests,), -1, dtype=torch.int32, device='cpu',
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

        # Allocator for Mamba state slots (CPU for bookkeeping).
        self.mamba_state_free_slots = torch.arange(
            self.max_requests, dtype=torch.int32, device='cpu',
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

        self.reset_varlen_metadata()

    def reset(self) -> None:
        """
        Resets all Mamba states and frees all allocated slots.
        """
        self.request_to_mamba_state_idx.fill_(-1)

        self.reset_varlen_metadata()

        # Re-initialize the free slot pool
        self.mamba_state_free_slots = torch.arange(
            self.max_requests, dtype=torch.int32, device='cpu',
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
                seq_len = end - start
                n_chunks = max(1, (seq_len + chunk_size - 1) // chunk_size)
                boundaries = [min(start + (k + 1) * chunk_size, end) for k in range(n_chunks)]
                chunk_boundaries.extend(boundaries)
                chunk_to_seq_list.extend([i] * n_chunks)
                last_chunk_idx_list.append(len(chunk_boundaries) - 2)

            # Pad to fixed size for CUDA graph compatibility
            padded_max_chunks = padded_token_count // chunk_size + padded_prefill_count
            last_boundary = chunk_boundaries[-1]
            pad_b = padded_max_chunks + 1 - len(chunk_boundaries)
            if pad_b > 0:
                chunk_boundaries.extend([last_boundary] * pad_b)
            pad_s = padded_max_chunks - len(chunk_to_seq_list)
            if pad_s > 0:
                chunk_to_seq_list.extend([0] * pad_s)

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
                cu = self._cu_seqlens_buffer[: real_prefill_count + 1]
                lengths = (cu[1:] - cu[:-1]).to(torch.int64)
                seq_indices = torch.arange(
                    real_prefill_count, dtype=torch.int32, device=self.device
                )
                seq_starts = cu[:real_prefill_count].to(torch.int32)
                self._conv_seq_idx_buffer[:real_tokens] = torch.repeat_interleave(
                    seq_indices, lengths
                )
                self._conv_seq_start_buffer[:real_tokens] = torch.repeat_interleave(
                    seq_starts, lengths
                )
            if padded_token_count > real_tokens:
                self._conv_seq_idx_buffer[real_tokens:padded_token_count] = 0
                self._conv_seq_start_buffer[real_tokens:padded_token_count] = 0

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
            # Transfer counts to CPU (single sync) for per_request_counts and total check
            counts_list = intermediate_counts_gpu.tolist()
            total = sum(counts_list)

            if total > 0:
                # Compute cumulative chunk counts from cu_seqlens (already on GPU)
                cu = self._cu_seqlens_buffer[: real_prefill_count + 1]
                seq_lens = (cu[1 : real_prefill_count + 1] - cu[:real_prefill_count]).to(
                    torch.int64
                )
                num_chunks = torch.clamp((seq_lens + chunk_size - 1) // chunk_size, min=1)
                cum_chunks = torch.zeros(
                    real_prefill_count + 1, dtype=torch.int64, device=self.device
                )
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
                self.per_request_intermediate_counts = counts_list
            else:
                # All counts are 0
                self._intermediate_chunk_indices_buffer.fill_(0)
                self._intermediate_abs_positions_buffer.fill_(self.d_conv)
                self.intermediate_count = 0
                self.per_request_intermediate_counts = counts_list

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

    def compute_cpu_metadata(
        self,
        active_mamba_indices: torch.Tensor,
        token_to_request_idx: torch.Tensor,
        cpu_cu_query: torch.Tensor,
        batch_dimensions: InferenceBatchDimensions,
        padded_batch_dimensions: InferenceBatchDimensions,
        enable_chunked_prefill: bool,
        intermediate_offsets_gpu: Optional[torch.Tensor] = None,
        intermediate_counts_gpu: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute all Mamba metadata on CPU. Returns dict for load_from_cpu().

        This performs the same computation as update() but entirely on CPU,
        producing CPU tensors that are later copied to GPU buffers.

        Args:
            active_mamba_indices: CPU tensor of Mamba slot indices for active requests.
            token_to_request_idx: CPU tensor mapping tokens to request indices.
            cpu_cu_query: CPU cumulative query lengths from MHA metadata computation.
            batch_dimensions: Dimensions of the current batch.
            padded_batch_dimensions: Dimensions of the padded batch.
            enable_chunked_prefill: Whether chunked prefill is enabled.
            intermediate_offsets_gpu: GPU tensor of per-request intermediate offsets, or None.
            intermediate_counts_gpu: GPU tensor of per-request intermediate counts, or None.
        """
        real_decode_count = batch_dimensions.decode_req_count
        real_prefill_count = batch_dimensions.prefill_req_count
        padded_decode_count = padded_batch_dimensions.decode_req_count
        padded_prefill_count = padded_batch_dimensions.prefill_req_count
        padded_token_count = padded_batch_dimensions.token_count
        chunk_size = self.mamba_chunk_size

        result = {
            "padded_decode_count": padded_decode_count,
            "padded_prefill_count": padded_prefill_count,
            "padded_token_count": padded_token_count,
            "real_decode_count": real_decode_count,
            "real_prefill_count": real_prefill_count,
        }

        # Decode batch indices.
        if padded_decode_count > 0:
            cpu_decode = torch.full((padded_decode_count,), -1, dtype=torch.int32)
            cpu_decode[:real_decode_count] = active_mamba_indices[:real_decode_count]
            result["batch_indices_decode"] = cpu_decode

        # Prefill batch indices.
        if padded_prefill_count > 0:
            cpu_prefill = torch.full((padded_prefill_count,), -1, dtype=torch.int32)
            if real_prefill_count > 0:
                start = real_decode_count
                cpu_prefill[:real_prefill_count] = active_mamba_indices[
                    start : start + real_prefill_count
                ]
            result["batch_indices_prefill"] = cpu_prefill

            # seq_idx: normalized token-to-request mapping for prefill tokens.
            prefill_start_req = real_decode_count
            end_prefill_req = real_decode_count + real_prefill_count
            start_token = cpu_cu_query[prefill_start_req].item()
            end_token = cpu_cu_query[end_prefill_req].item()
            seq_len = end_token - start_token

            cpu_seq_idx = torch.full((padded_token_count,), -1, dtype=torch.int32)
            if seq_len > 0:
                raw = token_to_request_idx[start_token:end_token]
                cpu_seq_idx[:seq_len] = raw - raw[0]
            result["seq_idx"] = cpu_seq_idx
            result["seq_len"] = seq_len

            # cu_seqlens for prefill.
            cpu_cu_seqlens = torch.zeros(padded_prefill_count + 1, dtype=torch.int32)
            if real_prefill_count > 0:
                cpu_cu_seqlens[1 : real_prefill_count + 1] = (
                    cpu_cu_query[prefill_start_req + 1 : end_prefill_req + 1]
                    - cpu_cu_query[prefill_start_req]
                )
            if real_prefill_count < padded_prefill_count:
                last_val = cpu_cu_seqlens[real_prefill_count].item()
                cpu_cu_seqlens[real_prefill_count + 1 :] = last_val
            result["cu_seqlens"] = cpu_cu_seqlens

            cu_seqlens_list = cpu_cu_seqlens[: real_prefill_count + 1].tolist()
            real_prefill_tokens = cu_seqlens_list[real_prefill_count] if real_prefill_count > 0 else 0
            result["cu_seqlens_list"] = cu_seqlens_list
            result["real_prefill_token_count"] = real_prefill_tokens

            # Chunk metadata (Python loop, pure CPU).
            cu_seqlens_all = cpu_cu_seqlens[: padded_prefill_count + 1].tolist()
            chunk_boundaries = [0]
            last_chunk_idx_list = []
            chunk_to_seq_list = []

            for i in range(padded_prefill_count):
                start = cu_seqlens_all[i]
                end = cu_seqlens_all[i + 1]
                s_len = end - start
                n_chunks = max(1, (s_len + chunk_size - 1) // chunk_size)
                boundaries = [min(start + (k + 1) * chunk_size, end) for k in range(n_chunks)]
                chunk_boundaries.extend(boundaries)
                chunk_to_seq_list.extend([i] * n_chunks)
                last_chunk_idx_list.append(len(chunk_boundaries) - 2)

            padded_max_chunks = padded_token_count // chunk_size + padded_prefill_count
            last_boundary = chunk_boundaries[-1]
            pad_b = padded_max_chunks + 1 - len(chunk_boundaries)
            if pad_b > 0:
                chunk_boundaries.extend([last_boundary] * pad_b)
            pad_s = padded_max_chunks - len(chunk_to_seq_list)
            if pad_s > 0:
                chunk_to_seq_list.extend([0] * pad_s)

            n_cu = padded_max_chunks + 1
            result["cu_chunk_seqlens"] = torch.tensor(
                chunk_boundaries[:n_cu], dtype=torch.int32,
            )
            result["last_chunk_indices"] = torch.tensor(
                last_chunk_idx_list, dtype=torch.int32,
            )
            result["seq_idx_for_varlen"] = torch.tensor(
                chunk_to_seq_list[:padded_max_chunks], dtype=torch.int32,
            )
            result["padded_max_chunks"] = padded_max_chunks

            # Conv1d per-token metadata (CPU repeat_interleave).
            if real_prefill_tokens > 0:
                cu_t = cpu_cu_seqlens[: real_prefill_count + 1]
                lengths = (cu_t[1:] - cu_t[:-1]).to(torch.int64)
                seq_indices = torch.arange(real_prefill_count, dtype=torch.int32)
                seq_starts = cu_t[:real_prefill_count].to(torch.int32)
                cpu_conv_seq_idx = torch.zeros(padded_token_count, dtype=torch.int32)
                cpu_conv_seq_start = torch.zeros(padded_token_count, dtype=torch.int32)
                cpu_conv_seq_idx[:real_prefill_tokens] = torch.repeat_interleave(
                    seq_indices, lengths,
                )
                cpu_conv_seq_start[:real_prefill_tokens] = torch.repeat_interleave(
                    seq_starts, lengths,
                )
            else:
                cpu_conv_seq_idx = torch.zeros(padded_token_count, dtype=torch.int32)
                cpu_conv_seq_start = torch.zeros(padded_token_count, dtype=torch.int32)
            result["conv_seq_idx"] = cpu_conv_seq_idx
            result["conv_seq_start"] = cpu_conv_seq_start

            # Intermediate metadata (needs GPU data -- defer to load_from_cpu).
            result["intermediate_offsets_gpu"] = intermediate_offsets_gpu
            result["intermediate_counts_gpu"] = intermediate_counts_gpu

        # device_decode_prefill scalars.
        if padded_decode_count > 0 and padded_prefill_count > 0:
            result["decode_prefill_0"] = cpu_cu_query[real_decode_count].item()
            result["decode_prefill_1"] = (
                cpu_cu_query[real_decode_count + real_prefill_count].item()
                - cpu_cu_query[real_decode_count].item()
            )

        return result

    def load_from_cpu(self, d: dict) -> None:
        """Copy pre-computed CPU metadata into GPU buffers.

        Args:
            d: Dict returned by compute_cpu_metadata().
        """
        padded_decode_count = d["padded_decode_count"]
        padded_prefill_count = d["padded_prefill_count"]
        padded_token_count = d["padded_token_count"]
        real_prefill_count = d["real_prefill_count"]

        if padded_decode_count > 0:
            self._batch_indices_decode_buffer[:padded_decode_count].copy_(
                d["batch_indices_decode"], non_blocking=True,
            )
            self.batch_indices_decode = self._batch_indices_decode_buffer[:padded_decode_count]

        if padded_prefill_count > 0:
            self._batch_indices_prefill_buffer[:padded_prefill_count].copy_(
                d["batch_indices_prefill"], non_blocking=True,
            )
            self.batch_indices_prefill = self._batch_indices_prefill_buffer[:padded_prefill_count]

            seq_len = d["seq_len"]
            self._seq_idx_buffer[:, :padded_token_count].copy_(
                d["seq_idx"][:padded_token_count].unsqueeze(0), non_blocking=True,
            )
            self.seq_idx = self._seq_idx_buffer[:, :padded_token_count]

            self._cu_seqlens_buffer[: padded_prefill_count + 1].copy_(
                d["cu_seqlens"], non_blocking=True,
            )
            self.cu_seqlens = self._cu_seqlens_buffer[: padded_prefill_count + 1]
            self.cu_seqlens_list = d["cu_seqlens_list"]
            self.real_prefill_token_count = d["real_prefill_token_count"]

            padded_max_chunks = d["padded_max_chunks"]
            n_cu = padded_max_chunks + 1
            self._cu_chunk_seqlens_buffer[:n_cu].copy_(
                d["cu_chunk_seqlens"], non_blocking=True,
            )
            self.cu_chunk_seqlens = self._cu_chunk_seqlens_buffer[:n_cu]

            self._last_chunk_indices_buffer[:padded_prefill_count].copy_(
                d["last_chunk_indices"], non_blocking=True,
            )
            self.last_chunk_indices = self._last_chunk_indices_buffer[:padded_prefill_count]

            self._seq_idx_for_varlen_buffer[:padded_max_chunks].copy_(
                d["seq_idx_for_varlen"], non_blocking=True,
            )
            self.seq_idx_for_varlen = self._seq_idx_for_varlen_buffer[:padded_max_chunks]

            self._conv_seq_idx_buffer[:padded_token_count].copy_(
                d["conv_seq_idx"][:padded_token_count], non_blocking=True,
            )
            self.conv_seq_idx = self._conv_seq_idx_buffer[:padded_token_count]

            self._conv_seq_start_buffer[:padded_token_count].copy_(
                d["conv_seq_start"][:padded_token_count], non_blocking=True,
            )
            self.conv_seq_start = self._conv_seq_start_buffer[:padded_token_count]

            # Intermediate metadata still requires GPU computation.
            self._update_intermediate_metadata(
                d["intermediate_offsets_gpu"],
                d["intermediate_counts_gpu"],
                real_prefill_count,
            )

        if padded_decode_count > 0 and padded_prefill_count > 0:
            self._device_decode_prefill_buffer[0] = d["decode_prefill_0"]
            self._device_decode_prefill_buffer[1] = d["decode_prefill_1"]
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
