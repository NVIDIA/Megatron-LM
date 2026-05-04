# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional, Tuple

import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.ssm.ops.gdp.common import CHUNK_SIZE as GDP_CHUNK_SIZE
from megatron.core.ssm.ops.gdp.metadata import build_gdp_chunk_descriptors, max_gdp_chunk_counts

# Maximum intermediate state extraction offsets per request. The 3 candidates
# are: KV divergence boundary, last block-aligned boundary, and penultimate
# block boundary (see PrefixCachedMambaMetadata.compute_and_store_offsets).
MAX_INTERMEDIATE_OFFSETS_PER_REQUEST = 3


class MambaMetadata:
    """Manages the metadata tensors required for Mamba layers during inference."""

    def __init__(
        self,
        max_requests: int,
        max_tokens: int,
        *,
        mamba_chunk_size: int = 128,
        d_conv: int = 0,
        decode_indices_dtype: torch.dtype = torch.int64,
        gdp_num_householder: int = 0,
    ):
        """
        Initializes the Mamba metadata.

        Args:
            max_requests (int): The maximum number of concurrent requests.
            max_tokens (int): The maximum number of tokens.
            mamba_chunk_size (int): The chunk size used by the Mamba SSM Triton kernels.
            d_conv (int): Convolution window size (from mamba_conv_states_shape[-1]).
                Used for vectorized conv state extraction at intermediate offsets.
            decode_indices_dtype (torch.dtype): Dtype for decode state-slot indices.
            gdp_num_householder (int): Number of Householder copies of the Gated
                Delta Product layers, or 0 if the model has none. When non-zero,
                the GDP chunk descriptors are allocated and maintained alongside
                the Mamba ones. They are kept separate rather than derived from
                the Mamba chunk metadata: the forked GDP kernels chunk at a fixed
                64 tokens (independent of `mamba_chunk_size`), index chunks as
                `(sequence, chunk-within-sequence)` pairs rather than by token
                boundary, and additionally need a chunking of the Householder-
                expanded stream, which no Mamba2 buffer describes.
        """
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.mamba_chunk_size = mamba_chunk_size
        self.d_conv = d_conv
        self.device = torch.cuda.current_device()
        assert decode_indices_dtype in (torch.int32, torch.int64)
        self.decode_indices_dtype = decode_indices_dtype

        # Maximum possible chunks across all batch configurations
        self.max_chunks = max_tokens // mamba_chunk_size + max_requests

        # Map from requests to slots in the static Mamba state buffer (CPU for bookkeeping).
        self.request_to_mamba_state_idx = torch.full(
            (self.max_requests,), -1, dtype=torch.int32, device='cpu'
        )

        # Map from requests to slots in the static Mamba state buffer for active decode requests.
        # Non-BIK decode uses int64 for selective_state_update; BIK uses int32
        # for the exact causal-conv1d update kernel.
        self._batch_indices_decode_buffer = torch.full(
            (self.max_requests,), -1, dtype=self.decode_indices_dtype, device=self.device
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

        # Gated Delta Product chunk descriptors (see the constructor docstring).
        self.gdp_num_householder = gdp_num_householder
        if gdp_num_householder > 0:
            self.max_gdp_chunks, self.max_gdp_chunks_dp = max_gdp_chunk_counts(
                max_tokens, max_requests, gdp_num_householder
            )
            self._gdp_chunk_indices_buffer = torch.zeros(
                (self.max_gdp_chunks, 2), dtype=torch.int32, device=self.device
            )
            self._gdp_chunk_indices_dp_buffer = torch.zeros(
                (self.max_gdp_chunks_dp, 2), dtype=torch.int32, device=self.device
            )
            self._gdp_chunk_offsets_buffer = torch.zeros(
                max_requests + 1, dtype=torch.int32, device=self.device
            )

        # Conv1d per-token metadata (request ID and request start position)
        self._conv_seq_idx_buffer = torch.zeros(max_tokens, dtype=torch.int32, device=self.device)
        self._conv_seq_start_buffer = torch.zeros(max_tokens, dtype=torch.int32, device=self.device)

        # Allocator for Mamba state slots (CPU for bookkeeping).
        self.mamba_state_free_slots = torch.arange(
            self.max_requests, dtype=torch.int32, device='cpu'
        )
        self.mamba_state_free_slot_count = self.max_requests

        # Intermediate-state extraction (prefix caching) lives on `PrefixCachedMambaMetadata`.
        self.intermediate_ssm_out: Optional[torch.Tensor] = None
        self.intermediate_conv_out: Optional[torch.Tensor] = None

        # Coalesced production path: pinned CPU views + shared GPU views bound
        # by DynamicInferenceContext so that the per-step Mamba metadata fields
        # ride along with the single coalesced H2D in transfer_bookkeeping_to_gpu.
        # The legacy update() path above keeps using the standalone _*_buffer
        # tensors (exercised only by unit tests that construct MambaMetadata
        # without a context).
        self._cpu_bufs = None
        self._gpu_view = None

        self.reset_varlen_metadata()

    def bind_cpu_buffers(self, bufs: dict) -> None:
        """Attach pinned CPU views from DynamicInferenceContext._cpu_bookkeeping_buf.

        ``bufs`` maps field names to 1D (or (1, max_tokens) for ``seq_idx``)
        pinned CPU views that compute_cpu_metadata writes into. The matching
        GPU views on the other side of the H2D are exposed via
        :meth:`bind_gpu_buffers`.
        """
        self._cpu_bufs = bufs

    def bind_gpu_buffers(self, gpu_view) -> None:
        """Attach shared GPU views from the context's :class:`ContextGPUView`."""
        self._gpu_view = gpu_view

    def reset(self) -> None:
        """
        Resets all Mamba states and frees all allocated slots.
        """
        self.request_to_mamba_state_idx.fill_(-1)

        self.reset_varlen_metadata()

        torch.arange(self.max_requests, out=self.mamba_state_free_slots)
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

        # Gated Delta Product chunk descriptor views
        self.gdp_chunk_indices = None
        self.gdp_chunk_indices_dp = None
        self.gdp_chunk_offsets = None
        self.gdp_intermediate_chunk_indices = None

        # Python-side precomputed values
        self.real_prefill_token_count = 0
        self.cu_seqlens_list = [0]

        # Intermediate state extraction views
        self.intermediate_chunk_indices = None
        self.intermediate_abs_positions = None
        self.intermediate_real_count = None
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
            intermediate_offsets_gpu, intermediate_counts_gpu: Used by
                `PrefixCachedMambaMetadata`; accepted (and ignored) here so callers can use
                one signature regardless of subclass.
        """
        del intermediate_offsets_gpu, intermediate_counts_gpu
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

            if self.gdp_num_householder > 0:
                self._fill_gdp_chunk_descriptors(
                    cu_seqlens_all, padded_prefill_count, padded_token_count
                )

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

        if padded_decode_count > 0 and padded_prefill_count > 0:
            self._device_decode_prefill_buffer[0] = cu_seqlens[real_decode_count]
            self._device_decode_prefill_buffer[1] = (
                cu_seqlens[real_decode_count + real_prefill_count] - cu_seqlens[real_decode_count]
            )
            self.device_decode_prefill = self._device_decode_prefill_buffer

    def _build_gdp_descriptor_tensors(
        self, cu_seqlens_all: list, padded_prefill_count: int, padded_token_count: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Build the GDP chunk descriptors as CPU int32 tensors.

        Shared by both destinations (the standalone device buffers and the bound
        pinned CPU views) so the two layouts cannot drift apart.

        Returns:
            `(chunk_indices, chunk_indices_dp, chunk_offsets, num_chunks,
            num_chunks_dp)`, the three tensors ready to copy into either
            destination plus the per-step array lengths.
        """
        chunk_indices, chunk_indices_dp, chunk_offsets, num_chunks, num_chunks_dp = (
            build_gdp_chunk_descriptors(
                cu_seqlens_all, padded_prefill_count, self.gdp_num_householder, padded_token_count
            )
        )
        return (
            torch.tensor(chunk_indices, dtype=torch.int32).view(num_chunks, 2),
            torch.tensor(chunk_indices_dp, dtype=torch.int32).view(num_chunks_dp, 2),
            torch.tensor(chunk_offsets, dtype=torch.int32),
            num_chunks,
            num_chunks_dp,
        )

    def _fill_gdp_chunk_descriptors(
        self, cu_seqlens_all: list, padded_prefill_count: int, padded_token_count: int
    ) -> None:
        """Build the GDP chunk descriptors into the standalone device buffers.

        The `update()` path only, i.e. callers that construct a MambaMetadata
        with no context to bind buffers from. A context goes through
        `_write_gdp_cpu_buffers` and the coalesced H2D instead.
        """
        chunk_indices, chunk_indices_dp, chunk_offsets, num_chunks, num_chunks_dp = (
            self._build_gdp_descriptor_tensors(
                cu_seqlens_all, padded_prefill_count, padded_token_count
            )
        )
        self._gdp_chunk_indices_buffer[:num_chunks].copy_(chunk_indices)
        self._gdp_chunk_indices_dp_buffer[:num_chunks_dp].copy_(chunk_indices_dp)
        self._gdp_chunk_offsets_buffer[: padded_prefill_count + 1].copy_(chunk_offsets)
        self.gdp_chunk_indices = self._gdp_chunk_indices_buffer[:num_chunks]
        self.gdp_chunk_indices_dp = self._gdp_chunk_indices_dp_buffer[:num_chunks_dp]
        self.gdp_chunk_offsets = self._gdp_chunk_offsets_buffer[: padded_prefill_count + 1]

    def _write_gdp_cpu_buffers(
        self, bufs: dict, cu_seqlens_all: list, padded_prefill_count: int, padded_token_count: int
    ) -> dict:
        """Build the GDP chunk descriptors into the bound pinned CPU views.

        Runs on the host, outside any CUDA graph; the values land on the device
        via the single coalesced H2D. The descriptor lengths depend only on the
        padded batch shape, so a replayed graph sees the same grid sizes it was
        captured with.

        Returns:
            The per-step array lengths `load_from_cpu` needs to slice the
            GPU views after the transfer.
        """
        chunk_indices, chunk_indices_dp, chunk_offsets, num_chunks, num_chunks_dp = (
            self._build_gdp_descriptor_tensors(
                cu_seqlens_all, padded_prefill_count, padded_token_count
            )
        )
        bufs['gdp_chunk_indices'][:num_chunks] = chunk_indices
        bufs['gdp_chunk_indices_dp'][:num_chunks_dp] = chunk_indices_dp
        bufs['gdp_chunk_offsets'][: padded_prefill_count + 1] = chunk_offsets
        return {"gdp_num_chunks": num_chunks, "gdp_num_chunks_dp": num_chunks_dp}

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
        """Compute all Mamba metadata on CPU, writing directly into the bound
        pinned CPU views.

        The values written here are transferred to GPU by the single coalesced
        H2D in :meth:`DynamicInferenceContext.transfer_bookkeeping_to_gpu`.
        The returned dict contains only Python scalars. On `PrefixCachedMambaMetadata`
        it also carries the intermediate GPU tensors, which :meth:`load_from_cpu`
        consumes after the H2D.

        Args:
            active_mamba_indices: CPU tensor of Mamba slot indices for active requests.
            token_to_request_idx: CPU tensor mapping tokens to request indices.
            cpu_cu_query: CPU cumulative query lengths from MHA metadata computation.
            batch_dimensions: Dimensions of the current batch.
            padded_batch_dimensions: Dimensions of the padded batch.
            enable_chunked_prefill: Whether chunked prefill is enabled.
            intermediate_offsets_gpu, intermediate_counts_gpu: Used by
                `PrefixCachedMambaMetadata`; accepted (and ignored) here so callers can use
                one signature regardless of subclass.
        """
        del intermediate_offsets_gpu, intermediate_counts_gpu
        assert self._cpu_bufs is not None, "bind_cpu_buffers() must be called first"
        bufs = self._cpu_bufs

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

        # Decode batch indices (write into pinned view; padded slots = -1).
        if padded_decode_count > 0:
            bufs['batch_indices_decode'][:real_decode_count] = active_mamba_indices[
                :real_decode_count
            ]
            if padded_decode_count > real_decode_count:
                bufs['batch_indices_decode'][real_decode_count:padded_decode_count] = -1

        # Prefill batch indices, seq_idx, cu_seqlens, chunk/conv metadata.
        if padded_prefill_count > 0:
            if real_prefill_count > 0:
                start = real_decode_count
                bufs['batch_indices_prefill'][:real_prefill_count] = active_mamba_indices[
                    start : start + real_prefill_count
                ]
            if padded_prefill_count > real_prefill_count:
                bufs['batch_indices_prefill'][real_prefill_count:padded_prefill_count] = -1

            # seq_idx: normalized token-to-request mapping for prefill tokens.
            prefill_start_req = real_decode_count
            end_prefill_req = real_decode_count + real_prefill_count
            start_token = cpu_cu_query[prefill_start_req].item()
            end_token = cpu_cu_query[end_prefill_req].item()
            seq_len = end_token - start_token

            if seq_len > 0:
                raw = token_to_request_idx[start_token:end_token]
                bufs['seq_idx'][0, :seq_len] = raw - raw[0]
            if padded_token_count > seq_len:
                bufs['seq_idx'][0, seq_len:padded_token_count] = -1
            result["seq_len"] = seq_len

            # cu_seqlens for prefill.
            cu_seqlens_view = bufs['cu_seqlens']
            cu_seqlens_view[0] = 0
            if real_prefill_count > 0:
                cu_seqlens_view[1 : real_prefill_count + 1] = (
                    cpu_cu_query[prefill_start_req + 1 : end_prefill_req + 1]
                    - cpu_cu_query[prefill_start_req]
                )
            if real_prefill_count < padded_prefill_count:
                last_val = cu_seqlens_view[real_prefill_count].item()
                cu_seqlens_view[real_prefill_count + 1 : padded_prefill_count + 1] = last_val

            cu_seqlens_list = cu_seqlens_view[: real_prefill_count + 1].tolist()
            real_prefill_tokens = (
                cu_seqlens_list[real_prefill_count] if real_prefill_count > 0 else 0
            )
            result["cu_seqlens_list"] = cu_seqlens_list
            result["real_prefill_token_count"] = real_prefill_tokens

            # Chunk metadata (Python loop, pure CPU).
            cu_seqlens_all = cu_seqlens_view[: padded_prefill_count + 1].tolist()
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
            bufs['cu_chunk_seqlens'][:n_cu] = torch.tensor(
                chunk_boundaries[:n_cu], dtype=torch.int32
            )
            bufs['last_chunk_indices'][:padded_prefill_count] = torch.tensor(
                last_chunk_idx_list, dtype=torch.int32
            )
            bufs['seq_idx_for_varlen'][:padded_max_chunks] = torch.tensor(
                chunk_to_seq_list[:padded_max_chunks], dtype=torch.int32
            )
            result["padded_max_chunks"] = padded_max_chunks
            if self.gdp_num_householder > 0:
                result.update(
                    self._write_gdp_cpu_buffers(
                        bufs, cu_seqlens_all, padded_prefill_count, padded_token_count
                    )
                )

            # Conv1d per-token metadata (CPU repeat_interleave).
            conv_seq_idx_view = bufs['conv_seq_idx']
            conv_seq_start_view = bufs['conv_seq_start']
            if real_prefill_tokens > 0:
                cu_t = cu_seqlens_view[: real_prefill_count + 1]
                lengths = (cu_t[1:] - cu_t[:-1]).to(torch.int64)
                seq_indices = torch.arange(real_prefill_count, dtype=torch.int32)
                seq_starts = cu_t[:real_prefill_count].to(torch.int32)
                conv_seq_idx_view[:real_prefill_tokens] = torch.repeat_interleave(
                    seq_indices, lengths
                )
                conv_seq_start_view[:real_prefill_tokens] = torch.repeat_interleave(
                    seq_starts, lengths
                )
            if padded_token_count > real_prefill_tokens:
                conv_seq_idx_view[real_prefill_tokens:padded_token_count] = 0
                conv_seq_start_view[real_prefill_tokens:padded_token_count] = 0

        # device_decode_prefill scalars.
        if padded_decode_count > 0 and padded_prefill_count > 0:
            result["decode_prefill_0"] = cpu_cu_query[real_decode_count].item()
            result["decode_prefill_1"] = (
                cpu_cu_query[real_decode_count + real_prefill_count].item()
                - cpu_cu_query[real_decode_count].item()
            )

        return result

    def load_from_cpu(self, d: dict) -> None:
        """Point state attributes at the freshly-transferred shared GPU views.

        No H2D copies happen here: the Mamba metadata fields were transferred
        as part of the coalesced bookkeeping H2D. This method just slices the
        bound GPU views to the per-step sizes; `PrefixCachedMambaMetadata` extends
        it with the intermediate metadata computation.

        Args:
            d: Dict returned by compute_cpu_metadata().
        """
        assert self._gpu_view is not None, "bind_gpu_buffers() must be called first"
        v = self._gpu_view

        padded_decode_count = d["padded_decode_count"]
        padded_prefill_count = d["padded_prefill_count"]
        padded_token_count = d["padded_token_count"]

        if padded_decode_count > 0:
            self.batch_indices_decode = v.mamba_batch_indices_decode[:padded_decode_count]

        if padded_prefill_count > 0:
            self.batch_indices_prefill = v.mamba_batch_indices_prefill[:padded_prefill_count]
            self.seq_idx = v.mamba_seq_idx[:, :padded_token_count]
            self.cu_seqlens = v.mamba_cu_seqlens[: padded_prefill_count + 1]
            self.cu_seqlens_list = d["cu_seqlens_list"]
            self.real_prefill_token_count = d["real_prefill_token_count"]

            padded_max_chunks = d["padded_max_chunks"]
            self.cu_chunk_seqlens = v.mamba_cu_chunk_seqlens[: padded_max_chunks + 1]
            self.last_chunk_indices = v.mamba_last_chunk_indices[:padded_prefill_count]
            self.seq_idx_for_varlen = v.mamba_seq_idx_for_varlen[:padded_max_chunks]
            self.conv_seq_idx = v.mamba_conv_seq_idx[:padded_token_count]
            self.conv_seq_start = v.mamba_conv_seq_start[:padded_token_count]

            if self.gdp_num_householder > 0:
                self.gdp_chunk_indices = v.gdp_chunk_indices[: d["gdp_num_chunks"]]
                self.gdp_chunk_indices_dp = v.gdp_chunk_indices_dp[: d["gdp_num_chunks_dp"]]
                self.gdp_chunk_offsets = v.gdp_chunk_offsets[: padded_prefill_count + 1]

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

        return int(mamba_idx)

    def detach_state_slot(self, request_idx: int) -> int:
        """Detach and return a request's live state slot without freeing it."""

        mamba_idx = int(self.request_to_mamba_state_idx[request_idx].item())
        if mamba_idx < 0:
            raise RuntimeError(f"Request index {request_idx} has no live Mamba state slot")
        self.request_to_mamba_state_idx[request_idx] = -1
        return mamba_idx

    def free_slot(self, mamba_idx: int) -> None:
        """Return one unbound slot to the live Mamba state pool."""

        if not 0 <= mamba_idx < self.max_requests:
            raise ValueError(f"Mamba state slot {mamba_idx} is outside the live state pool")
        if self.mamba_state_free_slot_count >= self.max_requests:
            raise RuntimeError("Cannot free a Mamba state slot when the pool is already full")
        self.mamba_state_free_slots[self.mamba_state_free_slot_count] = mamba_idx
        self.mamba_state_free_slot_count += 1

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

        return mamba_idx.clone()

    def _return_slots(self, mamba_indices: torch.Tensor) -> None:
        """Return live state slots to the free-slot stack."""

        if mamba_indices.numel() == 0:
            return
        start = self.mamba_state_free_slot_count
        end = start + mamba_indices.numel()
        if end > self.max_requests:
            raise RuntimeError("Mamba state free-slot pool overflow")
        self.mamba_state_free_slots[start:end] = mamba_indices.to(torch.int32)
        self.mamba_state_free_slot_count = end

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
        self._return_slots(mamba_indices_to_free)

        # Invalidate the Mamba state index for the finished requests
        self.request_to_mamba_state_idx[request_indices] = -1


class PrefixCachedMambaMetadata(MambaMetadata):
    """`MambaMetadata` extended with prefix-cache intermediate-state extraction."""

    def __init__(
        self,
        max_requests: int,
        max_tokens: int,
        *,
        num_mamba_layers: int,
        conv_states_shape: tuple,
        ssm_states_shape: tuple,
        conv_states_dtype: torch.dtype,
        ssm_states_dtype: torch.dtype,
        max_intermediate_count: int,
        mamba_chunk_size: int = 128,
        d_conv: int = 0,
        decode_indices_dtype: torch.dtype = torch.int64,
        gdp_num_householder: int = 0,
    ):
        """
        Args (beyond `MambaMetadata`):
            num_mamba_layers (int): Number of SSM layers; sizes the scratch output buffers.
            conv_states_shape / ssm_states_shape: Per-slot state shapes (excluding the
                layer and slot dims).
            conv_states_dtype / ssm_states_dtype: Dtypes of the state tensors.
            max_intermediate_count (int): Per-step bound on Mamba intermediate-state extractions.
        """
        super().__init__(
            max_requests,
            max_tokens,
            mamba_chunk_size=mamba_chunk_size,
            d_conv=d_conv,
            decode_indices_dtype=decode_indices_dtype,
            gdp_num_householder=gdp_num_householder,
        )

        # Intermediate state extraction buffers (CUDA graph compatible). Sized by
        # the per-step token-budget cap shared from DynamicInferenceContext.
        self.max_intermediate_count = max_intermediate_count
        self._intermediate_chunk_indices_buffer = torch.zeros(
            self.max_intermediate_count, dtype=torch.int64, device=self.device
        )
        self._intermediate_abs_positions_buffer = torch.full(
            (self.max_intermediate_count,), d_conv, dtype=torch.int32, device=self.device
        )
        # Runtime real-count tensor read by the fused gather+scatter Triton
        # kernels (intermediate_extraction.py). Fixed-address, rewritten each step
        # so captured CUDA graphs stay valid while the kernels skip padded slots
        # (pid_slot >= real_count).
        self._intermediate_real_count_buffer = torch.zeros(1, dtype=torch.int32, device=self.device)
        if gdp_num_householder > 0:
            # GDP's own offset -> chunk-row mapping for prefix-caching state
            # extraction. It cannot reuse the Mamba one: the chunk size differs
            # (64 vs mamba_chunk_size) and GDP's per-chunk states are indexed by
            # the chunk they enter, not the chunk they leave.
            self._gdp_intermediate_chunk_indices_buffer = torch.zeros(
                max_intermediate_count, dtype=torch.int64, device=self.device
            )

        # Pre-allocated "scratch" output buffers for CUDA graph compatible
        # extraction (GPU): per-step staging that the kernels write intermediate
        # states into before `MambaSlotAllocator.commit_intermediate_states` copies
        # them to the durable cache. The budget accounting in
        # DynamicInferenceContext refers to these as the "scratch" buffers.
        self.intermediate_ssm_out = torch.zeros(
            (num_mamba_layers, self.max_intermediate_count) + ssm_states_shape,
            dtype=ssm_states_dtype,
            device=self.device,
        )
        self.intermediate_conv_out = torch.zeros(
            (num_mamba_layers, self.max_intermediate_count) + conv_states_shape,
            dtype=conv_states_dtype,
            device=self.device,
        )

        # Per-request CPU bookkeeping for intermediate state extraction.
        k = MAX_INTERMEDIATE_OFFSETS_PER_REQUEST
        self._intermediate_offsets_cpu = torch.zeros(
            (max_requests, k), dtype=torch.int32, device='cpu'
        )
        self._intermediate_counts_cpu = torch.zeros(max_requests, dtype=torch.int32, device='cpu')
        self._intermediate_block_ids_cpu = torch.full(
            (max_requests, k), -1, dtype=torch.int32, device='cpu'
        )
        self._eos_cache_block_id_cpu = torch.full(
            (max_requests,), -1, dtype=torch.int32, device='cpu'
        )
        # CPU flag to skip the commit pipeline when nothing was extracted.
        self._has_intermediates = False

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
        super().update(
            active_mamba_indices=active_mamba_indices,
            token_to_request_idx=token_to_request_idx,
            cu_seqlens=cu_seqlens,
            batch_dimensions=batch_dimensions,
            padded_batch_dimensions=padded_batch_dimensions,
            enable_chunked_prefill=enable_chunked_prefill,
        )
        if padded_batch_dimensions.prefill_req_count > 0:
            # Convert per-request token offsets to chunk indices and absolute positions,
            # padded to fixed size for CUDA graph compat.
            self._update_intermediate_metadata(
                intermediate_offsets_gpu,
                intermediate_counts_gpu,
                batch_dimensions.prefill_req_count,
                padded_batch_dimensions.prefill_req_count,
            )

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
        result = super().compute_cpu_metadata(
            active_mamba_indices=active_mamba_indices,
            token_to_request_idx=token_to_request_idx,
            cpu_cu_query=cpu_cu_query,
            batch_dimensions=batch_dimensions,
            padded_batch_dimensions=padded_batch_dimensions,
            enable_chunked_prefill=enable_chunked_prefill,
        )
        # Intermediate metadata still requires GPU data: defer to load_from_cpu.
        if result["padded_prefill_count"] > 0:
            result["intermediate_offsets_gpu"] = intermediate_offsets_gpu
            result["intermediate_counts_gpu"] = intermediate_counts_gpu
        return result

    def load_from_cpu(self, d: dict) -> None:
        super().load_from_cpu(d)
        if d["padded_prefill_count"] > 0:
            # Intermediate metadata reads from the just-transferred cu_seqlens
            # to compute chunk indices & absolute positions for state extraction.
            self._update_intermediate_metadata(
                d["intermediate_offsets_gpu"],
                d["intermediate_counts_gpu"],
                d["real_prefill_count"],
                d["padded_prefill_count"],
                cu_seqlens_gpu=self._gpu_view.mamba_cu_seqlens,
            )

    def _update_intermediate_metadata(
        self,
        intermediate_offsets_gpu: Optional[torch.Tensor],
        intermediate_counts_gpu: Optional[torch.Tensor],
        real_prefill_count: int,
        padded_prefill_count: int,
        cu_seqlens_gpu: Optional[torch.Tensor] = None,
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
            padded_prefill_count: Prefill request count after batch padding
                (equals the captured graph bucket under CUDA graphs, or the
                round-up-padded count in eager mode; always >= real_prefill_count).
                Bounds the exposed/padded extent of the intermediate views via
                ``max_count`` so CUDA graph replay always touches a fixed-size
                region within the scratch buffers.
            cu_seqlens_gpu: GPU cu_seqlens tensor to read from. Defaults to
                the legacy standalone ``_cu_seqlens_buffer`` used by
                :meth:`update`; the coalesced production path passes the
                shared ``ContextGPUView.mamba_cu_seqlens`` view.
        """
        chunk_size = self.mamba_chunk_size
        # Cap at the token-budget bound so the per-step views never exceed the
        # buffers, even for high-prefill-count graph buckets where
        # padded_prefill_count * MAX_INTERMEDIATE_OFFSETS_PER_REQUEST would.
        max_count = min(
            padded_prefill_count * MAX_INTERMEDIATE_OFFSETS_PER_REQUEST, self.max_intermediate_count
        )
        if cu_seqlens_gpu is None:
            cu_seqlens_gpu = self._cu_seqlens_buffer

        if intermediate_offsets_gpu is not None and real_prefill_count > 0:
            # counts_list is CPU-cheap (source is already CPU from MambaSlotAllocator).
            counts_list = intermediate_counts_gpu.tolist()
            total = sum(counts_list)

            # Ensure GPU copies for vectorized GPU ops below.
            if not intermediate_offsets_gpu.is_cuda:
                intermediate_offsets_gpu = intermediate_offsets_gpu.to(
                    self.device, non_blocking=True
                )
            if not intermediate_counts_gpu.is_cuda:
                intermediate_counts_gpu = intermediate_counts_gpu.to(self.device, non_blocking=True)

            if total > 0:
                # Reuse the actual chunk layout. Context-aligned prefills can
                # have a partial first chunk, so ceil(seq_len / chunk_size)
                # undercounts chunks and shifts snapshots of later requests.
                cu = cu_seqlens_gpu[: real_prefill_count + 1]
                cum_chunks = torch.zeros(real_prefill_count, dtype=torch.int64, device=self.device)
                cum_chunks[1:] = self.last_chunk_indices[: real_prefill_count - 1] + 1

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
                # The token-budget bound guarantees this; fail loudly rather than
                # silently overrun the scratch buffers if the candidate-offset
                # logic in MambaSlotAllocator.compute_and_store_offsets changes.
                assert real_count <= self.max_intermediate_count, (
                    f"Mamba intermediate count {real_count} exceeds buffer size "
                    f"{self.max_intermediate_count}"
                )
                self._intermediate_chunk_indices_buffer[:real_count] = valid_chunk_indices
                self._intermediate_abs_positions_buffer[:real_count] = valid_abs_positions.to(
                    torch.int32
                )

                # Same offsets, GDP's chunking. No -1: GDP's per-chunk states are
                # the states *entering* each chunk, so the state after `offset`
                # tokens is row `offset // 64` of the sequence's chunk range,
                # whereas Mamba's are the states leaving each chunk.
                if self.gdp_num_householder > 0:
                    # The GDP chunk descriptors must already be built for this step:
                    # the view published below is what enables extraction, and it
                    # would otherwise carry the previous step's rows.
                    assert self.gdp_chunk_offsets is not None, (
                        "GDP chunk descriptors must be built before intermediate "
                        "extraction metadata."
                    )
                    gdp_cu_chunk_offsets = self.gdp_chunk_offsets[:real_prefill_count].to(
                        torch.int64
                    )
                    gdp_indices_2d = gdp_cu_chunk_offsets.unsqueeze(1).expand_as(offsets) + (
                        offsets // GDP_CHUNK_SIZE
                    )
                    self._gdp_intermediate_chunk_indices_buffer[:real_count] = gdp_indices_2d[
                        valid_mask
                    ]

                # Pad unused slots with safe defaults for CUDA graph replay:
                # - chunk_indices=0: reads from chunk 0 (always exists), output ignored
                # - abs_positions=d_conv: conv gather reads tokens [0..d_conv-1].
                #   These are within bounds only when the prefill has at least
                #   d_conv tokens; shorter sequences (e.g. small CUDA-graph warmup
                #   buckets) would overrun the token axis, so ssm_prefill clamps
                #   the gather positions into range. The gathered state is unused.
                if real_count < max_count:
                    self._intermediate_chunk_indices_buffer[real_count:max_count].fill_(0)
                    self._intermediate_abs_positions_buffer[real_count:max_count].fill_(self.d_conv)
                    if self.gdp_num_householder > 0:
                        self._gdp_intermediate_chunk_indices_buffer[real_count:max_count].fill_(0)

                self.intermediate_count = real_count
                self.per_request_intermediate_counts = counts_list
            else:
                # All counts are 0
                self._intermediate_chunk_indices_buffer[:max_count] = 0
                self._intermediate_abs_positions_buffer[:max_count] = self.d_conv
                if self.gdp_num_householder > 0:
                    self._gdp_intermediate_chunk_indices_buffer[:max_count] = 0
                self.intermediate_count = 0
                self.per_request_intermediate_counts = counts_list

            self.intermediate_chunk_indices = self._intermediate_chunk_indices_buffer[:max_count]
            self.intermediate_abs_positions = self._intermediate_abs_positions_buffer[:max_count]
            if self.gdp_num_householder > 0:
                self.gdp_intermediate_chunk_indices = self._gdp_intermediate_chunk_indices_buffer[
                    :max_count
                ]
            # Publish real_count to the fixed-address GPU tensor the scatter
            # kernels consult. fill_ is async (no host sync) and keeps the tensor
            # at the same address captured graphs reference.
            self._intermediate_real_count_buffer.fill_(self.intermediate_count)
            self.intermediate_real_count = self._intermediate_real_count_buffer
        else:
            # No extraction: fill with safe defaults for CUDA graph warmup
            # (same rationale as padding comment above; abs_positions=d_conv may
            # exceed a sub-d_conv warmup sequence, so ssm_prefill clamps the
            # gather positions into range and the gathered state is unused)
            self._intermediate_chunk_indices_buffer[:max_count] = 0
            self._intermediate_abs_positions_buffer[:max_count] = self.d_conv
            self.intermediate_count = 0
            self.per_request_intermediate_counts = []
            self.intermediate_chunk_indices = self._intermediate_chunk_indices_buffer[:max_count]
            self.intermediate_abs_positions = self._intermediate_abs_positions_buffer[:max_count]
            if self.gdp_num_householder > 0:
                self._gdp_intermediate_chunk_indices_buffer[:max_count] = 0
                self.gdp_intermediate_chunk_indices = self._gdp_intermediate_chunk_indices_buffer[
                    :max_count
                ]
            self._intermediate_real_count_buffer.fill_(0)
            self.intermediate_real_count = self._intermediate_real_count_buffer


    # -------------------------------------------------------------------------
    # Intermediate state tracking
    # -------------------------------------------------------------------------

    def compute_and_store_offsets(
        self,
        ctx,
        req,
        current_id: int,
        skip_tokens: int,
        prefill_chunk_length: int,
        num_matched_blocks: int,
        matched_block_ids: list,
        overall_required_blocks: int,
    ) -> None:
        """Stage reusable recurrent states at interior offsets and/or an aligned chunk endpoint.

        Args:
            ctx: The owning :class:`DynamicInferenceContext`.
            req: The inference request.
            current_id: Context request index.
            skip_tokens: Number of tokens being skipped (mamba match).
            prefill_chunk_length: Total prefill chunk length before skipping.
            num_matched_blocks: Number of KV-matched blocks.
            matched_block_ids: List of matched KV block IDs.
            overall_required_blocks: Total blocks needed for this request.
        """
        bs = ctx.block_size_tokens
        prompt_len = len(req.prompt_tokens)

        # Absolute token position (from the prompt start) where THIS chunk's
        # computed tokens begin. The first chunk computes from `skip_tokens` (the
        # prefix that was skipped); continuation chunks compute from
        # `finished_chunk_token_count` (with skip_tokens == 0). Framing the
        # boundary offsets against this chunk start -- rather than assuming the
        # first chunk -- lets us extract Mamba state at block boundaries that fall
        # in ANY chunk. In particular the last complete block of a multi-chunk
        # prompt lives in a continuation chunk; it was previously unreachable, so
        # non-block-aligned prompts never cached a usable resume boundary and
        # later turns could not skip prefill.
        chunk_start = req.finished_chunk_token_count + skip_tokens
        seq_len = prefill_chunk_length - skip_tokens  # tokens computed this chunk
        chunk_end = req.finished_chunk_token_count + prefill_chunk_length

        # Candidate absolute block boundaries at which to cache Mamba state.
        kv_div_abs = num_matched_blocks * bs
        last_aligned_abs = (prompt_len // bs) * bs  # last complete block boundary
        penultimate_abs = (overall_required_blocks - 1) * bs

        # Quantum every SSM mixer in the model agrees is a chunk boundary. States
        # can only be extracted there, and it is a multiple of the Mamba kernel
        # chunk size (asserted in MambaSlotAllocator.__init__), so the offset ->
        # chunk-index conversion in `_update_intermediate_metadata` stays consistent.
        ssm_chunk_alignment = ctx.ssm_chunk_alignment

        # Keep only boundaries that land inside this chunk's computed tokens and on
        # an SSM chunk boundary (required for mid-sequence state extraction).
        offsets_set = set()
        for abs_pos in (kv_div_abs, last_aligned_abs, penultimate_abs):
            offset = abs_pos - chunk_start
            if offset > 0 and offset < seq_len and offset % ssm_chunk_alignment == 0:
                offsets_set.add(offset)

        offsets = sorted(offsets_set)
        count = len(offsets)

        # CPU bookkeeping writes (no GPU kernel launches).
        if count > 0:
            abs_tokens_cpu = torch.tensor([chunk_start + o for o in offsets], dtype=torch.int64)
            block_indices_cpu = abs_tokens_cpu // bs - 1
            bids_cpu = ctx.request_to_kv_block_ids[current_id][block_indices_cpu]

            self._intermediate_offsets_cpu[current_id, :count] = torch.tensor(
                offsets, dtype=torch.int32
            )
            self._intermediate_block_ids_cpu[current_id, :count] = bids_cpu.to(torch.int32)
            self._has_intermediates = True
        self._intermediate_counts_cpu[current_id] = count

        # At a block-aligned chunk end, the request's live state is exactly the
        # state for that block boundary and can be cached directly. This covers
        # both aligned final prompts and non-final boundaries, which cannot use
        # intermediate extraction because their offset equals `seq_len`.
        if chunk_end > 0 and chunk_end % bs == 0:
            last_block_idx = chunk_end // bs - 1
            if last_block_idx >= 0:
                self._eos_cache_block_id_cpu[current_id] = ctx.request_to_kv_block_ids[current_id][
                    last_block_idx
                ]
                self._has_intermediates = True
            else:
                self._eos_cache_block_id_cpu[current_id] = -1
        else:
            self._eos_cache_block_id_cpu[current_id] = -1

    def get_intermediate_cpu_data(self, ctx):
        """Return ``(offsets_cpu, counts_cpu)`` slices for the current prefill batch.

        Returns:
            Tuple of (offsets_cpu, counts_cpu) where:
                offsets_cpu: [prefill_count, 3] int32 CPU tensor
                counts_cpu: [prefill_count] int32 CPU tensor
            Returns (None, None) if no prefill requests or no intermediates.
        """
        if not self._has_intermediates:
            return None, None

        prefill_count = ctx.batch_dimensions.prefill_req_count
        if prefill_count == 0:
            return None, None

        prefill_start = ctx.paused_request_count + ctx.batch_dimensions.decode_req_count
        offsets = self._intermediate_offsets_cpu[prefill_start : prefill_start + prefill_count]
        counts = self._intermediate_counts_cpu[prefill_start : prefill_start + prefill_count]
        return offsets, counts

    def clear_intermediate_state(self, ctx) -> None:
        """Clear per-request intermediate state for the current prefill batch."""
        prefill_count = ctx.batch_dimensions.prefill_req_count
        if prefill_count > 0:
            prefill_start = ctx.paused_request_count + ctx.batch_dimensions.decode_req_count
            end = prefill_start + prefill_count
            self._intermediate_counts_cpu[prefill_start:end].fill_(0)
            self._intermediate_offsets_cpu[prefill_start:end].fill_(0)
            self._intermediate_block_ids_cpu[prefill_start:end].fill_(-1)
            self._eos_cache_block_id_cpu[prefill_start:end].fill_(-1)
        self._has_intermediates = False

    def clear_request_entries(self, request_indexes: torch.Tensor) -> None:
        """Clear per-request bookkeeping for the given context indexes."""
        self._intermediate_counts_cpu[request_indexes] = 0
        self._intermediate_offsets_cpu[request_indexes] = 0
        self._intermediate_block_ids_cpu[request_indexes] = -1
        self._eos_cache_block_id_cpu[request_indexes] = -1

    def reset_intermediate_state(self) -> None:
        """Full wipe: zero the scratch output buffers and all per-request CPU bookkeeping."""
        self.intermediate_ssm_out.zero_()
        self.intermediate_conv_out.zero_()
        self._intermediate_offsets_cpu.fill_(0)
        self._intermediate_counts_cpu.fill_(0)
        self._intermediate_block_ids_cpu.fill_(-1)
        self._eos_cache_block_id_cpu.fill_(-1)
        self._has_intermediates = False
