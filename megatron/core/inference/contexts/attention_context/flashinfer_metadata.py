# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
FlashInfer attention metadata for dynamic inference.

Provides a unified metadata class that uses simple dispatch logic:
- All decode requests → BatchDecodeWithPagedKVCacheWrapper
- Otherwise (any prefill) → BatchPrefillWithPagedKVCacheWrapper
"""

import torch
from torch import Tensor

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.transformer.enums import AttnBackend

from .metadata_base import MetadataBase

try:
    import flashinfer

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False


class FlashInferMetadata(MetadataBase):
    """
    Unified FlashInfer attention metadata.

    Simple dispatch logic:
    - All decode requests → BatchDecodeWithPagedKVCacheWrapper
    - Otherwise (any prefill) → BatchPrefillWithPagedKVCacheWrapper
    """

    # Map AttnBackend enum to FlashInfer backend strings
    BACKEND_MAP = {
        AttnBackend.flashinfer_fa2: "fa2",
        AttnBackend.flashinfer_fa3: "fa3",
        AttnBackend.flashinfer_trt: "trtllm-gen",
    }

    def __init__(
        self,
        max_requests: int,
        max_kv_block_count: int,
        block_size_tokens: int,
        backend: AttnBackend,
        workspace_size: int = 512 * 1024 * 1024,  # 512MB default
    ):
        """
        Initialize FlashInfer metadata.

        Args:
            max_requests: Maximum number of concurrent requests
            max_kv_block_count: Maximum number of KV blocks per request
            block_size_tokens: Number of tokens per KV block (page size)
            backend: FlashInfer backend type (fa2, fa3, trt)
            workspace_size: Size of workspace buffer for FlashInfer wrappers
        """
        super().__init__()

        if not HAVE_FLASHINFER:
            raise ImportError("flashinfer is required for FlashInfer attention backend")

        self.device = torch.cuda.current_device()
        self.max_requests = max_requests
        self.max_kv_block_count = max_kv_block_count
        self.block_size_tokens = block_size_tokens
        self.backend = backend
        self.flashinfer_backend = self.BACKEND_MAP.get(backend, "fa2")

        # Model parameters (set via set_model_params)
        self._num_qo_heads = None
        self._num_kv_heads = None
        self._head_dim = None
        self._params_dtype = None

        # Pre-allocate buffers for FlashInfer
        # qo_indptr: cumulative query/output lengths [batch_size + 1]
        self._qo_indptr_buf = torch.zeros(
            max_requests + 1, dtype=torch.int32, device=self.device
        )

        # paged_kv_indptr: cumulative block counts [batch_size + 1]
        self._paged_kv_indptr_buf = torch.zeros(
            max_requests + 1, dtype=torch.int32, device=self.device
        )

        # paged_kv_indices: flattened block indices [total_blocks]
        max_total_blocks = max_requests * max_kv_block_count
        self._paged_kv_indices_buf = torch.zeros(
            max_total_blocks, dtype=torch.int32, device=self.device
        )

        # paged_kv_last_page_len: tokens in last page per request [batch_size]
        self._paged_kv_last_page_len_buf = torch.zeros(
            max_requests, dtype=torch.int32, device=self.device
        )

        # Workspace buffer for FlashInfer
        self.workspace_buffer = torch.empty(
            workspace_size, dtype=torch.uint8, device=self.device
        )

        # kv_seq_lengths buffer for cu_kv_lengths compatibility
        self._kv_seq_lengths_buf = torch.zeros(
            max_requests, dtype=torch.int32, device=self.device
        )

        # cu_kv_seq_lengths buffer for cu_kv_lengths compatibility
        self._cu_kv_seq_lengths_buf = torch.zeros(
            max_requests + 1, dtype=torch.int32, device=self.device
        )

        # block_table buffer for key_value_cache compatibility
        self._block_table_buf = torch.zeros(
            (max_requests, max_kv_block_count), dtype=torch.int32, device=self.device
        )

        # Runtime state
        self._is_all_decode = False
        self._batch_size = 0
        self._total_blocks = 0
        self._max_seqlen_q = 0
        self._max_seqlen_k = 0

    def set_model_params(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        params_dtype: torch.dtype,
    ):
        """
        Set model parameters needed for FlashInfer planning.

        Args:
            num_qo_heads: Number of query/output heads
            num_kv_heads: Number of key/value heads
            head_dim: Dimension of each attention head
            params_dtype: Data type for parameters (e.g., torch.float16)
        """
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._params_dtype = params_dtype

    @property
    def prefill_wrapper(self):
        """Return the prefill wrapper. Must be implemented by subclasses."""
        raise NotImplementedError

    @property
    def decode_wrapper(self):
        """Return the decode wrapper. Must be implemented by subclasses."""
        raise NotImplementedError

    def update(
        self,
        request_query_lengths: Tensor,
        request_kv_length_offsets: Tensor,
        request_to_kv_block_ids: Tensor,
        batch_dimensions: InferenceBatchDimensions,
        padded_batch_dimensions: InferenceBatchDimensions,
    ):
        """
        Update metadata from request states.

        Args:
            request_query_lengths: Query token count per request (real_batch_size,)
            request_kv_length_offsets: KV offset per request (real_batch_size,)
            request_to_kv_block_ids: Block IDs per request (real_batch_size, max_kv_blocks)
            batch_dimensions: Real batch dimensions
            padded_batch_dimensions: Padded batch dimensions for CUDA graphs
        """
        real_batch_size = batch_dimensions.req_count
        padded_batch_size = padded_batch_dimensions.req_count

        self._batch_size = padded_batch_size

        # Determine if all requests are decode (query_length == 1 for all)
        self._is_all_decode = (
            padded_batch_dimensions.prefill_req_count == 0
            and padded_batch_dimensions.decode_req_count > 0
        )

        # Build qo_indptr: cumulative query lengths
        self._qo_indptr_buf[0] = 0
        if real_batch_size > 0:
            cumsum = torch.cumsum(request_query_lengths, dim=0)
            self._qo_indptr_buf[1 : real_batch_size + 1] = cumsum
            # Pad remaining entries
            if padded_batch_size > real_batch_size:
                last_val = cumsum[-1].item()
                self._qo_indptr_buf[real_batch_size + 1 : padded_batch_size + 1] = last_val

        # Compute KV sequence lengths and block counts
        kv_seq_lengths = request_kv_length_offsets + request_query_lengths
        kv_block_counts = (kv_seq_lengths + self.block_size_tokens - 1) // self.block_size_tokens

        # Store kv_seq_lengths for cu_kv_lengths compatibility
        if real_batch_size > 0:
            self._kv_seq_lengths_buf[:real_batch_size] = kv_seq_lengths
            # Pad remaining entries
            if padded_batch_size > real_batch_size:
                self._kv_seq_lengths_buf[real_batch_size:padded_batch_size] = 0

        # Build cu_kv_seq_lengths: cumulative KV sequence lengths
        self._cu_kv_seq_lengths_buf[0] = 0
        if real_batch_size > 0:
            cumsum_kv = torch.cumsum(self._kv_seq_lengths_buf[:padded_batch_size], dim=0)
            self._cu_kv_seq_lengths_buf[1 : padded_batch_size + 1] = cumsum_kv

        # Compute max_seqlen_q and max_seqlen_k
        if padded_batch_dimensions.prefill_req_count == 0:
            self._max_seqlen_q = 1
        else:
            self._max_seqlen_q = max(2, request_query_lengths.max().item()) if real_batch_size > 0 else 1

        self._max_seqlen_k = kv_seq_lengths.max().item() if real_batch_size > 0 else 0

        # Build paged_kv_indptr: cumulative block counts
        self._paged_kv_indptr_buf[0] = 0
        if real_batch_size > 0:
            cumsum_blocks = torch.cumsum(kv_block_counts, dim=0)
            self._paged_kv_indptr_buf[1 : real_batch_size + 1] = cumsum_blocks
            self._total_blocks = cumsum_blocks[-1].item()
            # Pad remaining entries
            if padded_batch_size > real_batch_size:
                self._paged_kv_indptr_buf[
                    real_batch_size + 1 : padded_batch_size + 1
                ] = self._total_blocks
        else:
            self._total_blocks = 0

        # Flatten block table to paged_kv_indices
        if real_batch_size > 0 and self._total_blocks > 0:
            # Extract valid block IDs from block table
            idx = 0
            for i in range(real_batch_size):
                num_blocks = kv_block_counts[i].item()
                self._paged_kv_indices_buf[idx : idx + num_blocks] = request_to_kv_block_ids[
                    i, :num_blocks
                ]
                idx += num_blocks

        # Compute last page lengths
        if real_batch_size > 0:
            last_page_lens = kv_seq_lengths - (kv_block_counts - 1) * self.block_size_tokens
            self._paged_kv_last_page_len_buf[:real_batch_size] = last_page_lens
            # Pad remaining entries
            if padded_batch_size > real_batch_size:
                self._paged_kv_last_page_len_buf[real_batch_size:padded_batch_size] = 1

        # Store block table for key_value_cache compatibility
        if real_batch_size > 0:
            self._block_table_buf[:real_batch_size, :] = request_to_kv_block_ids[:real_batch_size, :]
            if padded_batch_size > real_batch_size:
                self._block_table_buf[real_batch_size:padded_batch_size, :] = -1

        # Plan the appropriate wrapper
        self._plan_wrapper(padded_batch_size)

        # Store state data
        self.state_data = {
            "qo_indptr": self._qo_indptr_buf[: padded_batch_size + 1],
            "paged_kv_indptr": self._paged_kv_indptr_buf[: padded_batch_size + 1],
            "paged_kv_indices": self._paged_kv_indices_buf[: self._total_blocks],
            "paged_kv_last_page_len": self._paged_kv_last_page_len_buf[:padded_batch_size],
            "is_all_decode": self._is_all_decode,
            "batch_size": padded_batch_size,
            # Compatibility keys for cu_query_lengths() and cu_kv_lengths()
            "cu_query_seq_lengths": self._qo_indptr_buf[: padded_batch_size + 1],  # alias to qo_indptr
            "cu_kv_seq_lengths": self._cu_kv_seq_lengths_buf[: padded_batch_size + 1],
            "kv_seq_lengths": self._kv_seq_lengths_buf[:padded_batch_size],
            "max_seqlen_q": self._max_seqlen_q,
            "max_seqlen_k": self._max_seqlen_k,
            "block_table": self._block_table_buf[:padded_batch_size, :],
        }

    def _plan_wrapper(self, batch_size: int):
        """Plan the FlashInfer wrapper with current metadata."""
        if batch_size == 0:
            return

        if self._is_all_decode:
            # Use decode wrapper
            self.decode_wrapper.plan(
                indptr=self._paged_kv_indptr_buf[: batch_size + 1],
                indices=self._paged_kv_indices_buf[: self._total_blocks],
                last_page_len=self._paged_kv_last_page_len_buf[:batch_size],
                num_qo_heads=self._num_qo_heads,
                num_kv_heads=self._num_kv_heads,
                head_dim=self._head_dim,
                page_size=self.block_size_tokens,
                q_data_type=self._params_dtype,
                kv_data_type=self._params_dtype,
                block_tables=self._block_table_buf[:batch_size, :],
            )
        else:
            # Use prefill wrapper (uses head_dim_qk instead of head_dim)
            self.prefill_wrapper.plan(
                qo_indptr=self._qo_indptr_buf[: batch_size + 1],
                paged_kv_indptr=self._paged_kv_indptr_buf[: batch_size + 1],
                paged_kv_indices=self._paged_kv_indices_buf[: self._total_blocks],
                paged_kv_last_page_len=self._paged_kv_last_page_len_buf[:batch_size],
                num_qo_heads=self._num_qo_heads,
                num_kv_heads=self._num_kv_heads,
                head_dim_qk=self._head_dim,
                page_size=self.block_size_tokens,
                q_data_type=self._params_dtype,
                kv_data_type=self._params_dtype,
                causal=True,
                block_tables=self._block_table_buf[:batch_size, :],
            )

    def attention(
        self,
        q: Tensor,
        kv_cache: Tensor,
        softmax_scale: float = None,
        layer_idx: int = 0,
    ) -> Tensor:
        """
        Run FlashInfer attention.

        Args:
            q: Query tensor of shape (batch, 1, num_heads, head_dim) or (tokens, num_heads, head_dim)
            kv_cache: KV cache tensor with layout M_N2HCD [blocks, 2, num_kv_heads, block_size, head_dim]
            softmax_scale: Optional softmax scale
            layer_idx: Layer index (unused for now)

        Returns:
            Output tensor of shape (batch, 1, num_heads, head_dim)
        """
        # Squeeze sequence dimension if present: (batch, 1, heads, dim) -> (batch, heads, dim)
        if q.dim() == 4:
            q_input = q.squeeze(1)
        else:
            q_input = q

        # Run appropriate wrapper
        if self._is_all_decode:
            output = self.decode_wrapper.run(q_input, kv_cache)
        else:
            output = self.prefill_wrapper.run(q_input, kv_cache)

        # Restore sequence dimension: (batch, heads, dim) -> (batch, 1, heads, dim)
        return output.unsqueeze(1)

    def reset(self):
        """Reset metadata for next batch."""
        self._qo_indptr_buf.fill_(0)
        self._paged_kv_indptr_buf.fill_(0)
        self._paged_kv_indices_buf.fill_(0)
        self._paged_kv_last_page_len_buf.fill_(0)
        self._kv_seq_lengths_buf.fill_(0)
        self._cu_kv_seq_lengths_buf.fill_(0)
        self._block_table_buf.fill_(0)
        self._is_all_decode = False
        self._batch_size = 0
        self._total_blocks = 0
        self._max_seqlen_q = 0
        self._max_seqlen_k = 0
        self.state_data = {}


class GraphFlashInferMetadata(FlashInferMetadata):
    """
    FlashInfer metadata for CUDA graph mode.

    Pre-binds buffers to wrappers and caches wrappers per batch size.
    """

    def __init__(
        self,
        max_requests: int,
        max_kv_block_count: int,
        block_size_tokens: int,
        backend: AttnBackend,
        workspace_size: int = 512 * 1024 * 1024,
    ):
        super().__init__(
            max_requests=max_requests,
            max_kv_block_count=max_kv_block_count,
            block_size_tokens=block_size_tokens,
            backend=backend,
            workspace_size=workspace_size,
        )

        # Cache wrappers per batch size
        self._prefill_wrappers_by_bs = {}
        self._decode_wrappers_by_bs = {}

    def _get_prefill_wrapper(self, batch_size: int):
        """Get or create prefill wrapper for given batch size."""
        if batch_size not in self._prefill_wrappers_by_bs:
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "HND",
                use_cuda_graph=True,
                qo_indptr_buf=self._qo_indptr_buf[: batch_size + 1],
                paged_kv_indptr_buf=self._paged_kv_indptr_buf[: batch_size + 1],
                paged_kv_indices_buf=self._paged_kv_indices_buf,
                paged_kv_last_page_len_buf=self._paged_kv_last_page_len_buf[:batch_size],
                backend=self.flashinfer_backend,
            )
            self._prefill_wrappers_by_bs[batch_size] = wrapper
        return self._prefill_wrappers_by_bs[batch_size]

    def _get_decode_wrapper(self, batch_size: int):
        """Get or create decode wrapper for given batch size."""
        if batch_size not in self._decode_wrappers_by_bs:
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "HND",
                use_tensor_cores=True,
                use_cuda_graph=True,
                paged_kv_indptr_buffer=self._paged_kv_indptr_buf[: batch_size + 1],
                paged_kv_indices_buffer=self._paged_kv_indices_buf,
                paged_kv_last_page_len_buffer=self._paged_kv_last_page_len_buf[:batch_size],
                backend=self.flashinfer_backend,
            )
            self._decode_wrappers_by_bs[batch_size] = wrapper
        return self._decode_wrappers_by_bs[batch_size]

    @property
    def prefill_wrapper(self):
        return self._get_prefill_wrapper(self._batch_size)

    @property
    def decode_wrapper(self):
        return self._get_decode_wrapper(self._batch_size)


class NonGraphFlashInferMetadata(FlashInferMetadata):
    """
    FlashInfer metadata for non-CUDA graph mode.

    Creates wrappers lazily without buffer binding.
    """

    def __init__(
        self,
        max_requests: int,
        max_kv_block_count: int,
        block_size_tokens: int,
        backend: AttnBackend,
        workspace_size: int = 512 * 1024 * 1024,
    ):
        super().__init__(
            max_requests=max_requests,
            max_kv_block_count=max_kv_block_count,
            block_size_tokens=block_size_tokens,
            backend=backend,
            workspace_size=workspace_size,
        )

        # Lazy wrapper initialization
        self._prefill_wrapper = None
        self._decode_wrapper = None

    @property
    def prefill_wrapper(self):
        if self._prefill_wrapper is None:
            self._prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "HND",
                use_cuda_graph=False,
                backend=self.flashinfer_backend,
            )
        return self._prefill_wrapper

    @property
    def decode_wrapper(self):
        if self._decode_wrapper is None:
            self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "HND",
                use_tensor_cores=True,
                use_cuda_graph=False,
                backend=self.flashinfer_backend,
            )
        return self._decode_wrapper
