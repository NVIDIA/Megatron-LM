# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch

try:
    import flashinfer
except ImportError:
    flashinfer = None

from .mha_splitpd_metadata import MHASplitPDMetadata
from megatron.core.transformer.enums import AttnBackend
from .triton import attn_partial_copy_triton, attn_merge_triton
from megatron.core.utils import nvtx_range_push, nvtx_range_pop


class MHAFlashInferMetadata(MHASplitPDMetadata):
    """
    Base class for FlashInfer metadata, extending MHASplitPDMetadata.
    Adds FlashInfer wrapper support while reusing all buffer allocation and Triton computation.
    """

    def __init__(
        self,
        block_count_total,
        max_kv_block_count,
        max_requests,
        block_size_tokens,
        max_seqlen,
        backend: AttnBackend = AttnBackend.flashinfer_fa2
    ):
        super().__init__(
            block_count_total, max_kv_block_count, max_requests, block_size_tokens, max_seqlen
        )

        if flashinfer is None:
            raise ImportError(
                "FlashInfer is not installed. Please install it to use FlashInfer metadata classes."
            )

        self.backend = backend
        # Map backend enum to FlashInfer backend strings
        self.flashinfer_backend_map = {
            AttnBackend.flashinfer_fa2: "fa2",
            AttnBackend.flashinfer_fa3: "fa3",
            AttnBackend.flashinfer_trt: "trtllm-gen",
        }

        if self.backend not in self.flashinfer_backend_map:
            raise ValueError(
                f"Backend {self.backend} is not a valid FlashInfer backend. "
                f"Valid options: {list(self.flashinfer_backend_map.keys())}"
            )

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
        request_query_lengths: torch.Tensor,
        request_kv_length_offsets: torch.Tensor,
        request_to_kv_block_ids: torch.Tensor,
        real_config,
        padded_config,
    ):
        """
        Update metadata using parent's Triton kernel, then plan FlashInfer wrappers.

        Args:
            request_query_lengths: (>real_batch_size,) query lengths for each request
            request_kv_length_offsets: (>real_batch_size,) KV cache offsets for each request
            request_to_kv_block_ids: (>real_batch_size, max_kv_blocks) block table mapping
            real_config: Configuration object containing real batch settings
            padded_config: Configuration object containing padded batch settings
        """

        # Call parent to compute all layouts via Triton kernel
        super().update(
            request_query_lengths,
            request_kv_length_offsets,
            request_to_kv_block_ids,
            real_config,
            padded_config,
        )

        # Plan FlashInfer wrappers using computed metadata
        self._plan_flashinfer_wrappers(real_config, padded_config)

    def _plan_flashinfer_wrappers(self, real_config, padded_config):
        """
        Plan FlashInfer wrappers using metadata computed by parent class.

        Args:
            real_config: Configuration object containing real batch settings
            padded_config: Configuration object containing padded batch settings
        """
        pf_target = padded_config.prefill_req_count
        dc_target = padded_config.decode_req_count

        # Plan prefill wrapper if there are prefill requests
        if pf_target > 0:
            pf_wrapper = self.prefill_wrapper
            if pf_wrapper is not None:
                pf_wrapper.plan(
                    qo_indptr=self.state_data["prefill_qo_indptr"],
                    paged_kv_indptr=self.state_data["prefill_paged_kv_indptr"],
                    paged_kv_indices=self._prefill_paged_kv_indices_buf,
                    paged_kv_last_page_len=self.state_data["prefill_paged_kv_last_page_len"],
                    num_qo_heads=self._num_qo_heads,
                    num_kv_heads=self._num_kv_heads,
                    head_dim_qk=self._head_dim,
                    page_size=self.block_size_tokens,
                    q_data_type=self._params_dtype,
                    kv_data_type=self._params_dtype,
                    causal=True,
                    block_tables=self.state_data["prefill_block_table"],
                )

        # Plan decode wrapper if there are decode requests
        if dc_target > 0:
            dec_wrapper = self.decode_wrapper
            if dec_wrapper is not None:
                dec_wrapper.plan(
                    indptr=self.state_data["decode_paged_kv_indptr"],
                    indices=self._prefill_paged_kv_indices_buf,  # Reuse prefill indices
                    last_page_len=self.state_data["decode_paged_kv_last_page_len"],
                    num_qo_heads=self._num_qo_heads,
                    num_kv_heads=self._num_kv_heads,
                    head_dim=self._head_dim,
                    page_size=self.block_size_tokens,
                    q_data_type=self._params_dtype,
                    kv_data_type=self._params_dtype,
                    block_tables=self.state_data["decode_block_table"],
                )

    def set_model_params(self, num_qo_heads: int, num_kv_heads: int, head_dim: int, params_dtype: torch.dtype):
        """
        Set model parameters needed for FlashInfer planning.

        Args:
            num_qo_heads: Number of query/output heads
            num_kv_heads: Number of key/value heads
            head_dim: Dimension of each head
            params_dtype: Data type for parameters
        """
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._params_dtype = params_dtype

    def attention(self, q: torch.Tensor, kv_cache: torch.Tensor, softmax_scale=None, layer_idx=0):
        """
        Run FlashInfer attention for both prefill and decode phases.

        Args:
            q: Query tensor (batch, 1, num_heads, head_dim)
            kv_cache: KV cache tensor for this layer in M_N2HCD layout [N, 2, H, C, D]
            softmax_scale: Optional softmax scaling factor (unused by FlashInfer)

        Returns:
            Attention output tensor (batch, 1, num_heads, head_dim)
        """
        # Squeeze query: (batch, 1, heads, dim) -> (batch, heads, dim)
        q_fi = q.squeeze(1)

        # Get wrappers (implemented by subclasses)
        prefill_wrapper = self.prefill_wrapper
        decode_wrapper = self.decode_wrapper

        # prefill only:
        if self.padded_config.prefill_req_count > 0 and self.padded_config.decode_req_count == 0:
            o_fi = prefill_wrapper.run(q_fi, kv_cache)
            return o_fi.unsqueeze(1)
        
        if self.padded_config.decode_req_count > 0 and self.padded_config.prefill_req_count == 0:
            o_fi = decode_wrapper.run(q_fi, kv_cache)
            return o_fi.unsqueeze(1)

        # Prepare prefill input buffer using triton copy
        q_pf_input = torch.empty_like(q_fi)
        attn_partial_copy_triton(
            q_fi,
            q_pf_input,
            self.state_data["device_decode_prefill"],
            check_bounds=False
        )

        # Run prefill if there are prefill requests
        nvtx_range_push("prefill_wrapper.run")
        if self.padded_config.prefill_req_count > 0:
            o_fi_pf = prefill_wrapper.run(q_pf_input, kv_cache)
        else:
            o_fi_pf = torch.empty_like(q_fi)
        nvtx_range_pop("prefill_wrapper.run")

        # Run decode if there are decode requests
        nvtx_range_push("decode_wrapper.run")
        if self.padded_config.decode_req_count > 0:
            o_fi_dc = decode_wrapper.run(
                q_fi[:self.padded_config.decode_req_count],
                kv_cache
            )
        else:
            o_fi_dc = torch.empty_like(q_fi)
        nvtx_range_pop("decode_wrapper.run")

        # Merge prefill and decode outputs
        o_fi = torch.empty_like(q_fi, device=torch.cuda.current_device())
        attn_merge_triton(
            decode_tensor=o_fi_dc,
            prefill_tensor=o_fi_pf,
            output_tensor=o_fi,
            device_dc=self.state_data["device_decode_prefill"],
            pf_useful_from_beginning=True
        )
        
        # Unsqueeze back: (batch, heads, dim) -> (batch, 1, heads, dim)
        return o_fi.unsqueeze(1)


class GraphMHAFlashInferMetadata(MHAFlashInferMetadata):
    """
    FlashInfer metadata for CUDA graph mode.
    Creates wrappers per batch size configuration with buffer binding.
    """

    def __init__(
        self,
        block_count_total,
        max_kv_block_count,
        max_requests,
        block_size_tokens,
        max_seqlen,
        backend: AttnBackend = AttnBackend.flashinfer_fa2,
        prefill_workspace_size: int = 128 * 1024 * 1024,  # 128MB default
        decode_workspace_size: int = 128 * 1024 * 1024,   # 128MB default
    ):
        super().__init__(
            block_count_total, max_kv_block_count, max_requests, block_size_tokens, max_seqlen, backend
        )

        device = torch.cuda.current_device()

        # Allocate workspace buffers for FlashInfer
        self.prefill_workspace_buffer = torch.empty(
            prefill_workspace_size, dtype=torch.uint8, device=device
        )
        self.decode_workspace_buffer = torch.empty(
            decode_workspace_size, dtype=torch.uint8, device=device
        )

        # Per-batch-size wrapper caches for CUDA graph mode
        self._prefill_wrappers_by_bs = {}
        self._decode_wrappers_by_bs = {}

        # Track current batch sizes for wrapper retrieval
        self._current_pf_bs = 0
        self._current_dc_bs = 0

    def get_prefill_wrapper_for_batch_size(self, batch_size: int):
        """
        Get or create a prefill wrapper for the specified batch size.

        Args:
            batch_size: Number of prefill requests

        Returns:
            FlashInfer BatchPrefillWithPagedKVCacheWrapper configured for this batch size
        """
        if batch_size <= 0:
            return None

        if batch_size not in self._prefill_wrappers_by_bs:
            # Create wrapper with buffer slices bound for CUDA graph mode
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.prefill_workspace_buffer,
                "HND",
                use_cuda_graph=True,
                qo_indptr_buf=self._prefill_qo_indptr_buf[:batch_size + 1],
                paged_kv_indptr_buf=self._prefill_paged_kv_indptr_buf[:batch_size + 1],
                paged_kv_indices_buf=self._prefill_paged_kv_indices_buf,
                paged_kv_last_page_len_buf=self._prefill_paged_kv_last_page_len_buf[:batch_size],
                backend=self.flashinfer_backend_map[self.backend],
            )
            self._prefill_wrappers_by_bs[batch_size] = wrapper

        return self._prefill_wrappers_by_bs[batch_size]

    def get_decode_wrapper_for_batch_size(self, batch_size: int):
        """
        Get or create a decode wrapper for the specified batch size.

        Args:
            batch_size: Number of decode requests

        Returns:
            FlashInfer BatchDecodeWithPagedKVCacheWrapper configured for this batch size
        """
        if batch_size <= 0:
            return None

        if batch_size not in self._decode_wrappers_by_bs:
            # Create wrapper with buffer slices bound for CUDA graph mode
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.decode_workspace_buffer,
                "HND",
                use_tensor_cores=True,
                use_cuda_graph=True,
                paged_kv_indptr_buffer=self._decode_paged_kv_indptr_buf[:batch_size + 1],
                paged_kv_indices_buffer=self._prefill_paged_kv_indices_buf,  # Reuse prefill indices
                paged_kv_last_page_len_buffer=self._decode_paged_kv_last_page_len_buf[:batch_size],
                backend=self.flashinfer_backend_map[self.backend],
            )
            self._decode_wrappers_by_bs[batch_size] = wrapper

        return self._decode_wrappers_by_bs[batch_size]

    @property
    def prefill_wrapper(self):
        """Return the prefill wrapper for the current padded batch size."""
        return self.get_prefill_wrapper_for_batch_size(self._current_pf_bs)

    @property
    def decode_wrapper(self):
        """Return the decode wrapper for the current padded batch size."""
        return self.get_decode_wrapper_for_batch_size(self._current_dc_bs)

    def _plan_flashinfer_wrappers(self, real_config, padded_config):
        """
        Update current batch sizes and plan FlashInfer wrappers.

        Args:
            real_config: Configuration object containing real batch settings
            padded_config: Configuration object containing padded batch settings
        """
        # Update current batch sizes for wrapper retrieval
        self._current_pf_bs = padded_config.prefill_req_count
        self._current_dc_bs = padded_config.decode_req_count

        # Call parent implementation to plan wrappers
        super()._plan_flashinfer_wrappers(real_config, padded_config)


class NonGraphMHAFlashInferMetadata(MHAFlashInferMetadata):
    """
    FlashInfer metadata for non-CUDA-graph mode.
    Creates wrappers lazily without buffer binding.
    """

    def __init__(
        self,
        block_count_total,
        max_kv_block_count,
        max_requests,
        block_size_tokens,
        max_seqlen,
        backend: AttnBackend = AttnBackend.flashinfer_fa2,
        prefill_workspace_size: int = 128 * 1024 * 1024,  # 128MB default
        decode_workspace_size: int = 128 * 1024 * 1024,   # 128MB default
    ):
        super().__init__(
            block_count_total, max_kv_block_count, max_requests, block_size_tokens, max_seqlen, backend
        )

        device = torch.cuda.current_device()

        # Allocate workspace buffers for FlashInfer
        self.prefill_workspace_buffer = torch.empty(
            prefill_workspace_size, dtype=torch.uint8, device=device
        )
        self.decode_workspace_buffer = torch.empty(
            decode_workspace_size, dtype=torch.uint8, device=device
        )

        # Lazy wrapper initialization
        self._prefill_wrapper = None
        self._decode_wrapper = None

    @property
    def prefill_wrapper(self):
        """Lazily create and return the prefill wrapper."""
        if self._prefill_wrapper is None:
            self._prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.prefill_workspace_buffer,
                "HND",
                use_cuda_graph=False,
                backend=self.flashinfer_backend_map[self.backend],
            )
        return self._prefill_wrapper

    @property
    def decode_wrapper(self):
        """Lazily create and return the decode wrapper."""
        if self._decode_wrapper is None:
            self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.decode_workspace_buffer,
                "HND",
                use_tensor_cores=True,
                use_cuda_graph=False,
                backend=self.flashinfer_backend_map[self.backend],
            )
        return self._decode_wrapper
