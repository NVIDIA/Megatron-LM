# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""BF16 decode with a Gluon TMA/tcgen05 scorer and Triton exact top-k/attention.

The implementation calls no CuTe DSL or FlashInfer GPU operators. Scoring uses
native KV page tables; there is no expanded-page planning step.
"""

from __future__ import annotations

import torch
import triton

from ._dsa_triton_decode import attention as dsa_gqa_inference_kernels


class TritonDSADecodeWorkspace:
    """Reusable workspace for single-token DSA decode on Blackwell.

    Allocate one workspace per concurrently executing stream. Warm one call with
    the actual query/cache buffers before graph capture. Captured graphs read new
    tensor contents, including lengths, requests, and the native page table. The
    returned output is overwritten on the next call.

    Input indexer vectors have dimension 128. KV caches have one head, and native
    page sizes must be multiples of 128. Every positive context length must fit
    max_context and reference valid allocated pages in block_table. Active request
    indices must address valid rows of block_table. Score inputs must be NaN-free.
    The optimized top-k path uses count crossing to remain exact, with a GPU radix
    fallback for non-finite ranges or an oversized crossing bucket.

    Args:
        batch_size: Number of padded decode rows.
        num_query_heads: Attention query heads per row.
        head_dim: Attention key/value dimension.
        index_topk: Selected keys: 512, 1024, or 2048.
        max_context: Context capacity, at most 1048576.
        page_size: Tokens per native page, divisible by 128.
        device: SM100 or SM103 CUDA device.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        num_query_heads: int,
        head_dim: int,
        index_topk: int,
        max_context: int,
        page_size: int,
        device: torch.device,
    ) -> None:
        from triton.experimental.gluon import language as gl
        from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

        from ._dsa_triton_decode.score import score_tcgen
        from ._dsa_triton_decode.topk import (
            chunk_lengths,
            exact_topk,
            gather_chunk_candidates,
            remap_chunk_indices,
        )

        device = torch.device(device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) not in (
            (10, 0),
            (10, 3),
        ):
            raise ValueError("Native Triton DSA decode requires SM100 or SM103")
        if batch_size <= 0 or num_query_heads <= 0 or head_dim <= 0 or head_dim % 32:
            raise ValueError(
                "Batch/heads must be positive; head_dim must be a positive multiple of 32"
            )
        if index_topk not in (512, 1024, 2048) or not 0 < max_context <= 1048576:
            raise ValueError(
                "Supported top-k values are 512/1024/2048, with context capacity 1..1048576"
            )
        if page_size <= 0 or page_size % 128:
            raise ValueError("Native Triton DSA requires pages divisible by 128")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        self.device = device
        self.batch_size = batch_size
        self.num_query_heads = num_query_heads
        self.head_dim = head_dim
        self.index_topk = index_topk
        self.max_context = max_context
        self.page_size = page_size
        self.score_width = triton.cdiv(max_context, 4) * 4
        self._chunks = triton.cdiv(max_context, 65536)
        if self._chunks > 1:
            self.score_width = self._chunks * 65536
            self._chunk_lengths = torch.empty(
                batch_size * self._chunks, dtype=torch.int32, device=device
            )
            self._candidate_lengths = torch.empty(batch_size, dtype=torch.int32, device=device)
            self._chunk_indices = torch.empty(
                batch_size * self._chunks, index_topk, dtype=torch.int32, device=device
            )
            self._chunk_flags = torch.empty_like(self._chunk_lengths)
            self._candidates = torch.empty(
                batch_size, self._chunks * index_topk, dtype=torch.float32, device=device
            )
            self._selected = torch.empty(batch_size, index_topk, dtype=torch.int32, device=device)
        self._chunk_lengths_kernel = chunk_lengths
        self._gather_candidates = gather_chunk_candidates
        self._remap_indices = remap_chunk_indices
        self.scores = torch.empty(batch_size, self.score_width, dtype=torch.float32, device=device)
        self.indices = torch.empty(batch_size, index_topk, dtype=torch.int32, device=device)
        self.output = torch.empty(
            batch_size, num_query_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        self.used_radix_fallback = torch.empty(batch_size, dtype=torch.int32, device=device)
        self.request_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        self._score = score_tcgen
        self._topk = exact_topk
        self._descriptor_type = TensorDescriptor
        self._descriptor_layout = gl.NVMMASharedLayout(
            swizzle_byte_width=128, element_bitwidth=16, rank=2, transposed=False
        )
        self._descriptor_key = None
        self._query_descriptor = None
        self._key_descriptor = None
        self._score_workers = 8
        self._score_stages = 3
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self._slices = dsa_gqa_inference_kernels._num_slices(batch_size, 1, index_topk, sms)
        self._slice_len = triton.cdiv(index_topk, self._slices)
        partial_rows = batch_size * self._slices * num_query_heads
        self._partial = torch.empty(partial_rows, head_dim, dtype=torch.float32, device=device)
        self._maxima = torch.empty(partial_rows, dtype=torch.float32, device=device)
        self._sums = torch.empty_like(self._maxima)

    def __call__(
        self,
        *,
        query: torch.Tensor,
        index_query: torch.Tensor,
        index_key_cache: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lengths: torch.Tensor,
        softmax_scale: float,
        request_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score, select exact top-k, and attend into preallocated output storage."""
        requests = self.request_indices if request_indices is None else request_indices
        tensors = (
            query,
            index_query,
            index_key_cache,
            key_cache,
            value_cache,
            block_table,
            context_lengths,
            requests,
        )
        if any(t.device != self.device for t in tensors):
            raise ValueError("Every input must be on the workspace CUDA device")
        if (
            query.shape != self.output.shape
            or query.dtype != torch.bfloat16
            or query.stride(-1) != 1
        ):
            raise ValueError("query must be BF16 [batch_size,num_query_heads,head_dim]")
        if query.stride(1) != self.head_dim:
            raise ValueError("Query heads must occupy compact rows")
        if (
            index_query.shape != (self.batch_size, 128)
            or index_query.dtype != torch.bfloat16
            or not index_query.is_contiguous()
        ):
            raise ValueError("index_query must be contiguous BF16 [batch_size,128]")
        if (
            index_key_cache.ndim != 3
            or index_key_cache.shape[1:] != (self.page_size, 128)
            or index_key_cache.dtype != torch.bfloat16
            or not index_key_cache.is_contiguous()
        ):
            raise ValueError("index_key_cache must be contiguous BF16 [blocks,page_size,128]")
        if key_cache.shape != value_cache.shape or key_cache.shape != (
            index_key_cache.shape[0],
            self.page_size,
            1,
            self.head_dim,
        ):
            raise ValueError("Key/value caches must be [blocks,page_size,1,head_dim]")
        if key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16:
            raise ValueError("Key/value caches must use BF16")
        if key_cache.stride(-1) != 1 or value_cache.stride(-1) != 1:
            raise ValueError("Cache head dimensions must have unit stride")
        for name, tensor in (("context_lengths", context_lengths), ("request_indices", requests)):
            if (
                tensor.dtype != torch.int32
                or tensor.shape != (self.batch_size,)
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous int32 [batch_size]")
        if block_table.dtype != torch.int32 or block_table.ndim != 2 or block_table.stride(-1) != 1:
            raise ValueError("block_table must have compact int32 rows")

        descriptor_key = (
            index_query.data_ptr(),
            index_key_cache.data_ptr(),
            index_key_cache.shape[0],
        )
        with torch.cuda.device(self.device):
            if descriptor_key != self._descriptor_key:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Warm the workspace with these query/cache buffers before capture"
                    )
                self._query_descriptor = self._descriptor_type.from_tensor(
                    index_query, [8, 128], self._descriptor_layout
                )
                self._key_descriptor = self._descriptor_type.from_tensor(
                    index_key_cache.view(-1, 128), [128, 128], self._descriptor_layout
                )
                self._descriptor_key = descriptor_key

            self._score[(self.batch_size, self._score_workers)](
                self._query_descriptor,
                self._key_descriptor,
                block_table,
                context_lengths,
                requests,
                self.scores,
                stride_table=block_table.stride(0),
                WIDTH=self.score_width,
                PAGE=self.page_size,
                WORKERS=self._score_workers,
                STAGES=self._score_stages,
                num_warps=4,
            )
            if self._chunks == 1:
                self._topk[(self.batch_size,)](
                    self.scores,
                    context_lengths,
                    self.indices,
                    self.used_radix_fallback,
                    WIDTH=self.score_width,
                    K=self.index_topk,
                    num_warps=32,
                    enable_fp_fusion=False,
                )
            else:
                # A global top-k is contained in the union of each disjoint chunk's top-k.
                self._chunk_lengths_kernel[(self.batch_size,)](
                    context_lengths,
                    self._chunk_lengths,
                    self._candidate_lengths,
                    CHUNKS=self._chunks,
                    K=self.index_topk,
                )
                self._topk[(self.batch_size * self._chunks,)](
                    self.scores,
                    self._chunk_lengths,
                    self._chunk_indices,
                    self._chunk_flags,
                    WIDTH=65536,
                    K=self.index_topk,
                    num_warps=32,
                    enable_fp_fusion=False,
                )
                self._gather_candidates[(self.batch_size, self._chunks)](
                    self.scores,
                    self._chunk_indices,
                    self._candidates,
                    WIDTH=self.score_width,
                    CHUNKS=self._chunks,
                    K=self.index_topk,
                )
                self._topk[(self.batch_size,)](
                    self._candidates,
                    self._candidate_lengths,
                    self._selected,
                    self.used_radix_fallback,
                    WIDTH=self._chunks * self.index_topk,
                    K=self.index_topk,
                    num_warps=32,
                    enable_fp_fusion=False,
                )
                self._remap_indices[(self.batch_size,)](
                    self._chunk_indices,
                    self._selected,
                    context_lengths,
                    self.indices,
                    CHUNKS=self._chunks,
                    K=self.index_topk,
                )
            dsa_gqa_inference_kernels._partial_kernel[(self.batch_size, 1, self._slices)](
                query,
                key_cache,
                value_cache,
                block_table,
                requests,
                self.indices,
                context_lengths,
                self._partial,
                self._maxima,
                self._sums,
                query.stride(0),
                query.stride(1),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                value_cache.stride(0),
                value_cache.stride(1),
                value_cache.stride(2),
                block_table.stride(0),
                self.indices.stride(0),
                key_cache.shape[0],
                block_table.shape[1],
                self.index_topk,
                self._slice_len,
                self._slices,
                softmax_scale * dsa_gqa_inference_kernels.LOG2E,
                HEADS_PER_GROUP=self.num_query_heads,
                HEAD_TILE=max(16, triton.next_power_of_2(self.num_query_heads)),
                HEAD_DIM=self.head_dim,
                HEAD_DIM_TILE=triton.next_power_of_2(self.head_dim),
                VALUE_DIM=self.head_dim,
                VALUE_DIM_TILE=triton.next_power_of_2(self.head_dim),
                PAGE_SIZE=self.page_size,
                BLOCK=dsa_gqa_inference_kernels.BLOCK_N,
                num_warps=4,
                num_stages=2,
            )
            dsa_gqa_inference_kernels._combine_kernel[(self.batch_size, self.num_query_heads)](
                self._partial,
                self._maxima,
                self._sums,
                self.output,
                self.output.stride(0),
                self.output.stride(1),
                self.num_query_heads,
                self._slices,
                VALUE_DIM=self.head_dim,
                VALUE_DIM_TILE=triton.next_power_of_2(self.head_dim),
                num_warps=4,
            )
        return self.output
