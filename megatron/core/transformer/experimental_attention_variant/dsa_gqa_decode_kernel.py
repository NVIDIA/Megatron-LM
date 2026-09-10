# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""BF16 CuTeDSL paged decode kernels for simplified DSA/NVSA.

The kernel computes one FP32 score row per decode request. The indexer head
dimension is specialized as a ``cutlass.Constexpr``. K is read directly from
the paged cache through ``block_table``; TopK remains a separate caller operation.
The score width and its row stride are runtime values, so growing a request's
KV length does not require recompiling the indexer.

The sparse-attention kernel consumes those TopK indices and fuses paged K/V
gather, QK, FP32 softmax, and PV into one launch over all decode requests.
"""

from __future__ import annotations

import torch

# pylint: disable=missing-function-docstring

try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack

    HAVE_CUTEDSL = True
except ImportError:
    cutlass = None
    cute = None
    utils = None
    from_dlpack = None
    HAVE_CUTEDSL = False


THREADS = 128
WARPS = THREADS // 32
TOKENS_PER_WARP = 8
TOKENS_PER_BLOCK = WARPS * TOKENS_PER_WARP
ATTENTION_THREADS = 256
ATTENTION_WARPS = ATTENTION_THREADS // 32
LOG2E = 1.4426950408889634


if HAVE_CUTEDSL:

    class _DecodeIndexer:
        def __init__(self, batch_size: int, page_size: int, max_pages: int, num_pages: int) -> None:
            self.batch_size = batch_size
            self.page_size = page_size
            self.max_pages = max_pages
            self.num_pages = num_pages

        @cute.kernel
        def score_kernel(
            self,
            query: cute.Tensor,
            key_cache: cute.Tensor,
            block_table: cute.Tensor,
            context_lengths: cute.Tensor,
            output: cute.Tensor,
            index_head_dim: cutlass.Constexpr,
        ):
            thread, _, _ = cute.arch.thread_idx()
            score_block, request, _ = cute.arch.block_idx()
            lane = thread % 32
            warp_id = cute.arch.warp_idx()

            allocator = utils.SmemAllocator()
            shared_query = allocator.allocate_tensor(
                cutlass.BFloat16,
                cute.make_ordered_layout((index_head_dim,), order=(0,)),
                byte_alignment=16,
            )
            for item in cutlass.range_constexpr((index_head_dim + THREADS - 1) // THREADS):
                dimension = thread + item * THREADS
                if dimension < index_head_dim:
                    shared_query[dimension] = query[request, dimension]
            cute.arch.barrier()

            context_length = context_lengths[request]
            first_token = score_block * TOKENS_PER_BLOCK
            for token_in_warp in cutlass.range_constexpr(TOKENS_PER_WARP):
                logical_token = first_token + token_in_warp * WARPS + warp_id
                if logical_token < output.shape[1]:
                    score = -cutlass.Float32.inf
                    if logical_token < context_length:
                        logical_page = logical_token // self.page_size
                        token_in_page = logical_token - logical_page * self.page_size
                        physical_page = cutlass.Int32(-1)
                        if logical_page < self.max_pages:
                            physical_page = block_table[request, logical_page]
                        if physical_page >= 0 and physical_page < self.num_pages:
                            partial = cutlass.Float32(0.0)
                            for item in cutlass.range_constexpr(index_head_dim // 32):
                                dimension = lane + item * 32
                                partial = partial + cutlass.Float32(
                                    shared_query[dimension]
                                ) * cutlass.Float32(
                                    key_cache[physical_page, token_in_page, dimension]
                                )
                            score = cute.arch.warp_reduction_sum(partial)
                    if lane == 0:
                        output[request, logical_token] = score

        @cute.jit
        def __call__(
            self,
            query,
            key_cache,
            block_table,
            context_lengths,
            output,
            index_head_dim: cutlass.Constexpr,
        ):
            self.score_kernel(
                query, key_cache, block_table, context_lengths, output, index_head_dim
            ).launch(
                grid=(
                    (output.shape[1] + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK,
                    self.batch_size,
                    1,
                ),
                block=(THREADS, 1, 1),
            )

    @cute.jit
    def _block_reduce_max(value: cutlass.Float32, scratch: cute.Tensor, thread: cutlass.Int32):
        lane = thread % 32
        warp_id = thread // 32
        warp_value = cute.arch.warp_reduction_max(value)
        if lane == 0:
            scratch[warp_id] = warp_value
        cute.arch.barrier()

        block_value = -cutlass.Float32.inf
        if warp_id == 0 and lane < ATTENTION_WARPS:
            block_value = scratch[lane]
        block_value = cute.arch.warp_reduction_max(block_value)
        if thread == 0:
            scratch[0] = block_value
        cute.arch.barrier()
        return scratch[0]

    @cute.jit
    def _block_reduce_sum(value: cutlass.Float32, scratch: cute.Tensor, thread: cutlass.Int32):
        lane = thread % 32
        warp_id = thread // 32
        warp_value = cute.arch.warp_reduction_sum(value)
        if lane == 0:
            scratch[warp_id] = warp_value
        cute.arch.barrier()

        block_value = cutlass.Float32(0.0)
        if warp_id == 0 and lane < ATTENTION_WARPS:
            block_value = scratch[lane]
        block_value = cute.arch.warp_reduction_sum(block_value)
        if thread == 0:
            scratch[0] = block_value
        cute.arch.barrier()
        return scratch[0]

    class _DecodeAttention:
        def __init__(self, batch_size: int, page_size: int, max_pages: int, num_pages: int) -> None:
            self.batch_size = batch_size
            self.page_size = page_size
            self.max_pages = max_pages
            self.num_pages = num_pages

        @cute.kernel
        def attention_kernel(
            self,
            query: cute.Tensor,
            key_cache: cute.Tensor,
            value_cache: cute.Tensor,
            block_table: cute.Tensor,
            topk_indices: cute.Tensor,
            context_lengths: cute.Tensor,
            output: cute.Tensor,
            num_query_heads: cutlass.Constexpr,
            num_query_groups: cutlass.Constexpr,
            query_head_dim: cutlass.Constexpr,
            value_head_dim: cutlass.Constexpr,
            topk_width: cutlass.Constexpr,
            softmax_scale: cutlass.Constexpr,
        ):
            thread, _, _ = cute.arch.thread_idx()
            request, query_head, _ = cute.arch.block_idx()
            lane = thread % 32
            warp_id = cute.arch.warp_idx()
            query_group_size = num_query_heads // num_query_groups
            key_group = query_head // query_group_size

            allocator = utils.SmemAllocator()
            shared_query = allocator.allocate_tensor(
                cutlass.BFloat16,
                cute.make_ordered_layout((query_head_dim,), order=(0,)),
                byte_alignment=16,
            )
            shared_scores = allocator.allocate_tensor(
                cutlass.Float32,
                cute.make_ordered_layout((topk_width,), order=(0,)),
                byte_alignment=16,
            )
            shared_pages = allocator.allocate_tensor(
                cutlass.Int32,
                cute.make_ordered_layout((topk_width,), order=(0,)),
                byte_alignment=16,
            )
            shared_offsets = allocator.allocate_tensor(
                cutlass.Int32,
                cute.make_ordered_layout((topk_width,), order=(0,)),
                byte_alignment=16,
            )
            reduce_scratch = allocator.allocate_tensor(
                cutlass.Float32,
                cute.make_ordered_layout((ATTENTION_WARPS,), order=(0,)),
                byte_alignment=16,
            )

            for item in cutlass.range_constexpr(
                (query_head_dim + ATTENTION_THREADS - 1) // ATTENTION_THREADS
            ):
                dimension = thread + item * ATTENTION_THREADS
                if dimension < query_head_dim:
                    shared_query[dimension] = query[request, query_head, dimension]

            context_length = context_lengths[request]
            valid_count = context_length
            if valid_count > topk_width:
                valid_count = topk_width
            slot = thread
            while slot < topk_width:
                physical_page = cutlass.Int32(-1)
                token_in_page = cutlass.Int32(0)
                if slot < valid_count:
                    logical_token = topk_indices[request, slot]
                    if logical_token >= 0 and logical_token < context_length:
                        logical_page = logical_token // self.page_size
                        token_in_page = logical_token - logical_page * self.page_size
                        if logical_page < self.max_pages:
                            page = block_table[request, logical_page]
                            if page >= 0 and page < self.num_pages:
                                physical_page = page
                shared_pages[slot] = physical_page
                shared_offsets[slot] = token_in_page
                slot = slot + ATTENTION_THREADS
            cute.arch.barrier()

            slot = warp_id
            while slot < topk_width:
                score = -cutlass.Float32.inf
                physical_page = shared_pages[slot]
                if physical_page >= 0:
                    token_in_page = shared_offsets[slot]
                    partial = cutlass.Float32(0.0)
                    for item in cutlass.range_constexpr(query_head_dim // 32):
                        dimension = lane + item * 32
                        partial = partial + cutlass.Float32(
                            shared_query[dimension]
                        ) * cutlass.Float32(
                            key_cache[physical_page, token_in_page, key_group, dimension]
                        )
                    score = cute.arch.warp_reduction_sum(partial)
                if lane == 0:
                    shared_scores[slot] = score
                slot = slot + ATTENTION_WARPS
            cute.arch.barrier()

            local_max = -cutlass.Float32.inf
            slot = thread
            while slot < topk_width:
                local_max = cute.arch.fmax(local_max, shared_scores[slot])
                slot = slot + ATTENTION_THREADS
            row_max = _block_reduce_max(local_max, reduce_scratch, thread)

            local_sum = cutlass.Float32(0.0)
            slot = thread
            while slot < topk_width:
                weight = cute.math.exp2(
                    (shared_scores[slot] - row_max) * cutlass.Float32(softmax_scale * LOG2E),
                    fastmath=True,
                )
                shared_scores[slot] = weight
                local_sum = local_sum + weight
                slot = slot + ATTENTION_THREADS
            row_sum = _block_reduce_sum(local_sum, reduce_scratch, thread)

            dimension = thread
            while dimension < value_head_dim:
                accumulator = cutlass.Float32(0.0)
                slot = cutlass.Int32(0)
                while slot < valid_count:
                    physical_page = shared_pages[slot]
                    if physical_page >= 0:
                        probability = (shared_scores[slot] / row_sum).to(cutlass.BFloat16)
                        accumulator = accumulator + cutlass.Float32(probability) * cutlass.Float32(
                            value_cache[physical_page, shared_offsets[slot], key_group, dimension]
                        )
                    slot = slot + 1
                output[request, query_head, dimension] = accumulator.to(cutlass.BFloat16)
                dimension = dimension + ATTENTION_THREADS

        @cute.jit
        def __call__(
            self,
            query,
            key_cache,
            value_cache,
            block_table,
            topk_indices,
            context_lengths,
            output,
            num_query_heads: cutlass.Constexpr,
            num_query_groups: cutlass.Constexpr,
            query_head_dim: cutlass.Constexpr,
            value_head_dim: cutlass.Constexpr,
            topk_width: cutlass.Constexpr,
            softmax_scale: cutlass.Constexpr,
        ):
            self.attention_kernel(
                query,
                key_cache,
                value_cache,
                block_table,
                topk_indices,
                context_lengths,
                output,
                num_query_heads,
                num_query_groups,
                query_head_dim,
                value_head_dim,
                topk_width,
                softmax_scale,
            ).launch(grid=(self.batch_size, num_query_heads, 1), block=(ATTENTION_THREADS, 1, 1))


_COMPILED: dict[tuple, object] = {}
_OPERATORS: dict[tuple, object] = {}
_ATTENTION_COMPILED: dict[tuple, object] = {}
_ATTENTION_OPERATORS: dict[tuple, object] = {}


def _validate(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lengths: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if not HAVE_CUTEDSL:
        raise RuntimeError("CuTeDSL is required for the DSA-GQA decode indexer kernel.")
    tensors = (query, key_cache, block_table, context_lengths, output)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("All decode indexer tensors must be CUDA tensors.")
    if any(tensor.device != query.device for tensor in tensors[1:]):
        raise ValueError("All decode indexer tensors must be on the same CUDA device.")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("All decode indexer tensors must be contiguous.")
    if query.dtype != torch.bfloat16 or query.dim() != 2:
        raise TypeError("query must be a contiguous BF16 tensor with shape [batch,head_dim].")
    index_head_dim = query.size(1)
    if index_head_dim <= 0 or index_head_dim % 32 != 0:
        raise ValueError("indexer head dimension must be a positive multiple of 32.")
    if (
        key_cache.dtype != torch.bfloat16
        or key_cache.dim() != 3
        or key_cache.size(2) != index_head_dim
    ):
        raise TypeError(
            "key_cache must be contiguous BF16 with shape [pages,page_size,head_dim] "
            "matching query."
        )
    if block_table.dtype != torch.int32 or block_table.dim() != 2:
        raise TypeError("block_table must be contiguous int32 with shape [batch,max_pages].")
    if block_table.size(0) != query.size(0):
        raise ValueError("block_table batch dimension must match query.")
    if context_lengths.dtype != torch.int32 or context_lengths.shape != (query.size(0),):
        raise TypeError("context_lengths must be contiguous int32 with shape [batch].")
    if output.dtype != torch.float32 or output.dim() != 2 or output.size(0) != query.size(0):
        raise TypeError("out must be contiguous FP32 with shape [batch,max_context_length].")
    if key_cache.size(1) <= 0:
        raise ValueError("key_cache page size must be positive.")
    if query.size(0) and output.size(1):
        major, _ = torch.cuda.get_device_capability(query.device)
        if major < 8:
            raise RuntimeError("The CuTeDSL decode indexer requires SM80 or newer.")


def _compile(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lengths: torch.Tensor,
    output: torch.Tensor,
):
    key = (tuple(query.shape), tuple(key_cache.shape), tuple(block_table.shape), query.device.index)
    compiled = _COMPILED.get(key)
    if compiled is None:
        index_head_dim = query.size(1)
        operator = _DecodeIndexer(
            query.size(0), key_cache.size(1), block_table.size(1), key_cache.size(0)
        )
        _OPERATORS[key] = operator
        # Both the width and the outer row stride change at every decode step.
        # Keep the contiguous layout contract, but make its inner extent dynamic.
        dynamic_output = from_dlpack(
            output, assumed_align=4, enable_tvm_ffi=True
        ).mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=1)
        compiled = cute.compile(
            operator,
            from_dlpack(query, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(key_cache, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(block_table, assumed_align=4, enable_tvm_ffi=True),
            from_dlpack(context_lengths, assumed_align=4, enable_tvm_ffi=True),
            dynamic_output,
            index_head_dim,
            options="--enable-tvm-ffi",
        )
        _COMPILED[key] = compiled
    return compiled


@torch.no_grad()
def dsa_gqa_decode_indexer_score(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lengths: torch.Tensor,
    *,
    out: torch.Tensor,
) -> None:
    """Compute paged BF16 simplified-indexer decode scores.

    Args:
        query: BF16 indexer query with shape ``[batch,head_dim]``.
        key_cache: Paged BF16 indexer keys with shape
            ``[num_pages,page_size,head_dim]``.
        block_table: Int32 physical-page table with shape ``[batch,max_pages]``.
        context_lengths: Int32 valid K lengths with shape ``[batch]``.
        out: Caller-owned FP32 scores with shape ``[batch,max_context_length]``.
            Invalid positions are written as negative infinity.
    """
    _validate(query, key_cache, block_table, context_lengths, out)
    if query.size(0) == 0 or out.size(1) == 0:
        return
    compiled = _compile(query, key_cache, block_table, context_lengths, out)
    compiled(query, key_cache, block_table, context_lengths, out)


def _validate_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    context_lengths: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if not HAVE_CUTEDSL:
        raise RuntimeError("CuTeDSL is required for the DSA-GQA decode attention kernel.")
    tensors = (query, key_cache, value_cache, block_table, topk_indices, context_lengths, output)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("All decode attention tensors must be CUDA tensors.")
    if any(tensor.device != query.device for tensor in tensors[1:]):
        raise ValueError("All decode attention tensors must be on the same CUDA device.")
    if any(not tensor.is_contiguous() for tensor in tensors[1:]):
        raise ValueError("Decode attention cache, metadata, and output tensors must be contiguous.")
    if query.dtype != torch.bfloat16 or query.dim() != 3:
        raise TypeError("query must be BF16 with shape [batch,heads,head_dim].")
    batch_size, num_query_heads, query_head_dim = query.shape
    if query_head_dim <= 0 or query_head_dim % 32 != 0:
        raise ValueError("query head dimension must be a positive multiple of 32.")
    if (
        query.stride(-1) != 1
        or query.stride(-2) != query_head_dim
        or query.stride(0) < num_query_heads * query_head_dim
        or query.data_ptr() % 16 != 0
    ):
        raise ValueError(
            "query may use a pitched token stride but must have compact, 16-byte-aligned "
            "head rows."
        )
    if key_cache.dtype != torch.bfloat16 or key_cache.dim() != 4:
        raise TypeError(
            "key_cache must be contiguous BF16 with shape [pages,page_size,groups,head_dim]."
        )
    if value_cache.dtype != torch.bfloat16 or value_cache.dim() != 4:
        raise TypeError(
            "value_cache must be contiguous BF16 with shape "
            "[pages,page_size,groups,value_head_dim]."
        )
    if key_cache.shape[:3] != value_cache.shape[:3] or key_cache.size(3) != query_head_dim:
        raise ValueError("Key/value cache geometry must agree and key head_dim must match query.")
    num_query_groups = key_cache.size(2)
    if num_query_groups <= 0 or num_query_heads % num_query_groups != 0:
        raise ValueError("Query heads must be divisible by the number of KV groups.")
    if block_table.dtype != torch.int32 or block_table.dim() != 2:
        raise TypeError("block_table must be contiguous int32 with shape [batch,max_pages].")
    if block_table.size(0) != batch_size:
        raise ValueError("block_table batch dimension must match query.")
    if (
        topk_indices.dtype != torch.int32
        or topk_indices.dim() != 2
        or topk_indices.size(0) != batch_size
        or topk_indices.size(1) == 0
    ):
        raise TypeError("topk_indices must be contiguous int32 with shape [batch,topk].")
    if context_lengths.dtype != torch.int32 or context_lengths.shape != (batch_size,):
        raise TypeError("context_lengths must be contiguous int32 with shape [batch].")
    if output.dtype != torch.bfloat16 or output.shape != (
        batch_size,
        num_query_heads,
        value_cache.size(3),
    ):
        raise TypeError("out must be contiguous BF16 with shape [batch,heads,value_head_dim].")
    if key_cache.size(1) <= 0:
        raise ValueError("KV cache page size must be positive.")
    if batch_size:
        major, _ = torch.cuda.get_device_capability(query.device)
        if major < 8:
            raise RuntimeError("The CuTeDSL decode attention kernel requires SM80 or newer.")


def _compile_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    context_lengths: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
):
    key = (
        tuple(query.shape),
        tuple(query.stride()),
        tuple(key_cache.shape),
        tuple(value_cache.shape),
        tuple(block_table.shape),
        tuple(topk_indices.shape),
        tuple(output.shape),
        float(softmax_scale),
        query.device.index,
    )
    compiled = _ATTENTION_COMPILED.get(key)
    if compiled is None:
        batch_size, num_query_heads, query_head_dim = query.shape
        num_query_groups = key_cache.size(2)
        value_head_dim = value_cache.size(3)
        topk_width = topk_indices.size(1)
        operator = _DecodeAttention(
            batch_size, key_cache.size(1), block_table.size(1), key_cache.size(0)
        )
        _ATTENTION_OPERATORS[key] = operator
        compiled = cute.compile(
            operator,
            from_dlpack(query, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(key_cache, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(value_cache, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(block_table, assumed_align=4, enable_tvm_ffi=True),
            from_dlpack(topk_indices, assumed_align=4, enable_tvm_ffi=True),
            from_dlpack(context_lengths, assumed_align=4, enable_tvm_ffi=True),
            from_dlpack(output, assumed_align=16, enable_tvm_ffi=True),
            num_query_heads,
            num_query_groups,
            query_head_dim,
            value_head_dim,
            topk_width,
            float(softmax_scale),
            options="--enable-tvm-ffi",
        )
        _ATTENTION_COMPILED[key] = compiled
    return compiled


@torch.no_grad()
def dsa_gqa_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    context_lengths: torch.Tensor,
    softmax_scale: float,
    *,
    out: torch.Tensor,
) -> None:
    """Run fused paged BF16 sparse attention for a batch of decode requests.

    Args:
        query: BF16 query with shape ``[batch,query_heads,head_dim]``.
        key_cache: Paged BF16 keys with shape
            ``[num_pages,page_size,query_groups,head_dim]``.
        value_cache: Paged BF16 values with shape
            ``[num_pages,page_size,query_groups,value_head_dim]``.
        block_table: Int32 physical-page table with shape ``[batch,max_pages]``.
        topk_indices: Int32 logical token positions with shape ``[batch,topk]``.
        context_lengths: Int32 valid K lengths with shape ``[batch]``.
        softmax_scale: Attention score scale specialized at compile time.
        out: Caller-owned BF16 output with shape
            ``[batch,query_heads,value_head_dim]``.
    """
    _validate_attention(
        query, key_cache, value_cache, block_table, topk_indices, context_lengths, out
    )
    if query.size(0) == 0:
        return
    compiled = _compile_attention(
        query,
        key_cache,
        value_cache,
        block_table,
        topk_indices,
        context_lengths,
        out,
        softmax_scale,
    )
    compiled(query, key_cache, value_cache, block_table, topk_indices, context_lengths, out)


__all__ = ["HAVE_CUTEDSL", "dsa_gqa_decode_attention", "dsa_gqa_decode_indexer_score"]
