# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paged Triton DSA prefill and Gluon/Triton decode."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.contexts.attention_context.dsa_metadata import DSARowPlan
from megatron.core.tuning import use_deterministic_mode

SCORE_BUFFER_BYTES = 1 << 30


@torch.no_grad()
def _sparse_attention_rows(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    plan: DSARowPlan,
    max_context_length: int,
    index_topk: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Route prefill rows to their causal top-k keys and attend over the paged cache."""
    from . import dsa_gqa_inference_kernels

    num_rows, num_query_heads, _ = query.shape
    block_table = plan.block_table.to(torch.int32).contiguous()
    plan = plan._replace(block_table=block_table)
    index_query = index_query.contiguous()
    topk_indices = _materialized_topk(
        index_query=index_query,
        index_key_cache=index_key_cache,
        plan=plan,
        max_context_length=max_context_length,
        index_topk=index_topk,
    )
    output = query.new_empty((num_rows, num_query_heads, value_cache.size(-1)))
    dsa_gqa_inference_kernels.dsa_gqa_sparse_attention(
        query,
        key_cache,
        value_cache,
        block_table,
        plan.row_requests,
        plan.row_lengths,
        topk_indices,
        softmax_scale,
        out=output,
    )
    return output


def _materialized_topk(
    *,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    plan: DSARowPlan,
    max_context_length: int,
    index_topk: int,
) -> torch.Tensor:
    """Each row's top-k by scoring its keys into a buffer and selecting from it."""
    from megatron.core.transformer.experimental_attention_variant import dsa_gqa_inference_kernels

    num_rows = plan.row_requests.numel()
    width = max_context_length
    # Score and select in windows of rows so the score buffer stays bounded at long context;
    # the window count depends only on the padded rows and the width (CUDA-graph safe).
    window = max(1, min(num_rows, SCORE_BUFFER_BYTES // (width * torch.float32.itemsize)))
    scores = torch.empty((window, width), dtype=torch.float32, device=index_query.device)
    topk_indices = torch.empty((num_rows, index_topk), dtype=torch.int32, device=index_query.device)
    for start in range(0, num_rows, window):
        stop = min(start + window, num_rows)
        window_scores = scores[: stop - start]
        dsa_gqa_inference_kernels.dsa_gqa_indexer_scores(
            index_query,
            index_key_cache,
            plan.block_table,
            plan.row_lengths,
            plan.block_starts,
            plan.block_rows,
            plan.block_requests,
            out=window_scores,
            fill_invalid=True,
            row_offset=start,
        )
        selected = window_scores.topk(min(index_topk, width), dim=-1).indices.to(torch.int32)
        topk_indices[start:stop] = F.pad(selected, (0, index_topk - selected.size(1)), value=-1)
    return topk_indices


@torch.no_grad()
def dynamic_sparse_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    inference_context: BaseInferenceContext,
    index_topk: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Simplified-DSA attention over paged caches for a decode, prefill, or mixed batch.

    Every token is a row that routes to the top-k of its causal prefix
    (``DSAMetadata.plan``), so all step types share one path with
    launch sizes fixed by the padded batch and can run as CUDA graphs. The caches already
    hold this step's keys, values, and indexer keys.

    Args:
        query: Query ``[T,1,Hq,D]`` or ``[T,Hq,D]`` for the padded active tokens.
        key_cache: Paged keys ``[blocks,block_size,G,D]``.
        value_cache: Paged values ``[blocks,block_size,G,Dv]``.
        index_query: Indexer queries ``[T,1,1,Di]``.
        index_key_cache: Paged indexer keys ``[blocks,block_size,Di]``.
        inference_context: Dynamic inference context for this step.
        index_topk: Number of keys each query attends to.
        softmax_scale: Attention softmax scale.

    Returns:
        Attention output ``[T,1,Hq*Dv]``; rows for padding tokens are zero.
    """
    if inference_context.num_speculative_tokens:
        raise NotImplementedError(
            "DSA inference decodes one token per request; MTP speculative decoding is not "
            "supported."
        )
    if query.dim() == 3:
        query = query.unsqueeze(1)
    num_tokens, _, num_query_heads, _ = query.shape
    metadata = inference_context.dsa_metadata
    plan = metadata.plan()
    if plan.row_requests.numel() != num_tokens:
        raise RuntimeError(
            f"DSA row metadata covers {plan.row_requests.numel()} tokens, but the step has "
            f"{num_tokens}."
        )
    # Graphed steps use the graph's static bound; eager steps the batch's longest context.
    max_context_length = (
        inference_context.max_sequence_length
        if inference_context.using_cuda_graph_this_step()
        else max(1, metadata.step().max_kv_length)
    )
    output = _native_sparse_attention(
        query=query[:, 0],
        key_cache=key_cache,
        value_cache=value_cache,
        index_query=index_query.reshape(num_tokens, -1),
        index_key_cache=index_key_cache,
        inference_context=inference_context,
        plan=plan,
        max_context_length=max_context_length,
        index_topk=index_topk,
        softmax_scale=softmax_scale,
    )
    return output.view(num_tokens, 1, num_query_heads * value_cache.size(-1))


@torch.no_grad()
def _native_sparse_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    inference_context: BaseInferenceContext,
    plan: DSARowPlan,
    max_context_length: int,
    index_topk: int,
    softmax_scale: float,
) -> torch.Tensor:
    rows = plan.decode_lengths.numel()
    max_context = inference_context.max_sequence_length
    metadata = inference_context.dsa_metadata
    if metadata.topk_workspace is None:
        from flashinfer.topk_varlen.kernels.gvr2_topk_host import workspace_bytes

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm DSA decode before capturing its shared top-k scratch")
        metadata.topk_workspace = {
            "gvr2_workspace": torch.zeros(workspace_bytes(), dtype=torch.uint8, device=query.device)
        }
    lengths = plan.decode_lengths
    native = dsa_gqa_decode(
        query=query[:rows],
        index_query=index_query[:rows].contiguous(),
        index_key_cache=index_key_cache,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=plan.block_table,
        context_lengths=lengths,
        index_topk=index_topk,
        max_context=max_context,
        topk_workspace=metadata.topk_workspace,
        request_indices=plan.row_requests[:rows],
        softmax_scale=softmax_scale,
    )
    if plan.decode_only:
        return native

    # Mixed steps retain the prefill implementation; decode rows are computed only
    # by native decode, with zero lengths in the prefill score/attention path.
    prefill_lengths = plan.row_lengths.clone()
    prefill_lengths[:rows] = torch.where(lengths > 0, 0, prefill_lengths[:rows])
    output = _sparse_attention_rows(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        index_query=index_query,
        index_key_cache=index_key_cache,
        plan=plan._replace(row_lengths=prefill_lengths),
        max_context_length=max_context_length,
        index_topk=index_topk,
        softmax_scale=softmax_scale,
    )
    output[:rows] = torch.where((lengths > 0)[:, None, None], native, output[:rows])
    return output


@torch.no_grad()
def dsa_gqa_decode(
    *,
    query: torch.Tensor,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lengths: torch.Tensor,
    index_topk: int,
    max_context: int,
    topk_workspace: dict[str, torch.Tensor],
    softmax_scale: float,
    request_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Blackwell DSA decode with forward-local activation buffers.

    Args:
        query: BF16 attention queries ``[batch, heads, head_dim]``.
        index_query: Contiguous BF16 indexer queries ``[batch, 128]``.
        index_key_cache: Contiguous BF16 indexer keys ``[blocks, page_size, 128]``.
        key_cache: Paged BF16 keys ``[blocks, page_size, 1, head_dim]``.
        value_cache: Paged BF16 values with the same shape as key_cache.
        block_table: INT32 request-to-page mapping with compact rows.
        context_lengths: INT32 valid key counts ``[batch]``.
        index_topk: Number of selected keys: 512, 1024, or 2048.
        max_context: Static context capacity, at most 1048576.
        topk_workspace: Preinitialized FlashInfer scratch, shared by serial calls.
            Concurrent calls must use separate scratch buffers.
        softmax_scale: Attention softmax scale.
        request_indices: INT32 request IDs for each row; defaults to row indices.

    Returns:
        BF16 attention output with the same shape as query. The allocator owns
        intermediate lifetimes, including allocations in a CUDA graph pool.
    """
    import triton
    from flashinfer import top_k_varlen
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    from . import dsa_gqa_inference_kernels as kernels

    if use_deterministic_mode():
        raise RuntimeError("FlashInfer GVR2 top-k does not support deterministic mode")

    device = query.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        raise ValueError("Native Triton DSA decode requires SM100 or SM103")
    if query.ndim != 3 or key_cache.ndim != 4:
        raise ValueError("Expected THD query and paged BHGD key cache")
    batch_size, num_query_heads, head_dim = query.shape
    page_size = key_cache.shape[1]
    if batch_size <= 0 or num_query_heads <= 0 or head_dim <= 0 or head_dim % 32:
        raise ValueError("Batch/heads must be positive; head_dim must be a positive multiple of 32")
    if index_topk not in (512, 1024, 2048) or not 0 < max_context <= 1048576:
        raise ValueError(
            "Supported top-k values are 512/1024/2048, with context capacity 1..1048576"
        )
    if page_size <= 0 or page_size % 128:
        raise ValueError("Native Triton DSA requires pages divisible by 128")
    requests = (
        torch.arange(batch_size, dtype=torch.int32, device=device)
        if request_indices is None
        else request_indices
    )
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
    if any(t.device != device for t in tensors):
        raise ValueError("Every input must be on the query CUDA device")
    if query.dtype != torch.bfloat16 or query.stride(-1) != 1:
        raise ValueError("query must be BF16 [batch_size,num_query_heads,head_dim]")
    if query.stride(1) != head_dim:
        raise ValueError("Query heads must occupy compact rows")
    if (
        index_query.shape != (batch_size, 128)
        or index_query.dtype != torch.bfloat16
        or not index_query.is_contiguous()
    ):
        raise ValueError("index_query must be contiguous BF16 [batch_size,128]")
    if (
        index_key_cache.ndim != 3
        or index_key_cache.shape[1:] != (page_size, 128)
        or index_key_cache.dtype != torch.bfloat16
        or not index_key_cache.is_contiguous()
    ):
        raise ValueError("index_key_cache must be contiguous BF16 [blocks,page_size,128]")
    if key_cache.shape != value_cache.shape or key_cache.shape != (
        index_key_cache.shape[0],
        page_size,
        1,
        head_dim,
    ):
        raise ValueError("Key/value caches must be [blocks,page_size,1,head_dim]")
    if key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16:
        raise ValueError("Key/value caches must use BF16")
    if key_cache.stride(-1) != 1 or value_cache.stride(-1) != 1:
        raise ValueError("Cache head dimensions must have unit stride")
    for name, tensor in (("context_lengths", context_lengths), ("request_indices", requests)):
        if (
            tensor.dtype != torch.int32
            or tensor.shape != (batch_size,)
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous int32 [batch_size]")
    if block_table.dtype != torch.int32 or block_table.ndim != 2 or block_table.stride(-1) != 1:
        raise ValueError("block_table must have compact int32 rows")

    score_width = triton.cdiv(max(max_context, index_topk), 4) * 4
    scores = torch.empty(batch_size, score_width, dtype=torch.float32, device=device)
    indices = torch.empty(batch_size, index_topk, dtype=torch.int32, device=device)
    output = torch.empty(batch_size, num_query_heads, head_dim, dtype=torch.bfloat16, device=device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    slices = kernels._num_slices(batch_size, 1, index_topk, sms)
    slice_len = triton.cdiv(index_topk, slices)
    partial_rows = batch_size * slices * num_query_heads
    partial = torch.empty(partial_rows, head_dim, dtype=torch.float32, device=device)
    maxima = torch.empty(partial_rows, dtype=torch.float32, device=device)
    sums = torch.empty_like(maxima)
    descriptor_layout = gl.NVMMASharedLayout(
        swizzle_byte_width=128, element_bitwidth=16, rank=2, transposed=False
    )
    query_descriptor = TensorDescriptor.from_tensor(index_query, [8, 128], descriptor_layout)
    key_descriptor = TensorDescriptor.from_tensor(
        index_key_cache.view(-1, 128), [128, 128], descriptor_layout
    )
    score_workers, score_stages = 8, 3

    with torch.cuda.device(device):
        kernels.score_tcgen[(batch_size, score_workers)](
            query_descriptor,
            key_descriptor,
            block_table,
            context_lengths,
            requests,
            scores,
            stride_table=block_table.stride(0),
            WIDTH=score_width,
            PAGE=page_size,
            WORKERS=score_workers,
            STAGES=score_stages,
            num_warps=4,
        )
        top_k_varlen(
            scores,
            context_lengths,
            index_topk,
            backend="gvr_2",
            out_indices=indices,
            workspace=topk_workspace,
        )
        kernels._partial_kernel[(batch_size, 1, slices)](
            query,
            key_cache,
            value_cache,
            block_table,
            requests,
            indices,
            context_lengths,
            partial,
            maxima,
            sums,
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            block_table.stride(0),
            indices.stride(0),
            key_cache.shape[0],
            block_table.shape[1],
            index_topk,
            slice_len,
            slices,
            softmax_scale * kernels.LOG2E,
            HEADS_PER_GROUP=num_query_heads,
            HEAD_TILE=max(16, triton.next_power_of_2(num_query_heads)),
            HEAD_DIM=head_dim,
            HEAD_DIM_TILE=triton.next_power_of_2(head_dim),
            VALUE_DIM=head_dim,
            VALUE_DIM_TILE=triton.next_power_of_2(head_dim),
            PAGE_SIZE=page_size,
            BLOCK=kernels.BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        kernels._combine_kernel[(batch_size, num_query_heads)](
            partial,
            maxima,
            sums,
            output,
            output.stride(0),
            output.stride(1),
            num_query_heads,
            slices,
            VALUE_DIM=head_dim,
            VALUE_DIM_TILE=triton.next_power_of_2(head_dim),
            num_warps=4,
        )
    return output
