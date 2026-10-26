# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paged Triton DSA prefill and Gluon/Triton decode."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from megatron.core.inference.contexts import BaseInferenceContext

SCORE_BUFFER_BYTES = 1 << 30


@torch.no_grad()
def _sparse_attention_rows(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_metadata: tuple,
    max_context_length: int,
    index_topk: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Route prefill rows to their causal top-k keys and attend over the paged cache."""
    from . import dsa_gqa_inference_kernels

    row_requests, row_lengths, block_starts, block_rows, block_requests, _ = row_metadata
    num_rows, num_query_heads, _ = query.shape
    block_table = block_table.to(torch.int32).contiguous()
    index_query = index_query.contiguous()
    topk_indices = _materialized_topk(
        index_query=index_query,
        index_key_cache=index_key_cache,
        block_table=block_table,
        row_metadata=row_metadata,
        max_context_length=max_context_length,
        index_topk=index_topk,
    )
    output = query.new_empty((num_rows, num_query_heads, value_cache.size(-1)))
    dsa_gqa_inference_kernels.dsa_gqa_sparse_attention(
        query,
        key_cache,
        value_cache,
        block_table,
        row_requests,
        row_lengths,
        topk_indices,
        softmax_scale,
        out=output,
    )
    return output


def _materialized_topk(
    *,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_metadata: tuple,
    max_context_length: int,
    index_topk: int,
) -> torch.Tensor:
    """Each row's top-k by scoring its keys into a buffer and selecting from it."""
    from megatron.core.transformer.experimental_attention_variant import dsa_gqa_inference_kernels

    row_requests, row_lengths, block_starts, block_rows, block_requests, _ = row_metadata
    num_rows = row_requests.numel()
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
            block_table,
            row_lengths,
            block_starts,
            block_rows,
            block_requests,
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
    block_table: torch.Tensor,
    inference_context: BaseInferenceContext,
    index_topk: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Simplified-DSA attention over paged caches for a decode, prefill, or mixed batch.

    Every token is a row that routes to the top-k of its causal prefix
    (``DynamicInferenceContext.dsa_row_metadata``), so all step types share one path with
    launch sizes fixed by the padded batch and can run as CUDA graphs. The caches already
    hold this step's keys, values, and indexer keys.

    Args:
        query: Query ``[T,1,Hq,D]`` or ``[T,Hq,D]`` for the padded active tokens.
        key_cache: Paged keys ``[blocks,block_size,G,D]``.
        value_cache: Paged values ``[blocks,block_size,G,Dv]``.
        index_query: Indexer queries ``[T,1,1,Di]``.
        index_key_cache: Paged indexer keys ``[blocks,block_size,Di]``.
        block_table: Per-request physical block ids ``[requests,max_blocks]``.
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
    row_metadata = inference_context.dsa_row_metadata()
    if row_metadata[0].numel() != num_tokens:
        raise RuntimeError(
            f"DSA row metadata covers {row_metadata[0].numel()} tokens, but the step has "
            f"{num_tokens}."
        )
    # Graphed steps use the graph's static bound; eager steps the batch's longest context.
    max_context_length = inference_context.active_attn_metadata["mha_metadata"].state_data[
        "max_seqlen_k"
    ]
    output = _native_sparse_attention(
        query=query[:, 0],
        key_cache=key_cache,
        value_cache=value_cache,
        index_query=index_query.reshape(num_tokens, -1),
        index_key_cache=index_key_cache,
        block_table=block_table,
        inference_context=inference_context,
        row_metadata=row_metadata,
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
    block_table: torch.Tensor,
    inference_context: BaseInferenceContext,
    row_metadata: tuple,
    max_context_length: int,
    index_topk: int,
    softmax_scale: float,
) -> torch.Tensor:
    from .dsa_gqa_triton_decode import TritonDSADecodeWorkspace

    rows = min(query.shape[0], inference_context.padded_active_request_count)
    max_context = inference_context.max_sequence_length
    cache_key = (
        index_key_cache.data_ptr(),
        rows,
        query.shape[1:],
        max_context,
        index_topk,
        key_cache.shape[1],
    )
    cache = inference_context._dsa_decode_workspaces
    if cache_key not in cache:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the native DSA workspace before CUDA graph capture")
        workspace = TritonDSADecodeWorkspace(
            batch_size=rows,
            num_query_heads=query.shape[1],
            head_dim=query.shape[2],
            index_topk=index_topk,
            max_context=max_context,
            page_size=key_cache.shape[1],
            device=query.device,
        )
        stable_query = torch.empty(
            rows, index_query.shape[1], dtype=index_query.dtype, device=query.device
        )
        cache[cache_key] = workspace, stable_query
    workspace, stable_query = cache[cache_key]
    stable_query.copy_(index_query[:rows])
    lengths = inference_context.dsa_decode_lengths[:rows]
    native = workspace(
        query=query[:rows],
        index_query=stable_query,
        index_key_cache=index_key_cache,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        context_lengths=lengths,
        request_indices=row_metadata[0][:rows],
        softmax_scale=softmax_scale,
    )
    if inference_context.is_decode_only():
        return native

    # Mixed steps retain the prefill implementation; decode rows are computed only
    # by the native workspace, with zero lengths in the prefill score/attention path.
    prefill_lengths = row_metadata[1].clone()
    prefill_lengths[:rows] = torch.where(lengths > 0, 0, prefill_lengths[:rows])
    prefill_metadata = (row_metadata[0], prefill_lengths, *row_metadata[2:])
    output = _sparse_attention_rows(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        index_query=index_query,
        index_key_cache=index_key_cache,
        block_table=block_table,
        row_metadata=prefill_metadata,
        max_context_length=max_context_length,
        index_topk=index_topk,
        softmax_scale=softmax_scale,
    )
    output[:rows] = torch.where((lengths > 0)[:, None, None], native, output[:rows])
    return output
