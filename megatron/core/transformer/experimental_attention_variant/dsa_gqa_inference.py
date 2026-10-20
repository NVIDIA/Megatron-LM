# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paged Triton and FlashInfer DSA inference, separate from the training module."""

from __future__ import annotations

import logging
import math
from functools import cache, partial
from typing import Optional

import torch

from megatron.core.inference.contexts import BaseInferenceContext

try:
    import deep_select
    HAVE_DEEP_SELECT = True
except ImportError:
    deep_select = None
    HAVE_DEEP_SELECT = False

logger = logging.getLogger(__name__)

def _paged_cache_rows(
    cache: torch.Tensor, block_table_row: torch.Tensor, num_tokens: int, block_size_tokens: int
) -> torch.Tensor:
    """Gather logical positions ``[0, num_tokens)`` from one request's paged cache."""
    positions = torch.arange(num_tokens, device=cache.device)
    physical_blocks = block_table_row[positions // block_size_tokens].long()
    return cache[physical_blocks, positions % block_size_tokens]


@cache
def _log_decode_topk_backend(use_deep_select: bool) -> None:
    """Say once per backend which top-k decode uses, since the fallback is much slower."""
    if use_deep_select:
        logger.info("DSA decode selects its top-k with DeepSelect.")
    else:
        logger.warning(
            "DSA decode selects its top-k with torch.topk over the whole score buffer; "
            "install DeepSelect (SM100/SM103, top-k <= 4096) for decode cost that follows "
            "the real context lengths."
        )


def _deep_select_decode_supported(device: torch.device, index_topk: int) -> bool:
    """DeepSelect's CUDA build has SM100a/SM103a kernels for top-k up to 4096."""
    return (
        HAVE_DEEP_SELECT
        and 0 < index_topk <= 4096
        and torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))
    )


@torch.no_grad()
def _sparse_decode(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lengths: torch.Tensor,
    max_context_length: int,
    index_topk: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Route each single-token decode query to its top-k cached keys and attend to them.

    ``max_context_length`` must bound every entry of ``kv_lengths``. DeepSelect reads
    each score row only up to its request's length, so the indexer skips the rest and
    the cost follows the real lengths; ``torch.topk`` reads whole rows, so the indexer
    fills positions past each length with negative infinity.

    Returns:
        Attention output ``[requests,Hq,Dv]``.
    """
    from megatron.core.transformer.experimental_attention_variant import dsa_gqa_decode_triton

    num_requests, num_query_heads, _ = query.shape
    kv_lengths = kv_lengths.to(torch.int32).contiguous()
    block_table = block_table.to(torch.int32).contiguous()
    use_deep_select = _deep_select_decode_supported(query.device, index_topk)
    _log_decode_topk_backend(use_deep_select)
    width = max_context_length
    if use_deep_select:
        # DeepSelect needs row strides aligned to its requirement and at least top-k
        # columns; the indexer never writes past a row's length.
        alignment = deep_select.get_stride_requirement()[0] // torch.float32.itemsize
        width = -(-max(width, index_topk) // alignment) * alignment
    scores = torch.empty((num_requests, width), dtype=torch.float32, device=query.device)
    dsa_gqa_decode_triton.dsa_gqa_decode_indexer_score(
        index_query.contiguous(),
        index_key_cache,
        block_table,
        kv_lengths,
        out=scores,
        fill_invalid=not use_deep_select,
    )
    if use_deep_select:
        _, topk_indices = deep_select.topk(
            scores,
            index_topk,
            end=kv_lengths,
            indices_type=torch.int32,
            return_value=False,
            idx_oob_fill_value=-1,
        )
    else:
        topk_indices = scores.topk(min(index_topk, width), dim=-1).indices.to(torch.int32)
        topk_indices = F.pad(topk_indices, (0, index_topk - topk_indices.size(1)), value=-1)
    output = query.new_empty((num_requests, num_query_heads, value_cache.size(-1)))
    dsa_gqa_decode_triton.dsa_gqa_decode_attention(
        query,
        key_cache,
        value_cache,
        block_table,
        topk_indices,
        kv_lengths,
        softmax_scale,
        out=output,
    )
    return output


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
    """Simplified-DSA attention over paged caches for a mixed decode/prefill batch.

    The dynamic context orders decode requests before prefill requests, and its
    caches already hold this step's keys, values, and indexer keys.

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
    from megatron.core.transformer.experimental_attention_variant import dsa_cute_kernels

    if inference_context.num_speculative_tokens:
        raise NotImplementedError(
            "Simplified CuTe DSA decodes one token per request; "
            "MTP speculative decoding is not supported."
        )
    if query.dim() == 3:
        query = query.unsqueeze(1)
    num_tokens, _, num_query_heads, _ = query.shape
    value_head_dim = value_cache.size(-1)
    output = query.new_zeros((num_tokens, 1, num_query_heads * value_head_dim))
    state_data = inference_context.active_attn_metadata["mha_metadata"].state_data
    index_query = index_query.reshape(num_tokens, -1)

    if inference_context.using_cuda_graph_this_step():
        # A captured graph replays on later steps, so use only sizes fixed by the graph's
        # padded batch and never read per-step metadata on the host. Padded requests have
        # zero KV length and produce zero output.
        if not inference_context.is_decode_only():
            raise NotImplementedError(
                "Simplified CuTe DSA supports decode-only CUDA graphs; "
                "use --decode-only-cuda-graphs."
            )
        num_decode = inference_context.padded_batch_dimensions.decode_req_count
        output[:num_decode, 0] = _sparse_decode(
            query=query[:num_decode, 0],
            key_cache=key_cache,
            value_cache=value_cache,
            index_query=index_query[:num_decode],
            index_key_cache=index_key_cache,
            block_table=block_table[:num_decode],
            kv_lengths=state_data["kv_seq_lengths"][:num_decode],
            max_context_length=state_data["max_seqlen_k"],
            index_topk=index_topk,
            softmax_scale=softmax_scale,
        ).flatten(1)
        return output

    request_count = inference_context.get_active_request_count()
    if request_count == 0:
        return output
    query_lengths = state_data["query_lengths"][:request_count].tolist()
    kv_lengths = state_data["kv_seq_lengths"][:request_count]

    num_decode = inference_context.num_decode_requests
    if num_decode:
        if any(length != 1 for length in query_lengths[:num_decode]):
            raise NotImplementedError("CuTe DSA decode supports one query token per request.")
        output[:num_decode, 0] = _sparse_decode(
            query=query[:num_decode, 0],
            key_cache=key_cache,
            value_cache=value_cache,
            index_query=index_query[:num_decode],
            index_key_cache=index_key_cache,
            block_table=block_table[:num_decode],
            kv_lengths=kv_lengths[:num_decode],
            max_context_length=int(kv_lengths[:num_decode].max().item()),
            index_topk=index_topk,
            softmax_scale=softmax_scale,
        ).flatten(1)

    query_start = num_decode
    block_size_tokens = inference_context.block_size_tokens
    for request_idx in range(num_decode, request_count):
        query_length = query_lengths[request_idx]
        kv_length = int(kv_lengths[request_idx].item())
        query_end = query_start + query_length
        request_blocks = block_table[request_idx]
        request_output = dsa_cute_kernels.run_sparse_attention_inference(
            query=query[query_start:query_end, 0],
            key=_paged_cache_rows(key_cache, request_blocks, kv_length, block_size_tokens),
            value=_paged_cache_rows(value_cache, request_blocks, kv_length, block_size_tokens),
            q_indexer=index_query[query_start:query_end],
            k_indexer=_paged_cache_rows(
                index_key_cache, request_blocks, kv_length, block_size_tokens
            ),
            key_offset=kv_length - query_length,
            indexer_topk=index_topk,
        )
        output[query_start:query_end, 0] = request_output.flatten(1)
        query_start = query_end
    return output

