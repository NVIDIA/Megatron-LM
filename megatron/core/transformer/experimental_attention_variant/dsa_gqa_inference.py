# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paged Triton and FlashInfer DSA inference, separate from the training module."""

from __future__ import annotations

import logging
from functools import cache
from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core.inference.contexts import BaseInferenceContext

try:
    import deep_select

    HAVE_DEEP_SELECT = True
except ImportError:
    deep_select = None
    HAVE_DEEP_SELECT = False

SCORE_BUFFER_BYTES = 1 << 30

logger = logging.getLogger(__name__)


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


@cache
def _log_fused_topk() -> None:
    logger.info("DSA selects its top-k with the fused selection (no score buffer).")


def _deep_select_decode_supported(device: torch.device, index_topk: int) -> bool:
    """DeepSelect's CUDA build has SM100a/SM103a kernels for top-k up to 4096."""
    return (
        HAVE_DEEP_SELECT
        and 0 < index_topk <= 4096
        and torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))
    )


def _select_topk(
    scores: torch.Tensor, row_lengths: torch.Tensor, index_topk: int, use_deep_select: bool
) -> torch.Tensor:
    """Each row's top-k positions among its first ``row_lengths[t]`` scores, -1 padded."""
    if use_deep_select:
        _, topk_indices = deep_select.topk(
            scores,
            index_topk,
            end=row_lengths,
            indices_type=torch.int32,
            return_value=False,
            idx_oob_fill_value=-1,
        )
        return topk_indices
    width = scores.size(1)
    topk_indices = scores.topk(min(index_topk, width), dim=-1).indices.to(torch.int32)
    return F.pad(topk_indices, (0, index_topk - topk_indices.size(1)), value=-1)


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
    filter_blocks: Optional[tuple] = None,
    decode_only: bool = False,
) -> torch.Tensor:
    """Route every row to its top-k cached keys and attend to them.

    ``row_metadata`` is ``(row_requests, row_lengths, block_starts, block_rows,
    block_requests, num_blocks)`` (``DynamicInferenceContext.dsa_row_metadata``), and
    ``max_context_length`` bounds every row length. DeepSelect reads each score row only up
    to its length, so the indexer skips the rest and the cost follows the real lengths;
    ``torch.topk`` reads whole rows, so the indexer fills positions past each length with
    negative infinity.

    Given ``filter_blocks`` (``DynamicInferenceContext.dsa_filter_blocks``), contexts bounded
    by at least ``FUSED_MIN_CONTEXT`` keys take the fused selection of
    ``dsa_gqa_fused_select`` where the device supports it, which never materializes the
    scores; ``decode_only`` says every row is its request's only query token this step.

    Returns:
        Attention output ``[rows,Hq,Dv]``.
    """
    from megatron.core.transformer.experimental_attention_variant import (
        dsa_gqa_fused_select,
        dsa_gqa_inference_kernels,
    )

    row_requests, row_lengths, block_starts, block_rows, block_requests, _ = row_metadata
    num_rows, num_query_heads, _ = query.shape
    block_table = block_table.to(torch.int32).contiguous()
    index_query = index_query.contiguous()
    use_fused = (
        filter_blocks is not None
        and max_context_length >= dsa_gqa_fused_select.FUSED_MIN_CONTEXT
        and dsa_gqa_fused_select.fused_select_supported(
            query.device, index_topk, index_query.size(-1), index_key_cache.size(1)
        )
    )
    if use_fused:
        _log_fused_topk()
        topk_indices = dsa_gqa_fused_select.fused_select_topk(
            index_query,
            index_key_cache,
            block_table,
            row_requests,
            row_lengths,
            (block_starts, block_rows, block_requests),
            filter_blocks,
            width=max_context_length,
            index_topk=index_topk,
            decode_only=decode_only,
        )
    else:
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
    use_deep_select = _deep_select_decode_supported(index_query.device, index_topk)
    _log_decode_topk_backend(use_deep_select)
    width = max_context_length
    if use_deep_select:
        # DeepSelect needs row strides aligned to its requirement and at least top-k
        # columns; the indexer never writes past a row's length.
        alignment = deep_select.get_stride_requirement()[0] // torch.float32.itemsize
        width = -(-max(width, index_topk) // alignment) * alignment
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
            fill_invalid=not use_deep_select,
            row_offset=start,
        )
        topk_indices[start:stop] = _select_topk(
            window_scores, row_lengths[start:stop], index_topk, use_deep_select
        )
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
    backend: str = "triton",
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
        backend: ``triton`` (Megatron's kernels) or ``flashinfer`` (``dsa_gqa_flashinfer``).

    Returns:
        Attention output ``[T,1,Hq*Dv]``; rows for padding tokens are zero.
    """
    if inference_context.num_speculative_tokens:
        raise NotImplementedError(
            "DSA inference decodes one token per request; MTP speculative decoding is not "
            "supported."
        )
    if backend not in ("triton", "triton_gluon", "flashinfer"):
        raise ValueError(f"Unsupported dynamic DSA-GQA backend: {backend}")
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
    if backend == "triton_gluon":
        return _native_sparse_attention(
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
        ).view(num_tokens, 1, num_query_heads * value_cache.size(-1))
    if backend == "flashinfer":
        from megatron.core.transformer.experimental_attention_variant import dsa_gqa_flashinfer

        if not dsa_gqa_flashinfer.flashinfer_dsa_supported(
            query.device, index_topk, index_query.size(-1), key_cache.size(1), num_query_heads
        ):
            raise RuntimeError(
                "dsa_inference_backend='flashinfer' needs FlashInfer with DSA kernels, CuTe DSL "
                "4.7+, FA4, DeepSelect and Triton on SM100/SM103, top-k 512/1024/2048, indexer "
                "dim 128, and a KV block size that is a multiple of 64."
            )
        output = dsa_gqa_flashinfer.flashinfer_sparse_attention(
            query=query[:, 0],
            key_cache=key_cache,
            value_cache=value_cache,
            index_query=index_query.reshape(num_tokens, -1),
            index_key_cache=index_key_cache,
            block_table=block_table.to(torch.int32),
            # The configured bound, eager or graphed: FlashInfer compiles per launch shape.
            plan=inference_context.dsa_flashinfer_plan(inference_context.max_sequence_length),
            index_topk=index_topk,
            softmax_scale=softmax_scale,
        )
        return output.view(num_tokens, 1, num_query_heads * value_cache.size(-1))
    output = _sparse_attention_rows(
        query=query[:, 0],
        key_cache=key_cache,
        value_cache=value_cache,
        index_query=index_query.reshape(num_tokens, -1),
        index_key_cache=index_key_cache,
        block_table=block_table,
        row_metadata=row_metadata,
        max_context_length=max_context_length,
        index_topk=index_topk,
        softmax_scale=softmax_scale,
        filter_blocks=inference_context.dsa_filter_blocks(),
        decode_only=inference_context.is_decode_only(),
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
        filter_blocks=inference_context.dsa_filter_blocks(),
        decode_only=False,
    )
    output[:rows] = torch.where((lengths > 0)[:, None, None], native, output[:rows])
    return output
