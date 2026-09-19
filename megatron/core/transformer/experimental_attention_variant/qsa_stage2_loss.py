# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bounded Stage-2 QSA KL prototype for compact, preselected block IDs.

The caller owns hard TopK and passes only its selected *complete* blocks. Block
IDs index a flat table, so packed documents need no rectangular
``[number_of_documents, maximum_document_blocks]`` allocation. Query rows may
be a CP-local subset, while main-attention keys must be in global token order.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from megatron.core.transformer.experimental_attention_variant import dsa_indexer_loss, dsa_masking
from megatron.core.utils import get_pg_size


class _SelectedBlockScore(torch.autograd.Function):
    """Recompute ReLU scores in backward instead of retaining ``[Q, H, K]``."""

    @staticmethod
    def forward(ctx, index_query, compressed_key, block_ids, valid):
        gathered = compressed_key[block_ids.clamp_min(0)]
        compute_dtype = torch.float64 if index_query.dtype == torch.float64 else torch.float32
        scores = torch.einsum(
            "qhd,qkd->qhk", index_query.to(compute_dtype), gathered.to(compute_dtype)
        )
        logits = (scores.relu().sum(dim=1) / math.sqrt(index_query.size(-1))).masked_fill(
            ~valid, 0.0
        )
        ctx.save_for_backward(index_query, compressed_key, block_ids, valid)
        return logits

    @staticmethod
    def backward(ctx, grad_logits):
        index_query, compressed_key, block_ids, valid = ctx.saved_tensors
        compute_dtype = torch.float64 if index_query.dtype == torch.float64 else torch.float32
        safe_ids = block_ids.clamp_min(0)
        query = index_query.to(compute_dtype)
        gathered = compressed_key[safe_ids].to(compute_dtype)
        active = (torch.einsum("qhd,qkd->qhk", query, gathered) > 0) & valid.unsqueeze(1)
        grad_scores = (
            grad_logits.to(compute_dtype).unsqueeze(1) * active / math.sqrt(query.size(-1))
        )
        grad_query = torch.einsum("qhk,qkd->qhd", grad_scores, gathered)
        grad_gathered = torch.einsum("qhk,qhd->qkd", grad_scores, query)
        grad_key = torch.zeros_like(compressed_key, dtype=compute_dtype)
        grad_key.index_add_(
            0, safe_ids.reshape(-1), grad_gathered.reshape(-1, grad_gathered.size(-1))
        )
        return grad_query.to(index_query.dtype), grad_key.to(compressed_key.dtype), None, None


@torch.no_grad()
def _teacher_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    block_ids: torch.Tensor,
    block_starts: torch.Tensor,
    document_starts: torch.Tensor,
    query_positions: torch.Tensor,
    valid: torch.Tensor,
    *,
    compress_ratio: int,
    softmax_scale: Optional[float],
    tp_group: Optional[torch.distributed.ProcessGroup],
) -> torch.Tensor:
    """Selected-token softmax, all-head sum, token L1, block MaxPool, block L1."""
    rows, heads, dim = query.shape
    kv_heads = key.size(1)
    budget = block_ids.size(1)
    offsets = torch.arange(compress_ratio, device=query.device)
    selected_ids = block_starts[block_ids.clamp_min(0)].unsqueeze(-1) + offsets
    selected_valid = valid.unsqueeze(-1).expand(-1, -1, compress_ratio)
    selected_ids = selected_ids.reshape(rows, budget * compress_ratio)
    selected_valid = selected_valid.reshape(rows, budget * compress_ratio)

    # A causal query also sees its own incomplete block. It has at most R-1
    # tokens and is deliberately excluded from the KL's block MaxPool.
    tail_start = document_starts + ((query_positions + 1) // compress_ratio) * compress_ratio
    tail_offsets = torch.arange(compress_ratio - 1, device=query.device)
    tail_ids = tail_start.unsqueeze(-1) + tail_offsets
    tail_valid = tail_ids <= document_starts.unsqueeze(-1) + query_positions.unsqueeze(-1)
    token_ids = torch.cat((selected_ids, tail_ids), dim=-1)
    route_valid = torch.cat((selected_valid, tail_valid), dim=-1)
    gathered = key.detach()[token_ids.clamp(0, key.size(0) - 1)]
    grouped_query = query.detach().unflatten(1, (kv_heads, heads // kv_heads))
    grouped_key = gathered.permute(0, 2, 1, 3)
    scale = dim**-0.5 if softmax_scale is None else softmax_scale
    compute_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    scores = (
        torch.einsum(
            "rhgd,rhkd->rhgk", grouped_query.to(compute_dtype), grouped_key.to(compute_dtype)
        )
        * scale
    )
    scores = scores.masked_fill(~route_valid[:, None, None, :], -torch.inf)
    has_routes = route_valid.any(dim=-1)
    scores = torch.where(has_routes[:, None, None, None], scores, torch.zeros_like(scores))
    probs = torch.softmax(scores, dim=-1).masked_fill(~route_valid[:, None, None, :], 0.0)
    token_mass = probs.sum(dim=(1, 2)).contiguous()
    if get_pg_size(tp_group) > 1:
        # Every TP rank must execute the same number of chunks in the same order.
        torch.distributed.all_reduce(token_mass, group=tp_group)
    token_mass = dsa_indexer_loss.normalize_indexer_target(token_mass)
    block_mass = token_mass[:, : budget * compress_ratio].unflatten(-1, (budget, compress_ratio))
    target = block_mass.amax(dim=-1).masked_fill(~valid, 0.0)
    return dsa_indexer_loss.normalize_indexer_target(target)


def qsa_stage2_sparse_kl(
    index_query: torch.Tensor,
    compressed_key: torch.Tensor,
    block_ids: torch.Tensor,
    block_starts: torch.Tensor,
    teacher_query: torch.Tensor,
    teacher_key: torch.Tensor,
    document_starts: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    compress_ratio: int,
    loss_coeff: float,
    softmax_scale: Optional[float] = None,
    calculate_per_token_loss: bool = False,
    query_valid_rows: Optional[torch.Tensor] = None,
    query_chunk_size: int = 128,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Compute local-query Stage-2 KL without dense ``[Q, all_blocks]`` logits.

    ``block_ids`` are detached absolute IDs in ``compressed_key``; ``-1`` pads
    absent slots. ``block_starts`` maps each complete block to its first global
    token. ``document_starts`` and ``query_positions`` have one entry per local
    query row, which permits packed documents and CP-local query ownership.
    Callers must detach the backbone hidden input before indexer projection.
    """
    if (
        isinstance(loss_coeff, bool)
        or not isinstance(loss_coeff, (int, float))
        or loss_coeff < 0
        or not math.isfinite(loss_coeff)
    ):
        raise ValueError("loss_coeff must be finite and non-negative")
    if softmax_scale is not None and (
        isinstance(softmax_scale, bool)
        or not isinstance(softmax_scale, (int, float))
        or softmax_scale <= 0
        or not math.isfinite(softmax_scale)
    ):
        raise ValueError("softmax_scale must be finite and positive")
    if loss_coeff == 0:
        return index_query.new_zeros(())
    if compress_ratio <= 1 or query_chunk_size <= 0:
        raise ValueError("compress_ratio and query_chunk_size must exceed one and zero")
    if index_query.ndim != 3 or compressed_key.ndim != 2:
        raise ValueError("index_query must be [Q,H,D] and compressed_key [P,D]")
    rows = index_query.size(0)
    if (
        block_ids.ndim != 2
        or block_ids.size(0) != rows
        or block_starts.shape != (compressed_key.size(0),)
        or document_starts.shape != (rows,)
        or query_positions.shape != (rows,)
    ):
        raise ValueError("compact block and query-position shapes do not match")
    if teacher_query.ndim != 3 or teacher_key.ndim != 3 or teacher_query.size(0) != rows:
        raise ValueError("teacher query/key must be [Q,H,D] and [T,Hkv,D]")
    if teacher_query.size(-1) != teacher_key.size(-1) or teacher_query.size(1) % teacher_key.size(
        1
    ):
        raise ValueError("teacher GQA head dimensions do not match")
    if compressed_key.size(0) == 0 or teacher_key.size(0) == 0:
        raise ValueError("KL requires non-empty compressed and attention keys")
    if (
        bool((block_ids < -1).any())
        or bool((block_starts < 0).any())
        or bool((block_starts + compress_ratio > teacher_key.size(0)).any())
    ):
        raise ValueError("block IDs or complete block token spans are invalid")
    if bool(
        (
            (query_positions < 0)
            | (document_starts < 0)
            | (document_starts + query_positions >= teacher_key.size(0))
        ).any()
    ):
        raise ValueError("local query position is outside global attention keys")
    valid = block_ids >= 0
    if bool((block_ids >= compressed_key.size(0)).any()):
        raise ValueError("selected block ID is outside compressed_key")
    selected_starts = block_starts[block_ids.clamp_min(0)]
    first = document_starts.unsqueeze(-1)
    complete_end = first + ((query_positions.unsqueeze(-1) + 1) // compress_ratio) * compress_ratio
    if bool(
        (
            valid
            & (
                (selected_starts < first)
                | (selected_starts + compress_ratio > complete_end)
                | ((selected_starts - first) % compress_ratio != 0)
            )
        ).any()
    ):
        raise ValueError("selected block crosses a document or causal boundary")
    if query_valid_rows is not None and query_valid_rows.shape != (rows,):
        raise ValueError("query_valid_rows must have one entry per local query")
    loss_dtype = torch.float64 if index_query.dtype == torch.float64 else torch.float32
    kl_sum = index_query.new_zeros((), dtype=loss_dtype)
    for start in range(0, rows, query_chunk_size):
        end = min(start + query_chunk_size, rows)
        chunk_valid = valid[start:end]
        if query_valid_rows is not None:
            chunk_valid = chunk_valid & query_valid_rows[start:end, None]
        logits = _SelectedBlockScore.apply(
            index_query[start:end], compressed_key, block_ids[start:end], chunk_valid
        )
        target = _teacher_chunk(
            teacher_query[start:end],
            teacher_key,
            block_ids[start:end],
            block_starts,
            document_starts[start:end],
            query_positions[start:end],
            chunk_valid,
            compress_ratio=compress_ratio,
            softmax_scale=softmax_scale,
            tp_group=tp_group,
        )
        log_probs = dsa_masking.masked_log_softmax(logits, chunk_valid)
        kl_sum = kl_sum + dsa_indexer_loss.indexer_kl_sum(target, log_probs, chunk_valid)
    valid_count = query_valid_rows.sum() if query_valid_rows is not None else None
    reduced = dsa_indexer_loss.reduce_indexer_kl_sum(
        kl_sum,
        num_rows=rows,
        calculate_per_token_loss=calculate_per_token_loss,
        valid_row_count=valid_count,
    )
    return reduced * loss_coeff
