# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Experimental bounded-memory document-local QSA selected-ID router.

The packed pool has one row per complete block, with an upper-bound allocation
of ``tokens // ratio`` rows. A program for each query reads only its own
document's visible complete blocks. This backend is deliberately opt-in.
"""

import math
from typing import Tuple

import torch
import triton
import triton.language as tl


def pool_complete_blocks(
    raw_keys: torch.Tensor,
    doc_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_docs: int,
    ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool complete blocks into a flat allocation bounded by ``T // ratio``.

    Returns pooled keys, document block prefix/count, each pool row's document
    and relative block ID, and a validity mask for unused upper-bound rows.
    """
    tokens, dim = raw_keys.shape
    doc_long = doc_ids.long()
    doc_lengths = torch.zeros(num_docs, dtype=torch.int32, device=raw_keys.device)
    doc_lengths.scatter_add_(0, doc_long, torch.ones_like(doc_ids, dtype=torch.int32))
    blocks_per_doc = doc_lengths // ratio
    block_prefix = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=raw_keys.device),
            blocks_per_doc.cumsum(0, dtype=torch.int32),
        )
    )
    complete = positions < blocks_per_doc[doc_long] * ratio
    destination = block_prefix[doc_long] + positions // ratio
    upper_bound = tokens // ratio
    pooled_fp32 = torch.zeros((upper_bound, dim), dtype=torch.float32, device=raw_keys.device)
    pooled_fp32.index_add_(0, destination[complete].long(), raw_keys[complete].float())
    pooled = (pooled_fp32 / ratio).to(raw_keys.dtype)
    global_block = torch.arange(upper_bound, dtype=torch.int32, device=raw_keys.device)
    block_doc = torch.searchsorted(block_prefix[1:].contiguous(), global_block, right=True)
    block_doc = block_doc.clamp(max=num_docs - 1)
    block_valid = global_block < block_prefix[-1]
    block_relative = torch.where(
        block_valid, global_block - block_prefix[block_doc], torch.zeros_like(global_block)
    )
    return pooled, block_prefix, blocks_per_doc, block_doc, block_relative, block_valid


@triton.jit
def _route_ids(
    Q,
    Pooled,
    Doc,
    Pos,
    Prefix,
    Count,
    Out,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_PAD: tl.constexpr,
    D_PAD: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    doc = tl.load(Doc + row)
    position = tl.load(Pos + row)
    prefix = tl.load(Prefix + doc)
    available = tl.load(Count + doc)
    visible = tl.minimum((position + 1) // R, available)
    output_slots = tl.arange(0, K_PAD)

    if visible <= K:
        direct = tl.where(output_slots < visible, output_slots, -1)
        tl.store(Out + row * K + output_slots, direct, output_slots < K)
    else:
        heads = tl.arange(0, HEAD_PAD)
        dim = tl.arange(0, D_PAD)
        q = tl.load(
            Q + row * H * D + heads[:, None] * D + dim[None, :],
            (heads[:, None] < H) & (dim[None, :] < D),
            other=0,
        ).to(tl.float32)
        best = tl.full((K_PAD,), 0, tl.uint64)
        candidate_slots = tl.arange(0, K_PAD)
        for chunk in range(tl.cdiv(visible, BLOCK_N)):
            block = chunk * BLOCK_N + tl.arange(0, BLOCK_N)
            key = tl.load(
                Pooled + (prefix + block[None, :]) * D + dim[:, None],
                (block[None, :] < visible) & (dim[:, None] < D),
                other=0,
            ).to(tl.float32)
            dots = tl.dot(q, key, input_precision="ieee")
            scores = tl.sum(tl.maximum(dots, 0), 0) * SCALE
            score_bits = scores.to(tl.uint32, bitcast=True).to(tl.uint64)
            rank = (score_bits << 32) | (0xFFFFFFFF - block.to(tl.uint64))
            rank = tl.where(block < visible, rank, 0)
            rank_padded = tl.where(
                candidate_slots < BLOCK_N, tl.gather(rank, candidate_slots % BLOCK_N, 0), 0
            )
            best = tl.topk(tl.cat(best, rank_padded, can_reorder=True), K_PAD)
        chosen = (0xFFFFFFFF - (best & 0xFFFFFFFF)).to(tl.int32)
        tl.store(Out + row * K + output_slots, chosen, output_slots < K)


@torch.no_grad()
def select_document_local_ids(
    queries: torch.Tensor,
    pooled: torch.Tensor,
    doc_ids: torch.Tensor,
    positions: torch.Tensor,
    block_prefix: torch.Tensor,
    blocks_per_doc: torch.Tensor,
    *,
    ratio: int,
    topk: int,
) -> torch.Tensor:
    """Return document-relative IDs ``[T,topk]`` without cross-document scores."""
    tokens, heads, dim = queries.shape
    if not queries.is_cuda or not pooled.is_cuda:
        raise ValueError("document-local QSA routing requires CUDA")
    if queries.dtype != torch.bfloat16 or pooled.dtype != torch.bfloat16:
        raise NotImplementedError("document-local QSA routing currently requires BF16")
    if heads > 8 or dim > 128 or topk > 512 or ratio <= 0:
        raise NotImplementedError("unsupported QSA document-local router geometry")
    if block_prefix.shape != (blocks_per_doc.numel() + 1,):
        raise ValueError("block_prefix must have one more entry than blocks_per_doc")
    k_pad = max(16, triton.next_power_of_2(topk))
    out = torch.empty((tokens, topk), dtype=torch.int32, device=queries.device)
    _route_ids[(tokens,)](
        queries.contiguous(),
        pooled.contiguous(),
        doc_ids.contiguous(),
        positions.contiguous(),
        block_prefix.contiguous(),
        blocks_per_doc.contiguous(),
        out,
        heads,
        dim,
        ratio,
        topk,
        k_pad,
        min(k_pad, 128),
        16,
        max(32, triton.next_power_of_2(dim)),
        1.0 / math.sqrt(dim),
        num_warps=8,
    )
    return out
