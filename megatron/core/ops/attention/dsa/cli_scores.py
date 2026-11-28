# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Recompute raw DSA indexer logits over the selected slots for CLI."""

import torch
import triton
import triton.language as tl


@triton.jit
def _selected_indexer_logits(
    Q,
    K,
    W,
    TopK,
    Length,
    Logits,
    QUERY_ROWS: tl.constexpr,
    KEY_ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    TOPK: tl.constexpr,
    SLOT_TILES: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row = (pid // SLOT_TILES).to(tl.int64)
    batch = row // QUERY_ROWS
    slot = (pid % SLOT_TILES) * BLOCK_K + tl.arange(0, BLOCK_K)
    head = tl.arange(0, BLOCK_H)
    feature = tl.arange(0, BLOCK_D)

    length = tl.load(Length + row)
    index = tl.load(TopK + row * TOPK + slot, slot < TOPK, other=-1)
    valid = (slot < length) & (index >= 0) & (index < KEY_ROWS)
    q = tl.load(
        Q + (row * HEADS + head[:, None]) * DIM + feature[None, :],
        (head[:, None] < HEADS) & (feature[None, :] < DIM),
        other=0,
    )
    k = tl.load(
        K + (batch * KEY_ROWS + index[None, :]) * DIM + feature[:, None],
        valid[None, :] & (feature[:, None] < DIM),
        other=0,
    )
    w = tl.load(W + row * HEADS + head, head < HEADS, other=0).to(tl.float32)
    dots = tl.dot(q, k, out_dtype=tl.float32)
    score = tl.sum(tl.maximum(dots, 0.0) * w[:, None], axis=0)
    tl.store(Logits + row * TOPK + slot, tl.where(valid, score, 0.0), slot < TOPK)


def cli_selected_indexer_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Return FP32 weighted ReLU scores in the original Top-K order.

    Inputs use contiguous BSHD queries, BSD keys, BSH weights, and BSK
    indices. Invalid slots are zero and are masked by the loss.
    """
    batch, query_rows, heads, dim = q.shape
    topk = indices.size(-1)
    logits = torch.empty(indices.shape, device=q.device, dtype=torch.float32)
    if logits.numel() == 0:
        return logits
    block_k = 64
    tiles = triton.cdiv(topk, block_k)
    _selected_indexer_logits[(batch * query_rows * tiles,)](
        q,
        k,
        weights,
        indices,
        lengths,
        logits,
        query_rows,
        k.size(1),
        heads,
        dim,
        topk,
        tiles,
        max(16, triton.next_power_of_2(heads)),
        max(16, triton.next_power_of_2(dim)),
        block_k,
    )
    return logits
