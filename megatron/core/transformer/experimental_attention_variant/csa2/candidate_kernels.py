# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused candidate-set indexer scoring for DeepSeek-V4.1 reindex layers.

A reindex layer scores only the candidate blocks published by the candidate-source layer
(``csa2_candidate_topk_blocks`` blocks of ``csa2_candidate_block_size`` compressed entries per
query). The reference path scores every compressed entry densely (``[rows, n_comp]`` fp32),
masks the non-candidates and the causally unreachable entries with ``-inf`` and runs a top-k
over the full width; at 128K that is a 2 GB score matrix per 4096-row chunk and a top-k over
131072 columns for a set of at most 16384 candidates.

The Triton kernel here computes ``sum_h relu(q_h . k) * w_h`` directly for the candidate
entries of each row, with the invalid-block, out-of-range and causal masks applied inside the
kernel, and writes a ``[rows, topk_blocks * block_size]`` fp32 candidate score matrix (8x
smaller at the released configuration). The top-k then runs over the candidate width and the
winning entries are mapped back to segment-local compressed ids. Numerics: bf16 queries,
keys and head weights, fp32 accumulation, the ``(d_i * h_i) ** -0.5`` scale applied in fp32
before the ReLU (equivalent for a positive scale), the same contract as the merged cuDNN
dense indexer kernel used for the candidate-source layer. The eager reference remains the
semantic oracle (``tests/unit_tests/models/deepseek_v41/test_dsv41_candidate_kernel.py``).
"""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _candidate_scores_kernel(
        q_ptr,  # [R, H, D] bf16
        k_ptr,  # [N, D] bf16
        w_ptr,  # [R, H] bf16 (raw head weights)
        blocks_ptr,  # [R, B] int32 candidate block ids (-1 invalid)
        visible_ptr,  # [R] int32 causal limit (entries < visible are reachable)
        out_ptr,  # [R, B * BS] fp32
        n_keys,
        scale,
        stride_qr,
        stride_qh,
        stride_kn,
        stride_wr,
        stride_br,
        stride_or,
        H: tl.constexpr,
        D: tl.constexpr,
        BS: tl.constexpr,  # block size (compressed entries per candidate block)
        NB: tl.constexpr,  # candidate blocks per program
    ):
        row = tl.program_id(0)
        blk0 = tl.program_id(1) * NB
        KEYS: tl.constexpr = NB * BS

        # candidate block ids for this tile -> key ids (the host guarantees B % NB == 0)
        b_off = tl.arange(0, NB)
        block_ids = tl.load(blocks_ptr + row * stride_br + blk0 + b_off)  # [NB] int32
        j = tl.arange(0, BS)
        blocks_2d = block_ids[:, None] + tl.zeros((NB, BS), dtype=tl.int32)  # [NB, BS]
        key_ids = tl.reshape(blocks_2d * BS + j[None, :], (KEYS,))
        block_valid = tl.reshape(blocks_2d, (KEYS,)) >= 0
        visible = tl.load(visible_ptr + row)
        valid = block_valid & (key_ids < visible) & (key_ids < n_keys)
        safe_ids = tl.where(valid, key_ids, 0)

        d = tl.arange(0, D)
        # keys tile [KEYS, D] (bf16), transposed for the dot
        k_tile = tl.load(
            k_ptr + safe_ids[:, None] * stride_kn + d[None, :], mask=valid[:, None], other=0.0
        )
        h = tl.arange(0, H)
        q_tile = tl.load(q_ptr + row * stride_qr + h[:, None] * stride_qh + d[None, :])  # [H, D]
        logits = tl.dot(q_tile, tl.trans(k_tile))  # [H, KEYS] fp32
        logits = tl.maximum(logits * scale, 0.0)
        w = tl.load(w_ptr + row * stride_wr + h).to(tl.float32)  # [H]
        scores = tl.sum(logits * w[:, None], axis=0)  # [KEYS]
        scores = tl.where(valid, scores, float("-inf"))
        out_off = blk0 * BS + tl.arange(0, KEYS)
        tl.store(out_ptr + row * stride_or + out_off, scores)


def candidate_scores_available() -> bool:
    """Triton present and a CUDA device available."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def candidate_kernel_supported(
    heads: int, dim: int, block_size: int, n_blocks: int, blocks_per_program: int = 16
) -> bool:
    """Geometry the Triton kernel can compile and launch: ``tl.arange`` extents (heads, dim,
    block size, blocks per program) must be powers of two, ``tl.dot`` needs at least 16 heads
    and 16 dims, and the candidate blocks must tile evenly. Callers use this to choose the dense
    path for unsupported geometry instead of the eager candidate reference, whose gathered keys
    (``[rows, candidates, dim]``) would exceed the dense score budget several times over."""
    return (
        _pow2(heads)
        and heads >= 16
        and _pow2(dim)
        and dim >= 16
        and _pow2(block_size)
        and _pow2(blocks_per_program)
        and n_blocks % blocks_per_program == 0
    )


def _candidate_scores_reference(
    q_rows: torch.Tensor,
    keys: torch.Tensor,
    raw_head_weights: torch.Tensor,
    scale: float,
    candidate_blocks: torch.Tensor,
    block_size: int,
    visible: torch.Tensor,
) -> torch.Tensor:
    """Eager equivalent of the kernel: ``[rows, topk_blocks * block_size]`` fp32 with ``-inf``
    for invalid blocks, out-of-range and unreachable entries."""
    rows, n_blocks = candidate_blocks.shape
    n_keys = keys.size(0)
    key_ids = candidate_blocks.long().unsqueeze(-1) * block_size + torch.arange(
        block_size, device=keys.device
    ).view(1, 1, -1)
    key_ids = key_ids.view(rows, -1)
    valid = (
        (candidate_blocks.unsqueeze(-1) >= 0).expand(rows, n_blocks, block_size).reshape(rows, -1)
    )
    valid = valid & (key_ids < visible.view(-1, 1).long()) & (key_ids < n_keys)
    gathered = keys.float()[key_ids.clamp(0, max(n_keys - 1, 0))]  # [rows, K, D]
    logits = torch.einsum("rhd,rkd->rhk", q_rows.float(), gathered) * scale
    scores = (torch.relu(logits) * raw_head_weights.float().unsqueeze(-1)).sum(dim=1)
    return scores.masked_fill(~valid, float("-inf"))


def candidate_scores(
    q_rows: torch.Tensor,
    keys: torch.Tensor,
    raw_head_weights: torch.Tensor,
    scale: float,
    candidate_blocks: torch.Tensor,
    block_size: int,
    visible: torch.Tensor,
    blocks_per_program: int = 16,
) -> torch.Tensor:
    """Candidate-set indexer scores ``[rows, topk_blocks * block_size]`` (fp32, ``-inf`` where the
    entry is not a valid, reachable candidate).

    Args:
        q_rows: ``[rows, h_i, d_i]`` rotated indexer queries.
        keys: ``[n_keys, d_i]`` rotated index keys of the segment.
        raw_head_weights: ``[rows, h_i]`` unscaled head weights.
        scale: ``(d_i * h_i) ** -0.5``.
        candidate_blocks: ``[rows, topk_blocks]`` int32 block ids, ``-1`` invalid.
        block_size: compressed entries per block.
        visible: ``[rows]`` causal limits (entries ``< visible`` are reachable).
    """
    rows, n_blocks = candidate_blocks.shape
    heads, dim = q_rows.shape[1], q_rows.shape[2]
    if (
        not candidate_scores_available()
        or not q_rows.is_cuda
        or not candidate_kernel_supported(heads, dim, block_size, n_blocks, blocks_per_program)
    ):
        # Eager oracle (tests, CPU). Training callers check candidate_kernel_supported() first
        # and take the dense path instead: this reference gathers [rows, candidates, dim] keys.
        return _candidate_scores_reference(
            q_rows, keys, raw_head_weights, scale, candidate_blocks, block_size, visible
        )
    q = q_rows.to(torch.bfloat16).contiguous()
    k = keys.to(torch.bfloat16).contiguous()
    w = raw_head_weights.to(torch.bfloat16).contiguous()
    blocks = candidate_blocks.to(torch.int32).contiguous()
    vis = visible.to(torch.int32).contiguous()
    out = torch.empty((rows, n_blocks * block_size), dtype=torch.float32, device=q.device)
    grid = (rows, n_blocks // blocks_per_program)
    _candidate_scores_kernel[grid](
        q,
        k,
        w,
        blocks,
        vis,
        out,
        keys.size(0),
        float(scale),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        w.stride(0),
        blocks.stride(0),
        out.stride(0),
        H=heads,
        D=dim,
        BS=block_size,
        NB=blocks_per_program,
        num_warps=4,
    )
    return out


def candidate_topk_ids(
    cand_scores: torch.Tensor, candidate_blocks: torch.Tensor, block_size: int, topk: int
) -> torch.Tensor:
    """Top-``k`` over the candidate width mapped back to segment-local compressed ids: ``[rows, k]``
    int32, sorted ascending, ``-1`` where fewer than ``k`` valid candidates exist."""
    rows, width = cand_scores.shape
    k = min(topk, width)
    if k == 0:
        return cand_scores.new_empty((rows, 0), dtype=torch.int32)
    top_scores, top_idx = cand_scores.topk(k, dim=-1, sorted=False)
    block = candidate_blocks.long().gather(1, top_idx // block_size)
    ids = block * block_size + (top_idx % block_size)
    ids = torch.where(torch.isfinite(top_scores), ids, torch.full_like(ids, -1))
    # sort ascending with -1 (invalid) last: sort by (id if valid else large)
    order_key = torch.where(ids >= 0, ids, torch.full_like(ids, 2**31 - 1))
    order = order_key.argsort(dim=-1)
    return ids.gather(1, order).to(torch.int32)
