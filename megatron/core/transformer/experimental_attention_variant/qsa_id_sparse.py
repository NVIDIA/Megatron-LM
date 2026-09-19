# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Experimental selected-ID QSA GQA kernel; opt-in QSA backend in an isolated prototype.

``block_ids[b, q, :]`` contains distinct, document-relative complete block IDs,
with ``-1`` for invalid slots. ``positions[b, q]`` is the query's document-relative
token position. The last incomplete block is included directly by the kernel.

This reference-oriented Triton implementation prioritizes exact routing and
bounded O(S*K) work. It is not tuned for production or 256K training.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_indices(
    Ids,
    Pos,
    row,
    seq_len: tl.constexpr,
    topk: tl.constexpr,
    ratio: tl.constexpr,
    block_n: tl.constexpr,
    chunk,
):
    """Selected block slots and the incomplete tail, with masked global KV positions."""
    b = row // seq_len
    q = row % seq_len
    q_pos = tl.load(Pos + row)
    doc_start = q - q_pos
    slots = chunk * block_n + tl.arange(0, block_n)
    block_slot = slots // ratio
    offset = slots % ratio
    selected = block_slot < topk
    block_id = tl.load(Ids + row * topk + block_slot, selected, other=-1)
    tail_pos = ((q_pos + 1) // ratio) * ratio + offset
    rel_pos = tl.where(selected, block_id * ratio + offset, tail_pos)
    kv_idx = doc_start + rel_pos
    valid = tl.where(
        selected,
        (block_id >= 0) & (block_id * ratio + ratio - 1 <= q_pos),
        (block_slot == topk) & (tail_pos <= q_pos),
    )
    valid = valid & (kv_idx >= 0) & (kv_idx < seq_len)
    return kv_idx, valid


@triton.jit
def _qsa_id_fwd(
    Q,
    K,
    V,
    Ids,
    Pos,
    O,
    Lse,
    seq_len: tl.constexpr,
    hq: tl.constexpr,
    hk: tl.constexpr,
    dim: tl.constexpr,
    topk: tl.constexpr,
    ratio: tl.constexpr,
    scale: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1)
    b = row // seq_len
    q = row % seq_len
    kh = h // (hq // hk)
    dims = tl.arange(0, dim)
    q_vec = tl.load(Q + ((b * hq + h) * seq_len + q) * dim + dims).to(tl.float32)
    m = tl.full((), -float("inf"), tl.float32)
    denom = tl.full((), 0.0, tl.float32)
    acc = tl.full((dim,), 0.0, tl.float32)
    for chunk in range(tl.cdiv((topk + 1) * ratio, block_n)):
        kv_idx, valid = _qsa_indices(Ids, Pos, row, seq_len, topk, ratio, block_n, chunk)
        keys = tl.load(
            K + ((b * hk + kh) * seq_len + kv_idx[:, None]) * dim + dims[None, :],
            valid[:, None],
            other=0,
        ).to(tl.float32)
        values = tl.load(
            V + ((b * hk + kh) * seq_len + kv_idx[:, None]) * dim + dims[None, :],
            valid[:, None],
            other=0,
        ).to(tl.float32)
        scores = tl.sum(keys * q_vec[None, :], 1) * scale
        scores = tl.where(valid, scores, -float("inf"))
        new_m = tl.maximum(m, tl.max(scores, 0))
        # Every query has its own token in the incomplete tail. Earlier chunks may
        # contain no valid IDs, so avoid -inf - -inf before the first valid key.
        alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - new_m))
        probs = tl.exp(scores - new_m)
        probs = tl.where(valid, probs, 0.0)
        acc = acc * alpha + tl.sum(probs[:, None] * values, 0)
        denom = denom * alpha + tl.sum(probs, 0)
        m = new_m
    tl.store(O + ((b * hq + h) * seq_len + q) * dim + dims, acc / denom)
    tl.store(Lse + (b * hq + h) * seq_len + q, m + tl.log(denom))


@triton.jit
def _qsa_id_bwd(
    Q,
    K,
    V,
    Ids,
    Pos,
    O,
    Lse,
    DO,
    DQ,
    DK,
    DV,
    seq_len: tl.constexpr,
    hq: tl.constexpr,
    hk: tl.constexpr,
    dim: tl.constexpr,
    topk: tl.constexpr,
    ratio: tl.constexpr,
    scale: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1)
    b = row // seq_len
    q = row % seq_len
    kh = h // (hq // hk)
    dims = tl.arange(0, dim)
    q_ptr = ((b * hq + h) * seq_len + q) * dim + dims
    q_vec = tl.load(Q + q_ptr).to(tl.float32)
    out = tl.load(O + q_ptr).to(tl.float32)
    do = tl.load(DO + q_ptr).to(tl.float32)
    delta = tl.sum(out * do, 0)
    lse = tl.load(Lse + (b * hq + h) * seq_len + q)
    dq = tl.full((dim,), 0.0, tl.float32)
    for chunk in range(tl.cdiv((topk + 1) * ratio, block_n)):
        kv_idx, valid = _qsa_indices(Ids, Pos, row, seq_len, topk, ratio, block_n, chunk)
        kv_ptr = ((b * hk + kh) * seq_len + kv_idx[:, None]) * dim + dims[None, :]
        keys = tl.load(K + kv_ptr, valid[:, None], other=0).to(tl.float32)
        values = tl.load(V + kv_ptr, valid[:, None], other=0).to(tl.float32)
        scores = tl.sum(keys * q_vec[None, :], 1) * scale
        p = tl.exp(scores - lse)
        p = tl.where(valid, p, 0.0)
        dp = tl.sum(values * do[None, :], 1)
        ds = p * (dp - delta) * scale
        dq += tl.sum(ds[:, None] * keys, 0)
        tl.atomic_add(DK + kv_ptr, ds[:, None] * q_vec[None, :], valid[:, None], sem="relaxed")
        tl.atomic_add(DV + kv_ptr, p[:, None] * do[None, :], valid[:, None], sem="relaxed")
    tl.store(DQ + q_ptr, dq)


class _QSASparseIdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, block_ids, positions, ratio, scale):
        """Run sparse forward and retain tensors needed to recompute gradients."""
        b, hq, seq_len, dim = query.shape
        hk = key.shape[1]
        topk = block_ids.shape[-1]
        output = torch.empty_like(query)
        lse = torch.empty((b, hq, seq_len), device=query.device, dtype=torch.float32)
        _qsa_id_fwd[(b * seq_len, hq)](
            query,
            key,
            value,
            block_ids,
            positions,
            output,
            lse,
            seq_len,
            hq,
            hk,
            dim,
            topk,
            ratio,
            scale,
            32,
            num_warps=4,
        )
        ctx.save_for_backward(query, key, value, block_ids, positions, output, lse)
        ctx.ratio = ratio
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Accumulate Q/K/V gradients for the fixed selected-ID routing."""
        query, key, value, block_ids, positions, output, lse = ctx.saved_tensors
        b, hq, seq_len, dim = query.shape
        hk, topk = key.shape[1], block_ids.shape[-1]
        dq = torch.empty_like(query)
        dk = torch.zeros_like(key, dtype=torch.float32)
        dv = torch.zeros_like(value, dtype=torch.float32)
        _qsa_id_bwd[(b * seq_len, hq)](
            query,
            key,
            value,
            block_ids,
            positions,
            output,
            lse,
            grad_output.contiguous(),
            dq,
            dk,
            dv,
            seq_len,
            hq,
            hk,
            dim,
            topk,
            ctx.ratio,
            ctx.scale,
            32,
            num_warps=4,
        )
        return dq, dk.to(key.dtype), dv.to(value.dtype), None, None, None, None


def validate_qsa_block_ids(block_ids, positions, ratio):
    """Debug-only validation of the QSA producer contract; synchronizes CUDA."""
    topk = block_ids.shape[-1]
    visible = (positions + 1) // ratio
    valid = block_ids >= 0
    expected_count = visible.clamp_max(topk)
    bad_count = valid.sum(dim=-1) != expected_count
    bad_range = ((block_ids < -1) | (block_ids >= visible.unsqueeze(-1))).any(dim=-1)
    sentinel = torch.iinfo(block_ids.dtype).max
    sorted_ids = torch.where(valid, block_ids, sentinel).sort(dim=-1).values
    duplicate = (
        (sorted_ids[..., 1:] == sorted_ids[..., :-1]) & (sorted_ids[..., :-1] != sentinel)
    ).any(dim=-1)
    if bool((bad_count | bad_range | duplicate).any().item()):
        raise ValueError(
            "Selected QSA IDs must be distinct, visible, and fill every available top-k slot"
        )


def qsa_sparse_attention_id(
    query, key, value, block_ids, positions, *, ratio=4, scale=None, validate=False
):
    """Sparse attention from selected complete block IDs and the incomplete tail.

    Inputs are contiguous ``Q [B,Hq,S,D]``, ``K/V [B,Hkv,S,D]``,
    ``block_ids [B,S,K]`` int32, and ``positions [B,S]`` int32. Each query's
    block IDs must be unique and document-relative; invalid slots contain -1.
    Every row selects exactly ``min(visible_complete_blocks, K)`` IDs.
    ``positions`` must count from zero within each packed document.
    ``validate=True`` checks this producer contract but synchronizes CUDA, so
    it is for tests/debugging only.
    """
    if not all(t.is_cuda and t.is_contiguous() for t in (query, key, value, block_ids, positions)):
        raise ValueError("QSA selected-ID prototype requires contiguous CUDA tensors")
    if query.ndim != 4 or key.shape != value.shape or block_ids.ndim != 3 or positions.ndim != 2:
        raise ValueError("Invalid selected-ID QSA tensor shapes")
    b, hq, s, d = query.shape
    if (
        key.shape[0] != b
        or key.shape[2:] != (s, d)
        or block_ids.shape[:2] != (b, s)
        or positions.shape != (b, s)
    ):
        raise ValueError("Inconsistent selected-ID QSA dimensions")
    if hq % key.shape[1] or not triton.next_power_of_2(d) == d or d > 256:
        raise ValueError("GQA head grouping or head dimension is unsupported")
    if block_ids.dtype != torch.int32 or positions.dtype != torch.int32 or ratio < 1:
        raise ValueError("Expected int32 IDs/positions and positive compression ratio")
    if block_ids.shape[-1] < 1 or ratio > 32 or 32 % ratio:
        raise ValueError("Prototype requires positive top-k and a ratio dividing 32")
    if scale is None:
        scale = d**-0.5
    if validate:
        validate_qsa_block_ids(block_ids, positions, ratio)
    return _QSASparseIdFunction.apply(query, key, value, block_ids, positions, ratio, scale)
