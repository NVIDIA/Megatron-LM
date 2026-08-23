# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Forked from `fla/ops/gated_delta_product/chunk_deltaproduct_o.py` in
# flash-linear-attention v0.5.1 (https://github.com/fla-org/flash-linear-attention).
#
# Licensed under the MIT license; see the LICENSE file in the repository root.

"""Intra-chunk output for the Gated Delta Product.

Combines the inter-chunk state from `chunk_h` with the causal within-chunk
attention. The queries live on the unexpanded token stream while the keys and
values live on the Householder-expanded one, so the within-chunk term is
accumulated over the `num_householder` copies, each strided `H*K` (or `H*V`)
apart.
"""

import torch

from .common import (
    HAVE_TRITON,
    IS_NVIDIA_HOPPER,
    check_shared_mem,
    exp2,
    prepare_chunk_indices,
    tl,
    triton,
)

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


@triton.heuristics(
    {
        'USE_G': lambda args: args['g'] is not None,
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    num_householder: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Emit one chunk of output from the chunk-boundary state plus local attention."""
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * num_householder * H + i_h) * K
    v += (bos * num_householder * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_m = tl.where(m_A, exp2(b_g[:, None] - b_g[None, :]), 0)
        b_o = b_o * exp2(b_g)[:, None]
    else:
        b_m = ((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)).to(tl.float32)

    for i_dp in range(num_householder):
        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(
                k + i_dp * H * K,
                (K, T),
                (1, num_householder * H * K),
                (i_k * BK, i_t * BT),
                (BK, BT),
                (0, 1),
            )
            # [BT, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            # [BK, BT]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BT, BK] @ [BK, BT] -> [BT, BT]
            b_A += tl.dot(b_q, b_k)
        b_A = b_A * b_m
        p_v = tl.make_block_ptr(
            v + i_dp * H * V,
            (T, V),
            (H * V * num_householder, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o += tl.dot(b_A.to(b_v.dtype), b_v)
    b_o = b_o * scale
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_product_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    num_householder: int = 1,
    chunk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the Gated Delta Product outputs.

    Args:
        q: Queries `[B, T, H, K]` on the unexpanded token stream.
        k: Keys `[B, T*M, H, K]` on the Householder-expanded stream.
        v: Corrected values `[B, T*M, H, V]` from `chunk_h`.
        h: State at each chunk boundary, from `chunk_h`.
        g: Within-chunk cumulative log2 decays `[B, T, H]`, or `None`.
        scale: Score scale.
        cu_seqlens: Sequence boundaries `[N+1]` on the unexpanded stream.
        chunk_size: Chunk length.
        num_householder: Number of Householder copies `M`.
        chunk_indices: Precomputed chunk descriptors for the unexpanded stream.
            Derived from `cu_seqlens` when omitted, which synchronizes on the
            device.

    Returns the outputs `[B, T, H, V]`.
    """
    assert HAVE_TRITON, "chunk_gated_delta_product_fwd_o requires Triton"
    assert (
        q.shape[1] * num_householder == k.shape[1]
    ), "q.shape[1] * num_householder must be equal to k.shape[1]"
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    # Zeros, not a poison value: with a padded token dimension the tail belongs
    # to no chunk program, and the padding contract requires it to read back zero.
    o = v.new_zeros(B, T, H, V)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        num_householder=num_householder,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return o
