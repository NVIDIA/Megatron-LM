# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Forked from `fla/ops/gated_delta_product/chunk_deltaproduct_h.py` in
# flash-linear-attention v0.5.1 (https://github.com/fla-org/flash-linear-attention).
#
# Licensed under the MIT license; see the LICENSE file in the repository root.

"""Inter-chunk state recurrence for the Gated Delta Product.

Carries the matrix-valued state forward across chunks, emitting the state at
each chunk boundary (for the intra-chunk output kernel) and the corrected
values `v_new`. The Householder expansion shows up here as the stride between
state stores: the state is only checkpointed once every `num_householder`
expanded chunks, because that is one chunk of the original token stream.

The key dimension is handled in fixed 64-wide registers, which is why `K` is
capped at 256.
"""

import torch

from .common import HAVE_TRITON, exp2, prepare_chunk_indices, prepare_chunk_offsets, tl, triton


@triton.heuristics(
    {
        'USE_G': lambda args: args['g'] is not None,
        'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
        'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
        'HAS_STATE_INDICES': lambda args: args['state_indices'] is not None,
        'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_G'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_product_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    state_indices,
    ht_slot_stride,
    ht_head_stride,
    cu_seqlens,
    chunk_offsets,
    T,
    num_householder: tl.constexpr,  # number of delta products
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    HAS_STATE_INDICES: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Sweep one sequence's chunks, carrying the `[K, V]` state in registers."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * tl.cdiv(T // num_householder, BT)

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h) * K * V
    v += (bos * H + i_h) * V
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h) * V
    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if HAS_STATE_INDICES:
        # Dynamic batching: `ht` is the per-request cache, addressed by slot.
        # A padding request carries -1 and must leave the cache untouched.
        i_s = tl.load(state_indices + i_n).to(tl.int64)
    else:
        i_s = i_n
    if STORE_FINAL_STATE:
        if HAS_STATE_INDICES:
            ht = ht + i_s * ht_slot_stride + i_h * ht_head_stride
        else:
            ht = ht + i_nh * K * V

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        if i_t % num_householder == 0:
            i_t_true = i_t // num_householder
            p_h1 = tl.make_block_ptr(
                h + i_t_true * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h2 = tl.make_block_ptr(
                    h + i_t_true * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_h3 = tl.make_block_ptr(
                    h + i_t_true * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_h4 = tl.make_block_ptr(
                    h + i_t_true * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v_new = tl.zeros([BT, BV], dtype=tl.float32)
        p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v_new += tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h4.to(b_w.dtype))
        b_v_new = -b_v_new + tl.load(p_v, boundary_check=(0, 1))

        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v_new, b_v_new.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v_new = b_v_new * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = exp2(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last
        b_v_new = b_v_new.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v_new)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v_new)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v_new)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v_new)
    # epilogue
    if STORE_FINAL_STATE and i_s >= 0:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_product_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    num_householder: int = 1,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the inter-chunk state recurrence.

    Args:
        k: Keys `[B, T, H, K]` on the Householder-expanded token stream.
        w: WY key factor, shaped like `k`.
        u: WY value factor `[B, T, H, V]`.
        g: Within-chunk cumulative log2 decays on the expanded stream, or `None`.
        initial_state: Starting state `[N, H, K, V]`, or `None` for zeros.
        output_final_state: Whether to return the final state.
        chunk_size: Chunk length.
        save_new_value: Whether to emit the corrected values `v_new`.
        cu_seqlens: Sequence boundaries `[N+1]` on the expanded stream.
        num_householder: Number of Householder copies `M`.
        chunk_indices: Chunk descriptors for the *unexpanded* stream. Derived
            from `cu_seqlens // num_householder` when omitted, which
            synchronizes on the device.
        chunk_offsets: Per-sequence prefix sum of unexpanded chunk counts.
            Derived from `cu_seqlens` when omitted.
        state: `[S, H, K, V]` per-request state cache for dynamic batching,
            written in place at `state_indices` instead of into a dense
            `final_state`. `-1` slots are skipped.
        state_indices: `[N]` cache slot per sequence, or `None` for a dense
            `[N, H, K, V]` final state.

    Returns `(h, v_new, final_state)`. `h` holds the state at each unexpanded
    chunk boundary.
    """
    assert HAVE_TRITON, "chunk_gated_delta_product_fwd_h requires Triton"
    B, T, H, K, V = *k.shape, u.shape[-1]
    assert T % num_householder == 0, "T must be divisible by num_householder"
    T_true = T // num_householder
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens // num_householder, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T_true, BT), None
    else:
        N = len(cu_seqlens) - 1
        NT = len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens // num_householder, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    h = k.new_empty(B, NT, H, K, V)

    # Slot indices without a cache to index would leave the slot/head strides at
    # zero below, aliasing every sequence onto slot 0 while the padding mask still
    # runs -- wrong results behind well-formed output.
    assert (
        state_indices is None or state is not None
    ), "state_indices requires the state cache it indexes into"

    if state is not None:
        assert state.shape[1:] == (
            H,
            K,
            V,
        ), f"state is expected to have shape [num_slots, {H}, {K}, {V}], got {tuple(state.shape)}"
        assert (
            state.stride(3) == 1 and state.stride(2) == V
        ), "the last two dimensions of the state cache must be contiguous"
        final_state = state
    else:
        final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), N * H)

    chunk_gated_delta_product_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state,
        state_indices=state_indices,
        ht_slot_stride=state.stride(0) if state is not None else 0,
        ht_head_stride=state.stride(1) if state is not None else 0,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        num_householder=num_householder,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return h, v_new, final_state
