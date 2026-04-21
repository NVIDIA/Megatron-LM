# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Referenced from https://github.com/tile-ai/tilelang/blob/main/examples

import torch

from .sparse_mla_fwd import _ref_sparse_mla_fwd
from .utils import prepare_token_indices

try:
    import tilelang
    import tilelang.language as T

    HAS_TILELANG = True
except ImportError:
    HAS_TILELANG = False

if HAS_TILELANG:

    @tilelang.jit(out_idx=[-1])
    def _tl_preprocess(H, D, block_ND=32, num_stages=5, dtype="bfloat16", accum_dtype="float"):
        assert dtype == "bfloat16" and accum_dtype == "float"
        S = T.symbolic("S")
        shape = [S, H, D]

        @T.prim_func
        def preprocess_kernel(
            O: T.Tensor(shape, dtype),
            dO: T.Tensor(shape, dtype),
            Delta: T.Tensor([S, H], accum_dtype),
        ):
            with T.Kernel(H, T.ceildiv(S, block_ND)) as (bx, by):
                o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
                do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
                delta = T.alloc_fragment([block_ND], accum_dtype)
                acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
                T.clear(acc)
                for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
                    T.copy(
                        O[
                            by * block_ND : (by + 1) * block_ND,
                            bx,
                            k * block_ND : (k + 1) * block_ND,
                        ],
                        o,
                    )
                    T.copy(
                        dO[
                            by * block_ND : (by + 1) * block_ND,
                            bx,
                            k * block_ND : (k + 1) * block_ND,
                        ],
                        do,
                    )
                    for i, j in T.Parallel(block_ND, block_ND):
                        acc[i, j] += o[i, j] * do[i, j]
                T.reduce_sum(acc, delta, 1)
                T.copy(delta, Delta[by * block_ND : (by + 1) * block_ND, bx])

        return preprocess_kernel

    @tilelang.jit(out_idx=[-1])
    def _tl_postprocess(
        D, D_tail, kv_group=1, block_N=64, threads=128, dtype="bfloat16", accum_dtype="float"
    ):
        assert dtype == "bfloat16" and accum_dtype == "float"
        S_kv = T.symbolic("S_kv")
        dkv_shape = [S_kv, kv_group, D + D_tail]

        @T.prim_func
        def postprocess_kernel(
            dKV: T.Tensor(dkv_shape, accum_dtype), dKV_out: T.Tensor(dkv_shape, dtype)
        ):
            with T.Kernel(T.ceildiv(S_kv, block_N), kv_group, threads=threads) as (bx, by):
                T.copy(
                    dKV[bx * block_N : (bx + 1) * block_N, by, :],
                    dKV_out[bx * block_N : (bx + 1) * block_N, by, :],
                )

        return postprocess_kernel

    @tilelang.jit(
        out_idx=[-2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        },
    )
    def _tl_bwd(
        H,
        D,
        D_tail,
        topk,
        kv_group=1,
        sm_scale=None,
        is_causal=True,
        block_size=32,
        num_stages=0,
        threads=256,
        indices_dtype="int32",
        dtype="bfloat16",
        accum_dtype="float",
    ):
        assert kv_group == 1
        assert topk % block_size == 0
        assert is_causal
        assert dtype == "bfloat16"
        assert accum_dtype == "float"
        assert indices_dtype == "int32"

        if sm_scale is None:
            sm_scale = (D + D_tail) ** (-0.5)

        B_plus_one = T.symbolic("B_plus_one")
        S = T.symbolic("S")

        if H > 64:
            assert H % 64 == 0
            kv_group_view = H // 64
        else:
            kv_group_view = 1
        H_kv = H // kv_group_view

        q_shape = [S, H, D + D_tail]
        k_shape = [S, kv_group, D + D_tail]
        o_shape = [S, H, D]
        indices_shape = [S, kv_group, topk]
        delta_shape = [S, H]
        lse_shape = [S, H]
        offsets_shape = [B_plus_one]
        token_indices_shape = [S, 2]

        padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
        BS = block_size
        NS = tilelang.cdiv(topk, block_size)
        split_store = 2

        @T.prim_func
        def sparse_mla_bwd_kernel(
            Q: T.Tensor(q_shape, dtype),
            KV: T.Tensor(k_shape, dtype),
            dO: T.Tensor(o_shape, dtype),
            Indices: T.Tensor(indices_shape, indices_dtype),
            Lse: T.Tensor(lse_shape, accum_dtype),
            Delta: T.Tensor(delta_shape, accum_dtype),
            Offsets: T.Tensor(offsets_shape, indices_dtype),
            TokenIndices: T.Tensor(token_indices_shape, indices_dtype),
            dQ: T.Tensor(q_shape, dtype),
            dKV: T.Tensor(k_shape, accum_dtype),
        ):
            with T.Kernel(S, kv_group_view, threads=threads) as (b_s_i, bz):
                Q_shared = T.alloc_shared([padded_H, D], dtype)
                Q_tail_shared = T.alloc_shared([padded_H, D_tail], dtype)
                KV_shared = T.alloc_shared([BS, D], dtype)
                KV_tail_shared = T.alloc_shared([BS, D_tail], dtype)
                dO_shared = T.alloc_shared([padded_H, D], dtype)
                mask = T.alloc_fragment([BS], "bool")

                P_shared_cast = T.alloc_shared([padded_H, BS], dtype)
                dP_shared_cast = T.alloc_shared([padded_H, BS], dtype)
                dQ_shared = T.alloc_shared([padded_H, D], dtype)
                dQ_tail_shared = T.alloc_shared([padded_H, D_tail], dtype)

                acc_p = T.alloc_fragment([padded_H, BS], accum_dtype)
                acc_dp = T.alloc_fragment([padded_H, BS], accum_dtype)
                acc_dq = T.alloc_fragment([padded_H, D], accum_dtype)
                acc_dq_tail = T.alloc_fragment([padded_H, D_tail], accum_dtype)
                acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
                acc_dkv_tail = T.alloc_fragment([BS, D_tail], accum_dtype)
                acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)
                acc_dkv_tail_shared = T.alloc_shared([BS // split_store, D_tail], accum_dtype)

                b_i, s_i = TokenIndices[b_s_i, 0], TokenIndices[b_s_i, 1]
                bos = Offsets[b_i]
                max_kv_i = s_i

                T.copy(Q[bos + s_i, bz * padded_H : (bz + 1) * padded_H, :D], Q_shared)
                T.copy(Q[bos + s_i, bz * padded_H : (bz + 1) * padded_H, D:], Q_tail_shared)
                T.copy(dO[bos + s_i, bz * padded_H : (bz + 1) * padded_H, :D], dO_shared)

                T.clear(acc_dq)
                T.clear(acc_dq_tail)

                T.annotate_layout(
                    {
                        dQ_shared: tilelang.layout.make_swizzled_layout(dQ_shared),
                        dQ_tail_shared: tilelang.layout.make_swizzled_layout(dQ_tail_shared),
                    }
                )

                for i_i in T.Pipelined(NS, num_stages=num_stages):
                    for bi_i in T.Parallel(BS):
                        idx = Indices[bos + s_i, 0, i_i * BS + bi_i]
                        mask[bi_i] = (idx <= max_kv_i) & (idx != -1)

                    for h_i, bi_i in T.Parallel(padded_H, BS):
                        acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                    for bi_i, d_i in T.Parallel(BS, D):
                        KV_shared[bi_i, d_i] = KV[
                            bos + Indices[bos + s_i, 0, i_i * BS + bi_i], 0, d_i
                        ]
                    T.gemm(
                        Q_shared,
                        KV_shared,
                        acc_p,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    for bi_i, d_i in T.Parallel(BS, D_tail):
                        KV_tail_shared[bi_i, d_i] = KV[
                            bos + Indices[bos + s_i, 0, i_i * BS + bi_i], 0, D + d_i
                        ]
                    T.gemm(
                        Q_tail_shared,
                        KV_tail_shared,
                        acc_p,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    for h_i, bi_i in T.Parallel(padded_H, BS):
                        acc_p[h_i, bi_i] = T.exp(
                            acc_p[h_i, bi_i] * sm_scale - Lse[bos + s_i, bz * padded_H + h_i]
                        )

                    T.copy(acc_p, P_shared_cast)
                    T.gemm(
                        dO_shared,
                        KV_shared,
                        acc_dp,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )

                    for h_i, bi_i in T.Parallel(padded_H, BS):
                        acc_dp[h_i, bi_i] = (
                            acc_p[h_i, bi_i]
                            * (acc_dp[h_i, bi_i] - Delta[bos + s_i, bz * padded_H + h_i])
                            * sm_scale
                        )

                    T.copy(acc_dp, dP_shared_cast)
                    T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(
                        dP_shared_cast, KV_tail_shared, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol
                    )

                    T.gemm(
                        dP_shared_cast,
                        Q_shared,
                        acc_dkv,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )
                    T.gemm(
                        P_shared_cast,
                        dO_shared,
                        acc_dkv,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    T.clear(acc_dkv_tail)
                    T.gemm(
                        dP_shared_cast,
                        Q_tail_shared,
                        acc_dkv_tail,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    for s in range(split_store):
                        for bi_i, d_i in T.Parallel(BS, D):
                            if bi_i < BS // split_store:
                                acc_dkv_shared[bi_i, d_i] = acc_dkv[
                                    bi_i + s * (BS // split_store), d_i
                                ]
                        for bi_i, d_i in T.Parallel(BS, D_tail):
                            if bi_i < BS // split_store:
                                acc_dkv_tail_shared[bi_i, d_i] = acc_dkv_tail[
                                    bi_i + s * (BS // split_store), d_i
                                ]
                        for bi_i, d_i in T.Parallel(BS // split_store, D):
                            T.atomic_add(
                                dKV[
                                    bos
                                    + Indices[
                                        bos + s_i, 0, i_i * BS + bi_i + s * (BS // split_store)
                                    ],
                                    0,
                                    d_i,
                                ],
                                acc_dkv_shared[bi_i, d_i],
                            )
                        for bi_i, d_i in T.Parallel(BS // split_store, D_tail):
                            T.atomic_add(
                                dKV[
                                    bos
                                    + Indices[
                                        bos + s_i, 0, i_i * BS + bi_i + s * (BS // split_store)
                                    ],
                                    0,
                                    D + d_i,
                                ],
                                acc_dkv_tail_shared[bi_i, d_i],
                            )

                T.copy(acc_dq, dQ_shared)
                T.copy(acc_dq_tail, dQ_tail_shared)
                T.copy(dQ_shared, dQ[bos + s_i, bz * padded_H : (bz + 1) * padded_H, :D])
                T.copy(dQ_tail_shared, dQ[bos + s_i, bz * padded_H : (bz + 1) * padded_H, D:])

        return sparse_mla_bwd_kernel


def _ref_sparse_mla_bwd(q, kv, do, indices, offsets, sm_scale, d_v):
    """Unfused sparse MLA backward (THD format). Recomputes forward via autograd."""
    with torch.enable_grad():
        q = q.detach().clone().requires_grad_(True)
        kv = kv.detach().clone().requires_grad_(True)
        o_ref = _ref_sparse_mla_fwd(q, kv, indices, offsets, sm_scale, d_v)[0]
        o_ref.backward(do)
    return q.grad, kv.grad


def sparse_mla_bwd_interface(q, kv, o, do, indices, lse, offsets, sm_scale, d_v, use_unfused=False):
    """Sparse MLA backward. THD format. (o and lse are only used by the fused path.)"""
    if use_unfused or not HAS_TILELANG:
        return _ref_sparse_mla_bwd(q, kv, do, indices, offsets, sm_scale, d_v)

    S, H, dim_plus_tail_dim = q.shape
    D = d_v
    D_tail = dim_plus_tail_dim - D
    _, kv_group, _ = kv.shape
    topk = indices.shape[-1]

    token_indices = prepare_token_indices(offsets)
    preprocess_kernel = _tl_preprocess(H, D)
    threads = 256 if H > 16 else 128
    bwd_kernel = _tl_bwd(H, D, D_tail, topk, kv_group, sm_scale, threads=threads)
    postprocess_kernel = _tl_postprocess(D, D_tail, kv_group)

    delta = preprocess_kernel(o, do)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    dq = bwd_kernel(q, kv, do, indices, lse, delta, offsets, token_indices, dkv)
    dkv = postprocess_kernel(dkv)
    return dq, dkv
