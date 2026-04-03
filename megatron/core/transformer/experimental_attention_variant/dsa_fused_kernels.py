# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
This module provides 5 interface functions for DSA operations, each with built-in
fused (TileLang) and unfused (PyTorch) paths controlled by a `use_unfused` flag:

  1. sparse_mla_fwd_interface            - Sparse MLA forward pass
  2. sparse_mla_bwd_interface            - Sparse MLA backward pass
  3. indexer_topk_reducesum_interface    - Indexer top-k selection + index score
  4. indexer_bwd_interface               - Indexer backward pass
  5. sparse_mla_topk_reducesum_interface - Attention score computation for indexer loss

Also provides an autograd wrapper for the DSA function.
"""

import functools
import math
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from einops import einsum as einops_einsum

try:
    import tilelang
    import tilelang.language as T

    HAS_TILELANG = True
except ImportError:
    HAS_TILELANG = False


# =====================================================================================
# Section A: Utilities
# =====================================================================================


def _tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Single-entry cache for functions with tensor inputs."""
    last_args: tuple | None = None
    last_kwargs: dict | None = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result
        if (
            (last_args is not None and last_kwargs is not None)
            and (len(args) == len(last_args) and len(kwargs) == len(last_kwargs))
            and all(a is b for a, b in zip(args, last_args, strict=False))
            and all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items())
        ):
            return last_result
        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


@_tensor_cache
def _prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@_tensor_cache
def _prepare_position_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.cat(
        [
            torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
            for n in _prepare_lens(cu_seqlens).unbind()
        ]
    )


@_tensor_cache
def _prepare_sequence_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return _prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1


@_tensor_cache
def prepare_token_indices(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    """Convert cumulative sequence lengths to per-token (batch_id, position) pairs."""
    position_ids = _prepare_position_ids(cu_seqlens)
    return torch.stack([_prepare_sequence_ids(cu_seqlens), position_ids], 1).to(cu_seqlens)


# =====================================================================================
# Section B: TileLang kernel definitions
# =====================================================================================

if HAS_TILELANG:

    # --- Sparse MLA Forward ---
    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _tl_sparse_mla_fwd(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group=1,
        sm_scale=None,
        is_causal=True,
        block_I=32,
        num_stages=2,
        threads=128,
    ):
        assert dim == tilelang.math.next_power_of_2(dim)
        assert tail_dim == tilelang.math.next_power_of_2(tail_dim)
        assert is_causal
        assert topk % block_I == 0
        if sm_scale is None:
            sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        batch_plus_one = T.symbolic("batch_plus_one")
        seq_len = T.symbolic("seq_len")

        head_kv = heads // kv_group
        q_shape = [seq_len, heads, dim + tail_dim]
        kv_shape = [seq_len, kv_group, dim + tail_dim]
        o_shape = [seq_len, heads, dim]
        indices_shape = [seq_len, kv_group, topk]
        lse_shape = [seq_len, heads]
        offsets_shape = [batch_plus_one]
        token_indices_shape = [seq_len, 2]
        indices_dtype = "int32"
        dtype = "bfloat16"
        accum_dtype = "float"

        padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
        if padded_H != head_kv:
            assert kv_group == 1
        BI = block_I
        NI = tilelang.cdiv(topk, block_I)
        D = dim
        D_tail = tail_dim

        if head_kv > 64:
            assert head_kv % 64 == 0
            REPLICATE_H = head_kv // 64
        else:
            REPLICATE_H = 1
        H_per_block = padded_H if REPLICATE_H == 1 else 64

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),
            KV: T.Tensor(kv_shape, dtype),
            Indices: T.Tensor(indices_shape, indices_dtype),
            Offsets: T.Tensor(offsets_shape, indices_dtype),
            TokenIndices: T.Tensor(token_indices_shape, indices_dtype),
            Output: T.Tensor(o_shape, dtype),
            Lse: T.Tensor(lse_shape, accum_dtype),
        ):
            with T.Kernel(seq_len * REPLICATE_H, kv_group, threads=threads) as (bx, by):
                Q_shared = T.alloc_shared([H_per_block, D], dtype)
                Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
                KV_shared = T.alloc_shared([BI, D], dtype)
                K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
                mask = T.alloc_fragment([BI], "bool")

                acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                S_shared = T.alloc_shared([H_per_block, BI], dtype)
                sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                alpha = T.alloc_fragment([H_per_block], accum_dtype)
                m_i = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

                T.fill(acc_o, 0)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))

                b_s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
                b_i, s_i = TokenIndices[b_s_i, 0], TokenIndices[b_s_i, 1]
                bos, eos = Offsets[b_i], Offsets[b_i + 1]
                g_i = by
                max_kv_i = s_i

                H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
                H1 = H0 + H_per_block

                T.copy(Q[bos + s_i, H0:H1, :D], Q_shared)
                T.copy(Q[bos + s_i, H0:H1, D:], Q_tail_shared)

                for i_i in T.Pipelined(NI, num_stages=num_stages):
                    for bi_i in T.Parallel(BI):
                        mask[bi_i] = (Indices[bos + s_i, g_i, i_i * BI + bi_i] <= max_kv_i) & (
                            Indices[bos + s_i, g_i, i_i * BI + bi_i] != -1
                        )

                    for bi_i, d_i in T.Parallel(BI, D):
                        KV_shared[bi_i, d_i] = KV[
                            bos + Indices[bos + s_i, g_i, i_i * BI + bi_i], g_i, d_i
                        ]
                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        K_tail_shared[bi_i, d_i] = KV[
                            bos + Indices[bos + s_i, g_i, i_i * BI + bi_i], g_i, D + d_i
                        ]

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                    T.gemm(
                        Q_shared,
                        KV_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.gemm(
                        Q_tail_shared,
                        K_tail_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        alpha[h_i] = T.exp((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D):
                        acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] /= sumexp[h_i]
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = T.log(sumexp[h_i]) + m_i[h_i] * sm_scale

                T.copy(acc_o, Output[bos + s_i, H0:H1, :])
                T.copy(sumexp, Lse[bos + s_i, H0:H1])

        return main

    # --- Sparse MLA Backward: preprocess, postprocess, bwd ---
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

    # --- Indexer Top-K + ReduceSum ---
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tl_indexer_topk_reducesum(
        heads: int,
        dim: int,
        topk: int,
        sm_scale: Optional[float] = None,
        block_K: int = 32,
        dtype: str = "bfloat16",
        num_stages: int = 0,
        num_threads: int = 128,
    ):
        assert topk == tilelang.math.next_power_of_2(topk)
        assert topk % block_K == 0
        assert heads <= 64 and heads % 8 == 0
        assert num_stages == 0

        batch_plus_one = T.symbolic("batch_plus_one")
        seq_len = T.symbolic("seq_len")
        INT32 = "int32"
        FP32 = "float"

        index_q_shape = [seq_len, heads, dim]
        weights_shape = [seq_len, heads]
        index_k_shape = [seq_len, dim]
        topk_indices_shape = [seq_len, topk]
        offsets_shape = [batch_plus_one]
        token_indices_shape = [seq_len, 2]

        N = 2 * topk
        num_iters = int(round(math.log2(N)))
        if sm_scale is None:
            sm_scale = dim**-0.5

        @T.macro
        def bitonic_sort(
            topk_index_shared: T.SharedBuffer([N], dtype=INT32),
            topk_value_shared: T.SharedBuffer([N], dtype=FP32),
        ):
            T.sync_threads()
            for i1 in T.serial(num_iters):
                for i2 in T.serial(i1 + 1):
                    for i in T.Parallel(N):
                        ascending = (i & (1 << (i1 + 1))) != 0
                        j = i ^ (1 << (i1 - i2))
                        if i < j and (
                            (ascending and topk_value_shared[i] > topk_value_shared[j])
                            or (not ascending and topk_value_shared[i] < topk_value_shared[j])
                        ):
                            val = topk_value_shared[i]
                            topk_value_shared[i] = topk_value_shared[j]
                            topk_value_shared[j] = val
                            idx = topk_index_shared[i]
                            topk_index_shared[i] = topk_index_shared[j]
                            topk_index_shared[j] = idx
                    T.sync_threads()

        @T.prim_func
        def tl_indexer_topk_reducesum_kernel(
            IndexQ: T.Tensor(index_q_shape, dtype),
            Weights: T.Tensor(weights_shape, dtype),
            IndexK: T.Tensor(index_k_shape, dtype),
            TopkIndices: T.Tensor(topk_indices_shape, INT32),
            ReduceSum: T.Tensor(topk_indices_shape, FP32),
            Offsets: T.Tensor(offsets_shape, INT32),
            TokenIndices: T.Tensor(token_indices_shape, INT32),
        ):
            with T.Kernel(seq_len, threads=num_threads) as (bx):
                i_b, i_t = TokenIndices[bx, 0], TokenIndices[bx, 1]
                bos, eos = Offsets[i_b], Offsets[i_b + 1]
                num_blocks = T.ceildiv(i_t + 1, block_K)

                topk_index_shared = T.alloc_shared([N], dtype=INT32)
                topk_value_shared = T.alloc_shared([N], dtype=FP32)
                T.fill(topk_index_shared, -1)
                T.fill(topk_value_shared, float("-inf"))
                T.sync_threads()

                index_q_shared = T.alloc_shared([heads, dim], dtype=dtype)
                T.copy(IndexQ[bos + i_t, :, :], index_q_shared)
                T.sync_threads()

                weights_frag = T.alloc_shared([heads], dtype=dtype)
                T.copy(Weights[bos + i_t, :], weights_frag)
                T.sync_threads()

                for i, j in T.Parallel(heads, dim):
                    index_q_shared[i, j] = index_q_shared[i, j] * sm_scale
                T.sync_threads()

                for bk_i in T.Pipelined(num_blocks, num_stages=num_stages):
                    k_st = bk_i * block_K
                    k_ed = T.min((bk_i + 1) * block_K, eos - bos)

                    index_k_shared = T.alloc_shared([block_K, dim], dtype=dtype)
                    for i, j in T.Parallel(block_K, dim):
                        index_k_shared[i, j] = T.if_then_else(
                            k_st + i < k_ed, IndexK[bos + k_st + i, j], 0
                        )
                    T.sync_threads()

                    logits = T.alloc_fragment((block_K, heads), FP32)
                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        logits,
                        transpose_A=False,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.sync_threads()

                    for i, j in T.Parallel(block_K, heads):
                        logits[i, j] = T.max(logits[i, j], 0) * weights_frag[j]
                    T.sync_threads()

                    logits_sum = T.alloc_fragment(block_K, FP32)
                    T.reduce_sum(logits, logits_sum, dim=1)
                    T.sync_threads()

                    offset = T.alloc_var(INT32)
                    if k_st >= topk:
                        offset = topk + (k_st % topk)
                    else:
                        offset = k_st
                    T.sync_threads()
                    for i in T.Parallel(block_K):
                        if k_st + i > i_t:
                            logits_sum[i] = float("-inf")
                        j = offset + i
                        topk_index_shared[j] = k_st + i
                        topk_value_shared[j] = logits_sum[i]
                    T.sync_threads()

                    if k_ed > topk and k_ed % topk == 0:
                        bitonic_sort(topk_index_shared, topk_value_shared)

                bitonic_sort(topk_index_shared, topk_value_shared)

                logits_max_frag = T.alloc_fragment([1], dtype=FP32)
                logits_frag = T.alloc_fragment([topk], dtype=FP32)
                reducesum_shared = T.alloc_shared([topk], dtype=FP32)

                T.copy(topk_value_shared[:topk], logits_frag)
                T.sync_threads()
                T.reduce_max(logits_frag, logits_max_frag, dim=-1)
                T.sync_threads()
                for i in T.Parallel(topk):
                    logits_frag[i] = T.exp(logits_frag[i] - logits_max_frag[0])
                T.sync_threads()
                lse_frag = T.alloc_fragment([1], dtype=FP32)
                T.reduce_sum(logits_frag, lse_frag)
                T.sync_threads()
                for i in T.Parallel(topk):
                    reducesum_shared[i] = logits_frag[i] / lse_frag[0]
                T.sync_threads()

                for i in T.Parallel(topk):
                    if topk_index_shared[i] > i_t:
                        topk_index_shared[i] = -1
                T.sync_threads()

                T.copy(topk_index_shared[:topk], TopkIndices[bos + i_t, :])
                T.copy(reducesum_shared[:topk], ReduceSum[bos + i_t, :])

        return tl_indexer_topk_reducesum_kernel

    # --- Indexer Backward ---
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_WGMMA: True,
        }
    )
    def _tl_indexer_bwd(
        heads: int,
        dim: int,
        topk: int,
        sm_scale: Optional[float] = None,
        block_I: int = 32,
        num_stages: int = 0,
        num_threads: int = 128,
    ):
        assert num_stages == 0
        assert topk == tilelang.math.next_power_of_2(topk)
        assert topk % block_I == 0
        assert heads <= 64 and heads % 8 == 0

        batch_plus_one = T.symbolic("batch_plus_one")
        seq_len = T.symbolic("seq_len")
        dtype = "bfloat16"
        accum_dtype = "float"
        integer_dtype = "int32"

        index_q_shape = [seq_len, heads, dim]
        weights_shape = [seq_len, heads]
        index_k_shape = [seq_len, dim]
        shape_p = [seq_len, topk]
        topk_indices_shape = [seq_len, topk]
        offsets_shape = [batch_plus_one]
        token_indices_shape = [seq_len, 2]
        if sm_scale is None:
            sm_scale = dim**-0.5

        @T.prim_func
        def tl_indexer_bwd_kernel(
            IndexQ: T.Tensor(index_q_shape, dtype),
            Weights: T.Tensor(weights_shape, dtype),
            IndexK: T.Tensor(index_k_shape, dtype),
            dIndexQ: T.Tensor(index_q_shape, dtype),
            dWeights: T.Tensor(weights_shape, dtype),
            dIndexK: T.Tensor(index_k_shape, accum_dtype),
            AttnScore: T.Tensor(shape_p, accum_dtype),
            IndexScore: T.Tensor(shape_p, accum_dtype),
            TopkIndices: T.Tensor(topk_indices_shape, integer_dtype),
            Offsets: T.Tensor(offsets_shape, integer_dtype),
            TokenIndices: T.Tensor(token_indices_shape, integer_dtype),
        ):
            with T.Kernel(seq_len, threads=num_threads) as (bx):
                i_b, i_t = TokenIndices[bx, 0], TokenIndices[bx, 1]
                bos = Offsets[i_b]
                num_blocks = T.ceildiv(topk, block_I)

                index_q_shared = T.alloc_shared([heads, dim], dtype=dtype)
                weights_shared = T.alloc_shared([heads], dtype=dtype)
                d_index_q_frag = T.alloc_fragment([heads, dim], dtype=accum_dtype)
                d_weights_frag = T.alloc_fragment([heads], dtype=accum_dtype)

                T.copy(IndexQ[bos + i_t, :, :], index_q_shared)
                T.copy(Weights[bos + i_t, :], weights_shared)
                T.fill(d_index_q_frag, 0)
                T.fill(d_weights_frag, 0)

                for i, j in T.Parallel(heads, dim):
                    index_q_shared[i, j] = index_q_shared[i, j] * sm_scale

                for bi_i in T.Pipelined(num_blocks, num_stages=num_stages):
                    i_st = bi_i * block_I
                    i_ed = (bi_i + 1) * block_I

                    indices_shared = T.alloc_shared([block_I], dtype=integer_dtype)
                    T.copy(TopkIndices[bos + i_t, i_st:i_ed], indices_shared)

                    index_k_shared = T.alloc_shared([block_I, dim], dtype=dtype)
                    for i, j in T.Parallel(block_I, dim):
                        pos = indices_shared[i]
                        index_k_shared[i, j] = T.if_then_else(
                            (pos > -1) & (pos <= i_t), IndexK[bos + pos, j], 0
                        )

                    attn_score_shared = T.alloc_shared([block_I], dtype=accum_dtype)
                    index_score_shared = T.alloc_shared([block_I], dtype=accum_dtype)
                    for i in T.Parallel(block_I):
                        attn_score_shared[i] = AttnScore[bos + i_t, i_st + i]
                        index_score_shared[i] = IndexScore[bos + i_t, i_st + i]

                    logits = T.alloc_fragment((block_I, heads), accum_dtype)
                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        logits,
                        transpose_A=False,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    for i, j in T.Parallel(block_I, heads):
                        logits[i, j] = T.max(logits[i, j], 0)

                    d_weights_i = T.alloc_fragment((block_I, heads), accum_dtype)
                    for i, j in T.Parallel(block_I, heads):
                        d_weights_i[i, j] = (index_score_shared[i] - attn_score_shared[i]) * logits[
                            i, j
                        ]
                    T.reduce_sum(d_weights_i, d_weights_frag, dim=0, clear=False)

                    d_logits_qk = T.alloc_shared((block_I, heads), accum_dtype)
                    d_logits_qk_cast1 = T.alloc_fragment((block_I, heads), dtype)
                    d_logits_qk_cast2 = T.alloc_fragment((block_I, heads), dtype)

                    for i, j in T.Parallel(block_I, heads):
                        d_relu = T.alloc_var(accum_dtype)
                        if logits[i, j] > 0:
                            d_relu = 1.0
                        else:
                            d_relu = 0.0
                        d_logits_qk[i, j] = (
                            (index_score_shared[i] - attn_score_shared[i])
                            * d_relu
                            * weights_shared[j]
                        )

                    T.copy(d_logits_qk, d_logits_qk_cast1)
                    T.gemm(
                        d_logits_qk_cast1,
                        index_k_shared,
                        d_index_q_frag,
                        transpose_A=True,
                        transpose_B=False,
                        clear_accum=False,
                    )

                    T.copy(d_logits_qk, d_logits_qk_cast2)
                    d_index_k_frag = T.alloc_fragment([block_I, dim], dtype=accum_dtype)
                    T.gemm(
                        d_logits_qk_cast2,
                        index_q_shared,
                        d_index_k_frag,
                        transpose_A=False,
                        transpose_B=False,
                        clear_accum=True,
                    )

                    for i, j in T.Parallel(block_I, dim):
                        pos = indices_shared[i]
                        if (pos > -1) & (pos <= i_t):
                            T.atomic_add(dIndexK[bos + pos, j], d_index_k_frag[i, j])

                for i, j in T.Parallel(heads, dim):
                    d_index_q_frag[i, j] = d_index_q_frag[i, j] * sm_scale

                T.copy(d_index_q_frag, dIndexQ[bos + i_t, :, :])
                T.copy(d_weights_frag, dWeights[bos + i_t, :])

        return tl_indexer_bwd_kernel

    # --- Sparse MLA TopK ReduceSum ---
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tl_sparse_mla_topk_reducesum(
        heads, dim, tail_dim, topk, kv_group=1, sm_scale=None, block_I=32, num_stages=2, threads=128
    ):
        assert dim == tilelang.math.next_power_of_2(dim)
        assert tail_dim == tilelang.math.next_power_of_2(tail_dim)
        assert topk % block_I == 0
        if sm_scale is None:
            sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        batch_plus_one = T.symbolic("batch_plus_one")
        seq_len = T.symbolic("seq_len")
        seq_len_kv = T.symbolic("seq_len_kv")

        head_kv = heads // kv_group
        indices_dtype = "int32"
        dtype = "bfloat16"
        accum_dtype = "float"

        padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
        if padded_H != head_kv:
            assert kv_group == 1
        BI = block_I
        NI = tilelang.cdiv(topk, block_I)
        D = dim
        D_tail = tail_dim

        if head_kv > 64:
            assert head_kv % 64 == 0
            REPLICATE_H = head_kv // 64
        else:
            REPLICATE_H = 1
        H_per_block = padded_H if REPLICATE_H == 1 else 64

        q_shape = [seq_len, heads, dim + tail_dim]
        kv_shape = [seq_len_kv, kv_group, dim + tail_dim]
        indices_shape = [seq_len, kv_group, topk]
        lse_shape = [seq_len, heads]
        reducesum_shape = [seq_len, kv_group, REPLICATE_H, topk]
        offsets_shape = [batch_plus_one]
        token_indices_shape = [seq_len, 2]

        @T.prim_func
        def tl_sparse_mla_topk_reducesum_kernel(
            Q: T.Tensor(q_shape, dtype),
            KV: T.Tensor(kv_shape, dtype),
            Indices: T.Tensor(indices_shape, indices_dtype),
            Lse: T.Tensor(lse_shape, accum_dtype),
            Offsets: T.Tensor(offsets_shape, indices_dtype),
            TokenIndices: T.Tensor(token_indices_shape, indices_dtype),
            ReduceSum: T.Tensor(reducesum_shape, accum_dtype),
        ):
            with T.Kernel(seq_len * REPLICATE_H, kv_group, threads=threads) as (bx, by):
                Q_shared = T.alloc_shared([H_per_block, D], dtype)
                Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
                KV_shared = T.alloc_shared([BI, D], dtype)
                K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
                mask = T.alloc_fragment([BI], "bool")

                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                reducesum = T.alloc_fragment([BI], accum_dtype)
                lse = T.alloc_fragment([H_per_block], accum_dtype)
                T.fill(lse, 0)

                b_s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
                b_i, s_i = TokenIndices[b_s_i, 0], TokenIndices[b_s_i, 1]
                bos, eos = Offsets[b_i], Offsets[b_i + 1]
                r_i = bx % REPLICATE_H
                g_i = by
                max_kv_i = s_i

                H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
                H1 = H0 + H_per_block

                T.copy(Q[bos + s_i, H0:H1, :D], Q_shared)
                T.copy(Q[bos + s_i, H0:H1, D:], Q_tail_shared)
                T.copy(Lse[bos + s_i, H0:H1], lse)

                for i_i in T.Pipelined(NI, num_stages=num_stages):
                    for bi_i in T.Parallel(BI):
                        mask[bi_i] = (Indices[bos + s_i, g_i, i_i * BI + bi_i] <= max_kv_i) & (
                            Indices[bos + s_i, g_i, i_i * BI + bi_i] != -1
                        )

                    for bi_i, d_i in T.Parallel(BI, D):
                        KV_shared[bi_i, d_i] = KV[
                            bos + Indices[bos + s_i, g_i, i_i * BI + bi_i], g_i, d_i
                        ]
                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        K_tail_shared[bi_i, d_i] = KV[
                            bos + Indices[bos + s_i, g_i, i_i * BI + bi_i], g_i, D + d_i
                        ]

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                    T.gemm(
                        Q_shared,
                        KV_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.gemm(
                        Q_tail_shared,
                        K_tail_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp(acc_s[h_i, bi_i] * sm_scale - lse[h_i])
                    T.reduce_sum(acc_s, reducesum, dim=0)
                    T.copy(reducesum, ReduceSum[bos + s_i, g_i, r_i, i_i * BI : i_i * BI + BI])

        return tl_sparse_mla_topk_reducesum_kernel


# =====================================================================================
# Section C: Unfused reference implementations
# =====================================================================================


def _ref_sparse_mla_fwd(Q, KV, Indices, offsets, sm_scale, dim_v):
    """Unfused sparse MLA forward (THD format). Returns (output, lse)."""
    Q = Q.float()
    KV = KV.float()
    all_o = []
    all_lse = []
    for i in range(offsets.shape[0] - 1):
        q = Q[None, offsets[i] : offsets[i + 1]]
        kv = KV[None, offsets[i] : offsets[i + 1]]
        indices = Indices[None, offsets[i] : offsets[i + 1]].clone()

        indices = indices.transpose(1, 2)
        b, sq, h, dim_q = q.shape
        b, sk, g, _ = kv.shape
        k = kv
        v = kv[..., :dim_v]
        g_index = g
        h_index = h // g

        compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32, device=q.device).view(
            -1, 1
        ) >= torch.arange(0, sk, dtype=torch.int32, device=q.device).view(1, -1)

        indices[(indices < 0) | (indices > sk)] = sk
        mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
        mask = mask[..., :-1]
        mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
        mask = mask.view(b, g_index, 1, sq, sk)

        q = q.view(b, sq, g, -1, dim_q)
        score = torch.einsum("bmghd,bngd->bghmn", q, k)
        sm_scale_val = dim_q**-0.5 if sm_scale is None else sm_scale
        score = score.masked_fill(~mask, float("-inf")).mul(sm_scale_val)
        p = score.softmax(dim=-1)
        p = p.view(b, g_index, h_index, -1, sq, sk).view(b, g, -1, sq, sk)
        o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
        o = o.reshape(b, sq, h, dim_v)
        all_o.append(o.squeeze(0))

        max_logits = score.amax(dim=-1).float()
        lse = torch.log((score.float() - max_logits.unsqueeze(-1)).exp().sum(dim=-1)) + max_logits
        lse = lse.reshape(b, h, sq).permute(0, 2, 1).squeeze(0)
        all_lse.append(lse)

    o = torch.cat(all_o, dim=0).to(torch.bfloat16)
    lse = torch.cat(all_lse, dim=0)
    return o, lse


def _ref_sparse_mla_bwd(q, kv, do, indices, offsets, sm_scale, d_v):
    """Unfused sparse MLA backward (THD format). Recomputes forward via autograd."""
    with torch.enable_grad():
        q = q.detach().clone().requires_grad_(True)
        kv = kv.detach().clone().requires_grad_(True)
        o_ref = _ref_sparse_mla_fwd(q, kv, indices, offsets, sm_scale, d_v)[0]
        o_ref.backward(do)
    return q.grad, kv.grad


def _ref_indexer_topk_reducesum(Q, Weights, K, topk, offsets):
    """Unfused indexer topk + softmax (THD format). Returns (topk_indices, topk_score)."""

    all_topk_indices = []
    all_topk_score = []
    for i in range(offsets.shape[0] - 1):
        q = Q[offsets[i] : offsets[i + 1]]
        weights = Weights[offsets[i] : offsets[i + 1]]
        k = K[offsets[i] : offsets[i + 1]]
        softmax_scale = q.shape[-1] ** -0.5
        s = q.shape[0]
        mask = (
            torch.arange(s, device=q.device)[:, None] >= torch.arange(s, device=q.device)[None, :]
        )
        logits = einops_einsum(q, k, "s1 h k, s2 k -> s1 h s2")
        logits = F.relu(logits)
        logits = (logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32) * softmax_scale
        logits = torch.where(mask, logits, float("-inf"))
        topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)
        topk_score = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
        all_topk_indices.append(topk_indices)
        all_topk_score.append(topk_score)
    return torch.cat(all_topk_indices, dim=0), torch.cat(all_topk_score, dim=0)


def _ref_indexer_bwd(Q, Weights, K, TopkIndices, AttnScore, offsets):
    """Unfused indexer backward (THD format). Returns (index_score, dQ, dWeights, dK)."""

    with torch.enable_grad():
        Q = Q.detach().clone().requires_grad_(True)
        Weights = Weights.detach().clone().requires_grad_(True)
        K = K.detach().clone().requires_grad_(True)
        softmax_scale = Q.shape[-1] ** -0.5
        all_loss = []
        all_log_topk_prob = []
        for i in range(offsets.shape[0] - 1):
            q = Q[offsets[i] : offsets[i + 1]]
            weights = Weights[offsets[i] : offsets[i + 1]]
            k = K[offsets[i] : offsets[i + 1]]
            topk_indices = TopkIndices[offsets[i] : offsets[i + 1]]
            attn_score = AttnScore[offsets[i] : offsets[i + 1]]
            s = q.shape[0]
            mask = (
                torch.arange(s, device=q.device)[:, None]
                >= torch.arange(s, device=q.device)[None, :]
            )
            logits = einops_einsum(q, k, "s1 h k, s2 k -> s1 h s2") * softmax_scale
            logits = F.relu(logits)
            score = (logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32)
            score = torch.where(mask, score, float("-inf"))
            valid_mask = topk_indices >= 0
            safe_indices = topk_indices.clamp(min=0)
            topk_value = torch.gather(score, dim=-1, index=safe_indices.to(torch.int64))
            topk_value = torch.where(valid_mask, topk_value, float("-inf"))
            log_topk_prob = F.log_softmax(topk_value, dim=-1, dtype=torch.float32)
            loss = F.kl_div(
                log_topk_prob.clip(-100, 0),
                attn_score.log().clip(-100, 0),
                log_target=True,
                reduction="sum",
            )
            all_loss.append(loss)
            all_log_topk_prob.append(log_topk_prob)
        loss = torch.stack(all_loss).sum()
        loss.backward()
    log_topk_prob = torch.cat(all_log_topk_prob, dim=0)
    return log_topk_prob.exp(), Q.grad, Weights.grad, K.grad


def _ref_sparse_mla_topk_reducesum(Q, K, TopkIndices, offsets, sm_scale):
    """
    Unfused sparse MLA topk score calculation.

    Args:
        Q: [s, heads, dim]
        K: [s, 1, dim]
        TopkIndices: [s, topk]
        offsets: [batch + 1]
        sm_scale: float

    Returns:
        attn_score: [seq_len, topk]
    """
    assert Q.ndim == 3 and K.ndim == 3 and TopkIndices.ndim == 2 and offsets.ndim == 1
    assert K.shape[0] == Q.shape[0] and K.shape[1] == 1 and K.shape[2] == Q.shape[2]
    assert TopkIndices.shape[0] == Q.shape[0]

    all_topk_score = []
    for i in range(offsets.shape[0] - 1):
        q = Q[offsets[i] : offsets[i + 1]].float()
        k = K[offsets[i] : offsets[i + 1]].squeeze(-2).float()
        topk_indices = TopkIndices[offsets[i] : offsets[i + 1]]
        seq_len = q.shape[0]
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).tril().unsqueeze(-2)
        score = torch.einsum("thd,sd->ths", q, k) * sm_scale
        score = torch.where(mask, score, float("-inf"))
        score = F.softmax(score, dim=-1, dtype=torch.float32)
        score = score.sum(dim=-2)
        valid_mask = topk_indices >= 0
        safe_indices = topk_indices.clamp(min=0)
        score = torch.gather(score, dim=-1, index=safe_indices.to(torch.int64))
        score = torch.where(valid_mask, score, 0.0)
        score = score / score.sum(dim=-1, keepdim=True)
        all_topk_score.append(score)
    return torch.cat(all_topk_score, dim=0)


# =====================================================================================
# Section D: Interface functions
# =====================================================================================


def sparse_mla_fwd_interface(q, kv, indices, offsets, sm_scale, d_v, use_unfused=False):
    """Sparse MLA forward (THD format)."""
    if use_unfused:
        return _ref_sparse_mla_fwd(q, kv, indices, offsets, sm_scale, d_v)
    else:
        seq_len, heads, dim_plus_tail_dim = q.shape
        dim = d_v
        tail_dim = dim_plus_tail_dim - dim
        _, kv_group, _ = kv.shape
        _, _, topk = indices.shape
        token_indices = prepare_token_indices(offsets)
        kernel = _tl_sparse_mla_fwd(
            heads, dim, tail_dim, topk, kv_group, sm_scale, block_I=32, num_stages=2, threads=128
        )
        out, lse = kernel(q, kv, indices, offsets, token_indices)
        return out, lse


def sparse_mla_bwd_interface(q, kv, o, do, indices, lse, offsets, sm_scale, d_v, use_unfused=False):
    """Sparse MLA backward. THD format. (o and lse are only used by the fused path.)"""
    if use_unfused:
        return _ref_sparse_mla_bwd(q, kv, do, indices, offsets, sm_scale, d_v)
    else:
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


def indexer_topk_reducesum_interface(q, weights, k, topk, offsets, use_unfused=False):
    """Indexer topk + softmax. THD format: q [S,H,D], weights [S,H], k [S,D]."""
    if use_unfused:
        return _ref_indexer_topk_reducesum(q, weights, k, topk, offsets)
    else:
        _, heads, dim = q.shape
        token_indices = prepare_token_indices(offsets)
        seq_len = q.shape[0]
        kernel = _tl_indexer_topk_reducesum(heads=heads, dim=dim, topk=topk, dtype="bfloat16")
        topk_indices = torch.zeros((seq_len, topk), device=q.device, dtype=torch.int32)
        topk_score = torch.zeros((seq_len, topk), device=q.device, dtype=torch.float32)
        kernel(q, weights, k, topk_indices, topk_score, offsets, token_indices)
        return topk_indices, topk_score


def indexer_bwd_interface(
    q, weights, k, attn_score, index_score, topk_indices, offsets, use_unfused=False
):
    """Indexer backward. THD format."""
    if use_unfused:
        return _ref_indexer_bwd(q, weights, k, topk_indices, attn_score, offsets)[1:]
    else:
        _, heads, dim = q.shape
        topk = topk_indices.shape[-1]
        token_indices = prepare_token_indices(offsets)
        dq = torch.zeros_like(q)
        dweights = torch.zeros_like(weights)
        dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
        kernel = _tl_indexer_bwd(heads, dim, topk)
        kernel(
            q,
            weights,
            k,
            dq,
            dweights,
            dk,
            attn_score,
            index_score,
            topk_indices,
            offsets,
            token_indices,
        )
        return dq, dweights, dk.to(q.dtype)


def sparse_mla_topk_reducesum_interface(
    q, kv, topk_indices, lse, offsets, dim_v, sm_scale, use_unfused=False
):
    """Sparse MLA topk reducesum. THD format."""
    if use_unfused:
        # ref expects: Q [s, h, d], K [s, 1, d], TopkIndices [s, topk], sm_scale
        # topk_indices: [s, 1, topk] -> [s, topk]
        topk_indices_2d = topk_indices.squeeze(-2) if topk_indices.ndim == 3 else topk_indices
        attn_score = _ref_sparse_mla_topk_reducesum(q, kv, topk_indices_2d, offsets, sm_scale)
        # output: [s, topk] -> [s, 1, topk] to match fused output
        return attn_score.unsqueeze(-2)
    else:
        seq_len, heads, dim_plus_tail_dim = q.shape
        tail_dim = dim_plus_tail_dim - dim_v
        topk = topk_indices.shape[-1]
        kv_group = kv.shape[-2] if kv.ndim == 3 else 1
        REPLICATE_H = max(heads // 64, 1)
        token_indices = prepare_token_indices(offsets)
        reducesum = torch.zeros(
            [seq_len, kv_group, REPLICATE_H, topk], dtype=torch.float32, device=q.device
        )
        kernel = _tl_sparse_mla_topk_reducesum(
            heads=heads,
            dim=dim_v,
            tail_dim=tail_dim,
            topk=topk,
            kv_group=kv_group,
            sm_scale=sm_scale,
        )
        kernel(q, kv, topk_indices, lse, offsets, token_indices, reducesum)
        reducesum = reducesum.sum(dim=-2)
        attn_score = reducesum / reducesum.sum(dim=-1, keepdim=True)
        return attn_score


# =====================================================================================
# Section E: DSAFunction -- autograd wrapper
# =====================================================================================


def _sbhd_to_thd(tensor):
    s, b, *rest = tensor.shape
    return tensor.transpose(0, 1).reshape(b * s, *rest).contiguous()


def _thd_to_sbhd(tensor, s, b):
    return tensor.reshape(b, s, *tensor.shape[1:]).transpose(0, 1).contiguous()


class DSAFunction(torch.autograd.Function):
    """Autograd function for DSA with indexer.

    Combines indexer forward/backward and sparse MLA forward/backward into a single
    autograd function. Handles sbhd <-> thd format conversion internally.

    Forward:
      1. indexer_topk_reducesum -> topk_indices, index_score
      2. sparse_mla_fwd -> output, lse

    Backward:
      1. sparse_mla_topk_reducesum -> attn_score (for indexer loss gradient)
      2. sparse_mla_bwd -> dq, dk (MLA gradients from upstream do)
      3. indexer_bwd -> dindex_q, dweights, dindex_k (indexer loss gradients, scaled by loss_coeff)
    """

    @staticmethod
    def forward(
        ctx,
        query,  # [s, b, np, hn]
        key,  # [s, b, 1, hn]
        index_q,  # [s, b, index_h, index_d]
        index_k,  # [s, b, index_d]
        weights,  # [s, b, index_h]
        offsets,  # [b+1] int32 cumulative sequence lengths
        topk,  # int
        v_channels,  # int
        sm_scale,  # float
        loss_coeff,  # float
        loss_logger,  # callable or None
        tp_group,  # process group or None
        use_unfused,  # bool
    ):
        assert tp_group is None or tp_group.size() == 1  # TP not supported yet
        assert query.ndim == 4
        sq, b = query.shape[:2]
        if offsets is None:
            offsets = torch.arange(0, b + 1, dtype=torch.int32, device=query.device) * sq
        else:
            assert b == 1

        query = _sbhd_to_thd(query)
        key = _sbhd_to_thd(key)
        index_q = _sbhd_to_thd(index_q)
        index_k = _sbhd_to_thd(index_k)
        weights = _sbhd_to_thd(weights)

        # 1. Indexer forward: topk selection + index score
        topk_indices, index_score = indexer_topk_reducesum_interface(
            index_q, weights, index_k, topk, offsets, use_unfused=use_unfused
        )

        # 2. Sparse MLA forward
        o, lse = sparse_mla_fwd_interface(
            query,
            key,
            topk_indices.unsqueeze(-2),
            offsets,
            sm_scale=sm_scale,
            d_v=v_channels,
            use_unfused=use_unfused,
        )

        ctx.save_for_backward(
            query, key, index_q, index_k, weights, topk_indices, index_score, o, lse, offsets
        )
        ctx.sq = sq
        ctx.b = b
        ctx.v_channels = v_channels
        ctx.sm_scale = sm_scale
        ctx.loss_coeff = loss_coeff
        ctx.loss_logger = loss_logger
        ctx.use_unfused = use_unfused

        o = _thd_to_sbhd(o, sq, b)
        return o

    @staticmethod
    def backward(ctx, do):
        (query, key, index_q, index_k, weights, topk_indices, index_score, o, lse, offsets) = (
            ctx.saved_tensors
        )

        do = _sbhd_to_thd(do)

        # 1. Compute attn_score for indexer backward
        attn_score = sparse_mla_topk_reducesum_interface(
            query,
            key,
            topk_indices.unsqueeze(-2),
            lse,
            offsets,
            dim_v=ctx.v_channels,
            sm_scale=ctx.sm_scale,
            use_unfused=ctx.use_unfused,
        ).squeeze(-2)

        # Log indexer loss (monitoring only)
        if ctx.loss_logger is not None:
            log_index = F.log_softmax(index_score, dim=-1, dtype=torch.float32)
            kl_loss = F.kl_div(
                log_index.clip(-100, 0),
                attn_score.log().clip(-100, 0),
                log_target=True,
                reduction="sum",
            )
            ctx.loss_logger(kl_loss * ctx.loss_coeff)

        # 2. Sparse MLA backward
        dq, dk = sparse_mla_bwd_interface(
            query,
            key,
            o,
            do,
            topk_indices.unsqueeze(-2),
            lse,
            offsets,
            sm_scale=ctx.sm_scale,
            d_v=ctx.v_channels,
            use_unfused=ctx.use_unfused,
        )

        # 3. Indexer backward
        dindex_q, dweights, dindex_k = indexer_bwd_interface(
            index_q,
            weights,
            index_k,
            attn_score,
            index_score,
            topk_indices,
            offsets,
            use_unfused=ctx.use_unfused,
        )

        # Scale indexer gradients by loss_coeff
        dindex_q *= ctx.loss_coeff
        dweights *= ctx.loss_coeff
        dindex_k *= ctx.loss_coeff

        dq = _thd_to_sbhd(dq, ctx.sq, ctx.b)
        dk = _thd_to_sbhd(dk, ctx.sq, ctx.b)
        dindex_q = _thd_to_sbhd(dindex_q, ctx.sq, ctx.b)
        dindex_k = _thd_to_sbhd(dindex_k, ctx.sq, ctx.b)
        dweights = _thd_to_sbhd(dweights, ctx.sq, ctx.b)

        return dq, dk, dindex_q, dindex_k, dweights, None, None, None, None, None, None, None, None
