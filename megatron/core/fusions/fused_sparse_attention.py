# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import tilelang
    from tilelang import language as T

    HAVE_TILELANG = True
except ImportError:
    HAVE_TILELANG = False

if not HAVE_TILELANG:
    tilelang = MagicMock()
    tilelang.jit = null_decorator
    T = MagicMock()


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_attention_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    """
    Forward kernel.
    """
    assert dim == tilelang.math.next_power_of_2(
        dim
    ), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim
    ), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert (
        topk % block_I == 0
    ), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    k_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    v_shape = [batch, seq_len_kv, kv_group, dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, other wise you should handle Q copy and "
            "Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H "
            "would be handled automatically)"
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            K_shared = T.alloc_shared([BI, D], dtype)
            V_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            O_shared = T.alloc_shared([H_per_block, D], dtype)
            Lse_shared = T.alloc_shared([H_per_block], accum_dtype)
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
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i
            max_kv_i = q_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):

                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i

                for bi_i, d_i in T.Parallel(BI, D):
                    K_shared[bi_i, d_i] = K[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = K[
                        b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i
                    ]

                for bi_i, d_i in T.Parallel(BI, D):
                    V_shared[bi_i, d_i] = V[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
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
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, O_shared)
            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse_shared)
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


def sparse_attention_fwd_interface(
    q,
    k,
    v,
    indices,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v=512,
    block_I=64,
    num_stages=2,
    threads=256,
):
    """
    Forward kernel interface.
    """
    assert HAVE_TILELANG, "tilelang is not installed"

    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = k.shape

    dim = d_v

    assert k.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert k.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    kernel = sparse_attention_fwd(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group,
        sm_scale,
        is_casual,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
    )
    out, lse = kernel(q, k, v, indices)
    return out, lse


@tilelang.jit(out_idx=[-1])
def preprocess(B, S, H, D, block_ND=32, num_stages=5, dtype="bfloat16", accum_dtype="float"):
    """
    Preprocess kernel for backward.
    """
    assert dtype == "bfloat16"
    assert accum_dtype == "float"
    shape = [B, S, H, D]

    @T.prim_func
    def preprocess_kernel(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([B, S, H], accum_dtype),
    ):
        with T.Kernel(H, T.ceildiv(S, block_ND), B) as (bx, by, bz):
            o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            delta = T.alloc_fragment([block_ND], accum_dtype)
            acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
                T.copy(
                    O[
                        bz,
                        by * block_ND : (by + 1) * block_ND,
                        bx,
                        k * block_ND : (k + 1) * block_ND,
                    ],
                    o,
                )
                T.copy(
                    dO[
                        bz,
                        by * block_ND : (by + 1) * block_ND,
                        bx,
                        k * block_ND : (k + 1) * block_ND,
                    ],
                    do,
                )
                for i, j in T.Parallel(block_ND, block_ND):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, by * block_ND : (by + 1) * block_ND, bx])

    return preprocess_kernel


@tilelang.jit(out_idx=[-1])
def postprocess(
    B, S_kv, D, D_tail, kv_group=1, block_N=64, threads=128, dtype="bfloat16", accum_dtype="float"
):
    """
    Postprocess kernel for backward.
    """
    assert dtype == "bfloat16"
    assert accum_dtype == "float"
    dkv_shape = [B, S_kv, kv_group, D + D_tail]

    @T.prim_func
    def postprocess_kernel(
        dKV: T.Tensor(dkv_shape, accum_dtype), dKV_out: T.Tensor(dkv_shape, dtype)
    ):
        with T.Kernel(T.ceildiv(S_kv, block_N), kv_group, B, threads=threads) as (bx, by, bz):
            T.copy(
                dKV[bz, bx * block_N : (bx + 1) * block_N, by, :],
                dKV_out[bz, bx * block_N : (bx + 1) * block_N, by, :],
            )

    return postprocess_kernel


@tilelang.jit(
    out_idx=[-3],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_attention_bwd(
    B,
    S,
    S_kv,
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
    """
    Backward kernel.
    """
    assert is_causal == True, 'non-casual is not supported now'
    assert (
        topk % block_size == 0
    ), 'otherwise will load some index=0 thus causing wrong kv to be loaded'
    assert dtype == "bfloat16"
    assert accum_dtype == "float"
    assert indices_dtype == "int32"

    if sm_scale is None:
        sm_scale = (D + D_tail) ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    H_kv = H // kv_group
    q_shape = [B, S, H, D + D_tail]
    k_shape = [B, S_kv, kv_group, D + D_tail]
    v_shape = [B, S_kv, kv_group, D]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, kv_group, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]
    assert indices_dtype == "int32"
    assert dtype == "bfloat16"
    assert accum_dtype == "float"

    H = H_kv
    padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2

    @T.prim_func
    def sparse_mla_bwd_kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dK: T.Tensor(k_shape, accum_dtype),
        dV: T.Tensor(v_shape, accum_dtype),
    ):
        with T.Kernel(S, B, kv_group, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([padded_H, D], dtype)
            Q_tail_shared = T.alloc_shared([padded_H, D_tail], dtype)
            K_shared = T.alloc_shared([BS, D], dtype)
            K_tail_shared = T.alloc_shared([BS, D_tail], dtype)
            V_shared = T.alloc_shared([BS, D], dtype)
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
            acc_dk = T.alloc_fragment([BS, D], accum_dtype)
            acc_dk_tail = T.alloc_fragment([BS, D_tail], accum_dtype)
            acc_dv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dk_shared = T.view(K_shared, shape=[BS // split_store, D], dtype=accum_dtype)
            acc_dv_shared = T.view(V_shared, shape=[BS // split_store, D], dtype=accum_dtype)
            acc_dk_tail_shared = T.view(
                K_tail_shared, shape=[BS // split_store, D_tail], dtype=accum_dtype
            )

            max_kv_i = s_i

            T.copy(Q[by, s_i, bz * padded_H : (bz + 1) * padded_H, :D], Q_shared)
            T.copy(Q[by, s_i, bz * padded_H : (bz + 1) * padded_H, D:], Q_tail_shared)
            T.copy(dO[by, s_i, bz * padded_H : (bz + 1) * padded_H, :D], dO_shared)

            T.clear(acc_dq)
            T.clear(acc_dq_tail)

            T.annotate_layout(
                {
                    dQ_shared: tilelang.layout.make_swizzled_layout(dQ_shared),
                    dQ_tail_shared: tilelang.layout.make_swizzled_layout(dQ_tail_shared),
                }
            )

            # Process each block of indices
            for i_i in T.Pipelined(NS, num_stages=num_stages):
                # Check which indices are valid
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, bz, i_i * BS + bi_i] <= max_kv_i

                # Compute attention scores
                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                # Load KV, V for this block of indices
                for bi_i, d_i in T.Parallel(BS, D):
                    K_shared[bi_i, d_i] = K[by, Indices[by, s_i, bz, i_i * BS + bi_i], bz, d_i]

                T.gemm(Q_shared, K_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for bi_i, d_i in T.Parallel(BS, D_tail):
                    K_tail_shared[bi_i, d_i] = K[
                        by, Indices[by, s_i, bz, i_i * BS + bi_i], bz, D + d_i
                    ]
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_p,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2
                        - Lse[by, s_i, bz * padded_H + h_i]
                    )

                T.copy(acc_p, P_shared_cast)

                # Load KV, V for this block of indices
                for bi_i, d_i in T.Parallel(BS, D):
                    V_shared[bi_i, d_i] = V[by, Indices[by, s_i, bz, i_i * BS + bi_i], bz, d_i]

                T.gemm(
                    dO_shared,
                    V_shared,
                    acc_dp,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )

                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i]
                        * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * padded_H + h_i])
                        * sm_scale
                    )

                T.copy(acc_dp, dP_shared_cast)
                T.gemm(dP_shared_cast, K_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(dP_shared_cast, K_tail_shared, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol)

                T.gemm(
                    dP_shared_cast,
                    Q_shared,
                    acc_dk,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )

                T.gemm(
                    P_shared_cast,
                    dO_shared,
                    acc_dv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )

                T.clear(acc_dk_tail)
                T.gemm(
                    dP_shared_cast,
                    Q_tail_shared,
                    acc_dk_tail,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, D):
                        if bi_i < BS // split_store:
                            acc_dk_shared[bi_i, d_i] = acc_dk[bi_i + s * (BS // split_store), d_i]
                        if bi_i < BS // split_store:
                            acc_dv_shared[bi_i, d_i] = acc_dv[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS, D_tail):
                        if bi_i < BS // split_store:
                            acc_dk_tail_shared[bi_i, d_i] = acc_dk_tail[
                                bi_i + s * (BS // split_store), d_i
                            ]

                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        T.atomic_addx4(
                            dK[
                                by,
                                Indices[by, s_i, bz, i_i * BS + bi_i + s * (BS // split_store)],
                                bz,
                                d_i * 4,
                            ],
                            acc_dk_shared[bi_i, d_i * 4],
                        )
                        T.atomic_addx4(
                            dV[
                                by,
                                Indices[by, s_i, bz, i_i * BS + bi_i + s * (BS // split_store)],
                                bz,
                                d_i * 4,
                            ],
                            acc_dv_shared[bi_i, d_i * 4],
                        )

                    # Atomically update dKV, dKV_tail tensors
                    for bi_i, d_i in T.Parallel(BS // split_store, D_tail // 4):
                        T.atomic_addx4(
                            dK[
                                by,
                                Indices[by, s_i, bz, i_i * BS + bi_i + s * (BS // split_store)],
                                bz,
                                D + d_i * 4,
                            ],
                            acc_dk_tail_shared[bi_i, d_i * 4],
                        )

            # Store the accumulated dQ
            T.copy(acc_dq, dQ_shared)
            T.copy(acc_dq_tail, dQ_tail_shared)

            T.copy(dQ_shared, dQ[by, s_i, bz * padded_H : (bz + 1) * padded_H, :D])
            T.copy(dQ_tail_shared, dQ[by, s_i, bz * padded_H : (bz + 1) * padded_H, D:])

    return sparse_mla_bwd_kernel


def sparse_attention_bwd_interface(
    q,
    k,
    v,
    o,
    do,
    indices,
    lse,
    sm_scale=None,
    is_casual=True,
    return_kernel=False,
    delta=None,
    d_v=512,
):
    """
    Backward kernel interface.
    """
    assert HAVE_TILELANG, "tilelang is not installed"

    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert indices.is_contiguous()
    assert lse.is_contiguous()
    B, S, H, dim_plus_tail_dim = q.shape
    _, S_kv, kv_group, _ = k.shape
    assert k.shape[-1] == dim_plus_tail_dim
    assert k.shape[0] == B
    # dim should be assigned
    D = d_v

    D_tail = dim_plus_tail_dim - D
    topk = indices.shape[-1]
    assert indices.shape == (B, S, kv_group, topk)
    assert lse.shape == (B, S, H)

    # Get kernels
    preprocess_kernel = preprocess(B, S, H, D)

    bwd_kernel = sparse_attention_bwd(B, S, S_kv, H, D, D_tail, topk, kv_group, sm_scale, is_casual)
    postprocess_kernel_k = postprocess(B, S_kv, D, D_tail, kv_group)
    postprocess_kernel_v = postprocess(B, S_kv, D, 0, kv_group)

    if delta is None:
        delta = preprocess_kernel(o, do)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    dq = bwd_kernel(q, k, v, do, indices, lse, delta, dk, dv)
    dk = postprocess_kernel_k(dk)
    dv = postprocess_kernel_v(dv)

    return dq, dk, dv


class FusedSparseAttention(torch.autograd.Function):
    """
    Fused sparse attention kernel wrapper.
    """

    @staticmethod
    def forward(ctx, q, k, v, indices, d_v, sm_scale=None):
        """
        Forward pass.
        """
        out, lse = sparse_attention_fwd_interface(q, k, v, indices, sm_scale=sm_scale, d_v=d_v)
        ctx.save_for_backward(q, k, v, indices, out, lse)
        ctx.sm_scale = sm_scale
        ctx.d_v = d_v
        return out

    @staticmethod
    def backward(ctx, do):
        """
        Backward pass.
        """
        q, k, v, indices, out, lse = ctx.saved_tensors
        dq, dk, dv = sparse_attention_bwd_interface(
            q, k, v, out, do, indices, lse, sm_scale=ctx.sm_scale, d_v=ctx.d_v
        )
        return dq, dk, dv, None, None
