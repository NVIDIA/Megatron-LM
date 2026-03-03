# ruff: noqa
# Adapted from:
# https://github.com/tile-ai/tilelang/blob/e666d2d3cc483829c57618c9ebf2e4f4ada0819d/
# examples/deepseek_v32/sparse_mla_fwd.py
import os
from collections import OrderedDict

import tilelang
import torch
from tilelang import language as T

_TILELANG_KERNEL_CACHE_MAX = 64
_tilelang_sparse_mla_fwd_kernel_cache = OrderedDict()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _round_up(x: int, multiple: int) -> int:
    if multiple <= 1:
        return x
    return _ceil_div(x, multiple) * multiple


def _cache_put_lru(cache: OrderedDict, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _TILELANG_KERNEL_CACHE_MAX:
        cache.popitem(last=False)


def _get_sparse_mla_fwd_kernel(
    heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    kv_group: int,
    sm_scale,
    is_causal: bool,
    block_I: int,
    num_stages: int,
    threads: int,
):
    key = (heads, dim, tail_dim, topk, kv_group, sm_scale, is_causal, block_I, num_stages, threads)
    kernel = _tilelang_sparse_mla_fwd_kernel_cache.pop(key, None)
    if kernel is None:
        kernel = sparse_mla_fwd(
            heads,
            dim,
            tail_dim,
            topk,
            kv_group,
            sm_scale,
            is_causal,
            block_I=block_I,
            num_stages=num_stages,
            threads=threads,
        )
    _cache_put_lru(_tilelang_sparse_mla_fwd_kernel_cache, key, kernel)
    return kernel


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd(
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
    """Build sparse-MLA forward kernel."""
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
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, otherwise handle Q/Output copy with "
            "your own mask (for kv_group==1, g_i*padded_H:(g_i+1)*padded_H is handled)"
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
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
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
                    # Changed here for thd
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] != -1

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i
                    ]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i
                    ]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(
                    Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
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
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
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
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


def sparse_mla_fwd_interface(
    q,
    kv,
    indices,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v=512,
    block_I=64,
    num_stages=2,
    threads=256,
):
    """Run sparse-MLA forward kernel and return (out, lse)."""
    seq_bucket = _env_int("MCORE_DSA_TILELANG_SEQ_BUCKET", 256)
    topk_bucket = _env_int("MCORE_DSA_TILELANG_TOPK_BUCKET", block_I)

    q = q.unsqueeze(0)
    kv = kv.unsqueeze(0)
    indices = indices.unsqueeze(0)

    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, kv_dim = kv.shape
    assert (
        kv_dim == dim_plus_tail_dim
    ), "q and kv must have the same embedding dimension on the last axis"
    assert (
        dim_plus_tail_dim == 576
    ), "TileLang sparse MLA fwd is currently specialized for dim_plus_tail_dim=576"
    dim = d_v
    assert 0 < dim <= dim_plus_tail_dim, f"d_v must be in (0, {dim_plus_tail_dim}], but got {dim}"

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    seq_len_bucketed = _round_up(seq_len, seq_bucket)
    seq_len_kv_bucketed = _round_up(seq_len_kv, seq_bucket)
    topk_bucketed = _round_up(_round_up(topk, topk_bucket), block_I)

    if seq_len_bucketed != seq_len:
        q_padded = torch.zeros(
            (batch, seq_len_bucketed, heads, dim_plus_tail_dim), dtype=q.dtype, device=q.device
        )
        q_padded[:, :seq_len].copy_(q)
        q = q_padded

    if seq_len_kv_bucketed != seq_len_kv:
        kv_padded = torch.zeros(
            (batch, seq_len_kv_bucketed, kv_group, dim_plus_tail_dim),
            dtype=kv.dtype,
            device=kv.device,
        )
        kv_padded[:, :seq_len_kv].copy_(kv)
        kv = kv_padded

    if seq_len_bucketed != seq_len or topk_bucketed != topk:
        indices_padded = torch.full(
            (batch, seq_len_bucketed, kv_group, topk_bucketed),
            -1,
            dtype=indices.dtype,
            device=indices.device,
        )
        indices_padded[:, :seq_len, :, :topk].copy_(indices)
        indices = indices_padded

    kernel = _get_sparse_mla_fwd_kernel(
        heads=heads,
        dim=dim,
        tail_dim=tail_dim,
        topk=topk_bucketed,
        kv_group=kv_group,
        sm_scale=sm_scale,
        is_causal=is_casual,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
    )
    out, lse = kernel(q, kv, indices)
    out = out[:, :seq_len].contiguous().squeeze(0)
    lse = lse[:, :seq_len].contiguous().squeeze(0)
    return out, lse
