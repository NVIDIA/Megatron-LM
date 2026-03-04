# ruff: noqa
# Adapted from:
# https://github.com/tile-ai/tilelang/blob/4ff81c7d40803d269569e157e847623e84553f78/
# examples/deepseek_v32/sparse_mla_bwd.py
import os
import threading
from collections import OrderedDict

import tilelang
import torch
from tilelang import language as T

_SPARSE_MLA_BWD_BLOCK_SIZE = 32
_tilelang_sparse_mla_preprocess_kernel_cache = OrderedDict()
_tilelang_sparse_mla_bwd_kernel_cache = OrderedDict()
_tilelang_sparse_mla_postprocess_kernel_cache = OrderedDict()
_tilelang_sparse_mla_bwd_cache_lock = threading.Lock()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


_TILELANG_KERNEL_CACHE_MAX = _env_int("MCORE_DSA_TILELANG_KERNEL_CACHE_MAX", 512)


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


def _normalize_sm_scale(sm_scale):
    if sm_scale is None:
        return None
    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.detach().item())
    else:
        sm_scale = float(sm_scale)
    # Avoid tiny floating-point jitter creating cache-key churn.
    return round(sm_scale, 12)


def _get_preprocess_kernel(H: int, D: int):
    key = (H, D)
    with _tilelang_sparse_mla_bwd_cache_lock:
        kernel = _tilelang_sparse_mla_preprocess_kernel_cache.pop(key, None)
        if kernel is None:
            kernel = preprocess(H, D)
        _cache_put_lru(_tilelang_sparse_mla_preprocess_kernel_cache, key, kernel)
        return kernel


def _get_bwd_kernel(
    H: int, D: int, D_tail: int, topk: int, kv_group: int, sm_scale, is_causal: bool
):
    key = (H, D, D_tail, topk, kv_group, _normalize_sm_scale(sm_scale), is_causal)
    with _tilelang_sparse_mla_bwd_cache_lock:
        kernel = _tilelang_sparse_mla_bwd_kernel_cache.pop(key, None)
        if kernel is None:
            kernel = bwd(H, D, D_tail, topk, kv_group, sm_scale, is_causal)
        _cache_put_lru(_tilelang_sparse_mla_bwd_kernel_cache, key, kernel)
        return kernel


def _get_postprocess_kernel(D: int, D_tail: int, kv_group: int):
    key = (D, D_tail, kv_group)
    with _tilelang_sparse_mla_bwd_cache_lock:
        kernel = _tilelang_sparse_mla_postprocess_kernel_cache.pop(key, None)
        if kernel is None:
            kernel = postprocess(D, D_tail, kv_group)
        _cache_put_lru(_tilelang_sparse_mla_postprocess_kernel_cache, key, kernel)
        return kernel


@tilelang.jit(out_idx=[-1])
def preprocess(H, D, block_ND=32, num_stages=5, dtype=T.bfloat16, accum_dtype=T.float32):
    """Build preprocessing kernel that computes Delta = sum(O * dO) per row/head."""
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    shape = [batch, seq_len, H, D]

    @T.prim_func
    def preprocess_kernel(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([batch, seq_len, H], accum_dtype),
    ):
        with T.Kernel(H, T.ceildiv(seq_len, block_ND), batch) as (bx, by, bz):
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
    D, D_tail, kv_group=1, block_N=64, threads=128, dtype=T.bfloat16, accum_dtype=T.float32
):
    """Build postprocess kernel that casts/exports accumulated dKV."""
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    batch = T.dynamic("batch")
    seq_len_kv = T.dynamic("seq_len_kv")
    dkv_shape = [batch, seq_len_kv, kv_group, D + D_tail]

    @T.prim_func
    def postprocess_kernel(
        dKV: T.Tensor(dkv_shape, accum_dtype), dKV_out: T.Tensor(dkv_shape, dtype)
    ):
        with T.Kernel(T.ceildiv(seq_len_kv, block_N), kv_group, batch, threads=threads) as (
            bx,
            by,
            bz,
        ):
            T.copy(
                dKV[bz, bx * block_N : (bx + 1) * block_N, by, :],
                dKV_out[bz, bx * block_N : (bx + 1) * block_N, by, :],
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
def bwd(
    H,
    D,
    D_tail,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_size=32,
    num_stages=0,
    threads=128,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    """Build sparse-MLA backward kernel."""
    assert is_causal == True, "non-casual is not supported now"
    assert (
        topk % block_size == 0
    ), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    assert indices_dtype == T.int32

    if sm_scale is None:
        sm_scale = (D + D_tail) ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    H_kv = H // kv_group
    q_shape = [batch, seq_len, H, D + D_tail]
    k_shape = [batch, seq_len_kv, kv_group, D + D_tail]
    o_shape = [batch, seq_len, H, D]
    indices_shape = [batch, seq_len, kv_group, topk]
    delta_shape = [batch, seq_len, H]
    lse_shape = [batch, seq_len, H]
    assert indices_dtype == T.int32
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32

    H = H_kv
    padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
    block_H = min(64, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
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
        dQ: T.Tensor(q_shape, dtype),
        dKV: T.Tensor(k_shape, accum_dtype),
    ):
        with T.Kernel(seq_len, batch, kv_group * NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            Q_tail_shared = T.alloc_shared([block_H, D_tail], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            KV_tail_shared = T.alloc_shared([BS, D_tail], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, D], dtype)
            dQ_tail_shared = T.alloc_shared([block_H, D_tail], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dq_tail = T.alloc_fragment([block_H, D_tail], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_tail = T.alloc_fragment([BS, D_tail], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)
            acc_dkv_tail_shared = T.alloc_shared([BS // split_store, D_tail], accum_dtype)

            # max_kv_i = s_i

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, D:], Q_tail_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)

            T.clear(acc_dq)
            T.clear(acc_dq_tail)

            # Process each block of indices
            for i_i in T.Pipelined(NS, num_stages=num_stages):
                # Check which indices are valid
                for bi_i in T.Parallel(BS):
                    # Changed here for thd
                    mask[bi_i] = Indices[by, s_i, bz // NH, i_i * BS + bi_i] != -1

                # Compute attention scores
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                # Load KV, V for this block of indices
                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[
                        by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, d_i
                    ]

                T.gemm(
                    Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol
                )

                for bi_i, d_i in T.Parallel(BS, D_tail):
                    KV_tail_shared[bi_i, d_i] = KV[
                        by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, D + d_i
                    ]
                T.gemm(
                    Q_tail_shared,
                    KV_tail_shared,
                    acc_p,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2
                        - Lse[by, s_i, bz * block_H + h_i]
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

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i]
                        * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i])
                        * sm_scale
                    )

                T.copy(acc_dp, dP_shared_cast)
                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol)

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
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS, D_tail):
                        if bi_i < BS // split_store:
                            acc_dkv_tail_shared[bi_i, d_i] = acc_dkv_tail[
                                bi_i + s * (BS // split_store), d_i
                            ]

                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        T.atomic_addx4(
                            dKV[
                                by,
                                Indices[
                                    by, s_i, bz // NH, i_i * BS + bi_i + s * (BS // split_store)
                                ],
                                bz // NH,
                                d_i * 4,
                            ],
                            acc_dkv_shared[bi_i, d_i * 4],
                        )

                    # Atomically update dKV, dKV_tail tensors
                    for bi_i, d_i in T.Parallel(BS // split_store, D_tail // 4):
                        T.atomic_addx4(
                            dKV[
                                by,
                                Indices[
                                    by, s_i, bz // NH, i_i * BS + bi_i + s * (BS // split_store)
                                ],
                                bz // NH,
                                D + d_i * 4,
                            ],
                            acc_dkv_tail_shared[bi_i, d_i * 4],
                        )

            # Store the accumulated dQ
            T.copy(acc_dq, dQ_shared)
            T.copy(acc_dq_tail, dQ_tail_shared)

            T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])
            T.copy(dQ_tail_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, D:])

    return sparse_mla_bwd_kernel


def sparse_mla_bwd(
    q, kv, o, do, indices, lse, sm_scale=None, is_casual=True, return_kernel=False, delta=None
):
    """Run sparse-MLA backward kernels and return (dq, dkv)."""
    seq_bucket = _env_int("MCORE_DSA_TILELANG_SEQ_BUCKET", 256)
    topk_bucket = _env_int("MCORE_DSA_TILELANG_TOPK_BUCKET", _SPARSE_MLA_BWD_BLOCK_SIZE)

    q = q.unsqueeze(0)
    kv = kv.unsqueeze(0)
    o = o.unsqueeze(0)
    do = do.unsqueeze(0)
    indices = indices.unsqueeze(0)
    lse = lse.unsqueeze(0)

    assert q.is_contiguous()
    assert kv.is_contiguous()
    assert indices.is_contiguous()
    assert lse.is_contiguous()
    B, S, H, dim_plus_tail_dim = q.shape
    _, S_kv, kv_group, _ = kv.shape
    assert kv.shape[-1] == dim_plus_tail_dim
    assert kv.shape[0] == B
    # This copied kernel currently assumes a fixed base value-channel dimension.
    D = 512
    assert (
        dim_plus_tail_dim >= D
    ), f"Invalid dimensions: dim_plus_tail_dim={dim_plus_tail_dim} is smaller than base D={D}"

    D_tail = dim_plus_tail_dim - D
    topk = indices.shape[-1]
    assert indices.shape == (B, S, kv_group, topk)
    assert lse.shape == (B, S, H)

    seq_bucketed = _round_up(S, seq_bucket)
    seq_kv_bucketed = _round_up(S_kv, seq_bucket)
    topk_bucketed = _round_up(_round_up(topk, topk_bucket), _SPARSE_MLA_BWD_BLOCK_SIZE)

    if seq_bucketed != S:
        q_padded = torch.zeros(
            (B, seq_bucketed, H, dim_plus_tail_dim), dtype=q.dtype, device=q.device
        )
        q_padded[:, :S].copy_(q)
        q = q_padded

        o_padded = torch.zeros((B, seq_bucketed, H, D), dtype=o.dtype, device=o.device)
        o_padded[:, :S].copy_(o)
        o = o_padded

        do_padded = torch.zeros((B, seq_bucketed, H, D), dtype=do.dtype, device=do.device)
        do_padded[:, :S].copy_(do)
        do = do_padded

        lse_padded = torch.zeros((B, seq_bucketed, H), dtype=lse.dtype, device=lse.device)
        lse_padded[:, :S].copy_(lse)
        lse = lse_padded

    if seq_kv_bucketed != S_kv:
        kv_padded = torch.zeros(
            (B, seq_kv_bucketed, kv_group, dim_plus_tail_dim), dtype=kv.dtype, device=kv.device
        )
        kv_padded[:, :S_kv].copy_(kv)
        kv = kv_padded

    if seq_bucketed != S or topk_bucketed != topk:
        indices_padded = torch.full(
            (B, seq_bucketed, kv_group, topk_bucketed),
            -1,
            dtype=indices.dtype,
            device=indices.device,
        )
        indices_padded[:, :S, :, :topk].copy_(indices)
        indices = indices_padded

    if delta is not None:
        if delta.ndim == 2:
            delta = delta.unsqueeze(0)
        if seq_bucketed != S:
            delta_padded = torch.zeros((B, seq_bucketed, H), dtype=delta.dtype, device=delta.device)
            delta_padded[:, :S].copy_(delta)
            delta = delta_padded

    # Get kernels
    preprocess_kernel = _get_preprocess_kernel(H, D)
    bwd_kernel = _get_bwd_kernel(H, D, D_tail, topk_bucketed, kv_group, sm_scale, is_casual)
    postprocess_kernel = _get_postprocess_kernel(D, D_tail, kv_group)

    if delta is None:
        delta = preprocess_kernel(o, do)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    dq = bwd_kernel(q, kv, do, indices, lse, delta, dkv)
    dkv = postprocess_kernel(dkv)

    dq = dq[:, :S].contiguous()
    dkv = dkv[:, :S_kv].contiguous()

    dq = dq.squeeze(0)
    dkv = dkv.squeeze(0)

    return dq, dkv
