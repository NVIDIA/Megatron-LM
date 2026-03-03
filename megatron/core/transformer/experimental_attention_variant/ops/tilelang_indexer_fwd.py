# ruff: noqa
# Adapted from:
# https://github.com/tile-ai/tilelang/blob/4956b5835fa554af6c03d4a6289cad44bf310869/
# examples/deepseek_v32/fp8_lighting_indexer.py
from collections import OrderedDict

import tilelang
import torch
from tilelang import language as T

_TILELANG_KERNEL_CACHE_MAX = 64
_tilelang_indexer_fwd_kernel_cache = OrderedDict()
_tilelang_indexer_clean_logits_kernel_cache = OrderedDict()


def _cache_put_lru(cache: OrderedDict, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _TILELANG_KERNEL_CACHE_MAX:
        cache.popitem(last=False)


def _get_clean_logits_kernel(threads: int = 512, block_K: int = 4096):
    key = (threads, block_K)
    kernel = _tilelang_indexer_clean_logits_kernel_cache.pop(key, None)
    if kernel is None:
        kernel = clean_logits_(threads=threads, block_K=block_K)
    _cache_put_lru(_tilelang_indexer_clean_logits_kernel_cache, key, kernel)
    return kernel


def _get_indexer_fwd_kernel(
    heads: int, index_dim: int, block_N: int = 256, num_stages: int = 3, threads: int = 512
):
    key = (heads, index_dim, block_N, num_stages, threads)
    kernel = _tilelang_indexer_fwd_kernel_cache.pop(key, None)
    if kernel is None:
        kernel = tl_indexer_fwd_impl(
            heads=heads,
            index_dim=index_dim,
            block_N=block_N,
            num_stages=num_stages,
            threads=threads,
        )
    _cache_put_lru(_tilelang_indexer_fwd_kernel_cache, key, kernel)
    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tl_indexer_fwd_impl(heads, index_dim, block_N=256, num_stages=3, threads=512, block_Q=None):
    """Build tilelang forward kernel for sparse indexer logits."""
    if block_Q is None:
        block_Q = 128 // heads
    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    logits_shape = [seq_len, seq_len_kv]

    @T.prim_func
    def tl_indexer_fwd_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            logits_shared = T.alloc_shared([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)

            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                q_idx = seq_len_i + bq_i
                if q_idx < seq_len:
                    k_s = T.max(T.min(CuSeqLenKS[q_idx], seq_len_kv), 0)
                    cu_k_s_min = T.min(cu_k_s_min, k_s)
            for bq_i in T.serial(block_Q):
                q_idx = seq_len_i + bq_i
                if q_idx < seq_len:
                    k_e = T.max(T.min(CuSeqLenKE[q_idx], seq_len_kv), 0)
                    cu_k_e_max = T.max(cu_k_e_max, k_e)

            # Clamp bounds to [0, seq_len_kv] and normalize empty rows.
            cu_k_s_min = T.max(cu_k_s_min, 0)
            cu_k_s_min = T.min(cu_k_s_min, seq_len_kv)
            cu_k_e_max = T.max(cu_k_e_max, 0)
            cu_k_e_max = T.min(cu_k_e_max, seq_len_kv)
            if cu_k_e_max < cu_k_s_min:
                cu_k_e_max = cu_k_s_min

            for bq_i, h_i, d_i in T.Parallel(block_Q, heads, index_dim):
                q_idx = seq_len_i + bq_i
                index_q_shared[bq_i * heads + h_i, d_i] = T.if_then_else(
                    q_idx < seq_len, IndexQ[q_idx * heads + h_i, d_i], 0
                )
            for bq_i, h_i in T.Parallel(block_Q, heads):
                q_idx = seq_len_i + bq_i
                weights[bq_i, h_i] = T.if_then_else(q_idx < seq_len, Weights[q_idx, h_i], 0)

            for nbn_i in T.Pipelined(
                T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages
            ):
                for bn_i, d_i in T.Parallel(block_N, index_dim):
                    k_idx = cu_k_s_min + nbn_i * block_N + bn_i
                    index_k_shared[bn_i, d_i] = T.if_then_else(
                        k_idx >= 0 and k_idx < cu_k_e_max, IndexK[k_idx, d_i], 0
                    )

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (
                        T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]
                    )

                T.reduce_sum(s_reshaped, logits_shared, dim=-1, clear=True)

                # Keep this write deterministic to satisfy data-race verification.
                for bq_i in T.serial(block_Q):
                    q_idx = seq_len_i + bq_i
                    if q_idx < seq_len:
                        for bn_i in T.serial(block_N):
                            k_idx = cu_k_s_min + nbn_i * block_N + bn_i
                            if k_idx >= 0 and k_idx < cu_k_e_max:
                                Logits[q_idx, k_idx] = logits_shared[bn_i, bq_i]

    return tl_indexer_fwd_kernel


@tilelang.jit
def clean_logits_(threads: int = 512, block_K: int = 4096):
    """Build kernel that masks out invalid key ranges in logits."""
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx < seq_len_kv and (idx < cu_k_s or idx >= cu_k_e):
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_logits_kernel


def indexer_fwd_interface(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True):
    """Run indexer forward kernel and optionally clean logits by row bounds."""
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]

    tl_indexer_fwd_kernel = _get_indexer_fwd_kernel(heads=heads, index_dim=index_dim)
    logits = torch.empty([seq_len, seq_len_kv], device=q.device, dtype=torch.float32)
    tl_indexer_fwd_kernel(
        q.view(seq_len * heads, index_dim), kv, logits, weights, cu_seqlen_ks, cu_seqlen_ke
    )

    if clean_logits:
        clean_logits_kernel = _get_clean_logits_kernel()
        clean_logits_kernel(logits, cu_seqlen_ks, cu_seqlen_ke)
    return logits
