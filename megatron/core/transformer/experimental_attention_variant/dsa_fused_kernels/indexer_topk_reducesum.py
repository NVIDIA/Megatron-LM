import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import einsum as einops_einsum

from .utils import prepare_token_indices

try:
    import tilelang
    import tilelang.language as T

    HAS_TILELANG = True
except ImportError:
    HAS_TILELANG = False

if HAS_TILELANG:

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


def indexer_topk_reducesum_interface(q, weights, k, topk, offsets, use_unfused=False):
    """Indexer topk + softmax. THD format: q [S,H,D], weights [S,H], k [S,D]."""
    if use_unfused or not HAS_TILELANG:
        return _ref_indexer_topk_reducesum(q, weights, k, topk, offsets)

    _, heads, dim = q.shape
    token_indices = prepare_token_indices(offsets)
    seq_len = q.shape[0]
    kernel = _tl_indexer_topk_reducesum(heads=heads, dim=dim, topk=topk, dtype="bfloat16")
    topk_indices = torch.zeros((seq_len, topk), device=q.device, dtype=torch.int32)
    topk_score = torch.zeros((seq_len, topk), device=q.device, dtype=torch.float32)
    kernel(q, weights, k, topk_indices, topk_score, offsets, token_indices)
    return topk_indices, topk_score
