# ruff: noqa
# Adapted from:
# https://github.com/tile-ai/tilelang/blob/4956b5835fa554af6c03d4a6289cad44bf310869/
# examples/dsa_sparse_finetune/indexer_bwd.py
from collections import OrderedDict

import tilelang as tl
import tilelang.language as T
import torch

BF16 = T.bfloat16
FP32 = T.float32
INT32 = T.int32
_TILELANG_KERNEL_CACHE_MAX = 64
_tilelang_indexer_bwd_kernel_cache = OrderedDict()

pass_configs = {
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def _cache_put_lru(cache: OrderedDict, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _TILELANG_KERNEL_CACHE_MAX:
        cache.popitem(last=False)


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _canonical_topk(topk: int, block_i: int = 32) -> int:
    return _round_up(_next_power_of_two(topk), block_i)


def _get_indexer_bwd_kernel(heads: int, dim: int, topk: int):
    key = (heads, dim, topk)
    kernel = _tilelang_indexer_bwd_kernel_cache.pop(key, None)
    if kernel is None:
        kernel = tl_indexer_bwd_impl(heads, dim, topk)
    _cache_put_lru(_tilelang_indexer_bwd_kernel_cache, key, kernel)
    return kernel


@tl.jit(pass_configs=pass_configs)
def tl_indexer_bwd_impl(
    heads: int, dim: int, topk: int, block_I: int = 32, num_stages: int = 0, num_threads: int = 128
):
    """Build tilelang backward kernel for sparse indexer."""
    assert num_stages == 0
    assert topk == tl.math.next_power_of_2(topk)
    assert topk % block_I == 0
    assert heads <= 64 and heads % 8 == 0
    seq_len = T.symbolic("seq_len")
    q_seq_len = T.symbolic("q_seq_len")

    dtype: str = BF16
    accum_dtype: str = FP32
    index_q_shape = [q_seq_len, heads, dim]
    weights_shape = [q_seq_len, heads]
    index_k_shape = [seq_len, dim]
    shape_p = [q_seq_len, topk]
    topk_indices_shape = [q_seq_len, topk]

    pad_heads = heads
    if heads < 16:
        pad_heads = 16

    @T.prim_func
    def tl_indexer_bwd_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),
        IndexK: T.Tensor(index_k_shape, dtype),
        Weights: T.Tensor(weights_shape, FP32),
        TopkIndices: T.Tensor(topk_indices_shape, INT32),
        OGrad: T.Tensor(shape_p, FP32),
        dIndexQ: T.Tensor(index_q_shape, dtype),
        dWeights: T.Tensor(weights_shape, FP32),
        dIndexK: T.Tensor(index_k_shape, FP32),
    ):

        with T.Kernel(q_seq_len, threads=num_threads) as (bx):
            index_q_shared = T.alloc_shared([pad_heads, dim], dtype=FP32)
            weights_shared = T.alloc_shared([pad_heads], dtype=FP32)
            index_k_shared = T.alloc_shared([block_I, dim], dtype=FP32)
            indices_shared = T.alloc_shared([block_I], dtype=INT32)
            d_index_q_frag = T.alloc_fragment([pad_heads, dim], dtype=accum_dtype)
            d_weights_frag = T.alloc_fragment([pad_heads], dtype=accum_dtype)
            d_index_k_frag = T.alloc_fragment([block_I, dim], dtype=accum_dtype)
            logits = T.alloc_fragment((block_I, pad_heads), dtype=accum_dtype)
            _logits = T.alloc_shared((block_I, pad_heads), dtype=accum_dtype)
            grad = T.alloc_shared([block_I], dtype=FP32)

            num_blocks = T.ceildiv(topk, block_I)
            for i, j in T.Parallel(pad_heads, dim):
                index_q_shared[i, j] = T.if_then_else(i < heads, IndexQ[bx, i, j], 0)
            for i in T.Parallel(heads):
                weights_shared[i] = Weights[bx, i]

            T.fill(d_index_q_frag, 0)
            T.fill(d_weights_frag, 0)

            # for bi_i in T.Pipelined(num_blocks, num_stages=num_stages):
            for bi_i in T.serial(num_blocks):
                for i in T.Parallel(block_I):
                    if bi_i * block_I + i < topk:
                        indices_shared[i] = TopkIndices[bx, bi_i * block_I + i]
                        grad[i] = OGrad[bx, bi_i * block_I + i]

                T.sync_threads()
                for i, j in T.Parallel(block_I, dim):
                    index_k_shared[i, j] = T.if_then_else(
                        indices_shared[i] > -1 and indices_shared[i] < seq_len,
                        IndexK[indices_shared[i], j],
                        0,
                    )

                T.sync_threads()
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

                d_weights_i = T.alloc_fragment((block_I, pad_heads), accum_dtype)
                for i, j in T.Parallel(block_I, heads):
                    d_weights_i[i, j] = grad[i] * logits[i, j]
                T.reduce_sum(d_weights_i, d_weights_frag, dim=0, clear=False)

                for i, j in T.Parallel(block_I, pad_heads):
                    _logits[i, j] = T.if_then_else(
                        logits[i, j] > 0 and j < heads, grad[i] * weights_shared[j], 0
                    )
                T.sync_threads()
                T.gemm(
                    _logits,
                    index_k_shared,
                    d_index_q_frag,
                    transpose_A=True,
                    transpose_B=False,
                    clear_accum=False,
                )

                T.gemm(
                    _logits,
                    index_q_shared,
                    d_index_k_frag,
                    transpose_A=False,
                    transpose_B=False,
                    clear_accum=True,
                )

                for i, j in T.Parallel(block_I, dim):
                    if indices_shared[i] > -1 and indices_shared[i] < seq_len:
                        T.atomic_add(dIndexK[indices_shared[i], j], d_index_k_frag[i, j])

            T.copy(d_index_q_frag[:heads, :], dIndexQ[bx, :, :])
            T.copy(d_weights_frag[:heads], dWeights[bx, :])

    return tl_indexer_bwd_kernel


def indexer_bwd_interface(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_scores: torch.Tensor,
):
    """Run indexer backward kernel and return gradients for q/w/k."""
    _, head_num, head_dim = index_q.shape
    k_top = int(topk_indices.shape[1])
    assert k_top > 0, "topk must be positive"
    padded_topk = _canonical_topk(k_top)

    if padded_topk != k_top:
        padded_indices = torch.full(
            (topk_indices.size(0), padded_topk),
            -1,
            dtype=topk_indices.dtype,
            device=topk_indices.device,
        )
        padded_indices[:, :k_top].copy_(topk_indices)
        topk_indices = padded_indices

        padded_grad_scores = torch.zeros(
            (grad_scores.size(0), padded_topk), dtype=grad_scores.dtype, device=grad_scores.device
        )
        padded_grad_scores[:, :k_top].copy_(grad_scores)
        grad_scores = padded_grad_scores

    grad_scores = grad_scores.contiguous()
    grad_q = torch.empty_like(index_q)
    grad_w = torch.empty_like(weights, dtype=torch.float32)
    grad_k = torch.zeros_like(index_k, dtype=torch.float32)

    bwd_kernel = _get_indexer_bwd_kernel(head_num, head_dim, padded_topk)
    bwd_kernel(
        index_q.contiguous(),
        index_k.contiguous(),
        weights.squeeze(-1).contiguous(),
        topk_indices.contiguous(),
        grad_scores,
        grad_q,
        grad_w.squeeze(-1),
        grad_k,
    )

    return grad_q, grad_w, grad_k
