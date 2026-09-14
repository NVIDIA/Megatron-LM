# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TileLang kernels for the sparse DSA indexer KL target and score gradient."""

import threading
from collections import OrderedDict

import torch

from .tilelang_utils import (
    HAVE_TILELANG,
    T,
    _get_cached_kernel,
    _normalize_sm_scale,
    require_tilelang,
    tilelang,
    tilelang_jit,
)

_target_kernel_cache = OrderedDict()
_kl_kernel_cache = OrderedDict()
_kernel_cache_lock = threading.Lock()

_PASS_CONFIGS = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True} if HAVE_TILELANG else {}


def _get_target_kernel(
    heads: int,
    dim: int,
    topk: int,
    softmax_scale: float,
    block_h: int = 32,
    block_i: int = 64,
    num_stages: int = 2,
    threads: int = 256,
):
    scale = _normalize_sm_scale(softmax_scale)
    key = (heads, dim, topk, scale, block_h, block_i, num_stages, threads)
    return _get_cached_kernel(
        _target_kernel_cache,
        _kernel_cache_lock,
        key,
        lambda: sparse_indexer_target(
            heads=heads,
            dim=dim,
            topk=topk,
            softmax_scale=scale,
            block_h=block_h,
            block_i=block_i,
            num_stages=num_stages,
            threads=threads,
        ),
    )


def _get_kl_kernel(topk: int, block_i: int = 256, threads: int = 256):
    key = (topk, block_i, threads)
    return _get_cached_kernel(
        _kl_kernel_cache,
        _kernel_cache_lock,
        key,
        lambda: sparse_indexer_kl(topk=topk, block_i=block_i, threads=threads),
    )


@tilelang_jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def sparse_indexer_target(  # pragma: no cover
    heads: int,
    dim: int,
    topk: int,
    softmax_scale: float,
    block_h: int = 32,
    block_i: int = 64,
    num_stages: int = 2,
    threads: int = 256,
):
    """Build a kernel that sums selected-key attention probabilities over local heads."""
    require_tilelang()
    assert heads > 0
    assert dim % 16 == 0
    assert topk % block_i == 0

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32
    num_tiles = tilelang.cdiv(topk, block_i)
    num_head_tiles = tilelang.cdiv(heads, block_h)
    scale_log2 = softmax_scale * 1.4426950408889634

    @T.prim_func
    def main(
        Query: T.Tensor([seq_len, heads, dim], dtype),  # type: ignore
        Key: T.Tensor([seq_len_kv, dim], dtype),  # type: ignore
        Indices: T.Tensor([seq_len, topk], index_dtype),  # type: ignore
        Target: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as row:
            query_shared = T.alloc_shared([block_h, dim], dtype)
            key_shared = T.alloc_shared([block_i, dim], dtype)
            scores = T.alloc_fragment([block_h, block_i], accum_dtype)
            probabilities = T.alloc_fragment([block_h, block_i], accum_dtype)
            target_tile = T.alloc_fragment([block_i], accum_dtype)
            valid = T.alloc_fragment([block_i], "bool")
            row_max = T.alloc_fragment([block_h], accum_dtype)
            previous_max = T.alloc_fragment([block_h], accum_dtype)
            tile_max = T.alloc_fragment([block_h], accum_dtype)
            row_sum = T.alloc_fragment([block_h], accum_dtype)
            tile_sum = T.alloc_fragment([block_h], accum_dtype)
            alpha = T.alloc_fragment([block_h], accum_dtype)

            for item in T.Parallel(topk):
                Target[row, item] = 0

            for head_tile in T.serial(num_head_tiles):
                for head, d in T.Parallel(block_h, dim):
                    head_index = head_tile * block_h + head
                    query_shared[head, d] = T.if_then_else(
                        head_index < heads, Query[row, head_index, d], 0
                    )
                T.fill(row_max, -(2**30))
                T.fill(row_sum, 0)

                for tile in T.Pipelined(num_tiles, num_stages=num_stages):
                    for item in T.Parallel(block_i):
                        index = Indices[row, tile * block_i + item]
                        valid[item] = index >= 0 and index < seq_len_kv
                    for item, d in T.Parallel(block_i, dim):
                        index = Indices[row, tile * block_i + item]
                        safe_index = T.max(T.min(index, seq_len_kv - 1), 0)
                        key_shared[item, d] = T.if_then_else(valid[item], Key[safe_index, d], 0)
                    T.gemm(
                        query_shared,
                        key_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for head, item in T.Parallel(block_h, block_i):
                        scores[head, item] = T.if_then_else(
                            valid[item] and head_tile * block_h + head < heads,
                            scores[head, item],
                            -T.infinity(accum_dtype),
                        )
                    T.copy(row_max, previous_max)
                    T.reduce_max(scores, tile_max, dim=1, clear=True)
                    for head in T.Parallel(block_h):
                        row_max[head] = T.max(previous_max[head], tile_max[head])
                        alpha[head] = T.exp2((previous_max[head] - row_max[head]) * scale_log2)
                    for head, item in T.Parallel(block_h, block_i):
                        probabilities[head, item] = T.if_then_else(
                            valid[item] and head_tile * block_h + head < heads,
                            T.exp2((scores[head, item] - row_max[head]) * scale_log2),
                            0,
                        )
                    T.reduce_sum(probabilities, tile_sum, dim=1, clear=True)
                    for head in T.Parallel(block_h):
                        row_sum[head] = row_sum[head] * alpha[head] + tile_sum[head]

                for tile in T.Pipelined(num_tiles, num_stages=num_stages):
                    for item in T.Parallel(block_i):
                        index = Indices[row, tile * block_i + item]
                        valid[item] = index >= 0 and index < seq_len_kv
                    for item, d in T.Parallel(block_i, dim):
                        index = Indices[row, tile * block_i + item]
                        safe_index = T.max(T.min(index, seq_len_kv - 1), 0)
                        key_shared[item, d] = T.if_then_else(valid[item], Key[safe_index, d], 0)
                    T.gemm(
                        query_shared,
                        key_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for head, item in T.Parallel(block_h, block_i):
                        scores[head, item] = T.if_then_else(
                            valid[item] and head_tile * block_h + head < heads,
                            scores[head, item],
                            -T.infinity(accum_dtype),
                        )
                        probabilities[head, item] = T.if_then_else(
                            valid[item]
                            and head_tile * block_h + head < heads
                            and row_sum[head] > 0,
                            T.exp2((scores[head, item] - row_max[head]) * scale_log2)
                            / row_sum[head],
                            0,
                        )
                    T.reduce_sum(probabilities, target_tile, dim=0, clear=True)
                    for item in T.Parallel(block_i):
                        Target[row, tile * block_i + item] += target_tile[item]

    return main


@tilelang_jit(out_idx=[-2, -1], pass_configs=_PASS_CONFIGS)
def sparse_indexer_kl(topk: int, block_i: int = 256, threads: int = 256):  # pragma: no cover
    """Build a kernel that computes sparse KL row sums and gradients for indexer logits."""
    require_tilelang()
    assert topk % block_i == 0

    seq_len = T.dynamic("seq_len")
    accum_dtype = T.float32
    num_tiles = tilelang.cdiv(topk, block_i)
    log2_e = 1.4426950408889634
    ln_2 = 0.6931471805599453
    eps = 1.0e-10

    @T.prim_func
    def main(
        Target: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        IndexLogits: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        ValidMask: T.Tensor([seq_len, topk], "bool"),  # type: ignore
        GradLogits: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        KLRows: T.Tensor([seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as row:
            logits = T.alloc_fragment([1, block_i], accum_dtype)
            target = T.alloc_fragment([1, block_i], accum_dtype)
            probabilities = T.alloc_fragment([1, block_i], accum_dtype)
            kl_terms = T.alloc_fragment([1, block_i], accum_dtype)
            valid = T.alloc_fragment([block_i], "bool")
            row_max = T.alloc_fragment([1], accum_dtype)
            previous_max = T.alloc_fragment([1], accum_dtype)
            tile_max = T.alloc_fragment([1], accum_dtype)
            row_sum = T.alloc_fragment([1], accum_dtype)
            tile_sum = T.alloc_fragment([1], accum_dtype)
            target_sum = T.alloc_fragment([1], accum_dtype)
            target_tile_sum = T.alloc_fragment([1], accum_dtype)
            kl_sum = T.alloc_fragment([1], accum_dtype)
            kl_tile_sum = T.alloc_fragment([1], accum_dtype)

            T.fill(row_max, -(2**30))
            T.fill(row_sum, 0)
            T.fill(target_sum, 0)
            T.fill(kl_sum, 0)

            for tile in T.serial(num_tiles):
                for item in T.Parallel(block_i):
                    valid[item] = ValidMask[row, tile * block_i + item]
                    logits[0, item] = T.if_then_else(
                        valid[item],
                        IndexLogits[row, tile * block_i + item],
                        -T.infinity(accum_dtype),
                    )
                    target[0, item] = T.if_then_else(
                        valid[item], Target[row, tile * block_i + item], 0
                    )
                T.copy(row_max, previous_max)
                T.reduce_max(logits, tile_max, dim=1, clear=True)
                row_max[0] = T.max(previous_max[0], tile_max[0])
                for item in T.Parallel(block_i):
                    probabilities[0, item] = T.if_then_else(
                        valid[item], T.exp2((logits[0, item] - row_max[0]) * log2_e), 0
                    )
                T.reduce_sum(probabilities, tile_sum, dim=1, clear=True)
                row_sum[0] = (
                    row_sum[0] * T.exp2((previous_max[0] - row_max[0]) * log2_e) + tile_sum[0]
                )
                T.reduce_sum(target, target_tile_sum, dim=1, clear=True)
                target_sum[0] += target_tile_sum[0]

            for tile in T.serial(num_tiles):
                for item in T.Parallel(block_i):
                    valid[item] = ValidMask[row, tile * block_i + item]
                    logits[0, item] = T.if_then_else(
                        valid[item],
                        IndexLogits[row, tile * block_i + item],
                        -T.infinity(accum_dtype),
                    )
                    target[0, item] = T.if_then_else(
                        valid[item] and target_sum[0] > 0,
                        Target[row, tile * block_i + item] / target_sum[0],
                        0,
                    )
                    probabilities[0, item] = T.if_then_else(
                        valid[item] and row_sum[0] > 0,
                        T.exp2((logits[0, item] - row_max[0]) * log2_e) / row_sum[0],
                        0,
                    )
                    GradLogits[row, tile * block_i + item] = T.if_then_else(
                        valid[item], probabilities[0, item] - target[0, item], 0
                    )
                    kl_terms[0, item] = T.if_then_else(
                        valid[item] and target[0, item] > 0,
                        target[0, item]
                        * (
                            T.log2(T.max(target[0, item], eps)) * ln_2
                            - (logits[0, item] - row_max[0])
                            + T.log2(T.max(row_sum[0], eps)) * ln_2
                        ),
                        0,
                    )
                T.reduce_sum(kl_terms, kl_tile_sum, dim=1, clear=True)
                kl_sum[0] += kl_tile_sum[0]

            KLRows[row] = kl_sum[0]

    return main


def sparse_indexer_target_interface(
    query: torch.Tensor, key: torch.Tensor, topk_indices: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """Compute the local-head sparse attention target on selected top-k keys."""
    require_tilelang()
    seq_len, heads, dim = query.shape
    topk = topk_indices.size(1)
    kernel = _get_target_kernel(heads, dim, topk, softmax_scale)
    return kernel(query, key, topk_indices)


def sparse_indexer_kl_interface(
    target: torch.Tensor, index_logits: torch.Tensor, valid_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute unscaled indexer KL sum and its exact gradient with respect to logits."""
    require_tilelang()
    kernel = _get_kl_kernel(valid_mask.size(1))
    grad_logits, kl_rows = kernel(target, index_logits, valid_mask)
    return kl_rows.sum(), grad_logits


class SparseIndexerKLLoss(torch.autograd.Function):  # pragma: no cover
    """Autograd bridge from fused sparse KL score gradients to the TileLang indexer."""

    @staticmethod
    def forward(ctx, target, index_logits, valid_mask):
        """Compute the sparse indexer KL loss and save its logits gradient."""
        kl_sum, grad_logits = sparse_indexer_kl_interface(target, index_logits, valid_mask)
        ctx.save_for_backward(grad_logits)
        return kl_sum

    @staticmethod
    def backward(ctx, grad_output):
        """Scale the saved index-logits gradient for the backward pass."""
        (grad_logits,) = ctx.saved_tensors
        return None, grad_logits * grad_output, None


if not HAVE_TILELANG:
    SparseIndexerKLLoss = None
