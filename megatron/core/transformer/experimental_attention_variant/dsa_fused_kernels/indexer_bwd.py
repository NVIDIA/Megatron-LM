# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Referenced from https://github.com/tile-ai/tilelang/blob/main/examples

from typing import Optional

import torch
import torch.nn.functional as F

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
            logits = torch.einsum("shk,tk->sht", q, k) * softmax_scale
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


def indexer_bwd_interface(
    q, weights, k, attn_score, index_score, topk_indices, offsets, use_unfused=False
):
    """Indexer backward. THD format."""
    if use_unfused or not HAS_TILELANG:
        return _ref_indexer_bwd(q, weights, k, topk_indices, attn_score, offsets)[1:]

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
