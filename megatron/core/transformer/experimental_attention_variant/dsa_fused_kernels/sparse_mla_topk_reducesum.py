# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Referenced from https://github.com/tile-ai/tilelang/blob/main/examples

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


def sparse_mla_topk_reducesum_interface(
    q, kv, topk_indices, lse, offsets, dim_v, sm_scale, use_unfused=False
):
    """Sparse MLA topk reducesum. THD format."""
    if use_unfused or not HAS_TILELANG:
        topk_indices_2d = topk_indices.squeeze(-2) if topk_indices.ndim == 3 else topk_indices
        attn_score = _ref_sparse_mla_topk_reducesum(q, kv, topk_indices_2d, offsets, sm_scale)
        return attn_score.unsqueeze(-2)

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
        heads=heads, dim=dim_v, tail_dim=tail_dim, topk=topk, kv_group=kv_group, sm_scale=sm_scale
    )
    kernel(q, kv, topk_indices, lse, offsets, token_indices, reducesum)
    reducesum = reducesum.sum(dim=-2)
    attn_score = reducesum / reducesum.sum(dim=-1, keepdim=True)
    return attn_score
