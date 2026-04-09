import torch

from .utils import prepare_token_indices

try:
    import tilelang
    import tilelang.language as T

    HAS_TILELANG = True
except ImportError:
    HAS_TILELANG = False

if HAS_TILELANG:

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


def sparse_mla_fwd_interface(q, kv, indices, offsets, sm_scale, d_v, use_unfused=False):
    """Sparse MLA forward (THD format)."""
    if use_unfused or not HAS_TILELANG:
        return _ref_sparse_mla_fwd(q, kv, indices, offsets, sm_scale, d_v)

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
