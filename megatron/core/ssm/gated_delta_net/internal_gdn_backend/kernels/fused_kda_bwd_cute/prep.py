"""Chunk-parallel inputs of the KDA backward state recurrence (Triton).

For every 64-token chunk of every sequence and head, with G the chunk-local
log2 gate cumsum and G_last its last valid row:

* qg = q 2^G and kg = k 2^(G_last - G) (zero past the sequence end);
* w = Akk (k beta 2^G).

These are the parts of FLA's ``recompute_w_u_fwd`` and ``chunk_kda_bwd_dAv``
that the dS recurrence kernel (``dhu``) needs, without ``u`` or ``dA``; the
recurrence kernel computes dv = Aqk^T dO itself.  Chunks are located on device
from ``chunk_offsets``, so the launch needs only an upper bound on the chunk
count and no host synchronization.
"""

import torch
import triton
import triton.language as tl

_BT = 64
_D = 128


@triton.jit
def _prep_kernel(
    q,
    k,
    gc,
    beta,
    akk,
    cu_seqlens,
    chunk_offsets,
    qg,
    kg,
    w,
    num_sequences,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    if chunk >= tl.load(chunk_offsets + num_sequences):
        return
    # The sequence holding this chunk: the last s with chunk_offsets[s] <= chunk.
    lo = 0
    hi = num_sequences - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tl.load(chunk_offsets + mid) <= chunk:
            lo = mid
        else:
            hi = mid - 1
    bos = tl.load(cu_seqlens + lo)
    length = tl.load(cu_seqlens + lo + 1) - bos
    i_t = chunk - tl.load(chunk_offsets + lo)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < length
    rows = (bos + o_t).to(tl.int64)
    o_c = tl.arange(0, BT)
    p_sq = rows[:, None] * (H * BT) + head * BT + o_c[None, :]
    m_tok = m_t[:, None]
    # One program covers all key channels, so Akk is read once per chunk.
    b_akk = tl.load(akk + p_sq, mask=m_tok, other=0.0)
    b_b = tl.load(beta + rows * H + head, mask=m_t, other=0.0)
    last = (bos + tl.minimum(i_t * BT + BT, length) - 1).to(tl.int64)
    for i_k in tl.static_range(K // BK):
        o_k = i_k * BK + tl.arange(0, BK)
        p_tok = rows[:, None] * (H * K) + head * K + o_k[None, :]
        b_g = tl.load(gc + p_tok, mask=m_tok, other=0.0)
        b_gn = tl.load(gc + last * (H * K) + head * K + o_k)
        b_q = tl.load(q + p_tok, mask=m_tok, other=0.0).to(tl.float32)
        b_k = tl.load(k + p_tok, mask=m_tok, other=0.0).to(tl.float32)
        gamma = tl.exp2(b_g)
        tl.store(qg + p_tok, (b_q * gamma).to(qg.dtype.element_ty), mask=m_tok)
        b_kg = b_k * tl.where(m_tok, tl.exp2(b_gn[None, :] - b_g), 0.0)
        tl.store(kg + p_tok, b_kg.to(kg.dtype.element_ty), mask=m_tok)
        b_kb = (b_k * b_b[:, None] * gamma).to(k.dtype.element_ty)
        b_w = tl.dot(b_akk, b_kb)
        tl.store(w + p_tok, b_w.to(w.dtype.element_ty), mask=m_tok)


def kda_dhu_prep(q, k, gc, beta, akk, cu_seqlens, chunk_offsets, max_chunks, num_warps=4, block_k=64):
    """Returns ``(qg, kg, w)`` (BF16, shaped like ``q``).

    ``max_chunks`` is an upper bound on ``chunk_offsets[-1]``.
    """
    _, _, heads, dim = q.shape
    qg, kg, w = torch.empty_like(q), torch.empty_like(q), torch.empty_like(q)
    _prep_kernel[(max_chunks, heads)](
        q,
        k,
        gc,
        beta,
        akk,
        cu_seqlens,
        chunk_offsets,
        qg,
        kg,
        w,
        cu_seqlens.numel() - 1,
        H=heads,
        K=dim,
        BK=block_k,
        BT=_BT,
        num_warps=num_warps,
    )
    return qg, kg, w
