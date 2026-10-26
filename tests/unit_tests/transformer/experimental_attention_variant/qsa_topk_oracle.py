# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Test oracle for the QSA indexer top-k: the opt2 selection, kept out of the model code.

The production path (:func:`triton_qsa_select.qsa_select_topk_keys`) must select exactly
the blocks of the opt2 implementation: per chunk of 2048 blocks, a fused Triton kernel
scores the blocks (per head an fp32 IEEE dot, relu, sequential head sum, ``/ sqrt(d)``)
and encodes unique int64 ``(score bits + 1, inverted block id)`` keys, which a running
``torch.topk`` merges. The same arithmetic gives bit-identical scores, and the unique keys
make the ``(score desc, block id asc)`` order exact, so the two selections must match bit
for bit (including exact score ties).
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

ID_BITS = 21  # block-id field width in the int64 sort keys (supports 2M blocks)


@triton.autotune(configs=[triton.Config({}, num_warps=w) for w in [4, 8]], key=['BM', 'BN', 'DI'])
@triton.jit
def qsa_indexer_keys_kernel(
    q,  # [s, NH, DI] indexer queries (post norm/RoPE)
    bk,  # [n_blocks, DI] pooled block keys
    keys,  # [s, cb] int64 output
    m_t,  # [s] number of visible complete blocks per query (exclusive upper block id)
    m_lo,  # [s] first visible block id per query (read only when HAS_LO)
    j0,  # first block id of this chunk
    cb,  # chunk width (number of blocks scored)
    s,
    scale,  # 1 / sqrt(DI)
    max_id_code,  # (1 << IDB) - 1
    NH: tl.constexpr,
    DI: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IDB: tl.constexpr,
    HAS_LO: tl.constexpr,
):
    i_m, i_n = tl.program_id(0), tl.program_id(1)
    o_m = i_m * BM + tl.arange(0, BM)
    o_n = i_n * BN + tl.arange(0, BN)
    o_d = tl.arange(0, DI)
    m_m = o_m < s
    m_n = o_n < cb

    # [DI, BN] block-key tile for this chunk
    b_k = tl.load(bk + (j0 + o_n)[None, :] * DI + o_d[:, None], mask=m_n[None, :], other=0.0).to(
        tl.float32
    )

    b_acc = tl.zeros([BM, BN], dtype=tl.float32)
    for h in tl.static_range(NH):
        b_q = tl.load(
            q + (o_m[:, None] * NH + h) * DI + o_d[None, :], mask=m_m[:, None], other=0.0
        ).to(tl.float32)
        # fp32 IEEE dot: keeps scores reproducible and tie behavior stable
        b_acc += tl.maximum(tl.dot(b_q, b_k, input_precision="ieee"), 0.0)

    score = b_acc * scale
    # relu scores are >= 0, so the fp32 bit pattern is order-isomorphic to the value
    bits = score.to(tl.int32, bitcast=True).to(tl.int64)
    key = ((bits + 1) << IDB) | (max_id_code - (j0 + o_n))[None, :]
    visible = (j0 + o_n)[None, :] < tl.load(m_t + o_m, mask=m_m, other=0)[:, None]
    if HAS_LO:
        # Packed documents share one block-key buffer: a query sees only its own
        # document's blocks, i.e. ids in [m_lo, m_t).
        visible &= (j0 + o_n)[None, :] >= tl.load(m_lo + o_m, mask=m_m, other=0)[:, None]
    key = tl.where(visible & m_n[None, :] & m_m[:, None], key, 0)
    tl.store(keys + o_m[:, None] * cb + o_n[None, :], key, mask=m_m[:, None] & m_n[None, :])


def qsa_indexer_keys(
    q: torch.Tensor,  # [s, NH, DI]
    block_keys: torch.Tensor,  # [n_blocks, DI]
    m_t: torch.Tensor,  # [s]
    j0: int,
    cb: int,
    keys_out: torch.Tensor,  # [s, cb] int64 (reused buffer)
    m_lo: Optional[torch.Tensor] = None,  # [s] first visible block id, default 0
) -> torch.Tensor:
    """Fill ``keys_out`` with int64 (score, block_id) sort keys for blocks [j0, j0+cb).

    Block ``j`` is visible to query ``t`` iff ``m_lo[t] <= j < m_t[t]``. ``m_lo`` lets
    packed documents share one block-key buffer with global block ids; whether it is
    present is a launch-time constant, never a data-dependent one.
    """
    s, NH, DI = q.shape
    BM, BN = 64, 64
    grid = (triton.cdiv(s, BM), triton.cdiv(cb, BN))
    qsa_indexer_keys_kernel[grid](
        q=q,
        bk=block_keys,
        keys=keys_out,
        m_t=m_t,
        m_lo=m_t if m_lo is None else m_lo,
        j0=j0,
        cb=cb,
        s=s,
        scale=DI**-0.5,
        max_id_code=(1 << ID_BITS) - 1,
        NH=NH,
        DI=DI,
        BM=BM,
        BN=BN,
        IDB=ID_BITS,
        HAS_LO=m_lo is not None,
    )
    return keys_out


def oracle_topk(
    q: torch.Tensor,  # [s, b, NH, DI]
    block_keys: torch.Tensor,  # [n_blocks, b, DI]
    m_t: torch.Tensor,  # [s]
    n_blocks: int,
    k: int,
    m_lo: Optional[torch.Tensor] = None,  # [s]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """opt2 ``QSAIndexer._streaming_topk`` on CUDA -> (order ``[b, s, k]``, picked)."""
    s, b = q.shape[0], q.shape[1]
    chunk = max(k, 2048)
    assert n_blocks < (1 << ID_BITS)
    id_code_max = (1 << ID_BITS) - 1
    q_c = [q[:, bi].contiguous() for bi in range(b)]
    bk_c = [block_keys[:, bi].contiguous() for bi in range(b)]
    m_t_c = m_t.contiguous()
    m_lo_c = None if m_lo is None else m_lo.contiguous()
    run_keys = torch.zeros((b, s, k), dtype=torch.int64, device=q.device)
    for j0 in range(0, n_blocks, chunk):
        w = min(j0 + chunk, n_blocks) - j0
        key = torch.empty(b, s, w, dtype=torch.int64, device=q.device)
        for bi in range(b):
            qsa_indexer_keys(q_c[bi], bk_c[bi], m_t_c, j0, w, key[bi], m_lo=m_lo_c)
        run_keys = torch.cat([run_keys, key], dim=-1).topk(k, dim=-1).values
    picked = run_keys > 0
    order = id_code_max - (run_keys & id_code_max)
    return torch.where(picked, order, torch.zeros_like(order)), picked
