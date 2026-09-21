# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Portions adapted from HuggingFace transformers (Apache-2.0),
# src/transformers/models/qwen4_exp/modeling_qwen4_exp.py @ 99e19a9a
# (Qwen4ExpTextQSAIndexer, Qwen4ExpTextAttention, Qwen4ExpTextRMSNorm,
# apply_rotary_pos_emb). Copyright 2026 The Qwen team, Alibaba Group and
# the HuggingFace Inc. team.

"""Zero-dependency reference oracle for QSA (Qwen Sparse Attention).

Pure-torch, no Megatron imports. Two indexer implementations are provided and
tested against each other:

* :func:`reference_indexer_selected_sets` — a literal port of the HF O(B*S)
  python-loop indexer (visible-prefix blocking, driven by an explicit
  attention mask), kept as the semantic ground truth.
* :func:`batched_indexer_selected_mask` — the batched causal-prefix-blocking
  formulation used by the Megatron implementation (fixed 4-token grid from
  the sequence/document start, segmented at ``cu_seqlens`` for THD).

Their equivalence on unpadded causal inputs follows from the causal mask making
the visible prefix of token ``t`` exactly ``[0, t]``, so the explicit mask and the
fixed grid cut the same blocks; ``test_qsa_reference.py`` pins it for both BSHD
and THD.

One deliberate deviation from the HF code (QSA semantics decision D4): both
implementations use a *deterministic* top-k with ``(score desc, block_id asc)``
lexicographic tie-breaking instead of ``torch.topk``. ReLU scores tie at zero
en masse, and nondeterministic ties break recompute route-consistency and
cross-TP-rank agreement. HF-faithful caveat kept on purpose: when
``(t - seg_start + 1) % ratio == 0`` the tail is empty and the block containing
the current token must *win* top-k to be attended — it is not force-kept.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Set

import torch

# ---------------------------------------------------------------------------
# RoPE / RMSNorm primitives (HF-faithful)
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first ``cos.shape[-1]`` dims of ``t``; pass the rest through.

    ``cos``/``sin`` must already be broadcastable to ``t``'s leading dims.
    """
    rotary_dim = cos.shape[-1]
    t_rope, t_nope = t[..., :rotary_dim], t[..., rotary_dim:]
    t_rope = (t_rope * cos) + (rotate_half(t_rope) * sin)
    return torch.cat([t_rope, t_nope], dim=-1)


def build_rope_cos_sin(
    positions: torch.Tensor, rotary_dim: int, rope_theta: float, dtype: torch.dtype
):
    """cos/sin of shape ``positions.shape + (rotary_dim,)``, fp32-computed.

    Text-only layout: for identical T/H/W position grids the HF mrope
    recomposition is the identity, so this reduces to
    ``cat(freqs, freqs, dim=-1)`` over ``rotary_dim // 2`` inverse frequencies.
    """
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=positions.device)
            / rotary_dim
        )
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq  # [..., rotary_dim // 2]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Zero-centered-gamma RMSNorm: ``normalize(x) * (1 + w)``, fp32 internals."""
    out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out.type_as(x)


# ---------------------------------------------------------------------------
# Deterministic top-k (QSA semantics, D4)
# ---------------------------------------------------------------------------


def deterministic_topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Top-``k`` indices of a 1-D score vector under ``(score desc, index asc)``.

    A stable descending sort preserves ascending original order among equal
    scores, which is exactly the lexicographic tie-break.
    """
    assert scores.dim() == 1
    k = min(k, scores.numel())
    _, order = torch.sort(scores, descending=True, stable=True)
    return order[:k]


def deterministic_topk_indices_batched(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Row-wise deterministic top-k over the last axis. Returns ``[..., k]``."""
    k = min(k, scores.shape[-1])
    _, order = torch.sort(scores, dim=-1, descending=True, stable=True)
    return order[..., :k]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class QSAIndexerParams:
    """``index_qk_proj: hidden -> (n_heads + 1) * head_dim``, q/k RMSNorm."""

    index_qk_proj_weight: torch.Tensor  # [(n_heads + 1) * head_dim, hidden]
    q_norm_weight: torch.Tensor  # [head_dim]
    k_norm_weight: torch.Tensor  # [head_dim]
    n_heads: int = 4
    head_dim: int = 128
    token_budget: int = 2048
    compress_ratio: int = 4
    rotary_dim: int = 64
    rope_theta: float = 1e7
    rms_norm_eps: float = 1e-6

    @property
    def block_topk(self) -> int:
        return self.token_budget // self.compress_ratio

    @staticmethod
    def init_random(hidden_size: int = 2560, dtype=torch.float32, device="cpu", seed: int = 0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        w = torch.randn((4 + 1) * 128, hidden_size, generator=g) * hidden_size**-0.5
        return QSAIndexerParams(
            index_qk_proj_weight=w.to(device=device, dtype=dtype),
            q_norm_weight=torch.zeros(128, device=device, dtype=dtype),
            k_norm_weight=torch.zeros(128, device=device, dtype=dtype),
        )


@dataclass
class QSAAttentionParams:
    """Main-attention weights (GQA + interleaved output gate + qk-norm)."""

    q_proj_weight: torch.Tensor  # [n_heads * head_dim * 2, hidden] (query|gate per head)
    k_proj_weight: torch.Tensor  # [n_kv_heads * head_dim, hidden]
    v_proj_weight: torch.Tensor  # [n_kv_heads * head_dim, hidden]
    o_proj_weight: torch.Tensor  # [hidden, n_heads * head_dim]
    q_norm_weight: torch.Tensor  # [head_dim]
    k_norm_weight: torch.Tensor  # [head_dim]
    n_heads: int = 24
    n_kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64  # head_dim * partial_rotary_factor(0.25)
    rope_theta: float = 1e7
    rms_norm_eps: float = 1e-6

    @staticmethod
    def init_random(
        hidden_size: int = 2560,
        n_heads: int = 24,
        n_kv_heads: int = 2,
        head_dim: int = 256,
        dtype=torch.float32,
        device="cpu",
        seed: int = 0,
    ):
        g = torch.Generator(device="cpu").manual_seed(seed)

        def w(rows):
            return (torch.randn(rows, hidden_size, generator=g) * hidden_size**-0.5).to(
                device=device, dtype=dtype
            )

        o = (
            torch.randn(hidden_size, n_heads * head_dim, generator=g) * (n_heads * head_dim) ** -0.5
        ).to(device=device, dtype=dtype)
        return QSAAttentionParams(
            q_proj_weight=w(n_heads * head_dim * 2),
            k_proj_weight=w(n_kv_heads * head_dim),
            v_proj_weight=w(n_kv_heads * head_dim),
            o_proj_weight=o,
            q_norm_weight=torch.zeros(head_dim, device=device, dtype=dtype),
            k_norm_weight=torch.zeros(head_dim, device=device, dtype=dtype),
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )


# ---------------------------------------------------------------------------
# Shared indexer front-end: projections, norms, RoPE
# ---------------------------------------------------------------------------


def _indexer_q_and_rawk(hidden: torch.Tensor, p: QSAIndexerParams, positions: torch.Tensor):
    """hidden ``[..., S, H]`` -> (q ``[..., S, n_heads, head_dim]`` RoPE'd,
    raw_keys ``[..., S, head_dim]`` un-normalized, un-rotated)."""
    qk = hidden @ p.index_qk_proj_weight.t()
    q, token_k = torch.split(qk, [p.n_heads * p.head_dim, p.head_dim], dim=-1)
    q = q.reshape(*q.shape[:-1], p.n_heads, p.head_dim)
    q = rms_norm(q, p.q_norm_weight, p.rms_norm_eps)
    cos, sin = build_rope_cos_sin(positions, p.rotary_dim, p.rope_theta, hidden.dtype)
    q = apply_partial_rope(q, cos.unsqueeze(-2), sin.unsqueeze(-2))
    return q, token_k


# ---------------------------------------------------------------------------
# Literal HF port (visible-prefix blocking, python loops) — ground truth
# ---------------------------------------------------------------------------


def reference_indexer_selected_sets(
    hidden: torch.Tensor,  # [B, S, H]
    params: QSAIndexerParams,
    positions: torch.Tensor,  # [B, S] int64 (per-document under packing)
    visible: torch.Tensor,  # [B, S, S] bool, visible[b, t, j] = query t sees token j
    use_deterministic_topk: bool = True,
) -> List[List[Set[int]]]:
    """Per-(batch, query) selected-token sets, HF loop semantics verbatim."""
    B, S, _ = hidden.shape
    p = params
    q_all, raw_keys_all = _indexer_q_and_rawk(hidden, p, positions)

    out: List[List[Set[int]]] = []
    for b in range(B):
        raw_keys = raw_keys_all[b]  # [S, head_dim]
        rows: List[Set[int]] = []
        for t in range(S):
            local_visible = torch.nonzero(visible[b, t], as_tuple=False).flatten()
            m = local_visible.shape[0] // p.compress_ratio
            if m > 0:
                block_tokens = local_visible[: m * p.compress_ratio].view(m, p.compress_ratio)
                key_groups = raw_keys.index_select(0, block_tokens.flatten()).view(
                    m, p.compress_ratio, p.head_dim
                )
                pooled = key_groups.float().mean(dim=1).to(raw_keys.dtype)
                pooled = rms_norm(pooled, p.k_norm_weight, p.rms_norm_eps)
                group_start_pos = positions[b].index_select(0, block_tokens[:, 0])
                cos, sin = build_rope_cos_sin(
                    group_start_pos, p.rotary_dim, p.rope_theta, hidden.dtype
                )
                block_keys = apply_partial_rope(pooled, cos, sin)  # [m, head_dim]

                # scores: relu(q_h · k_block) summed over heads, fp32
                scores = q_all[b, t].float() @ block_keys.float().t()  # [n_heads, m]
                scores = torch.relu(scores).sum(dim=0) / math.sqrt(p.head_dim)  # [m]

                if use_deterministic_topk:
                    sel_blocks = deterministic_topk_indices(scores, p.block_topk)
                else:
                    sel_blocks = scores.topk(min(p.block_topk, m), dim=0).indices
                selected = block_tokens.index_select(0, sel_blocks).flatten()
            else:
                selected = torch.empty(0, dtype=torch.long, device=hidden.device)
            tail = local_visible[m * p.compress_ratio :]
            rows.append(set(torch.cat([selected, tail]).tolist()))
        out.append(rows)
    return out


# ---------------------------------------------------------------------------
# Batched causal-prefix-blocking formulation (training form)
# ---------------------------------------------------------------------------


def batched_indexer_selected_mask(
    hidden: torch.Tensor,  # [B, S, H] (BSHD) or [T, H] (THD with cu_seqlens)
    params: QSAIndexerParams,
    cu_seqlens: Optional[torch.Tensor] = None,  # [n_docs + 1] int, THD only
) -> torch.Tensor:
    """Bool selection mask ``[B, S, S]`` (or ``[T, T]`` for THD).

    Fixed 4-token grid per document; pooled keys / norms / RoPE computed once
    per document; block j visible to query t iff entirely inside the causal
    prefix; deterministic top-k per query; per-query ragged tail unconditional.
    Positions restart at every ``cu_seqlens`` boundary; blocks never cross
    documents. Only causal, unpadded inputs are supported (see the
    equivalence proof for why that is the exact HF semantics).
    """
    p = params
    if hidden.dim() == 2:
        assert cu_seqlens is not None, "THD input requires cu_seqlens"
        T = hidden.shape[0]
        assert int(cu_seqlens[-1]) == T
        mask = torch.zeros(T, T, dtype=torch.bool, device=hidden.device)
        for d in range(cu_seqlens.numel() - 1):
            s, e = int(cu_seqlens[d]), int(cu_seqlens[d + 1])
            mask[s:e, s:e] = _batched_indexer_one_doc(hidden[s:e], p)
        return mask.unsqueeze(0)

    assert cu_seqlens is None, "BSHD input must not pass cu_seqlens"
    B, S, _ = hidden.shape
    return torch.stack([_batched_indexer_one_doc(hidden[b], p) for b in range(B)], dim=0)


def _batched_indexer_one_doc(hidden: torch.Tensor, p: QSAIndexerParams) -> torch.Tensor:
    """One contiguous causal document ``[S, H]`` -> bool mask ``[S, S]``."""
    S = hidden.shape[0]
    device = hidden.device
    positions = torch.arange(S, device=device)
    q, raw_keys = _indexer_q_and_rawk(hidden, p, positions)  # [S, nh, hd], [S, hd]

    r = p.compress_ratio
    n_blocks = S // r
    t = torch.arange(S, device=device)
    m_t = (t + 1) // r  # complete blocks visible to query t

    if n_blocks > 0:
        pooled = raw_keys[: n_blocks * r].view(n_blocks, r, p.head_dim).float().mean(dim=1)
        pooled = rms_norm(pooled.to(raw_keys.dtype), p.k_norm_weight, p.rms_norm_eps)
        block_start_pos = positions[: n_blocks * r : r]
        cos, sin = build_rope_cos_sin(block_start_pos, p.rotary_dim, p.rope_theta, hidden.dtype)
        block_keys = apply_partial_rope(pooled, cos, sin)  # [n_blocks, hd]

        # [S, nh, hd] x [n_blocks, hd] -> [S, nh, n_blocks] -> relu-sum-heads fp32
        scores = torch.einsum("shd,bd->shb", q.float(), block_keys.float())
        scores = torch.relu(scores).sum(dim=1) / math.sqrt(p.head_dim)  # [S, n_blocks]

        block_ids = torch.arange(n_blocks, device=device)
        blk_visible = block_ids.unsqueeze(0) < m_t.unsqueeze(1)  # [S, n_blocks]
        scores = scores.masked_fill(~blk_visible, float("-inf"))
        order = deterministic_topk_indices_batched(scores, p.block_topk)  # [S, k]
        # keep only selections that are actually visible (score > -inf)
        picked = torch.gather(scores, 1, order) > float("-inf")

        mask = torch.zeros(S, S, dtype=torch.bool, device=device)
        # expand selected block ids to their 4 tokens
        tok = order.unsqueeze(-1) * r + torch.arange(r, device=device)  # [S, k, r]
        tok = tok.reshape(S, -1)
        keep = picked.unsqueeze(-1).expand(-1, -1, r).reshape(S, -1)
        rows = t.unsqueeze(1).expand_as(tok)
        mask[rows[keep], tok[keep]] = True
    else:
        mask = torch.zeros(S, S, dtype=torch.bool, device=device)

    # unconditional ragged tail: [m_t * r, t]
    j = torch.arange(S, device=device)
    tail = (j.unsqueeze(0) >= (m_t * r).unsqueeze(1)) & (j.unsqueeze(0) <= t.unsqueeze(1))
    return mask | tail


# ---------------------------------------------------------------------------
# Main attention (GQA + interleaved output gate + partial RoPE + qk-norm)
# ---------------------------------------------------------------------------


def reference_attention_forward(
    hidden: torch.Tensor,  # [B, S, H]
    params: QSAAttentionParams,
    selection_mask: torch.Tensor,  # [B, S, S] bool (already includes causality)
    positions: Optional[torch.Tensor] = None,  # [B, S]; default arange
) -> torch.Tensor:
    """Eager HF-semantics forward. Differentiable; use for fwd/bwd parity."""
    B, S, _ = hidden.shape
    p = params
    if positions is None:
        positions = torch.arange(S, device=hidden.device).expand(B, S)

    # q_proj emits query and gate chunked per head: view(..., h, d*2) -> chunk(2, -1)
    q_full = hidden @ p.q_proj_weight.t()  # [B, S, n_heads * head_dim * 2]
    q, gate = torch.chunk(q_full.view(B, S, p.n_heads, p.head_dim * 2), 2, dim=-1)
    q = rms_norm(q, p.q_norm_weight, p.rms_norm_eps)  # per-head, before RoPE
    k = (hidden @ p.k_proj_weight.t()).view(B, S, p.n_kv_heads, p.head_dim)
    k = rms_norm(k, p.k_norm_weight, p.rms_norm_eps)
    v = (hidden @ p.v_proj_weight.t()).view(B, S, p.n_kv_heads, p.head_dim)

    cos, sin = build_rope_cos_sin(positions, p.rotary_dim, p.rope_theta, hidden.dtype)
    q = apply_partial_rope(q, cos.unsqueeze(-2), sin.unsqueeze(-2))
    k = apply_partial_rope(k, cos.unsqueeze(-2), sin.unsqueeze(-2))

    # GQA: repeat kv heads
    rep = p.n_heads // p.n_kv_heads
    k = k.repeat_interleave(rep, dim=2)
    v = v.repeat_interleave(rep, dim=2)

    scaling = p.head_dim**-0.5
    attn = torch.einsum("bqhd,bkhd->bhqk", q, k) * scaling
    attn = attn.masked_fill(~selection_mask.unsqueeze(1), torch.finfo(attn.dtype).min)
    attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
    out = torch.einsum("bhqk,bkhd->bqhd", attn, v)  # [B, S, n_heads, head_dim]

    out = out.reshape(B, S, -1) * torch.sigmoid(gate.reshape(B, S, -1))
    return out @ p.o_proj_weight.t()


def build_causal_visible(S: int, device="cpu") -> torch.Tensor:
    """[S, S] causal bool mask for one document."""
    return torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))


def build_thd_visible(cu_seqlens: torch.Tensor, device="cpu") -> torch.Tensor:
    """[T, T] per-document causal bool mask for a packed batch."""
    T = int(cu_seqlens[-1])
    mask = torch.zeros(T, T, dtype=torch.bool, device=device)
    for d in range(cu_seqlens.numel() - 1):
        s, e = int(cu_seqlens[d]), int(cu_seqlens[d + 1])
        mask[s:e, s:e] = build_causal_visible(e - s, device)
    return mask


def build_thd_positions(cu_seqlens: torch.Tensor, device="cpu") -> torch.Tensor:
    """[T] per-document position ids restarting at each cu_seqlens boundary."""
    parts = [
        torch.arange(int(cu_seqlens[d + 1]) - int(cu_seqlens[d]), device=device)
        for d in range(cu_seqlens.numel() - 1)
    ]
    return torch.cat(parts)
