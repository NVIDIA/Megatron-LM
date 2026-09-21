# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pure-PyTorch reference for MiniMax Sparse Attention (MSA), MiniMax-M3 text model.

Non-product code (P0 golden). Mirrors the semantics of Hugging Face
``transformers.models.minimax_m3_vl.modeling_minimax_m3_vl`` (transformers 5.16.1):

* Gemma-style RMSNorm: normalize in high precision, scale by ``(1 + weight)``.
* Per-head QK-norm applied *before* partial RoPE.
* Partial RoPE: NeoX / ``rotate_half`` layout on the first ``rotary_dim`` dims
  (``inv_freq`` computed with ``dim = rotary_dim``), remaining dims pass through.
* Indexer: index-Q ``[n_idx_heads x idx_dim]`` and a single shared index-K
  ``[1 x idx_dim]``, each Gemma-normed and partially roped like the main branch;
  token scores ``iQ . iK`` (no ``1/sqrt(d)`` scale); causal ``-inf`` at token
  level; block max-pool; the query's own block is forced in (``local_blocks``);
  ``topk`` block ids left-packed with ``-1`` right padding.
* Sparse attention is realised as dense causal attention restricted to the
  selected key blocks (softmax normalised once over all visible tokens).

Every function is dtype-agnostic so the module can run in float64 for
``gradcheck``-grade comparisons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Elementary pieces
# --------------------------------------------------------------------------- #
def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with ``(1 + weight)`` scaling, computed in the widest of fp32 / input dtype."""
    compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    xf = x.to(compute_dtype)
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + weight.to(compute_dtype))
    return out.to(x.dtype)


def rope_cos_sin(
    position_ids: torch.Tensor, rotary_dim: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of shape ``[B, S, rotary_dim]`` (HF ``cat((freqs, freqs))`` layout)."""
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=compute_dtype, device=position_ids.device) / rotary_dim)
    )
    freqs = position_ids.to(compute_dtype)[..., None] * inv_freq  # [B, S, rotary_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x``: ``[B, H, S, D]``; ``cos``/``sin``: ``[B, S, rotary_dim]``. Rotates ``x[..., :rotary_dim]``."""
    rotary_dim = cos.shape[-1]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([x_rot, x_pass], dim=-1)


# --------------------------------------------------------------------------- #
# Indexer
# --------------------------------------------------------------------------- #
def msa_block_scores(
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    position_ids: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Token scores ``iQ . iK^T`` -> causal mask -> block max-pool.

    ``idx_q``: ``[B, H_idx, S_q, D]``; ``idx_k``: ``[B, 1, S_k, D]``;
    ``position_ids``: ``[B, S_q]`` (absolute key slot of each query).
    Returns ``[B, H_idx, S_q, N_kv]`` with ``-inf`` for blocks entirely in the future.
    """
    B, H, S_q, _ = idx_q.shape
    S_k = idx_k.shape[2]
    n_kv_blocks = -(-S_k // block_size)
    pad = n_kv_blocks * block_size - S_k
    scores = torch.matmul(idx_q, idx_k.transpose(-1, -2))  # [B, H, S_q, S_k]  (no 1/sqrt(d), as HF)
    k_pos = torch.arange(S_k, device=idx_q.device)
    future = k_pos[None, None, None, :] > position_ids[:, None, :, None]
    scores = scores.masked_fill(future, float("-inf"))
    if pad:
        scores = F.pad(scores, (0, pad), value=float("-inf"))
    scores = scores.view(B, H, S_q, n_kv_blocks, block_size)
    return scores.amax(dim=-1)


def msa_select_blocks(
    block_scores: torch.Tensor,
    position_ids: torch.Tensor,
    block_size: int,
    topk_blocks: int,
    local_blocks: int = 1,
) -> torch.Tensor:
    """Top-k block ids per (batch, index head, query): ``[B, H_idx, S_q, min(topk, N_kv)]``.

    Blocks ``q_block - local_blocks + 1 .. q_block`` are forced in (score ``+inf``).
    Slots whose score is ``-inf`` (future / empty) are reported as ``-1``; because
    ``-inf`` sorts last, valid ids are left-packed.
    """
    B, H, S_q, n_kv_blocks = block_scores.shape
    scores = block_scores.clone()
    q_block = position_ids // block_size  # [B, S_q]
    if local_blocks > 0:
        local = torch.arange(local_blocks, device=scores.device)
        local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0)  # [B, S_q, local]
        local_idx = local_idx.unsqueeze(1).expand(-1, H, -1, -1)
        scores.scatter_(-1, local_idx, float("inf"))
    k = min(topk_blocks, n_kv_blocks)
    top_scores, top_idx = scores.topk(k, dim=-1)
    return top_idx.masked_fill(top_scores == float("-inf"), -1)


def block_indices_to_mask(
    block_indices: torch.Tensor,
    position_ids: torch.Tensor,
    key_length: int,
    block_size: int,
    num_attention_heads: int,
) -> torch.Tensor:
    """Expand block ids into a boolean keep-mask ``[B, H_q, S_q, S_k]`` (selected block AND causal).

    Index head ``g`` serves query heads ``g * (H_q / H_idx) .. (g + 1) * (H_q / H_idx) - 1``
    (``repeat_interleave`` on the head axis, matching HF ``repeat_kv``).
    """
    B, H_idx, S_q, _ = block_indices.shape
    n_kv_blocks = -(-key_length // block_size)
    safe = block_indices.masked_fill(block_indices < 0, n_kv_blocks)
    keep_blk = torch.zeros(B, H_idx, S_q, n_kv_blocks + 1, dtype=torch.bool, device=block_indices.device)
    keep_blk.scatter_(-1, safe.long(), True)
    keep_blk = keep_blk[..., :n_kv_blocks]
    keep = keep_blk.repeat_interleave(block_size, dim=-1)[..., :key_length]
    keep = keep.repeat_interleave(num_attention_heads // H_idx, dim=1)
    k_pos = torch.arange(key_length, device=block_indices.device)
    future = k_pos[None, None, None, :] > position_ids[:, None, :, None]
    return keep & ~future


def causal_mask(position_ids: torch.Tensor, key_length: int) -> torch.Tensor:
    """Plain causal keep-mask ``[B, 1, S_q, S_k]`` (used by the dense baseline)."""
    k_pos = torch.arange(key_length, device=position_ids.device)
    return ~(k_pos[None, None, None, :] > position_ids[:, None, :, None])


def masked_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keep: torch.Tensor, scale: float
) -> torch.Tensor:
    """Dense GQA attention with a boolean keep-mask. ``q``: ``[B, H_q, S, D]``; ``k``/``v``: ``[B, H_kv, S_k, D]``."""
    n_rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~keep, float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float64 if q.dtype == torch.float64 else torch.float32)
    return torch.matmul(probs.to(q.dtype), v)


# --------------------------------------------------------------------------- #
# Full sparse-attention layer (mirror of HF ``MiniMaxM3VLAttention`` + ``MiniMaxM3VLIndexer``)
# --------------------------------------------------------------------------- #
@dataclass
class MSARefConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int = 128
    rotary_dim: int = 64
    rope_theta: float = 5.0e6
    rms_norm_eps: float = 1.0e-6
    index_n_heads: int = 4
    index_head_dim: int = 128
    index_block_size: int = 128
    index_topk_blocks: int = 16
    index_local_blocks: int = 1
    sparse: bool = True  # False -> plain full causal attention (dense layers L0-L2)


class MSARefAttention(nn.Module):
    """Reference attention layer. Parameter names follow the HF *module* names."""

    def __init__(self, cfg: MSARefConfig, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.cfg = cfg
        H, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        kw = dict(bias=False, dtype=dtype)
        self.q_proj = nn.Linear(cfg.hidden_size, H * D, **kw)
        self.k_proj = nn.Linear(cfg.hidden_size, Hkv * D, **kw)
        self.v_proj = nn.Linear(cfg.hidden_size, Hkv * D, **kw)
        self.o_proj = nn.Linear(H * D, cfg.hidden_size, **kw)
        self.q_norm = nn.Parameter(torch.zeros(D, dtype=dtype))
        self.k_norm = nn.Parameter(torch.zeros(D, dtype=dtype))
        if cfg.sparse:
            Hi, Di = cfg.index_n_heads, cfg.index_head_dim
            self.index_q_proj = nn.Linear(cfg.hidden_size, Hi * Di, **kw)
            self.index_k_proj = nn.Linear(cfg.hidden_size, Di, **kw)
            self.index_q_norm = nn.Parameter(torch.zeros(Di, dtype=dtype))
            self.index_k_norm = nn.Parameter(torch.zeros(Di, dtype=dtype))
        self.scaling = D**-0.5

    # -- indexer -----------------------------------------------------------
    def indexer(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor):
        cfg = self.cfg
        B, S, _ = hidden.shape
        iq = self.index_q_proj(hidden).view(B, S, cfg.index_n_heads, cfg.index_head_dim)
        iq = gemma_rmsnorm(iq, self.index_q_norm, cfg.rms_norm_eps).transpose(1, 2)
        ik = self.index_k_proj(hidden).view(B, S, 1, cfg.index_head_dim)
        ik = gemma_rmsnorm(ik, self.index_k_norm, cfg.rms_norm_eps).transpose(1, 2)
        # HF slices cos[..., :index_head_dim]; with rotary_dim <= index_head_dim that is the full cos.
        iq = apply_partial_rope(iq, cos[..., : cfg.index_head_dim], sin[..., : cfg.index_head_dim])
        ik = apply_partial_rope(ik, cos[..., : cfg.index_head_dim], sin[..., : cfg.index_head_dim])
        block_scores = msa_block_scores(iq, ik, position_ids, cfg.index_block_size)
        return msa_select_blocks(
            block_scores, position_ids, cfg.index_block_size, cfg.index_topk_blocks, cfg.index_local_blocks
        )

    # -- forward -----------------------------------------------------------
    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor | None = None):
        """Returns ``(output [B, S, hidden], block_indices | None)``."""
        cfg = self.cfg
        B, S, _ = hidden.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=hidden.device).unsqueeze(0).expand(B, -1)
        cos, sin = rope_cos_sin(position_ids, cfg.rotary_dim, cfg.rope_theta, hidden.dtype)

        H, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q = gemma_rmsnorm(self.q_proj(hidden).view(B, S, H, D), self.q_norm, cfg.rms_norm_eps).transpose(1, 2)
        k = gemma_rmsnorm(self.k_proj(hidden).view(B, S, Hkv, D), self.k_norm, cfg.rms_norm_eps).transpose(1, 2)
        v = self.v_proj(hidden).view(B, S, Hkv, D).transpose(1, 2)
        q = apply_partial_rope(q, cos, sin)
        k = apply_partial_rope(k, cos, sin)

        block_indices = None
        if cfg.sparse:
            block_indices = self.indexer(hidden, cos, sin, position_ids)
            keep = block_indices_to_mask(block_indices, position_ids, S, cfg.index_block_size, H)
        else:
            keep = causal_mask(position_ids, S)
        out = masked_attention(q, k, v, keep, self.scaling)
        out = out.transpose(1, 2).reshape(B, S, H * D)
        return self.o_proj(out), block_indices

    def dense_reference(self, hidden: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Same weights, full causal attention (ignores the indexer). Used for the degenerate test."""
        sparse, self.cfg.sparse = self.cfg.sparse, False
        try:
            out, _ = self.forward(hidden, position_ids)
        finally:
            self.cfg.sparse = sparse
        return out


def eligible_blocks_leq_topk(seq_len: int, block_size: int, topk_blocks: int) -> bool:
    """True when every query can see at most ``topk`` blocks, i.e. MSA degenerates to dense causal attention."""
    return math.ceil(seq_len / block_size) <= topk_blocks
