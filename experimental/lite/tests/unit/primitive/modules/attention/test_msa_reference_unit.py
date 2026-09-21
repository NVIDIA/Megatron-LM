# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pure-torch reference of MiniMax Sparse Attention and its invariants (CPU, float64).

``MSAReferenceAttention`` mirrors HF ``MiniMaxM3VLAttention`` + ``MiniMaxM3VLIndexer``:
Gemma RMSNorm ``(1 + w)``, per-head QK-norm before partial RoPE, indexer scores without
``1/sqrt(d)``, token-level causal mask, block max-pool, forced local block, left-packed
top-k with ``-1`` padding, and sparse attention realised as masked dense attention.

The reference is what the GPU tests compare kernels against, so its own semantics are
pinned here: degenerate sequences equal dense causal attention, truly sparse sequences
differ, the selected block ids satisfy the structural invariants, and (when
``transformers`` ships ``minimax_m3_vl``) it agrees with the HF module in float64.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = pytest.mark.mlite


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    compute = torch.float64 if x.dtype == torch.float64 else torch.float32
    xf = x.to(compute)
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.to(compute))).to(x.dtype)


def rope_cos_sin(position_ids: torch.Tensor, rotary_dim: int, theta: float, dtype: torch.dtype):
    compute = torch.float64 if dtype == torch.float64 else torch.float32
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=compute, device=position_ids.device) / rotary_dim))
    freqs = position_ids.to(compute)[..., None] * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    return torch.cat([x_rot * cos + rotate_half(x_rot) * sin, x_pass], dim=-1)


def msa_block_scores(idx_q: torch.Tensor, idx_k: torch.Tensor, position_ids: torch.Tensor, block_size: int) -> torch.Tensor:
    """``[B, H_idx, S_q, D] x [B, 1, S_k, D]`` -> causal block-max scores ``[B, H_idx, S_q, N_kv]``."""
    B, H, S_q, _ = idx_q.shape
    S_k = idx_k.shape[2]
    n_kv = -(-S_k // block_size)
    scores = torch.matmul(idx_q, idx_k.transpose(-1, -2))
    k_pos = torch.arange(S_k, device=idx_q.device)
    scores = scores.masked_fill(k_pos[None, None, None, :] > position_ids[:, None, :, None], float("-inf"))
    pad = n_kv * block_size - S_k
    if pad:
        scores = F.pad(scores, (0, pad), value=float("-inf"))
    return scores.view(B, H, S_q, n_kv, block_size).amax(dim=-1)


def msa_select_blocks(block_scores: torch.Tensor, position_ids: torch.Tensor, block_size: int, topk: int, local_blocks: int = 1):
    """Top-k block ids ``[B, H_idx, S_q, min(topk, N_kv)]``; local blocks forced in, ``-1`` where no block is visible."""
    B, H, S_q, n_kv = block_scores.shape
    scores = block_scores.clone()
    q_block = position_ids // block_size
    if local_blocks > 0:
        local = torch.arange(local_blocks, device=scores.device)
        local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0).unsqueeze(1).expand(-1, H, -1, -1)
        scores.scatter_(-1, local_idx, float("inf"))
    top_scores, top_idx = scores.topk(min(topk, n_kv), dim=-1)
    return top_idx.masked_fill(top_scores == float("-inf"), -1)


def block_indices_to_mask(block_indices: torch.Tensor, position_ids: torch.Tensor, key_length: int, block_size: int, num_q_heads: int):
    """Boolean keep-mask ``[B, H_q, S_q, S_k]`` = selected block AND causal (index head g serves q heads g*n_rep..)."""
    B, H_idx, S_q, _ = block_indices.shape
    n_kv = -(-key_length // block_size)
    safe = block_indices.masked_fill(block_indices < 0, n_kv)
    keep = torch.zeros(B, H_idx, S_q, n_kv + 1, dtype=torch.bool, device=block_indices.device)
    keep.scatter_(-1, safe.long(), True)
    keep = keep[..., :n_kv].repeat_interleave(block_size, dim=-1)[..., :key_length]
    keep = keep.repeat_interleave(num_q_heads // H_idx, dim=1)
    k_pos = torch.arange(key_length, device=block_indices.device)
    return keep & ~(k_pos[None, None, None, :] > position_ids[:, None, :, None])


def causal_mask(position_ids: torch.Tensor, key_length: int) -> torch.Tensor:
    k_pos = torch.arange(key_length, device=position_ids.device)
    return ~(k_pos[None, None, None, :] > position_ids[:, None, :, None])


def masked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keep: torch.Tensor, scale: float) -> torch.Tensor:
    """Dense GQA attention restricted by a boolean keep-mask; softmax accumulates in fp32/fp64."""
    n_rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~keep, float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float64 if q.dtype == torch.float64 else torch.float32)
    return torch.matmul(probs.to(q.dtype), v)


@dataclass
class MSAReferenceConfig:
    hidden_size: int = 256
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 64
    rotary_dim: int = 32
    rope_theta: float = 5.0e6
    rms_norm_eps: float = 1.0e-6
    index_n_heads: int = 2
    index_head_dim: int = 64
    index_block_size: int = 128
    index_topk_blocks: int = 4
    index_local_blocks: int = 1
    sparse: bool = True


class MSAReferenceAttention(nn.Module):
    """Reference attention layer; parameter names follow the HF module names."""

    def __init__(self, cfg: MSAReferenceConfig, dtype: torch.dtype = torch.float64):
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
        Hi, Di = cfg.index_n_heads, cfg.index_head_dim
        self.index_q_proj = nn.Linear(cfg.hidden_size, Hi * Di, **kw)
        self.index_k_proj = nn.Linear(cfg.hidden_size, Di, **kw)
        self.index_q_norm = nn.Parameter(torch.zeros(Di, dtype=dtype))
        self.index_k_norm = nn.Parameter(torch.zeros(Di, dtype=dtype))
        self.scaling = D**-0.5

    def indexer(self, hidden, cos, sin, position_ids):
        cfg = self.cfg
        B, S, _ = hidden.shape
        iq = gemma_rmsnorm(self.index_q_proj(hidden).view(B, S, cfg.index_n_heads, cfg.index_head_dim), self.index_q_norm, cfg.rms_norm_eps)
        ik = gemma_rmsnorm(self.index_k_proj(hidden).view(B, S, 1, cfg.index_head_dim), self.index_k_norm, cfg.rms_norm_eps)
        iq = apply_partial_rope(iq.transpose(1, 2), cos[..., : cfg.index_head_dim], sin[..., : cfg.index_head_dim])
        ik = apply_partial_rope(ik.transpose(1, 2), cos[..., : cfg.index_head_dim], sin[..., : cfg.index_head_dim])
        scores = msa_block_scores(iq, ik, position_ids, cfg.index_block_size)
        return msa_select_blocks(scores, position_ids, cfg.index_block_size, cfg.index_topk_blocks, cfg.index_local_blocks)

    def forward(self, hidden, position_ids=None, *, sparse: bool | None = None):
        cfg = self.cfg
        sparse = cfg.sparse if sparse is None else sparse
        B, S, _ = hidden.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=hidden.device).unsqueeze(0).expand(B, -1)
        cos, sin = rope_cos_sin(position_ids, cfg.rotary_dim, cfg.rope_theta, hidden.dtype)
        H, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q = apply_partial_rope(gemma_rmsnorm(self.q_proj(hidden).view(B, S, H, D), self.q_norm, cfg.rms_norm_eps).transpose(1, 2), cos, sin)
        k = apply_partial_rope(gemma_rmsnorm(self.k_proj(hidden).view(B, S, Hkv, D), self.k_norm, cfg.rms_norm_eps).transpose(1, 2), cos, sin)
        v = self.v_proj(hidden).view(B, S, Hkv, D).transpose(1, 2)
        block_indices = None
        if sparse:
            block_indices = self.indexer(hidden, cos, sin, position_ids)
            keep = block_indices_to_mask(block_indices, position_ids, S, cfg.index_block_size, H)
        else:
            keep = causal_mask(position_ids, S)
        out = masked_attention(q, k, v, keep, self.scaling).transpose(1, 2).reshape(B, S, H * D)
        return self.o_proj(out), block_indices


def _init(module: nn.Module, std: float = 0.05) -> None:
    for name, p in module.named_parameters():
        nn.init.normal_(p, std=0.1 if name.endswith("norm") else std)


def _layer(**overrides) -> tuple[MSAReferenceConfig, MSAReferenceAttention]:
    torch.manual_seed(0)
    cfg = MSAReferenceConfig(**overrides)
    layer = MSAReferenceAttention(cfg)
    _init(layer)
    return cfg, layer


@pytest.mark.parametrize("seq", [128, 500, 512])  # multiple / non-multiple / exactly topk*block
def test_degenerate_sequence_equals_dense_causal_attention(seq):
    cfg, layer = _layer()
    assert math.ceil(seq / cfg.index_block_size) <= cfg.index_topk_blocks
    x = torch.randn(2, seq, cfg.hidden_size, dtype=torch.float64)
    sparse, idx = layer(x)
    dense, _ = layer(x, sparse=False)
    assert (sparse - dense).abs().max().item() < 1e-10
    assert idx.shape[-1] == math.ceil(seq / cfg.index_block_size)


def test_sparse_sequence_differs_from_dense_only_after_topk_blocks():
    cfg, layer = _layer()
    seq = 1024  # 8 blocks > top-4
    x = torch.randn(1, seq, cfg.hidden_size, dtype=torch.float64)
    sparse, idx = layer(x)
    dense, _ = layer(x, sparse=False)
    early = cfg.index_topk_blocks * cfg.index_block_size
    assert idx.shape[-1] == cfg.index_topk_blocks
    assert (sparse[:, :early] - dense[:, :early]).abs().max().item() < 1e-10
    assert (sparse[:, early:] - dense[:, early:]).abs().max().item() > 1e-6


@pytest.mark.parametrize("seq", [1000, 1024])
def test_block_index_invariants(seq):
    cfg, layer = _layer()
    x = torch.randn(1, seq, cfg.hidden_size, dtype=torch.float64)
    _, idx = layer(x)
    q_block = torch.arange(seq) // cfg.index_block_size
    valid = idx >= 0
    assert bool((idx[valid] <= q_block[None, None, :, None].expand_as(idx)[valid]).all()), "future block selected"
    assert bool((idx == q_block[None, None, :, None]).any(-1).all()), "local block missing"
    srt = idx.sort(-1).values
    assert not bool(((srt[..., 1:] == srt[..., :-1]) & (srt[..., 1:] >= 0)).any()), "duplicate block id"
    assert bool((((~valid).int().cumsum(-1) > 0) == ~valid).all()), "-1 padding is not right-packed"
    assert bool((valid.sum(-1) == torch.clamp(q_block + 1, max=cfg.index_topk_blocks)[None, None, :]).all())


def _hf_available() -> bool:
    try:
        import transformers.models.minimax_m3_vl.modeling_minimax_m3_vl  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _hf_available(), reason="transformers without minimax_m3_vl")
@pytest.mark.parametrize("seq", [1000, 1024])
def test_reference_matches_hf_attention_float64(seq):
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLAttention,
        MiniMaxM3VLRotaryEmbedding,
    )

    cfg, ref = _layer()
    hf_cfg = MiniMaxM3VLTextConfig(
        vocab_size=1024, hidden_size=cfg.hidden_size, num_hidden_layers=1,
        num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, rms_norm_eps=cfg.rms_norm_eps, rope_theta=cfg.rope_theta, rotary_dim=cfg.rotary_dim,
        partial_rotary_factor=cfg.rotary_dim / cfg.head_dim, index_n_heads=cfg.index_n_heads,
        index_head_dim=cfg.index_head_dim, index_block_size=cfg.index_block_size,
        index_topk_blocks=cfg.index_topk_blocks, index_local_blocks=1,
        layer_types=["minimax_m3_sparse"], mlp_layer_types=["dense"], attn_implementation="eager",
    )
    hf = MiniMaxM3VLAttention(hf_cfg, layer_idx=0).double()
    hf_rope = MiniMaxM3VLRotaryEmbedding(hf_cfg).double()
    with torch.no_grad():
        hf.q_proj.weight.copy_(ref.q_proj.weight)
        hf.k_proj.weight.copy_(ref.k_proj.weight)
        hf.v_proj.weight.copy_(ref.v_proj.weight)
        hf.o_proj.weight.copy_(ref.o_proj.weight)
        hf.q_norm.weight.copy_(ref.q_norm)
        hf.k_norm.weight.copy_(ref.k_norm)
        hf.indexer.q_proj.weight.copy_(ref.index_q_proj.weight)
        hf.indexer.k_proj.weight.copy_(ref.index_k_proj.weight)
        hf.indexer.q_norm.weight.copy_(ref.index_q_norm)
        hf.indexer.k_norm.weight.copy_(ref.index_k_norm)
    x = torch.randn(2, seq, cfg.hidden_size, dtype=torch.float64)
    pos = torch.arange(seq).unsqueeze(0).expand(2, -1)
    with torch.no_grad():
        cos, sin = hf_rope(x, pos)
        hf_out, _ = hf(x, (cos, sin), attention_mask=None, position_ids=pos)
        hf_idx = hf.indexer(x, (cos, sin), None, pos)
        ref_out, ref_idx = ref(x, pos)
    assert torch.equal(hf_idx.sort(-1).values, ref_idx.sort(-1).values), "block indices differ from HF"
    rel = ((hf_out - ref_out).abs().max() / hf_out.abs().max()).item()
    print(f"msa_reference_vs_hf seq={seq} max_rel_diff={rel:.3e}")
    assert rel < 1e-6
