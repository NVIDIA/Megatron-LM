# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax Sparse Attention (MSA) core-attention kernels.

Semantics (MiniMax-M3, arXiv:2606.13392 §3.1): every query token ``i`` of GQA
group ``r`` attends to the union of ``k`` selected 128-token KV blocks
``I_i^(r)`` (its own block always included) restricted to keys ``j <= i``. The
selection is **per query token** (not per query block) and shared by the query
heads of a group.

Backends
* ``dense``  – reference: materialise the ``[B, H_q, S_q, S_k]`` keep-mask and run
  masked softmax attention in the input dtype (fp64 capable). O(S^2) memory.
* ``flex``   – default trainable path: ``torch.nn.attention.flex_attention`` with a
  per-token ``mask_mod`` reading the selection table; the BlockMask is built from
  the block-level union of the selections (no S_q x S_k intermediate) and skips
  unselected 128x128 tiles. Exact (fp32 rel ~2e-7 with TF32 off), fwd+bwd native.
  Caveat (torch 2.13, B200): the compiled fp32 kernel is only IEEE-exact for
  ``head_dim == 128``; ``head_dim == 64`` runs at TF32 precision (rel ~7e-4)
  regardless of ``ALLOW_TF32``. M3 uses 128; fp32 alignment tests must too.

Both consume the same selection table produced by :func:`block_indices_to_table`
from HF-format block indices ``[B, H_idx, S_q, K]`` (``-1`` right padding).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_FLEX = None  # lazily compiled flex_attention


def block_indices_to_table(block_idx: Tensor, n_kv_blocks: int) -> Tensor:
    """``[B, H_idx, S_q, K]`` int (``-1`` = unused) -> bool ``[B, H_idx, S_q, n_kv_blocks]``."""
    B, H, S, _ = block_idx.shape
    safe = block_idx.masked_fill(block_idx < 0, n_kv_blocks).long()
    table = torch.zeros(B, H, S, n_kv_blocks + 1, dtype=torch.bool, device=block_idx.device)
    table.scatter_(-1, safe, True)
    return table[..., :n_kv_blocks]


def keep_mask_from_table(
    table: Tensor, position_ids: Tensor, key_length: int, block_size: int, num_q_heads: int
) -> Tensor:
    """Dense boolean keep-mask ``[B, H_q, S_q, S_k]`` = selected block AND causal (token level)."""
    B, H_idx, S_q, _ = table.shape
    keep = table.repeat_interleave(block_size, dim=-1)[..., :key_length]
    keep = keep.repeat_interleave(num_q_heads // H_idx, dim=1)
    k_pos = torch.arange(key_length, device=table.device)
    future = k_pos[None, None, None, :] > position_ids[:, None, :, None]
    return keep & ~future


def msa_dense_attention(
    q: Tensor, k: Tensor, v: Tensor, table: Tensor, position_ids: Tensor, *, scale: float, block_size: int
) -> Tensor:
    """Reference backend. ``q``: ``[B, H_q, S_q, D]``; ``k``/``v``: ``[B, H_kv, S_k, D]``. Returns ``[B, H_q, S_q, D]``."""
    B, H_q, S_q, D = q.shape
    H_kv, S_k = k.shape[1], k.shape[2]
    n_rep = H_q // H_kv
    keep = keep_mask_from_table(table, position_ids, S_k, block_size, H_q)
    k_rep = k.repeat_interleave(n_rep, dim=1)
    v_rep = v.repeat_interleave(n_rep, dim=1)
    scores = torch.matmul(q, k_rep.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~keep, float("-inf"))
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    probs = torch.softmax(scores, dim=-1, dtype=acc).to(q.dtype)
    return torch.matmul(probs, v_rep)


def _get_flex():
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention

        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX


def build_flex_block_mask(table: Tensor, position_ids: Tensor, num_q_heads: int, key_length: int, block_size: int):
    """BlockMask for flex_attention from the per-token selection table (GQA group -> q heads).

    Built directly with ``BlockMask.from_kv_blocks`` from the block-level union of each 128-query block's
    selections (plus a block-level causal cut), so nothing of size ``S_q x S_k`` is materialised
    (``create_block_mask`` evaluates ``mask_mod`` densely with int64 indices: 64 GiB at 16K for 32 heads).
    Every listed block is "partial": the kernel still applies the exact per-token ``mask_mod`` inside it.
    """
    from torch.nn.attention.flex_attention import BlockMask

    B, H_idx, S_q, n_kv = table.shape
    n_rep = num_q_heads // H_idx
    pos = position_ids

    def mask_mod(b, h, q_idx, kv_idx):
        return table[b, h // n_rep, q_idx, kv_idx // block_size] & (kv_idx <= pos[b, q_idx])

    n_qb = -(-S_q // block_size)
    pad_q = n_qb * block_size - S_q
    t = table if pad_q == 0 else torch.nn.functional.pad(table, (0, 0, 0, pad_q))
    qpos = pos if pad_q == 0 else torch.nn.functional.pad(pos, (0, pad_q), value=-1)
    blocks = t.view(B, H_idx, n_qb, block_size, n_kv).any(dim=3)  # union of the block's per-token selections
    kv_start = torch.arange(n_kv, device=table.device) * block_size
    qpos_max = qpos.view(B, n_qb, block_size).amax(dim=-1)  # block-level causal: drop kv blocks entirely in the future
    blocks = blocks & (kv_start[None, None, None, :] <= qpos_max[:, None, :, None])
    blocks = blocks.repeat_interleave(n_rep, dim=1)  # [B, H_q, n_qb, n_kv]
    kv_num_blocks = blocks.sum(dim=-1, dtype=torch.int32)
    kv_indices = torch.sort(~blocks, dim=-1, stable=True).indices.to(torch.int32)  # selected blocks first, ascending
    return BlockMask.from_kv_blocks(
        kv_num_blocks, kv_indices, BLOCK_SIZE=block_size, mask_mod=mask_mod, seq_lengths=(S_q, key_length)
    )


def msa_flex_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    table: Tensor,
    position_ids: Tensor,
    *,
    scale: float,
    block_size: int,
    block_mask=None,
) -> Tensor:
    """flex_attention backend. Same contract as :func:`msa_dense_attention`."""
    if block_mask is None:
        block_mask = build_flex_block_mask(table, position_ids, q.shape[1], k.shape[2], block_size)
    return _get_flex()(q, k, v, block_mask=block_mask, scale=scale, enable_gqa=q.shape[1] != k.shape[1])


BACKENDS = ("dense", "flex", "magi")  # "magi" is handled by MSAttention itself, not by this function


def msa_core_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_idx: Tensor,
    position_ids: Tensor,
    *,
    block_size: int,
    scale: float | None = None,
    backend: str = "flex",
) -> Tensor:
    """Dispatch on ``backend``. ``block_idx``: HF-format ``[B, H_idx, S_q, K]`` with ``-1`` padding."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown MSA backend {backend!r}; expected one of {BACKENDS}")
    if backend == "magi":
        raise ValueError("backend='magi' bypasses msa_core_attention entirely (see MSAttention.forward / kernels.magi_msa)")
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    n_kv_blocks = -(-k.shape[2] // block_size)
    table = block_indices_to_table(block_idx, n_kv_blocks)
    if backend == "dense":
        return msa_dense_attention(q, k, v, table, position_ids, scale=scale, block_size=block_size)
    return msa_flex_attention(q, k, v, table, position_ids, scale=scale, block_size=block_size)


def msa_is_degenerate(seq_len: int, block_size: int, topk_blocks: int) -> bool:
    """True when every query can see at most ``topk`` blocks, i.e. MSA == dense causal attention."""
    return math.ceil(seq_len / block_size) <= topk_blocks
