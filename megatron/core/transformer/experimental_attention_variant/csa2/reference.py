# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Framework-free reference operators for DeepSeek-V4.1 CSA2.

Everything in this file is plain PyTorch that runs on CPU and GPU, uses no fused kernels
and no process groups. It serves two purposes:

1. It is the numerical definition of CSA2 that the fused / context-parallel paths are
   tested against.
2. It is the execution path of the M0 skeleton (tiny models, single GPU).

Layouts follow Megatron-Core: sequence first, ``[s, b, ...]``. Index tensors use ``-1``
for "no position".

Semantics follow the public DeepSeek-V4.1 reference (``inference/model.py`` and the
``sparse_attn`` kernel in ``inference/kernel.py``); the code is an independent
implementation.
"""

from typing import Optional, Union

import torch
import torch.nn.functional as F

# Finite stand-in for -inf when a query row has no valid key at all: exp(x - x) stays
# well defined and such rows produce an all-zero attention output, which is the
# convention of the reference kernel.
_NEG_LARGE = -1e30


def sliding_window_indices(seq_len: int, window: int, device=None) -> torch.Tensor:
    """Key positions each query may attend to inside a causal sliding window.

    Returns ``[seq_len, window]`` int32 with entries ``i - k`` for ``k in [0, window)`` and
    ``-1`` where the position would be negative. Order inside a row is irrelevant to the
    consumers (they treat every slot independently).
    """
    offsets = torch.arange(window, device=device, dtype=torch.int64)
    positions = torch.arange(seq_len, device=device, dtype=torch.int64)
    idx = positions.unsqueeze(1) - offsets.unsqueeze(0)
    return torch.where(idx >= 0, idx, torch.full_like(idx, -1)).to(torch.int32)


def compressed_visible_counts(seq_len: int, ratio: int, device=None) -> torch.Tensor:
    """Number of compressed entries a query at position ``i`` may see: ``(i + 1) // ratio``.

    A compressed entry ``j`` summarises tokens ``[j * ratio, (j + 1) * ratio)`` and becomes
    visible once the query has passed the group's last token.
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    return (torch.arange(1, seq_len + 1, device=device, dtype=torch.int64)) // ratio


def compressed_causal_mask(
    seq_len: int, n_compressed: int, ratio: int, device=None
) -> torch.Tensor:
    """``[seq_len, n_compressed]`` bool, True where query ``i`` may attend entry ``j``."""
    visible = compressed_visible_counts(seq_len, ratio, device=device)
    entries = torch.arange(n_compressed, device=device, dtype=torch.int64)
    return entries.unsqueeze(0) < visible.unsqueeze(1)


def pool_groups_with_softmax_gate(
    values: torch.Tensor, gate_logits: torch.Tensor, ratio: int
) -> torch.Tensor:
    """Pool consecutive groups of ``ratio`` rows with a softmax gate over each group.

    Args:
        values: ``[s, ..., d]`` in float32.
        gate_logits: same shape as ``values``; the softmax runs over the ``ratio`` rows of a
            group, independently per feature.
        ratio: group size. Trailing rows that do not fill a group are dropped.

    Returns:
        ``[s // ratio, ..., d]`` float32.
    """
    seq_len = values.size(0)
    n_groups = seq_len // ratio
    cutoff = n_groups * ratio
    grouped_values = values[:cutoff].unflatten(0, (n_groups, ratio))
    grouped_gates = gate_logits[:cutoff].unflatten(0, (n_groups, ratio))
    weights = torch.softmax(grouped_gates, dim=1)
    return (grouped_values * weights).sum(dim=1)


def _candidate_block_scores(
    scores: torch.Tensor, visible_counts: Union[torch.Tensor, int], block_size: int
) -> torch.Tensor:
    """Block scores (max over the block) with the block holding the newest reachable
    position pinned to +inf: it is only partially filled and could otherwise be outscored by
    an older, complete block (reference convention)."""
    n_positions = scores.size(-1)
    pad = (-n_positions) % block_size
    padded = F.pad(scores, (0, pad), value=float("-inf"))
    block_scores = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    n_blocks = block_scores.size(-1)
    if isinstance(visible_counts, int):
        newest_block = torch.tensor((visible_counts - 1) // block_size, device=scores.device)
    else:
        newest_block = (visible_counts - 1) // block_size
    block_ids = torch.arange(n_blocks, device=scores.device)
    pinned = (
        block_ids == newest_block.unsqueeze(-1)
        if newest_block.dim() > 0
        else block_ids == newest_block
    )
    return block_scores.masked_fill(pinned, float("inf"))


def select_candidate_block_ids(
    scores: torch.Tensor,
    visible_counts: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """First level of the two-level top-k: ids of the best ``topk_blocks`` blocks per query.

    Args:
        scores: ``[..., n_positions]`` with unreachable positions already at ``-inf``.
        visible_counts: per-row count of reachable positions, broadcastable to
            ``scores.shape[:-1]``, or a plain int.
        topk_blocks: blocks to keep.
        block_size: compressed positions per block.

    Returns:
        ``[..., topk_blocks]`` int32 block ids, ``-1`` where fewer reachable blocks exist.
        This compact form is what the shared state stores; consumers expand it with
        :func:`candidate_blocks_to_mask`.
    """
    block_scores = _candidate_block_scores(scores, visible_counts, block_size)
    n_blocks = block_scores.size(-1)
    k = min(topk_blocks, n_blocks)
    top_scores, top_ids = block_scores.topk(k, dim=-1)
    ids = torch.where(top_scores > float("-inf"), top_ids, torch.full_like(top_ids, -1))
    if k < topk_blocks:
        ids = F.pad(ids, (0, topk_blocks - k), value=-1)
    return ids.to(torch.int32)


def candidate_blocks_to_mask(
    block_ids: torch.Tensor, n_positions: int, block_size: int
) -> torch.Tensor:
    """Expand candidate block ids ``[..., topk_blocks]`` to a bool mask ``[..., n_positions]``."""
    n_blocks = -(-n_positions // block_size)
    # Invalid ids (-1) are routed to a scratch column so they never overwrite a kept block.
    keep = torch.zeros(
        (*block_ids.shape[:-1], n_blocks + 1), dtype=torch.bool, device=block_ids.device
    )
    target = torch.where(block_ids >= 0, block_ids, torch.full_like(block_ids, n_blocks))
    keep.scatter_(-1, target.long(), torch.ones_like(target, dtype=torch.bool))
    keep = keep[..., :n_blocks]
    return keep.repeat_interleave(block_size, dim=-1)[..., :n_positions]


def select_candidate_blocks(
    scores: torch.Tensor,
    visible_counts: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Mask form of :func:`select_candidate_block_ids`: True inside a kept block."""
    ids = select_candidate_block_ids(scores, visible_counts, topk_blocks, block_size)
    return candidate_blocks_to_mask(ids, scores.size(-1), block_size)


def indexer_topk_indices(
    scores: torch.Tensor, visible_counts: torch.Tensor, topk: int
) -> torch.Tensor:
    """Second level: top-``k`` compressed positions per query, sorted, ``-1`` where invalid.

    An entry is valid only if it is causally reachable *and* its score is finite, so
    positions excluded by a candidate mask (``-inf``) are never resurrected when fewer than
    ``k`` candidates remain.

    Args:
        scores: ``[s, b, n_compressed]`` float32 with excluded positions at ``-inf``.
        visible_counts: ``[s]`` reachable counts per query position.
        topk: entries to keep (clamped to ``n_compressed``).

    Returns:
        ``[s, b, k]`` int32.
    """
    n_compressed = scores.size(-1)
    k = min(topk, n_compressed)
    if k == 0:
        return scores.new_empty((*scores.shape[:-1], 0), dtype=torch.int32)
    top_ids = scores.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values
    reachable = top_ids < visible_counts.view(-1, *([1] * (scores.dim() - 1)))
    finite = torch.isfinite(scores.gather(-1, top_ids))
    valid = reachable & finite
    return torch.where(valid, top_ids, torch.full_like(top_ids, -1)).to(torch.int32)


def indexer_scores(
    q_index: torch.Tensor, k_index: torch.Tensor, head_weights: torch.Tensor
) -> torch.Tensor:
    """Indexer relevance of every compressed entry for every query.

    Args:
        q_index: ``[s, b, h_i, d_i]`` indexer queries (RoPE applied).
        k_index: ``[n, b, d_i]`` indexer keys, one per compressed entry (RoPE applied).
        head_weights: ``[s, b, h_i]`` per-head mixing weights, already multiplied by the
            reference scale ``d_i^-0.5 * h_i^-0.5``.

    Returns:
        ``[s, b, n]`` float32: ``sum_h relu(q_h . k) * w_h``.
    """
    raw = torch.einsum("sbhd,nbd->sbhn", q_index.float(), k_index.float())
    weighted = torch.relu(raw) * head_weights.float().unsqueeze(-1)
    return weighted.sum(dim=2)


def sparse_attention_with_sink(
    query: torch.Tensor,
    keys: torch.Tensor,
    attn_sink: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Multi-query sparse attention with a learned per-head sink, gather based.

    Args:
        query: ``[s, b, h, d]``.
        keys: ``[n, b, d]`` single-head keys (key == value).
        attn_sink: ``[h]`` float32 sink logits.
        indices: ``[s, b, k]`` int positions into ``keys`` (``-1`` = no position).
        softmax_scale: scale on ``q . k``.

    Returns:
        ``[s, b, h, d]`` in ``query.dtype``.
    """
    s, b, h, d = query.shape
    k = indices.size(-1)
    batch_ids = torch.arange(b, device=query.device).view(1, b, 1).expand(s, b, k)
    # Gather from an fp32 copy of the keys: each key position is gathered by many queries (up to
    # the window length for window keys, top-k x consumer layers for compressed keys) and the
    # gather's backward scatter-adds those contributions in the gathered dtype. Casting after the
    # gather would accumulate the key / compressed-KV gradients in bf16.
    gathered = keys.float()[indices.clamp_min(0).long(), batch_ids]  # [s, b, k, d]

    logits = torch.einsum("sbhd,sbkd->sbhk", query.float(), gathered) * softmax_scale
    invalid = (indices < 0).unsqueeze(2)
    logits = logits.masked_fill(invalid, _NEG_LARGE)

    sink = attn_sink.float().view(1, 1, h, 1)
    row_max = torch.maximum(logits.amax(dim=-1, keepdim=True), sink)
    probs = torch.exp(logits - row_max)
    probs = probs.masked_fill(invalid, 0.0)
    denom = probs.sum(dim=-1, keepdim=True) + torch.exp(sink - row_max)
    out = torch.einsum("sbhk,sbkd->sbhd", probs / denom, gathered.float())
    return out.to(query.dtype)


def dense_reference_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    attn_sink: torch.Tensor,
    allowed: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Dense masked attention with sink, used by tests to cross-check the sparse gather form.

    Args:
        query: ``[s, b, h, d]``; keys: ``[n, b, d]``; attn_sink: ``[h]``.
        allowed: ``[s, n]`` bool, True where query ``i`` may attend key ``j``.
    """
    logits = torch.einsum("sbhd,nbd->sbhn", query.float(), keys.float()) * softmax_scale
    logits = logits.masked_fill(~allowed.view(allowed.size(0), 1, 1, allowed.size(1)), _NEG_LARGE)
    sink = attn_sink.float().view(1, 1, -1, 1)
    row_max = torch.maximum(logits.amax(dim=-1, keepdim=True), sink)
    probs = torch.exp(logits - row_max)
    probs = probs.masked_fill(~allowed.view(allowed.size(0), 1, 1, allowed.size(1)), 0.0)
    denom = probs.sum(dim=-1, keepdim=True) + torch.exp(sink - row_max)
    return torch.einsum("sbhn,nbd->sbhd", probs / denom, keys.float()).to(query.dtype)


def concat_window_and_compressed_indices(
    window_indices: torch.Tensor,
    compressed_indices: Optional[torch.Tensor],
    n_window_keys: int,
    batch_size: int,
) -> torch.Tensor:
    """Merge window indices ``[s, w]`` with compressed indices ``[s, b, k]`` into one
    ``[s, b, w + k]`` index tensor addressing ``cat([window_keys, compressed_keys])``.

    Compressed indices are shifted by ``n_window_keys``; ``-1`` entries stay ``-1``.
    """
    s = window_indices.size(0)
    window = window_indices.unsqueeze(1).expand(s, batch_size, -1)
    if compressed_indices is None or compressed_indices.size(-1) == 0:
        return window.contiguous()
    shifted = torch.where(
        compressed_indices >= 0,
        compressed_indices + n_window_keys,
        torch.full_like(compressed_indices, -1),
    )
    return torch.cat([window, shifted.to(window.dtype)], dim=-1)
