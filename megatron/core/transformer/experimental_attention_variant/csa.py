# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace
from megatron.core.models.common.embeddings import RotaryEmbedding, apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import csa_cp_utils as cp_utils
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    FusedDSAIndexerLoss,
    fused_qk_topk_naive,
    fused_qk_topk_naive_thd,
    rotate_activation,
)
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    FusedIndexerSparseAttnFromTopkFunc,
    batch_of_row,
    build_flat_topk_idxs,
    dsa_sparse_attn,
    fused_indexer_sparse_attn,
    indexer_topk,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

# ---------------------------------------------------------------------------
# Helper functions for index computation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _get_window_topk_idxs_cached(window_size: int, seqlen: int, device_str: str) -> torch.Tensor:
    """Compute sliding-window indices for a single sequence (cached).

    Returns:
        indices: [seqlen, window_size] int tensor, -1 for invalid positions.
    """
    base = torch.arange(seqlen, device=device_str).unsqueeze(1)
    offsets = torch.arange(window_size, device=device_str)
    matrix = (base - window_size + 1).clamp(min=0) + offsets
    matrix = torch.where(matrix > base, -1, matrix)
    return matrix


def get_window_topk_idxs(
    window_size: int, batch_size: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    """Sliding-window indices [batch, seqlen, window_size]."""
    matrix = _get_window_topk_idxs_cached(window_size, seqlen, str(device))
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


@lru_cache(maxsize=8)
def _get_compress_topk_idxs_cached(
    ratio: int, seqlen: int, offset: int, device_str: str
) -> torch.Tensor:
    """Compute all-compressed-positions indices for a single sequence (cached).

    Returns:
        indices: [seqlen, seqlen // ratio] int tensor, -1 for future positions.
    """
    n_compressed = seqlen // ratio
    matrix = torch.arange(n_compressed, device=device_str).repeat(seqlen, 1)
    mask = matrix >= torch.arange(1, seqlen + 1, device=device_str).unsqueeze(1) // ratio
    matrix = torch.where(mask, -1, matrix + offset)
    return matrix


def get_compress_topk_idxs(
    ratio: int, batch_size: int, seqlen: int, offset: int, device: torch.device
) -> torch.Tensor:
    """All-compressed-position indices [batch, seqlen, seqlen // ratio]."""
    matrix = _get_compress_topk_idxs_cached(ratio, seqlen, offset, str(device))
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


# ---------------------------------------------------------------------------
# THD (packed) variants of the index helpers above.
#
# Both produce per-row local-to-segment indices in the SAME index space that
# ``dsa_kernels.local_to_global_flat(..., cu_seqlens_q=..., cu_seqlens_kv=...)``
# expects: each row is one query token in the packed layout, each value is
# either ``-1`` (invalid / future position) or a non-negative local KV id in
# ``[0, seqlen_kv_full[batch_of_row])`` where ``seqlen_kv_full[b] =
# seqlen_kv[b] + seqlen_compressed[b]``. Window indices live in
# ``[0, seqlen_kv[b])``; compressed indices live in
# ``[seqlen_kv[b], seqlen_kv[b] + seqlen_compressed[b])``.
#
# These mirror the SBHD helpers above but cannot be lru-cached because
# their output shape depends on the per-batch ``cu_seqlens`` tensors.
# ---------------------------------------------------------------------------


def get_window_topk_idxs_thd(
    window_size: int,
    cu_seqlens_q: torch.Tensor,
    device: torch.device,
    total_q: Optional[int] = None,
) -> torch.Tensor:
    """Sliding-window indices for a packed THD layout.

    For each query token ``i`` in segment ``b`` (with ``pos_in_seq =
    i - cu_seqlens_q[b]``), the window covers the last ``window_size``
    KV positions within the same segment's original KV region:
    indices ``[max(0, pos-window_size+1), ..., pos]``; positions
    extending before the start of the segment are emitted as ``-1``.

    Args:
        window_size: number of positions per window.
        cu_seqlens_q: ``(B+1,)`` int32 cumulative Q lengths
            (self-attention: same as KV lengths).
        device: tensor device.
        total_q: total number of query tokens (avoids a GPU→CPU sync
            when the caller already knows it, e.g. from ``x.shape[0]``).

    Returns:
        ``(total_q, window_size)`` int32 — LOCAL (per-segment) KV indices.
    """
    if total_q is None:
        total_q = int(cu_seqlens_q[-1].item())
    batch_of_token = batch_of_row(cu_seqlens_q, total_q=total_q)
    token_idx = torch.arange(total_q, device=device, dtype=cu_seqlens_q.dtype)
    pos_in_seq = token_idx - cu_seqlens_q[batch_of_token]  # (total_q,)

    offsets = torch.arange(window_size, device=device, dtype=cu_seqlens_q.dtype)
    matrix = (pos_in_seq - window_size + 1).clamp(min=0).unsqueeze(1) + offsets.unsqueeze(0)
    matrix = torch.where(matrix > pos_in_seq.unsqueeze(1), torch.full_like(matrix, -1), matrix)
    return matrix.int()


def get_compress_topk_idxs_thd(
    ratio: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    device: torch.device,
    total_q: Optional[int] = None,
    max_n_compressed: Optional[int] = None,
) -> torch.Tensor:
    """All compressed-position indices for a packed THD layout.

    For each query token ``i`` in segment ``b`` (``pos_in_seq = i -
    cu_seqlens_q[b]``), the valid compressed positions within that
    segment are ``[0, 1, ..., (pos+1) // ratio - 1]`` (clamped to
    ``seqlen_compressed[b]``). The returned indices are already shifted
    by the per-segment offset ``seqlen_kv[b]`` so that they live in the
    *full* per-segment KV index space ``[seqlen_kv[b], seqlen_kv[b] +
    seqlen_compressed[b])`` — exactly mirroring the SBHD helper's
    ``offset=sq`` shift.

    Args:
        ratio: indexer compression ratio.
        cu_seqlens_q: ``(B+1,)`` int32 cumulative Q lengths.
        cu_seqlens_kv: ``(B+1,)`` int32 cumulative original-KV lengths
            (used to derive the per-segment compressed-offset).
        cu_seqlens_compressed: ``(B+1,)`` int32 cumulative compressed-KV
            lengths (== Compressor's second return value).
        device: tensor device.
        total_q: total number of query tokens (avoids a GPU→CPU sync
            when the caller already knows it, e.g. from ``x.shape[0]``).
        max_n_compressed: max compressed sequence length across segments
            (avoids a GPU→CPU sync when the caller can derive it, e.g.
            ``max_seqlen_q // ratio``).

    Returns:
        ``(total_q, max_compressed_per_seq)`` int32 — LOCAL (per-segment)
        full-KV indices, ``-1`` for future positions.
    """
    if total_q is None:
        total_q = int(cu_seqlens_q[-1].item())
    seq_lens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
    seq_lens_compressed = cu_seqlens_compressed[1:] - cu_seqlens_compressed[:-1]
    if max_n_compressed is None:
        if seq_lens_compressed.numel() == 0:
            return torch.empty((total_q, 0), dtype=torch.int32, device=device)
        max_n_compressed = int(seq_lens_compressed.max().item())
    if max_n_compressed == 0:
        return torch.empty((total_q, 0), dtype=torch.int32, device=device)

    batch_of_token = batch_of_row(cu_seqlens_q, total_q=total_q)
    token_idx = torch.arange(total_q, device=device, dtype=cu_seqlens_q.dtype)
    pos_in_seq = token_idx - cu_seqlens_q[batch_of_token]

    n_valid_per_row = ((pos_in_seq + 1) // ratio).clamp(max=seq_lens_compressed[batch_of_token])
    offset_per_row = seq_lens_kv[batch_of_token]  # compressed starts at seqlen_kv[b]

    col_idx = (
        torch.arange(max_n_compressed, device=device, dtype=cu_seqlens_q.dtype)
        .unsqueeze(0)
        .expand(total_q, -1)
    )
    valid = col_idx < n_valid_per_row.unsqueeze(1)
    matrix = torch.where(valid, col_idx + offset_per_row.unsqueeze(1), torch.full_like(col_idx, -1))
    return matrix.int()


def build_cu_seqlens_kv_full(
    cu_seqlens_kv: torch.Tensor, cu_seqlens_compressed: torch.Tensor
) -> torch.Tensor:
    """Cumulative sequence lengths for the per-segment-concatenated
    ``kv_full_thd = cat_per_seg([kv_thd, compressed_kv_thd])``.

    ``kv_full_thd[cu_seqlens_kv_full[b] + i]`` for ``i in [0, seqlen_kv[b])``
    is ``kv_thd[cu_seqlens_kv[b] + i]``; for ``i in [seqlen_kv[b],
    seqlen_kv[b] + seqlen_compressed[b])`` it's
    ``compressed_kv_thd[cu_seqlens_compressed[b] + (i - seqlen_kv[b])]``.
    """
    full_lens = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]) + (
        cu_seqlens_compressed[1:] - cu_seqlens_compressed[:-1]
    )
    return torch.cat(
        [
            torch.zeros(1, dtype=cu_seqlens_kv.dtype, device=cu_seqlens_kv.device),
            full_lens.cumsum(0).to(cu_seqlens_kv.dtype),
        ]
    )


def cat_per_segment(
    kv_thd: torch.Tensor,
    compressed_kv_thd: Optional[torch.Tensor],
    cu_seqlens_kv: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    cu_seqlens_kv_full: torch.Tensor,
) -> torch.Tensor:
    """Build ``kv_full_thd`` by per-segment concatenation of ``kv_thd`` and
    ``compressed_kv_thd`` (the THD equivalent of ``torch.cat([kv,
    compressed_kv], dim=0)`` in the SBHD path).

    Fully vectorized: computes destination indices for all tokens via
    ``batch_of_row`` + offset arithmetic and writes with two indexed
    assignments — no Python loop, no GPU→CPU sync.

    Args:
        kv_thd: ``(total_kv, *trailing)``.
        compressed_kv_thd: ``(total_comp, *trailing)`` or ``None`` if every
            segment had ``seqlen < ratio`` (returns ``kv_thd`` unchanged).
        cu_seqlens_kv:        ``(B+1,)`` int32.
        cu_seqlens_compressed:``(B+1,)`` int32.
        cu_seqlens_kv_full:   ``(B+1,)`` int32 (computed by
            :func:`build_cu_seqlens_kv_full`).

    Returns:
        ``(total_kv_full, *trailing)`` packed concat.
    """
    if compressed_kv_thd is None:
        return kv_thd

    total_kv = kv_thd.shape[0]
    total_kv_full = total_kv + compressed_kv_thd.shape[0]
    device = kv_thd.device
    out_shape = (total_kv_full,) + tuple(kv_thd.shape[1:])
    out = torch.empty(out_shape, dtype=kv_thd.dtype, device=device)

    # KV tokens: dst[i] = cu_full[b] + (i - cu_kv[b])
    batch_of_kv = batch_of_row(cu_seqlens_kv, total_q=total_kv)
    src_kv = torch.arange(total_kv, device=device, dtype=cu_seqlens_kv.dtype)
    dst_kv = cu_seqlens_kv_full[batch_of_kv] + (src_kv - cu_seqlens_kv[batch_of_kv])
    out[dst_kv] = kv_thd

    # Compressed tokens: dst[j] = cu_full[b] + kv_len[b] + (j - cu_comp[b]).
    # ``compressed_kv_thd`` may use a static graph-capture capacity. Rows beyond
    # ``cu_seqlens_compressed[-1]`` are tail padding and no valid index can
    # reference them.
    total_comp_capacity = compressed_kv_thd.shape[0]
    if total_comp_capacity > 0:
        kv_lens = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
        src_comp = torch.arange(
            total_comp_capacity, device=device, dtype=cu_seqlens_compressed.dtype
        )
        num_sequences = cu_seqlens_compressed.shape[0] - 1
        batch_of_comp = batch_of_row(cu_seqlens_compressed, total_q=total_comp_capacity).clamp(
            max=max(num_sequences - 1, 0)
        )
        valid_comp = src_comp < cu_seqlens_compressed[-1]
        dst_comp = (
            cu_seqlens_kv_full[batch_of_comp]
            + kv_lens[batch_of_comp]
            + (src_comp - cu_seqlens_compressed[batch_of_comp])
        )
        dst_comp = torch.where(valid_comp, dst_comp, total_kv + src_comp)
        out[dst_comp] = compressed_kv_thd

    return out


# ---------------------------------------------------------------------------
# Helper functions for RoPE
# ---------------------------------------------------------------------------


def _stride_tables_per_segment(
    *tables: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    ratio: int,
    total_comp: Optional[int] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Per-segment strided slicing for THD packed layout with ``ratio > 1``.

    THD analogue of the SBHD ``table[:total:ratio][:seq_len]`` trick.  For each
    segment of compressed length *s*, takes ``table[:s*ratio:ratio][:s]`` and
    concatenates all segments into a packed table aligned with
    ``cu_seqlens_compressed``.

    Fully vectorized: builds a flat gather index on GPU and applies a single
    ``index_select`` per table — no Python loop, no GPU→CPU sync.

    Args:
        *tables: one or more 1-D-leading rotary tables, each ``(max_len, ...)``.
        cu_seqlens_compressed: ``(B+1,)`` int32 cumulative compressed lengths.
        ratio: compression stride (> 1).
        total_comp: total number of compressed tokens (avoids a GPU→CPU sync
            when the caller already knows it, e.g. from ``x.shape[0]``).
    """
    if total_comp is None:
        total_comp = int(cu_seqlens_compressed[-1].item())
    if total_comp == 0:
        ret = [t[:0] for t in tables]
        return ret[0] if len(ret) == 1 else tuple(ret)

    device = cu_seqlens_compressed.device
    row_idx = torch.arange(total_comp, device=device, dtype=cu_seqlens_compressed.dtype)
    num_sequences = cu_seqlens_compressed.shape[0] - 1
    batch_ids = batch_of_row(cu_seqlens_compressed, total_q=total_comp).clamp(
        max=max(num_sequences - 1, 0)
    )
    valid = row_idx < cu_seqlens_compressed[-1]
    local_pos = row_idx - cu_seqlens_compressed[batch_ids]
    table_len = tables[0].shape[0]
    gather_idx = torch.where(
        valid, (local_pos * ratio).clamp(max=table_len - 1), torch.zeros_like(local_pos)
    )

    gather_idx_expanded = gather_idx.view(-1, *([1] * (tables[0].ndim - 1)))
    results = [torch.gather(t, 0, gather_idx_expanded.expand(-1, *t.shape[1:])) for t in tables]
    return results[0] if len(results) == 1 else tuple(results)


def _apply_fused_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    cu_seqlens: Optional[torch.Tensor],
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Apply the fused MLA RoPE kernel with automatic 3-D / 4-D handling."""
    packed_seq = cu_seqlens is not None

    # Strip the dummy batch axis for packed sequences: (total, 1, h, d) → (total, h, d)
    squeezed_b = packed_seq and x.dim() == 4 and x.size(1) == 1
    if squeezed_b:
        x = x.squeeze(1)

    # Add a dummy head axis for non-packed sequences: (b, s, d) → (b, s, 1, d)
    squeeze_head = not packed_seq and x.dim() == 3
    if squeeze_head:
        x = x.unsqueeze(-2)

    out = fused_mla_rope_inplace(
        x,
        cos,
        sin,
        nope_dim,
        pos_dim,
        cu_seqlens,
        cp_group.rank(),
        cp_group.size(),
        remove_interleaving=True,
    )

    if squeezed_b:
        out = out.unsqueeze(1)
    if squeeze_head:
        out = out.squeeze(-2)
    return out


def _apply_unfused_rope(
    x: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    config: TransformerConfig,
    cu_seqlens: Optional[torch.Tensor],
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Apply unfused RoPE (split, rotate, concat) with 3-D / 4-D handling.

    DSv4 forces ``mscale=1.0`` — the model relies on Q/KV RMS-norm +
    unit-magnitude rotation, not Yarn's concentration factor.
    """
    packed_seq = cu_seqlens is not None

    # Drop dummy ``b=1`` from packed 4-D ``(total, 1, h, d)`` callers.
    squeezed_b = packed_seq and x.dim() == 4 and x.size(1) == 1
    # Packed 3-D ``(total, 1, d)``: collapse batch and add a temporary head dim.
    squeezed_b_3d = packed_seq and x.dim() == 3 and x.size(1) == 1
    if squeezed_b:
        x = x.squeeze(1)
    elif squeezed_b_3d:
        x = x.squeeze(1).unsqueeze(-2)

    # Non-packed 3-D ``(b, s, d)``: add a temporary head dim.
    squeeze_head = not packed_seq and x.dim() == 3
    if squeeze_head:
        x = x.unsqueeze(-2)

    x_nope, x_pe = torch.split(x, [nope_dim, pos_dim], dim=-1)
    x_pe = apply_rotary_pos_emb(
        x_pe,
        rotary_pos_emb,
        config=config,
        cu_seqlens=cu_seqlens,
        mscale=1.0,
        cp_group=cp_group,
        mla_rotary_interleaved=True,
        mla_output_remove_interleaving=True,
    )
    out = torch.cat([x_nope, x_pe], dim=-1)

    if squeezed_b:
        out = out.unsqueeze(1)
    elif squeezed_b_3d:
        out = out.squeeze(-2).unsqueeze(1)
    elif squeeze_head:
        out = out.squeeze(-2)
    return out


def _apply_rope(
    x: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    rotary_pos_emb_module: RotaryEmbedding,
    config: TransformerConfig,
    rotary_seq_len: int,
    ratio: int = 1,
    cp_group: torch.distributed.ProcessGroup = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen_rope: Optional[int] = None,
) -> torch.Tensor:
    """Apply RoPE to the last ``pos_dim`` dims, leaving the rest unchanged.

    Accepts both 3-D ``[seq, batch, head_dim]`` and 4-D ``[seq, batch, heads, head_dim]``
    inputs.  When the input is 3-D a temporary head dimension is inserted for
    ``apply_rotary_pos_emb`` and removed before returning.

    Two layouts:

    * **SBHD** (``cu_seqlens=None``): builds a single rotary table of length
      ``rotary_seq_len * ratio`` and slices with stride ``ratio``.
    * **THD packed** (``cu_seqlens`` supplied): per-segment strided tables via
      ``_stride_tables_per_segment``.

    Args:
        max_seqlen_rope: pre-computed ``max(seg_lens) * ratio`` for the
            THD + ``ratio > 1`` path (avoids a GPU→CPU sync when the
            caller already knows the max original sequence length).
    """
    packed_seq = cu_seqlens is not None

    if packed_seq:
        if max_seqlen_rope is not None:
            max_total = max_seqlen_rope
        else:
            seg_lens_t = cu_seqlens[1:] - cu_seqlens[:-1]
            max_total = int(seg_lens_t.max().item()) * ratio if seg_lens_t.numel() > 0 else 0
    else:
        max_total = None

    # Fused path applies to THD unconditionally and to SBHD only for
    # non-"rope" types (plain "rope" always goes through the unfused path).
    use_fused = config.apply_rope_fusion and (packed_seq or config.rope_type != "rope")

    if use_fused:
        # ``mscale=1.0`` keeps the cached cos/sin free of yarn's
        # concentration factor so the fused kernel matches the unfused
        # split-rotate path (DSv4 "pure rotation" contract).
        if packed_seq:
            cos, sin = rotary_pos_emb_module.get_cached_cos_sin(
                max_total, dtype=x.dtype, packed_seq=False, mscale=1.0
            )
            if ratio > 1:
                cos, sin = _stride_tables_per_segment(
                    cos, sin, cu_seqlens_compressed=cu_seqlens, ratio=ratio, total_comp=x.shape[0]
                )
                return _apply_fused_rope(x, cos, sin, nope_dim, pos_dim, None, cp_group)
        else:
            total = rotary_seq_len * ratio if ratio > 1 else rotary_seq_len
            cos, sin = rotary_pos_emb_module.get_cached_cos_sin(
                total, dtype=x.dtype, packed_seq=False, mscale=1.0
            )
            if ratio > 1:
                cos = cos[:total:ratio][:rotary_seq_len]
                sin = sin[:total:ratio][:rotary_seq_len]
        return _apply_fused_rope(x, cos, sin, nope_dim, pos_dim, cu_seqlens, cp_group)

    # ---- Unfused path: build rotary_pos_emb tensor ----------------------
    if packed_seq:
        rope_result = rotary_pos_emb_module(max_total, packed_seq=False)
        rotary_pos_emb = rope_result[0] if isinstance(rope_result, tuple) else rope_result
        if ratio > 1:
            rotary_pos_emb = _stride_tables_per_segment(
                rotary_pos_emb, cu_seqlens_compressed=cu_seqlens, ratio=ratio, total_comp=x.shape[0]
            )
            return _apply_unfused_rope(x, rotary_pos_emb, nope_dim, pos_dim, config, None, cp_group)
    else:
        total = rotary_seq_len * ratio if ratio > 1 else rotary_seq_len
        rope_result = rotary_pos_emb_module(total, packed_seq=False)
        rotary_pos_emb = rope_result[0] if isinstance(rope_result, tuple) else rope_result
        if ratio > 1:
            rotary_pos_emb = rotary_pos_emb[:total:ratio][:rotary_seq_len]

    return _apply_unfused_rope(x, rotary_pos_emb, nope_dim, pos_dim, config, cu_seqlens, cp_group)


# ---------------------------------------------------------------------------
# Sparse attention kernel (unfused, differentiable)
# ---------------------------------------------------------------------------


def unfused_compressed_sparse_attn(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Differentiable sparse attention with MQA + learnable attention sink.

    Layout is detected from ``query.ndim``:

    * **SBHD** (4-D query):
        query        ``(sq, b, np, hn)``    multi-head Q.
        kv_full      ``(n_kv, b, hn)``      single-head MQA KV (original
                                            + compressed concatenated).
        topk_indices ``(b, sq, topk)``      int32 **LOCAL per-batch** ids
                                            (``-1`` invalid).
        Returns ``(sq, b, np * hn)``.

    * **THD** (3-D query — callers should pre-``squeeze(1)`` the dummy b=1 dim):
        query        ``(total_q, np, hn)``  packed multi-head Q.
        kv_full      ``(total_kv, hn)``     packed single-head MQA KV.
        topk_indices ``(total_q, topk)``    int32 **flat-global** ids into
                                            ``kv_full`` (``-1`` invalid).
        Returns ``(total_q, np * hn)``.

    The math (gather → MQA scores → softmax with sink → weighted sum) is
    identical for both layouts; SBHD adds permute / globalize-indices /
    unpermute around the call.

    Args:
        attn_sink: ``(np,)`` per-head learnable bias for the sink term.
        softmax_scale: scalar applied to ``Q · K^T`` before softmax.
    """
    is_thd = query.ndim == 3

    # ----------- Layout-specific input prep -------------------------------
    if is_thd:
        q_flat = query  # (rows, np, hn)
        kv_flat = kv_full  # (n_kv, hn)
        global_indices = topk_indices  # (rows, topk)
    else:
        sq, b, np_, hn = query.size()
        n_kv = kv_full.size(0)
        # b-major flatten of query and kv_full.
        q_flat = query.permute(1, 0, 2, 3).reshape(b * sq, np_, hn)
        kv_flat = kv_full.permute(1, 0, 2).reshape(b * n_kv, hn)
        # Globalize topk_indices: ``global = batch_idx * n_kv + local``.
        valid = topk_indices >= 0
        batch_ids = torch.arange(b, device=query.device).view(b, 1, 1)
        global_indices = torch.where(valid, topk_indices + batch_ids * n_kv, topk_indices).reshape(
            b * sq, -1
        )

    # ----------- Shared core: gather, MQA softmax with sink, sum ---------
    rows, np_, hn = q_flat.shape

    safe_indices = global_indices.clamp(min=0).long()
    safe_indices_exp = safe_indices.unsqueeze(-1).expand(-1, -1, hn)
    kv_gathered = torch.gather(
        kv_flat.unsqueeze(0).expand(rows, -1, -1), dim=1, index=safe_indices_exp
    )  # (rows, topk, hn)

    q_f = q_flat.float()
    kv_g = kv_gathered.float()
    scores = torch.einsum("inh,ikh->ink", q_f, kv_g) * softmax_scale  # (rows, np, topk)

    invalid_mask = (global_indices < 0).unsqueeze(1)  # (rows, 1, topk)
    scores = scores.masked_fill(invalid_mask, float("-inf"))

    sink = attn_sink.view(1, np_, 1).float()
    scores_max = scores.max(dim=-1, keepdim=True).values
    scores_max = torch.max(scores_max, sink)

    exp_scores = torch.exp(scores - scores_max)
    exp_sink = torch.exp(sink - scores_max)
    attn_weights = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + exp_sink)

    output = torch.einsum("ink,ikh->inh", attn_weights, kv_g)
    output = output.to(query.dtype)

    # ----------- Layout-specific output reshape ---------------------------
    if is_thd:
        return output.reshape(rows, np_ * hn)
    return output.reshape(b, sq, np_ * hn).permute(1, 0, 2).contiguous()


def _unfused_indexer_sparse_attn_from_topk(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    indexer_topk_indices: torch.Tensor,
    compressed_kv: torch.Tensor,
    softmax_scale: float,
    indexer_softmax_scale: float,
    loss_coeff: float,
    calculate_per_token_loss: bool,
    global_query_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch sparse attention plus caller-supplied top-k indexer loss for THD CP.

    This mirrors the fused CP top-k path at the tensor-contract level:
    ``topk_indices`` indexes ``kv_full`` for sparse attention, and
    ``indexer_topk_indices`` indexes rank-major compressed K for indexer loss.
    """
    output = unfused_compressed_sparse_attn(query, kv_full, attn_sink, topk_indices, softmax_scale)

    total_q, np_, hn = query.shape
    indexer_topk = indexer_topk_indices.shape[-1]
    if indexer_topk == 0:
        return output, query.new_zeros((), dtype=torch.float32)

    valid = indexer_topk_indices >= 0
    row_valid = valid.any(dim=-1, keepdim=True)
    safe_indices = indexer_topk_indices.clamp(min=0).long()

    index_scores_full = torch.einsum("rhd,kd->rhk", q_indexer.float(), k_indexer.float())
    index_scores_full = torch.relu(index_scores_full)
    weights_scaled = weights.float() * float(indexer_softmax_scale)
    index_scores_full = index_scores_full * weights_scaled.unsqueeze(-1)
    index_scores_full = index_scores_full.sum(dim=1)

    predict_logits = torch.gather(index_scores_full, dim=-1, index=safe_indices)
    predict_logits = predict_logits.masked_fill(~valid, float("-inf"))
    predict_logits = predict_logits.masked_fill(~row_valid, 0.0)
    predict = torch.softmax(predict_logits, dim=-1, dtype=torch.float32)
    predict = predict * row_valid.float()

    selected_kv = compressed_kv.detach().index_select(0, safe_indices.reshape(-1))
    selected_kv = selected_kv.reshape(total_q, indexer_topk, hn)

    # Match the fused CP top-k path: the target uses the attention
    # probability mass of the compressed top-k prefix, normalized by the same
    # sink-aware denominator that FlashMLA reports as lse_indexer.
    attn_scores = torch.einsum("rhd,rkd->rhk", query.detach().float(), selected_kv.float())
    attn_scores = attn_scores * softmax_scale
    attn_scores = attn_scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    sink = attn_sink.view(1, np_, 1).float()
    scores_max = torch.maximum(attn_scores.max(dim=-1, keepdim=True).values, sink)
    exp_scores = torch.exp(attn_scores - scores_max)
    exp_sink = torch.exp(sink - scores_max)
    attn_probs = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + exp_sink)
    attn_probs = attn_probs * row_valid.unsqueeze(1).float()
    target = attn_probs.sum(dim=1)

    eps = torch.finfo(torch.float32).tiny
    target = target.clamp(min=eps)
    predict = predict.clamp(min=eps)
    kl_per_row = (target * (torch.log(target) - torch.log(predict))).sum(dim=-1)
    kl_per_row = torch.where(row_valid.squeeze(-1), kl_per_row, torch.zeros_like(kl_per_row))
    raw_local_loss = kl_per_row.sum()

    if calculate_per_token_loss:
        indexer_loss = raw_local_loss * loss_coeff
    else:
        if global_query_rows <= 0:
            raise RuntimeError(f"global_query_rows must be positive, got {global_query_rows}.")
        indexer_loss = raw_local_loss * (float(loss_coeff) / float(global_query_rows))
    return output, indexer_loss


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


@dataclass
class CompressorSubmodules:
    """Submodule specs for CSA and HCA Compressor."""

    linear_wkv: Union[ModuleSpec, type] = None
    linear_wgate: Union[ModuleSpec, type] = None
    norm: Union[ModuleSpec, type] = None


class Compressor(MegatronModule):
    """Gated pooling compressor for CSA and HCA sparse attention.

    Compresses a sequence of tokens into a shorter sequence by pooling groups of
    ``compress_ratio`` tokens using learned gated weights.

    For ``compress_ratio == 4``, overlapping compression is used (``coff = 2``).
    For ``compress_ratio == 128``, non-overlapping compression is used (``coff = 1``).

    Arbitrary-seqlen handling (same rule for SBHD and THD):
        Per-segment ``cutoff = (seqlen // ratio) * ratio = seqlen - (seqlen % ratio)``.
        Only the first ``cutoff`` tokens are pooled, producing ``seqlen // ratio``
        compressed entries. The trailing ``seqlen % ratio`` tokens are NOT
        compressed and have no compressed-KV representation — they rely on the
        sliding window for attention. This matches inference behavior (a
        decode token sitting in an incomplete buffer of 1..ratio-1 tokens has
        no compressed entry either) and avoids train/inference mismatch from
        padding-to-ratio.

        Causal-mask consequence: under the codebase's ``(i+1) // ratio``
        convention, a query token at 0-indexed position ``i`` attends to
        ``min((i+1) // ratio, n_compressed_in_segment)`` compressed entries.
        The ``clamp`` (in ``get_compress_topk_idxs*`` / kernel-level
        ``_indexer_topk_core``) ensures positions in the dropped tail (and
        positions in segments shorter than ``ratio``) never index past
        ``n_compressed_in_segment``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CompressorSubmodules,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
        rotary_pos_emb: nn.Module = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = compress_ratio == 4
        self.coff = 1 + int(self.overlap)
        self.rotate = rotate
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim

        self.rotary_pos_emb = rotary_pos_emb

        proj_out_dim = self.coff * head_dim

        self.linear_wkv = build_module(
            submodules.linear_wkv,
            config.hidden_size,
            proj_out_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.linear_wgate = build_module(
            submodules.linear_wgate,
            config.hidden_size,
            proj_out_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        # keep to high precision
        _ape = torch.empty(
            compress_ratio, proj_out_dim, device=torch.cuda.current_device(), dtype=torch.float32
        )
        config.init_method(_ape)
        self.ape = nn.Parameter(_ape)

        norm_config = copy.copy(config)
        norm_config.normalization = "RMSNorm"
        self.norm = build_module(
            submodules.norm, config=norm_config, hidden_size=head_dim, eps=config.layernorm_epsilon
        )

    def _overlap_transform(self, tensor: torch.Tensor, fill_value: float = 0) -> torch.Tensor:
        """Apply overlapping window transform for 4x compression.

        Input shape:  [n_groups, ratio, b, coff * head_dim]
        Output shape: [n_groups, 2 * ratio, b, head_dim]

        Used by the SBHD path where all groups belong to the same sequence.
        """
        n_groups, ratio, b_dim, _ = tensor.size()
        d = self.head_dim
        new_tensor = tensor.new_full((n_groups, 2 * ratio, b_dim, d), fill_value)
        new_tensor[:, ratio:] = tensor[:, :, :, d:]
        new_tensor[1:, :ratio] = tensor[:-1, :, :, :d]
        return new_tensor

    def _overlap_transform_thd(
        self, tensor: torch.Tensor, is_first_in_seg: torch.Tensor, fill_value: float = 0
    ) -> torch.Tensor:
        """Batched overlapping window transform for THD packed layout.

        Like :meth:`_overlap_transform` but operates on the flat
        ``(total_comp, ratio, b, coff * head_dim)`` tensor from all segments
        at once. ``is_first_in_seg`` is a ``(total_comp,)`` bool mask that
        is ``True`` for each compressed entry that starts a new segment
        (i.e. has no predecessor group to pull from).

        Input shape:  [total_comp, ratio, b, coff * head_dim]
        Output shape: [total_comp, 2 * ratio, b, head_dim]
        """
        n, ratio, b_dim, _ = tensor.size()
        d = self.head_dim
        new_tensor = tensor.new_full((n, 2 * ratio, b_dim, d), fill_value)
        new_tensor[:, ratio:] = tensor[:, :, :, d:]
        # Previous group's first-half data — shift by 1 along dim-0.
        prev_data = torch.roll(tensor[:, :, :, :d], shifts=1, dims=0)
        # Zero-fill (or fill_value-fill) segment boundaries.
        prev_data[is_first_in_seg] = fill_value
        new_tensor[:, :ratio] = prev_data
        return new_tensor

    def _forward_sbhd(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """SBHD path. ``x`` is ``(sq, b, hidden_size)``; returns
        ``(sq // ratio, b, head_dim)`` or ``None`` when ``sq < ratio``.
        """
        sq = x.size(0)
        ratio = self.compress_ratio

        if sq < ratio:
            return None

        kv, _ = self.linear_wkv(x)  # (sq, b, coff * head_dim)
        score, _ = self.linear_wgate(x)  # (sq, b, coff * head_dim)

        cutoff = (sq // ratio) * ratio
        if cutoff < sq:
            kv = kv[:cutoff]
            score = score[:cutoff]
        n_compressed = cutoff // ratio

        _, b_dim, _ = kv.shape
        kv = kv.view(n_compressed, ratio, b_dim, -1)
        score = score.view(n_compressed, ratio, b_dim, -1)
        score = score + self.ape.view(1, ratio, 1, -1)
        if self.overlap:
            kv = self._overlap_transform(kv, fill_value=0)
            score = self._overlap_transform(score, fill_value=float("-inf"))
        weights = torch.softmax(score, dim=1, dtype=torch.float32).to(kv.dtype)
        kv = (kv * weights).sum(dim=1)  # [n_compressed, b, head_dim]
        kv = self.norm(kv.to(x.dtype))
        kv = _apply_rope(
            kv,
            self.head_dim - self.qk_pos_emb_head_dim,
            self.qk_pos_emb_head_dim,
            self.rotary_pos_emb,
            self.config,
            n_compressed,
            ratio=ratio,
            cp_group=self.pg_collection.cp,
        )

        if self.rotate:
            kv = rotate_activation(kv)
        return kv

    def _forward_thd(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_q: Optional[int] = None,
        rope_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """THD per-segment compression — fully vectorized.

        Linear projections are token-wise on the flat input. The gated
        softmax + reduce is batched across ALL compressed entries from all
        segments via a ``(total_compressed, ratio)`` gather index.

        Args:
            x:           ``(total, 1, hidden_size)`` packed bf16.
            cu_seqlens:  ``(B+1,)`` int32 cumulative seq lengths
                (matches ``packed_seq_params.cu_seqlens_q``).
            max_seqlen_q: max original sequence length (avoids a GPU→CPU
                sync when building the rotary table for ``ratio > 1``).
            rope_positions: CP compressor-prep group ids. When supplied,
                ``x`` is already packed into ``ratio``-sized groups.

        Returns:
            ``(compressed_thd, cu_seqlens_compressed)`` where
            ``compressed_thd`` is ``(total_compressed, 1, head_dim)`` bf16
            (or ``None`` when no sequence has ``seg_len >= ratio``) and
            ``cu_seqlens_compressed`` is ``(B+1,)`` int32 with
            ``cu_seqlens_compressed[b+1] - cu_seqlens_compressed[b] = seqlen_b // ratio``.
        """
        ratio = self.compress_ratio
        device = x.device
        pre_grouped = rope_positions is not None
        cu_seqlens_compressed = cp_utils.build_global_compressed_cu_seqlens(cu_seqlens, ratio)

        if pre_grouped:
            if max_seqlen_q is None:
                raise RuntimeError("DSv4 THD CP compressor input requires max_seqlen_q for RoPE.")
            if x.shape[0] % ratio != 0:
                raise RuntimeError(
                    "DSv4 THD CP compressor input expects rows divisible by ratio: "
                    f"x={x.shape[0]}, ratio={ratio}."
                )
            total_comp = x.shape[0] // ratio
            if rope_positions.shape[0] < total_comp:
                raise RuntimeError(
                    "DSv4 THD compressor received too few compressed group ids: "
                    f"ids={rope_positions.shape[0]}, compressed={total_comp}"
                )
        else:
            if max_seqlen_q is not None:
                num_sequences = max(int(cu_seqlens.shape[0]) - 1, 0)
                total_comp = min(
                    int(x.shape[0]) // ratio, num_sequences * (max_seqlen_q // ratio)
                )
            else:
                total_comp = int(cu_seqlens_compressed[-1].item())

        if total_comp == 0:
            return None, cu_seqlens_compressed

        # Token-wise projections on the FULL flat input — no boundary issue.
        kv, _ = self.linear_wkv(x)  # (total, 1, coff * head_dim)
        score, _ = self.linear_wgate(x)  # (total, 1, coff * head_dim)

        if pre_grouped:
            # Compressor-prep already groups rows as ``[g * ratio, (g + 1) * ratio)``.
            kv_grouped = kv.reshape(total_comp, ratio, 1, -1)
            score_grouped = score.reshape(total_comp, ratio, 1, -1)
            local_pos = None
        else:
            # Build gather index: (total_comp, ratio). ``total_comp`` can be a
            # static capacity for CUDA graph capture, so rows beyond the true
            # ``cu_seqlens_compressed[-1]`` are mapped to a safe source row and left
            # as tail padding by downstream index lowering.
            row_idx = torch.arange(total_comp, device=device, dtype=cu_seqlens_compressed.dtype)
            num_sequences = cu_seqlens_compressed.shape[0] - 1
            batch_ids = batch_of_row(cu_seqlens_compressed, total_q=total_comp).clamp(
                max=max(num_sequences - 1, 0)
            )
            valid_comp = row_idx < cu_seqlens_compressed[-1]
            local_pos = row_idx - cu_seqlens_compressed[batch_ids]
            local_pos = torch.where(valid_comp, local_pos, torch.zeros_like(local_pos))
            # (total_comp, 1) + (1, ratio)  →  (total_comp, ratio)
            base = cu_seqlens[batch_ids].unsqueeze(1) + local_pos.unsqueeze(1) * ratio
            base = torch.where(valid_comp.unsqueeze(1), base, torch.zeros_like(base))
            offsets = torch.arange(ratio, device=device, dtype=base.dtype).unsqueeze(0)
            gather_idx = base + offsets  # (total_comp, ratio)

            # Index flat kv/score into ``(total_comp, ratio, 1, coff * d)`` groups.
            kv_grouped = kv[gather_idx]  # (total_comp, ratio, 1, coff * d)
            score_grouped = score[gather_idx]  # same

        # APE: (ratio, coff * d) → broadcast (1, ratio, 1, coff * d).
        score_grouped = score_grouped + self.ape.view(1, ratio, 1, -1)

        if self.overlap:
            if pre_grouped:
                is_first = rope_positions[:total_comp] == 0
            else:
                is_first = local_pos == 0  # (total_comp,)
            kv_grouped = self._overlap_transform_thd(kv_grouped, is_first, fill_value=0)
            score_grouped = self._overlap_transform_thd(
                score_grouped, is_first, fill_value=float("-inf")
            )

        # Batched softmax + weighted sum — single kernel for all entries.
        # (total_comp, [2*]ratio, 1, [coff*]d)  →  (total_comp, 1, head_dim)
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float32).to(kv_grouped.dtype)
        compressed_thd = (kv_grouped * weights).sum(dim=1)

        compressed_thd = self.norm(compressed_thd.to(x.dtype))

        if not pre_grouped:
            # RoPE: applied in a single vectorized THD call.
            max_seqlen_rope = (max_seqlen_q // ratio) * ratio if max_seqlen_q is not None else None
            compressed_thd = _apply_rope(
                compressed_thd,
                self.head_dim - self.qk_pos_emb_head_dim,
                self.qk_pos_emb_head_dim,
                self.rotary_pos_emb,
                self.config,
                rotary_seq_len=0,
                ratio=ratio,
                cp_group=self.pg_collection.cp,
                cu_seqlens=cu_seqlens_compressed,
                max_seqlen_rope=max_seqlen_rope,
            )
        else:
            rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                int(max_seqlen_q), dtype=compressed_thd.dtype, packed_seq=True, mscale=1.0
            )
            compressed_thd = cp_utils.apply_thd_cp_compressed_rope_fused(
                compressed_thd,
                rotary_pos_cos,
                rotary_pos_sin,
                rope_positions,
                ratio,
                self.head_dim - self.qk_pos_emb_head_dim,
                self.qk_pos_emb_head_dim,
            )

        if self.rotate:
            compressed_thd = rotate_activation(compressed_thd)
        return compressed_thd, cu_seqlens_compressed

    def forward(
        self, x: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> Union[Optional[torch.Tensor], Tuple[Optional[torch.Tensor], torch.Tensor]]:
        """Compress hidden states into a shorter KV sequence.

        Two layouts are supported:

        * **SBHD** (default, ``packed_seq_params=None``): ``x`` is
          ``(sq, b, hidden_size)``; returns ``(sq // ratio, b, head_dim)``
          (single ``Tensor``) or ``None`` when ``sq < ratio``.
        * **THD packed** (``packed_seq_params.qkv_format == 'thd'``):
          ``x`` is ``(total, 1, hidden_size)``; returns
          ``(compressed_thd, cu_seqlens_compressed)`` (a 2-tuple) so the
          caller can build ``kv_full`` per-sequence. ``compressed_thd``
          may be ``None`` when every sequence is shorter than ``ratio``.
        """
        nvtx_range_push("compressor")
        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if is_thd:
            cu_seqlens = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
            max_seqlen_q = (
                int(packed_seq_params.max_seqlen_q)
                if packed_seq_params.max_seqlen_q is not None
                else None
            )
            result = self._forward_thd(x, cu_seqlens, max_seqlen_q=max_seqlen_q)
        else:
            result = self._forward_sbhd(x)
        nvtx_range_pop("compressor")
        return result


# ---------------------------------------------------------------------------
# CSAIndexer
# ---------------------------------------------------------------------------


@dataclass
class CSAIndexerSubmodules:
    """Submodule specs for CSAIndexer."""

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None
    compressor: Union[ModuleSpec, type] = None


class CSAIndexer(MegatronModule):
    """Learned top-k retrieval over compressed positions for CSA sparse attention.

    Computes index scores to select the most relevant compressed KV positions for each
    query.  Reuses the scoring logic from ``DSAIndexer`` (einsum -> relu -> weight -> sum
    -> topk) and ``rotate_activation`` (Hadamard transform) from ``dsa.py``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CSAIndexerSubmodules,
        compress_ratio: int,
        rotary_pos_emb: nn.Module = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        self.compress_ratio = compress_ratio
        self.hidden_size = config.hidden_size
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
        self.q_lora_rank = (
            config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size
        )

        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        self.rotary_pos_emb = rotary_pos_emb

        # Q projection
        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        # Weights projection
        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        # Own compressor (smaller head_dim, with Hadamard rotation)
        self.compressor = build_module(
            submodules.compressor,
            config=config,
            compress_ratio=compress_ratio,
            head_dim=self.index_head_dim,
            rotate=True,
            rotary_pos_emb=rotary_pos_emb,
            pg_collection=pg_collection,
        )

    def _forward_thd_query_weights_cp(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        cu_seqlens_padded: torch.Tensor,
        max_position: int,
        chunk_ranges: Tuple[Tuple[int, int], ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute local THD indexer Q/weights for the current CP row order."""
        local_rows, bsz, _ = x.size()
        if bsz != 1:
            raise RuntimeError(f"DSv4 THD CP indexer expects bsz=1, got {bsz}.")

        q, _ = self.linear_wq_b(qr)
        q = q.reshape(local_rows, bsz, self.index_n_heads, self.index_head_dim).squeeze(1)
        rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
            int(max_position), dtype=q.dtype, packed_seq=True, mscale=1.0
        )
        q = cp_utils.apply_thd_cp_local_rope_fused(
            q,
            rotary_pos_cos,
            rotary_pos_sin,
            self.index_head_dim - self.qk_pos_emb_head_dim,
            self.qk_pos_emb_head_dim,
            cu_seqlens_padded,
            chunk_ranges,
        )
        q = rotate_activation(q)

        weights, _ = self.linear_weights_proj(x)
        weights = weights.squeeze(1) * (self.index_n_heads**-0.5)
        return q, weights

    def forward_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Compute Q, compressed K, and weights before top-k selection.

        Two layouts:

        * **SBHD** (``packed_seq_params=None``): inputs are ``x (sq, b, h)``
          and ``qr (sq, b, q_lora_rank)``. Returns ``(q, k, weights)``:
          ``q (sq, b, n_heads, head_dim)``, ``k (sq // ratio, b, head_dim)``,
          ``weights (sq, b, n_heads)``.
        * **THD packed** (``packed_seq_params.qkv_format == 'thd'``):
          inputs are ``x (total, 1, h)`` and ``qr (total, 1, q_lora_rank)``.
          Returns ``(q, k, weights, cu_seqlens_compressed)`` where ``q
          (total, 1, n_heads, head_dim)``, ``k (total_comp, 1, head_dim)``
          (``None`` if every sequence is shorter than ``ratio``),
          ``weights (total, 1, n_heads)``, and ``cu_seqlens_compressed
          (B+1,)`` int32 is the second return value from
          ``self.compressor(x, packed_seq_params=...)``.
        """
        nvtx_range_push("indexer_before_topk")

        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'

        sq, bsz, _ = x.size()  # in THD: sq = total_q, bsz = 1.

        # ``cu_seqlens_q`` is None for SBHD; ``_apply_rope`` and
        # ``self.compressor.forward`` are both layout-aware.
        cu_seqlens_q = None
        max_seqlen_q = None
        if is_thd:
            cu_seqlens_q = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
            max_seqlen_q = (
                int(packed_seq_params.max_seqlen_q)
                if packed_seq_params.max_seqlen_q is not None
                else None
            )

        # Q path — projection is token-wise so it works for either layout;
        # ``_apply_rope`` selects SBHD vs THD packed mode internally
        # based on whether ``cu_seqlens`` is supplied.
        q, _ = self.linear_wq_b(qr)
        q = q.reshape(sq, bsz, self.index_n_heads, self.index_head_dim)
        q = _apply_rope(
            q,
            self.index_head_dim - self.qk_pos_emb_head_dim,
            self.qk_pos_emb_head_dim,
            self.rotary_pos_emb,
            self.config,
            rotary_seq_len=sq,
            ratio=1,
            cp_group=self.pg_collection.cp,
            cu_seqlens=cu_seqlens_q,
            max_seqlen_rope=max_seqlen_q,
        )
        q = rotate_activation(q)

        # K path: own compressor. SBHD returns ``k``; THD returns the
        # 2-tuple ``(k_thd, cu_seqlens_compressed)``.
        compressor_out = self.compressor(x, packed_seq_params=packed_seq_params)

        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.index_n_heads**-0.5)

        nvtx_range_pop("indexer_before_topk")
        if is_thd:
            k, cu_seqlens_compressed = compressor_out
            return q, k, weights, cu_seqlens_compressed
        return q, compressor_out, weights

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (index_scores, topk_indices).

        Two layouts:

        * **SBHD** (default): the original PyTorch reference path using
          :func:`fused_qk_topk_naive` with caller-supplied ``mask``.
          Returns ``(index_scores (b, sq, sk), topk_indices (b, sq, topk))``.
        * **THD packed** (``packed_seq_params.qkv_format == 'thd'``): the
          THD analogue that loops per-segment and delegates each one to
          :func:`fused_qk_topk_naive` with ``b=1``, then aggregates
          per-segment LOCAL top-K ids into a flat
          ``(total_q, topk)`` tensor. The per-segment causal
          mask is built internally from
          :attr:`self.compress_ratio`; ``mask`` is ignored. Returns
          ``(None, topk_indices)`` — per-segment scores are not
          surfaced because their shapes are heterogeneous and the only
          current caller
          (:meth:`CompressedSparseAttention._forward_thd` force_unfused
          inference) discards them.
        """
        nvtx_range_push("indexer")
        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if is_thd:
            q, k, weights, cu_seqlens_compressed_idx = self.forward_before_topk(
                x, qr, packed_seq_params
            )
            cu_seqlens_q = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
            nvtx_range_push("indexer_qk_topk")
            if k is None:
                # Every segment is shorter than ``ratio`` → no compressed
                # indexer K. Return an all--1 topk so downstream
                # consumers treat all positions as invalid.
                total_q = q.shape[0]
                index_scores = None
                topk_indices = torch.full(
                    (total_q, self.index_topk), -1, dtype=torch.int64, device=q.device
                )
            else:
                # Squeeze the dummy ``b=1`` dim that ``forward_before_topk``
                # carries (matching the THD shape contract used by the
                # cuDNN indexer kernels).
                q_thd = q.squeeze(1)
                k_thd = k.squeeze(1)
                w_thd = weights.squeeze(1)
                effective_topk = min(self.index_topk, k_thd.shape[0])
                index_scores, topk_indices = fused_qk_topk_naive_thd(
                    q_thd,
                    k_thd,
                    w_thd,
                    index_topk=effective_topk,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_compressed_idx,
                    ratio=self.compress_ratio,
                )
            nvtx_range_pop("indexer_qk_topk")
            nvtx_range_pop("indexer")
            return index_scores, topk_indices

        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)
        nvtx_range_push("indexer_qk_topk")
        effective_topk = min(self.index_topk, k.size(0))
        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, effective_topk, mask)
        nvtx_range_pop("indexer_qk_topk")
        nvtx_range_pop("indexer")
        return index_scores, topk_indices


# ---------------------------------------------------------------------------
# CompressedSparseAttention (core attention)
# ---------------------------------------------------------------------------


@dataclass
class CompressedSparseAttentionSubmodules:
    """Submodule specs for CompressedSparseAttention."""

    compressor: Union[ModuleSpec, type] = None
    indexer: Union[ModuleSpec, type] = None


class CompressedSparseAttention(MegatronModule):
    """Sparse core attention for CompressedSparseAttention.

    Combines sliding window attention with compressed KV attention.  The spec always
    provides compressor and indexer submodule specs; this ``__init__`` inspects
    ``config.csa_compress_ratios[layer_idx]`` and conditionally builds them:

    * ``ratio == 0``:  window-only (compressor and indexer NOT built)
    * ``ratio == 4``:  window + 4x compressed + learned Indexer (both built)
    * ``ratio == 128``: window + 128x compressed, attend to all (compressor built only)
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CompressedSparseAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: Optional[ProcessGroupCollection] = None,
        rotary_pos_emb: nn.Module = None,
        compress_ratio: int = 0,
        is_mtp_layer: bool = False,
    ):
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        self.layer_number = layer_number
        if is_mtp_layer:
            self.layer_number = self.layer_number + self.config.num_layers
        self.compress_ratio = compress_ratio
        self.window_size = config.csa_window_size
        self.v_head_dim = config.v_head_dim

        self.n_local_heads = config.num_attention_heads

        if softmax_scale is None:
            softmax_scale = config.v_head_dim**-0.5
        self.softmax_scale = softmax_scale

        self.apply_dsa_kernel_fusion = config.apply_dsa_kernel_fusion

        # Learnable attention sink per head
        self.attn_sink = nn.Parameter(torch.zeros(self.n_local_heads, dtype=torch.float32))

        # Conditionally build Compressor (ratio > 1)
        if self.compress_ratio > 1 and submodules.compressor is not None:
            self.compressor = build_module(
                submodules.compressor,
                config=config,
                compress_ratio=self.compress_ratio,
                head_dim=config.v_head_dim,
                rotate=False,
                rotary_pos_emb=rotary_pos_emb,
                pg_collection=pg_collection,
            )
        else:
            self.compressor = None

        # Conditionally build Indexer (ratio == 4)
        if (
            self.compress_ratio == 4
            and not config.csa_dense_mode
            and submodules.indexer is not None
        ):
            self.indexer = build_module(
                submodules.indexer,
                config=config,
                compress_ratio=self.compress_ratio,
                rotary_pos_emb=rotary_pos_emb,
                pg_collection=pg_collection,
            )
        else:
            self.indexer = None

    # ------------------------------------------------------------------
    # Private helpers – each owns one logical slice of the forward pass.
    # ------------------------------------------------------------------

    def _build_kv_full(
        self, kv: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """Concatenate original KV with compressed KV (if applicable).

        Returns:
            kv_full:       [n_kv, b, v_head_dim]  original + compressed KV.
            compressed_kv: [n_compressed, b, v_head_dim] or None.
            n_compressed:  number of compressed positions (0 when unused).
        """
        if self.compressor is not None and self.compress_ratio > 1:
            compressed_kv = self.compressor(x)
            if compressed_kv is not None:
                kv_full = torch.cat([kv, compressed_kv], dim=0)
                n_compressed = compressed_kv.size(0)
            else:
                kv_full = kv
                compressed_kv = None
                n_compressed = 0
        else:
            kv_full = kv
            compressed_kv = None
            n_compressed = 0
        return kv_full, compressed_kv, n_compressed

    def _forward_unfused_csa(
        self,
        query: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_full: torch.Tensor,
        compressed_kv: Optional[torch.Tensor],
        n_compressed: int,
        offset: int,
        window_idxs: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """PyTorch unfused path.

        Returns ``(output, indexer_loss)``.
        """
        sq, b, np, hn = query.size()
        indexer_loss = None

        if self.compress_ratio > 1 and n_compressed > 0:
            nvtx_range_push("compressed_indices")
            if self.indexer is not None:
                x_det = x.detach()
                qr_det = qr.detach()

                causal_mask = (
                    torch.arange(n_compressed, device=x.device).unsqueeze(0).expand(sq, -1)
                )
                positions = torch.arange(1, sq + 1, device=x.device).unsqueeze(1)
                causal_mask = (
                    torch.where(causal_mask >= positions // self.compress_ratio, float("-inf"), 0.0)
                    .unsqueeze(0)
                    .expand(b, -1, -1)
                )  # [b, sq, n_compressed]

                if self.training and torch.is_grad_enabled():
                    q_indexer, k_indexer, weights_indexer = self.indexer.forward_before_topk(
                        x_det, qr_det, packed_seq_params
                    )
                    indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)
                    key_for_loss = compressed_kv.unsqueeze(2).expand(-1, -1, np, -1)
                    # ``FusedDSAIndexerLoss`` does not accept a separate
                    # indexer_softmax_scale; apply it here via the
                    # weights-scaling trick so the effective weights match
                    # the pre-scale-split behaviour.
                    weights_for_unfused = weights_indexer.float() * self.indexer.softmax_scale
                    topk_indices_compressed, indexer_loss = FusedDSAIndexerLoss.apply(
                        q_indexer,
                        weights_for_unfused,
                        k_indexer,
                        query.detach(),
                        key_for_loss.detach(),
                        self.softmax_scale,
                        min(self.indexer.index_topk, n_compressed),
                        indexer_loss_coeff,
                        causal_mask,
                        getattr(self.config, "dsa_indexer_use_sparse_loss", True),
                        self.indexer.pg_collection,
                        self.config.calculate_per_token_loss,
                    )
                    if indexer_loss_coeff > 0:
                        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                            loss=indexer_loss,
                            layer_number=self.layer_number,
                            num_layers=self.config.num_layers + (self.config.mtp_num_layers or 0),
                        )
                else:
                    _, topk_indices_compressed = self.indexer(
                        x_det, qr_det, mask=causal_mask, packed_seq_params=packed_seq_params
                    )

                n_valid_per_pos = positions // self.compress_ratio  # [sq, 1]
                valid = topk_indices_compressed < n_valid_per_pos
                compress_topk_idxs = torch.where(
                    valid, topk_indices_compressed + offset, torch.tensor(-1, device=x.device)
                )
            else:
                compress_topk_idxs = get_compress_topk_idxs(
                    self.compress_ratio, b, sq, offset, query.device
                )

            topk_idxs = torch.cat([window_idxs, compress_topk_idxs], dim=-1)
            nvtx_range_pop("compressed_indices")
        else:
            topk_idxs = window_idxs

        topk_idxs = topk_idxs.int()

        nvtx_range_push("sparse_attn_kernel")
        output = unfused_compressed_sparse_attn(
            query, kv_full, self.attn_sink.float(), topk_idxs, self.softmax_scale
        )
        nvtx_range_pop("sparse_attn_kernel")
        return output, indexer_loss

    def _forward_fused_no_indexer(
        self,
        query: torch.Tensor,
        kv_full: torch.Tensor,
        n_compressed: int,
        offset: int,
        window_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """Path A: fused sparse attn with window or deterministic compressed indices."""
        sq, b, np, hn = query.size()

        nvtx_range_push("compressed_indices")
        if self.compress_ratio > 1 and n_compressed > 0:
            compress_topk_idxs = get_compress_topk_idxs(
                self.compress_ratio, b, sq, offset, query.device
            )
            flat_idxs, _ = build_flat_topk_idxs(window_idxs, compress_topk_idxs, batch_size=b)
        else:
            flat_idxs, _ = build_flat_topk_idxs(window_idxs, batch_size=b)
        nvtx_range_pop("compressed_indices")

        nvtx_range_push("sparse_attn_kernel")
        output = dsa_sparse_attn(
            query, kv_full, self.attn_sink.float(), flat_idxs, self.softmax_scale
        )
        nvtx_range_pop("sparse_attn_kernel")
        return output

    def _forward_fused_indexer_inference(
        self,
        query: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_full: torch.Tensor,
        n_compressed: int,
        offset: int,
        window_idxs: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams],
    ) -> torch.Tensor:
        """Path C: separate indexer forward (no loss) + fused sparse attn (compact)."""
        b = query.size(1)

        nvtx_range_push("compressed_indices")
        x_det = x.detach()
        qr_det = qr.detach()
        q_indexer, k_indexer, weights_indexer = self.indexer.forward_before_topk(
            x_det, qr_det, packed_seq_params
        )
        topk_indices_cmp, _ = indexer_topk(
            q_indexer,
            k_indexer,
            weights_indexer,
            self.indexer.index_topk,
            self.compress_ratio,
            indexer_softmax_scale=self.indexer.softmax_scale,
        )
        compress_topk_idxs = torch.where(topk_indices_cmp >= 0, topk_indices_cmp + offset, -1)
        flat_idxs, flat_tlen = build_flat_topk_idxs(
            window_idxs, compress_topk_idxs, batch_size=b, compact=True
        )
        nvtx_range_pop("compressed_indices")

        nvtx_range_push("sparse_attn_kernel")
        output = dsa_sparse_attn(
            query,
            kv_full,
            self.attn_sink.float(),
            flat_idxs,
            self.softmax_scale,
            topk_length=flat_tlen,
        )
        nvtx_range_pop("sparse_attn_kernel")
        return output

    def _forward_fused_indexer_training(
        self,
        query: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_full: torch.Tensor,
        n_compressed: int,
        offset: int,
        window_idxs: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Path B: fused indexer (with loss) + fused sparse attn.

        Returns ``(output, indexer_loss)``.
        """
        nvtx_range_push("compressed_indices")
        x_det = x.detach()
        qr_det = qr.detach()
        q_indexer, k_indexer, weights_indexer = self.indexer.forward_before_topk(
            x_det, qr_det, packed_seq_params
        )
        nvtx_range_pop("compressed_indices")

        indexer_loss_coeff = self.config.dsa_indexer_loss_coeff or 0.0

        nvtx_range_push("sparse_attn_kernel")
        output, indexer_loss = fused_indexer_sparse_attn(
            query,
            kv_full,
            self.attn_sink.float(),
            window_idxs,
            q_indexer,
            k_indexer,
            weights_indexer,
            self.indexer.index_topk,
            self.compress_ratio,
            self.softmax_scale,
            self.indexer.softmax_scale,
            indexer_loss_coeff,
            sparse_loss=getattr(self.config, "dsa_indexer_use_sparse_loss", True),
            kv_offset=offset,
            calculate_per_token_loss=self.config.calculate_per_token_loss,
        )
        nvtx_range_pop("sparse_attn_kernel")

        if indexer_loss_coeff > 0:
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=indexer_loss,
                layer_number=self.layer_number,
                num_layers=self.config.num_layers + (self.config.mtp_num_layers or 0),
            )
        return output, indexer_loss

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor = None,
        qr: torch.Tensor = None,
        boundary_hidden: Optional[torch.Tensor] = None,
        boundary_kv: Optional[torch.Tensor] = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ) -> torch.Tensor:
        """Forward pass for CompressedSparseAttention.

        Args:
            query:  [sq, b, np, v_head_dim]
            key:    [sq, b, 1, v_head_dim]  (single-head MQA; head dim squeezed internally)
            value:  unused (key == value in MQA)
            attention_mask: attention mask (may be None for causal).
            x:      [sq, b, hidden_size]  original hidden states.
            qr:     [sq, b, q_lora_rank]  compressed query representation.

        Returns:
            output: [sq, b, np * v_head_dim]
        """
        nvtx_range_push("compressed_sparse_attn")

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            if self.pg_collection.cp is not None and self.pg_collection.cp.size() > 1:
                output = self._forward_thd_cp(
                    query, key, x, qr, packed_seq_params, boundary_hidden, boundary_kv
                )
            else:
                output = self._forward_thd(query, key, x, qr, packed_seq_params)
            nvtx_range_pop("compressed_sparse_attn")
            return output

        sq, b, np, hn = query.size()

        kv = key.squeeze(-2)  # [sq, b, 1, v_head_dim] -> [sq, b, v_head_dim]
        kv_full, compressed_kv, n_compressed = self._build_kv_full(kv, x)
        offset = sq  # compressed indices start after original positions
        window_idxs = get_window_topk_idxs(self.window_size, b, sq, query.device)

        has_indexer_compressed = (
            self.compress_ratio > 1 and n_compressed > 0 and self.indexer is not None
        )

        indexer_loss = None

        if not self.apply_dsa_kernel_fusion:
            output, indexer_loss = self._forward_unfused_csa(
                query,
                x,
                qr,
                kv_full,
                compressed_kv,
                n_compressed,
                offset,
                window_idxs,
                packed_seq_params,
            )
        elif has_indexer_compressed and self.training and torch.is_grad_enabled():
            output, indexer_loss = self._forward_fused_indexer_training(
                query, x, qr, kv_full, n_compressed, offset, window_idxs, packed_seq_params
            )
        elif has_indexer_compressed:
            output = self._forward_fused_indexer_inference(
                query, x, qr, kv_full, n_compressed, offset, window_idxs, packed_seq_params
            )
        else:
            output = self._forward_fused_no_indexer(
                query, kv_full, n_compressed, offset, window_idxs
            )

        if indexer_loss is not None:
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        nvtx_range_pop("compressed_sparse_attn")
        return output

    # ------------------------------------------------------------------
    # THD per-path helpers (called from _forward_thd)
    # ------------------------------------------------------------------

    def _forward_unfused_csa_thd(
        self,
        query: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_full_thd: torch.Tensor,
        compressed_kv: Optional[torch.Tensor],
        n_compressed_total: int,
        np_: int,
        total_q: int,
        device: torch.device,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        cu_seqlens_kv_full: torch.Tensor,
        cu_seqlens_compressed: torch.Tensor,
        window_idxs: torch.Tensor,
        max_seqlen_q: int,
        packed_seq_params: PackedSeqParams,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """PyTorch unfused path for THD.

        Mirrors :meth:`_forward_unfused_csa` for the SBHD layout.
        Returns ``(output, indexer_loss)`` where *output* is
        ``(total_q, 1, np * hn)``.
        """
        indexer_loss = None

        if self.compress_ratio > 1 and n_compressed_total > 0:
            if self.indexer is not None:
                x_det = x.detach()
                qr_det = qr.detach()

                if self.training and torch.is_grad_enabled():
                    q_indexer, k_indexer, weights_indexer, cu_seqlens_compressed_idx = (
                        self.indexer.forward_before_topk(x_det, qr_det, packed_seq_params)
                    )
                    if k_indexer is None:
                        raise RuntimeError(
                            "CompressedSparseAttention THD unfused Path B requires "
                            "at least one segment with compressed indexer K."
                        )
                    q_thd = q_indexer.squeeze(1)
                    w_thd = weights_indexer.squeeze(1)
                    k_thd = k_indexer.squeeze(1)
                    max_seqlen_compressed_idx = max_seqlen_q // self.compress_ratio

                    key_for_loss_thd = compressed_kv.unsqueeze(1).expand(-1, np_, -1)
                    weights_for_unfused = w_thd * self.indexer.softmax_scale
                    indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)

                    # ``_forward_thd`` (caller) absorbs trailing padded
                    # tokens into the last ``cu_seqlens_q[-1]`` bucket
                    # so ``batch_of_row`` doesn't OOB; the Compressor
                    # does the same to ``cu_seqlens_compressed_idx[-1]``.
                    # Both are correct for the sparse-attention path,
                    # but the per-segment indexer-loss loop in
                    # ``fwd/bwd_fused_indexer_loss_naive_thd`` would
                    # then iterate a "fake" absorbed-padding segment
                    # with ``seqlen_k_b < topk`` — triggering a write
                    # shape mismatch and a downstream ``scatter_`` OOB
                    # on its ``-1`` entries.
                    #
                    # Restore the original (pre-absorption) cu_seqlens
                    # for the loss path so the segment loop's
                    # ``if seqlen_k_b == 0: continue`` guard skips the
                    # padding-only iteration. When no padding exists,
                    # ``packed_seq_params`` already equals the absorbed
                    # version and this is a no-op.
                    cu_seqlens_q_for_loss = packed_seq_params.cu_seqlens_q
                    seg_lens_q = cu_seqlens_q_for_loss[1:] - cu_seqlens_q_for_loss[:-1]
                    cu_seqlens_compressed_idx_for_loss = torch.cat(
                        [
                            torch.zeros(
                                1,
                                dtype=cu_seqlens_q_for_loss.dtype,
                                device=cu_seqlens_q_for_loss.device,
                            ),
                            (seg_lens_q // self.compress_ratio).cumsum(0).to(
                                cu_seqlens_q_for_loss.dtype
                            ),
                        ]
                    )
                    topk_indices_cmp, indexer_loss = FusedDSAIndexerLoss.apply(
                        q_thd,
                        weights_for_unfused,
                        k_thd,
                        query.detach(),
                        key_for_loss_thd.detach(),
                        self.softmax_scale,
                        min(self.indexer.index_topk, max_seqlen_compressed_idx),
                        indexer_loss_coeff,
                        None,
                        getattr(self.config, "dsa_indexer_use_sparse_loss", True),
                        self.indexer.pg_collection,
                        self.config.calculate_per_token_loss,
                        cu_seqlens_q_for_loss,
                        cu_seqlens_compressed_idx_for_loss,
                        self.compress_ratio,
                    )

                    if indexer_loss_coeff > 0:
                        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                            loss=indexer_loss,
                            layer_number=self.layer_number,
                            num_layers=self.config.num_layers + (self.config.mtp_num_layers or 0),
                        )
                else:
                    _, topk_indices_cmp = self.indexer(
                        x_det, qr_det, mask=None, packed_seq_params=packed_seq_params
                    )

                # Shift into per-segment full-KV index space.
                if topk_indices_cmp.shape[-1] > 0:
                    seq_lens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                    batch_of_token = batch_of_row(cu_seqlens_q, total_q=total_q)
                    offset_per_row = seq_lens_kv[batch_of_token].unsqueeze(1)
                    # Per-segment causal post-filter — mirrors the SBHD
                    # ``_forward_unfused_csa`` post-filter. The training
                    # indexer (``fwd_fused_indexer_loss_naive_thd``)
                    # returns RAW per-segment top-K ids without sentinel
                    # for non-causal picks, so a query at intra-segment
                    # position ``i`` (0-indexed) may select compressed
                    # indices ``>= (i+1)//ratio`` whose pre-mask scores
                    # were ``-inf``; treat those as ``-1`` so the sparse
                    # attention skips them.
                    pos_in_seg = (
                        torch.arange(total_q, device=device, dtype=cu_seqlens_q.dtype)
                        - cu_seqlens_q[batch_of_token]
                    )
                    n_valid_per_row = ((pos_in_seg + 1) // self.compress_ratio).unsqueeze(1)
                    causal_valid = topk_indices_cmp < n_valid_per_row
                    is_valid = (topk_indices_cmp >= 0) & causal_valid
                    compress_topk_idxs = torch.where(
                        is_valid,
                        topk_indices_cmp + offset_per_row,
                        torch.full_like(topk_indices_cmp, -1),
                    )
                else:
                    compress_topk_idxs = topk_indices_cmp
            else:
                compress_topk_idxs = get_compress_topk_idxs_thd(
                    self.compress_ratio,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    cu_seqlens_compressed,
                    device,
                    total_q=total_q,
                    max_n_compressed=max_seqlen_q // self.compress_ratio,
                )

            topk_idxs = torch.cat([window_idxs, compress_topk_idxs], dim=-1)
        else:
            topk_idxs = window_idxs

        topk_idxs = topk_idxs.int()

        flat_idxs, _ = build_flat_topk_idxs(
            topk_idxs, batch_size=-1, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv_full
        )

        output = unfused_compressed_sparse_attn(
            query, kv_full_thd, self.attn_sink.float(), flat_idxs, self.softmax_scale
        )
        return output.unsqueeze(1), indexer_loss

    def _forward_fused_no_indexer_thd(
        self,
        query: torch.Tensor,
        kv_full_thd: torch.Tensor,
        total_q: int,
        device: torch.device,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        cu_seqlens_kv_full: torch.Tensor,
        cu_seqlens_compressed: torch.Tensor,
        n_compressed_total: int,
        window_idxs: torch.Tensor,
        max_seqlen_q: int = 0,
    ) -> torch.Tensor:
        """Path A (THD): fused sparse attn with window or deterministic
        compressed indices.

        Returns ``(total_q, 1, np * hn)`` — the attention output.
        """
        if self.compress_ratio > 1 and n_compressed_total > 0:
            compress_topk_idxs = get_compress_topk_idxs_thd(
                self.compress_ratio,
                cu_seqlens_q,
                cu_seqlens_kv,
                cu_seqlens_compressed,
                device,
                total_q=total_q,
                max_n_compressed=max_seqlen_q // self.compress_ratio,
            )
            flat_idxs, _ = build_flat_topk_idxs(
                window_idxs,
                compress_topk_idxs,
                batch_size=-1,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv_full,
            )
        else:
            flat_idxs, _ = build_flat_topk_idxs(
                window_idxs,
                batch_size=-1,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv_full,
            )

        output = dsa_sparse_attn(
            query, kv_full_thd, self.attn_sink.float(), flat_idxs, self.softmax_scale, is_thd=True
        )
        return output.unsqueeze(1)

    def _forward_fused_indexer_inference_thd(
        self,
        query: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_full_thd: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        total_q: int,
        device: torch.device,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        cu_seqlens_kv_full: torch.Tensor,
        window_idxs: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kv: int,
    ) -> torch.Tensor:
        """Path C (THD): separate indexer forward (no loss) + fused sparse attn (compact).

        Returns ``(total_q, 1, np * hn)`` — the attention output.
        """
        x_det = x.detach()
        qr_det = qr.detach()

        q_indexer, k_indexer, weights_indexer, cu_seqlens_compressed_idx = (
            self.indexer.forward_before_topk(x_det, qr_det, packed_seq_params)
        )
        q_thd = q_indexer.squeeze(1)
        w_thd = weights_indexer.squeeze(1)
        if k_indexer is None:
            topk_indices_cmp = torch.full((total_q, 0), -1, dtype=torch.int32, device=device)
        else:
            k_thd = k_indexer.squeeze(1)
            max_seqlen_compressed_idx = max_seqlen_q // self.compress_ratio
            topk_indices_cmp, _ = indexer_topk(
                q_thd,
                k_thd,
                w_thd,
                topk=self.indexer.index_topk,
                ratio=self.compress_ratio,
                indexer_softmax_scale=self.indexer.softmax_scale,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_compressed_idx,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_compressed_idx,
            )

        # Shift into per-segment full-KV index space.
        if topk_indices_cmp.shape[-1] > 0:
            seq_lens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
            batch_of_token = batch_of_row(cu_seqlens_q, total_q=total_q)
            offset_per_row = seq_lens_kv[batch_of_token].unsqueeze(1)
            compress_topk_idxs = torch.where(
                topk_indices_cmp >= 0,
                topk_indices_cmp + offset_per_row,
                torch.full_like(topk_indices_cmp, -1),
            )
        else:
            compress_topk_idxs = topk_indices_cmp

        flat_idxs, flat_tlen = build_flat_topk_idxs(
            window_idxs,
            compress_topk_idxs,
            batch_size=-1,
            compact=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv_full,
        )
        output = dsa_sparse_attn(
            query,
            kv_full_thd,
            self.attn_sink.float(),
            flat_idxs,
            self.softmax_scale,
            topk_length=flat_tlen,
            is_thd=True,
        )
        return output.unsqueeze(1)

    def _forward_fused_indexer_training_thd(
        self,
        query: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        total_q: int,
        np_: int,
        device: torch.device,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        cu_seqlens_kv_full: torch.Tensor,
        max_seqlen_q: int,
        compressed_kv: torch.Tensor,
        kv_full_thd: torch.Tensor,
        window_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """Path B (THD): fused indexer (with loss) + fused sparse attn.

        Returns ``(output, indexer_loss)`` where *output* is
        ``(total_q, 1, np * hn)``.
        """
        sparse_loss = getattr(self.config, "dsa_indexer_use_sparse_loss", True)
        indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)

        x_det = x.detach()
        qr_det = qr.detach()
        q_indexer, k_indexer, weights_indexer, cu_seqlens_compressed_idx = (
            self.indexer.forward_before_topk(x_det, qr_det, packed_seq_params)
        )
        if k_indexer is None:
            raise RuntimeError(
                "CompressedSparseAttention THD Path B requires at least "
                "one segment with compressed indexer K; got none. (Should "
                "be unreachable when ``n_compressed_total > 0``.)"
            )

        q_thd = q_indexer.squeeze(1)
        w_thd = weights_indexer.squeeze(1)
        k_thd = k_indexer.squeeze(1)

        max_seqlen_compressed_idx = max_seqlen_q // self.compress_ratio

        output, indexer_loss = fused_indexer_sparse_attn(
            query,
            kv_full_thd,
            self.attn_sink.float(),
            window_idxs,
            q_thd,
            k_thd,
            w_thd,
            self.indexer.index_topk,
            self.compress_ratio,
            self.softmax_scale,
            self.indexer.softmax_scale,
            indexer_loss_coeff,
            sparse_loss=sparse_loss,
            kv_offset=0,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_kv_full=cu_seqlens_kv_full,
            cu_seqlens_compressed_idx=cu_seqlens_compressed_idx,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_compressed_idx=max_seqlen_compressed_idx,
            compressed_kv=compressed_kv,
            calculate_per_token_loss=self.config.calculate_per_token_loss,
        )

        if indexer_loss_coeff > 0:
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=indexer_loss,
                layer_number=self.layer_number,
                num_layers=self.config.num_layers + (self.config.mtp_num_layers or 0),
            )
        output = output.unsqueeze(1)
        return output, indexer_loss

    def _forward_thd_cp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        boundary_hidden: Optional[torch.Tensor],
        boundary_kv: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """THD-packed context-parallel branch.

        This path keeps each rank on its local THD rows, uses the boundary
        tensors supplied by the caller, all-gathers fixed-capacity compressed KV,
        builds the local full-KV layout, then dispatches to either the indexer
        loss path or sparse attention.
        """
        # ---- Step 1: CP metadata and THD shape contract ----------------------
        cp_group = self.pg_collection.cp
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()

        l_local = query.shape[0]
        cu_seqlens_q = (
            packed_seq_params.cu_seqlens_q_padded
            if packed_seq_params.cu_seqlens_q_padded is not None
            else packed_seq_params.cu_seqlens_q
        )
        cu_seqlens_kv = (
            packed_seq_params.cu_seqlens_kv_padded
            if packed_seq_params.cu_seqlens_kv_padded is not None
            else packed_seq_params.cu_seqlens_kv
        )
        max_seqlen_q = int(packed_seq_params.max_seqlen_q)
        if l_local != key.shape[0]:
            raise RuntimeError("DSv4 THD CP path currently supports self-attention only.")

        # ---- Step 2: local CP rows, local KV, and boundary tensors ------------
        chunk_ranges = cp_utils.local_q_cp_chunk_ranges(
            self.config.csa_cp_partition_mode, l_local, cp_size, cp_rank
        )
        kv_local = key.squeeze(-2).squeeze(1)
        if boundary_hidden is None or boundary_kv is None:
            raise RuntimeError(
                "DSv4 THD CP path requires boundary_hidden and boundary_kv from "
                "the hidden-only boundary exchange and boundary KV projection path."
            )
        boundary_kv = boundary_kv.squeeze(-2).squeeze(1)
        d_window = boundary_hidden.shape[0] // len(chunk_ranges)
        compressed_kv_rank_major = kv_local.new_empty((0, kv_local.shape[-1]))
        cu_seqlens_compressed = None
        seq_ids_rank_major = torch.empty((0,), dtype=torch.int32, device=query.device)
        comp_ids_rank_major = torch.empty((0,), dtype=torch.int32, device=query.device)
        valid_rank_major = torch.empty((0,), dtype=torch.bool, device=query.device)
        indexer_topk_logical = rank_major_by_seq_major = None
        ratio = self.compress_ratio
        indexer = self.indexer
        indexer_loss_coeff = self.config.dsa_indexer_loss_coeff or 0.0
        training_with_grad = self.training and torch.is_grad_enabled()
        if self.compressor is not None and ratio > 1:
            # ---- Step 3: build fixed-capacity compressor input ----------------
            d_comp = 8 if ratio == 4 else ratio
            # Global compressed prefix sums: sequence i contributes floor(seq_len_i / ratio).
            cu_seqlens_compressed = cp_utils.build_global_compressed_cu_seqlens(
                cu_seqlens_kv, ratio
            )
            compressed_seq_major_rows = (l_local * cp_size) // ratio
            # Group local/boundary hidden rows into fixed-capacity compressor input.
            hidden_compact, cu_compact, _, comp_ids_local, _ = (
                cp_utils.build_cp_compressor_prep_compact_fused(
                    x,
                    boundary_hidden,
                    cu_seqlens_kv,
                    chunk_ranges,
                    ratio,
                    d_comp,
                    d_window,
                )
            )
            # Metadata maps each rank-major compressed slot to sequence/group ids.
            seq_ids_rank_major, comp_ids_rank_major, valid_rank_major = (
                cp_utils.build_cp_rank_major_compressed_metadata_fused(
                    cu_seqlens_kv, chunk_ranges, cp_size, ratio, d_comp
                )
            )

            if indexer is not None:
                # ---- Step 4: optional indexer compressed path -----------------
                if (
                    training_with_grad
                    and indexer_loss_coeff > 0
                    and not self.config.dsa_indexer_use_sparse_loss
                ):
                    raise RuntimeError(
                        "DSv4 THD CP path currently supports sparse indexer loss only. "
                        "Dense CP-aware indexer loss needs a CP-aware dense score kernel."
                    )

                q_indexer_cp, weights_indexer_cp = indexer._forward_thd_query_weights_cp(
                    x.detach(),
                    qr.detach(),
                    cu_seqlens_q,
                    max_seqlen_q,
                    chunk_ranges=chunk_ranges,
                )

                indexer_compressed_local, _ = indexer.compressor._forward_thd(
                    hidden_compact.detach(),
                    cu_compact,
                    max_seqlen_q=max_seqlen_q,
                    rope_positions=comp_ids_local,
                )
                # Gather indexer compressed K in rank-major order.
                k_indexer_rank_major = cp_utils.all_gather_fixed_cp_tensor(
                    indexer_compressed_local.squeeze(1), cp_group
                )

                # Repack valid compressed K to seq-major order for top-k.
                k_indexer_seq_major, rank_major_by_seq_major = (
                    cp_utils.repack_rank_major_compressed_to_seq_major_fused(
                        k_indexer_rank_major,
                        seq_ids_rank_major,
                        comp_ids_rank_major,
                        valid_rank_major,
                        cu_seqlens_compressed,
                        seq_major_rows=compressed_seq_major_rows,
                    )
                )
                # Top-k is still in logical compressed coordinates here.
                indexer_topk_logical = cp_utils.compute_cp_indexer_topk_logical_fused(
                    q_indexer_cp,
                    weights_indexer_cp,
                    k_indexer_seq_major,
                    cu_seqlens_q,
                    cu_seqlens_compressed,
                    chunk_ranges,
                    ratio,
                    indexer.index_topk,
                    indexer.softmax_scale,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_q // ratio,
                )

            # ---- Step 5: attention compressed KV path -------------------------
            compressed_kv_local, _ = self.compressor._forward_thd(
                hidden_compact,
                cu_compact,
                max_seqlen_q=max_seqlen_q,
                rope_positions=comp_ids_local,
            )
            # Gather attention compressed KV in the same rank-major layout.
            compressed_kv_rank_major = cp_utils.all_gather_fixed_cp_tensor(
                compressed_kv_local.squeeze(1), cp_group
            )
            # Build seq-major -> rank-major map if the indexer did not need it.
            if len(chunk_ranges) == 2 and rank_major_by_seq_major is None:
                _, rank_major_by_seq_major = (
                    cp_utils.repack_rank_major_compressed_to_seq_major_fused(
                        compressed_kv_rank_major,
                        seq_ids_rank_major,
                        comp_ids_rank_major,
                        valid_rank_major,
                        cu_seqlens_compressed,
                        seq_major_rows=compressed_seq_major_rows,
                    )
                )

        # ---- Step 6: pack the local full-KV layout ---------------------------
        # Pack local window KV, boundary KV, compressed KV, and tail padding.
        kv_full_thd, shared_compressed_base = cp_utils.pack_cp_kv_full_fused(
            kv_local,
            boundary_kv,
            compressed_kv_rank_major,
            seq_ids_rank_major,
            comp_ids_rank_major,
            valid_rank_major,
            cu_seqlens_kv,
            chunk_ranges,
            d_window,
            ratio,
            rank_major_by_seq_major=rank_major_by_seq_major,
            cu_seqlens_compressed=cu_seqlens_compressed,
        )
        if training_with_grad and indexer_topk_logical is not None:
            # ---- Step 7a: indexer-loss path ----------------------------------
            # Lower logical top-k to local KV and rank-major indexer-loss spaces.
            topk_idxs, indexer_topk_rank_major = cp_utils.build_cp_indexer_loss_indices_fused(
                cu_seqlens_q,
                cu_seqlens_compressed,
                chunk_ranges,
                d_window,
                self.window_size,
                ratio,
                indexer_topk_logical,
                rank_major_by_seq_major,
                shared_compressed_base=shared_compressed_base,
            )
            output, indexer_loss = (
                FusedIndexerSparseAttnFromTopkFunc.apply
                if self.apply_dsa_kernel_fusion
                else _unfused_indexer_sparse_attn_from_topk
            )(
                query,
                kv_full_thd,
                self.attn_sink.float(),
                topk_idxs.int(),
                q_indexer_cp,
                k_indexer_rank_major,
                weights_indexer_cp,
                indexer_topk_rank_major.int(),
                compressed_kv_rank_major,
                self.softmax_scale,
                indexer.softmax_scale,
                indexer_loss_coeff,
                self.config.calculate_per_token_loss,
                l_local * cp_size,
            )
            if indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_layers + (self.config.mtp_num_layers or 0),
                    reduce_group=cp_group,
                )
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
            return output.unsqueeze(1)

        # ---- Step 7b: sparse attention path ----------------------------------
        # Build sparse-attention indices into ``kv_full_thd``.
        topk_idxs, topk_length = cp_utils.build_cp_attention_indices_fused(
            cu_seqlens_q,
            chunk_ranges,
            d_window,
            self.window_size,
            ratio,
            indexer_topk_logical,
            max_n_compressed=max_seqlen_q // ratio if ratio > 1 else 0,
            rank_major_by_seq_major=rank_major_by_seq_major,
            cu_seqlens_compressed=cu_seqlens_compressed,
            shared_compressed_base=shared_compressed_base,
        )

        if self.apply_dsa_kernel_fusion:
            output = dsa_sparse_attn(
                query,
                kv_full_thd,
                self.attn_sink.float(),
                topk_idxs.int(),
                self.softmax_scale,
                topk_length=topk_length,
                is_thd=True,
            )
        else:
            output = unfused_compressed_sparse_attn(
                query, kv_full_thd, self.attn_sink.float(), topk_idxs.int(), self.softmax_scale
            )
        return output.unsqueeze(1)

    def _forward_thd(
        self,
        query: torch.Tensor,  # (total_q, np, hn)        TE THD convention
        key: torch.Tensor,  # (total_kv, 1, 1, hn)     packed, MQA
        x: torch.Tensor,  # (total_q, 1, hidden_size)
        qr: torch.Tensor,  # (total_q, 1, q_lora_rank)
        packed_seq_params: PackedSeqParams,
    ) -> torch.Tensor:
        """THD-packed branch of :meth:`forward`. See class docstring for layout.

        Performs common setup (shape validation, per-segment compression,
        full-KV layout construction, window indices) then dispatches to
        one of three per-path helpers:

        * :meth:`_forward_fused_no_indexer_thd` — window-only / window + all-compressed.
        * :meth:`_forward_fused_indexer_training_thd` — training + indexer + loss (returns
          directly with attached indexer loss).
        * :meth:`_forward_fused_indexer_inference_thd` — inference + indexer (no loss).

        Paths A and C return ``compress_topk_idxs`` which are globalized
        and fed to the fused/unfused sparse attention in Step 5 below.
        """
        # ---- Inputs / shape contract ----------------------------------------
        # query    : (total_q, np, hn)        multi-head Q (TE THD convention)
        # key      : (total_kv, 1, 1, hn)     packed single-head MQA KV (the
        #            DSv4 hybrid adds a dummy batch dim to keep the MQA-head
        #            unsqueeze symmetric with SBHD)
        # x, qr    : (total_q, 1, *)
        total_q, _np, _ = query.shape
        device = query.device

        cu_seqlens_q = (
            packed_seq_params.cu_seqlens_q_padded
            if packed_seq_params.cu_seqlens_q_padded is not None
            else packed_seq_params.cu_seqlens_q
        )
        cu_seqlens_kv = (
            packed_seq_params.cu_seqlens_kv_padded
            if packed_seq_params.cu_seqlens_kv_padded is not None
            else packed_seq_params.cu_seqlens_kv
        )

        # After pad_thd_for_cuda_graph the token tensors are padded to a
        # fixed ``max_seqlen`` but cu_seqlens[-1] still equals actual_T.
        # Tokens in [actual_T, total_q) are orphans that belong to no
        # segment, which makes batch_of_row return OOB segment IDs.
        # Absorb them into the last cu_seqlens bucket so every token maps
        # to a valid segment.  Clone first — packed_seq_params is still
        # read by the Compressor and Indexer which must see the originals.
        cu_seqlens_q = cu_seqlens_q.clone()
        cu_seqlens_q[-1:].fill_(total_q)
        cu_seqlens_kv = cu_seqlens_kv.clone()
        cu_seqlens_kv[-1:].fill_(total_q)

        max_seqlen_q = int(packed_seq_params.max_seqlen_q)
        max_seqlen_kv = int(packed_seq_params.max_seqlen_kv)

        # Squeeze the dummy b=1 and MQA head-dim to get the KV-flat layout.
        # (key arrives as (total_kv, 1, 1, hn) for MQA.)
        kv_thd = key.squeeze(-2).squeeze(1)  # (total_kv, hn)

        # ---- Step 2: per-segment compression --------------------------------
        if self.compressor is not None and self.compress_ratio > 1:
            compressed_kv, cu_seqlens_compressed = self.compressor(
                x, packed_seq_params=packed_seq_params
            )
            # compressed_kv: (max_total_comp, 1, hn) — static shape,
            # padded by the Compressor.
            compressed_kv = compressed_kv.squeeze(1)  # (max_total_comp, hn)
            n_compressed_total = compressed_kv.shape[0]
        else:
            compressed_kv = None
            cu_seqlens_compressed = torch.zeros_like(cu_seqlens_kv)
            n_compressed_total = 0

        # ---- Build full per-segment-concatenated KV layout ------------------
        cu_seqlens_kv_full = build_cu_seqlens_kv_full(cu_seqlens_kv, cu_seqlens_compressed)
        kv_full_thd = cat_per_segment(
            kv_thd, compressed_kv, cu_seqlens_kv, cu_seqlens_compressed, cu_seqlens_kv_full
        )

        # ---- Step 3: window indices (per-segment local) ---------------------
        window_idxs = get_window_topk_idxs_thd(
            self.window_size, cu_seqlens_q, device, total_q=total_q
        )  # (total_q, win_topk) local-to-segment

        # ---- Step 4: path dispatch --------------------------------------------
        is_training = self.training and torch.is_grad_enabled()
        has_indexer = (
            self.compress_ratio > 1 and n_compressed_total > 0 and self.indexer is not None
        )

        indexer_loss = None

        if not self.apply_dsa_kernel_fusion:
            output, indexer_loss = self._forward_unfused_csa_thd(
                query,
                x,
                qr,
                kv_full_thd,
                compressed_kv,
                n_compressed_total,
                _np,
                total_q,
                device,
                cu_seqlens_q,
                cu_seqlens_kv,
                cu_seqlens_kv_full,
                cu_seqlens_compressed,
                window_idxs,
                max_seqlen_q,
                packed_seq_params,
            )
        elif has_indexer and is_training:
            output, indexer_loss = self._forward_fused_indexer_training_thd(
                query,
                x,
                qr,
                packed_seq_params,
                total_q,
                _np,
                device,
                cu_seqlens_q,
                cu_seqlens_kv,
                cu_seqlens_kv_full,
                max_seqlen_q,
                compressed_kv,
                kv_full_thd,
                window_idxs,
            )
        elif has_indexer:
            output = self._forward_fused_indexer_inference_thd(
                query,
                x,
                qr,
                kv_full_thd,
                packed_seq_params,
                total_q,
                device,
                cu_seqlens_q,
                cu_seqlens_kv,
                cu_seqlens_kv_full,
                window_idxs,
                max_seqlen_q,
                max_seqlen_kv,
            )
        else:
            output = self._forward_fused_no_indexer_thd(
                query,
                kv_full_thd,
                total_q,
                device,
                cu_seqlens_q,
                cu_seqlens_kv,
                cu_seqlens_kv_full,
                cu_seqlens_compressed,
                n_compressed_total,
                window_idxs,
                max_seqlen_q=max_seqlen_q,
            )

        if indexer_loss is not None:
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        return output
