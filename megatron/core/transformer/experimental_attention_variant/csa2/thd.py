# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed-sequence (THD) helpers for DeepSeek-V4.1 CSA2.

A packed microbatch is one flat row axis holding several sequences ("segments") back to
back, described by cumulative lengths ``cu_seqlens`` (``[B + 1]``). Everything positional
in CSA2 is defined *within* a segment: sliding windows, compression groups, causal
visibility of compressed entries, candidate blocks and RoPE positions never cross a segment
boundary. These helpers translate between flat rows and (segment, position) pairs and build
the flat index tensors the reference sparse attention consumes.

Under context parallelism the flat axis is split into contiguous blocks; a rank owns rows
``[global_start, global_start + local_rows)`` of the global packed layout. All helpers take
``global_start`` so the same code serves CP 1 and CP > 1.

Framework free (plain PyTorch, CPU capable). Independent implementation.
"""

from typing import NamedTuple, Optional, Tuple

import torch


class RowMetadata(NamedTuple):
    """Segment bookkeeping for a contiguous interval of packed rows."""

    segment_ids: torch.Tensor  # [rows] int64, clamped for padding rows
    positions: torch.Tensor  # [rows] int64 position within the segment (0 for padding rows)
    valid: torch.Tensor  # [rows] bool, False for padding rows (inside or after the segments)


def packed_layout(packed_seq_params) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(starts, lengths)`` of the segments of a packed microbatch.

    ``starts`` (``[B + 1]`` int64) are the *physical* cumulative offsets, i.e.
    ``cu_seqlens_q_padded`` when the pack is padded, and ``lengths`` (``[B]`` int64) are the
    *valid* token counts from ``cu_seqlens_q``. Rows ``starts[b] + [lengths[b], starts[b+1])``
    are padding: they belong to no segment, are never compressed or hashed, attend nothing and
    are attended by nothing.
    """
    cu_valid = packed_seq_params.cu_seqlens_q.to(torch.int64)
    cu_phys = packed_seq_params.cu_seqlens_q_padded
    cu_phys = cu_valid if cu_phys is None else cu_phys.to(torch.int64)
    lengths = cu_valid[1:] - cu_valid[:-1]
    return cu_phys, lengths


def row_metadata(
    cu_seqlens: torch.Tensor,
    local_rows: int,
    global_start: int = 0,
    seq_lens: Optional[torch.Tensor] = None,
) -> RowMetadata:
    """Segment id, in-segment position and validity of rows ``global_start + [0, local_rows)``.

    ``cu_seqlens`` are the physical segment starts; ``seq_lens`` (``[B]``) the valid lengths
    when they differ from the physical ones (padded packs), see :func:`packed_layout`.
    """
    cu = cu_seqlens.to(torch.int64)
    rows = torch.arange(
        global_start, global_start + local_rows, device=cu.device, dtype=torch.int64
    )
    n_segments = cu.numel() - 1
    segment_ids = torch.bucketize(rows, cu[1:], right=True).clamp_max(n_segments - 1)
    starts = cu[segment_ids]
    if seq_lens is None:
        ends = cu[segment_ids + 1]
    else:
        ends = starts + seq_lens.to(torch.int64)[segment_ids]
    valid = (rows >= starts) & (rows < ends)
    positions = torch.where(valid, rows - starts, torch.zeros_like(rows))
    return RowMetadata(segment_ids, positions, valid)


def compressed_cu_seqlens(
    cu_seqlens: torch.Tensor, ratio: int, seq_lens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Cumulative compressed lengths: every segment contributes ``valid_len // ratio`` entries."""
    cu = cu_seqlens.to(torch.int64)
    lens = (cu[1:] - cu[:-1]) if seq_lens is None else seq_lens.to(torch.int64)
    return torch.cat([cu.new_zeros(1), (lens // ratio).cumsum(0)])


def compressed_entry_metadata(cu_seqlens: torch.Tensor, cu_comp: torch.Tensor, ratio: int):
    """For every compressed entry (sequence-major): segment id, group index and the flat row
    of its first token in the packed layout.

    Returns ``(segment_ids [n], group_ids [n], first_rows [n])``; group ``j`` of segment ``b``
    stands for RoPE position ``j * ratio`` and covers rows ``cu_seqlens[b] + j*ratio + [0, ratio)``.
    """
    cu = cu_seqlens.to(torch.int64)
    cu_comp = cu_comp.to(torch.int64)
    n_comp = int(cu_comp[-1])
    entries = torch.arange(n_comp, device=cu.device, dtype=torch.int64)
    segment_ids = torch.bucketize(entries, cu_comp[1:], right=True).clamp_max(cu.numel() - 2)
    group_ids = entries - cu_comp[segment_ids]
    first_rows = cu[segment_ids] + group_ids * ratio
    return segment_ids, group_ids, first_rows


def compressor_group_rows(first_rows: torch.Tensor, ratio: int, row_base: int = 0) -> torch.Tensor:
    """``[n_comp, ratio]`` indices of the rows pooled into each compressed entry.

    ``row_base`` shifts global flat rows into the index space of the buffer the caller pools
    from (e.g. a local buffer that starts at ``global_start - halo``).
    """
    offsets = torch.arange(ratio, device=first_rows.device, dtype=torch.int64)
    return first_rows.unsqueeze(1) + offsets.unsqueeze(0) - row_base


def owned_compressed_entries(
    first_rows: torch.Tensor, ratio: int, global_start: int, local_rows: int
) -> torch.Tensor:
    """Bool mask of the compressed entries whose *last* token lies in the local row block.

    The rank holding a group's last token owns it: with a left halo of at least ``ratio - 1``
    rows every owner can pool its groups, and each group is produced exactly once across CP.
    """
    last_rows = first_rows + ratio - 1
    return (last_rows >= global_start) & (last_rows < global_start + local_rows)


def window_indices_thd(
    meta: RowMetadata, window: int, local_row_base: int, halo: int
) -> torch.Tensor:
    """Flat sliding-window key indices for local rows into ``[halo | local]`` key rows.

    Local row ``i`` (global ``global_start + i``) attends keys ``i - k`` for ``k in [0, window)``
    as long as the key stays inside the same segment (``positions[i] - k >= 0``). The key buffer
    holds ``halo`` rows preceding the block, so local row ``i`` sits at ``halo + i`` and the
    key at offset ``-k`` sits at ``halo + i - k``; a key further back than the halo is ``-1``.

    Args:
        meta: metadata of the local rows.
        window: window size.
        local_row_base: unused placeholder for symmetry (kept 0); rows are local indices.
        halo: number of preceding rows available in the key buffer (0 at CP 1 / rank 0).
    """
    rows = torch.arange(meta.positions.numel(), device=meta.positions.device, dtype=torch.int64)
    offsets = torch.arange(window, device=rows.device, dtype=torch.int64)
    key_rows = rows.unsqueeze(1) - offsets.unsqueeze(0)  # local row of the key
    in_segment = offsets.unsqueeze(0) <= meta.positions.unsqueeze(1)
    available = key_rows >= -halo
    ok = in_segment & available & meta.valid.unsqueeze(1)
    idx = key_rows + halo - local_row_base
    return torch.where(ok, idx, torch.full_like(idx, -1)).to(torch.int32)


def visible_compressed_counts(meta: RowMetadata, cu_comp: torch.Tensor, ratio: int) -> torch.Tensor:
    """Per local row: number of compressed entries of its segment it may attend."""
    cu_comp = cu_comp.to(torch.int64)
    seg_len = cu_comp[meta.segment_ids + 1] - cu_comp[meta.segment_ids]
    counts = torch.minimum((meta.positions + 1) // ratio, seg_len)
    return torch.where(meta.valid, counts, torch.zeros_like(counts))


def segment_rows(meta: RowMetadata, segment: int) -> torch.Tensor:
    """Local row indices belonging to ``segment`` (valid rows only)."""
    return torch.nonzero((meta.segment_ids == segment) & meta.valid, as_tuple=False).squeeze(1)


def shift_compressed_indices(
    local_indices: torch.Tensor, segment_comp_start: int, compressed_base: int
) -> torch.Tensor:
    """Map segment-local compressed ids to flat key indices; ``-1`` stays ``-1``.

    ``compressed_base`` is where the compressed region starts in the key buffer
    (``halo + local_rows``), ``segment_comp_start`` is ``cu_comp[b]``.
    """
    shifted = local_indices.to(torch.int64) + segment_comp_start + compressed_base
    return torch.where(local_indices >= 0, shifted, torch.full_like(shifted, -1)).to(torch.int32)
