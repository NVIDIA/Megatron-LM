# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Packed CSA layout and index metadata using device tensor operations.

All capacities come from input shapes. Real lengths remain device tensors; no
sequence-length dependent host synchronization is needed in the forward path.
"""

import torch


def _prefix(lengths):
    return torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))


def _visible_groups(cu, start, end, ratio, halo):
    first = torch.div(
        (start - halo - cu[:-1]).clamp_min(0) + ratio - 1, ratio, rounding_mode="floor"
    )
    stop = torch.div(
        torch.minimum(cu[1:], cu.new_tensor(end)) - cu[:-1], ratio, rounding_mode="floor"
    )
    counts = (stop - first).clamp_min(0)
    counts = torch.where((cu[:-1] < end) & (cu[1:] > start), counts, 0)
    return first, counts


def _compact_compressor_input(hidden, boundary, cu, global_start, ratio, halo, capacity, cp_size):
    """Gather complete groups and their ratio-4 predecessors; retain autograd edges."""
    local_rows = hidden.shape[0]
    first, counts = _visible_groups(cu, global_start, global_start + local_rows, ratio, halo)
    local_comp_cu = _prefix(counts)
    comp_cu = _prefix(torch.div(cu[1:] - cu[:-1], ratio, rounding_mode="floor"))
    rows = torch.arange(capacity, dtype=torch.int32, device=cu.device)
    seq = torch.bucketize(rows, local_comp_cu[1:], right=True).clamp_max(cu.numel() - 2)
    valid = rows < local_comp_cu[-1]
    group_ids = first[seq] + rows - local_comp_cu[seq]
    source_rows = cu[seq, None] + group_ids[:, None] * ratio + torch.arange(ratio, device=cu.device)
    source_rows = (source_rows - global_start).reshape(-1).long()
    values = hidden.index_select(0, source_rows.clamp(0, local_rows - 1))
    mask_shape = (capacity * ratio,) + (1,) * (hidden.ndim - 1)
    if boundary.shape[0]:
        halo_rows = (source_rows + boundary.shape[0]).clamp(0, boundary.shape[0] - 1)
        halo_values = boundary.index_select(0, halo_rows)
        values = torch.where((source_rows < 0).view(mask_shape), halo_values, values)
    gathered = values * valid.repeat_interleave(ratio).view(mask_shape)
    positions = torch.where(valid, group_ids * ratio, 0)
    group_ids = torch.where(valid, group_ids, -1)

    # A group's final source token determines its canonical owner. Halo copies
    # are used only for local pooling, never duplicated in the global KV domain.
    logical_rows = torch.arange(local_rows * cp_size // ratio, dtype=torch.int32, device=cu.device)
    sequence = torch.bucketize(logical_rows, comp_cu[1:], right=True).clamp_max(cu.numel() - 2)
    group = logical_rows - comp_cu[sequence]
    owner = torch.div(cu[sequence] + (group + 1) * ratio - 1, local_rows, rounding_mode="floor")
    owner = owner.clamp(0, cp_size - 1)
    # Compute rank/segment metadata once, then gather each group's owner directly.
    # Avoid CP_size full passes over every compressed row.
    rank_start = torch.arange(cp_size, device=cu.device, dtype=cu.dtype)[:, None] * local_rows
    rank_first = torch.div(
        (rank_start - halo - cu[:-1]).clamp_min(0) + ratio - 1, ratio, rounding_mode="floor"
    )
    rank_stop = torch.div(
        torch.minimum(cu[1:], rank_start + local_rows) - cu[:-1], ratio, rounding_mode="floor"
    )
    rank_counts = (rank_stop - rank_first).clamp_min(0)
    rank_counts = torch.where(
        (cu[:-1] < rank_start + local_rows) & (cu[1:] > rank_start), rank_counts, 0
    )
    rank_prefix = rank_counts.cumsum(dim=1, dtype=torch.int32) - rank_counts
    physical = owner * capacity + rank_prefix[owner, sequence] + group - rank_first[owner, sequence]
    row_map = torch.where(logical_rows < comp_cu[-1], physical, -1)
    return gathered, group_ids, positions, comp_cu, row_map


_compiled_compactor = torch.compile(_compact_compressor_input, fullgraph=True)


def compact_compressor_input(hidden, boundary, cu, global_start, ratio, halo, capacity, cp_size):
    """Gather packed groups with fused CUDA indexing/masking and an eager CPU oracle."""
    implementation = _compiled_compactor if hidden.is_cuda else _compact_compressor_input
    return implementation(hidden, boundary, cu, global_start, ratio, halo, capacity, cp_size)


def _build_attention_indices(
    cu_seqlens,
    global_start,
    l_local,
    d_window,
    window_size,
    ratio,
    compressed_width,
    compressed_topk=None,
    cu_seqlens_compressed=None,
    seq_to_rank_row=None,
    for_indexer_loss=False,
    compressed_base=None,
    compressed_rows=None,
    compressed_is_sequence_major=False,
    cu_seqlens_unpadded=None,
    output_alignment=1,
):
    """Lower document-relative keys to physical KV rows, masking real/padded boundaries."""
    cu = cu_seqlens
    rows = torch.arange(global_start, global_start + l_local, device=cu.device, dtype=torch.int32)
    seq = torch.bucketize(rows, cu[1:], right=True).clamp_max(cu.numel() - 2)
    valid = (rows >= cu[seq]) & (rows < cu[seq + 1])
    if cu_seqlens_unpadded is not None:
        real_lengths = cu_seqlens_unpadded[1:] - cu_seqlens_unpadded[:-1]
        valid = valid & (rows - cu[seq] < real_lengths[seq])
    padding_mask = ~valid if cu_seqlens_unpadded is not None else None
    window = torch.maximum(rows - window_size + 1, cu[seq])[:, None] + torch.arange(
        window_size, device=cu.device
    )
    window = torch.where(
        valid[:, None] & (window <= rows[:, None]), window - global_start + d_window, -1
    )
    if compressed_base is None:
        compressed_base = d_window + l_local
    physical = torch.empty((l_local, 0), device=cu.device, dtype=torch.int32)
    if compressed_width:
        local_ids = compressed_topk
        if local_ids is None:
            local_ids = torch.arange(compressed_width, device=cu.device).expand(l_local, -1)
        visible = torch.div(rows - cu[seq] + 1, ratio, rounding_mode="floor")
        count = cu_seqlens_compressed[seq + 1] - cu_seqlens_compressed[seq]
        selected = (
            valid[:, None] & (local_ids >= 0) & (local_ids < torch.minimum(visible, count)[:, None])
        )
        logical = local_ids + cu_seqlens_compressed[seq, None]
        if compressed_is_sequence_major:
            physical = logical
        elif seq_to_rank_row.numel():
            physical = seq_to_rank_row[logical.clamp(0, seq_to_rank_row.numel() - 1).long()]
        else:
            physical = torch.full_like(logical, -1)
        physical = torch.where(
            selected & (physical >= 0) & (physical < compressed_rows), physical, -1
        ).int()
    compressed = torch.where(physical >= 0, physical + compressed_base, -1)
    indices = torch.cat(
        (compressed, window) if for_indexer_loss else (window, compressed), dim=-1
    ).int()
    width = indices.shape[1]
    if output_alignment > 1:
        indices = torch.nn.functional.pad(indices, (0, -width % output_alignment), value=-1)
    lengths = None if for_indexer_loss else (indices >= 0).sum(-1).int()
    return indices, lengths, physical if for_indexer_loss else None, padding_mask


_compiled_attention_indices = torch.compile(_build_attention_indices, fullgraph=True)


def build_attention_indices(
    cu_seqlens,
    global_start,
    l_local,
    d_window,
    window_size,
    ratio,
    compressed_width,
    compressed_topk=None,
    cu_seqlens_compressed=None,
    seq_to_rank_row=None,
    for_indexer_loss=False,
    compressed_base=None,
    compressed_rows=None,
    compressed_is_sequence_major=False,
    cu_seqlens_unpadded=None,
    output_alignment=1,
):
    """Lower document-relative keys without materializing each CUDA pointwise intermediate."""
    implementation = _compiled_attention_indices if cu_seqlens.is_cuda else _build_attention_indices
    return implementation(
        cu_seqlens,
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        compressed_topk=compressed_topk,
        cu_seqlens_compressed=cu_seqlens_compressed,
        seq_to_rank_row=seq_to_rank_row,
        for_indexer_loss=for_indexer_loss,
        compressed_base=compressed_base,
        compressed_rows=compressed_rows,
        compressed_is_sequence_major=compressed_is_sequence_major,
        cu_seqlens_unpadded=cu_seqlens_unpadded,
        output_alignment=output_alignment,
    )


def build_seq_lens(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    total_q: int,
    ratio: int,
    q_causal_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build each query row's visible compressed-key count, including its CP offset."""
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    row_idx = torch.arange(total_q, device=cu_seqlens_q.device, dtype=torch.int32)
    row_batch_ids = torch.bucketize(row_idx, cu_seqlens_q[1:], right=True).clamp(
        max=cu_seqlens_q.shape[0] - 2
    )
    row_valid = row_idx < cu_seqlens_q[-1]
    pos_in_seq = row_idx - cu_seqlens_q[row_batch_ids]
    if q_causal_offsets is not None:
        pos_in_seq = pos_in_seq + q_causal_offsets[row_batch_ids]
    pos_in_seq = torch.where(row_valid, pos_in_seq, torch.zeros_like(pos_in_seq))
    seqlen_kv_per_row = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1])[row_batch_ids]
    seq_lens = ((pos_in_seq + 1) // ratio).clamp(max=seqlen_kv_per_row).to(torch.int32).contiguous()
    return torch.where(row_valid, seq_lens, torch.zeros_like(seq_lens))


def _sanitize_topk(
    candidate_indices: torch.Tensor,
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    output_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask invalid/padded keys and align the selected rows for attention kernels."""
    if scores.shape[1] == 0:
        raise ValueError("scores must contain at least one column")
    if output_width is not None and output_width < candidate_indices.shape[1]:
        raise ValueError("output_width cannot be smaller than the candidate width")
    if output_width is not None and output_width > candidate_indices.shape[1]:
        padding = torch.full(
            (candidate_indices.shape[0], output_width - candidate_indices.shape[1]),
            -1,
            dtype=candidate_indices.dtype,
            device=candidate_indices.device,
        )
        candidate_indices = torch.cat((candidate_indices, padding), dim=-1)
    score_width = scores.shape[1]
    row_valid = (candidate_indices >= 0) & (candidate_indices < seq_lens.unsqueeze(1))
    sanitized_indices = candidate_indices.masked_fill(~row_valid, -1)
    safe_topk = sanitized_indices.clamp(min=0, max=score_width - 1).to(torch.long)
    selected_scores = torch.gather(scores, dim=-1, index=safe_topk)
    selected_valid = (
        (sanitized_indices >= 0)
        & (sanitized_indices < score_width)
        & torch.isfinite(selected_scores)
    )
    sanitized_indices = sanitized_indices.masked_fill(~selected_valid, -1)
    return sanitized_indices, (sanitized_indices >= 0).sum(dim=-1).int()


_compiled_sanitize_topk = torch.compile(_sanitize_topk, fullgraph=True)


def sanitize_topk(candidate_indices, scores, seq_lens, output_width=None):
    """Sanitize selected keys with fused CUDA masks/gathers or the CPU reference path."""
    implementation = _compiled_sanitize_topk if scores.is_cuda else _sanitize_topk
    return implementation(candidate_indices, scores, seq_lens, output_width)
