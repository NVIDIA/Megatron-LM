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


def compact_compressor_input(hidden, boundary, cu, global_start, ratio, halo, capacity, cp_size):
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
    source_rows = source_rows - global_start + boundary.shape[0]
    inputs = torch.cat((boundary, hidden), dim=0)
    safe_rows = source_rows.clamp(0, inputs.shape[0] - 1).long()
    gathered = inputs.index_select(0, safe_rows.flatten()).reshape(
        capacity * ratio, *hidden.shape[1:]
    )
    mask_shape = (capacity * ratio,) + (1,) * (hidden.ndim - 1)
    gathered = gathered * valid.repeat_interleave(ratio).view(mask_shape)
    positions = torch.where(valid, group_ids * ratio, 0)
    group_ids = torch.where(valid, group_ids, -1)

    # A group's final source token determines its canonical owner. Halo copies
    # are used only for local pooling, never duplicated in the global KV domain.
    logical_rows = torch.arange(local_rows * cp_size // ratio, dtype=torch.int32, device=cu.device)
    sequence = torch.bucketize(logical_rows, comp_cu[1:], right=True).clamp_max(cu.numel() - 2)
    group = logical_rows - comp_cu[sequence]
    owner = torch.div(cu[sequence] + (group + 1) * ratio - 1, local_rows, rounding_mode="floor")
    owner = owner.clamp(0, cp_size - 1)
    row_map = torch.full_like(logical_rows, -1)
    for rank in range(cp_size):
        rank_first, rank_counts = _visible_groups(
            cu, rank * local_rows, (rank + 1) * local_rows, ratio, halo
        )
        rank_cu = _prefix(rank_counts)
        physical = rank * capacity + rank_cu[sequence] + group - rank_first[sequence]
        row_map = torch.where((owner == rank) & (logical_rows < comp_cu[-1]), physical, row_map)
    return gathered, group_ids, positions, comp_cu, row_map


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


def sanitize_topk(
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
