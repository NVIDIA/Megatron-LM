# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Context parallel tensor layout helpers."""

from typing import List, Optional, Tuple

import torch

from megatron.core.tensor_parallel import all_to_all


def get_thd_context_parallel_rank_indices(
    cu_seqlens: torch.Tensor, cp_size: int, cp_rank: int, layout: str
) -> torch.Tensor:
    """Return global THD token indices owned by one CP rank in a layout.

    Args:
        cu_seqlens: Global packed-sequence cumulative lengths before CP partitioning.
        cp_size: Context-parallel group size.
        cp_rank: Context-parallel rank.
        layout: Either ``"zigzag"`` or ``"contiguous"``.

    The returned indices are ordered exactly as the rank-local THD tensor is stored:
    each packed sequence contributes its two local chunks in rank-local slot order.
    """
    if layout not in ("zigzag", "contiguous"):
        raise ValueError(f"Unsupported context-parallel layout {layout!r}.")
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}.")
    if cu_seqlens.dim() != 1:
        raise ValueError(f"cu_seqlens must be 1-D, got shape {tuple(cu_seqlens.shape)}.")

    cu = cu_seqlens.to(dtype=torch.long)
    if cu.numel() == 0 or cu[0].item() != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {cu_seqlens}.")

    seq_lens = torch.diff(cu)
    chunk_divisor = 2 * cp_size
    if torch.any(seq_lens <= 0):
        raise ValueError(f"All packed sequence lengths must be positive, got {seq_lens}.")
    if torch.any(seq_lens % chunk_divisor != 0):
        raise ValueError(
            "All packed sequence lengths must be divisible by "
            f"2 * cp_size ({chunk_divisor}) for zigzag/contiguous CP layout conversion, "
            f"got {seq_lens}."
        )

    total_tokens = int(cu[-1].item())
    positions = torch.arange(total_tokens, device=cu.device, dtype=torch.long)
    if total_tokens == 0:
        return positions

    seq_idx = torch.bucketize(positions, cu[1:], right=True)
    global_starts = cu[:-1]
    pos_in_seq = positions - global_starts[seq_idx]
    chunk_lens = (seq_lens // chunk_divisor)[seq_idx]
    chunk = pos_in_seq // chunk_lens
    offset = pos_in_seq - chunk * chunk_lens

    if layout == "zigzag":
        owner = torch.where(chunk < cp_size, chunk, 2 * cp_size - chunk - 1)
        local_slot = torch.where(chunk < cp_size, torch.zeros_like(chunk), torch.ones_like(chunk))
    else:
        owner = chunk // 2
        local_slot = chunk % 2

    local_starts = (global_starts // cp_size)[seq_idx]
    local_pos = local_starts + local_slot * chunk_lens + offset

    rank_mask = owner == cp_rank
    rank_positions = positions[rank_mask]
    rank_local_pos = local_pos[rank_mask]
    return rank_positions[torch.argsort(rank_local_pos)]


def zigzag_to_contiguous_chunks(
    x: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    seq_dim: int = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Permute CP chunks from Megatron zigzag layout to contiguous-time layout.

    SBHD tensors have two equal chunks per rank along ``seq_dim`` and use a
    chunk-level all-to-all. THD tensors pass global ``cu_seqlens`` and use one
    packed-token all-to-all over the whole local THD tensor.
    """
    if cu_seqlens is not None:
        return _zigzag_contiguous_thd_swap(
            x, cp_group, seq_dim, cu_seqlens, source_layout="zigzag", target_layout="contiguous"
        )
    return _zigzag_contiguous_chunk_swap(x, cp_group, seq_dim, to_contiguous=True)


def contiguous_to_zigzag_chunks(
    x: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    seq_dim: int = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inverse of :func:`zigzag_to_contiguous_chunks`."""
    if cu_seqlens is not None:
        return _zigzag_contiguous_thd_swap(
            x, cp_group, seq_dim, cu_seqlens, source_layout="contiguous", target_layout="zigzag"
        )
    return _zigzag_contiguous_chunk_swap(x, cp_group, seq_dim, to_contiguous=False)


def _zigzag_contiguous_thd_swap(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    seq_dim: int,
    cu_seqlens: torch.Tensor,
    source_layout: str,
    target_layout: str,
) -> torch.Tensor:
    """Single-all-to-all THD permutation between zigzag and contiguous layouts.

    The packed THD tensor stays packed: we first group local tokens by their
    target CP rank, exchange those groups once, then scatter received tokens
    back into the target rank-local order.
    """
    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return x
    cp_rank = cp_group.rank()

    if seq_dim != 0:
        x = x.movedim(seq_dim, 0)
    x = x.contiguous()

    cu = cu_seqlens.to(device=x.device, dtype=torch.long)
    source_by_rank = [
        get_thd_context_parallel_rank_indices(cu, cp_size, rank, source_layout)
        for rank in range(cp_size)
    ]
    target_by_rank = [
        get_thd_context_parallel_rank_indices(cu, cp_size, rank, target_layout)
        for rank in range(cp_size)
    ]

    local_source_indices = source_by_rank[cp_rank]
    local_target_indices = target_by_rank[cp_rank]
    if x.size(0) != local_source_indices.numel():
        raise ValueError(
            f"Local THD tensor length ({x.size(0)}) does not match {source_layout} "
            f"rank-{cp_rank} partition length ({local_source_indices.numel()})."
        )

    total_tokens = int(cu[-1].item())
    target_owner = torch.empty(total_tokens, device=x.device, dtype=torch.long)
    target_local_pos = torch.empty(total_tokens, device=x.device, dtype=torch.long)
    for rank, indices in enumerate(target_by_rank):
        target_owner[indices] = rank
        target_local_pos[indices] = torch.arange(indices.numel(), device=x.device)

    local_target_owner = target_owner[local_source_indices]
    local_target_pos = target_local_pos[local_source_indices]

    send_parts: List[torch.Tensor] = []
    input_split_sizes: List[int] = []
    for dst_rank in range(cp_size):
        dst_mask = local_target_owner == dst_rank
        dst_rows = dst_mask.nonzero(as_tuple=False).flatten()
        if dst_rows.numel() > 0:
            dst_rows = dst_rows[torch.argsort(local_target_pos[dst_rows])]
            send_part = x.index_select(0, dst_rows)
        else:
            send_part = x.narrow(0, 0, 0)
        send_parts.append(send_part)
        input_split_sizes.append(send_part.size(0))
    send_buf = torch.cat(send_parts, dim=0).contiguous()

    output_split_sizes: List[int] = []
    recv_target_positions: List[torch.Tensor] = []
    for src_rank in range(cp_size):
        src_indices = source_by_rank[src_rank]
        src_to_this_rank = target_owner[src_indices] == cp_rank
        recv_global_indices = src_indices[src_to_this_rank]
        if recv_global_indices.numel() > 0:
            recv_positions = target_local_pos[recv_global_indices]
            recv_positions = recv_positions[torch.argsort(recv_positions)]
        else:
            recv_positions = local_target_indices.narrow(0, 0, 0)
        recv_target_positions.append(recv_positions)
        output_split_sizes.append(recv_positions.numel())

    recv_buf = all_to_all(cp_group, send_buf, output_split_sizes, input_split_sizes)

    out_shape = (local_target_indices.numel(),) + tuple(x.shape[1:])
    out = x.new_empty(out_shape)
    offset = 0
    for recv_positions in recv_target_positions:
        recv_len = recv_positions.numel()
        if recv_len > 0:
            out[recv_positions] = recv_buf[offset : offset + recv_len]
            offset += recv_len

    if seq_dim != 0:
        out = out.movedim(0, seq_dim)
    return out.contiguous()


def _zigzag_contiguous_chunk_swap(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    seq_dim: int,
    to_contiguous: bool,
) -> torch.Tensor:
    """Single-all-to-all chunk permutation between zigzag and contiguous layouts.

    Each rank holds exactly two chunks along ``seq_dim``. The mapping from
    local (rank, slot) to (rank, slot) in the target layout is deterministic
    and depends only on ``cp_size`` and ``cp_rank``, so we pack send data in
    destination-rank order and use one ``all_to_all_single`` with unequal
    splits to route each chunk to its target rank.
    """
    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return x
    cp_rank = cp_group.rank()

    # Work with seq_dim at position 0.
    if seq_dim != 0:
        x = x.movedim(seq_dim, 0)
    x = x.contiguous()

    seq_len_local = x.size(0)
    assert seq_len_local % 2 == 0, (
        f"zigzag/contiguous chunk swap requires an even local sequence length, "
        f"got {seq_len_local}."
    )
    chunk_len = seq_len_local // 2

    def _rank_to_chunks(rank: int, in_zigzag: bool) -> Tuple[int, int]:
        """Global chunk indices at (slot 0, slot 1) for this rank."""
        if in_zigzag:
            return (rank, 2 * cp_size - rank - 1)
        return (2 * rank, 2 * rank + 1)

    def _chunk_to_dest(chunk_idx: int, target_zigzag: bool) -> Tuple[int, int]:
        """Destination (rank, slot) for a given global chunk index in the target layout."""
        if target_zigzag:
            if chunk_idx < cp_size:
                return chunk_idx, 0
            return 2 * cp_size - chunk_idx - 1, 1
        return chunk_idx // 2, chunk_idx % 2

    source_in_zigzag = to_contiguous
    target_in_zigzag = not to_contiguous

    local_chunk_indices = _rank_to_chunks(cp_rank, source_in_zigzag)
    local_dests = [_chunk_to_dest(c, target_in_zigzag) for c in local_chunk_indices]

    # Pack the send buffer so chunks are ordered by (dst_rank, dst_slot).
    local_slot_order = sorted(range(2), key=lambda s: local_dests[s])
    local_chunks = [x[:chunk_len], x[chunk_len:]]
    send_buf = torch.cat([local_chunks[s] for s in local_slot_order], dim=0).contiguous()

    input_split_chunks = [0] * cp_size
    for dst_rank, _ in local_dests:
        input_split_chunks[dst_rank] += 1

    # Mirror every source rank's packing logic so we know which received chunk
    # belongs in which local target slot.
    output_split_chunks = [0] * cp_size
    recv_dst_slots_per_source: List[List[int]] = [[] for _ in range(cp_size)]
    for src in range(cp_size):
        src_chunks = _rank_to_chunks(src, source_in_zigzag)
        src_dests = [_chunk_to_dest(c, target_in_zigzag) for c in src_chunks]
        src_slot_order = sorted(range(2), key=lambda s: src_dests[s])
        for s in src_slot_order:
            dst_rank, dst_slot = src_dests[s]
            if dst_rank == cp_rank:
                output_split_chunks[src] += 1
                recv_dst_slots_per_source[src].append(dst_slot)

    input_split_sizes = [n * chunk_len for n in input_split_chunks]
    output_split_sizes = [n * chunk_len for n in output_split_chunks]

    recv_buf = all_to_all(cp_group, send_buf, output_split_sizes, input_split_sizes)

    # Reassemble local chunks in target-layout slot order.
    target_slots: List[Optional[torch.Tensor]] = [None, None]
    offset = 0
    for src in range(cp_size):
        for dst_slot in recv_dst_slots_per_source[src]:
            target_slots[dst_slot] = recv_buf[offset : offset + chunk_len]
            offset += chunk_len
    assert all(t is not None for t in target_slots), "Incomplete chunk reassembly in CP swap"

    out = torch.cat(target_slots, dim=0)
    if seq_dim != 0:
        out = out.movedim(0, seq_dim)
    return out.contiguous()
