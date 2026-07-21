# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Context parallel zigzag sequence splitting helpers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


def zigzag_split_for_cp(
    tensor: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Split tensor along sequence dim using zigzag (striped) pattern for CP.

    Splits into 2*cp_size chunks; GPU i gets chunk[i] + chunk[2*cp_size-1-i].
    This balances causal-mask workload across CP ranks.

    Example (CP=2, seq=8): chunks [0,1,2,3] ->
        GPU0: chunk[0]+chunk[3] = tokens [0,1,6,7]
        GPU1: chunk[1]+chunk[2] = tokens [2,3,4,5]
    """
    if cp_size <= 1:
        return tensor
    seq_len = tensor.shape[seq_dim]
    assert (
        seq_len % (2 * cp_size) == 0
    ), f"seq_len={seq_len} must be divisible by 2*cp_size={2 * cp_size}"
    shape = list(tensor.shape)
    shape[seq_dim : seq_dim + 1] = [2 * cp_size, seq_len // (2 * cp_size)]
    tensor = tensor.view(*shape)
    idx = torch.tensor(
        [cp_rank, 2 * cp_size - cp_rank - 1],
        dtype=torch.long,
        device=tensor.device,
    )
    tensor = tensor.index_select(seq_dim, idx)
    shape[seq_dim : seq_dim + 2] = [seq_len // cp_size]
    return tensor.reshape(*shape)


def zigzag_reconstruct_from_cp_parts(
    parts: list[torch.Tensor] | tuple[torch.Tensor, ...], seq_dim: int = 1
) -> torch.Tensor:
    """Reconstruct a full sequence from per-rank zigzag CP shards."""
    cp_size = len(parts)
    if cp_size <= 1:
        return parts[0]
    local_len = parts[0].shape[seq_dim]
    assert (
        local_len % 2 == 0
    ), f"local seq_len={local_len} must be divisible by 2 for zigzag CP reconstruction"
    for idx, part in enumerate(parts):
        assert (
            part.shape == parts[0].shape
        ), f"CP part {idx} shape {tuple(part.shape)} != {tuple(parts[0].shape)}"

    chunk = local_len // 2
    full_len = local_len * cp_size
    out_shape = list(parts[0].shape)
    out_shape[seq_dim] = full_len
    full = torch.zeros(out_shape, dtype=parts[0].dtype, device=parts[0].device)
    for rank, part in enumerate(parts):
        first = part.narrow(seq_dim, 0, chunk)
        second = part.narrow(seq_dim, chunk, chunk)
        full.narrow(seq_dim, rank * chunk, chunk).copy_(first)
        full.narrow(seq_dim, full_len - (rank + 1) * chunk, chunk).copy_(second)
    return full


def zigzag_slice_for_cp(
    tensor: torch.Tensor, cp_rank: int, cp_size: int, seq_dim: int = 1
) -> torch.Tensor:
    """Return one rank's zigzag CP shard from a full sequence tensor."""
    if cp_size <= 1:
        return tensor
    seq_len = tensor.shape[seq_dim]
    assert (
        seq_len % (2 * cp_size) == 0
    ), f"seq_len={seq_len} must be divisible by 2*cp_size={2 * cp_size}"
    chunk = seq_len // (2 * cp_size)
    first = tensor.narrow(seq_dim, cp_rank * chunk, chunk)
    second_start = seq_len - (cp_rank + 1) * chunk
    second = tensor.narrow(seq_dim, second_start, chunk)
    return torch.cat((first, second), dim=seq_dim).contiguous()


def contiguous_slice_for_cp(
    tensor: torch.Tensor, cp_rank: int, cp_size: int, seq_dim: int = 1
) -> torch.Tensor:
    """Return one rank's contiguous CP shard from a full sequence tensor."""
    if cp_size <= 1:
        return tensor
    seq_len = tensor.shape[seq_dim]
    if seq_len % cp_size != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by cp_size={cp_size}")
    local_len = seq_len // cp_size
    return tensor.narrow(seq_dim, cp_rank * local_len, local_len).contiguous()


def contiguous_position_ids_for_cp(
    seq_len: int,
    cp_rank: int,
    cp_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return global position IDs for this CP rank under contiguous splitting."""
    if cp_size <= 1:
        return torch.arange(seq_len, device=device).unsqueeze(0)
    if seq_len % cp_size != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by cp_size={cp_size}")
    local_len = seq_len // cp_size
    start = cp_rank * local_len
    return torch.arange(start, start + local_len, device=device).unsqueeze(0)


def local_position_ids_for_cp(position_ids, *, batch, local_seq_len, cp_rank, cp_size):
    """Validate and contiguous-slice full-length position_ids to this CP rank."""
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.dim() != 2:
        raise ValueError("position_ids must have shape (S,) or (B, S).")
    if position_ids.size(0) == 1 and batch > 1:
        position_ids = position_ids.expand(batch, -1)
    if position_ids.size(0) != batch:
        raise ValueError(
            f"position_ids batch={position_ids.size(0)} does not match input batch={batch}."
        )
    if cp_size <= 1 or position_ids.size(1) == local_seq_len:
        return position_ids

    full_seq_len = local_seq_len * cp_size
    if position_ids.size(1) != full_seq_len:
        raise ValueError(
            "CP expects position_ids to be either CP-local or full-length; "
            f"got {position_ids.size(1)} for local_seq_len={local_seq_len}, cp={cp_size}."
        )
    return contiguous_slice_for_cp(position_ids, cp_rank, cp_size, seq_dim=1)


def local_sequence_tensor_for_cp(
    tensor,
    *,
    local_seq_len,
    cp_rank,
    cp_size,
    seq_dim=1,
    name: str = "tensor",
    unsqueeze_1d: bool = True,
):
    """Validate and contiguous-slice a full-length sequence tensor to this CP rank."""
    if tensor is None or cp_size <= 1:
        return tensor
    if unsqueeze_1d and tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    full_seq_len = local_seq_len * cp_size
    seq_len = tensor.size(seq_dim)
    if seq_len == local_seq_len:
        return tensor
    if seq_len != full_seq_len:
        raise ValueError(
            f"CP expects {name} to be either CP-local or full-length; "
            f"got {seq_len} for local_seq_len={local_seq_len}, cp={cp_size}."
        )
    return contiguous_slice_for_cp(tensor, cp_rank, cp_size, seq_dim=seq_dim)


def zigzag_to_contiguous_chunks(
    tensor: torch.Tensor,
    cp_group: dist.ProcessGroup | None,
    seq_dim: int = 1,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Swap a CP-local tensor from Megatron zigzag layout to contiguous chunks.

    Zigzag CP layout assigns rank ``r`` global chunks ``[r, 2*cp-r-1]``.
    Linear-attention all-gather CP kernels expect rank ``r`` to hold chunks
    ``[2*r, 2*r+1]``.

    SBHD tensors (``cu_seqlens is None``) hold exactly two equal chunks per rank
    and use a chunk-level all-to-all that preserves the local tensor shape. Packed
    THD tensors pass the *global* (pre-CP-split) ``cu_seqlens`` and use a single
    packed-token all-to-all whose routing is packing-aware, so per-sequence zigzag
    chunk boundaries are honoured even when sequences do not evenly divide the CP
    size. Mirrors upstream Megatron ``context_parallel_layout``.
    """
    if cu_seqlens is not None:
        return _zigzag_contiguous_thd_swap(
            tensor, cp_group, seq_dim, cu_seqlens, source_layout="zigzag", target_layout="contiguous"
        )
    return _zigzag_contiguous_chunk_swap(tensor, cp_group, seq_dim, to_contiguous=True)


def contiguous_to_zigzag_chunks(
    tensor: torch.Tensor,
    cp_group: dist.ProcessGroup | None,
    seq_dim: int = 1,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inverse of :func:`zigzag_to_contiguous_chunks`."""
    if cu_seqlens is not None:
        return _zigzag_contiguous_thd_swap(
            tensor, cp_group, seq_dim, cu_seqlens, source_layout="contiguous", target_layout="zigzag"
        )
    return _zigzag_contiguous_chunk_swap(tensor, cp_group, seq_dim, to_contiguous=False)


def _zigzag_contiguous_chunk_swap(
    tensor: torch.Tensor,
    cp_group: Optional[dist.ProcessGroup],
    seq_dim: int,
    *,
    to_contiguous: bool,
) -> torch.Tensor:
    cp_size = dist.get_world_size(cp_group) if cp_group is not None else 1
    if cp_size <= 1:
        return tensor
    cp_rank = dist.get_rank(cp_group)

    if seq_dim != 0:
        tensor = tensor.movedim(seq_dim, 0)
    tensor = tensor.contiguous()

    local_len = tensor.size(0)
    if local_len % 2 != 0:
        raise ValueError(
            f"zigzag/contiguous CP chunk swap requires even local sequence length, got {local_len}."
        )
    chunk_len = local_len // 2

    def rank_to_chunks(rank: int, in_zigzag: bool) -> tuple[int, int]:
        if in_zigzag:
            return rank, 2 * cp_size - rank - 1
        return 2 * rank, 2 * rank + 1

    def chunk_to_dest(chunk_idx: int, target_zigzag: bool) -> tuple[int, int]:
        if target_zigzag:
            if chunk_idx < cp_size:
                return chunk_idx, 0
            return 2 * cp_size - chunk_idx - 1, 1
        return chunk_idx // 2, chunk_idx % 2

    source_in_zigzag = to_contiguous
    target_in_zigzag = not to_contiguous
    local_chunks = [tensor[:chunk_len], tensor[chunk_len:]]
    local_chunk_indices = rank_to_chunks(cp_rank, source_in_zigzag)
    local_dests = [chunk_to_dest(chunk_idx, target_in_zigzag) for chunk_idx in local_chunk_indices]
    local_slot_order = sorted(range(2), key=lambda slot: local_dests[slot])
    send_buf = torch.cat([local_chunks[slot] for slot in local_slot_order], dim=0).contiguous()

    input_split_chunks = [0] * cp_size
    for dst_rank, _dst_slot in local_dests:
        input_split_chunks[dst_rank] += 1

    output_split_chunks = [0] * cp_size
    recv_dst_slots_per_source: list[list[int]] = [[] for _ in range(cp_size)]
    for src_rank in range(cp_size):
        src_chunks = rank_to_chunks(src_rank, source_in_zigzag)
        src_dests = [chunk_to_dest(chunk_idx, target_in_zigzag) for chunk_idx in src_chunks]
        src_slot_order = sorted(range(2), key=lambda slot: src_dests[slot])
        for slot in src_slot_order:
            dst_rank, dst_slot = src_dests[slot]
            if dst_rank == cp_rank:
                output_split_chunks[src_rank] += 1
                recv_dst_slots_per_source[src_rank].append(dst_slot)

    input_split_sizes = [count * chunk_len for count in input_split_chunks]
    output_split_sizes = [count * chunk_len for count in output_split_chunks]
    recv_shape = (sum(output_split_sizes), *send_buf.shape[1:])
    recv_buf = torch.empty(recv_shape, dtype=send_buf.dtype, device=send_buf.device)
    from torch.distributed.nn.functional import all_to_all_single

    recv_buf = all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=cp_group,
    )

    target_slots: list[torch.Tensor | None] = [None, None]
    offset = 0
    for src_rank in range(cp_size):
        for dst_slot in recv_dst_slots_per_source[src_rank]:
            target_slots[dst_slot] = recv_buf[offset : offset + chunk_len]
            offset += chunk_len
    if any(slot is None for slot in target_slots):
        raise RuntimeError("Incomplete CP chunk reassembly.")

    out = torch.cat([slot for slot in target_slots if slot is not None], dim=0)
    if seq_dim != 0:
        out = out.movedim(0, seq_dim)
    return out.contiguous()


def get_thd_context_parallel_rank_indices(
    cu_seqlens: torch.Tensor, cp_size: int, cp_rank: int, layout: str
) -> torch.Tensor:
    """Return global THD token indices owned by one CP rank in a layout.

    Args:
        cu_seqlens: Global packed-sequence cumulative lengths before CP partitioning.
        cp_size: Context-parallel group size.
        cp_rank: Context-parallel rank.
        layout: Either ``"zigzag"`` or ``"contiguous"``.

    The returned indices are ordered exactly as the rank-local THD tensor is stored.
    ``"zigzag"`` follows Megatron's per-sequence load-balanced chunk order; ``"contiguous"``
    partitions the flattened packed THD buffer into rank-contiguous spans. Mirrors upstream
    Megatron ``context_parallel_layout.get_thd_context_parallel_rank_indices``.
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

    if torch.any(torch.diff(cu) < 0):
        raise ValueError(f"cu_seqlens must be nondecreasing, got {cu_seqlens}.")

    nonduplicate_boundaries = torch.ones(cu.numel(), device=cu.device, dtype=torch.bool)
    nonduplicate_boundaries[1:] = cu[1:] != cu[:-1]
    cu = cu[nonduplicate_boundaries]

    total_tokens = int(cu[-1].item())
    positions = torch.arange(total_tokens, device=cu.device, dtype=torch.long)
    if total_tokens == 0:
        return positions

    seq_lens = torch.diff(cu)
    chunk_divisor = 2 * cp_size
    if torch.any(seq_lens % chunk_divisor != 0):
        raise ValueError(
            "All packed sequence lengths must be divisible by "
            f"2 * cp_size ({chunk_divisor}) for zigzag/contiguous CP layout conversion, "
            f"got {seq_lens}."
        )

    if layout == "contiguous":
        part_len = total_tokens // cp_size
        rank_start = cp_rank * part_len
        return positions[rank_start : rank_start + part_len]

    seq_idx = torch.bucketize(positions, cu[1:], right=True)
    global_starts = cu[:-1]
    pos_in_seq = positions - global_starts[seq_idx]
    chunk_lens = (seq_lens // chunk_divisor)[seq_idx]
    chunk = pos_in_seq // chunk_lens
    offset = pos_in_seq - chunk * chunk_lens

    owner = torch.where(chunk < cp_size, chunk, 2 * cp_size - chunk - 1)
    local_slot = torch.where(chunk < cp_size, torch.zeros_like(chunk), torch.ones_like(chunk))

    local_starts = (global_starts // cp_size)[seq_idx]
    local_pos = local_starts + local_slot * chunk_lens + offset

    rank_mask = owner == cp_rank
    rank_positions = positions[rank_mask]
    rank_local_pos = local_pos[rank_mask]
    return rank_positions[torch.argsort(rank_local_pos)]


def _zigzag_contiguous_thd_swap(
    tensor: torch.Tensor,
    cp_group: Optional[dist.ProcessGroup],
    seq_dim: int,
    cu_seqlens: torch.Tensor,
    *,
    source_layout: str,
    target_layout: str,
) -> torch.Tensor:
    """Single-all-to-all THD permutation between zigzag and contiguous layouts.

    The packed THD tensor stays packed: we first group local tokens by their target
    CP rank, exchange those groups once, then scatter received tokens back into the
    target rank-local order. Mirrors upstream Megatron
    ``context_parallel_layout._zigzag_contiguous_thd_swap`` (packing-aware routing
    from the *global* ``cu_seqlens``).
    """
    cp_size = dist.get_world_size(cp_group) if cp_group is not None else 1
    if cp_size <= 1:
        return tensor
    cp_rank = dist.get_rank(cp_group)

    if seq_dim != 0:
        tensor = tensor.movedim(seq_dim, 0)
    tensor = tensor.contiguous()

    cu = cu_seqlens.to(device=tensor.device, dtype=torch.long)
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
    if tensor.size(0) != local_source_indices.numel():
        raise ValueError(
            f"Local THD tensor length ({tensor.size(0)}) does not match {source_layout} "
            f"rank-{cp_rank} partition length ({local_source_indices.numel()})."
        )

    total_tokens = int(cu[-1].item())
    target_owner = torch.empty(total_tokens, device=tensor.device, dtype=torch.long)
    target_local_pos = torch.empty(total_tokens, device=tensor.device, dtype=torch.long)
    for rank, indices in enumerate(target_by_rank):
        target_owner[indices] = rank
        target_local_pos[indices] = torch.arange(indices.numel(), device=tensor.device)

    local_target_owner = target_owner[local_source_indices]
    local_target_pos = target_local_pos[local_source_indices]

    send_parts: list[torch.Tensor] = []
    input_split_sizes: list[int] = []
    for dst_rank in range(cp_size):
        dst_mask = local_target_owner == dst_rank
        dst_rows = dst_mask.nonzero(as_tuple=False).flatten()
        if dst_rows.numel() > 0:
            dst_rows = dst_rows[torch.argsort(local_target_pos[dst_rows])]
            send_part = tensor.index_select(0, dst_rows)
        else:
            send_part = tensor.narrow(0, 0, 0)
        send_parts.append(send_part)
        input_split_sizes.append(send_part.size(0))
    send_buf = torch.cat(send_parts, dim=0).contiguous()

    output_split_sizes: list[int] = []
    recv_target_positions: list[torch.Tensor] = []
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
        output_split_sizes.append(int(recv_positions.numel()))

    recv_shape = (sum(output_split_sizes), *send_buf.shape[1:])
    recv_buf = torch.empty(recv_shape, dtype=send_buf.dtype, device=send_buf.device)
    from torch.distributed.nn.functional import all_to_all_single

    recv_buf = all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=cp_group,
    )

    out_shape = (local_target_indices.numel(),) + tuple(tensor.shape[1:])
    out = tensor.new_empty(out_shape)
    offset = 0
    for recv_positions in recv_target_positions:
        recv_len = int(recv_positions.numel())
        if recv_len > 0:
            out[recv_positions] = recv_buf[offset : offset + recv_len]
            offset += recv_len

    if seq_dim != 0:
        out = out.movedim(0, seq_dim)
    return out.contiguous()


def zigzag_position_ids_for_cp(
    seq_len: int,
    cp_rank: int,
    cp_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return global position IDs for this CP rank under zigzag splitting.

    Returns shape [1, seq_len // cp_size] matching batch dim convention.
    """
    if cp_size <= 1:
        return torch.arange(seq_len, device=device).unsqueeze(0)
    chunk = seq_len // (2 * cp_size)
    first = torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk, device=device)
    second_start = (2 * cp_size - cp_rank - 1) * chunk
    second = torch.arange(second_start, second_start + chunk, device=device)
    return torch.cat([first, second]).unsqueeze(0)


def split_packed_for_cp(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    cp_rank: int,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Zigzag-split a packed sequence batch for context parallelism.

    Each sample defined by *cu_seqlens* is split with the same zigzag
    (striped) pattern as :func:`zigzag_split_for_cp`: tokens are divided
    into ``2 * cp_size`` chunks, and this rank keeps
    ``chunk[cp_rank] + chunk[2*cp_size - 1 - cp_rank]``.

    Returns:
        ``(input_ids, position_ids, cu_seqlens, max_seqlen)`` for this CP rank.
    """
    if cp_size <= 1:
        return input_ids, position_ids, cu_seqlens, max_seqlen

    num_seqs = cu_seqlens.size(0) - 1
    ids_parts: list[torch.Tensor] = []
    pos_parts: list[torch.Tensor] = []
    new_lengths: list[int] = []

    for i in range(num_seqs):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        seq_len = end - start
        assert (
            seq_len % (2 * cp_size) == 0
        ), f"Sample {i} length {seq_len} not divisible by 2*cp_size={2 * cp_size}"
        chunk = seq_len // (2 * cp_size)
        c1 = start + cp_rank * chunk
        c2 = start + (2 * cp_size - cp_rank - 1) * chunk
        ids_parts.append(input_ids[c1 : c1 + chunk])
        ids_parts.append(input_ids[c2 : c2 + chunk])
        pos_parts.append(position_ids[c1 : c1 + chunk])
        pos_parts.append(position_ids[c2 : c2 + chunk])
        new_lengths.append(2 * chunk)

    new_ids = torch.cat(ids_parts)
    new_pos = torch.cat(pos_parts)
    lens = torch.tensor(new_lengths, dtype=torch.int32, device=cu_seqlens.device)
    new_cu = torch.zeros(num_seqs + 1, dtype=torch.int32, device=cu_seqlens.device)
    torch.cumsum(lens, dim=0, out=new_cu[1:])
    return new_ids, new_pos, new_cu, max(new_lengths)


def build_headwise_section_perm(
    split_sections: tuple[int, ...] | list[int],
    cp_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Permutation over the hidden dim so a single contiguous ``1/cp`` slice equals
    the concatenation of each section's ``k``-th contiguous block.

    ``qkvzba`` is a concatenation of independently-headed sections (q, k, v, z,
    beta, alpha). Head-parallel CP wants each rank to own ``1/cp`` heads of *every*
    section. Applying this permutation before a plain hidden-scatter all-to-all makes
    rank ``k``'s contiguous hidden shard hold section-block ``k`` of each section, so
    the per-section split downstream (and the matching parameter slicing) is exact.

    Mirrors upstream Megatron ``_build_head_perm_for_split_sections``.
    """
    assert all(
        s % cp_size == 0 for s in split_sections
    ), f"split_sections {tuple(split_sections)} must each be divisible by cp_size={cp_size}."
    offset = 0
    parts = []
    for s in split_sections:
        parts.append(
            torch.arange(offset, offset + s, device=device, dtype=torch.long).view(cp_size, -1)
        )
        offset += s
    return torch.cat(parts, dim=-1).view(-1)


def get_parameter_local_cp_headwise(
    param: torch.Tensor,
    dim: int,
    cp_size: int,
    cp_rank: int,
    split_sections: Optional[list[int]] = None,
) -> torch.Tensor:
    """Return this CP rank's head-wise slice of a parameter.

    With ``split_sections`` the parameter is first split into sections along ``dim``,
    each section is sliced to its rank-``cp_rank`` contiguous block, and the blocks are
    re-concatenated — matching :func:`build_headwise_section_perm`. Slicing happens in
    the forward path so gradients flow back to the full (cp_size=1) parameter.

    Mirrors upstream Megatron ``get_parameter_local_cp_headwise``.
    """
    if cp_size <= 1:
        return param
    if split_sections is not None:
        pieces = torch.split(param, split_sections, dim=dim)
        return torch.cat(
            [get_parameter_local_cp_headwise(p, dim, cp_size, cp_rank) for p in pieces],
            dim=dim,
        )
    dim_size = param.size(dim)
    assert (
        dim_size % cp_size == 0
    ), f"parameter dim {dim} size {dim_size} not divisible by cp_size={cp_size}."
    slices: list[slice] = [slice(None)] * param.dim()
    block = dim_size // cp_size
    slices[dim] = slice(cp_rank * block, (cp_rank + 1) * block)
    return param[tuple(slices)]


def all_to_all_hidden_shards(
    send_parts: list[torch.Tensor],
    cp_group: dist.ProcessGroup | None,
) -> list[torch.Tensor]:
    """Even all-to-all over a CP group: ``send_parts[j]`` is delivered to rank ``j``;
    the returned ``recv[i]`` is the tensor rank ``i`` sent to this rank.

    All parts must share the same shape and dtype. Autograd-aware via
    ``torch.distributed.nn.functional.all_to_all_single``.
    """
    cp_size = dist.get_world_size(cp_group) if cp_group is not None else 1
    if cp_size <= 1:
        return [send_parts[0]]
    assert len(send_parts) == cp_size, (len(send_parts), cp_size)
    ref = send_parts[0]
    send = torch.stack([p.contiguous() for p in send_parts], dim=0)  # [cp, *shape]
    flat = send.reshape(cp_size, -1).contiguous()
    from torch.distributed.nn.functional import all_to_all_single

    recv_flat = all_to_all_single(torch.empty_like(flat), flat, group=cp_group)
    recv = recv_flat.reshape(cp_size, *ref.shape)
    return [recv[i] for i in range(cp_size)]


__all__ = [
    "all_to_all_hidden_shards",
    "build_headwise_section_perm",
    "contiguous_position_ids_for_cp",
    "contiguous_slice_for_cp",
    "contiguous_to_zigzag_chunks",
    "get_parameter_local_cp_headwise",
    "get_thd_context_parallel_rank_indices",
    "split_packed_for_cp",
    "zigzag_to_contiguous_chunks",
    "zigzag_reconstruct_from_cp_parts",
    "zigzag_position_ids_for_cp",
    "zigzag_slice_for_cp",
    "zigzag_split_for_cp",
]
