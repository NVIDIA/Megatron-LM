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
) -> torch.Tensor:
    """Swap a CP-local tensor from Megatron zigzag layout to contiguous chunks.

    Zigzag CP layout assigns rank ``r`` global chunks ``[r, 2*cp-r-1]``.
    Linear-attention all-gather CP kernels expect rank ``r`` to hold chunks
    ``[2*r, 2*r+1]``. The conversion is a chunk-level all-to-all and preserves
    the local tensor shape.
    """
    return _zigzag_contiguous_chunk_swap(tensor, cp_group, seq_dim, to_contiguous=True)


def contiguous_to_zigzag_chunks(
    tensor: torch.Tensor,
    cp_group: dist.ProcessGroup | None,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Inverse of :func:`zigzag_to_contiguous_chunks`."""
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


__all__ = [
    "contiguous_position_ids_for_cp",
    "contiguous_slice_for_cp",
    "contiguous_to_zigzag_chunks",
    "split_packed_for_cp",
    "zigzag_to_contiguous_chunks",
    "zigzag_reconstruct_from_cp_parts",
    "zigzag_position_ids_for_cp",
    "zigzag_slice_for_cp",
    "zigzag_split_for_cp",
]
