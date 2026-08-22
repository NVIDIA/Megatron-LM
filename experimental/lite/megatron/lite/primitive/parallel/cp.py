"""Context parallel zigzag sequence splitting helpers."""

from __future__ import annotations

import torch


def zigzag_split_for_cp(
    tensor: torch.Tensor, cp_rank: int, cp_size: int, seq_dim: int = 1,
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
    assert seq_len % (2 * cp_size) == 0, (
        f"seq_len={seq_len} must be divisible by 2*cp_size={2 * cp_size}"
    )
    shape = list(tensor.shape)
    shape[seq_dim:seq_dim + 1] = [2 * cp_size, seq_len // (2 * cp_size)]
    tensor = tensor.view(*shape)
    idx = torch.tensor(
        [cp_rank, 2 * cp_size - cp_rank - 1],
        dtype=torch.long, device=tensor.device,
    )
    tensor = tensor.index_select(seq_dim, idx)
    shape[seq_dim:seq_dim + 2] = [seq_len // cp_size]
    return tensor.reshape(*shape)


def zigzag_position_ids_for_cp(
    seq_len: int, cp_rank: int, cp_size: int, device: torch.device,
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
        assert seq_len % (2 * cp_size) == 0, (
            f"Sample {i} length {seq_len} not divisible by 2*cp_size={2 * cp_size}"
        )
        chunk = seq_len // (2 * cp_size)
        c1 = start + cp_rank * chunk
        c2 = start + (2 * cp_size - cp_rank - 1) * chunk
        ids_parts.append(input_ids[c1:c1 + chunk])
        ids_parts.append(input_ids[c2:c2 + chunk])
        pos_parts.append(position_ids[c1:c1 + chunk])
        pos_parts.append(position_ids[c2:c2 + chunk])
        new_lengths.append(2 * chunk)

    new_ids = torch.cat(ids_parts)
    new_pos = torch.cat(pos_parts)
    lens = torch.tensor(new_lengths, dtype=torch.int32, device=cu_seqlens.device)
    new_cu = torch.zeros(num_seqs + 1, dtype=torch.int32, device=cu_seqlens.device)
    torch.cumsum(lens, dim=0, out=new_cu[1:])
    return new_ids, new_pos, new_cu, max(new_lengths)


__all__ = [
    "split_packed_for_cp",
    "zigzag_position_ids_for_cp",
    "zigzag_split_for_cp",
]
