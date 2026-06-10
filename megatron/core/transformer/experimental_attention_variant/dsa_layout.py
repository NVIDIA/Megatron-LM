# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layout helpers for DeepSeek sparse attention."""

from typing import Optional, Tuple

import torch

from megatron.core.packed_seq_params import PackedSeqParams

__all__ = [
    "build_packed_allgather_cp_local_positions",
    "build_packed_allgather_cp_query_positions_and_key_reorder",
    "build_zigzag_allgather_cp_key_reorder",
    "build_zigzag_cp_local_positions",
    "ensure_sbhd",
    "extract_query_positions_from_position_ids",
    "get_cp_positions_from_layout",
    "get_packed_qk_cu_seqlens",
    "normalize_cp_comm_type",
]


def normalize_cp_comm_type(cp_comm_type: Optional[str]) -> str:
    """Normalize CP communication type to a canonical lowercase form."""
    if cp_comm_type is None:
        return "p2p"
    return cp_comm_type.replace("_", "").lower()


def ensure_sbhd(tensor: torch.Tensor, name: str) -> Tuple[torch.Tensor, bool]:
    """Ensure tensor is [s, b, h, d], allowing packed [t, h, d] input."""
    if tensor.ndim == 4:
        return tensor, False
    if tensor.ndim == 3:
        return tensor.unsqueeze(1), True
    raise ValueError(f"{name} must be 3D ([t,h,d]) or 4D ([s,b,h,d]), got {tensor.ndim}D")


def build_zigzag_cp_local_positions(
    seq_len: int, cp_size: int, cp_rank: int, device: torch.device
) -> torch.Tensor:
    """Build this CP rank's token positions under MCore zigzag sequence sharding."""
    if cp_size <= 1:
        return torch.arange(seq_len, device=device, dtype=torch.int64)
    if seq_len % (2 * cp_size) != 0:
        raise ValueError(
            "Zigzag CP expects the global sequence length to be divisible by 2 * cp_size, got "
            f"seq_len={seq_len}, cp_size={cp_size}"
        )

    chunk_len = seq_len // (2 * cp_size)
    front_chunk = cp_rank
    back_chunk = 2 * cp_size - cp_rank - 1
    return torch.cat(
        (
            torch.arange(
                front_chunk * chunk_len,
                (front_chunk + 1) * chunk_len,
                device=device,
                dtype=torch.int64,
            ),
            torch.arange(
                back_chunk * chunk_len,
                (back_chunk + 1) * chunk_len,
                device=device,
                dtype=torch.int64,
            ),
        ),
        dim=0,
    )


def build_zigzag_allgather_cp_key_reorder(
    sq: int, cp_size: int, device: torch.device
) -> torch.Tensor:
    """Build gathered-KV reorder index for non-packed zigzag allgather CP."""
    global_seq_len = sq * cp_size
    gathered_key_positions = torch.cat(
        [
            build_zigzag_cp_local_positions(global_seq_len, cp_size, rank, device)
            for rank in range(cp_size)
        ],
        dim=0,
    )
    return torch.argsort(gathered_key_positions)


def get_cp_positions_from_layout(
    sq: int,
    skv: int,
    cp_size: int,
    cp_rank: int,
    cp_comm_type: Optional[str],
    device: torch.device,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Infer query/key global token positions under CP allgather layout."""
    if cp_size <= 1:
        query_pos = torch.arange(sq, device=device, dtype=torch.int64)
        key_pos = torch.arange(skv, device=device, dtype=torch.int64)
        return query_pos, key_pos

    if normalize_cp_comm_type(cp_comm_type) != "allgather":
        raise NotImplementedError(
            "DSAttention context parallelism currently supports cp_comm_type=allgather only."
        )

    if skv == sq * cp_size:
        query_pos = build_zigzag_cp_local_positions(skv, cp_size, cp_rank, device)
        key_pos = torch.arange(skv, device=device, dtype=torch.int64)
        return query_pos, key_pos

    # Fallback for callers that pass uneven per-rank lengths. The non-packed MCore
    # dataloader uses zigzag layout, so the uniform case above is the expected path.
    query_offset = cp_rank * sq
    if (
        cp_group is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and cp_group.size() == cp_size
    ):
        local_len = torch.tensor([sq], device=device, dtype=torch.int64)
        all_lens = [torch.empty_like(local_len) for _ in range(cp_size)]
        torch.distributed.all_gather(all_lens, local_len, group=cp_group)
        query_offset = int(torch.stack(all_lens[:cp_rank]).sum().item()) if cp_rank > 0 else 0

    query_pos = torch.arange(sq, device=device, dtype=torch.int64) + query_offset
    key_pos = torch.arange(skv, device=device, dtype=torch.int64)
    return query_pos, key_pos


def build_packed_allgather_cp_local_positions(
    cu_seqlens: torch.Tensor, cp_size: int, cp_rank: int, device: torch.device
) -> torch.Tensor:
    """Build local packed-token positions for one CP rank under zigzag THD sharding.

    This mirrors the packed THD CP layout used by the surrounding training stack:
    each packed sequence is padded to a multiple of ``2 * cp_size`` and each rank
    receives the rank-local front chunk followed by the mirrored back chunk.
    """
    cu_seqlens_i64 = cu_seqlens.to(dtype=torch.int64)
    if cp_size <= 1:
        return torch.arange(int(cu_seqlens_i64[-1].item()), dtype=torch.int64, device=device)

    positions = []
    cu_seqlens_list = cu_seqlens_i64.tolist()
    for seq_start, seq_end in zip(cu_seqlens_list[:-1], cu_seqlens_list[1:]):
        seq_len = seq_end - seq_start
        if seq_len == 0:
            continue
        if seq_len % cp_size != 0:
            raise ValueError(
                "Packed DSA CP expects per-sequence padded lengths divisible by cp_size, got "
                f"seq_len={seq_len}, cp_size={cp_size}"
            )
        local_seq_len = seq_len // cp_size
        if local_seq_len % 2 != 0:
            raise ValueError(
                "Packed DSA CP expects per-rank packed sequence lengths divisible by 2, got "
                f"local_seq_len={local_seq_len}, seq_len={seq_len}, cp_size={cp_size}"
            )
        half_seq_len = local_seq_len // 2
        if half_seq_len == 0:
            continue

        front_start = seq_start + cp_rank * half_seq_len
        back_end = seq_end - cp_rank * half_seq_len
        back_start = back_end - half_seq_len

        positions.append(
            torch.arange(front_start, front_start + half_seq_len, dtype=torch.int64, device=device)
        )
        positions.append(torch.arange(back_start, back_end, dtype=torch.int64, device=device))

    if not positions:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.cat(positions, dim=0)


def build_packed_allgather_cp_query_positions_and_key_reorder(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build packed-query positions and gathered-KV reorder index for allgather CP.

    Queries stay in the local zigzag THD order for ``cp_rank``. Keys/values are
    manually all-gathered rank-by-rank, so their gathered tensor order is:
    rank0-local-packed, rank1-local-packed, ..., rank{cp_size-1}-local-packed.
    This helper returns the permutation that restores those gathered KV tensors
    to global packed order, matching the Slime GLM5 implementation semantics.
    """
    query_positions = build_packed_allgather_cp_local_positions(
        cu_seqlens_q, cp_size, cp_rank, device
    )
    gathered_key_positions = [
        build_packed_allgather_cp_local_positions(cu_seqlens_kv, cp_size, rank, device)
        for rank in range(cp_size)
    ]
    gathered_key_positions = torch.cat(gathered_key_positions, dim=0)
    key_reorder_idx = torch.argsort(gathered_key_positions)
    return query_positions, key_reorder_idx


def extract_query_positions_from_position_ids(
    position_ids: Optional[torch.Tensor], sq: int, device: torch.device
) -> Optional[torch.Tensor]:
    """Extract per-rank query positions from position_ids if compatible."""
    if position_ids is None:
        return None
    if position_ids.ndim == 2:
        if position_ids.size(0) > 1:
            assert torch.equal(
                position_ids[0], position_ids[-1]
            ), "Allgather-CP DSA expects identical position_ids across batch"
        query_pos = position_ids[0]
    elif position_ids.ndim == 1:
        query_pos = position_ids
    else:
        raise ValueError(f"position_ids should be 1D or 2D tensor, got {position_ids.ndim}D.")

    if query_pos.numel() != sq:
        return None
    return query_pos.to(device=device, dtype=torch.int64)


def get_packed_qk_cu_seqlens(
    packed_seq_params: PackedSeqParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select packed cu_seqlens for query and key/value streams."""
    cu_seqlens_q = (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else packed_seq_params.cu_seqlens_q
    )
    cu_seqlens = (
        packed_seq_params.cu_seqlens_kv_padded
        if packed_seq_params.cu_seqlens_kv_padded is not None
        else packed_seq_params.cu_seqlens_kv
    )
    cu_seqlens_kv = cu_seqlens

    if cu_seqlens_q is None and cu_seqlens_kv is None:
        raise ValueError("Packed sequence parameters must provide cu_seqlens for DSA masking.")
    if cu_seqlens_q is None:
        cu_seqlens_q = cu_seqlens_kv
    if cu_seqlens_kv is None:
        cu_seqlens_kv = cu_seqlens_q
    return cu_seqlens_q, cu_seqlens_kv
