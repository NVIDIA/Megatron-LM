# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layout helpers for DeepSeek sparse attention."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_pg_size

__all__ = [
    "PackedCPIndexerLayout",
    "build_packed_cp_indexer_layout",
    "build_packed_allgather_cp_local_positions",
    "build_packed_allgather_cp_query_positions_and_key_reorder",
    "build_zigzag_allgather_cp_key_reorder",
    "build_zigzag_cp_local_positions",
    "ensure_sbhd",
    "extract_query_positions_from_position_ids",
    "get_cp_positions_from_layout",
    "get_packed_qk_cu_seqlens",
    "normalize_cp_comm_type",
    "build_packed_allgather_cp_local_positions_from_host",
    "build_packed_allgather_cp_query_positions_and_key_reorder_from_host",
]


@dataclass(frozen=True)
class PackedCPIndexerLayout:
    """Segment metadata shared by packed-CP DSA indexer backends."""

    segment_q_lengths: torch.Tensor
    segment_k_lengths: torch.Tensor
    segment_cu_q: torch.Tensor
    segment_cu_k: torch.Tensor
    segment_key_starts: torch.Tensor
    source_indices: torch.Tensor


def build_packed_cp_indexer_layout(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    *,
    cp_size: int,
    cp_rank: int,
    key_size: int,
    local_key_layout: bool = False,
) -> PackedCPIndexerLayout:
    """Build packed-CP front/back segment metadata for fused DSA indexers.

    ``local_key_layout`` describes the single-sequence optimization where the
    key tensor contains only this CP rank's local front/back chunks. Otherwise,
    ``key_size`` is the globally ordered packed key length.
    """
    if cp_size <= 1 or not 0 <= cp_rank < cp_size:
        raise RuntimeError("packed CP indexer layout requires a valid CP rank and cp_size > 1")
    if cu_seqlens_q.shape != cu_seqlens_kv.shape or cu_seqlens_q.numel() < 2:
        raise RuntimeError("packed CP indexer layout requires matching non-empty q/k cu_seqlens")

    device = cu_seqlens_q.device
    cu_q = cu_seqlens_q.to(device=device, dtype=torch.int64).contiguous()
    cu_k = cu_seqlens_kv.to(device=device, dtype=torch.int64).contiguous()
    segment_divisor = 2 * cp_size

    if local_key_layout:
        if cu_q.numel() != 2 or key_size % 2 != 0:
            raise RuntimeError(
                "local-key packed CP indexer layout requires one sequence and even key rows"
            )
        half = key_size // 2
        segment_q_lengths = torch.full((2,), half, dtype=torch.int64, device=device)
        segment_k_lengths = torch.tensor((half, key_size), dtype=torch.int64, device=device)
        segment_key_starts = torch.zeros(2, dtype=torch.int64, device=device)
        total_segment_k = key_size + half
    else:
        if key_size % segment_divisor != 0:
            raise RuntimeError(
                f"packed CP key length must be divisible by {segment_divisor}, got {key_size}"
            )
        q_lengths = cu_q[1:] - cu_q[:-1]
        k_lengths = cu_k[1:] - cu_k[:-1]
        q_half = q_lengths // segment_divisor
        k_half = k_lengths // segment_divisor
        segment_q_lengths = torch.stack((q_half, q_half), dim=1).reshape(-1)
        segment_k_lengths = torch.stack(
            ((cp_rank + 1) * k_half, k_lengths - cp_rank * k_half), dim=1
        ).reshape(-1)
        segment_key_starts = cu_k[:-1].repeat_interleave(2)
        total_segment_k = key_size + key_size // segment_divisor

    zero = torch.zeros(1, dtype=torch.int64, device=device)
    segment_cu_q = torch.cat((zero, segment_q_lengths.cumsum(dim=0))).contiguous()
    segment_cu_k = torch.cat((zero, segment_k_lengths.cumsum(dim=0))).contiguous()

    segment_ids = torch.repeat_interleave(
        torch.arange(segment_k_lengths.numel(), device=device),
        segment_k_lengths,
        output_size=total_segment_k,
    )
    segment_offsets = torch.arange(total_segment_k, device=device, dtype=torch.int64)
    segment_offsets -= torch.repeat_interleave(
        segment_cu_k[:-1], segment_k_lengths, output_size=total_segment_k
    )
    source_indices = segment_key_starts.index_select(0, segment_ids) + segment_offsets
    return PackedCPIndexerLayout(
        segment_q_lengths=segment_q_lengths,
        segment_k_lengths=segment_k_lengths,
        segment_cu_q=segment_cu_q,
        segment_cu_k=segment_cu_k,
        segment_key_starts=segment_key_starts,
        source_indices=source_indices,
    )


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
        and get_pg_size(cp_group) == cp_size
    ):
        local_len = torch.tensor([sq], device=device, dtype=torch.int64)
        all_lens = [torch.empty_like(local_len) for _ in range(cp_size)]
        torch.distributed.all_gather(all_lens, local_len, group=cp_group)
        query_offset = int(torch.stack(all_lens[:cp_rank]).sum().item()) if cp_rank > 0 else 0

    query_pos = torch.arange(sq, device=device, dtype=torch.int64) + query_offset
    key_pos = torch.arange(skv, device=device, dtype=torch.int64)
    return query_pos, key_pos


def build_packed_allgather_cp_local_positions(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
    output_size: Optional[int] = None,
    *,
    cu_seqlens_cover_output: bool = False,
) -> torch.Tensor:
    """Build local packed-token positions for one CP rank under zigzag THD sharding.

    This mirrors the packed THD CP layout used by the surrounding training stack:
    each packed sequence is padded to a multiple of ``2 * cp_size`` and each rank
    receives the rank-local front chunk followed by the mirrored back chunk.
    """
    cu_seqlens_i64 = cu_seqlens.to(device=device, dtype=torch.int64)
    if cp_size <= 1:
        if output_size is None:
            output_size = int(cu_seqlens_i64[-1].item())
        return torch.arange(output_size, dtype=torch.int64, device=device)

    seq_starts = cu_seqlens_i64[:-1]
    seq_ends = cu_seqlens_i64[1:]
    seq_lens = seq_ends - seq_starts
    nonzero = seq_lens > 0
    seq_starts = seq_starts[nonzero]
    seq_ends = seq_ends[nonzero]
    seq_lens = seq_lens[nonzero]
    if seq_lens.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    # Host-side guard for CPU/test callers. In CUDA training these lengths are runtime tensors;
    # checking them here would add a sync, and padding divisibility is guaranteed by the pipeline.
    if cu_seqlens_i64.device.type == "cpu":
        bad_divisible = seq_lens[seq_lens % cp_size != 0]
        if bad_divisible.numel() > 0:
            raise ValueError(
                "Packed DSA CP expects per-sequence padded lengths divisible by cp_size, got "
                f"seq_len={int(bad_divisible[0].item())}, cp_size={cp_size}"
            )
        bad_local = seq_lens[(seq_lens // cp_size) % 2 != 0]
        if bad_local.numel() > 0:
            seq_len = int(bad_local[0].item())
            raise ValueError(
                "Packed DSA CP expects per-rank packed sequence lengths divisible by 2, got "
                f"local_seq_len={seq_len // cp_size}, seq_len={seq_len}, cp_size={cp_size}"
            )

    half_seq_lens = (seq_lens // cp_size) // 2
    front_starts = seq_starts + cp_rank * half_seq_lens
    back_starts = seq_ends - (cp_rank + 1) * half_seq_lens
    segment_starts = torch.stack((front_starts, back_starts), dim=1).reshape(-1)
    segment_lens = torch.stack((half_seq_lens, half_seq_lens), dim=1).reshape(-1)
    nonempty_segments = segment_lens > 0
    segment_starts = segment_starts[nonempty_segments]
    segment_lens = segment_lens[nonempty_segments]

    if output_size is None:
        output_size = int(segment_lens.sum().item())
    if output_size == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    if not cu_seqlens_cover_output:
        # Packed tensors may carry padded rows not represented by unpadded cu_seqlens.
        # Give those rows deterministic positions after all real tokens so KV reorder
        # keeps valid packed tokens ordered and moves padding to the suffix.
        pad_len = (
            torch.tensor(output_size, dtype=torch.int64, device=device) - segment_lens.sum()
        ).clamp_min(0)
        pad_start = cu_seqlens_i64[-1] + cp_rank * output_size
        segment_starts = torch.cat((segment_starts, pad_start.view(1)), dim=0)
        segment_lens = torch.cat((segment_lens, pad_len.view(1)), dim=0)

    segment_ids = torch.repeat_interleave(
        torch.arange(segment_lens.numel(), dtype=torch.int64, device=device),
        segment_lens,
        output_size=output_size,
    )
    segment_offsets = torch.arange(output_size, dtype=torch.int64, device=device)
    segment_offsets -= torch.repeat_interleave(
        torch.cumsum(segment_lens, dim=0) - segment_lens, segment_lens, output_size=output_size
    )
    return segment_starts.index_select(0, segment_ids) + segment_offsets


def build_packed_allgather_cp_all_rank_positions(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    device: torch.device,
    output_size: Optional[int] = None,
    *,
    cu_seqlens_cover_output: bool = False,
) -> torch.Tensor:
    """Build every CP rank's local packed positions at once: ``[cp_size, output_size]``.

    Row ``r`` is identical to
    ``build_packed_allgather_cp_local_positions(..., cp_rank=r, ...)``.

    Doing all ranks together is worth a separate function because every expensive
    step is rank-invariant. ``cp_rank`` enters only through ``front_starts`` and
    ``back_starts`` -- cheap elementwise expressions. In particular
    ``segment_lens`` is ``stack(half, half)`` and so does not depend on the rank,
    which means the ``nonzero``/``nonempty_segments`` boolean masks, and therefore
    all of the data-dependent output shapes that force a device-to-host size
    readback, are shared across ranks. The per-rank loop this replaces paid those
    readbacks ``cp_size`` times over; here they are paid once.
    """
    cu_seqlens_i64 = cu_seqlens.to(device=device, dtype=torch.int64)
    if cp_size <= 1:
        return build_packed_allgather_cp_local_positions(
            cu_seqlens,
            cp_size,
            0,
            device,
            output_size=output_size,
            cu_seqlens_cover_output=cu_seqlens_cover_output,
        ).unsqueeze(0)

    seq_starts = cu_seqlens_i64[:-1]
    seq_ends = cu_seqlens_i64[1:]
    seq_lens = seq_ends - seq_starts
    nonzero = seq_lens > 0
    seq_starts = seq_starts[nonzero]
    seq_ends = seq_ends[nonzero]
    seq_lens = seq_lens[nonzero]
    if seq_lens.numel() == 0:
        return torch.empty((cp_size, 0), dtype=torch.int64, device=device)

    # Host-side guard for CPU/test callers; mirrors the single-rank builder.
    if cu_seqlens_i64.device.type == "cpu":
        bad_divisible = seq_lens[seq_lens % cp_size != 0]
        if bad_divisible.numel() > 0:
            raise ValueError(
                "Packed DSA CP expects per-sequence padded lengths divisible by cp_size, got "
                f"seq_len={int(bad_divisible[0].item())}, cp_size={cp_size}"
            )
        bad_local = seq_lens[(seq_lens // cp_size) % 2 != 0]
        if bad_local.numel() > 0:
            seq_len = int(bad_local[0].item())
            raise ValueError(
                "Packed DSA CP expects per-rank packed sequence lengths divisible by 2, got "
                f"local_seq_len={seq_len // cp_size}, seq_len={seq_len}, cp_size={cp_size}"
            )

    half_seq_lens = (seq_lens // cp_size) // 2
    ranks = torch.arange(cp_size, dtype=torch.int64, device=device).unsqueeze(1)
    # [cp_size, n_seq]
    front_starts = seq_starts.unsqueeze(0) + ranks * half_seq_lens.unsqueeze(0)
    back_starts = seq_ends.unsqueeze(0) - (ranks + 1) * half_seq_lens.unsqueeze(0)
    # Interleave to [f0, b0, f1, b1, ...] per row, matching the single-rank builder.
    segment_starts = torch.stack((front_starts, back_starts), dim=2).reshape(cp_size, -1)
    segment_lens = torch.stack((half_seq_lens, half_seq_lens), dim=1).reshape(-1)
    nonempty_segments = segment_lens > 0
    segment_starts = segment_starts[:, nonempty_segments]
    segment_lens = segment_lens[nonempty_segments]

    if output_size is None:
        output_size = int(segment_lens.sum().item())
    if output_size == 0:
        return torch.empty((cp_size, 0), dtype=torch.int64, device=device)
    if not cu_seqlens_cover_output:
        pad_len = (
            torch.tensor(output_size, dtype=torch.int64, device=device) - segment_lens.sum()
        ).clamp_min(0)
        # Per-rank padding origin, matching cu_seqlens[-1] + cp_rank * output_size.
        pad_start = cu_seqlens_i64[-1] + ranks.reshape(-1) * output_size
        segment_starts = torch.cat((segment_starts, pad_start.unsqueeze(1)), dim=1)
        segment_lens = torch.cat((segment_lens, pad_len.view(1)), dim=0)

    segment_ids = torch.repeat_interleave(
        torch.arange(segment_lens.numel(), dtype=torch.int64, device=device),
        segment_lens,
        output_size=output_size,
    )
    segment_offsets = torch.arange(output_size, dtype=torch.int64, device=device)
    segment_offsets -= torch.repeat_interleave(
        torch.cumsum(segment_lens, dim=0) - segment_lens, segment_lens, output_size=output_size
    )
    return segment_starts.index_select(1, segment_ids) + segment_offsets.unsqueeze(0)


def build_packed_allgather_cp_query_positions_and_key_reorder(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
    local_output_size: Optional[int] = None,
    key_local_output_size: Optional[int] = None,
    global_output_size: Optional[int] = None,
    *,
    query_cu_seqlens_cover_output: bool = False,
    key_cu_seqlens_cover_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build packed-query positions and gathered-KV reorder index for allgather CP.

    Queries stay in the local zigzag THD order for ``cp_rank``. Keys/values are
    manually all-gathered rank-by-rank, so their gathered tensor order is:
    rank0-local-packed, rank1-local-packed, ..., rank{cp_size-1}-local-packed.
    This helper returns the permutation that restores those gathered KV tensors
    to global packed order, matching the Slime GLM5 implementation semantics.
    """
    query_positions = build_packed_allgather_cp_local_positions(
        cu_seqlens_q,
        cp_size,
        cp_rank,
        device,
        output_size=local_output_size,
        cu_seqlens_cover_output=query_cu_seqlens_cover_output,
    )
    if key_local_output_size is None:
        key_local_output_size = local_output_size
    # All ranks in one batched build. Row-major flattening reproduces exactly the
    # rank0-local, rank1-local, ... concatenation the gathered KV tensor is in.
    gathered_key_positions = build_packed_allgather_cp_all_rank_positions(
        cu_seqlens_kv,
        cp_size,
        device,
        output_size=key_local_output_size,
        cu_seqlens_cover_output=key_cu_seqlens_cover_output,
    ).reshape(-1)
    key_reorder_idx = torch.argsort(gathered_key_positions)
    if global_output_size is not None and key_reorder_idx.numel() != global_output_size:
        raise RuntimeError(
            f"Packed DSA CP key reorder length mismatch: got {key_reorder_idx.numel()}, "
            f"expected {global_output_size}"
        )
    return query_positions, key_reorder_idx


def _host_packed_cp_spans(
    host_cu_seqlens: List[int], cp_size: int, cp_rank: int
) -> List[Tuple[int, int]]:
    """One rank's zigzag layout as ``(global_start, length)`` spans, from host ints.

    The span decomposition is the same maths as the device builders above, but on a
    Python list there is nothing to synchronize on: every length is already known.
    Zero-length sequences contribute no spans, matching the device builders' filter.
    """
    spans: List[Tuple[int, int]] = []
    for seq_start, seq_end in zip(host_cu_seqlens[:-1], host_cu_seqlens[1:]):
        seq_len = seq_end - seq_start
        if seq_len == 0:
            continue
        # The zigzag layout requires it, and on host integers the check is free --
        # unlike the device builders, which can only validate without a sync on CPU
        # inputs. Without it, non-divisible lengths would silently drop tokens here
        # and leave uninitialized slots in the reorder buffer downstream.
        if seq_len % (2 * cp_size) != 0:
            raise ValueError(
                "Packed DSA CP expects per-sequence padded lengths divisible by "
                f"2 * cp_size ({2 * cp_size}), got seq_len={seq_len}"
            )
        half = seq_len // cp_size // 2
        if half <= 0:
            continue
        spans.append((seq_start + cp_rank * half, half))
        spans.append((seq_end - (cp_rank + 1) * half, half))
    return spans


def _host_to_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a freshly built host tensor to ``device`` without blocking the CPU."""
    if device.type == "cuda":
        return values.pin_memory().to(device, non_blocking=True)
    return values.to(device)


def build_packed_allgather_cp_local_positions_from_host(
    host_cu_seqlens: List[int],
    cp_size: int,
    cp_rank: int,
    device: torch.device,
    output_size: Optional[int] = None,
    *,
    cu_seqlens_cover_output: bool = False,
) -> torch.Tensor:
    """Host-side equivalent of :func:`build_packed_allgather_cp_local_positions`.

    The device builder runs ~20 kernels over a few dozen integers and pays two
    device-to-host size readbacks for its boolean-mask filters. When the compacted
    ``cu_seqlens`` are already on the host -- ``prebuild_thd_cp_partition_routes``
    stores them on ``PackedSeqParams`` at batch-construction time, where the one
    blocking copy is cheap because the CUDA queue is still shallow -- the whole
    table is a closed form over Python ints: zero kernels, zero synchronization,
    one asynchronous host-to-device copy of the finished table.
    """
    if cp_size <= 1:
        # Mirror the device builder: at cp_size <= 1 the local layout is the identity,
        # with no zigzag halving (and therefore no even-length requirement).
        if output_size is None:
            output_size = host_cu_seqlens[-1]
        return _host_to_device(torch.arange(output_size, dtype=torch.int64), device)
    spans = _host_packed_cp_spans(host_cu_seqlens, cp_size, cp_rank)
    real = (
        torch.cat([torch.arange(start, start + length) for start, length in spans])
        if spans
        else torch.empty(0, dtype=torch.int64)
    )
    total = real.numel()
    if output_size is None:
        output_size = total
    positions = torch.empty(output_size, dtype=torch.int64)
    n = min(total, output_size)
    positions[:n] = real[:n]
    if output_size > n:
        # The cover flag only ever accompanies output_size == total (dsa.py derives it
        # from host max-seqlen metadata), so this branch is the not-covered case; fill
        # it unconditionally rather than leave torch.empty garbage if a caller lies.
        pad_start = host_cu_seqlens[-1] + cp_rank * output_size
        positions[n:] = torch.arange(pad_start, pad_start + (output_size - n))
    return _host_to_device(positions, device)


def build_packed_allgather_cp_query_positions_and_key_reorder_from_host(
    host_cu_seqlens_q: List[int],
    host_cu_seqlens_kv: List[int],
    cp_size: int,
    cp_rank: int,
    device: torch.device,
    local_output_size: Optional[int] = None,
    key_local_output_size: Optional[int] = None,
    global_output_size: Optional[int] = None,
    *,
    query_cu_seqlens_cover_output: bool = False,
    key_cu_seqlens_cover_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Host-side equivalent of the query-positions + key-reorder wrapper.

    Besides removing the device builders' synchronizations, this replaces the
    ``argsort`` over the gathered key positions with a direct inverse permutation.
    The sort is unnecessary because the real positions of all ranks tile
    ``[0, total_tokens)`` exactly once (every padded sequence length is divisible
    by ``2 * cp_size``), so the rank of a value in sorted order *is* the value;
    and the padding pseudo-positions ``total + r * output_size + j`` are unique
    and already ascending in ``(rank, j)`` order. Both facts are pinned by the
    bit-equality tests against the device path.
    """
    query_positions = build_packed_allgather_cp_local_positions_from_host(
        host_cu_seqlens_q,
        cp_size,
        cp_rank,
        device,
        output_size=local_output_size,
        cu_seqlens_cover_output=query_cu_seqlens_cover_output,
    )
    if key_local_output_size is None:
        key_local_output_size = local_output_size

    kv_total = host_cu_seqlens_kv[-1]
    if cp_size <= 1:
        # Identity layout: the gathered order is already the global order.
        out = key_local_output_size if key_local_output_size is not None else kv_total
        return query_positions, _host_to_device(torch.arange(out, dtype=torch.int64), device)
    spans_by_rank = [
        _host_packed_cp_spans(host_cu_seqlens_kv, cp_size, rank) for rank in range(cp_size)
    ]
    real_len = sum(length for _, length in spans_by_rank[0]) if spans_by_rank else 0
    if real_len * cp_size != kv_total:
        raise ValueError(
            "Packed DSA CP spans do not tile the key stream: "
            f"{cp_size} ranks x {real_len} real tokens != total {kv_total}"
        )
    out = key_local_output_size if key_local_output_size is not None else real_len
    pad = max(0, out - real_len)
    n = cp_size * out
    if global_output_size is not None and n != global_output_size:
        raise RuntimeError(
            f"Packed DSA CP key reorder length mismatch: got {n}, " f"expected {global_output_size}"
        )
    key_reorder_idx = torch.empty(n, dtype=torch.int64)
    for rank, spans in enumerate(spans_by_rank):
        local = 0
        for global_start, length in spans:
            key_reorder_idx[global_start : global_start + length] = torch.arange(
                rank * out + local, rank * out + local + length
            )
            local += length
        if pad > 0:
            key_reorder_idx[kv_total + rank * pad : kv_total + (rank + 1) * pad] = torch.arange(
                rank * out + local, rank * out + out
            )
    return query_positions, _host_to_device(key_reorder_idx, device)


def extract_query_positions_from_position_ids(
    position_ids: Optional[torch.Tensor], sq: int, device: torch.device
) -> Optional[torch.Tensor]:
    """Extract per-rank query positions from position_ids if compatible."""
    if position_ids is None:
        return None
    if position_ids.ndim == 2:
        # ``torch.equal`` on CUDA forces a per-forward host/device sync, so only run the eager
        # cross-batch consistency check off the CUDA training path (tests/CPU). On CUDA we rely on
        # the dataloader contract that DSA position_ids are identical across the batch dimension.
        if position_ids.size(0) > 1 and not position_ids.is_cuda:
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
