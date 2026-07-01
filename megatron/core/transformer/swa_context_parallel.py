# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Context Parallel helpers for Sliding Window Attention via Halo Exchange."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict

import torch


@dataclass(frozen=True)
class _HaloPart:
    """One contiguous source slice needed by a destination rank's halo."""

    dst_segment: int
    src_rank: int
    src_segment: int
    src_offset: int
    length: int


@dataclass(frozen=True)
class SwaP2PSegment:
    """One contiguous local sequence segment in CP order."""

    sequence_id: int
    sequence_start: int
    global_start: int
    length: int
    local_offset: int


def is_swa_p2p_cp_comm_type(cp_comm_type: str | None) -> bool:
    """Return whether ``cp_comm_type`` selects the SWA-specific P2P path."""
    return cp_comm_type is not None and cp_comm_type.replace("_", "").lower() == "swap2p"


def _rank_segment_chunk(rank: int, segment: int, cp_size: int) -> int:
    """Return the global chunk id for ``rank`` and local ``segment``."""
    if segment == 0:
        return rank
    return 2 * cp_size - rank - 1


def _build_swa_p2p_halo_plan(
    segments_by_rank: list[list[SwaP2PSegment]], window_size: int
) -> tuple[dict[tuple[int, int], list[_HaloPart]], dict[tuple[int, int], list[_HaloPart]]]:
    """Build deterministic send and receive plans for SWA halos."""
    cp_size = len(segments_by_rank)
    send_plan: dict[tuple[int, int], list[_HaloPart]] = {}
    parts_by_dst: dict[tuple[int, int], list[_HaloPart]] = {}

    if window_size <= 0:
        for dst_rank in range(cp_size):
            for dst_segment in range(len(segments_by_rank[dst_rank])):
                parts_by_dst[(dst_rank, dst_segment)] = []
        return send_plan, parts_by_dst

    for dst_rank in range(cp_size):
        for dst_segment, dst_info in enumerate(segments_by_rank[dst_rank]):
            segment_start = dst_info.global_start
            halo_start = max(dst_info.sequence_start, segment_start - window_size)
            parts: list[_HaloPart] = []

            if halo_start < segment_start:
                for src_rank, src_segments in enumerate(segments_by_rank):
                    for src_segment, src_info in enumerate(src_segments):
                        if src_info.sequence_id != dst_info.sequence_id:
                            continue
                        src_start = max(halo_start, src_info.global_start)
                        src_end = min(segment_start, src_info.global_start + src_info.length)
                        length = src_end - src_start
                        if length <= 0:
                            continue
                        part = _HaloPart(
                            dst_segment=dst_segment,
                            src_rank=src_rank,
                            src_segment=src_segment,
                            src_offset=src_start - src_info.global_start,
                            length=length,
                        )
                        parts.append(part)
                        if src_rank != dst_rank:
                            send_plan.setdefault((src_rank, dst_rank), []).append(part)

            parts.sort(
                key=lambda part: (
                    segments_by_rank[part.src_rank][part.src_segment].global_start + part.src_offset
                )
            )
            parts_by_dst[(dst_rank, dst_segment)] = parts

    for parts in send_plan.values():
        parts.sort(
            key=lambda part: (
                segments_by_rank[part.src_rank][part.src_segment].global_start + part.src_offset
            )
        )

    return send_plan, parts_by_dst


def build_swa_p2p_nonpacked_segments(local_seq_len: int, cp_size: int) -> list[list[SwaP2PSegment]]:
    """Build segment metadata for MCore non-packed zigzag CP layout."""
    if cp_size <= 0:
        raise ValueError(f"swa_p2p expects a positive cp_size, got {cp_size}")
    if local_seq_len % 2 != 0:
        raise ValueError(f"swa_p2p expects an even local sequence length, got {local_seq_len}")

    chunk_len = local_seq_len // 2
    segments_by_rank: list[list[SwaP2PSegment]] = []
    for rank in range(cp_size):
        local_offset = 0
        rank_segments = []
        for segment in (0, 1):
            chunk = _rank_segment_chunk(rank, segment, cp_size)
            rank_segments.append(
                SwaP2PSegment(
                    sequence_id=0,
                    sequence_start=0,
                    global_start=chunk * chunk_len,
                    length=chunk_len,
                    local_offset=local_offset,
                )
            )
            local_offset += chunk_len
        segments_by_rank.append(rank_segments)
    return segments_by_rank


def _cu_seqlens_to_list(cu_seqlens: torch.Tensor) -> list[int]:
    """Convert 1-D or batch-leading cu_seqlens to a Python list."""
    if cu_seqlens.dim() == 2:
        if cu_seqlens.size(0) != 1:
            raise ValueError(
                f"swa_p2p expects packed cu_seqlens with batch size 1, got {cu_seqlens.shape}"
            )
        cu_seqlens = cu_seqlens[0]
    if cu_seqlens.dim() != 1:
        raise ValueError(f"swa_p2p expects 1-D packed cu_seqlens, got {cu_seqlens.shape}")
    return [int(v) for v in cu_seqlens.detach().cpu().tolist()]


def build_swa_p2p_packed_segments(
    cu_seqlens_padded: torch.Tensor, cp_size: int
) -> list[list[SwaP2PSegment]]:
    """Build segment metadata for packed THD zigzag CP layout."""
    if cp_size <= 0:
        raise ValueError(f"swa_p2p expects a positive cp_size, got {cp_size}")
    cu = _cu_seqlens_to_list(cu_seqlens_padded)
    if any(end < start for start, end in zip(cu[:-1], cu[1:])):
        raise ValueError(f"swa_p2p expects monotonic cu_seqlens, got {cu}")
    local_offsets = [0 for _ in range(cp_size)]
    segments_by_rank: list[list[SwaP2PSegment]] = [[] for _ in range(cp_size)]

    for sequence_id, (sequence_start, sequence_end) in enumerate(zip(cu[:-1], cu[1:])):
        sequence_len = sequence_end - sequence_start
        if sequence_len % (2 * cp_size) != 0:
            raise ValueError(
                "swa_p2p packed CP expects padded sequence lengths divisible by 2 * cp_size, "
                f"got sequence_len={sequence_len}, cp_size={cp_size}"
            )
        chunk_len = sequence_len // (2 * cp_size)
        for rank in range(cp_size):
            front_start = sequence_start + rank * chunk_len
            back_start = sequence_end - (rank + 1) * chunk_len
            for global_start in (front_start, back_start):
                segments_by_rank[rank].append(
                    SwaP2PSegment(
                        sequence_id=sequence_id,
                        sequence_start=sequence_start,
                        global_start=global_start,
                        length=chunk_len,
                        local_offset=local_offsets[rank],
                    )
                )
                local_offsets[rank] += chunk_len

    return segments_by_rank


def _get_cp_global_ranks(cp_group: torch.distributed.ProcessGroup) -> list[int]:
    """Return global ranks for a CP group."""
    try:
        return torch.distributed.get_process_group_ranks(cp_group)
    except AttributeError:
        try:
            return [
                torch.distributed.get_global_rank(cp_group, group_rank)
                for group_rank in range(cp_group.size())
            ]
        except AttributeError:
            return list(range(torch.distributed.get_world_size()))


def _wait_all(reqs: list[torch.distributed.Work]) -> None:
    """Wait for all distributed requests."""
    for req in reqs:
        req.wait()


def _run_p2p_ops(
    recv_tensors: dict[int, torch.Tensor],
    send_tensors: dict[int, torch.Tensor],
    cp_group: torch.distributed.ProcessGroup,
    cp_rank: int,
) -> None:
    """Run one batched P2P exchange with at most one send/recv tensor per peer."""
    if not recv_tensors and not send_tensors:
        return

    global_ranks = _get_cp_global_ranks(cp_group)
    ops: list[torch.distributed.P2POp] = []
    for peer in sorted(set(recv_tensors) | set(send_tensors)):
        if peer == cp_rank:
            continue
        global_peer = global_ranks[peer]
        if peer in recv_tensors:
            ops.append(
                torch.distributed.P2POp(
                    torch.distributed.irecv, recv_tensors[peer], global_peer, cp_group
                )
            )
        if peer in send_tensors:
            ops.append(
                torch.distributed.P2POp(
                    torch.distributed.isend, send_tensors[peer], global_peer, cp_group
                )
            )

    if ops:
        _wait_all(torch.distributed.batch_isend_irecv(ops))


class _SwaP2PHaloExchange(torch.autograd.Function):
    """Autograd function that returns previous-token halos and routes halo gradients back."""

    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        window_size: int,
        cp_group: torch.distributed.ProcessGroup,
        segments_by_rank: list[list[SwaP2PSegment]],
    ) -> tuple[torch.Tensor, ...]:
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        local_segments = segments_by_rank[cp_rank]
        if cp_size == 1 or window_size <= 0:
            empty = input_.new_empty((0, *input_.shape[1:]))
            ctx.cp_group = cp_group
            ctx.cp_rank = cp_rank
            ctx.cp_size = cp_size
            ctx.input_shape = tuple(input_.shape)
            ctx.send_plan = {}
            ctx.segments_by_rank = segments_by_rank
            ctx.parts_by_dst = {(cp_rank, segment): [] for segment in range(len(local_segments))}
            return tuple(empty for _ in local_segments)

        expected_local_seq_len = sum(segment.length for segment in local_segments)
        if input_.size(0) != expected_local_seq_len:
            raise ValueError(
                "swa_p2p input length does not match CP segment metadata: "
                f"got {input_.size(0)}, expected {expected_local_seq_len}"
            )

        send_plan, parts_by_dst = _build_swa_p2p_halo_plan(segments_by_rank, window_size)
        tensor_tail_shape = tuple(input_.shape[1:])

        send_tensors: dict[int, torch.Tensor] = {}
        recv_tensors: dict[int, torch.Tensor] = {}

        for peer in range(cp_size):
            if peer == cp_rank:
                continue
            recv_sections = send_plan.get((peer, cp_rank), [])
            if recv_sections:
                recv_len = sum(part.length for part in recv_sections)
                recv_tensors[peer] = input_.new_empty((recv_len, *tensor_tail_shape))

            send_sections = send_plan.get((cp_rank, peer), [])
            if send_sections:
                send_parts = [
                    input_.narrow(
                        0,
                        segments_by_rank[cp_rank][part.src_segment].local_offset + part.src_offset,
                        part.length,
                    ).contiguous()
                    for part in send_sections
                ]
                send_tensors[peer] = (
                    torch.cat(send_parts, dim=0) if len(send_parts) > 1 else send_parts[0]
                )

        _run_p2p_ops(recv_tensors, send_tensors, cp_group, cp_rank)

        recv_offsets: DefaultDict[int, int] = defaultdict(int)
        halos: list[torch.Tensor] = []
        for dst_segment in range(len(local_segments)):
            halo_parts: list[torch.Tensor] = []
            for part in parts_by_dst[(cp_rank, dst_segment)]:
                if part.src_rank == cp_rank:
                    halo_parts.append(
                        input_.narrow(
                            0,
                            segments_by_rank[cp_rank][part.src_segment].local_offset
                            + part.src_offset,
                            part.length,
                        )
                    )
                else:
                    offset = recv_offsets[part.src_rank]
                    halo_parts.append(recv_tensors[part.src_rank].narrow(0, offset, part.length))
                    recv_offsets[part.src_rank] += part.length

            if halo_parts:
                halos.append(torch.cat(halo_parts, dim=0))
            else:
                halos.append(input_.new_empty((0, *tensor_tail_shape)))

        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.input_shape = tuple(input_.shape)
        ctx.send_plan = send_plan
        ctx.segments_by_rank = segments_by_rank
        ctx.parts_by_dst = parts_by_dst
        return tuple(halos)

    @staticmethod
    def backward(ctx, *grad_halos: torch.Tensor):
        cp_group = ctx.cp_group
        cp_rank = ctx.cp_rank
        cp_size = ctx.cp_size
        send_plan = ctx.send_plan
        segments_by_rank = ctx.segments_by_rank
        parts_by_dst = ctx.parts_by_dst

        grad_input = grad_halos[0].new_zeros(ctx.input_shape)
        if cp_size == 1:
            return grad_input, None, None, None

        grad_parts_by_src: DefaultDict[int, list[torch.Tensor]] = defaultdict(list)
        for dst_segment, grad_halo in enumerate(grad_halos):
            halo_offset = 0
            for part in parts_by_dst[(cp_rank, dst_segment)]:
                grad_part = grad_halo.narrow(0, halo_offset, part.length)
                halo_offset += part.length
                if part.src_rank == cp_rank:
                    grad_input.narrow(
                        0,
                        segments_by_rank[cp_rank][part.src_segment].local_offset + part.src_offset,
                        part.length,
                    ).add_(grad_part)
                else:
                    grad_parts_by_src[part.src_rank].append(grad_part.contiguous())

        send_tensors: dict[int, torch.Tensor] = {}
        recv_tensors: dict[int, torch.Tensor] = {}

        for peer in range(cp_size):
            if peer == cp_rank:
                continue
            recv_sections = send_plan.get((cp_rank, peer), [])
            if recv_sections:
                recv_len = sum(part.length for part in recv_sections)
                recv_tensors[peer] = grad_input.new_empty((recv_len, *ctx.input_shape[1:]))

            send_sections = send_plan.get((peer, cp_rank), [])
            if send_sections:
                send_parts = grad_parts_by_src.get(peer, [])
                if not send_parts:
                    raise RuntimeError(
                        "swa_p2p backward halo plan mismatch: "
                        f"missing gradient parts for source rank {peer}."
                    )
                send_tensors[peer] = (
                    torch.cat(send_parts, dim=0) if len(send_parts) > 1 else send_parts[0]
                )

        _run_p2p_ops(recv_tensors, send_tensors, cp_group, cp_rank)

        for peer, recv_tensor in recv_tensors.items():
            offset = 0
            for part in send_plan[(cp_rank, peer)]:
                grad_input.narrow(
                    0,
                    segments_by_rank[cp_rank][part.src_segment].local_offset + part.src_offset,
                    part.length,
                ).add_(recv_tensor.narrow(0, offset, part.length))
                offset += part.length

        return grad_input, None, None, None


def get_swa_p2p_halos(
    input_: torch.Tensor,
    window_size: int,
    cp_group: torch.distributed.ProcessGroup,
    segments_by_rank: list[list[SwaP2PSegment]] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Return SWA previous-token halos for a CP tensor.

    Args:
        input_: Tensor with sequence first, in MCore CP local order
            ``[front_chunk, back_chunk]``.
        window_size: Causal sliding-window size on the left side.
        cp_group: Context-parallel process group.
        segments_by_rank: Optional CP segment metadata. If omitted, non-packed
            zigzag layout is inferred from ``input_``.

    Returns:
        A tuple of halos in rank-local segment order. Each halo is ordered by global token
        position and contains at most ``window_size`` tokens preceding that local segment.
    """
    if segments_by_rank is None:
        segments_by_rank = build_swa_p2p_nonpacked_segments(input_.size(0), cp_group.size())
    return _SwaP2PHaloExchange.apply(input_, window_size, cp_group, segments_by_rank)
