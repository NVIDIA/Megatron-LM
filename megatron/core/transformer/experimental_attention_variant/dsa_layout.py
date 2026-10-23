# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layout helpers for DeepSeek sparse attention."""

from dataclasses import dataclass, fields
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_pg_size, move_host_tensor_to_device, nvtx_range

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
    "invert_permutation",
    "normalize_cp_comm_type",
    "build_packed_allgather_cp_local_positions_from_host",
    "build_packed_allgather_cp_query_positions_and_key_reorder_from_host",
    # Packed-CP layout tables prebuilt at batch construction.
    "PackedCPLayoutGeometry",
    "PrebuiltPackedCPLayout",
    "attach_prebuilt_packed_cp_layout",
    "build_packed_cp_layout_from_host",
    "copy_prebuilt_packed_cp_layout_",
    "get_prebuilt_packed_cp_layout",
    "prebuild_packed_cp_layout",
    "resolve_packed_cp_cover_output_flags",
    "resolve_packed_cp_layout_geometry",
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


def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    """Return ``inv`` with ``inv[perm[i]] == i`` for a 1-D ``int64`` permutation ``perm``.

    ``x.index_select(0, perm)`` is then undone exactly by ``y.index_select(0, inv)``,
    which is what lets the CP key reorder use a plain gather as its backward instead
    of autograd's zero fill plus atomic ``index_add_``. That is only valid when
    ``perm`` is a bijection of ``[0, n)``. Host (CPU) inputs are checked here, where
    reading the values needs no device synchronization, and a host ``perm`` that is
    not a permutation raises ``ValueError``: it is a hard error, not a fallback to
    ``index_add_``, because such a table would already be wrong in the forward.
    Device inputs are deliberately not checked, since reading them back would add a
    synchronization: callers must pass a permutation by construction, such as an
    ``argsort`` output. The device scatter has a static output shape, so it is safe
    to issue while a CUDA graph is being captured.
    """
    if perm.dim() != 1 or perm.dtype != torch.int64:
        raise ValueError(
            "invert_permutation expects a 1-D int64 tensor, got "
            f"shape={tuple(perm.shape)}, dtype={perm.dtype}"
        )
    n = perm.numel()
    positions = torch.arange(n, dtype=torch.int64, device=perm.device)
    if perm.device.type != "cpu":
        return torch.empty(n, dtype=torch.int64, device=perm.device).scatter_(0, perm, positions)
    if n > 0 and (int(perm.min()) < 0 or int(perm.max()) >= n):
        raise ValueError(
            f"Expected a permutation of [0, {n}), got values in "
            f"[{int(perm.min())}, {int(perm.max())}]"
        )
    inv = torch.full((n,), -1, dtype=torch.int64)
    inv[perm] = positions
    # n in-range entries: a repeated entry necessarily leaves another slot unwritten.
    if bool((inv < 0).any()):
        raise ValueError(f"Expected a permutation of [0, {n}), got repeated entries")
    return inv


def build_zigzag_allgather_cp_key_reorder(
    sq: int, cp_size: int, device: torch.device, *, return_inverse_key_reorder: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Build gathered-KV reorder index for non-packed zigzag allgather CP.

    With ``return_inverse_key_reorder`` the inverse permutation is returned as well,
    as ``(reorder, inverse)``. The ranks' front/back chunks tile ``[0, sq * cp_size)``
    exactly once, so the gathered key positions are a permutation, the reorder is
    their ``argsort``, and the positions themselves are its inverse: no extra kernel.
    """
    global_seq_len = sq * cp_size
    gathered_key_positions = torch.cat(
        [
            build_zigzag_cp_local_positions(global_seq_len, cp_size, rank, device)
            for rank in range(cp_size)
        ],
        dim=0,
    )
    key_reorder_idx = torch.argsort(gathered_key_positions)
    if return_inverse_key_reorder:
        return key_reorder_idx, gathered_key_positions
    return key_reorder_idx


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
    return_inverse_key_reorder: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Build packed-query positions and gathered-KV reorder index for allgather CP.

    Queries stay in the local zigzag THD order for ``cp_rank``. Keys/values are
    manually all-gathered rank-by-rank, so their gathered tensor order is:
    rank0-local-packed, rank1-local-packed, ..., rank{cp_size-1}-local-packed.
    This helper returns the permutation that restores those gathered KV tensors
    to global packed order, matching the Slime GLM5 implementation semantics.

    With ``return_inverse_key_reorder`` a third tensor, the inverse permutation
    (see :func:`invert_permutation`), is returned as well. The reorder is an
    ``argsort`` output and therefore always a permutation of ``[0, n)``, so the
    inverse is scattered on the reorder's device without a host readback, and
    without a bijection check by design. DSAttention requests the triple, so a
    replacement of this builder inside ``dsa.py`` must forward that keyword.
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
    if return_inverse_key_reorder:
        return query_positions, key_reorder_idx, invert_permutation(key_reorder_idx)
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
    table is a closed form over Python ints: no device kernels or device-to-host
    size readbacks, followed by one host-to-device copy of the finished table.
    """
    if cp_size <= 1:
        # Mirror the device builder: at cp_size <= 1 the local layout is the identity,
        # with no zigzag halving (and therefore no even-length requirement).
        if output_size is None:
            output_size = host_cu_seqlens[-1]
        return move_host_tensor_to_device(torch.arange(output_size, dtype=torch.int64), device)
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
    return move_host_tensor_to_device(positions, device)


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
    return_inverse_key_reorder: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Host-side equivalent of the query-positions + key-reorder wrapper.

    Besides removing the device builders' synchronizations, this replaces the
    ``argsort`` over the gathered key positions with a direct inverse permutation.
    The sort is unnecessary because the real positions of all ranks tile
    ``[0, total_tokens)`` exactly once (every padded sequence length is divisible
    by ``2 * cp_size``), so the rank of a value in sorted order *is* the value;
    and the padding pseudo-positions ``total + r * output_size + j`` are unique
    and already ascending in ``(rank, j)`` order. Both facts are pinned by the
    bit-equality tests against the device path.

    With ``return_inverse_key_reorder`` the inverse permutation is returned as a
    third tensor. It is built on the host before either copy, which is also where
    the reorder table is checked to be a bijection (see :func:`invert_permutation`):
    a table that is not one raises ``ValueError`` instead of being copied. DSAttention
    requests the triple, so a replacement of this builder inside ``dsa.py`` must
    forward that keyword.
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

    def _finish(key_reorder_idx: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if not return_inverse_key_reorder:
            return query_positions, move_host_tensor_to_device(key_reorder_idx, device)
        # Invert (and so validate) before either copy is issued.
        key_reorder_inv = invert_permutation(key_reorder_idx)
        return (
            query_positions,
            move_host_tensor_to_device(key_reorder_idx, device),
            move_host_tensor_to_device(key_reorder_inv, device),
        )

    kv_total = host_cu_seqlens_kv[-1]
    if cp_size <= 1:
        # Identity layout: the gathered order is already the global order.
        out = key_local_output_size if key_local_output_size is not None else kv_total
        return _finish(torch.arange(out, dtype=torch.int64))
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
    return _finish(key_reorder_idx)


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


def resolve_packed_cp_cover_output_flags(
    packed_seq_params: PackedSeqParams, global_output_size: int
) -> Tuple[bool, bool]:
    """Return the ``(query, key)`` ``cu_seqlens_cover_output`` flags of the packed-CP layout.

    They are part of the layout's identity (the memo key and the prebuilt geometry), so
    ``DSAttention.forward`` and :func:`prebuild_packed_cp_layout` both derive them here.
    The flags are only claimed for a single packed sequence whose host ``max_seqlen``
    equals the global (all-rank) output size.
    """
    cu_seqlens_q, cu_seqlens_kv = get_packed_qk_cu_seqlens(packed_seq_params)
    single_sequence = cu_seqlens_q.numel() == 2 and cu_seqlens_kv.numel() == 2
    query_cover = (
        single_sequence
        and isinstance(packed_seq_params.max_seqlen_q, int)
        and packed_seq_params.max_seqlen_q == global_output_size
    )
    key_cover = (
        single_sequence
        and isinstance(packed_seq_params.max_seqlen_kv, int)
        and packed_seq_params.max_seqlen_kv == global_output_size
    )
    return query_cover, key_cover


class PackedCPLayoutGeometry(NamedTuple):
    """The request a packed-CP DSA layout answers.

    ``local_rows`` is the CP-local physical row count, i.e. the layout builders'
    ``local_output_size``; the global (gathered) size is ``local_rows * cp_size``. The
    cover flags are the builders' ``cu_seqlens_cover_output`` arguments. Together the
    fields carry exactly the information of ``DSAttention``'s lazy memo keys.
    """

    cp_size: int
    cp_rank: int
    local_rows: int
    query_cu_seqlens_cover_output: bool
    key_cu_seqlens_cover_output: bool


# PackedSeqParams attributes. Like the host cu_seqlens recorded by
# prebuild_thd_cp_partition_routes, these are dynamic attributes: they are not forwarded
# to TE and die with the per-microbatch metadata. DSAttention owns the lazy layout memo;
# its name lives here so copy_prebuilt_packed_cp_layout_ can drop it without importing dsa.
_PREBUILT_PACKED_CP_LAYOUT_ATTR = "_dsa_prebuilt_packed_cp_layout"
PACKED_CP_LAYOUT_MEMO_ATTR = "_dsa_packed_cp_layout_holder"
_HOST_CU_SEQLENS_ATTRS = ("thd_cp_host_cu_seqlens_q", "thd_cp_host_cu_seqlens_kv")


def _host_cu_seqlens(packed_seq_params) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """The host boundary lists recorded on ``packed_seq_params`` (``None`` when absent)."""
    host_cu_seqlens_q, host_cu_seqlens_kv = (
        getattr(packed_seq_params, name, None) for name in _HOST_CU_SEQLENS_ATTRS
    )
    return host_cu_seqlens_q, host_cu_seqlens_kv


def _is_capturing() -> bool:
    """True only while the current CUDA stream is being captured into a graph."""
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _table_device(device: Union[str, int, torch.device]) -> torch.device:
    """Return ``device`` as tensors created on it report their device.

    A CUDA device without an index places tensors on the current device, and tensors
    report CPU without an index, so ``torch.device("cuda")`` or ``"cpu:0"`` would never
    compare equal to an existing table's device.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if device.type == "cpu":
        return torch.device("cpu")
    return device


@dataclass(eq=False)
class PrebuiltPackedCPLayout:
    """Packed-CP DSA layout tables built before the model forward.

    ``query_positions`` (``[local_rows]``), ``kv_reorder_idx`` and its inverse
    permutation ``kv_reorder_inv_idx`` (both ``[local_rows * cp_size]``) are exactly
    the ``(positions, reorder, inverse)`` entry ``DSAttention.forward`` would otherwise
    memoize lazily under its ``"positions_and_reorder"`` key for ``geometry``;
    ``query_positions`` also serves the sequence-parallel ``"local_positions"`` request
    of the same geometry. The inverse is what the K/V and indexer-K reorders save for
    their gather backward. The tables are fixed-size for a fixed geometry, so a CUDA
    graph integration can capture them as static inputs and refresh them in place with
    :meth:`copy_` (see there for when that is safe).

    ``kv_reorder_inv_idx`` must invert ``kv_reorder_idx``. Build layouts with
    :func:`build_packed_cp_layout_from_host` (or :func:`prebuild_packed_cp_layout`),
    which take both from one host-builder call that checks this, and refresh them only
    with :meth:`copy_`, which replaces every table at once. :meth:`validate` checks the
    relation only for host-resident tables, since a device check would need a device
    read; a device layout whose inverse does not match its reorder gives a correct
    forward but wrong K/V and indexer-K gradients.

    Every tensor-valued field is a table: :meth:`named_tables`, :meth:`validate` and
    :meth:`copy_` iterate over the fields, and :meth:`_expected_table_numel` declares
    each table's length. An additional table is therefore one field plus one length
    entry, without changing the API.
    """

    _REQUIRED_TABLES = ("query_positions", "kv_reorder_idx", "kv_reorder_inv_idx")

    geometry: PackedCPLayoutGeometry
    query_positions: torch.Tensor
    kv_reorder_idx: torch.Tensor
    kv_reorder_inv_idx: torch.Tensor

    def __post_init__(self):
        self.validate()

    def _expected_table_numel(self) -> Dict[str, int]:
        local_rows = self.geometry.local_rows
        global_rows = local_rows * self.geometry.cp_size
        return {
            "query_positions": local_rows,
            "kv_reorder_idx": global_rows,
            "kv_reorder_inv_idx": global_rows,
        }

    def named_tables(self) -> Dict[str, torch.Tensor]:
        """Return every present table by field name, e.g. to register static graph inputs."""
        tables = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                tables[field.name] = value
        return tables

    @property
    def device(self) -> torch.device:
        """Device holding the tables."""
        return self.query_positions.device

    @property
    def nbytes(self) -> int:
        """Total size of the tables in bytes."""
        return sum(table.numel() * table.element_size() for table in self.named_tables().values())

    def validate(self) -> None:
        """Raise ``RuntimeError`` unless the tables are consistent with ``geometry``.

        These are invariants of a correctly built or refreshed layout; a violation is a
        bug in whoever produced or mutated it, so callers must not fall back silently.
        Every table's dtype, shape, device and contiguity are checked; for host-resident
        tables also that ``kv_reorder_inv_idx`` inverts ``kv_reorder_idx``. Device
        tables' values are never read, so this does not synchronize and is safe during a
        CUDA graph capture.
        """
        geometry = self.geometry
        if not isinstance(geometry, PackedCPLayoutGeometry):
            raise RuntimeError(
                f"Prebuilt packed-CP layout geometry must be a PackedCPLayoutGeometry, "
                f"got {type(geometry).__name__}."
            )
        if geometry.cp_size <= 1 or not 0 <= geometry.cp_rank < geometry.cp_size:
            raise RuntimeError(f"Prebuilt packed-CP layout has an invalid CP geometry: {geometry}.")
        if geometry.local_rows < 0:
            raise RuntimeError(f"Prebuilt packed-CP layout has negative local rows: {geometry}.")
        expected_numel = self._expected_table_numel()
        tables = self.named_tables()
        missing = sorted(set(self._REQUIRED_TABLES) - set(tables))
        if missing:
            raise RuntimeError(f"Prebuilt packed-CP layout is missing tables {missing}.")
        unknown = sorted(set(tables) - set(expected_numel))
        if unknown:
            raise RuntimeError(f"Prebuilt packed-CP layout has undeclared tables {unknown}.")
        device = self.device
        for name, table in tables.items():
            if (
                table.dtype != torch.int64
                or table.dim() != 1
                or table.numel() != expected_numel[name]
                or table.device != device
                or not table.is_contiguous()
            ):
                raise RuntimeError(
                    f"Prebuilt packed-CP layout table {name!r} is inconsistent with {geometry}: "
                    f"got dtype={table.dtype}, shape={tuple(table.shape)}, device={table.device}, "
                    f"contiguous={table.is_contiguous()}; expected a contiguous int64 vector of "
                    f"{expected_numel[name]} elements on {device}."
                )
        if device.type == "cpu":
            reorder, inverse = self.kv_reorder_idx, self.kv_reorder_inv_idx
            global_rows = reorder.numel()
            if global_rows and (
                int(inverse.min()) < 0
                or int(inverse.max()) >= global_rows
                or not torch.equal(reorder[inverse], torch.arange(global_rows))
            ):
                raise RuntimeError(
                    "Prebuilt packed-CP layout table 'kv_reorder_inv_idx' does not invert "
                    f"'kv_reorder_idx' for {geometry}. Build both with one "
                    "build_packed_cp_layout_from_host call and refresh them only with copy_()."
                )

    def copy_(self, source: "PrebuiltPackedCPLayout") -> "PrebuiltPackedCPLayout":
        """Refresh these tables in place from ``source``; keeps every table's storage.

        ``source`` must answer the same geometry (``cp_size``, ``cp_rank``,
        ``local_rows`` and both cover flags) with the same tables; it may live on
        another device, in which case ``Tensor.copy_`` performs the transfer. A
        mismatch raises instead of resizing or relabelling: a captured graph keeps
        reading the original storage, and DSA matches the geometry it derives from the
        PackedSeqParams it receives, so the static metadata must keep describing it.

        One call replaces every table, so the inverse reorder keeps inverting the
        reorder; never rewrite one table on its own (see the class docstring).

        The copy overwrites tables that autograd may have saved for a backward that
        has not run yet: the K/V and indexer-K reorders save the inverse reorder for
        their gather backward (the KPool gate reorder runs without grad and saves
        nothing). Refresh a layout only after the backward of every microbatch that
        consumed it has run: with separately captured forward and backward graphs and
        several microbatches in flight (1F1B or interleaved pipelining), keep one static
        layout per in-flight microbatch and geometry, not one per geometry. Eager
        autograd reports a premature refresh through its version counter; a graph
        replay silently reads the new tables.
        """
        if not isinstance(source, PrebuiltPackedCPLayout):
            raise TypeError(
                f"Can only refresh from a PrebuiltPackedCPLayout, got {type(source).__name__}."
            )
        if source is self:
            return self
        self.validate()
        source.validate()
        if self.geometry != source.geometry:
            raise ValueError(
                "Cannot refresh a prebuilt packed-CP layout across geometries: destination "
                f"{self.geometry}, source {source.geometry}."
            )
        destination_tables = self.named_tables()
        source_tables = source.named_tables()
        if destination_tables.keys() != source_tables.keys():
            raise ValueError(
                "Cannot refresh a prebuilt packed-CP layout with a different table set: "
                f"destination={sorted(destination_tables)}, source={sorted(source_tables)}."
            )
        # Equal geometries and validated tables imply equal shapes and dtypes.
        with torch.no_grad():
            for name, destination in destination_tables.items():
                destination.copy_(source_tables[name])
        return self


class _PrebuiltLayoutAttachment(NamedTuple):
    """A layout attached to a PackedSeqParams, bound to its host boundary lists."""

    layout: PrebuiltPackedCPLayout
    host_cu_seqlens_q: Optional[List[int]]
    host_cu_seqlens_kv: Optional[List[int]]


def build_packed_cp_layout_from_host(
    host_cu_seqlens_q: List[int],
    host_cu_seqlens_kv: List[int],
    geometry: PackedCPLayoutGeometry,
    device: torch.device,
) -> PrebuiltPackedCPLayout:
    """Build the packed-CP layout tables for ``geometry`` from host boundaries.

    This is the exact call ``DSAttention.forward`` makes for its
    ``"positions_and_reorder"`` memo entry, inverse reorder included, so the tables are
    bit-identical to the lazily built ones. The host builder inverts the reorder (and so
    checks that it is a permutation) before copying either table to ``device``.
    """
    local_rows = geometry.local_rows
    query_positions, kv_reorder_idx, kv_reorder_inv_idx = (
        build_packed_allgather_cp_query_positions_and_key_reorder_from_host(
            host_cu_seqlens_q,
            host_cu_seqlens_kv,
            cp_size=geometry.cp_size,
            cp_rank=geometry.cp_rank,
            device=torch.device(device),
            local_output_size=local_rows,
            key_local_output_size=local_rows,
            global_output_size=local_rows * geometry.cp_size,
            query_cu_seqlens_cover_output=geometry.query_cu_seqlens_cover_output,
            key_cu_seqlens_cover_output=geometry.key_cu_seqlens_cover_output,
            return_inverse_key_reorder=True,
        )
    )
    return PrebuiltPackedCPLayout(
        geometry=geometry,
        query_positions=query_positions,
        kv_reorder_idx=kv_reorder_idx,
        kv_reorder_inv_idx=kv_reorder_inv_idx,
    )


def resolve_packed_cp_layout_geometry(
    packed_seq_params: Optional[PackedSeqParams],
    *,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    local_rows: Optional[int] = None,
) -> Optional[PackedCPLayoutGeometry]:
    """Return the layout geometry DSA requests for ``packed_seq_params``, or ``None``.

    ``cp_size`` and ``cp_rank`` come from ``cp_group``, by default the realized group
    on ``packed_seq_params.cp_group``. ``local_rows`` defaults to
    ``thd_cp_host_cu_seqlens_q[-1] // cp_size``, the row count the THD CP routes
    assume; pass it when the packed tensors carry rows the boundaries do not cover. The
    cover flags are derived as ``DSAttention.forward`` derives them. ``None`` means no
    packed-CP layout applies: not packed THD, CP size 1, or ``local_rows`` unknown.
    """
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        return None
    if cp_group is None:
        cp_group = getattr(packed_seq_params, "cp_group", None)
    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size <= 1:
        return None
    if local_rows is None:
        host_cu_seqlens_q, _ = _host_cu_seqlens(packed_seq_params)
        if host_cu_seqlens_q is None or host_cu_seqlens_q[-1] % cp_size != 0:
            return None
        local_rows = host_cu_seqlens_q[-1] // cp_size
    local_rows = int(local_rows)
    query_cover, key_cover = resolve_packed_cp_cover_output_flags(
        packed_seq_params, local_rows * cp_size
    )
    return PackedCPLayoutGeometry(
        cp_size=cp_size,
        cp_rank=cp_group.rank(),
        local_rows=local_rows,
        query_cu_seqlens_cover_output=query_cover,
        key_cu_seqlens_cover_output=key_cover,
    )


def get_prebuilt_packed_cp_layout(
    packed_seq_params: Optional[PackedSeqParams],
) -> Optional[PrebuiltPackedCPLayout]:
    """Return the prebuilt packed-CP layout attached to ``packed_seq_params``, if current.

    A layout is bound to the host boundary lists (``thd_cp_host_cu_seqlens_{q,kv}``)
    present when it was attached. ``prebuild_thd_cp_partition_routes`` records new lists
    on every call, so re-preparing the metadata without prebuilding the layout again
    leaves it stale, and ``None`` is returned: DSA then never consumes tables built for
    other boundaries. Boundaries mutated in place are not detected; re-prepare instead.
    """
    if packed_seq_params is None:
        return None
    attachment = getattr(packed_seq_params, _PREBUILT_PACKED_CP_LAYOUT_ATTR, None)
    if attachment is None:
        return None
    host_cu_seqlens_q, host_cu_seqlens_kv = _host_cu_seqlens(packed_seq_params)
    if (
        host_cu_seqlens_q is not attachment.host_cu_seqlens_q
        or host_cu_seqlens_kv is not attachment.host_cu_seqlens_kv
    ):
        return None
    return attachment.layout


def attach_prebuilt_packed_cp_layout(
    packed_seq_params: PackedSeqParams, layout: Optional[PrebuiltPackedCPLayout]
) -> None:
    """Attach ``layout`` to ``packed_seq_params``; ``None`` detaches any existing layout.

    The layout is bound to the host boundary lists currently recorded on
    ``packed_seq_params`` (see :func:`get_prebuilt_packed_cp_layout`), so it must be
    built for exactly those boundaries, or, on metadata without host boundaries (such as
    a PackedSeqParams a graph integration rebuilds around static buffers), for the
    boundaries its ``cu_seqlens`` hold. Its tables must come from one
    :func:`build_packed_cp_layout_from_host` call, the inverse reorder included:
    :meth:`PrebuiltPackedCPLayout.validate` cannot check device tables' values.
    """
    if layout is None:
        if hasattr(packed_seq_params, _PREBUILT_PACKED_CP_LAYOUT_ATTR):
            delattr(packed_seq_params, _PREBUILT_PACKED_CP_LAYOUT_ATTR)
        return
    if not isinstance(layout, PrebuiltPackedCPLayout):
        raise TypeError(f"Expected a PrebuiltPackedCPLayout, got {type(layout).__name__}.")
    layout.validate()
    setattr(
        packed_seq_params,
        _PREBUILT_PACKED_CP_LAYOUT_ATTR,
        _PrebuiltLayoutAttachment(layout, *_host_cu_seqlens(packed_seq_params)),
    )


def copy_prebuilt_packed_cp_layout_(
    destination: PackedSeqParams, source: PackedSeqParams
) -> PrebuiltPackedCPLayout:
    """Refresh ``destination``'s DSA packed-CP state in place from ``source``.

    For graph integrations that keep static ``PackedSeqParams`` across replays. It
    copies the tables of ``source``'s current layout into the layout attached to
    ``destination`` (storage kept; see :meth:`PrebuiltPackedCPLayout.copy_` for the
    compatibility rules and for when a refresh is safe), points ``destination``'s host
    boundary lists at ``source``'s and re-binds the layout to them, and drops
    ``destination``'s lazy DSA layout memo, whose entries describe the previous pack.
    ``cu_seqlens`` buffers, CP routes and other metadata of ``destination`` remain the
    caller's to refresh.
    """
    # The raw attachment: refreshing is also how a stale destination becomes current.
    attachment = getattr(destination, _PREBUILT_PACKED_CP_LAYOUT_ATTR, None)
    source_layout = get_prebuilt_packed_cp_layout(source)
    if attachment is None or source_layout is None:
        raise RuntimeError(
            "Both PackedSeqParams must carry a prebuilt packed-CP layout "
            "(see prebuild_packed_cp_layout) to refresh one from the other."
        )
    layout = attachment.layout.copy_(source_layout)
    for name, value in zip(_HOST_CU_SEQLENS_ATTRS, _host_cu_seqlens(source)):
        setattr(destination, name, value)
    if hasattr(destination, PACKED_CP_LAYOUT_MEMO_ATTR):
        delattr(destination, PACKED_CP_LAYOUT_MEMO_ATTR)
    attach_prebuilt_packed_cp_layout(destination, layout)
    return layout


def prebuild_packed_cp_layout(
    packed_seq_params: Optional[PackedSeqParams],
    *,
    local_rows: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Optional[PrebuiltPackedCPLayout]:
    """Build and attach DSA's packed-CP layout tables at batch-construction time.

    Call after ``prebuild_thd_cp_partition_routes`` (``finalize_packed_seq_params``),
    which resolves ``packed_seq_params.cp_group`` and records the host ``cu_seqlens``
    used here. The tables are built for that realized group, so under dynamic CP they
    follow each microbatch's group; their sizes are ``local_rows`` and
    ``local_rows * cp_size``, i.e. static only for a fixed CP size and row count.

    Idempotent: a layout already attached for the same boundaries, geometry and device
    is returned as-is, so metadata prepared again (for example a static PackedSeqParams
    that is prepared inside every captured step) keeps its storage and builds nothing.
    Nothing is built during a CUDA graph capture, where the host-to-device copy would
    be captured from a transient staging buffer; DSA then reports the missing layout.

    Args:
        packed_seq_params: Per-microbatch metadata; the layout is attached to it.
        local_rows: CP-local physical rows of the packed tensors. Defaults to
            ``cu_seqlens[-1] // cp_size``, the row count the THD CP routes assume.
            Pass it explicitly when the tensors carry rows the boundaries do not cover.
        device: Destination of the tables. Defaults to the ``cu_seqlens`` device; a
            CUDA device without an index means the current device, as for tensor
            creation.

    Returns:
        The attached layout, or ``None`` (any stale layout detached) when the
        microbatch is not packed THD with CP > 1, the host boundaries were not
        recorded, ``local_rows`` cannot be inferred, or a capture is in progress.
        ``DSAttention`` then builds the layout lazily in an eager forward, as without
        prebuilding.
    """
    if packed_seq_params is None:
        return None
    geometry = resolve_packed_cp_layout_geometry(packed_seq_params, local_rows=local_rows)
    host_cu_seqlens_q, host_cu_seqlens_kv = _host_cu_seqlens(packed_seq_params)
    if geometry is None or host_cu_seqlens_q is None or host_cu_seqlens_kv is None:
        attach_prebuilt_packed_cp_layout(packed_seq_params, None)
        return None
    if device is None:
        device = get_packed_qk_cu_seqlens(packed_seq_params)[0].device
    device = _table_device(device)
    current = get_prebuilt_packed_cp_layout(packed_seq_params)
    if current is not None and current.geometry == geometry and current.device == device:
        return current
    attach_prebuilt_packed_cp_layout(packed_seq_params, None)
    if _is_capturing():
        return None
    with nvtx_range("dsa/prebuild_packed_cp_layout"):
        layout = build_packed_cp_layout_from_host(
            host_cu_seqlens_q, host_cu_seqlens_kv, geometry, device
        )
    attach_prebuilt_packed_cp_layout(packed_seq_params, layout)
    return layout
