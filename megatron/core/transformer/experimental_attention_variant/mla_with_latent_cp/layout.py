# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed zigzag layout validation and attention phase planning."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Literal, Protocol

import torch
import torch.distributed as dist
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams

from .utils import _require


@dataclass(frozen=True)
class PhaseSpec:
    """One compact attention matrix in the zigzag ring schedule."""

    phase: int
    owner: int
    kind: Literal["diagonal", "lower", "upper"]
    q_indices: Tensor
    kv_indices: Tensor
    cu_seqlens_q: Tensor
    cu_seqlens_kv: Tensor
    max_seqlen_q: int
    max_seqlen_kv: int
    causal: bool
    scatter_indices: Tensor | None = None
    q_slice: tuple[int, int] | None = None
    kv_slice: tuple[int, int] | None = None
    scatter_slice: tuple[int, int] | None = None


@dataclass(frozen=True)
class ZigZagLayout:
    """Validated view of already-zigzag THD storage."""

    cp_size: int
    cp_rank: int
    local_tokens: int
    cu_global: Tensor
    cu_full: Tensor
    cu_half: Tensor
    max_global: int
    max_full: int
    max_half: int
    front_indices: Tensor
    back_indices: Tensor
    phases: tuple[PhaseSpec, ...]


@dataclass(frozen=True)
class _LayoutCacheKey:
    """Semantic identity for reusable device-side phase metadata."""

    cu_seqlens: tuple[int, ...]
    local_tokens: int
    cp_size: int
    cp_rank: int
    device: torch.device
    max_global: int


class LatentCPLayoutAdapter(Protocol):
    """Extension seam for a future contiguous-to-zigzag layout conversion."""

    def prepare(
        self,
        local_hidden: Tensor,
        packed_seq_params: PackedSeqParams,
        cp_group: dist.ProcessGroup,
        *,
        tp_group: dist.ProcessGroup | None = None,
        sequence_parallel: bool = False,
    ) -> ZigZagLayout:
        """Validate layout metadata and return the per-rank phase plan."""
        ...


def _cu_from_lengths(lengths: Tensor) -> Tensor:
    zero = torch.zeros(1, dtype=torch.int32, device=lengths.device)
    cumulative = torch.cumsum(lengths, dim=0, dtype=torch.int32)
    return torch.cat((zero, cumulative)).contiguous()


def _packed_half_indices(
    local_lengths: tuple[int, ...], device: torch.device
) -> tuple[Tensor, Tensor]:
    """Return front/back row indices for per-sequence [F_r, B_r] storage."""

    front: list[Tensor] = []
    back: list[Tensor] = []
    offset = 0
    for length in local_lengths:
        half = length // 2
        front.append(
            torch.arange(offset, offset + half, dtype=torch.long, device=device)
        )
        back.append(
            torch.arange(
                offset + half, offset + length, dtype=torch.long, device=device
            )
        )
        offset += length

    if not front:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    return torch.cat(front), torch.cat(back)


def build_zigzag_layout(
    cu_global: Tensor,
    local_tokens: int,
    cp_size: int,
    cp_rank: int,
    *,
    max_global: int | None = None,
    cu_values: tuple[int, ...] | None = None,
) -> ZigZagLayout:
    """Validate packed ownership and build the three-shape zigzag phase schedule.

    cu_global always describes original global sequences. Derived cumulative lengths are
    backend-only metadata and must never be passed to RoPE.
    """

    _require(cu_global.ndim == 1 and cu_global.numel() >= 2, "cu_seqlens must be 1-D")
    _require(cu_global.dtype == torch.int32, "cu_seqlens must have dtype torch.int32")
    _require(cp_size > 0 and 0 <= cp_rank < cp_size, "invalid CP rank or size")
    if cu_values is None:
        cu_values = tuple(
            cu_global.detach().to(device="cpu", dtype=torch.long).tolist()
        )
    _require(len(cu_values) == cu_global.numel(), "host cu_seqlens size mismatch")
    _require(cu_values[0] == 0, "cu_seqlens must start at zero")

    global_lengths_values = tuple(
        stop - start for start, stop in zip(cu_values[:-1], cu_values[1:])
    )
    _require(
        all(length > 0 for length in global_lengths_values),
        "empty packed sequences are unsupported",
    )
    if cp_size > 1:
        _require(
            all(length % (2 * cp_size) == 0 for length in global_lengths_values),
            f"every global packed length must be divisible by 2*CP ({2 * cp_size})",
        )

    local_length_values = tuple(length // cp_size for length in global_lengths_values)
    # CP=1 is the exact no-ring degeneration. It has only a full/full diagonal
    # phase, so no artificial half-sequence divisibility requirement is needed.
    half_length_values = (
        tuple(length // (2 * cp_size) for length in global_lengths_values)
        if cp_size > 1
        else local_length_values
    )
    local_lengths = torch.tensor(
        local_length_values, dtype=torch.int32, device=cu_global.device
    )
    half_lengths = torch.tensor(
        half_length_values, dtype=torch.int32, device=cu_global.device
    )
    cu_full = _cu_from_lengths(local_lengths)
    cu_half = _cu_from_lengths(half_lengths)
    _require(
        sum(local_length_values) == local_tokens,
        "hidden token count disagrees with metadata",
    )

    full_indices = torch.arange(local_tokens, dtype=torch.long, device=cu_global.device)
    if cp_size > 1:
        front_indices, back_indices = _packed_half_indices(
            local_length_values, cu_global.device
        )
        if len(local_length_values) == 1:
            half = local_length_values[0] // 2
            front_slice = (0, half)
            back_slice = (half, local_tokens)
        else:
            front_slice = None
            back_slice = None
    else:
        front_indices = full_indices
        back_indices = torch.empty(0, dtype=torch.long, device=cu_global.device)
        front_slice = (0, local_tokens)
        back_slice = (0, 0)
    derived_max_global = max(global_lengths_values)
    if max_global is not None:
        _require(
            max_global == derived_max_global, "max_seqlen disagrees with cu_seqlens"
        )
    max_global = derived_max_global
    max_full = max(local_length_values)
    max_half = max(half_length_values)
    full_slice = (0, local_tokens)

    phases: list[PhaseSpec] = []
    for phase in range(cp_size):
        owner = (cp_rank - phase) % cp_size
        if phase == 0:
            phases.append(
                PhaseSpec(
                    phase,
                    owner,
                    "diagonal",
                    full_indices,
                    full_indices,
                    cu_full,
                    cu_full,
                    max_full,
                    max_full,
                    True,
                    q_slice=full_slice,
                    kv_slice=full_slice,
                )
            )
        elif phase <= cp_rank:
            phases.append(
                PhaseSpec(
                    phase,
                    owner,
                    "lower",
                    full_indices,
                    front_indices,
                    cu_full,
                    cu_half,
                    max_full,
                    max_half,
                    False,
                    q_slice=full_slice,
                    kv_slice=front_slice,
                )
            )
        else:
            phases.append(
                PhaseSpec(
                    phase,
                    owner,
                    "upper",
                    back_indices,
                    full_indices,
                    cu_half,
                    cu_full,
                    max_half,
                    max_full,
                    False,
                    scatter_indices=back_indices,
                    q_slice=back_slice,
                    kv_slice=full_slice,
                    scatter_slice=back_slice,
                )
            )

    return ZigZagLayout(
        cp_size=cp_size,
        cp_rank=cp_rank,
        local_tokens=local_tokens,
        cu_global=cu_global,
        cu_full=cu_full,
        cu_half=cu_half,
        max_global=max_global,
        max_full=max_full,
        max_half=max_half,
        front_indices=front_indices,
        back_indices=back_indices,
        phases=tuple(phases),
    )


class AlreadyZigZagTHDAdapter:
    """V1 layout adapter: validate an input that is already zigzag-partitioned."""

    _CACHE_CAPACITY = 16

    def __init__(self) -> None:
        self._layout_cache: OrderedDict[_LayoutCacheKey, ZigZagLayout] = OrderedDict()

    def _cached_layout(
        self,
        cu_global: Tensor,
        cu_values: tuple[int, ...],
        local_tokens: int,
        cp_size: int,
        cp_rank: int,
        max_global: int,
    ) -> ZigZagLayout:
        key = _LayoutCacheKey(
            cu_values, local_tokens, cp_size, cp_rank, cu_global.device, max_global
        )
        cached = self._layout_cache.pop(key, None)
        if cached is not None:
            self._layout_cache[key] = cached
            return replace(cached, cu_global=cu_global)

        layout = build_zigzag_layout(
            cu_global,
            local_tokens,
            cp_size,
            cp_rank,
            max_global=max_global,
            cu_values=cu_values,
        )
        self._layout_cache[key] = layout
        if len(self._layout_cache) > self._CACHE_CAPACITY:
            self._layout_cache.popitem(last=False)
        return layout

    def prepare(
        self,
        local_hidden: Tensor,
        packed_seq_params: PackedSeqParams,
        cp_group: dist.ProcessGroup,
        *,
        tp_group: dist.ProcessGroup | None = None,
        sequence_parallel: bool = False,
    ) -> ZigZagLayout:
        """Validate already-zigzag THD metadata and build its phase plan."""
        _require(packed_seq_params.qkv_format == "thd", "only THD format is supported")
        _require(
            packed_seq_params.cp_partition_mode == "zigzag",
            "only an already-zigzag CP partition is supported",
        )
        cu_q = packed_seq_params.cu_seqlens_q
        cu_kv = packed_seq_params.cu_seqlens_kv
        _require(
            isinstance(cu_q, Tensor) and isinstance(cu_kv, Tensor), "missing cu_seqlens"
        )
        _require(
            cu_q.is_cuda
            and cu_kv.is_cuda
            and cu_q.device == local_hidden.device
            and cu_kv.device == local_hidden.device,
            "cu_seqlens must be CUDA tensors colocated with hidden_states",
        )
        _require(
            cu_q.dtype == torch.int32 and cu_kv.dtype == torch.int32,
            "both Q and KV cu_seqlens must have dtype torch.int32",
        )
        _require(
            cu_q.is_contiguous() and cu_kv.is_contiguous(),
            "cu_seqlens must be contiguous",
        )
        _require(
            cu_q is cu_kv or torch.equal(cu_q, cu_kv),
            "self-attention requires equal Q/KV cu_seqlens",
        )
        route = packed_seq_params.cp_partition_route
        route_source = (
            packed_seq_params.cu_seqlens_q_padded
            if packed_seq_params.cu_seqlens_q_padded is not None
            else cu_q
        )
        if route is not None:
            _require(
                packed_seq_params.pad_between_seqs is False,
                "scheduler must reject inter-sequence/tail padding before route use",
            )
        else:
            for padded, valid, name in (
                (packed_seq_params.cu_seqlens_q_padded, cu_q, "Q"),
                (packed_seq_params.cu_seqlens_kv_padded, cu_kv, "KV"),
            ):
                _require(
                    padded is None or padded is valid or torch.equal(padded, valid),
                    f"{name} inter-sequence/tail padding is unsupported",
                )
        max_q = packed_seq_params.max_seqlen_q
        max_kv = packed_seq_params.max_seqlen_kv
        _require(
            isinstance(max_q, int)
            and not isinstance(max_q, bool)
            and isinstance(max_kv, int)
            and not isinstance(max_kv, bool)
            and max_q > 0
            and max_kv > 0,
            "Q and KV max_seqlen must be positive Python integers",
        )
        _require(max_q == max_kv, "self-attention requires equal Q/KV max_seqlen")
        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)
        if route is not None:
            _require(route.cp_size == cp_size, "stale THD route CP size")
            _require(route.cp_rank == cp_rank, "stale THD route CP rank")
            _require(
                route.source_cu_seqlens_id == id(route_source),
                "stale THD route metadata identity",
            )
            _require(bool(route.cu_seqlens), "THD route is missing host cu_seqlens")
            cu_values = route.cu_seqlens
        else:
            cu_values = tuple(cu_q.detach().to(device="cpu", dtype=torch.long).tolist())
        local_tokens = local_hidden.size(0)
        if sequence_parallel:
            _require(tp_group is not None, "sequence parallelism requires a TP group")
            local_tokens *= dist.get_world_size(tp_group)
        return self._cached_layout(
            cu_q, cu_values, local_tokens, cp_size, cp_rank, max_q
        )
