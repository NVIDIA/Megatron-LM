# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed zigzag layout validation and attention phase planning."""

from __future__ import annotations

from dataclasses import dataclass
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


def _packed_half_indices(local_lengths: Tensor) -> tuple[Tensor, Tensor]:
    """Return front/back row indices for per-sequence [F_r, B_r] storage."""

    device = local_lengths.device
    starts = _cu_from_lengths(local_lengths)[:-1]
    halves = local_lengths // 2
    front = [
        torch.arange(start, start + half, dtype=torch.long, device=device)
        for start, half in zip(starts.unbind(), halves.unbind())
    ]
    back = [
        torch.arange(start + half, start + length, dtype=torch.long, device=device)
        for start, half, length in zip(
            starts.unbind(), halves.unbind(), local_lengths.unbind()
        )
    ]
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
) -> ZigZagLayout:
    """Validate packed ownership and build the three-shape zigzag phase schedule.

    cu_global always describes original global sequences. Derived cumulative lengths are
    backend-only metadata and must never be passed to RoPE.
    """

    _require(cu_global.ndim == 1 and cu_global.numel() >= 2, "cu_seqlens must be 1-D")
    _require(cu_global.dtype == torch.int32, "cu_seqlens must have dtype torch.int32")
    _require(cp_size > 0 and 0 <= cp_rank < cp_size, "invalid CP rank or size")
    _require(int(cu_global[0].item()) == 0, "cu_seqlens must start at zero")

    global_lengths = cu_global[1:] - cu_global[:-1]
    _require(
        bool(torch.all(global_lengths > 0).item()),
        "empty packed sequences are unsupported",
    )
    if cp_size > 1:
        _require(
            bool(torch.all(torch.remainder(global_lengths, 2 * cp_size) == 0).item()),
            f"every global packed length must be divisible by 2*CP ({2 * cp_size})",
        )

    local_lengths = torch.div(global_lengths, cp_size, rounding_mode="floor")
    # CP=1 is the exact no-ring degeneration. It has only a full/full diagonal
    # phase, so no artificial half-sequence divisibility requirement is needed.
    half_lengths = (
        torch.div(global_lengths, 2 * cp_size, rounding_mode="floor")
        if cp_size > 1
        else local_lengths
    )
    cu_full = _cu_from_lengths(local_lengths)
    cu_half = _cu_from_lengths(half_lengths)
    _require(
        int(cu_full[-1].item()) == local_tokens,
        "hidden token count disagrees with metadata",
    )

    full_indices = torch.arange(local_tokens, dtype=torch.long, device=cu_global.device)
    if cp_size > 1:
        front_indices, back_indices = _packed_half_indices(local_lengths)
    else:
        front_indices = full_indices
        back_indices = torch.empty(0, dtype=torch.long, device=cu_global.device)
    derived_max_global = int(global_lengths.max().item())
    if max_global is not None:
        _require(
            max_global == derived_max_global, "max_seqlen disagrees with cu_seqlens"
        )
    max_global = derived_max_global
    max_full = int(local_lengths.max().item())
    max_half = int(half_lengths.max().item())

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
            torch.equal(cu_q, cu_kv), "self-attention requires equal Q/KV cu_seqlens"
        )
        for padded, valid, name in (
            (packed_seq_params.cu_seqlens_q_padded, cu_q, "Q"),
            (packed_seq_params.cu_seqlens_kv_padded, cu_kv, "KV"),
        ):
            _require(
                padded is None or torch.equal(padded, valid),
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
        local_tokens = local_hidden.size(0)
        if sequence_parallel:
            _require(tp_group is not None, "sequence parallelism requires a TP group")
            local_tokens *= dist.get_world_size(tp_group)
        return build_zigzag_layout(
            cu_q, local_tokens, cp_size, cp_rank, max_global=max_q
        )
