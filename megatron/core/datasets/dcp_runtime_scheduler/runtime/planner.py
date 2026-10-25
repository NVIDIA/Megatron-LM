# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU geometry, capacity and validated MCore rank-row contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


class RuntimeScheduleError(RuntimeError):
    """Raised when the runtime scheduler cannot produce a valid schedule (every rank raises the same error in Megatron)."""


def _ceil_to(value: int, unit: int) -> int:
    return (int(value) + unit - 1) // unit * unit


@dataclass(frozen=True)
class RuntimePlanGeometry:
    """Static geometry of the DPxCP domain; also the ``config`` object PackCostModel expects."""

    world_size: int
    tokens_per_rank: int
    ep_size: int
    allowed_cp_sizes: Tuple[int, ...]
    dataset_alignment: int
    max_group_seqs: Optional[int] = None

    def __post_init__(self):
        if self.world_size < 1 or self.tokens_per_rank < 1:
            raise ValueError("world_size and tokens_per_rank must be >= 1")
        if self.ep_size < 1 or self.world_size % self.ep_size:
            raise ValueError(
                f"expert_model_parallel_size ({self.ep_size}) must divide dp_size*cp_size ({self.world_size})"
            )
        if not self.allowed_cp_sizes or any(
            not 1 <= cp <= self.world_size for cp in self.allowed_cp_sizes
        ):
            raise ValueError(
                f"allowed_cp_sizes must be within 1..{self.world_size}: {self.allowed_cp_sizes}"
            )
        if self.dataset_alignment < 1 or self.dataset_alignment % 2:
            raise ValueError("dataset_alignment must be a positive even number")
        if self.max_group_seqs is not None and self.max_group_seqs < 1:
            raise ValueError("max_group_seqs must be >= 1 when set")

    def stored_length(self, length: int, cp: int) -> int:
        """Tokens one sequence occupies in a CP-``cp`` pack: the dataset padding (``dataset_alignment``) followed by MCore's
        re-padding to a multiple of ``2*cp`` (``_pack_sequences``)."""
        aligned = _ceil_to(int(length), self.dataset_alignment)
        return _ceil_to(aligned, 2 * int(cp))

    def capacity(self, cp: int) -> int:
        """Physical tokens of a CP-``cp`` microbatch pack."""
        return int(cp) * self.tokens_per_rank


@dataclass
class RuntimePlan:
    """One planned global batch (rows are what MCore consumes; the rest is telemetry)."""

    rows: List[List[List[int]]]
    groups: List[List[Tuple[int, int, List[int]]]]
    predicted_ms: float
    planning_s: float
    microbatches: int
    dummy_tokens: List[List[int]] = field(default_factory=list)


def encode_rows(rounds, world_size: int) -> List[List[List[int]]]:
    """``sched`` rounds (``[[{'cp', 'sample_ids'}]]``) -> MCore rows (``[mb][rank] -> ids``)."""
    rows = []
    for row in rounds:
        encoded = [list(group["sample_ids"]) for group in row for _ in range(int(group["cp"]))]
        if len(encoded) != world_size:
            raise RuntimeScheduleError(
                f"scheduler row covers {len(encoded)} ranks, expected {world_size}: "
                f"{[(g['cp'], len(g['sample_ids'])) for g in row]}"
            )
        rows.append(encoded)
    return rows


def decode_groups(row: Sequence[Sequence[int]]) -> List[Tuple[int, int, List[int]]]:
    """One MCore row -> ``[(start, cp, ids)]`` (maximal runs of equal id lists, rank order)."""
    groups = []
    first = 0
    n = len(row)
    while first < n:
        ids = list(row[first])
        end = first + 1
        while end < n and list(row[end]) == ids:
            end += 1
        groups.append((first, end - first, ids))
        first = end
    return groups


def validate_rows(
    rows: Sequence[Sequence[Sequence[int]]],
    lengths: Dict[int, int],
    geometry: RuntimePlanGeometry,
    min_cp_size: int,
) -> List[List[Tuple[int, int, List[int]]]]:
    """Check every invariant MCore relies on; return the decoded groups per microbatch.

    Raises :class:`RuntimeScheduleError` naming the microbatch, rank and reason instead of letting MCore fail on an
    IndexError / assert deep inside ``build_packed_microbatches``.
    """
    if not rows:
        raise RuntimeScheduleError("runtime scheduler produced no microbatch")
    world = geometry.world_size
    allowed = set(geometry.allowed_cp_sizes)
    seen: List[int] = []
    decoded = []
    for m, row in enumerate(rows):
        if len(row) != world:
            raise RuntimeScheduleError(
                f"microbatch {m}: row has {len(row)} rank entries, expected {world}"
            )
        groups = decode_groups(row)
        for start, cp, ids in groups:
            if not ids:
                raise RuntimeScheduleError(
                    f"microbatch {m}: rank {start} has no sequence (every rank must pack >= 1)"
                )
            if cp not in allowed or cp < min_cp_size:
                raise RuntimeScheduleError(
                    f"microbatch {m}: group at rank {start} has cp={cp}, allowed {sorted(allowed)} "
                    f"and min_dynamic_context_parallel_size={min_cp_size}"
                )
            if geometry.max_group_seqs is not None and len(ids) > geometry.max_group_seqs:
                raise RuntimeScheduleError(
                    f"microbatch {m}: group at rank {start} packs {len(ids)} sequences, "
                    f"cap is {geometry.max_group_seqs} (thd_max_packed_sequences)"
                )
            stored = 0
            for i in ids:
                if i not in lengths:
                    raise RuntimeScheduleError(
                        f"microbatch {m}: unknown sample id {i} at rank {start}"
                    )
                stored += geometry.stored_length(lengths[i], cp)
            if stored > geometry.capacity(cp):
                raise RuntimeScheduleError(
                    f"microbatch {m}: group at rank {start} (cp={cp}) stores {stored} tokens, capacity {geometry.capacity(cp)}"
                )
            seen.extend(ids)
        # a run of equal lists is contiguous by construction; the only way MCore's member_ranks check can fail is an id
        # appearing in two non-adjacent groups, which the multiset check below catches as a duplicate
        decoded.append(groups)
    if sorted(seen) != sorted(lengths):
        counts: Dict[int, int] = {}
        for i in seen:
            counts[i] = counts.get(i, 0) + 1
        missing = sorted(i for i in lengths if i not in counts)
        extra = sorted(i for i, c in counts.items() if c > 1 or i not in lengths)
        raise RuntimeScheduleError(
            f"schedule lost or duplicated samples: missing {missing}, duplicated/unknown {extra}"
        )
    return decoded


def dummy_tokens_per_group(
    groups, lengths: Dict[int, int], geometry: RuntimePlanGeometry
) -> List[List[int]]:
    """Padding MCore materialises per group under pad 'max' + append_dummy_seq (telemetry)."""
    return [
        [
            geometry.capacity(cp) - sum(geometry.stored_length(lengths[i], cp) for i in ids)
            for _, cp, ids in row
        ]
        for row in groups
    ]
