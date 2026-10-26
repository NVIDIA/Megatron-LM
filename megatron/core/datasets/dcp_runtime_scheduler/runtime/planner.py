# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU geometry, capacity and validated MCore rank-row contracts."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..models.cache import PackCostModel
from .timeline import build_timeline


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
    align_group_starts: bool = False

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

    def supports_group(self, start: int, cp: int) -> bool:
        """Whether this contiguous interval is supported by the runtime's groups."""
        return (
            cp in self.allowed_cp_sizes
            and 0 <= start <= self.world_size - cp
            and (not self.align_group_starts or start % cp == 0)
        )


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
            if not geometry.supports_group(start, cp):
                raise RuntimeScheduleError(
                    f"microbatch {m}: cp={cp} group at rank {start} requires a size-aligned start"
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


class RuntimePlanner(ABC):
    """Common model injection, validation, pricing and rank-row contract for both solvers."""

    def __init__(self, geometry, cost_model) -> None:
        geometry = replace(geometry, allowed_cp_sizes=tuple(sorted(set(geometry.allowed_cp_sizes))))
        self.geometry = geometry
        self.model = cost_model
        cost_model.check_domain(geometry, None)
        self.timeline = build_timeline(cost_model, geometry)
        prepare = getattr(cost_model, "prepare_guide", None)
        if prepare is not None:
            prepare(geometry.allowed_cp_sizes)
        self.costs = self._new_costs()
        self.last_statistics = {}

    @abstractmethod
    def plan(self, lengths, *, incumbent_rounds=None):
        """Search with the injected cost model and return a validated RuntimePlan."""

    def _new_costs(self):
        return PackCostModel(
            self.geometry,
            self.model.forward,
            self.model.backward,
            getattr(self.model, "program_costs", None),
            placement_period=(
                None
                if getattr(self.model, "placement_invariant", False)
                else self.geometry.world_size
            ),
            max_group_seqs=self.geometry.max_group_seqs or 64,
        )

    def _start_plan(self, lengths):
        lengths = dict(lengths)
        if not lengths or any(
            type(i) is not int or i < 0 or type(n) is not int or n <= 0 for i, n in lengths.items()
        ):
            raise ValueError(
                "lengths must map nonnegative integer IDs to positive integer logical lengths"
            )
        self.model.clear()
        self.costs = self._new_costs()
        return lengths

    def _make_plan(self, rounds, lengths, predicted, begin):
        rows = encode_rows(rounds, self.geometry.world_size)
        groups = validate_rows(rows, lengths, self.geometry, min(self.geometry.allowed_cp_sizes))
        return RuntimePlan(
            rows=rows,
            groups=groups,
            predicted_ms=predicted,
            planning_s=time.perf_counter() - begin,
            microbatches=len(rows),
            dummy_tokens=dummy_tokens_per_group(groups, lengths, self.geometry),
        )

    def _costed(self, rounds, lengths, guide=None):
        out = []
        for row in rounds:
            start, groups = 0, []
            for group in row:
                cp = group["cp"]
                values = tuple(lengths[i] for i in group["sample_ids"])
                value = (
                    (0.0, 0.0, guide(values, cp, ranks=tuple(range(start, start + cp))))
                    if guide is not None
                    else self.costs.pack(values, cp, start)
                )
                if value is None:
                    raise RuntimeScheduleError(f"cost model rejected CP{cp} pack {values}")
                groups.append((start, cp, *value))
                start += cp
            out.append(groups)
        return out

    def plan_padded(self, padded_lengths, *, logical_lengths=None, incumbent_rows=None):
        """MCore input [(global_sample_id, dataset-padded length)]; pure CPU.

        No LM shift is inferred. Exact logical lengths are optional and must ceil
        to the supplied padded lengths. Otherwise the cost input is explicitly
        the padded upper bound, not fabricated logical lengths.
        """
        items = list(padded_lengths.items() if hasattr(padded_lengths, "items") else padded_lengths)
        padded = dict(items)
        if (
            len(padded) != len(items)
            or not padded
            or any(
                type(i) is not int
                or i < 0
                or type(n) is not int
                or n <= 0
                or n % self.geometry.dataset_alignment
                for i, n in items
            )
        ):
            raise ValueError("unique nonnegative IDs and dataset-aligned positive lengths required")
        logical = padded if logical_lengths is None else dict(logical_lengths)
        if logical.keys() != padded.keys() or any(
            type(n) is not int
            or n <= 0
            or (n + self.geometry.dataset_alignment - 1)
            // self.geometry.dataset_alignment
            * self.geometry.dataset_alignment
            != padded[i]
            for i, n in logical.items()
        ):
            raise ValueError(
                "logical lengths must match every padded input after dataset alignment"
            )
        incumbent = None
        if incumbent_rows is not None:
            groups = validate_rows(
                incumbent_rows, padded, self.geometry, min(self.geometry.allowed_cp_sizes)
            )
            incumbent = [[dict(cp=cp, sample_ids=ids) for _, cp, ids in row] for row in groups]
        plan = self.plan(logical, incumbent_rounds=incumbent)
        validate_rows(plan.rows, padded, self.geometry, min(self.geometry.allowed_cp_sizes))
        self.last_statistics["cost_length_source"] = (
            "after_lm_logical" if logical_lengths is not None else "dataset_padded_upper_bound"
        )
        return plan
