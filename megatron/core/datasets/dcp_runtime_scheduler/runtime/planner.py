# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU geometry, capacity and validated MCore rank-row contracts."""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..models.cache import PackCostModel
from .timeline import build_timeline

# The packaged cost models price packs of at most this many sequences (their measurements
# used 64 THD metadata slots). It is a prediction-domain limit, not an execution limit:
# the execution limit is thd_max_packed_sequences, when the user sets it.
MAX_PRICED_GROUP_SEQUENCES = 64


class RuntimeScheduleError(RuntimeError):
    """Invalid runtime schedule; every Megatron rank raises the same error."""


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
    max_cp_size: Optional[int] = None

    def __post_init__(self):
        if self.world_size < 1 or self.tokens_per_rank < 1:
            raise ValueError("world_size and tokens_per_rank must be >= 1")
        if self.ep_size < 1 or self.world_size % self.ep_size:
            raise ValueError(
                f"expert_model_parallel_size ({self.ep_size}) must divide "
                f"dp_size*cp_size ({self.world_size})"
            )
        if not self.allowed_cp_sizes or any(
            not 1 <= cp <= self.world_size for cp in self.allowed_cp_sizes
        ):
            raise ValueError(
                f"allowed_cp_sizes must be within 1..{self.world_size}: {self.allowed_cp_sizes}"
            )
        if self.max_cp_size is not None:
            if type(self.max_cp_size) is not int or self.max_cp_size < 1:
                raise ValueError("max_cp_size must be a positive integer when set")
            allowed = tuple(cp for cp in self.allowed_cp_sizes if cp <= self.max_cp_size)
            if not allowed:
                raise ValueError(
                    f"max_cp_size={self.max_cp_size} leaves no allowed CP sizes "
                    f"in {self.allowed_cp_sizes}"
                )
            object.__setattr__(self, "allowed_cp_sizes", allowed)
        if self.dataset_alignment < 1 or self.dataset_alignment % 2:
            raise ValueError("dataset_alignment must be a positive even number")
        if self.max_group_seqs is not None and self.max_group_seqs < 1:
            raise ValueError("max_group_seqs must be >= 1 when set")

    def stored_length(self, length: int, cp: int) -> int:
        """Physical tokens after dataset alignment and CP re-padding by _pack_sequences."""
        aligned = _ceil_to(int(length), self.dataset_alignment)
        return _ceil_to(aligned, 2 * int(cp))

    def capacity(self, cp: int) -> int:
        """Physical tokens of a CP-``cp`` microbatch pack."""
        return int(cp) * self.tokens_per_rank

    def pack_unit(self, cp: int) -> int:
        """Smallest sequence length that a CP-``cp`` pack stores without padding."""
        return math.lcm(self.dataset_alignment, 2 * int(cp))

    def full_pack(self, cp: int, count: int) -> Optional[Tuple[int, ...]]:
        """``count`` equal sequences filling a CP-``cp`` pack without padding, or None."""
        unit = self.pack_unit(cp)
        length = self.capacity(cp) // int(count) // unit * unit
        return (length,) * int(count) if length else None

    def supports_group(self, start: int, cp: int) -> bool:
        """Whether this contiguous interval is supported by the runtime's groups."""
        return (
            cp in self.allowed_cp_sizes
            and 0 <= start <= self.world_size - cp
            and (not self.align_group_starts or start % cp == 0)
        )

    def validate_lengths(self, lengths: Dict[int, int]) -> None:
        """Reject samples that cannot fit any allowed CP width, including padding."""
        if not lengths or any(
            type(i) is not int or i < 0 or type(n) is not int or n <= 0 for i, n in lengths.items()
        ):
            raise ValueError(
                "lengths must map nonnegative integer IDs to positive integer logical lengths"
            )
        for sample_id, length in lengths.items():
            if not any(
                self.stored_length(length, cp) <= self.capacity(cp) for cp in self.allowed_cp_sizes
            ):
                required = (length + self.tokens_per_rank - 1) // self.tokens_per_rank
                raise RuntimeScheduleError(
                    f"sample {sample_id}: seq_len={length} requires CP >= {required} before "
                    f"padding, but no allowed CP size fits after alignment; "
                    f"max_cp_size={self.max_cp_size}, allowed_cp_sizes={self.allowed_cp_sizes}, "
                    f"max_seqlen_per_dp_cp_rank={self.tokens_per_rank}"
                )


def sample_length_limits(geometry: RuntimePlanGeometry, model) -> Dict[int, Optional[int]]:
    """Longest logical sample ``model`` prices at each allowed CP size; None without a limit.

    This asks the optional ``pack_limits``, which a model may answer by pricing packs: call
    it where a plan is made or an error is being reported, not on every rank at start-up.
    """
    declared = getattr(model, "pack_limits", None)
    return {
        cp: None if declared is None else declared(cp).max_sample_length
        for cp in geometry.allowed_cp_sizes
    }


def longest_priced_sample(geometry: RuntimePlanGeometry, model) -> int:
    """Longest logical sample that some allowed CP size both holds and prices."""
    return max(
        geometry.capacity(cp) if limit is None else min(geometry.capacity(cp), limit)
        for cp, limit in sample_length_limits(geometry, model).items()
    )


def unpriced_sample(geometry: RuntimePlanGeometry, model, lengths: Dict[int, int]):
    """First sample that fits an allowed CP size but that no such size prices, or None.

    ``validate_lengths`` rejects a sample no allowed CP size can hold. A cost model with a
    measured sample-length range may still price none of the sizes that hold it.
    """
    limits = sample_length_limits(geometry, model)
    if all(limit is None for limit in limits.values()):
        return None
    for sample_id, length in lengths.items():
        if not any(
            geometry.stored_length(length, cp) <= geometry.capacity(cp)
            and (limit is None or length <= limit)
            for cp, limit in limits.items()
        ):
            return sample_id
    return None


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

    Raises :class:`RuntimeScheduleError` naming the microbatch, rank and reason before an
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
                    f"cap is {geometry.max_group_seqs} (thd_max_packed_sequences, less one entry "
                    f"when a dummy tail sequence is appended, and at most the cost models' "
                    f"{MAX_PRICED_GROUP_SEQUENCES} sequences)"
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
                    f"microbatch {m}: group at rank {start} (cp={cp}) stores {stored} tokens, "
                    f"capacity {geometry.capacity(cp)}"
                )
            seen.extend(ids)
        # Equal-list runs are contiguous by construction. The multiset check below catches
        # any ID appearing in two non-adjacent groups as a duplicate.
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
        self._sequence_limits = {}
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
            max_group_seqs=self.geometry.max_group_seqs or MAX_PRICED_GROUP_SEQUENCES,
            sequence_limit=self.sequence_limit,
        )

    def sequence_limit(self, cp):
        """Sequences one CP-``cp`` pack may hold in any layout this planner builds.

        The cap of the geometry, lowered to what the cost model declares it can price at
        this CP size (``pack_limits``). It is asked on first use and kept: a model may
        have to price packs to answer, and only a process that plans should pay for that.
        """
        limit = self._sequence_limits.get(cp)
        if limit is None:
            limit = self.geometry.max_group_seqs or MAX_PRICED_GROUP_SEQUENCES
            declared = getattr(self.model, "pack_limits", None)
            if declared is not None:
                model_limit = declared(cp).max_sequences
                if model_limit is not None:
                    limit = min(limit, model_limit)
            self._sequence_limits[cp] = limit
        return limit

    def model_sequence_limits(self):
        """CP sizes whose packs the cost model limits below the cap of the geometry."""
        cap = self.geometry.max_group_seqs or MAX_PRICED_GROUP_SEQUENCES
        return {cp: limit for cp, limit in sorted(self._sequence_limits.items()) if limit < cap}

    def _no_layout(self, lengths, detail):
        """Explain which constraints left this global batch without a layout."""
        geometry = self.geometry
        longest = max(lengths, key=lengths.get)
        return RuntimeScheduleError(
            f"no layout gives each of the {geometry.world_size} DPxCP ranks a sample in every "
            f"microbatch: {len(lengths)} samples (longest: sample {longest}, "
            f"seq_len={lengths[longest]}), allowed_cp_sizes={geometry.allowed_cp_sizes}, "
            f"max_cp_size={geometry.max_cp_size}, at most "
            f"{geometry.max_group_seqs or MAX_PRICED_GROUP_SEQUENCES} sequences per CP group, "
            f"max_seqlen_per_dp_cp_rank={geometry.tokens_per_rank} ({detail})"
        )

    def _start_plan(self, lengths):
        lengths = dict(lengths)
        self.geometry.validate_lengths(lengths)
        self.model.clear()
        self.costs = self._new_costs()
        return lengths

    def _capacity_fallback(self, lengths, widths=None):
        """Split fixed-width packs to fill rank rows when search finds no layout.

        This is a capacity-only candidate, scored by the same E2E model as any
        search result. It is bounded by the sample count, not an exhaustive
        feasibility solver.
        """
        geometry = self.geometry
        allowed = geometry.allowed_cp_sizes if widths is None else widths
        cp = next(
            (
                width
                for width in reversed(allowed)
                if geometry.world_size % width == 0
                and self.sequence_limit(width) >= 1
                and all(
                    geometry.stored_length(length, width) <= geometry.capacity(width)
                    for length in lengths.values()
                )
            ),
            None,
        )
        if cp is None:
            return None
        limit = self.sequence_limit(cp)
        stored = {i: geometry.stored_length(n, cp) for i, n in lengths.items()}
        pending = sorted(lengths, key=lambda i: (-stored[i], -lengths[i], i))
        packs = []
        while pending:
            ids, used = [], 0
            for i in pending:
                if len(ids) >= limit:
                    break
                if used + stored[i] <= geometry.capacity(cp):
                    ids.append(i)
                    used += stored[i]
            selected = set(ids)
            pending = [i for i in pending if i not in selected]
            packs.append(ids)
        per_row = geometry.world_size // cp
        needed = (-len(packs)) % per_row
        if len(packs) + needed > len(lengths):
            return None
        for ids in tuple(packs):
            while needed and len(ids) > 1:
                packs.append([ids.pop()])
                needed -= 1
        return [
            [dict(cp=cp, sample_ids=ids) for ids in packs[start : start + per_row]]
            for start in range(0, len(packs), per_row)
        ]

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
        self.geometry.validate_lengths(logical)
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
