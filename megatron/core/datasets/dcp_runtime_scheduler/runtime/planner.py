# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU geometry, capacity and validated MCore rank-row contracts."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, replace
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


def longest_priced_sample(geometry: RuntimePlanGeometry, model) -> int:
    """Longest logical sample that an allowed CP size holds and that ``model`` prices."""
    capacity = geometry.capacity(max(geometry.allowed_cp_sizes))
    return min(capacity, getattr(model, "max_sample_length", None) or capacity)


def unpriced_sample(geometry: RuntimePlanGeometry, model, lengths: Dict[int, int]):
    """First sample beyond the measured sample-length range of ``model``, or None.

    ``validate_lengths`` has rejected a sample that no allowed CP size can hold.
    """
    limit = longest_priced_sample(geometry, model)
    return next((i for i, n in lengths.items() if n > limit), None)


@dataclass
class RuntimePlan:
    """One planned global batch (rows are what MCore consumes; the rest is telemetry)."""

    rows: List[List[List[int]]]
    groups: List[List[Tuple[int, int, List[int]]]]
    predicted_ms: float
    microbatches: int


def encode_rows(rounds) -> List[List[List[int]]]:
    """``sched`` rounds (``[[{'cp', 'sample_ids'}]]``) -> MCore rows (``[mb][rank] -> ids``)."""
    return [
        [list(group["sample_ids"]) for group in row for _ in range(int(group["cp"]))]
        for row in rounds
    ]


def decode_groups(row: Sequence[Sequence[int]]) -> List[Tuple[int, int, List[int]]]:
    """One MCore row -> ``[(start, cp, ids)]`` (maximal runs of equal id lists, rank order)."""
    groups, start = [], 0
    for ids, run in itertools.groupby(map(list, row)):
        groups.append((start, len(list(run)), ids))
        start += groups[-1][1]
    return groups


def validate_rows(
    rows: Sequence[Sequence[Sequence[int]]], lengths: Dict[int, int], geometry: RuntimePlanGeometry
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
            if cp not in allowed:
                raise RuntimeScheduleError(
                    f"microbatch {m}: group at rank {start} has cp={cp}, allowed {sorted(allowed)}"
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
        counts = Counter(seen)
        missing = sorted(i for i in lengths if i not in counts)
        extra = sorted(i for i, c in counts.items() if c > 1 or i not in lengths)
        raise RuntimeScheduleError(
            f"schedule lost or duplicated samples: missing {missing}, duplicated/unknown {extra}"
        )
    return decoded


class RuntimePlanner(ABC):
    """Common model injection, validation, pricing and rank-row contract for both solvers."""

    # Every plan has a microbatch count that is a multiple of this. A planner for more than
    # one pipeline stage sets it to the count multiple of the pipeline schedule.
    count_multiple = 1

    def __init__(self, geometry, cost_model) -> None:
        geometry = replace(geometry, allowed_cp_sizes=tuple(sorted(set(geometry.allowed_cp_sizes))))
        self.geometry = geometry
        self.model = cost_model
        cost_model.check_domain(geometry, None)
        self.timeline = build_timeline(cost_model, geometry)
        # Sequences one CP group may hold in any layout this planner builds.
        self.sequence_cap = geometry.max_group_seqs or MAX_PRICED_GROUP_SEQUENCES
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
            self.model.program_costs,
            placement_period=(
                None
                if getattr(self.model, "placement_invariant", False)
                else self.geometry.world_size
            ),
            sequence_limit=self.sequence_cap,
        )

    def _no_layout(self, lengths, detail):
        """Explain which constraints left this global batch without a layout."""
        geometry = self.geometry
        longest = max(lengths, key=lengths.get)
        if self.count_multiple > 1:
            detail += (
                f"; the count of microbatches is a multiple of {self.count_multiple}, the "
                "pipeline-parallel size"
            )
            # A necessary condition: every microbatch holds at least DPxCP / largest CP size
            # groups, and every group a sample. It is named only when the batch fails it.
            groups = geometry.world_size // max(geometry.allowed_cp_sizes)
            needed = self.count_multiple * groups
            if len(lengths) < needed:
                detail += f", which takes at least {needed} samples; this batch has {len(lengths)}"
        return RuntimeScheduleError(
            f"no layout gives each of the {geometry.world_size} DPxCP ranks a sample in every "
            f"microbatch: {len(lengths)} samples (longest: sample {longest}, "
            f"seq_len={lengths[longest]}), allowed_cp_sizes={geometry.allowed_cp_sizes}, "
            f"max_cp_size={geometry.max_cp_size}, at most "
            f"{self.sequence_cap} sequences per CP group, "
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
        feasibility solver. Its number of rank rows is a multiple of ``count_multiple``.
        """
        geometry = self.geometry
        allowed = geometry.allowed_cp_sizes if widths is None else widths
        cp = next(
            (
                width
                for width in reversed(allowed)
                if geometry.world_size % width == 0
                and all(
                    geometry.stored_length(length, width) <= geometry.capacity(width)
                    for length in lengths.values()
                )
            ),
            None,
        )
        if cp is None:
            return None
        stored = {i: geometry.stored_length(n, cp) for i, n in lengths.items()}
        pending = sorted(lengths, key=lambda i: (-stored[i], -lengths[i], i))
        packs = []
        while pending:
            ids, used = [], 0
            for i in pending:
                if len(ids) >= self.sequence_cap:
                    break
                if used + stored[i] <= geometry.capacity(cp):
                    ids.append(i)
                    used += stored[i]
            selected = set(ids)
            pending = [i for i in pending if i not in selected]
            packs.append(ids)
        per_row = geometry.world_size // cp
        # Whole rank rows, and a number of them that is a multiple of ``count_multiple``.
        needed = (-len(packs)) % (per_row * self.count_multiple)
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

    def _validated(self, rounds, lengths):
        """``rounds`` as rank rows that satisfy the runtime contract, with their groups."""
        rows = encode_rows(rounds)
        return rows, validate_rows(rows, lengths, self.geometry)

    def _make_plan(self, rounds, lengths, predicted):
        rows, groups = self._validated(rounds, lengths)
        return RuntimePlan(rows=rows, groups=groups, predicted_ms=predicted, microbatches=len(rows))

    def _costed(self, rounds, lengths):
        out = []
        for row in rounds:
            start, groups = 0, []
            for group in row:
                cp = group["cp"]
                values = tuple(lengths[i] for i in group["sample_ids"])
                value = self.costs.pack(values, cp, start)
                if value is None:
                    raise RuntimeScheduleError(f"cost model rejected CP{cp} pack {values}")
                groups.append((start, cp, *value))
                start += cp
            out.append(groups)
        return out

    def plan_padded(self, padded_lengths, *, logical_lengths=None, incumbent_rows=None):
        """Plan ``{sample id: dataset-padded length}``, or such pairs; pure CPU.

        The cost input is ``logical_lengths`` when given, else the padded upper bound; no LM
        shift is inferred. The caller checks the padded lengths, that the logical ones round
        up to them, and any incumbent rows, as the CPU worker does (``plan`` checks its own
        input again: the cost lengths and the rows, not the padding).
        """
        logical = dict(padded_lengths if logical_lengths is None else logical_lengths)
        incumbent = incumbent_rows and [
            [dict(cp=cp, sample_ids=ids) for _, cp, ids in decode_groups(row)]
            for row in incumbent_rows
        ]
        plan = self.plan(logical, incumbent_rounds=incumbent)
        if plan.microbatches % self.count_multiple:
            # The pipeline schedule cannot run such a count: no plan source may return one.
            raise RuntimeScheduleError(
                f"the runtime schedule has {plan.microbatches} microbatches, which is not a "
                f"multiple of {self.count_multiple}, the pipeline-parallel size"
            )
        self.last_statistics["cost_length_source"] = (
            "after_lm_logical" if logical_lengths is not None else "dataset_padded_upper_bound"
        )
        return plan
