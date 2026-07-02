# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Planning API for sequence packing and backend placement.

This module intentionally contains API types only. It does not move data,
materialize THD tensors, create process groups, or call forward/backward
schedules. The goal is to provide a shared contract that can be implemented by
online schedulers, offline planners, Dynamic CP placement, static CP placement,
HybridEP placement, and future backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable


class PlacementKind(str, Enum):
    """Logical backend selected for a planned pack assignment."""

    DATA_PARALLEL = "data_parallel"
    STATIC_CP = "static_cp"
    DYNAMIC_CP = "dynamic_cp"
    HYBRID_EP = "hybrid_ep"
    CUSTOM = "custom"


def _as_tuple(values: Sequence[int], field_name: str) -> Tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _validate_positive(value: int, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")


@dataclass(frozen=True)
class SequenceDescriptor:
    """Metadata for one source sequence or already-unpacked sub-sample."""

    sample_id: int
    num_tokens: int
    padded_num_tokens: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_positive(self.num_tokens, "num_tokens")
        if self.padded_num_tokens is not None and self.padded_num_tokens < self.num_tokens:
            raise ValueError(
                "padded_num_tokens must be greater than or equal to num_tokens, "
                f"got {self.padded_num_tokens} < {self.num_tokens}"
            )

    @property
    def materialized_tokens(self) -> int:
        """Token count the pack materializer should reserve for this sequence."""

        return self.padded_num_tokens if self.padded_num_tokens is not None else self.num_tokens


@dataclass(frozen=True)
class PackingConstraints:
    """Capacity and alignment constraints used by sequence-packing schedulers."""

    max_tokens_per_pack: int
    pad_to_multiple: int = 1
    sequence_parallel_size: int = 1
    backend_alignment: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_positive(self.max_tokens_per_pack, "max_tokens_per_pack")
        _validate_positive(self.pad_to_multiple, "pad_to_multiple")
        _validate_positive(self.sequence_parallel_size, "sequence_parallel_size")
        if self.backend_alignment is not None:
            _validate_positive(self.backend_alignment, "backend_alignment")


@dataclass(frozen=True)
class PackDescriptor:
    """Output of a sequence-packing scheduler.

    ``sequence_lengths`` represent real token lengths. ``padded_sequence_lengths``
    represent the lengths that should be used to materialize THD tensors when
    per-sequence padding is required.
    """

    pack_id: int
    sample_ids: Sequence[int]
    sequence_lengths: Sequence[int]
    padded_sequence_lengths: Optional[Sequence[int]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_ids = _as_tuple(self.sample_ids, "sample_ids")
        sequence_lengths = _as_tuple(self.sequence_lengths, "sequence_lengths")
        if len(sample_ids) != len(sequence_lengths):
            raise ValueError(
                "sample_ids and sequence_lengths must have the same length, "
                f"got {len(sample_ids)} != {len(sequence_lengths)}"
            )
        for length in sequence_lengths:
            _validate_positive(length, "sequence_lengths entries")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "sequence_lengths", sequence_lengths)

        if self.padded_sequence_lengths is not None:
            padded_lengths = _as_tuple(self.padded_sequence_lengths, "padded_sequence_lengths")
            if len(padded_lengths) != len(sequence_lengths):
                raise ValueError(
                    "padded_sequence_lengths and sequence_lengths must have the same length, "
                    f"got {len(padded_lengths)} != {len(sequence_lengths)}"
                )
            for padded_length, real_length in zip(padded_lengths, sequence_lengths):
                if padded_length < real_length:
                    raise ValueError(
                        "padded_sequence_lengths entries must be >= sequence_lengths entries, "
                        f"got {padded_length} < {real_length}"
                    )
            object.__setattr__(self, "padded_sequence_lengths", padded_lengths)

    @property
    def total_tokens(self) -> int:
        """Total real tokens in this pack."""

        return sum(self.sequence_lengths)

    @property
    def materialized_tokens(self) -> int:
        """Total tokens that will exist after per-sequence padding."""

        lengths = self.padded_sequence_lengths or self.sequence_lengths
        return sum(lengths)

    @property
    def cu_seqlens(self) -> Tuple[int, ...]:
        """Cumulative real sequence lengths for THD metadata."""

        return self._cumulative_lengths(self.sequence_lengths)

    @property
    def cu_seqlens_padded(self) -> Tuple[int, ...]:
        """Cumulative padded sequence lengths for THD materialization."""

        lengths = self.padded_sequence_lengths or self.sequence_lengths
        return self._cumulative_lengths(lengths)

    @staticmethod
    def _cumulative_lengths(lengths: Sequence[int]) -> Tuple[int, ...]:
        cursor = 0
        cumulative = [cursor]
        for length in lengths:
            cursor += int(length)
            cumulative.append(cursor)
        return tuple(cumulative)


@dataclass(frozen=True)
class RankGroup:
    """Logical rank group used by placement plans.

    This type deliberately stores logical rank IDs, not concrete
    ``torch.distributed.ProcessGroup`` objects, so plans can be serialized,
    tested on CPU, and produced by offline planners.
    """

    group_id: str
    ranks: Sequence[int]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ranks = _as_tuple(self.ranks, "ranks")
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"ranks must be unique, got {ranks}")
        if any(rank < 0 for rank in ranks):
            raise ValueError(f"ranks must be non-negative, got {ranks}")
        object.__setattr__(self, "ranks", ranks)


@dataclass(frozen=True)
class PlacementResources:
    """Logical resources available to a placement scheduler."""

    dp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    rank_groups: Mapping[str, RankGroup] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_positive(self.dp_size, "dp_size")
        _validate_positive(self.cp_size, "cp_size")
        _validate_positive(self.ep_size, "ep_size")
        _validate_positive(self.tp_size, "tp_size")
        _validate_positive(self.pp_size, "pp_size")


@dataclass(frozen=True)
class PackAssignment:
    """Placement of one pack onto logical executor ranks."""

    pack_id: int
    placement_kind: PlacementKind
    executor_ranks: Sequence[int]
    rank_group_id: Optional[str] = None
    local_cp_size: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ranks = _as_tuple(self.executor_ranks, "executor_ranks")
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"executor_ranks must be unique, got {ranks}")
        if any(rank < 0 for rank in ranks):
            raise ValueError(f"executor_ranks must be non-negative, got {ranks}")
        object.__setattr__(self, "executor_ranks", ranks)
        if self.local_cp_size is not None:
            _validate_positive(self.local_cp_size, "local_cp_size")


@dataclass(frozen=True)
class ExecutionGroup:
    """Barrier-safe group of pack assignments that can execute together."""

    group_id: int
    assignments: Sequence[PackAssignment]
    requires_barrier_after: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assignments = tuple(self.assignments)
        if not assignments:
            raise ValueError("assignments must not be empty")
        object.__setattr__(self, "assignments", assignments)


@dataclass(frozen=True)
class ExecutionPlan:
    """Top-level plan consumed by a future shared execution engine."""

    groups: Sequence[ExecutionGroup]
    num_microbatches: Optional[int] = None
    total_tokens: Optional[int] = None
    sequence_square_sum: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        if not groups:
            raise ValueError("groups must not be empty")
        object.__setattr__(self, "groups", groups)
        if self.num_microbatches is not None:
            _validate_positive(self.num_microbatches, "num_microbatches")
        if self.total_tokens is not None:
            _validate_positive(self.total_tokens, "total_tokens")
        if self.sequence_square_sum is not None:
            _validate_positive(self.sequence_square_sum, "sequence_square_sum")

    @property
    def assignment_count(self) -> int:
        """Number of pack assignments in all execution groups."""

        return sum(len(group.assignments) for group in self.groups)

    @property
    def pack_ids(self) -> Tuple[int, ...]:
        """Pack IDs in execution order."""

        return tuple(
            assignment.pack_id for group in self.groups for assignment in group.assignments
        )


@runtime_checkable
class SequencePackingScheduler(Protocol):
    """Generic sequence-packing planner.

    Implementations can be online or offline. They should decide which source
    sequences belong to each pack, but should not decide backend placement.
    """

    def build_packs(
        self, sequences: Sequence[SequenceDescriptor], constraints: PackingConstraints
    ) -> Sequence[PackDescriptor]:
        """Return pack descriptors for the provided source sequences."""


@runtime_checkable
class PlacementScheduler(Protocol):
    """Backend-specific placement planner.

    Implementations decide where already-built packs execute. For example,
    Dynamic CP maps packs to CP1/CP2/CP4 groups, while HybridEP may map packs to
    expert-parallel resources.
    """

    def build_plan(
        self, packs: Sequence[PackDescriptor], resources: PlacementResources
    ) -> ExecutionPlan:
        """Return an execution plan for the provided packs and resources."""


@runtime_checkable
class PlanExecutionEngine(Protocol):
    """Future shared engine that consumes an execution plan.

    The engine is expected to own data reroute, THD materialization, TP/PP
    metadata broadcast, barriers, and eventual train/eval integration. This PR
    defines the contract only.
    """

    def execute_plan(self, plan: ExecutionPlan, samples: Mapping[int, Any]) -> Any:
        """Execute or materialize ``plan`` using source samples keyed by sample ID."""
