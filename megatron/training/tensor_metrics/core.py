# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit, distributed tensor metric primitives for training observations.

This module deliberately does not discover tensors, process groups, or logging sinks. A caller
describes how each local tensor relates to ranks on named axes and supplies the corresponding
process groups to :class:`TensorMetricExecutor`. Metrics perform local work and explicitly request
the collectives needed to make progress.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import groupby
from typing import TypeAlias

import torch

__all__ = [
    "AllGather",
    "AllReduce",
    "Collective",
    "CollectiveCompletionSet",
    "CollectiveRequest",
    "CollectiveRequestSet",
    "FlatShard",
    "LogicalReductionMetric",
    "MetricGroup",
    "MetricResult",
    "MetricSite",
    "MetricStep",
    "MetricTensor",
    "Owned",
    "PerGroupTensorMetric",
    "Placement",
    "RankRelation",
    "Replica",
    "Shard",
    "TensorMetric",
    "TensorMetricExecutor",
]


@dataclass(frozen=True)
class MetricSite:
    """Location and kind of an observed tensor.

    Args:
        name: Stable, human-readable name for the observation site.
        kind: Source kind, such as ``parameter``, ``wgrad``, ``activation``, or ``dgrad``.
    """

    name: str
    kind: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MetricSite.name must not be empty.")
        if not self.kind:
            raise ValueError("MetricSite.kind must not be empty.")


@dataclass(frozen=True)
class Replica:
    """A value that is identical on every rank of an axis."""


@dataclass(frozen=True)
class Shard:
    """A value partitioned across an axis.

    Args:
        dim: Local tensor dimension partitioned by the axis. ``None`` represents a partition whose
            tensor dimension is not known; dimension-sensitive metrics may reject it.
    """

    dim: int | None


@dataclass(frozen=True)
class FlatShard:
    """A contiguous flat interval of a logical tensor held on one rank.

    The interval uses the logical tensor's contiguous row-major flattening. This placement is
    useful for DistributedOptimizer ranges, which may begin or end in the middle of any logical
    tensor dimension.

    Args:
        logical_shape: Shape of the tensor before sharding on this rank axis. It may still be local
            with respect to other rank axes, such as tensor parallelism.
        start: Inclusive flat offset of the local interval.
        end: Exclusive flat offset of the local interval.
    """

    logical_shape: tuple[int, ...]
    start: int
    end: int

    def __post_init__(self) -> None:
        logical_shape = tuple(self.logical_shape)
        object.__setattr__(self, "logical_shape", logical_shape)
        if any(size < 0 for size in logical_shape):
            raise ValueError("FlatShard.logical_shape dimensions must be non-negative.")
        logical_numel = 1
        for size in logical_shape:
            logical_numel *= size
        if self.start < 0 or self.end < self.start or self.end > logical_numel:
            raise ValueError(
                "FlatShard interval must satisfy 0 <= start <= end <= logical tensor size."
            )


@dataclass(frozen=True)
class Owned:
    """A value contributed by one rank of an axis.

    This commonly applies to parameters owned by the layerwise optimizer.

    Other ranks must participate with an empty or neutral local value before a metric requests a
    collective on this axis.

    Args:
        rank: Rank within the axis that owns the value.
    """

    rank: int

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("Owned.rank must be non-negative.")


Placement: TypeAlias = Replica | Shard | FlatShard | Owned


@dataclass(frozen=True)
class RankRelation:
    """Placement of a tensor relative to one named rank axis.

    Args:
        axis: Abstract axis identifier. The executor binds it to a concrete process group.
        placement: Relationship of the local tensor to other ranks on the axis.
    """

    axis: str
    placement: Placement

    def __post_init__(self) -> None:
        if not self.axis:
            raise ValueError("RankRelation.axis must not be empty.")


@dataclass(frozen=True)
class MetricTensor:
    """A local tensor annotated with observation sites and rank relations.

    Args:
        tensor: Rank-local value.
        sites: Observation sites contributing to the value.
        rank_relations: Placement on each relevant rank axis. Axis identifiers must be unique.
        is_placeholder: Whether this is a neutral rank-symmetric slot for a logical tensor stored
            on another rank, rather than a locally present value. An empty local shard is not a
            placeholder.
    """

    tensor: torch.Tensor
    sites: tuple[MetricSite, ...]
    rank_relations: tuple[RankRelation, ...] = ()
    is_placeholder: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "sites", tuple(self.sites))
        object.__setattr__(self, "rank_relations", tuple(self.rank_relations))
        if not self.sites:
            raise ValueError("MetricTensor.sites must not be empty.")
        axes = [relation.axis for relation in self.rank_relations]
        if len(axes) != len(set(axes)):
            raise ValueError("MetricTensor.rank_relations must contain each axis at most once.")

    def relation(self, axis: str) -> RankRelation:
        """Return the relation for an axis.

        Args:
            axis: Abstract rank axis identifier.

        Returns:
            Relation associated with ``axis``.

        Raises:
            KeyError: If the tensor has no relation for ``axis``.
        """
        for relation in self.rank_relations:
            if relation.axis == axis:
                return relation
        raise KeyError(f"MetricTensor has no rank relation for axis {axis!r}.")

    def with_tensor(
        self, tensor: torch.Tensor, rank_relations: Sequence[RankRelation] | None = None
    ) -> "MetricTensor":
        """Return a copy containing a new local tensor and optionally new relations.

        Args:
            tensor: New rank-local value.
            rank_relations: New relations, or ``None`` to preserve the existing relations.

        Returns:
            Updated metric tensor.
        """
        relations = self.rank_relations if rank_relations is None else tuple(rank_relations)
        return replace(self, tensor=tensor, rank_relations=relations)

    def with_placement(self, axis: str, placement: Placement) -> "MetricTensor":
        """Return a copy with one axis assigned a new placement.

        Args:
            axis: Existing abstract rank axis identifier.
            placement: New placement on that axis.

        Returns:
            Updated metric tensor.

        Raises:
            KeyError: If the tensor has no relation for ``axis``.
        """
        found = False
        relations = []
        for relation in self.rank_relations:
            if relation.axis == axis:
                relations.append(RankRelation(axis, placement))
                found = True
            else:
                relations.append(relation)
        if not found:
            raise KeyError(f"MetricTensor has no rank relation for axis {axis!r}.")
        relations = tuple(relations)
        is_placeholder = self.is_placeholder and any(
            isinstance(relation.placement, Owned) for relation in relations
        )
        return replace(
            self, rank_relations=relations, is_placeholder=is_placeholder
        )


@dataclass(frozen=True)
class MetricGroup:
    """A metric-selected group of prepared tensor contributions.

    Groups produced by one metric need not be disjoint. The same prepared contribution may appear
    in several groups when one preparation should feed several independently labeled results.

    Args:
        items: Prepared tensor contributions belonging to the group.
        label: Optional metric-defined label for the result produced by this group. The executor
            does not use labels to correlate work.
    """

    items: tuple[MetricTensor, ...]
    label: Hashable | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        if not self.items:
            raise ValueError("MetricGroup.items must not be empty.")


@dataclass(frozen=True)
class AllReduce:
    """Request an all-reduce across one explicitly selected rank axis.

    Args:
        op: PyTorch reduction operation to perform.
    """

    op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM


@dataclass(frozen=True)
class AllGather:
    """Request an equal-sized all-gather across one explicitly selected rank axis.

    Args:
        dim: Existing tensor dimension along which rank-local values are concatenated. ``None``
            stacks them along a new leading dimension.
    """

    dim: int | None = None


Collective: TypeAlias = AllReduce | AllGather


@dataclass(frozen=True)
class CollectiveRequest:
    """One collective requested by a metric.

    Args:
        value: Local collective contribution.
        axis: Abstract rank axis on which to communicate.
        collective: Collective operation to execute.
    """

    value: MetricTensor
    axis: str
    collective: Collective


@dataclass(frozen=True)
class CollectiveRequestSet:
    """Independent collectives requested together by one metric group.

    The executor may pack compatible requests from this and other request sets. It completes every
    request in the set before returning the corresponding :class:`CollectiveCompletionSet` to the
    metric.

    Args:
        requests: Independent collectives ready to execute. Must not be empty.
        continuation: Opaque group state returned unchanged with the completions.
    """

    requests: tuple[CollectiveRequest, ...]
    continuation: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "requests", tuple(self.requests))
        if not self.requests:
            raise ValueError("CollectiveRequestSet.requests must not be empty.")


@dataclass(frozen=True)
class CollectiveCompletionSet:
    """Completed request set returned to one metric group.

    Values have the same order as the corresponding requests, even when the executor reorders
    requests internally for communication batching.

    Args:
        values: One collective result per request, in request order. Each value has the targeted
            axis marked as replicated.
        continuation: Opaque group state supplied by the request set.
    """

    values: tuple[MetricTensor, ...]
    continuation: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if not self.values:
            raise ValueError("CollectiveCompletionSet.values must not be empty.")


@dataclass(frozen=True)
class MetricResult:
    """A fully computed metric value.

    Args:
        value: Computed metric value and its observation sites. Finalized results need not carry
            rank relations; those annotations are required while collective work is in progress.
        label: Optional metric-defined label distinguishing this result from other results.
    """

    value: MetricTensor
    label: Hashable | None = None


MetricStep: TypeAlias = CollectiveRequestSet | MetricResult


class TensorMetric(ABC):
    """Complete definition of tensor preparation, grouping, and distributed computation.

    A caller may use :meth:`accepts` to avoid constructing observations that cannot contribute.
    The executor applies the same site predicate before calling :meth:`prepare` while observed
    tensors are available. Callers may accumulate prepared contributions from several observation
    points before passing them to :meth:`groups` and the batch-native :meth:`start`. Selection,
    grouping, and group order must be deterministic across every rank that will participate in the
    resulting collectives.
    """

    name: str = ""
    source_kinds: frozenset[str] = frozenset()

    def accepts(self, site: MetricSite) -> bool:
        """Return whether an observation site may belong to this metric.

        This is a cheap, tensor-independent predicate that callers may invoke before constructing
        a :class:`MetricTensor`. The executor invokes it again before :meth:`prepare`. The default
        accepts every site when :attr:`source_kinds` is empty and otherwise accepts declared source
        kinds. Because callers may invoke this predicate more than once, it must be deterministic
        and free of side effects.

        Args:
            site: Stable name and source kind of a potential observation.

        Returns:
            Whether the site may contribute to the metric.
        """
        return not self.source_kinds or site.kind in self.source_kinds

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Locally prepare a potential batch of selected observed tensors.

        This method must not communicate. Metrics may override it to compact short-lived tensors
        into local contributions that retain the sites and rank relations needed for later
        grouping and communication, or to perform additional tensor-aware filtering. The executor
        supplies detached inputs whose sites have passed :meth:`accepts` and detaches returned
        contributions. The default preserves every input.

        Args:
            values: Selected observed tensors available together.

        Returns:
            Prepared contributions in input order, optionally with additional observations omitted.
        """
        return list(values)

    def groups(self, values: Sequence[MetricTensor]) -> list[MetricGroup]:
        """Form independent groups from accumulated prepared contributions.

        The default aggregates every contribution into one group labeled ``"global"``, keeping
        result cardinality bounded. Metrics performing per-tensor or other structured aggregation
        may override this method. Groups may overlap: reusing one prepared contribution in several
        groups produces several results without invoking :meth:`prepare` again. Each group is
        nevertheless executed independently, so overlapping groups may repeat communication.
        Selection and preparation have already happened.

        Args:
            values: Prepared contributions accumulated by the caller.

        Returns:
            One global group, or no groups when no contributions were prepared.
        """
        return [MetricGroup(tuple(values), label="global")] if values else []

    @abstractmethod
    def start(self, groups: Sequence[MetricGroup]) -> list[MetricStep]:
        """Start a batch of independent metric groups.

        Args:
            groups: Groups of prepared contributions available together.

        Returns:
            Request sets and completed results. Groups may be filtered by returning no
            corresponding step.
        """

    @abstractmethod
    def resume(self, completed: Sequence[CollectiveCompletionSet]) -> list[MetricStep]:
        """Resume a batch of independent metric groups.

        Args:
            completed: Completed request sets available together.

        Returns:
            Further request sets and completed results.
        """


class PerGroupTensorMetric(TensorMetric):
    """Convenience base adapting per-group implementations to the batch-native interface."""

    def start(self, groups: Sequence[MetricGroup]) -> list[MetricStep]:
        """Start groups independently using :meth:`start_group`.

        Args:
            groups: Groups of prepared contributions available together.

        Returns:
            Request sets and completed results. Groups may be filtered by returning no
            corresponding step from :meth:`start_group`.
        """
        return [step for group in groups if (step := self.start_group(group)) is not None]

    @abstractmethod
    def start_group(self, group: MetricGroup) -> MetricStep | None:
        """Start metric computation for one selected tensor group.

        Args:
            group: Prepared contributions belonging to one group.

        Returns:
            A collective request set, completed result, or ``None`` to filter the group.
        """

    def resume(self, completed: Sequence[CollectiveCompletionSet]) -> list[MetricStep]:
        """Resume groups independently using :meth:`resume_group`.

        Args:
            completed: Completed request sets available together.

        Returns:
            Further request sets and completed results.
        """
        return [
            step
            for completion_set in completed
            if (step := self.resume_group(completion_set)) is not None
        ]

    @abstractmethod
    def resume_group(self, completed: CollectiveCompletionSet) -> MetricStep | None:
        """Resume one grouped computation after its requested collectives.

        Args:
            completed: Collective outputs and the metric's opaque continuation.

        Returns:
            Another collective request set, a completed result, or ``None`` to filter the group.
        """


@dataclass(frozen=True)
class _LogicalReductionBranch:
    index: int
    remaining_axes: tuple[str, ...]


@dataclass(frozen=True)
class _LogicalReductionContinuation:
    group: MetricGroup
    branches: tuple[_LogicalReductionBranch, ...]
    contributions: tuple[torch.Tensor | None, ...]


_LOCAL_DIMENSION_REDUCTION_OPS: dict[
    torch.distributed.ReduceOp, Callable[[torch.Tensor], torch.Tensor]
] = {
    torch.distributed.ReduceOp.SUM: lambda tensor: tensor.sum(dim=0, dtype=tensor.dtype),
    torch.distributed.ReduceOp.PRODUCT: lambda tensor: tensor.prod(dim=0, dtype=tensor.dtype),
    torch.distributed.ReduceOp.MIN: lambda tensor: tensor.amin(dim=0),
    torch.distributed.ReduceOp.MAX: lambda tensor: tensor.amax(dim=0),
}

_LOCAL_BINARY_REDUCTION_OPS: dict[
    torch.distributed.ReduceOp, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
] = {
    torch.distributed.ReduceOp.BAND: torch.bitwise_and,
    torch.distributed.ReduceOp.BOR: torch.bitwise_or,
    torch.distributed.ReduceOp.BXOR: torch.bitwise_xor,
}


class LogicalReductionMetric(PerGroupTensorMetric):
    """Base class for simple reductions over complete logical tensors.

    A subclass converts each rank-local tensor into a fixed-shaped, mergeable contribution. For
    grouped metrics, it also defines how those contributions combine. This class all-reduces each
    contribution across every ``Shard`` axis and across every ``Owned`` axis, then calls
    :meth:`finalize`. Ranks without an ``Owned`` value must contribute the identity for
    :attr:`reduction_op`.
    Existing ``Replica`` axes require no communication.

    :meth:`prepare` creates the contributions. Each prepared contribution is reduced along all of
    its ``Shard`` axes, then all of its ``Owned`` axes, before the contributions in a group are
    combined. Every branch whose next collective is ready is returned in one request set, allowing
    the executor to batch compatible requests across heterogeneous layouts.

    The default :meth:`combine_contributions` applies :attr:`reduction_op` locally for associative
    PyTorch reductions with direct tensor equivalents. A metric may override it for different
    cross-tensor semantics. More complex layouts should override :meth:`start_group`. Subclasses
    may override :meth:`contribution_batch` to use fused or multi-tensor kernels.
    """

    reduction_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM

    @abstractmethod
    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create a mergeable contribution from one rank-local tensor.

        Args:
            tensor: Detached rank-local tensor.

        Returns:
            Fixed-shaped contribution that can be combined with :attr:`reduction_op`.
        """

    def contribution_batch(self, tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Create mergeable contributions for a potential batch of rank-local tensors.

        Args:
            tensors: Detached rank-local tensors available together.

        Returns:
            One mergeable contribution per input tensor, in input order.
        """
        return [self.contribution(tensor) for tensor in tensors]

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Convert selected tensors to a potential batch of mergeable contributions.

        Args:
            values: Selected observed tensors available together.

        Returns:
            One prepared contribution per tensor, in input order.

        Raises:
            ValueError: If :meth:`contribution_batch` returns the wrong number of contributions.
        """
        if not values:
            return []
        contributions = self.contribution_batch(tuple(value.tensor for value in values))
        if len(contributions) != len(values):
            raise ValueError(
                "LogicalReductionMetric.contribution_batch must return one contribution per tensor."
            )
        return [
            value.with_tensor(contribution) for value, contribution in zip(values, contributions)
        ]

    def combine_contributions(
        self, group: MetricGroup, contributions: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Combine per-tensor contributions within one selected group.

        Args:
            group: Selected tensor group corresponding to ``contributions``.
            contributions: One mergeable contribution per tensor in ``group``.

        Returns:
            One combined contribution.

        Raises:
            ValueError: If the contributions do not correspond to the group or a multi-tensor
                group uses a reduction without default local semantics.
        """
        if len(contributions) != len(group.items):
            raise ValueError(
                "Logical reduction contributions must correspond to the grouped tensors."
            )
        if len(contributions) == 1:
            return contributions[0]
        dimension_op = _LOCAL_DIMENSION_REDUCTION_OPS.get(self.reduction_op)
        if dimension_op is not None:
            return dimension_op(torch.stack(tuple(contributions), dim=0))
        local_op = _LOCAL_BINARY_REDUCTION_OPS.get(self.reduction_op)
        if local_op is None:
            raise ValueError(
                f"{self.reduction_op!r} has no default local multi-tensor implementation; "
                "override combine_contributions."
            )
        result = contributions[0]
        for contribution in contributions[1:]:
            result = local_op(result, contribution)
        return result

    def finalize(self, contribution: torch.Tensor) -> torch.Tensor:
        """Convert a fully reduced contribution into the reported metric value.

        Args:
            contribution: Combined contributions resolved across their described rank axes.

        Returns:
            Final metric value.
        """
        return contribution

    def start_group(self, group: MetricGroup) -> MetricStep:
        """Request the first required reduction for every prepared contribution in a group.

        Args:
            group: Prepared contributions for one result.

        Returns:
            First collective request set or completed result.
        """
        return self._start_from_contributions(group, tuple(value.tensor for value in group.items))

    def resume_group(self, completed: CollectiveCompletionSet) -> MetricStep:
        """Advance every completed logical-reduction branch.

        Args:
            completed: Completed branch reductions and their shared continuation.

        Returns:
            Next collective request set or completed result.

        Raises:
            ValueError: If the completion did not originate from this abstraction.
        """
        continuation = completed.continuation
        if not isinstance(continuation, _LogicalReductionContinuation):
            raise ValueError("Collective completion has an invalid logical reduction continuation.")
        if len(completed.values) != len(continuation.branches):
            raise ValueError(
                f"{type(self).__name__} requires one completion per logical reduction branch."
            )
        requests = []
        branches = []
        contributions = list(continuation.contributions)
        for value, branch in zip(completed.values, continuation.branches):
            if branch.remaining_axes:
                requests.append(
                    CollectiveRequest(
                        value=value,
                        axis=branch.remaining_axes[0],
                        collective=AllReduce(self.reduction_op),
                    )
                )
                branches.append(_LogicalReductionBranch(branch.index, branch.remaining_axes[1:]))
            else:
                contributions[branch.index] = value.tensor
        return self._request_or_finish(
            group=continuation.group,
            requests=requests,
            branches=branches,
            contributions=contributions,
        )

    def _start_from_contributions(
        self, group: MetricGroup, contributions: Sequence[torch.Tensor]
    ) -> MetricStep:
        if len(contributions) != len(group.items):
            raise ValueError(
                "Logical reduction contributions must correspond to the grouped tensors."
            )
        requests = []
        branches = []
        resolved_contributions: list[torch.Tensor | None] = [None] * len(contributions)
        for index, (grouped_value, contribution) in enumerate(zip(group.items, contributions)):
            value = grouped_value.with_tensor(contribution)
            shard_axes = tuple(
                relation.axis
                for relation in value.rank_relations
                if isinstance(relation.placement, (Shard, FlatShard))
            )
            owned_axes = tuple(
                relation.axis
                for relation in value.rank_relations
                if isinstance(relation.placement, Owned)
            )
            reduction_axes = shard_axes + owned_axes
            if reduction_axes:
                requests.append(
                    CollectiveRequest(
                        value=value, axis=reduction_axes[0], collective=AllReduce(self.reduction_op)
                    )
                )
                branches.append(_LogicalReductionBranch(index, reduction_axes[1:]))
            else:
                resolved_contributions[index] = contribution
        return self._request_or_finish(
            group=group, requests=requests, branches=branches, contributions=resolved_contributions
        )

    def _request_or_finish(
        self,
        group: MetricGroup,
        requests: Sequence[CollectiveRequest],
        branches: Sequence[_LogicalReductionBranch],
        contributions: Sequence[torch.Tensor | None],
    ) -> MetricStep:
        if requests:
            return CollectiveRequestSet(
                requests=tuple(requests),
                continuation=_LogicalReductionContinuation(
                    group=group, branches=tuple(branches), contributions=tuple(contributions)
                ),
            )
        resolved_contributions = []
        for contribution in contributions:
            if contribution is None:
                raise ValueError("A logical reduction branch did not produce a final contribution.")
            resolved_contributions.append(contribution)
        contribution = self.combine_contributions(group, resolved_contributions)
        sites = tuple(site for value in group.items for site in value.sites)
        return MetricResult(MetricTensor(self.finalize(contribution), sites=sites), group.label)


class TensorMetricExecutor:
    """Execute tensor metrics using caller-supplied process groups.

    Compatible requests are packed into a leading batch dimension before communication. No group
    is discovered from Megatron global state.

    Args:
        axis_process_groups: Mapping from abstract metric axes to concrete process groups. Mapping
            an axis to ``None`` explicitly selects the default world process group.
    """

    def __init__(
        self, axis_process_groups: Mapping[str, torch.distributed.ProcessGroup | None]
    ) -> None:
        self._axis_process_groups = dict(axis_process_groups)

    @torch.no_grad()
    def run(self, metric: TensorMetric, values: Sequence[MetricTensor]) -> list[MetricResult]:
        """Prepare, group, and run a metric to completion for observed tensors.

        Args:
            metric: Metric implementation to run.
            values: Observed tensors.

        Returns:
            Completed metric results in completion order.

        Raises:
            ValueError: If the metric produces an invalid collective request.
            RuntimeError: If communication is requested before ``torch.distributed`` is initialized.
        """
        prepared = self.prepare(metric, values)
        return self.complete(metric, self.start(metric, prepared))

    @torch.no_grad()
    def prepare(self, metric: TensorMetric, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Select and locally prepare observed tensors without communication.

        This method may be called repeatedly at observation points where short-lived tensors are
        available. Its outputs may be accumulated and passed together to :meth:`start` later. A
        value is selected only when every contributing site passes :meth:`TensorMetric.accepts`.
        The executor detaches selected inputs before calling the metric and detaches its returned
        contributions, so metric authors do not need to manage autograd connectivity.

        Args:
            metric: Metric implementation that owns site selection and local preparation.
            values: Observed tensors available together.

        Returns:
            Prepared metric contributions in deterministic input order.
        """
        selected_values = [
            value for value in values if all(metric.accepts(site) for site in value.sites)
        ]
        detached_values = [
            value.with_tensor(value.tensor.detach()) if value.tensor.requires_grad else value
            for value in selected_values
        ]
        prepared = metric.prepare(detached_values)
        return [
            value.with_tensor(value.tensor.detach()) if value.tensor.requires_grad else value
            for value in prepared
        ]

    @torch.no_grad()
    def start(self, metric: TensorMetric, values: Sequence[MetricTensor]) -> list[MetricStep]:
        """Group prepared contributions and start their distributed metric computations.

        Args:
            metric: Metric implementation to start.
            values: Prepared contributions returned by :meth:`prepare`.

        Returns:
            Results and explicit collective request sets, which may be passed to :meth:`complete`
            later.

        """
        return metric.start(metric.groups(values))

    @torch.no_grad()
    def complete(
        self, metric: TensorMetric, initial_steps: Sequence[MetricStep]
    ) -> list[MetricResult]:
        """Complete metric steps that may have been started earlier.

        Together with :meth:`prepare`, this separation lets short-lived tensors be compacted when
        they are observed while deferring their collectives until a later commit point.

        Args:
            metric: Metric implementation that produced ``initial_steps``.
            initial_steps: Request sets and results previously returned by :meth:`TensorMetric.start`
                or :meth:`TensorMetric.resume`.

        Returns:
            Completed metric results in completion order.

        Raises:
            ValueError: If the metric produces an invalid collective request.
            RuntimeError: If communication is requested before ``torch.distributed`` is initialized.
        """
        steps = list(initial_steps)
        results: list[MetricResult] = []

        while steps:
            request_sets: list[CollectiveRequestSet] = []
            for step in steps:
                if isinstance(step, MetricResult):
                    results.append(step)
                else:
                    request_sets.append(step)

            if not request_sets:
                break
            requests = [request for request_set in request_sets for request in request_set.requests]
            completed_values = self._execute_requests(requests)
            completion_sets = []
            offset = 0
            for request_set in request_sets:
                end = offset + len(request_set.requests)
                completion_sets.append(
                    CollectiveCompletionSet(
                        values=tuple(completed_values[offset:end]),
                        continuation=request_set.continuation,
                    )
                )
                offset = end
            steps = metric.resume(completion_sets)

        return results

    def _execute_requests(self, requests: Sequence[CollectiveRequest]) -> list[MetricTensor]:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "A tensor metric requested communication before distributed initialization."
            )

        indexed_requests = list(enumerate(requests))
        indexed_requests.sort(key=lambda entry: self._compatibility_key(entry[1]))
        completed_values: list[MetricTensor | None] = [None] * len(requests)
        for _, compatible_group in groupby(
            indexed_requests, key=lambda entry: self._compatibility_key(entry[1])
        ):
            entries = list(compatible_group)
            packed_values = self._execute_compatible([request for _, request in entries])
            for (index, _), value in zip(entries, packed_values):
                completed_values[index] = value
        return [value for value in completed_values if value is not None]

    def _compatibility_key(self, request: CollectiveRequest) -> tuple:
        tensor = request.value.tensor
        return (
            request.axis,
            type(request.collective).__name__,
            repr(request.collective),
            str(tensor.device),
            str(tensor.dtype),
            tuple(tensor.shape),
            str(tensor.layout),
        )

    def _execute_compatible(self, requests: Sequence[CollectiveRequest]) -> list[MetricTensor]:
        first = requests[0]
        self._validate_request(first)
        for request in requests[1:]:
            self._validate_request(request)

        group = self._axis_process_groups[first.axis]
        packed = torch.stack([request.value.tensor.detach() for request in requests])
        world_size = torch.distributed.get_world_size(group=group)
        if isinstance(first.collective, AllReduce):
            if world_size > 1:
                torch.distributed.all_reduce(packed, op=first.collective.op, group=group)
            outputs = packed.unbind()
        else:
            if world_size > 1:
                gathered = [torch.empty_like(packed) for _ in range(world_size)]
                torch.distributed.all_gather(gathered, packed, group=group)
            else:
                gathered = [packed]
            if first.collective.dim is None:
                outputs = tuple(
                    torch.stack([rank_values[index] for rank_values in gathered], dim=0)
                    for index in range(len(requests))
                )
            else:
                dim = _normalize_dim(first.collective.dim, first.value.tensor.ndim)
                outputs = tuple(
                    torch.cat([rank_values[index] for rank_values in gathered], dim=dim)
                    for index in range(len(requests))
                )

        return [
            request.value.with_tensor(output).with_placement(request.axis, Replica())
            for request, output in zip(requests, outputs)
        ]

    def _validate_request(self, request: CollectiveRequest) -> None:
        if request.axis not in self._axis_process_groups:
            raise ValueError(f"No process group was supplied for metric axis {request.axis!r}.")
        request.value.relation(request.axis)


def _normalize_dim(dim: int, ndim: int) -> int:
    normalized = dim if dim >= 0 else dim + ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"Tensor dimension {dim} is invalid for a {ndim}-dimensional tensor.")
    return normalized
