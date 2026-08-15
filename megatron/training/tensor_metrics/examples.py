# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Readable examples of tensor metrics with different communication patterns.

These examples are intentionally independent of one another.  There is a little duplication so
that a reader can understand any one metric without first learning a hierarchy of example helper
classes.  Production metrics may factor out policy or kernels once there is a demonstrated need.

The examples are not registered with the training loop.  They show how to select observation
sites, compact tensors during ``prepare()``, form result groups, and explicitly stage distributed
collectives using the primitives in :mod:`megatron.training.tensor_metrics`.
"""

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

import torch

from .core import (
    AllReduce,
    CollectiveCompletionSet,
    CollectiveRequest,
    CollectiveRequestSet,
    FlatShard,
    LogicalReductionMetric,
    MetricGroup,
    MetricResult,
    MetricSite,
    MetricStep,
    MetricTensor,
    Owned,
    PerGroupTensorMetric,
    RankRelation,
    Replica,
    Shard,
)

__all__ = [
    "FP8UnderflowFractionExample",
    "GlobalParameterAmaxExample",
    "LayerParameterL2NormExample",
    "MaxReplicaDriftExample",
    "MeanRowL2NormExample",
    "MultiGranularityParameterL2NormExample",
    "ParameterL2NormExample",
    "SampledMaxReplicaDriftExample",
    "TransformerEngineBatchedParameterL2NormExample",
]


class ParameterL2NormExample(LogicalReductionMetric):
    """Compute one L2 norm over all observed parameters.

    This is the smallest useful metric example.  ``source_kinds`` cheaply excludes observations
    other than parameters.  ``prepare()`` is inherited from ``LogicalReductionMetric``: it calls
    ``contribution()`` while each parameter is available and keeps only one sum-of-squares scalar.
    The default ``groups()`` puts all prepared scalars in one group labeled ``"global"``.

    ``LogicalReductionMetric`` sums each scalar over every ``Shard``, ``FlatShard``, and ``Owned``
    relation described by the observer, then sums the resolved parameter scalars locally.  Only
    after those linear reductions are complete does ``finalize()`` take the square root.
    """

    name = "example-parameter-l2"
    source_kinds = frozenset({"parameter"})
    reduction_op = torch.distributed.ReduceOp.SUM

    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compact one local parameter shard to its sum of squared magnitudes."""
        if tensor.dtype in (torch.float16, torch.bfloat16):
            accumulation_dtype = torch.float32
        elif tensor.dtype.is_floating_point or tensor.dtype.is_complex:
            accumulation_dtype = tensor.dtype
        else:
            accumulation_dtype = torch.float32
        return tensor.to(dtype=accumulation_dtype).abs().square().sum()

    def finalize(self, contribution: torch.Tensor) -> torch.Tensor:
        """Convert the complete sum of squares to an L2 norm."""
        return contribution.sqrt()


class GlobalParameterAmaxExample(LogicalReductionMetric):
    """Report the largest absolute value among all observed parameters.

    This is the MAX counterpart to :class:`ParameterL2NormExample`.  Each parameter is compacted
    to one local absolute maximum, and ``LogicalReductionMetric`` applies MAX over every shard or
    ownership axis and across the default global group.  Because absolute values are nonnegative,
    zero is the correct identity for an empty ``FlatShard`` or remotely ``Owned`` parameter.

    The reduction is exact and communicates only one scalar per parameter and rank axis.  It is a
    useful pattern for any metric whose contribution has a simple non-SUM merge operation.
    """

    name = "example-global-parameter-amax"
    source_kinds = frozenset({"parameter"})
    reduction_op = torch.distributed.ReduceOp.MAX

    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compact one local parameter shard to its largest absolute value."""
        magnitude = tensor.abs()
        return magnitude.amax() if magnitude.numel() else magnitude.new_zeros(())


class TransformerEngineBatchedParameterL2NormExample(LogicalReductionMetric):
    """Compute global parameter L2 using TransformerEngine's batched L2 kernel.

    The result and distributed semantics match :class:`ParameterL2NormExample`; only local
    preparation changes.  ``LogicalReductionMetric.prepare()`` passes all tensors available at an
    observation point to ``contribution_batch()``. This implementation groups compatible tensors
    by device and dtype, then calls TransformerEngine's ``multi_tensor_l2norm`` once per bucket
    with ``per_tensor=True``.

    Per-tensor output is essential here.  Different parameters may have different ``Shard``,
    ``FlatShard``, or ``Owned`` relations, so the executor still needs one contribution carrying each
    parameter's metadata.  TransformerEngine returns L2 norms; squaring them produces additive
    sum-of-squares contributions for the ordinary logical SUM reduction. Empty ownership slots bypass
    the kernel and contribute a float32 zero.

    The TransformerEngine import is deliberately local to ``contribution_batch()``. Importing this
    examples module therefore does not load an optional compiled extension unless this particular
    metric runs.  Nonempty inputs must be contiguous CUDA tensors supported by the installed
    TransformerEngine build.
    """

    name = "example-te-batched-parameter-l2"
    source_kinds = frozenset({"parameter"})
    reduction_op = torch.distributed.ReduceOp.SUM

    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Use the same batch path when called for one tensor."""
        return self.contribution_batch((tensor,))[0]

    def contribution_batch(self, tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Compute one sum-of-squares contribution per tensor using fused compatible batches."""
        try:
            from transformer_engine.pytorch.optimizers import (
                multi_tensor_applier,
                multi_tensor_l2norm,
            )
        except ImportError as error:
            raise RuntimeError(
                "TransformerEngineBatchedParameterL2NormExample requires TransformerEngine."
            ) from error

        contributions: list[torch.Tensor | None] = [None] * len(tensors)
        buckets: dict[tuple[torch.device, torch.dtype], list[tuple[int, torch.Tensor]]] = {}
        for index, tensor in enumerate(tensors):
            if tensor.numel() == 0:
                contributions[index] = torch.zeros((), dtype=torch.float32, device=tensor.device)
            else:
                if tensor.device.type != "cuda" or not tensor.is_contiguous():
                    raise ValueError(
                        "TransformerEngine batched L2 requires contiguous CUDA tensors."
                    )
                buckets.setdefault((tensor.device, tensor.dtype), []).append((index, tensor))

        for (device, _), entries in buckets.items():
            bucket_tensors = [tensor for _, tensor in entries]
            overflow_buffer = torch.zeros(1, dtype=torch.int32, device=device)
            _, per_tensor_norms = multi_tensor_applier(
                multi_tensor_l2norm, overflow_buffer, [bucket_tensors], True
            )
            if per_tensor_norms is None or per_tensor_norms.numel() != len(entries):
                raise RuntimeError(
                    "TransformerEngine multi_tensor_l2norm returned an invalid per-tensor result."
                )
            per_tensor_squares = per_tensor_norms.square().reshape(-1)
            for (index, _), square in zip(entries, per_tensor_squares):
                contributions[index] = square

        resolved_contributions = []
        for contribution in contributions:
            if contribution is None:
                raise RuntimeError("A batched parameter L2 input did not produce a contribution.")
            resolved_contributions.append(contribution)
        return resolved_contributions

    def finalize(self, contribution: torch.Tensor) -> torch.Tensor:
        """Convert the complete sum of squares to a global L2 norm."""
        return contribution.sqrt()


class LayerParameterL2NormExample(LogicalReductionMetric):
    """Compute one parameter L2 norm for every numbered model layer.

    This repeats the small amount of numerical code from :class:`ParameterL2NormExample` so the
    grouping example stands on its own.  The interesting addition is the naming policy:
    ``decoder.layers.3.mlp.linear_fc1.weight`` belongs to ``decoder.layers.3``.  ``accepts()`` can
    apply that policy before the observer constructs a ``MetricTensor``; ``prepare()`` checks the
    same policy for tensors carrying more than one site; and ``groups()`` assigns result labels.

    Group order follows first observation order.  That order, and the selected groups themselves,
    must be deterministic on ranks that will participate in the same collectives.
    """

    name = "example-layer-parameter-l2"
    source_kinds = frozenset({"parameter"})
    reduction_op = torch.distributed.ReduceOp.SUM
    _layer_pattern = re.compile(r"^(?P<layer>(?:[^.]+\.)*layers\.[0-9]+)(?:\.|$)")

    @classmethod
    def _site_layer(cls, site: MetricSite) -> str | None:
        match = cls._layer_pattern.match(site.name)
        return None if match is None else match.group("layer")

    @classmethod
    def _tensor_layer(cls, value: MetricTensor) -> str | None:
        labels = tuple(cls._site_layer(site) for site in value.sites)
        if labels[0] is None or any(label != labels[0] for label in labels[1:]):
            return None
        return labels[0]

    def accepts(self, site: MetricSite) -> bool:
        """Select parameter sites whose names identify a numbered layer."""
        return super().accepts(site) and self._site_layer(site) is not None

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Drop a combined tensor if its contributing sites do not identify one layer."""
        selected = [value for value in values if self._tensor_layer(value) is not None]
        return super().prepare(selected)

    def groups(self, values: Sequence[MetricTensor]) -> list[MetricGroup]:
        """Group prepared sum-of-squares contributions by layer name."""
        grouped: dict[str, list[MetricTensor]] = {}
        for value in values:
            label = self._tensor_layer(value)
            if label is None:
                raise ValueError("A prepared layer metric tensor must identify one layer.")
            grouped.setdefault(label, []).append(value)
        return [MetricGroup(tuple(items), label=label) for label, items in grouped.items()]

    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compact one local parameter shard to its sum of squared magnitudes."""
        if tensor.dtype in (torch.float16, torch.bfloat16):
            accumulation_dtype = torch.float32
        elif tensor.dtype.is_floating_point or tensor.dtype.is_complex:
            accumulation_dtype = tensor.dtype
        else:
            accumulation_dtype = torch.float32
        return tensor.to(dtype=accumulation_dtype).abs().square().sum()

    def finalize(self, contribution: torch.Tensor) -> torch.Tensor:
        """Convert one layer's complete sum of squares to its L2 norm."""
        return contribution.sqrt()


class MultiGranularityParameterL2NormExample(LogicalReductionMetric):
    """Report tensor, parameter-family, layer, and global parameter L2 norms together.

    Every parameter is compacted exactly once by ``prepare()``.  ``groups()`` then places the same
    prepared scalar in several overlapping ``MetricGroup`` objects.  This is the intended pattern
    when several results share local preparation but differ only in grouping.

    A family replaces a numbered ``layers.<index>`` segment with ``layers.*``.  For example, all
    ``decoder.layers.N.mlp.linear_fc1.weight`` parameters share one family.  Parameters outside a
    numbered layer still receive tensor, family, and global results, but no layer result.

    Overlapping groups save local work, not necessarily communication: groups execute
    independently, so a sharded scalar may be reduced once for each result containing it.  A
    future executor optimization could coalesce such identical requests without changing this
    metric's grouping policy.
    """

    name = "example-multi-granularity-parameter-l2"
    source_kinds = frozenset({"parameter"})
    reduction_op = torch.distributed.ReduceOp.SUM
    _layer_pattern = re.compile(r"^(?P<layer>(?:[^.]+\.)*layers\.[0-9]+)(?:\.|$)")

    @classmethod
    def _site_layer(cls, site: MetricSite) -> str | None:
        match = cls._layer_pattern.match(site.name)
        return None if match is None else match.group("layer")

    @classmethod
    def _tensor_layer(cls, value: MetricTensor) -> str | None:
        labels = tuple(cls._site_layer(site) for site in value.sites)
        if labels[0] is None or any(label != labels[0] for label in labels[1:]):
            return None
        return labels[0]

    @classmethod
    def _family_name(cls, name: str) -> str:
        match = cls._layer_pattern.match(name)
        if match is None:
            return name
        layer_collection = match.group("layer").rsplit(".", maxsplit=1)[0]
        return f"{layer_collection}.*{name[match.end('layer') :]}"

    @classmethod
    def _tensor_family(cls, value: MetricTensor) -> str:
        labels = tuple(cls._family_name(site.name) for site in value.sites)
        if any(label != labels[0] for label in labels[1:]):
            raise ValueError("A prepared family metric tensor must identify one family.")
        return labels[0]

    def groups(self, values: Sequence[MetricTensor]) -> list[MetricGroup]:
        """Reuse prepared contributions in four sets of independently labeled groups."""
        tensor_groups = [
            MetricGroup((value,), label=f"tensor/{'+'.join(site.name for site in value.sites)}")
            for value in values
        ]

        families: dict[str, list[MetricTensor]] = {}
        layers: dict[str, list[MetricTensor]] = {}
        for value in values:
            families.setdefault(self._tensor_family(value), []).append(value)
            layer = self._tensor_layer(value)
            if layer is not None:
                layers.setdefault(layer, []).append(value)

        family_groups = [
            MetricGroup(tuple(items), label=f"family/{label}") for label, items in families.items()
        ]
        layer_groups = [
            MetricGroup(tuple(items), label=f"layer/{label}") for label, items in layers.items()
        ]
        global_groups = [MetricGroup(tuple(values), label="global")] if values else []
        return tensor_groups + family_groups + layer_groups + global_groups

    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compact one local parameter shard to its sum of squared magnitudes."""
        if tensor.dtype in (torch.float16, torch.bfloat16):
            accumulation_dtype = torch.float32
        elif tensor.dtype.is_floating_point or tensor.dtype.is_complex:
            accumulation_dtype = tensor.dtype
        else:
            accumulation_dtype = torch.float32
        return tensor.to(dtype=accumulation_dtype).abs().square().sum()

    def finalize(self, contribution: torch.Tensor) -> torch.Tensor:
        """Convert each group's complete sum of squares to an L2 norm."""
        return contribution.sqrt()


class FP8UnderflowFractionExample(LogicalReductionMetric):
    """Report the fraction of wgrad elements below a supplied FP8 nonzero threshold.

    The caller supplies ``underflow_threshold`` in the original wgrad's units.  This is important:
    real FP8 training scales values before conversion, so there is no universally correct raw
    tensor threshold for "FP8 underflow."  The caller should derive the threshold from the actual
    FP8 format, quantization scale, and rounding rule being evaluated.

    A nonzero element with ``abs(value) < underflow_threshold`` counts as an underflow candidate.
    Exact zeros do not count as candidates, but all elements (including zeros) remain in the
    denominator. ``contribution()`` turns an arbitrarily large wgrad into the fixed contribution
    ``[candidate_count, element_count]``.  Both entries are additive, so ordinary logical SUM
    reduction handles shards, distributed-optimizer flat shards, ownership, and global grouping.
    """

    name = "example-fp8-underflow-fraction"
    source_kinds = frozenset({"wgrad"})
    reduction_op = torch.distributed.ReduceOp.SUM

    def __init__(self, underflow_threshold: float) -> None:
        if underflow_threshold <= 0:
            raise ValueError("FP8 underflow threshold must be positive.")
        self.underflow_threshold = underflow_threshold

    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Count local underflow candidates and total local elements."""
        magnitude = tensor.abs()
        candidates = ((magnitude != 0) & (magnitude < self.underflow_threshold)).sum(
            dtype=torch.float64
        )
        element_count = torch.tensor(tensor.numel(), dtype=torch.float64, device=tensor.device)
        return torch.stack((candidates, element_count))

    def finalize(self, contribution: torch.Tensor) -> torch.Tensor:
        """Divide the complete candidate count by the complete element count."""
        fraction = contribution[0] / contribution[1]
        return torch.where(contribution[1] > 0, fraction, torch.full_like(fraction, torch.nan))


class _RowL2Stage(Enum):
    RESOLVE_COLUMNS = auto()
    RESOLVE_ROW_POPULATION = auto()


@dataclass(frozen=True)
class _RowL2Branch:
    index: int
    stage: _RowL2Stage
    remaining_axes: tuple[str, ...]


@dataclass(frozen=True)
class _RowL2Continuation:
    group: MetricGroup
    branches: tuple[_RowL2Branch, ...]
    states: tuple[torch.Tensor | None, ...]


class MeanRowL2NormExample(PerGroupTensorMetric):
    """Report the population-weighted mean row L2 norm of selected wgrads.

    This is the first example that cannot use ``LogicalReductionMetric``.  A row split across a
    tensor-parallel axis must have its squared column contributions summed *before* the square
    root.  Rows split across another axis must have their final ``[norm_sum, row_count]`` state
    summed *after* the square root.  The continuation below makes those two stages explicit and
    exposes all tensor branches that are ready at a stage in one request set.

    A ``FlatShard`` can begin in the middle of a row.  ``_flat_shard_row_sums()`` scatters the
    shard's squared elements into a fixed vector with one slot per logical row; an all-reduce then
    reconstructs every row's sum of squares without gathering the original wgrad.  This is more
    state than a scalar metric, but much less than the full parameter for typical matrices.

    Unlike the preceding examples, ``prepare()`` intentionally retains the detached wgrad because
    the per-row state depends on its logical layout.  A production implementation could override
    batch execution with a multi-tensor kernel, but the communication stages would remain the
    same.
    """

    name = "example-mean-row-l2"
    source_kinds = frozenset({"wgrad"})

    def start_group(self, group: MetricGroup) -> MetricStep:
        """Create local row sums and request every initially ready column reduction."""
        requests: list[CollectiveRequest] = []
        branches: list[_RowL2Branch] = []
        states: list[torch.Tensor | None] = [None] * len(group.items)
        for index, value in enumerate(group.items):
            # A rank-symmetric placeholder has no local shape from which to construct row slots. It
            # can skip directly to the fixed population state because the Owned reduction will
            # combine this neutral contribution with the owner's state. An empty local FlatShard is
            # not a placeholder; it must construct zero row slots before reducing with its peers.
            if value.is_placeholder:
                if value.tensor.dtype in (torch.float16, torch.bfloat16):
                    accumulation_dtype = torch.float32
                elif value.tensor.dtype.is_floating_point:
                    accumulation_dtype = value.tensor.dtype
                else:
                    accumulation_dtype = torch.float32
                sum_and_count = torch.zeros(2, dtype=accumulation_dtype, device=value.tensor.device)
                population_axes = self._reduction_axes(value)
                self._advance(
                    index,
                    _RowL2Stage.RESOLVE_ROW_POPULATION,
                    value.with_tensor(sum_and_count),
                    population_axes,
                    requests,
                    branches,
                    states,
                )
                continue
            local_row_sums, column_axes = self._local_row_sums(value)
            self._advance(
                index,
                _RowL2Stage.RESOLVE_COLUMNS,
                local_row_sums,
                column_axes,
                requests,
                branches,
                states,
            )
        return self._request_or_finish(group, requests, branches, states)

    def resume_group(self, completed: CollectiveCompletionSet) -> MetricStep:
        """Advance all branches whose previous column or population reduction completed."""
        continuation = completed.continuation
        if not isinstance(continuation, _RowL2Continuation):
            raise ValueError("Collective completion has an invalid row L2 continuation.")
        if len(completed.values) != len(continuation.branches):
            raise ValueError("MeanRowL2NormExample requires one completion per active branch.")

        requests: list[CollectiveRequest] = []
        branches: list[_RowL2Branch] = []
        states = list(continuation.states)
        for value, branch in zip(completed.values, continuation.branches):
            self._advance(
                branch.index, branch.stage, value, branch.remaining_axes, requests, branches, states
            )
        return self._request_or_finish(continuation.group, requests, branches, states)

    @staticmethod
    def _local_row_sums(value: MetricTensor) -> tuple[MetricTensor, tuple[str, ...]]:
        flat_shards = tuple(
            relation.placement
            for relation in value.rank_relations
            if isinstance(relation.placement, FlatShard)
        )
        if len(flat_shards) > 1:
            raise ValueError("MeanRowL2NormExample supports at most one FlatShard.")

        logical_shape = flat_shards[0].logical_shape if flat_shards else tuple(value.tensor.shape)
        if not logical_shape or logical_shape[-1] == 0:
            raise ValueError("Mean row L2 norm requires a nonempty final tensor dimension.")
        accumulation_dtype = (
            torch.float32
            if value.tensor.dtype in (torch.float16, torch.bfloat16)
            else value.tensor.dtype
        )
        squared = value.tensor.to(dtype=accumulation_dtype).abs().square()
        if flat_shards:
            row_sums = MeanRowL2NormExample._flat_shard_row_sums(squared, flat_shards[0])
        else:
            row_sums = squared.sum(dim=-1)

        output_relations = []
        column_axes = []
        ndim = len(logical_shape)
        for relation in value.rank_relations:
            placement = relation.placement
            if isinstance(placement, FlatShard):
                output_relations.append(relation)
                column_axes.append(relation.axis)
            elif isinstance(placement, Shard):
                if placement.dim is None:
                    raise ValueError("Mean row L2 norm requires the dimension of every Shard.")
                shard_dim = placement.dim if placement.dim >= 0 else placement.dim + ndim
                if shard_dim < 0 or shard_dim >= ndim:
                    raise ValueError(
                        f"Tensor dimension {placement.dim} is invalid for a {ndim}-D tensor."
                    )
                if shard_dim == ndim - 1:
                    output_relations.append(relation)
                    column_axes.append(relation.axis)
                else:
                    output_relations.append(RankRelation(relation.axis, Shard(shard_dim)))
            else:
                output_relations.append(relation)
        return value.with_tensor(row_sums, output_relations), tuple(column_axes)

    @staticmethod
    def _flat_shard_row_sums(squared: torch.Tensor, placement: FlatShard) -> torch.Tensor:
        """Scatter one contiguous flat shard into fixed per-logical-row sum slots."""
        local_values = squared.reshape(-1)
        if local_values.numel() != placement.end - placement.start:
            raise ValueError("FlatShard interval length must equal the local tensor element count.")

        column_count = placement.logical_shape[-1]
        row_count = 1
        for size in placement.logical_shape[:-1]:
            row_count *= size
        row_sums = torch.zeros(row_count, dtype=squared.dtype, device=squared.device)
        if placement.start == placement.end:
            return row_sums.reshape(placement.logical_shape[:-1])

        start_column = placement.start % column_count
        first_size = min(column_count - start_column, local_values.numel())
        start_row = placement.start // column_count
        row_sums[start_row] = local_values[:first_size].sum()

        cursor = first_size
        full_row_count = (local_values.numel() - cursor) // column_count
        if full_row_count:
            full_rows = local_values[cursor : cursor + full_row_count * column_count].reshape(
                full_row_count, column_count
            )
            row_sums[start_row + 1 : start_row + 1 + full_row_count] = full_rows.sum(dim=1)
            cursor += full_row_count * column_count

        if cursor < local_values.numel():
            end_row = placement.end // column_count
            row_sums[end_row] = local_values[cursor:].sum()
        return row_sums.reshape(placement.logical_shape[:-1])

    def _advance(
        self,
        index: int,
        stage: _RowL2Stage,
        value: MetricTensor,
        remaining_axes: tuple[str, ...],
        requests: list[CollectiveRequest],
        branches: list[_RowL2Branch],
        states: list[torch.Tensor | None],
    ) -> None:
        if remaining_axes:
            requests.append(CollectiveRequest(value, remaining_axes[0], AllReduce()))
            branches.append(_RowL2Branch(index, stage, remaining_axes[1:]))
            return

        if stage is _RowL2Stage.RESOLVE_COLUMNS:
            row_norms = value.tensor.sqrt()
            sum_and_count = torch.stack(
                (row_norms.sum(), row_norms.new_tensor(row_norms.numel(), dtype=row_norms.dtype))
            )
            population_axes = self._reduction_axes(value)
            self._advance(
                index,
                _RowL2Stage.RESOLVE_ROW_POPULATION,
                value.with_tensor(sum_and_count),
                population_axes,
                requests,
                branches,
                states,
            )
            return

        if stage is _RowL2Stage.RESOLVE_ROW_POPULATION:
            states[index] = value.tensor
            return
        raise ValueError(f"Unexpected row L2 stage: {stage}.")

    @staticmethod
    def _reduction_axes(value: MetricTensor) -> tuple[str, ...]:
        """Keep rank-symmetric shard communication ahead of owner communication."""
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
        return shard_axes + owned_axes

    @staticmethod
    def _request_or_finish(
        group: MetricGroup,
        requests: Sequence[CollectiveRequest],
        branches: Sequence[_RowL2Branch],
        states: Sequence[torch.Tensor | None],
    ) -> MetricStep:
        if requests:
            return CollectiveRequestSet(
                tuple(requests), _RowL2Continuation(group, tuple(branches), tuple(states))
            )

        resolved = []
        for state in states:
            if state is None:
                raise ValueError("A row L2 branch did not produce its final state.")
            resolved.append(state)
        total = resolved[0] if len(resolved) == 1 else torch.stack(tuple(resolved)).sum(dim=0)
        sites = tuple(site for value in group.items for site in value.sites)
        return MetricResult(MetricTensor(total[0] / total[1], sites), group.label)


@dataclass(frozen=True)
class _ReplicaExtremaContinuation:
    group: MetricGroup
    remaining_axes: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _ReplicaDriftBranch:
    index: int
    remaining_axes: tuple[str, ...]


@dataclass(frozen=True)
class _ReplicaDriftContinuation:
    group: MetricGroup
    branches: tuple[_ReplicaDriftBranch, ...]
    states: tuple[torch.Tensor | None, ...]


class MaxReplicaDriftExample(PerGroupTensorMetric):
    """Report absolute and scale-aware parameter drift across an expected-replica axis.

    ``replica_axis`` names the relation to check.  Each selected tensor must describe that axis as
    ``Replica``: the placement records the intended layout, while this diagnostic verifies whether
    the values are actually identical.  Exact maximum pairwise drift is ``max(value) - min(value)``
    at each element, so the first stage performs full-tensor MAX and MIN all-reduces on the replica
    axis.  It reports a two-element tensor containing ``(max_absolute_drift, max_relative_drift)``.
    Relative drift divides each element's absolute drift by the larger replica magnitude, clamped
    to ``relative_scale_floor``.  This symmetric normalization makes 100 versus 101 small, makes a
    meaningful sign reversal large, and prevents insignificant differences near zero from
    dominating.  The metric MAX-reduces both components over any remaining ``Shard``,
    ``FlatShard``, or ``Owned`` axes.

    This exact diagnostic is intentionally expensive: it communicates each parameter twice and
    cannot compact it in ``prepare()``.  Sampling elements, checking a rotating subset of tensors,
    or comparing checksums are natural cheaper extensions, but they answer weaker questions.

    Args:
        replica_axis: Named rank axis whose intended replicas should be compared.
        relative_scale_floor: Positive absolute floor for the relative-drift denominator.
    """

    name = "example-max-replica-drift"
    source_kinds = frozenset({"parameter"})
    result_components = ("max_absolute_drift", "max_relative_drift")

    def __init__(self, replica_axis: str, relative_scale_floor: float = 1e-8) -> None:
        if not replica_axis:
            raise ValueError("Replica drift axis must not be empty.")
        if not math.isfinite(relative_scale_floor) or relative_scale_floor <= 0:
            raise ValueError("Replica drift relative scale floor must be positive and finite.")
        self.replica_axis = replica_axis
        self.relative_scale_floor = relative_scale_floor

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Validate that every selected tensor is intended to be replicated on the checked axis."""
        for value in values:
            try:
                placement = value.relation(self.replica_axis).placement
            except KeyError as error:
                raise ValueError(
                    f"Replica drift requires a {self.replica_axis!r} relation on every tensor."
                ) from error
            if not isinstance(placement, Replica):
                raise ValueError(
                    f"Replica drift axis {self.replica_axis!r} must have Replica placement."
                )
        return list(values)

    def start_group(self, group: MetricGroup) -> MetricStep:
        """Request full-tensor maxima and minima for every tensor in the group."""
        requests = []
        remaining_axes = []
        for value in group.items:
            requests.extend(
                (
                    CollectiveRequest(
                        value, self.replica_axis, AllReduce(torch.distributed.ReduceOp.MAX)
                    ),
                    CollectiveRequest(
                        value, self.replica_axis, AllReduce(torch.distributed.ReduceOp.MIN)
                    ),
                )
            )
            remaining_axes.append(
                tuple(
                    relation.axis
                    for relation in value.rank_relations
                    if relation.axis != self.replica_axis
                    and isinstance(relation.placement, (Shard, FlatShard, Owned))
                )
            )
        return CollectiveRequestSet(
            tuple(requests), _ReplicaExtremaContinuation(group, tuple(remaining_axes))
        )

    def resume_group(self, completed: CollectiveCompletionSet) -> MetricStep:
        """Compact extrema to drift scalars, or advance their remaining MAX reductions."""
        continuation = completed.continuation
        if isinstance(continuation, _ReplicaExtremaContinuation):
            return self._resume_extrema(completed.values, continuation)
        if isinstance(continuation, _ReplicaDriftContinuation):
            if len(completed.values) != len(continuation.branches):
                raise ValueError("Replica drift requires one completion per active branch.")
            requests: list[CollectiveRequest] = []
            branches: list[_ReplicaDriftBranch] = []
            states = list(continuation.states)
            for value, branch in zip(completed.values, continuation.branches):
                self._advance_drift(
                    branch.index, value, branch.remaining_axes, requests, branches, states
                )
            return self._request_or_finish(continuation.group, requests, branches, states)
        raise ValueError("Collective completion has an invalid replica drift continuation.")

    def _resume_extrema(
        self, values: Sequence[MetricTensor], continuation: _ReplicaExtremaContinuation
    ) -> MetricStep:
        if len(values) != 2 * len(continuation.group.items):
            raise ValueError("Replica drift requires MAX and MIN completions for every tensor.")
        requests: list[CollectiveRequest] = []
        branches: list[_ReplicaDriftBranch] = []
        states: list[torch.Tensor | None] = [None] * len(continuation.group.items)
        for index, axes in enumerate(continuation.remaining_axes):
            maximum, minimum = values[2 * index : 2 * index + 2]
            if maximum.tensor.numel():
                accumulation_dtype = (
                    torch.float32
                    if maximum.tensor.dtype in (torch.float16, torch.bfloat16)
                    else maximum.tensor.dtype
                )
                maximum_tensor = maximum.tensor.to(dtype=accumulation_dtype)
                minimum_tensor = minimum.tensor.to(dtype=accumulation_dtype)
                absolute_drift = (maximum_tensor - minimum_tensor).abs()
                relative_scale = torch.maximum(maximum_tensor.abs(), minimum_tensor.abs()).clamp_min(
                    self.relative_scale_floor
                )
                drift = torch.stack(
                    (absolute_drift.amax(), (absolute_drift / relative_scale).amax())
                )
            else:
                drift = maximum.tensor.new_zeros(2, dtype=torch.float32)
            self._advance_drift(index, maximum.with_tensor(drift), axes, requests, branches, states)
        return self._request_or_finish(continuation.group, requests, branches, states)

    @staticmethod
    def _advance_drift(
        index: int,
        value: MetricTensor,
        remaining_axes: tuple[str, ...],
        requests: list[CollectiveRequest],
        branches: list[_ReplicaDriftBranch],
        states: list[torch.Tensor | None],
    ) -> None:
        if remaining_axes:
            requests.append(
                CollectiveRequest(
                    value, remaining_axes[0], AllReduce(torch.distributed.ReduceOp.MAX)
                )
            )
            branches.append(_ReplicaDriftBranch(index, remaining_axes[1:]))
        else:
            states[index] = value.tensor

    @staticmethod
    def _request_or_finish(
        group: MetricGroup,
        requests: Sequence[CollectiveRequest],
        branches: Sequence[_ReplicaDriftBranch],
        states: Sequence[torch.Tensor | None],
    ) -> MetricStep:
        if requests:
            return CollectiveRequestSet(
                tuple(requests), _ReplicaDriftContinuation(group, tuple(branches), tuple(states))
            )
        resolved = []
        for state in states:
            if state is None:
                raise ValueError("A replica drift branch did not produce its final state.")
            resolved.append(state)
        maximum = (
            resolved[0]
            if len(resolved) == 1
            else torch.stack(tuple(resolved)).amax(dim=0)
        )
        sites = tuple(site for value in group.items for site in value.sites)
        return MetricResult(MetricTensor(maximum, sites), group.label)


class SampledMaxReplicaDriftExample(MaxReplicaDriftExample):
    """Estimate absolute and relative replica drift from a deterministic parameter sample.

    The exact drift algorithm in :class:`MaxReplicaDriftExample` is unchanged.  This subclass only
    replaces each parameter in ``prepare()`` with a compact sample, so the full-tensor MAX and MIN
    collectives operate on approximately ``1 / sample_factor`` of its elements.  The default
    ``sample_factor=100`` therefore communicates roughly one percent of every nonempty parameter.
    At least one element is retained.

    Each result component is the maximum over the sampled elements and is consequently a lower
    bound on its exact counterpart.  The metric has a distinct name so logs do not confuse those
    semantics.

    Sampling must select the same elements on every replica.  A stable digest of ``sample_seed``
    and the observation sites chooses the start and coprime stride of an affine permutation of the
    flattened tensor.  Taking a prefix of that permutation produces unique, deterministic indices
    with ``O(sample_size)`` temporary storage; unlike ``randperm``, it does not allocate one index
    per original element.  The fixed seed deliberately observes the same elements on every run.

    Args:
        replica_axis: Named rank axis whose intended replicas should be compared.
        sample_factor: Approximate reduction in elements communicated per parameter. Must be
            positive. A value of one samples the entire parameter.
        sample_seed: Stable integer selecting a different deterministic sample.
        relative_scale_floor: Positive absolute floor for the relative-drift denominator.
    """

    name = "example-sampled-max-replica-drift"

    def __init__(
        self,
        replica_axis: str,
        sample_factor: int = 100,
        sample_seed: int = 0,
        relative_scale_floor: float = 1e-8,
    ) -> None:
        super().__init__(replica_axis, relative_scale_floor)
        if sample_factor <= 0:
            raise ValueError("Replica drift sample factor must be positive.")
        self.sample_factor = sample_factor
        self.sample_seed = sample_seed

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Validate replica layouts and replace each nonempty parameter with its stable sample."""
        validated = super().prepare(values)
        sampled = []
        for value in validated:
            tensor = value.tensor
            if tensor.numel() == 0:
                sampled.append(value)
                continue
            if not tensor.is_contiguous():
                raise ValueError("Sampled replica drift requires contiguous tensors.")

            sample_count = max(1, (tensor.numel() + self.sample_factor - 1) // self.sample_factor)
            permutation_seed = self._permutation_seed(value)
            start = permutation_seed % tensor.numel()
            stride = ((permutation_seed >> 64) % tensor.numel()) or 1
            while math.gcd(stride, tensor.numel()) != 1:
                stride = stride + 1 if stride + 1 < tensor.numel() else 1

            indices = torch.arange(sample_count, dtype=torch.int64, device=tensor.device)
            indices.mul_(stride).add_(start).remainder_(tensor.numel())
            sampled.append(value.with_tensor(tensor.view(-1).index_select(0, indices)))
        return sampled

    def _permutation_seed(self, value: MetricTensor) -> int:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(self.sample_seed).encode("utf-8"))
        for site in value.sites:
            digest.update(b"\0")
            digest.update(site.kind.encode("utf-8"))
            digest.update(b"\0")
            digest.update(site.name.encode("utf-8"))
        return int.from_bytes(digest.digest(), byteorder="little")
