# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Reusable tensor metric implementations.

The distributed metric protocol lives in :mod:`megatron.training.tensor_metrics`.
"""

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import torch

from .core import (
    AllGather,
    AllReduce,
    CollectiveRequest,
    CollectiveStage,
    FlatShard,
    LogicalReductionMetric,
    MetricResult,
    MetricSite,
    MetricStep,
    MetricTensor,
    Owned,
    RankRelation,
    Shard,
    TensorMetric,
)

__all__ = [
    "GlobalL2NormMetric",
    "L2NormMetric",
    "LayerL2NormMetric",
    "LayerMaxMetric",
    "LayerNormalizedEntropyMetric",
    "LayerSampledMedianMetric",
    "MeanColumnL2NormMetric",
    "MeanRowL2NormMetric",
]


class L2NormMetric(LogicalReductionMetric):
    """Compute one L2 norm over all selected logical input tensors.

    Local reduction uses TransformerEngine's fused multi-tensor L2 kernel where the observed
    tensors allow it, and an equivalent per-tensor reduction everywhere else.
    """

    name = "l2"
    reduction_op = torch.distributed.ReduceOp.SUM

    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute one tensor's sum-of-squares contribution.

        Args:
            tensor: Detached rank-local tensor.

        Returns:
            Scalar sum of squared magnitudes in an accumulation-safe dtype.
        """
        accumulation_dtype = _accumulation_dtype(tensor.dtype)
        if not tensor.is_floating_point():
            return tensor.to(dtype=accumulation_dtype).abs().square().sum()
        return torch.linalg.vector_norm(tensor, 2, dtype=accumulation_dtype).square()

    def contribution_batch(self, tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Compute one additive sum-of-squares contribution per input tensor.

        Tensors the fused multi-tensor L2 kernel accepts are bucketed by device and dtype and
        reduced with one kernel call per bucket, requesting per-tensor output so each contribution
        keeps its own placement metadata. Everything the kernel cannot consume -- empty shards,
        host tensors, non-contiguous views, tensor subclasses, and other dtypes -- falls back to
        :meth:`contribution`, which produces the same value and dtype.

        Args:
            tensors: Detached rank-local tensors available together.

        Returns:
            One sum-of-squares contribution per input tensor, in input order.

        Raises:
            RuntimeError: If the fused kernel returns an unusable per-tensor result.
        """
        fused_impl = _fused_l2_norm_impl()
        contributions: list[torch.Tensor | None] = [None] * len(tensors)
        buckets: dict[tuple[torch.device, torch.dtype], list[tuple[int, torch.Tensor]]] = {}
        for index, tensor in enumerate(tensors):
            if fused_impl is not None and _accepts_fused_l2_norm(tensor):
                buckets.setdefault((tensor.device, tensor.dtype), []).append((index, tensor))
            else:
                contributions[index] = self.contribution(tensor)

        if fused_impl is not None:
            multi_tensor_applier, multi_tensor_l2norm = fused_impl
            for (device, _), entries in buckets.items():
                overflow_buffer = torch.zeros(1, dtype=torch.int32, device=device)
                _, per_tensor_norms = multi_tensor_applier(
                    multi_tensor_l2norm, overflow_buffer, [[tensor for _, tensor in entries]], True
                )
                if per_tensor_norms is None or per_tensor_norms.numel() != len(entries):
                    raise RuntimeError(
                        "The fused multi-tensor L2 kernel returned an invalid per-tensor result."
                    )
                per_tensor_squares = per_tensor_norms.square().reshape(-1)
                for (index, _), square in zip(entries, per_tensor_squares):
                    contributions[index] = square

        resolved = []
        for contribution in contributions:
            if contribution is None:
                raise RuntimeError("An L2 norm input did not produce a contribution.")
            resolved.append(contribution)
        return resolved

    def finalize(self, contribution: torch.Tensor) -> torch.Tensor:
        """Take the square root of a complete sum of squares.

        Args:
            contribution: Complete sum of squared magnitudes.

        Returns:
            L2 norm.
        """
        return contribution.sqrt()


class GlobalL2NormMetric(L2NormMetric):
    """Compute one L2 norm over all selected tensors in the local model stage.

    The reusable default selects parameters; training-specific subclasses may select another
    source kind.
    """

    name = "global-l2"
    source_kinds = frozenset({"parameter"})


class LayerL2NormMetric(L2NormMetric):
    """Compute one L2 norm over all selected tensors in each logical layer.

    Selected sites must contain a ``layers.<index>`` path segment. The full site-name prefix
    through the innermost such segment becomes the result label; for example, sites named
    ``decoder.layers.3.mlp.linear_fc1.weight`` contributes to ``decoder.layers.3``. The reusable
    default selects parameters; training-specific subclasses may select another source kind.
    """

    name = "layer-l2"
    source_kinds = frozenset({"parameter"})
    include_global = False
    _layer_pattern = re.compile(r"^(?P<layer>(?:[^.]+\.)*layers\.[0-9]+)(?:\.|$)")

    @classmethod
    def _site_layer_label(cls, site: MetricSite) -> str | None:
        match = cls._layer_pattern.match(site.name)
        return None if match is None else match.group("layer")

    @classmethod
    def _layer_label(cls, value: MetricTensor) -> str | None:
        layer_labels = [cls._site_layer_label(site) for site in value.sites]
        if layer_labels[0] is None:
            return None
        if any(label != layer_labels[0] for label in layer_labels[1:]):
            return None
        return layer_labels[0]

    @classmethod
    def _selected_values(cls, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        return [value for value in values if cls._layer_label(value) is not None]

    @classmethod
    def _reductions_by_layer(
        cls, values: Sequence[MetricTensor], *, include_global: bool = False
    ) -> list[tuple[str, Sequence[MetricTensor]]]:
        values_by_layer: dict[str, list[MetricTensor]] = {}
        for value in values:
            label = cls._layer_label(value)
            if label is None:
                raise ValueError("A prepared layer metric tensor must identify one logical layer.")
            values_by_layer.setdefault(label, []).append(value)
        reductions: list[tuple[str, Sequence[MetricTensor]]] = list(values_by_layer.items())
        if include_global and values:
            reductions.append(("global", values))
        return reductions

    def accepts(self, site: MetricSite) -> bool:
        """Select configured sites belonging to a numbered layer.

        Args:
            site: Potential observation site.

        Returns:
            Whether the site has a selected source kind and belongs to a named layer.
        """
        return super().accepts(site) and self._site_layer_label(site) is not None

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Reject tensors spanning multiple layers, then prepare L2 contributions."""
        return super().prepare(self._selected_values(values))

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        """Start one logical reduction for each numbered layer.

        Args:
            values: Prepared tensor contributions accumulated by the caller.

        Returns:
            One result or collective stage per layer, in first-observation order.
        """
        return self._start_logical_reductions(
            self._reductions_by_layer(values, include_global=self.include_global)
        )


class LayerMaxMetric(LogicalReductionMetric):
    """Compute the exact maximum value within each numbered model layer."""

    name = "layer-max"
    reduction_op = torch.distributed.ReduceOp.MAX
    include_global = False

    def accepts(self, site: MetricSite) -> bool:
        """Select sites belonging to a numbered layer."""
        return super().accepts(site) and LayerL2NormMetric._site_layer_label(site) is not None

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Compact each selected tensor to a rank-local maximum."""
        return super().prepare(LayerL2NormMetric._selected_values(values))

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        """Start one maximum reduction for each numbered layer."""
        return self._start_logical_reductions(
            LayerL2NormMetric._reductions_by_layer(values, include_global=self.include_global)
        )

    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return one maximum contribution, using negative infinity for an empty shard."""
        accumulation_dtype = _accumulation_dtype(tensor.dtype)
        if tensor.numel():
            return tensor.amax().to(dtype=accumulation_dtype)
        return torch.full((), float("-inf"), dtype=accumulation_dtype, device=tensor.device)


class LayerNormalizedEntropyMetric(LogicalReductionMetric):
    """Compute mean per-item categorical entropy by layer, normalized to the range [0, 1].

    The last tensor dimension is treated as a categorical distribution. Each input must already
    contain nonnegative scores normalized across that dimension. Entropy is divided by
    ``log(number_of_categories)`` before averaging over all items and distributed shards.
    """

    name = "layer-normalized-entropy"
    reduction_op = torch.distributed.ReduceOp.SUM
    include_global = False

    def accepts(self, site: MetricSite) -> bool:
        """Select sites belonging to a numbered layer."""
        return super().accepts(site) and LayerL2NormMetric._site_layer_label(site) is not None

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Compact each distribution tensor to an entropy sum and population count."""
        return super().prepare(LayerL2NormMetric._selected_values(values))

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        """Start one entropy reduction for each numbered layer."""
        return self._start_logical_reductions(
            LayerL2NormMetric._reductions_by_layer(values, include_global=self.include_global)
        )

    def contribution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return normalized entropy sum and number of distributions."""
        if tensor.ndim == 0 or tensor.shape[-1] < 2:
            raise ValueError(
                "Normalized entropy requires a final dimension with at least two categories."
            )
        scores = tensor.float()
        normalized_entropy = torch.special.entr(scores).sum(dim=-1) / math.log(tensor.shape[-1])
        return torch.stack(
            (normalized_entropy.sum(), normalized_entropy.new_tensor(normalized_entropy.numel()))
        )

    def finalize(self, contribution: torch.Tensor) -> torch.Tensor:
        """Divide the complete normalized-entropy sum by its population count."""
        return contribution[0] / contribution[1]


@dataclass(frozen=True)
class _SampledMedianContinuation:
    remaining_axes: tuple[str, ...]
    label: str


class LayerSampledMedianMetric(TensorMetric):
    """Estimate each layer's median from a deterministic sample of its tensor elements.

    Approximately ``1 / sample_factor`` of every nonempty local tensor is retained during
    :meth:`prepare`. Samples from repeated observations are concatenated locally, then explicitly
    all-gathered across every ``Shard`` axis before their median is computed. The participating
    ranks must have equally sized local samples, as required by :class:`AllGather`.

    Args:
        sample_factor: Approximate reduction in retained elements. A value of 100 samples about
            one percent, retaining at least one element from every nonempty tensor.
        sample_seed: Stable integer selecting a different deterministic sample.
    """

    name = "layer-sampled-median"
    include_global = False

    def __init__(self, sample_factor: int = 100, sample_seed: int = 0) -> None:
        if sample_factor <= 0:
            raise ValueError("Median sample factor must be positive.")
        self.sample_factor = sample_factor
        self.sample_seed = sample_seed

    def accepts(self, site: MetricSite) -> bool:
        """Select sites belonging to a numbered layer."""
        return super().accepts(site) and LayerL2NormMetric._site_layer_label(site) is not None

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Replace each selected tensor with a stable flat sample."""
        sampled = []
        for value in LayerL2NormMetric._selected_values(values):
            tensor = value.tensor.reshape(-1)
            if not tensor.numel():
                sampled.append(value.with_tensor(tensor))
                continue
            sample_count = max(1, (tensor.numel() + self.sample_factor - 1) // self.sample_factor)
            permutation_seed = self._permutation_seed(value)
            start = permutation_seed % tensor.numel()
            stride = ((permutation_seed >> 64) % tensor.numel()) or 1
            while math.gcd(stride, tensor.numel()) != 1:
                stride = stride + 1 if stride + 1 < tensor.numel() else 1
            indices = torch.arange(sample_count, dtype=torch.int64, device=tensor.device)
            indices.mul_(stride).add_(start).remainder_(tensor.numel())
            sampled.append(value.with_tensor(tensor.index_select(0, indices)))
        return sampled

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        """Start one sampled-median computation for each numbered layer."""
        return [
            self._start_sample(layer_values, label)
            for label, layer_values in LayerL2NormMetric._reductions_by_layer(
                values, include_global=self.include_global
            )
        ]

    def _start_sample(self, values: Sequence[MetricTensor], label: str) -> MetricStep:
        """Concatenate one result's samples and start gathering sharded populations."""
        relations = values[0].rank_relations
        if any(value.rank_relations != relations for value in values[1:]):
            raise ValueError("A sampled-median result requires identical rank relations.")
        unsupported = tuple(
            relation.axis
            for relation in relations
            if isinstance(relation.placement, (FlatShard, Owned))
        )
        if unsupported:
            raise ValueError(
                "Sampled median does not yet support FlatShard or Owned placements: "
                f"{unsupported}."
            )
        sites = tuple(site for value in values for site in value.sites)
        sample = torch.cat(tuple(value.tensor for value in values))
        sampled_value = MetricTensor(sample, sites, relations)
        shard_axes = tuple(
            relation.axis for relation in relations if isinstance(relation.placement, Shard)
        )
        return self._gather_or_finish(sampled_value, shard_axes, label)

    def resume(self, values: Sequence[MetricTensor], continuation: object) -> MetricStep:
        """Continue gathering one sample, or compute its median."""
        if not isinstance(continuation, _SampledMedianContinuation):
            raise ValueError("Invalid sampled-median continuation.")
        if len(values) != 1:
            raise ValueError("Sampled median requires exactly one collective completion.")
        return self._gather_or_finish(values[0], continuation.remaining_axes, continuation.label)

    @staticmethod
    def _gather_or_finish(
        value: MetricTensor, remaining_axes: tuple[str, ...], label: str
    ) -> MetricStep:
        if remaining_axes:
            return CollectiveStage(
                (CollectiveRequest(value, remaining_axes[0], AllGather(dim=0)),),
                _SampledMedianContinuation(remaining_axes[1:], label),
            )
        if not value.tensor.numel():
            raise ValueError("Sampled median requires at least one sampled element.")
        sample = value.tensor.to(dtype=_accumulation_dtype(value.tensor.dtype))
        return MetricResult(torch.quantile(sample, 0.5), label)

    def _permutation_seed(self, value: MetricTensor) -> int:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(self.sample_seed).encode("utf-8"))
        for site in value.sites:
            digest.update(b"\0")
            digest.update(site.kind.encode("utf-8"))
            digest.update(b"\0")
            digest.update(site.name.encode("utf-8"))
        return int.from_bytes(digest.digest(), byteorder="little")


class MeanRowL2NormMetric(TensorMetric):
    """Compute the population-weighted mean row L2 norm over all selected tensors.

    The last tensor dimension contains each row's columns. The default produces one global result
    by averaging across the combined row population of all selected tensors.
    """

    name = "mean-row-l2"

    def __init__(self) -> None:
        self._implementation = _MeanDimwiseL2NormMetric(norm_last_dim=True)

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        """Start one global mean row L2 norm computation.

        Args:
            values: Tensors whose row norms contribute to one population-weighted mean.

        Returns:
            One global result or collective stage, or no steps when no values were prepared.
        """
        if not values:
            return []
        return [self._implementation.start(values, "global")]

    def resume(self, values: Sequence[MetricTensor], continuation: object) -> MetricStep:
        """Resume one completed mean row L2 norm stage.

        Args:
            values: Completed norm or population reductions for ready tensor branches.
            continuation: Shared state supplied by the completed stage.

        Returns:
            Next collective stage or completed mean row L2 norm.
        """
        return self._implementation.resume(values, continuation)


class MeanColumnL2NormMetric(TensorMetric):
    """Compute the population-weighted mean column L2 norm over all selected tensors.

    The last tensor dimension indexes columns. The default produces one global result by averaging
    across the combined column population of all selected tensors.
    """

    name = "mean-column-l2"

    def __init__(self) -> None:
        self._implementation = _MeanDimwiseL2NormMetric(norm_last_dim=False)

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        """Start one global mean column L2 norm computation.

        Args:
            values: Tensors whose column norms contribute to one population-weighted mean.

        Returns:
            One global result or collective stage, or no steps when no values were prepared.
        """
        if not values:
            return []
        return [self._implementation.start(values, "global")]

    def resume(self, values: Sequence[MetricTensor], continuation: object) -> MetricStep:
        """Resume one completed mean column L2 norm stage.

        Args:
            values: Completed norm or population reductions for ready tensor branches.
            continuation: Shared state supplied by the completed stage.

        Returns:
            Next collective stage or completed mean column L2 norm.
        """
        return self._implementation.resume(values, continuation)


class _DimwiseL2Stage(Enum):
    RESOLVE_NORM = auto()
    RESOLVE_POPULATION = auto()


@dataclass(frozen=True)
class _DimwiseL2Branch:
    index: int
    stage: _DimwiseL2Stage
    remaining_axes: tuple[str, ...]


@dataclass(frozen=True)
class _DimwiseL2Continuation:
    label: str
    branches: tuple[_DimwiseL2Branch, ...]
    states: tuple[torch.Tensor | None, ...]


class _MeanDimwiseL2NormMetric:
    def __init__(self, norm_last_dim: bool) -> None:
        self._norm_last_dim = norm_last_dim

    def start(self, values: Sequence[MetricTensor], label: str) -> MetricStep:
        requests = []
        branches = []
        states: list[torch.Tensor | None] = [None] * len(values)
        for index, value in enumerate(values):
            if value.is_placeholder:
                accumulation_dtype = _accumulation_dtype(value.tensor.dtype)
                sum_and_count = torch.zeros(2, dtype=accumulation_dtype, device=value.tensor.device)
                population_axes = _dimwise_reduction_axes(value.rank_relations)
                self._advance_branch(
                    index=index,
                    stage=_DimwiseL2Stage.RESOLVE_POPULATION,
                    value=value.with_tensor(sum_and_count),
                    remaining_axes=population_axes,
                    requests=requests,
                    branches=branches,
                    states=states,
                )
                continue
            local_sum, norm_axes = self._local_norm_state(value)
            self._advance_branch(
                index=index,
                stage=_DimwiseL2Stage.RESOLVE_NORM,
                value=local_sum,
                remaining_axes=norm_axes,
                requests=requests,
                branches=branches,
                states=states,
            )
        return self._request_or_finish(label, requests, branches, states)

    def _local_norm_state(self, value: MetricTensor) -> tuple[MetricTensor, tuple[str, ...]]:
        flat_shards = tuple(
            relation.placement
            for relation in value.rank_relations
            if isinstance(relation.placement, FlatShard)
        )
        if len(flat_shards) > 1:
            raise ValueError("Row-wise and column-wise L2 metrics support at most one FlatShard.")
        logical_shape = flat_shards[0].logical_shape if flat_shards else tuple(value.tensor.shape)
        if not logical_shape:
            raise ValueError("Row-wise and column-wise L2 metrics require a non-scalar tensor.")
        ndim = len(logical_shape)
        norm_dims = (ndim - 1,) if self._norm_last_dim else tuple(range(ndim - 1))
        accumulation_dtype = _accumulation_dtype(value.tensor.dtype)
        if flat_shards:
            squared = value.tensor.to(dtype=accumulation_dtype).abs().square()
            local_sum = _flat_shard_dimwise_sum(squared, flat_shards[0], self._norm_last_dim)
        elif norm_dims and value.tensor.is_floating_point():
            # Reduce the norm dimensions in one kernel rather than materializing a cast and a
            # squared copy of a whole parameter.
            local_sum = torch.linalg.vector_norm(
                value.tensor, 2, dim=norm_dims, dtype=accumulation_dtype
            ).square()
        else:
            squared = value.tensor.to(dtype=accumulation_dtype).abs().square()
            local_sum = squared.sum(dim=norm_dims) if norm_dims else squared
        relations, norm_axes = _relations_after_dim_reduction(value.rank_relations, norm_dims, ndim)
        return value.with_tensor(local_sum, relations), norm_axes

    def resume(self, values: Sequence[MetricTensor], continuation: object) -> MetricStep:
        continuation = _expect_dimwise_l2_continuation(continuation)
        if len(values) != len(continuation.branches):
            raise ValueError(
                "A dimwise L2 metric requires one completion per active tensor branch."
            )
        requests = []
        branches = []
        states = list(continuation.states)
        for value, branch in zip(values, continuation.branches):
            self._advance_branch(
                index=branch.index,
                stage=branch.stage,
                value=value,
                remaining_axes=branch.remaining_axes,
                requests=requests,
                branches=branches,
                states=states,
            )
        return self._request_or_finish(continuation.label, requests, branches, states)

    def _advance_branch(
        self,
        index: int,
        stage: _DimwiseL2Stage,
        value: MetricTensor,
        remaining_axes: tuple[str, ...],
        requests: list[CollectiveRequest],
        branches: list[_DimwiseL2Branch],
        states: list[torch.Tensor | None],
    ) -> None:
        if remaining_axes:
            requests.append(CollectiveRequest(value, remaining_axes[0], AllReduce()))
            branches.append(_DimwiseL2Branch(index, stage, remaining_axes[1:]))
            return
        if stage is _DimwiseL2Stage.RESOLVE_NORM:
            norms = value.tensor.sqrt()
            sum_and_count = torch.stack(
                (norms.sum(), norms.new_tensor(norms.numel(), dtype=norms.dtype))
            )
            population_axes = _dimwise_reduction_axes(value.rank_relations)
            self._advance_branch(
                index=index,
                stage=_DimwiseL2Stage.RESOLVE_POPULATION,
                value=value.with_tensor(sum_and_count),
                remaining_axes=population_axes,
                requests=requests,
                branches=branches,
                states=states,
            )
            return
        if stage is _DimwiseL2Stage.RESOLVE_POPULATION:
            states[index] = value.tensor
            return
        raise ValueError(f"Unexpected dimwise L2 norm continuation stage: {stage}.")

    @staticmethod
    def _request_or_finish(
        label: str,
        requests: Sequence[CollectiveRequest],
        branches: Sequence[_DimwiseL2Branch],
        states: Sequence[torch.Tensor | None],
    ) -> MetricStep:
        if requests:
            return CollectiveStage(
                requests=tuple(requests),
                continuation=_DimwiseL2Continuation(
                    label=label, branches=tuple(branches), states=tuple(states)
                ),
            )
        resolved_states = []
        for state in states:
            if state is None:
                raise ValueError("A dimwise L2 tensor branch did not produce a final state.")
            resolved_states.append(state)
        if len(resolved_states) == 1:
            total = resolved_states[0]
        else:
            packed = torch.stack(tuple(resolved_states), dim=0)
            total = packed.sum(dim=0, dtype=packed.dtype)
        result = total[0] / total[1]
        return MetricResult(result, label)


def _dimwise_reduction_axes(relations: Sequence[RankRelation]) -> tuple[str, ...]:
    """Order shard reductions before owner reductions for every logical tensor branch."""
    shard_axes = tuple(
        relation.axis
        for relation in relations
        if isinstance(relation.placement, (Shard, FlatShard))
    )
    owned_axes = tuple(
        relation.axis for relation in relations if isinstance(relation.placement, Owned)
    )
    return shard_axes + owned_axes


def _relations_after_dim_reduction(
    relations: tuple[RankRelation, ...], reduced_dims: tuple[int, ...], input_ndim: int
) -> tuple[tuple[RankRelation, ...], tuple[str, ...]]:
    output_relations = []
    reduction_axes = []
    for relation in relations:
        placement = relation.placement
        if isinstance(placement, FlatShard):
            output_relations.append(relation)
            reduction_axes.append(relation.axis)
            continue
        if not isinstance(placement, Shard):
            output_relations.append(relation)
            continue
        if placement.dim is None:
            raise ValueError(
                "A row-wise or column-wise metric requires the dimension for every Shard placement."
            )
        shard_dim = _normalize_dim(placement.dim, input_ndim)
        if shard_dim in reduced_dims:
            output_relations.append(relation)
            reduction_axes.append(relation.axis)
            continue
        output_dim = shard_dim - sum(reduced_dim < shard_dim for reduced_dim in reduced_dims)
        output_relations.append(RankRelation(relation.axis, Shard(output_dim)))
    return tuple(output_relations), tuple(reduction_axes)


def _flat_shard_dimwise_sum(
    squared: torch.Tensor, placement: FlatShard, norm_last_dim: bool
) -> torch.Tensor:
    local_values = squared.reshape(-1)
    if local_values.numel() != placement.end - placement.start:
        raise ValueError("FlatShard interval length must equal the local tensor element count.")
    column_count = placement.logical_shape[-1]
    if column_count == 0:
        raise ValueError("Row-wise and column-wise L2 metrics require a nonempty last dimension.")
    row_count = 1
    for size in placement.logical_shape[:-1]:
        row_count *= size
    output_size = row_count if norm_last_dim else column_count
    local_sum = torch.zeros(output_size, dtype=squared.dtype, device=squared.device)
    if placement.start == placement.end:
        output_shape = placement.logical_shape[:-1] if norm_last_dim else (column_count,)
        return local_sum.reshape(output_shape)

    start_column = placement.start % column_count
    first_segment_size = min(column_count - start_column, local_values.numel())
    if norm_last_dim:
        start_row = placement.start // column_count
        local_sum[start_row] = local_values[:first_segment_size].sum()
    else:
        local_sum[start_column : start_column + first_segment_size].add_(
            local_values[:first_segment_size]
        )

    cursor = first_segment_size
    full_row_count = (local_values.numel() - cursor) // column_count
    if full_row_count:
        full_rows = local_values[cursor : cursor + full_row_count * column_count].reshape(
            full_row_count, column_count
        )
        if norm_last_dim:
            start_row = placement.start // column_count + 1
            local_sum[start_row : start_row + full_row_count] = full_rows.sum(dim=1)
        else:
            local_sum.add_(full_rows.sum(dim=0))
        cursor += full_row_count * column_count

    if cursor < local_values.numel():
        if norm_last_dim:
            end_row = placement.end // column_count
            local_sum[end_row] = local_values[cursor:].sum()
        else:
            local_sum[: local_values.numel() - cursor].add_(local_values[cursor:])

    output_shape = placement.logical_shape[:-1] if norm_last_dim else (column_count,)
    return local_sum.reshape(output_shape)


def _normalize_dim(dim: int, ndim: int) -> int:
    normalized = dim if dim >= 0 else dim + ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"Tensor dimension {dim} is invalid for a {ndim}-dimensional tensor.")
    return normalized


_FUSED_L2_NORM_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})


def _fused_l2_norm_impl() -> tuple[Callable[..., object], object] | None:
    """Return the multi-tensor applier and L2 kernel, or ``None`` without TransformerEngine.

    The import stays inside the call so that a build without TransformerEngine degrades to the
    per-tensor path instead of failing, and so that the kernel remains substitutable.
    """
    try:
        from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm
    except ImportError:
        return None
    return multi_tensor_applier, multi_tensor_l2norm


def _accepts_fused_l2_norm(tensor: torch.Tensor) -> bool:
    """Return whether the fused multi-tensor L2 kernel can consume a tensor.

    Tensor subclasses such as quantized parameters are excluded because the kernel reads their
    storage directly and would report a norm of the underlying representation.
    """
    return (
        type(tensor) is torch.Tensor
        and tensor.numel() > 0
        and tensor.device.type == "cuda"
        and tensor.is_contiguous()
        and tensor.dtype in _FUSED_L2_NORM_DTYPES
    )


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype.is_floating_point and torch.finfo(dtype).bits < 32:
        return torch.float32
    if dtype.is_floating_point or dtype.is_complex:
        return dtype
    return torch.float32


def _expect_dimwise_l2_continuation(continuation: object) -> _DimwiseL2Continuation:
    if not isinstance(continuation, _DimwiseL2Continuation):
        raise ValueError("Collective completion has an invalid dimwise L2 continuation.")
    return continuation
