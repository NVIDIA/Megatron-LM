# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor metrics for diagnosing MoE router load-balancing behavior."""

# A compact first-line router health configuration is:
#   --tensor-metrics \
#     layer-router-health:100 \
#     layer-router-decision-entropy:100 \
#     layer-router-logits-l2:100
#
# The health bundle reports global and worst-layer rollups. Enable the decomposition,
# routing-balance, or expert-bias families when one of those signals needs deeper diagnosis.

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from megatron.core.transformer.moe.router_diagnostics import (
    ROUTER_DIAGNOSTIC_CHANNEL_COUNT,
    RouterDiagnosticChannel,
)

from .core import (
    AllReduce,
    CollectiveCompletionSet,
    CollectiveRequest,
    CollectiveRequestSet,
    FlatShard,
    MetricGroup,
    MetricResult,
    MetricSite,
    MetricStep,
    MetricTensor,
    Owned,
    PerGroupTensorMetric,
    Shard,
)
from .definitions import (
    LayerL2NormMetric,
    LayerMaxMetric,
    LayerNormalizedEntropyMetric,
    LayerSampledMedianMetric,
)

__all__ = [
    "LayerRouterDecisionEntropyMetric",
    "LayerRouterExpertBiasMetric",
    "LayerRouterHealthMetric",
    "LayerRouterLogitsL2NormMetric",
    "LayerRouterLogitsMaxMetric",
    "LayerRouterLogitsSampledMedianMetric",
    "LayerRouterRoutingBalanceMetric",
    "LayerRouterSeqAuxDecompositionMetric",
]


class LayerRouterLogitsL2NormMetric(LayerL2NormMetric):
    """Report exact per-layer and global router-logit L2 norms."""

    name = "layer-router-logits-l2"
    source_kinds = frozenset({"router_logits"})
    include_global = True


class LayerRouterLogitsMaxMetric(LayerMaxMetric):
    """Report exact per-layer and global router-logit maxima."""

    name = "layer-router-logits-max"
    source_kinds = frozenset({"router_logits"})
    include_global = True


class LayerRouterLogitsSampledMedianMetric(LayerSampledMedianMetric):
    """Report sampled per-layer and global router-logit medians."""

    name = "layer-router-logits-sampled-median"
    source_kinds = frozenset({"router_logits"})
    include_global = True


class LayerRouterDecisionEntropyMetric(LayerNormalizedEntropyMetric):
    """Report per-layer and global normalized router decision entropy."""

    name = "layer-router-decision-entropy"
    source_kinds = frozenset({"router_scores"})
    include_global = True


@dataclass(frozen=True)
class _RouterOperation:
    name: str
    statistic: str
    aggregation: str
    reduction_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM
    worst_is_min: bool = False


@dataclass(frozen=True)
class _RouterMetricLabel:
    layer: str
    operation: _RouterOperation

    def __str__(self) -> str:
        return f"{self.layer}/{self.operation.name}"


@dataclass(frozen=True)
class _RouterMetricContinuation:
    remaining_axes: tuple[str, ...]
    label: _RouterMetricLabel


class _LayerRouterDiagnosticMetric(PerGroupTensorMetric):
    source_kinds = frozenset({"router_diagnostics"})
    operations: tuple[_RouterOperation, ...] = ()
    include_layer_groups = True
    include_worst_layer_group = False

    def accepts(self, site: MetricSite) -> bool:
        """Select router diagnostic sites belonging to numbered layers."""
        return super().accepts(site) and LayerL2NormMetric._site_layer_label(site) is not None

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Retain an independent copy of each compact router diagnostic observation."""
        prepared = []
        for value in LayerL2NormMetric._selected_values(values):
            tensor = value.tensor
            if (
                tensor.ndim != 3
                or tensor.shape[0] == 0
                or tensor.shape[1] != ROUTER_DIAGNOSTIC_CHANNEL_COUNT
                or tensor.shape[2] < 2
            ):
                raise ValueError(
                    "Router diagnostic metrics require shape "
                    "[batch, ROUTER_DIAGNOSTIC_CHANNEL_COUNT, num_experts]."
                )
            prepared.append(value.with_tensor(tensor.float().clone()))
        return prepared

    def groups(self, values: Sequence[MetricTensor]) -> list[MetricGroup]:
        """Create configured layer, global, and worst-layer diagnostic result groups."""
        grouped_values: dict[str, list[MetricTensor]] = {}
        for value in values:
            layer = LayerL2NormMetric._layer_label(value)
            if layer is None:
                raise ValueError("A router diagnostic tensor must identify one logical layer.")
            grouped_values.setdefault(layer, []).append(value)

        groups = []
        if self.include_layer_groups:
            groups.extend(
                MetricGroup(tuple(layer_values), label=_RouterMetricLabel(layer, operation))
                for layer, layer_values in grouped_values.items()
                for operation in self.operations
            )
        if values:
            groups.extend(
                MetricGroup(tuple(values), label=_RouterMetricLabel("global", operation))
                for operation in self.operations
            )
        if values and self.include_worst_layer_group:
            groups.extend(
                MetricGroup(tuple(values), label=_RouterMetricLabel("worst-layer", operation))
                for operation in self.operations
            )
        return groups

    def start_group(self, group: MetricGroup) -> MetricStep:
        """Build a compact contribution and reduce it over data-parallel ranks."""
        label = group.label
        if not isinstance(label, _RouterMetricLabel):
            raise ValueError("Invalid router diagnostic metric label.")
        relations = group.items[0].rank_relations
        if any(value.rank_relations != relations for value in group.items[1:]):
            raise ValueError("A router diagnostic group requires identical rank relations.")
        unsupported = tuple(
            relation.axis
            for relation in relations
            if isinstance(relation.placement, (FlatShard, Owned))
            or (isinstance(relation.placement, Shard) and relation.axis != "dp")
        )
        if unsupported:
            raise NotImplementedError(
                "Router diagnostic metrics do not yet support sharded tensor or context "
                f"parallel observations; unsupported axes: {unsupported}."
            )

        contribution = self._group_contribution(group, label)
        sites = tuple(site for value in group.items for site in value.sites)
        value = MetricTensor(contribution, sites, relations)
        reduction_axes = tuple(
            relation.axis for relation in relations if isinstance(relation.placement, Shard)
        )
        return self._reduce_or_finish(value, reduction_axes, label)

    def _group_contribution(self, group: MetricGroup, label: _RouterMetricLabel) -> torch.Tensor:
        if label.layer == "worst-layer":
            return torch.stack(
                tuple(
                    self._contribution(
                        torch.cat(tuple(value.tensor for value in layer_values), dim=0),
                        label.operation,
                    )
                    for layer_values in self._values_by_layer(group).values()
                )
            )
        if label.layer != "global" or label.operation.aggregation != "global":
            diagnostics = torch.cat(tuple(value.tensor for value in group.items), dim=0)
            return self._contribution(diagnostics, label.operation)

        return torch.stack(
            tuple(
                self._contribution(
                    torch.cat(tuple(value.tensor for value in layer_values), dim=0), label.operation
                )
                for layer_values in self._values_by_layer(group).values()
            )
        )

    @staticmethod
    def _values_by_layer(group: MetricGroup) -> dict[str, list[MetricTensor]]:
        values_by_layer: dict[str, list[MetricTensor]] = {}
        for value in group.items:
            layer = LayerL2NormMetric._layer_label(value)
            if layer is None:
                raise ValueError("A router diagnostic rollup requires numbered layers.")
            values_by_layer.setdefault(layer, []).append(value)
        return values_by_layer

    def resume_group(self, completed: CollectiveCompletionSet) -> MetricStep:
        """Finish a distributed diagnostic statistic after its reduction."""
        continuation = completed.continuation
        if not isinstance(continuation, _RouterMetricContinuation):
            raise ValueError("Invalid router diagnostic metric continuation.")
        if len(completed.values) != 1:
            raise ValueError("A router diagnostic metric requires one collective completion.")
        return self._reduce_or_finish(
            completed.values[0], continuation.remaining_axes, continuation.label
        )

    def _reduce_or_finish(
        self, value: MetricTensor, remaining_axes: tuple[str, ...], label: _RouterMetricLabel
    ) -> MetricStep:
        if remaining_axes:
            return CollectiveRequestSet(
                (
                    CollectiveRequest(
                        value, remaining_axes[0], AllReduce(label.operation.reduction_op)
                    ),
                ),
                _RouterMetricContinuation(remaining_axes[1:], label),
            )
        if label.layer == "worst-layer":
            layer_results = torch.stack(
                tuple(
                    self._finalize(contribution, label.operation) for contribution in value.tensor
                )
            )
            result = layer_results.amin() if label.operation.worst_is_min else layer_results.amax()
        else:
            result = self._finalize(value.tensor, label.operation)
        return MetricResult(MetricTensor(result, value.sites), label)

    @staticmethod
    def _contribution(diagnostics: torch.Tensor, operation: _RouterOperation) -> torch.Tensor:
        if operation.aggregation == "global":
            weights = diagnostics[:, RouterDiagnosticChannel.VALID_TOKEN_COUNT, 0]
            mean_score = diagnostics[:, RouterDiagnosticChannel.MEAN_SCORE]
            aux_load = diagnostics[:, RouterDiagnosticChannel.AUX_LOAD]
            actual_load = diagnostics[:, RouterDiagnosticChannel.ACTUAL_LOAD]
            global_contribution = torch.cat(
                tuple(
                    (distribution * weights.unsqueeze(-1)).sum(dim=0)
                    for distribution in (mean_score, aux_load, actual_load)
                )
            )
            if operation.statistic != "seq_global_aux_loss_gap":
                return global_contribution
            valid = weights > 0
            seq_losses = _per_sequence_statistic(diagnostics, "seq_aux_loss")[valid]
            return torch.cat(
                (
                    torch.stack((seq_losses.sum(), seq_losses.new_tensor(seq_losses.numel()))),
                    global_contribution,
                )
            )

        valid = diagnostics[:, RouterDiagnosticChannel.VALID_TOKEN_COUNT, 0] > 0
        values = _per_sequence_statistic(diagnostics, operation.statistic)[valid]
        if operation.aggregation == "mean":
            return torch.stack((values.sum(), values.new_tensor(values.numel())))
        if operation.aggregation == "max":
            return values.amax() if values.numel() else values.new_tensor(float("-inf"))
        if operation.aggregation == "min":
            return values.amin() if values.numel() else values.new_tensor(float("inf"))
        raise ValueError(f"Unknown router diagnostic aggregation {operation.aggregation!r}.")

    @staticmethod
    def _finalize(contribution: torch.Tensor, operation: _RouterOperation) -> torch.Tensor:
        if operation.aggregation == "global" and contribution.ndim == 2:
            layer_results = torch.stack(
                tuple(
                    _LayerRouterDiagnosticMetric._finalize(layer_contribution, operation)
                    for layer_contribution in contribution
                )
            )
            return layer_results.mean()
        if operation.aggregation == "mean":
            return contribution[0] / contribution[1].clamp_min(1)
        if operation.aggregation in {"max", "min"}:
            return contribution
        if operation.aggregation != "global":
            raise ValueError(f"Unknown router diagnostic aggregation {operation.aggregation!r}.")

        if operation.statistic == "seq_global_aux_loss_gap":
            seq_loss = contribution[0] / contribution[1].clamp_min(1)
            global_batch_loss = _LayerRouterDiagnosticMetric._finalize(
                contribution[2:], _RouterOperation("", "global_batch_aux_loss", "global")
            )
            return seq_loss - global_batch_loss

        num_experts = contribution.numel() // 3
        mean_score = contribution[:num_experts]
        aux_load = contribution[num_experts : 2 * num_experts]
        actual_load = contribution[2 * num_experts :]
        mean_score = mean_score / mean_score.sum().clamp_min(1)
        aux_load = aux_load / aux_load.sum().clamp_min(1)
        actual_load = actual_load / actual_load.sum().clamp_min(1)
        if operation.statistic == "global_batch_aux_loss":
            return num_experts * (aux_load * mean_score).sum()
        if operation.statistic == "actual_load_imbalance":
            return _load_imbalance(actual_load)
        if operation.statistic == "aux_actual_tv":
            return 0.5 * (aux_load - actual_load).abs().sum()
        if operation.statistic == "aux_max_over_mean":
            return aux_load.amax() * num_experts
        if operation.statistic == "actual_max_over_mean":
            return actual_load.amax() * num_experts
        raise ValueError(f"Unknown global router statistic {operation.statistic!r}.")


class LayerRouterHealthMetric(_LayerRouterDiagnosticMetric):
    """Report compact global and worst-layer MoE router health signals.

    Balanced baselines are one for the loss and max-over-mean metrics and zero for imbalance and
    loss-gap metrics. Larger top-k boundary margins are healthier.
    """

    name = "layer-router-health"
    include_layer_groups = False
    include_worst_layer_group = True
    operations = (
        _RouterOperation("seq-loss", "seq_aux_loss", "mean"),
        _RouterOperation("seq-assignment-imbalance", "aux_load_imbalance", "mean"),
        _RouterOperation("seq-score-imbalance", "score_imbalance", "mean"),
        _RouterOperation("global-batch-loss", "global_batch_aux_loss", "global"),
        _RouterOperation("seq-global-loss-gap", "seq_global_aux_loss_gap", "global"),
        _RouterOperation("global-aux-max-over-mean", "aux_max_over_mean", "global"),
        _RouterOperation("global-actual-imbalance", "actual_load_imbalance", "global"),
        _RouterOperation("global-actual-max-over-mean", "actual_max_over_mean", "global"),
        _RouterOperation(
            "topk-boundary-relative-margin",
            "topk_boundary_relative_margin",
            "mean",
            worst_is_min=True,
        ),
    )


class LayerRouterSeqAuxDecompositionMetric(_LayerRouterDiagnosticMetric):
    """Decompose the sequence auxiliary loss into load, score, and alignment terms."""

    name = "layer-router-seq-aux-decomposition"
    operations = (
        _RouterOperation("loss-mean", "seq_aux_loss", "mean"),
        _RouterOperation("loss-max", "seq_aux_loss", "max", torch.distributed.ReduceOp.MAX),
        _RouterOperation("assignment-imbalance-mean", "aux_load_imbalance", "mean"),
        _RouterOperation("score-imbalance-mean", "score_imbalance", "mean"),
        _RouterOperation("imbalance-coupling-mean", "imbalance_coupling", "mean"),
        _RouterOperation("imbalance-alignment-mean", "imbalance_alignment", "mean"),
    )


class LayerRouterRoutingBalanceMetric(_LayerRouterDiagnosticMetric):
    """Compare unbiased auxiliary assignments with the dispatched assignments."""

    name = "layer-router-routing-balance"
    operations = (
        _RouterOperation("actual-imbalance-mean", "actual_load_imbalance", "mean"),
        _RouterOperation(
            "actual-imbalance-max", "actual_load_imbalance", "max", torch.distributed.ReduceOp.MAX
        ),
        _RouterOperation("aux-actual-tv-mean", "aux_actual_tv", "mean"),
        _RouterOperation("topk-overlap-mean", "aux_actual_overlap", "mean"),
        _RouterOperation(
            "topk-overlap-min", "aux_actual_overlap", "min", torch.distributed.ReduceOp.MIN
        ),
        _RouterOperation("aux-max-over-mean", "aux_max_over_mean", "mean"),
        _RouterOperation("actual-max-over-mean", "actual_max_over_mean", "mean"),
        _RouterOperation("actual-inactive-fraction", "actual_inactive_fraction", "mean"),
        _RouterOperation("global-actual-imbalance", "actual_load_imbalance", "global"),
        _RouterOperation("global-aux-actual-tv", "aux_actual_tv", "global"),
        _RouterOperation("global-aux-max-over-mean", "aux_max_over_mean", "global"),
        _RouterOperation("global-actual-max-over-mean", "actual_max_over_mean", "global"),
    )


class LayerRouterExpertBiasMetric(_LayerRouterDiagnosticMetric):
    """Track expert-bias magnitude and its relationship to router preferences and load."""

    name = "layer-router-expert-bias"
    operations = (
        _RouterOperation("mean", "bias_mean", "mean"),
        _RouterOperation("std", "bias_std", "mean"),
        _RouterOperation("rms", "bias_rms", "mean"),
        _RouterOperation("range", "bias_range", "mean"),
        _RouterOperation("abs-max", "bias_abs_max", "mean"),
        _RouterOperation("score-correlation", "bias_score_correlation", "mean"),
        _RouterOperation("aux-load-correlation", "bias_aux_load_correlation", "mean"),
        _RouterOperation("actual-load-correlation", "bias_actual_load_correlation", "mean"),
    )


def _per_sequence_statistic(diagnostics: torch.Tensor, statistic: str) -> torch.Tensor:
    score = diagnostics[:, RouterDiagnosticChannel.MEAN_SCORE]
    aux_load = diagnostics[:, RouterDiagnosticChannel.AUX_LOAD]
    actual_load = diagnostics[:, RouterDiagnosticChannel.ACTUAL_LOAD]
    bias = diagnostics[:, RouterDiagnosticChannel.EXPERT_BIAS]
    num_experts = diagnostics.shape[-1]
    uniform = 1.0 / num_experts
    aux_delta = aux_load - uniform
    score_delta = score - uniform

    if statistic == "seq_aux_loss":
        return num_experts * (aux_load * score).sum(dim=-1)
    if statistic == "aux_load_imbalance":
        return num_experts * aux_delta.square().sum(dim=-1)
    if statistic == "score_imbalance":
        return num_experts * score_delta.square().sum(dim=-1)
    if statistic == "imbalance_coupling":
        return num_experts * (aux_delta * score_delta).sum(dim=-1)
    if statistic == "imbalance_alignment":
        coupling = (aux_delta * score_delta).sum(dim=-1)
        denominator = (aux_delta.square().sum(dim=-1) * score_delta.square().sum(dim=-1)).sqrt()
        return torch.where(denominator > 0, coupling / denominator, torch.zeros_like(coupling))
    if statistic == "actual_load_imbalance":
        return num_experts * (actual_load - uniform).square().sum(dim=-1)
    if statistic == "aux_actual_tv":
        return 0.5 * (aux_load - actual_load).abs().sum(dim=-1)
    if statistic == "aux_actual_overlap":
        return diagnostics[:, RouterDiagnosticChannel.AUX_ACTUAL_OVERLAP, 0]
    if statistic == "aux_max_over_mean":
        return aux_load.amax(dim=-1) * num_experts
    if statistic == "actual_max_over_mean":
        return actual_load.amax(dim=-1) * num_experts
    if statistic == "actual_inactive_fraction":
        return (actual_load == 0).float().mean(dim=-1)
    if statistic == "topk_boundary_relative_margin":
        return diagnostics[:, RouterDiagnosticChannel.TOPK_BOUNDARY_RELATIVE_MARGIN, 0]
    if statistic == "bias_mean":
        return bias.mean(dim=-1)
    if statistic == "bias_std":
        return bias.std(dim=-1, correction=0)
    if statistic == "bias_rms":
        return bias.square().mean(dim=-1).sqrt()
    if statistic == "bias_range":
        return bias.amax(dim=-1) - bias.amin(dim=-1)
    if statistic == "bias_abs_max":
        return bias.abs().amax(dim=-1)
    if statistic == "bias_score_correlation":
        return _correlation(bias, score)
    if statistic == "bias_aux_load_correlation":
        return _correlation(bias, aux_load)
    if statistic == "bias_actual_load_correlation":
        return _correlation(bias, actual_load)
    raise ValueError(f"Unknown per-sequence router statistic {statistic!r}.")


def _load_imbalance(load: torch.Tensor) -> torch.Tensor:
    num_experts = load.shape[-1]
    return num_experts * (load - 1.0 / num_experts).square().sum(dim=-1)


def _correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left - left.mean(dim=-1, keepdim=True)
    right = right - right.mean(dim=-1, keepdim=True)
    numerator = (left * right).sum(dim=-1)
    denominator = (left.square().sum(dim=-1) * right.square().sum(dim=-1)).sqrt()
    return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))
