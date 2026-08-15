# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training integration for configured tensor metrics."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeAlias

import torch

from megatron.core.optimizer import MegatronOptimizer
from megatron.core.parameter_names import CanonicalParameterNameMap
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_observation import capture_tensor_observations
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.utils import unwrap_model
from megatron.training.global_vars import get_tensorboard_writer, get_wandb_writer
from megatron.training.utils import get_nvtx_range

from .core import (
    MetricResult,
    MetricSite,
    MetricTensor,
    RankRelation,
    Replica,
    Shard,
    TensorMetric,
    TensorMetricExecutor,
)
from .definitions import (
    GlobalL2NormMetric,
    LayerL2NormMetric,
    MeanColumnL2NormMetric,
    MeanRowL2NormMetric,
)
from .optimizer_sources import (
    _build_optimizer_parameter_manifest,
    _optimizer_metric_tensors,
    _OptimizerParameterManifestEntry,
)
from .router_metrics import (
    LayerRouterDecisionEntropyMetric,
    LayerRouterExpertBiasMetric,
    LayerRouterHealthMetric,
    LayerRouterLogitsL2NormMetric,
    LayerRouterLogitsMaxMetric,
    LayerRouterLogitsSampledMedianMetric,
    LayerRouterRoutingBalanceMetric,
    LayerRouterSeqAuxDecompositionMetric,
)

__all__ = [
    "MetricResultSink",
    "ScheduledMetric",
    "TrainingTensorMetricObserver",
    "build_tensor_metric_observer",
    "parse_tensor_metric_specs",
]


MetricResultSink: TypeAlias = Callable[[TensorMetric, Sequence[MetricResult], int], None]


@dataclass(frozen=True)
class ScheduledMetric:
    """A metric paired with an externally configurable training interval.

    Args:
        metric: Metric to execute when the schedule is due.
        interval: Positive number of training iterations between observations.
    """

    metric: TensorMetric
    interval: int

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("ScheduledMetric.interval must be positive.")

    def is_due(self, iteration: int) -> bool:
        """Return whether the metric is due on a zero-based training iteration.

        Args:
            iteration: Zero-based training iteration.

        Returns:
            Whether this iteration completes an interval.
        """
        return (iteration + 1) % self.interval == 0


class _GlobalParameterL2NormMetric(GlobalL2NormMetric):
    name = "global-param-l2"


class _LayerParameterL2NormMetric(LayerL2NormMetric):
    name = "layer-param-l2"


class _GlobalParameterMeanRowL2NormMetric(MeanRowL2NormMetric):
    name = "global-param-mean-row-l2"
    source_kinds = frozenset({"parameter"})


class _GlobalParameterMeanColumnL2NormMetric(MeanColumnL2NormMetric):
    name = "global-param-mean-column-l2"
    source_kinds = frozenset({"parameter"})


class _GlobalWgradL2NormMetric(GlobalL2NormMetric):
    name = "global-wgrad-l2"
    source_kinds = frozenset({"wgrad"})


class _LayerWgradL2NormMetric(LayerL2NormMetric):
    name = "layer-wgrad-l2"
    source_kinds = frozenset({"wgrad"})
    include_global = True


class _LayerResidualAccumulatorL2NormMetric(LayerL2NormMetric):
    name = "layer-residual-accumulator-l2"
    source_kinds = frozenset({"residual_accumulator"})
    include_global = True


class _LayerResidualContributionL2NormMetric(LayerL2NormMetric):
    name = "layer-residual-contribution-l2"
    source_kinds = frozenset({"residual_contribution"})
    include_global = True


class _GlobalOutputLogitsL2NormMetric(GlobalL2NormMetric):
    name = "global-output-logits-l2"
    source_kinds = frozenset({"output_logits"})


class _GlobalMTPLogitsL2NormMetric(GlobalL2NormMetric):
    name = "global-mtp-logits-l2"
    source_kinds = frozenset({"mtp_logits"})


_TENSOR_METRIC_FACTORIES: dict[str, Callable[[], TensorMetric]] = {
    "global-param-l2": _GlobalParameterL2NormMetric,
    "layer-param-l2": _LayerParameterL2NormMetric,
    "global-param-mean-row-l2": _GlobalParameterMeanRowL2NormMetric,
    "global-param-mean-column-l2": _GlobalParameterMeanColumnL2NormMetric,
    "global-wgrad-l2": _GlobalWgradL2NormMetric,
    "layer-wgrad-l2": _LayerWgradL2NormMetric,
    "layer-residual-accumulator-l2": _LayerResidualAccumulatorL2NormMetric,
    "layer-residual-contribution-l2": _LayerResidualContributionL2NormMetric,
    "global-output-logits-l2": _GlobalOutputLogitsL2NormMetric,
    "global-mtp-logits-l2": _GlobalMTPLogitsL2NormMetric,
    "layer-router-logits-l2": LayerRouterLogitsL2NormMetric,
    "layer-router-logits-max": LayerRouterLogitsMaxMetric,
    "layer-router-logits-sampled-median": LayerRouterLogitsSampledMedianMetric,
    "layer-router-decision-entropy": LayerRouterDecisionEntropyMetric,
    "layer-router-health": LayerRouterHealthMetric,
    "layer-router-seq-aux-decomposition": LayerRouterSeqAuxDecompositionMetric,
    "layer-router-routing-balance": LayerRouterRoutingBalanceMetric,
    "layer-router-expert-bias": LayerRouterExpertBiasMetric,
}

_OPTIMIZER_SOURCE_KINDS = frozenset({"parameter", "wgrad"})
_FORWARD_SOURCE_KINDS = frozenset(
    {
        "residual_accumulator",
        "residual_contribution",
        "output_logits",
        "mtp_logits",
        "router_logits",
        "router_scores",
        "router_diagnostics",
    }
)
_SUPPORTED_SOURCE_KINDS = _OPTIMIZER_SOURCE_KINDS | _FORWARD_SOURCE_KINDS


def parse_tensor_metric_specs(specifications: Sequence[str]) -> tuple[ScheduledMetric, ...]:
    """Parse configured tensor metrics in ``NAME:INTERVAL`` form.

    Args:
        specifications: Metric names paired with positive iteration intervals.

    Returns:
        Scheduled metric instances in configuration order.

    Raises:
        ValueError: If a specification is malformed, duplicated, unknown, or has an invalid
            interval.
    """
    scheduled_metrics = []
    configured_names = set()
    available_names = ", ".join(sorted(_TENSOR_METRIC_FACTORIES))
    for specification in specifications:
        name, separator, interval_text = specification.strip().rpartition(":")
        if not separator or not name or not interval_text:
            raise ValueError(
                f"Invalid tensor metric specification {specification!r}; expected NAME:INTERVAL."
            )
        if name not in _TENSOR_METRIC_FACTORIES:
            raise ValueError(
                f"Unknown tensor metric {name!r}; available tensor metrics: {available_names}."
            )
        if name in configured_names:
            raise ValueError(f"Tensor metric {name!r} was configured more than once.")
        try:
            interval = int(interval_text)
        except ValueError as error:
            raise ValueError(
                f"Tensor metric {name!r} requires an integer interval, got {interval_text!r}."
            ) from error
        if interval <= 0:
            raise ValueError(f"Tensor metric {name!r} requires a positive interval.")
        configured_names.add(name)
        scheduled_metrics.append(ScheduledMetric(_TENSOR_METRIC_FACTORIES[name](), interval))
    return tuple(scheduled_metrics)


def build_tensor_metric_observer(
    specifications: Sequence[str], result_sink: MetricResultSink | None = None
) -> "TrainingTensorMetricObserver | None":
    """Build the training observer selected by tensor metric CLI specifications.

    Args:
        specifications: Metric names paired with intervals.
        result_sink: Optional result consumer receiving the one-based reporting iteration. The
            default writes existing scalar logging sinks.

    Returns:
        Configured observer, or ``None`` when no metrics were requested.
    """
    scheduled_metrics = parse_tensor_metric_specs(specifications)
    if not scheduled_metrics:
        return None
    return TrainingTensorMetricObserver(scheduled_metrics, result_sink=result_sink)


class TrainingTensorMetricObserver:
    """Observe short-lived forward tensors and optimizer tensors during a training iteration.

    Forward tensors are prepared while available and only compact prepared states survive until the
    pre-step commit. A metric using a forward source must therefore make its prepared state
    independent of the observed tensor; partial CUDA graphs may reuse an output buffer after the
    callback returns. Parameter metrics observe optimizer/master values when locally available,
    and wgrad metrics observe finalized ``main_grad`` or ``grad`` values at that commit point.
    Ordinary, Distributed, LayerWise, and Chained optimizer storage is represented explicitly
    across tensor, expert, data, and expert-data parallel axes. Pipeline parallelism,
    full-iteration CUDA graphs, GTP weight rematerialization, and FSDP parameter storage remain
    unsupported.

    Args:
        scheduled_metrics: Metrics paired with independent iteration intervals.
        result_sink: Optional result consumer receiving the one-based reporting iteration. The
            default writes TensorBoard or WandB and falls back to stdout on the last rank.
    """

    def __init__(
        self,
        scheduled_metrics: Sequence[ScheduledMetric],
        result_sink: MetricResultSink | None = None,
    ) -> None:
        self.scheduled_metrics = tuple(scheduled_metrics)
        self._result_sink = _write_scalar_results if result_sink is None else result_sink
        self._parameter_names: CanonicalParameterNameMap | None = None
        self._parameter_names_key: tuple[tuple[int, ...], int, int] | None = None
        self._optimizer_parameter_manifest: tuple[
            _OptimizerParameterManifestEntry, ...
        ] | None = None
        self._optimizer_parameter_manifest_key: tuple[int, int] | None = None
        self._module_names: dict[int, str] | None = None
        self._module_names_key: tuple[int, ...] | None = None
        self._active_forward_metrics: tuple[ScheduledMetric, ...] = ()
        self._active_forward_executor: TensorMetricExecutor | None = None
        self._active_pg_collection: ProcessGroupCollection | None = None
        self._active_module_names: dict[int, str] | None = None
        self._prepared_forward_values: dict[int, list[MetricTensor]] | None = None
        self._prepared_forward_iteration: int | None = None

    @contextmanager
    def observe_forward_backward(
        self,
        *,
        model: Sequence[torch.nn.Module],
        iteration: int | None,
        pg_collection: ProcessGroupCollection,
    ) -> Iterator[None]:
        """Observe forward tensors produced during one forward-backward execution.

        Args:
            model: Wrapped model chunks participating in the training step.
            iteration: Zero-based training iteration.
            pg_collection: Concrete model process groups.
        """
        if iteration is None:
            raise ValueError("Configured tensor metrics require a training iteration.")
        _validate_supported_topology(pg_collection)
        due_metrics = self._due_metrics(iteration)
        self._validate_metric_sources(due_metrics)
        forward_metrics = tuple(
            scheduled
            for scheduled in due_metrics
            if _metric_source_kinds(scheduled.metric) & _FORWARD_SOURCE_KINDS
        )
        if not forward_metrics:
            yield
            return
        if self._active_forward_metrics:
            raise RuntimeError("Tensor metric forward observation scopes must not be nested.")

        source_kinds = frozenset().union(
            *(
                _metric_source_kinds(scheduled.metric) & _FORWARD_SOURCE_KINDS
                for scheduled in forward_metrics
            )
        )
        if "router_diagnostics" in source_kinds:
            unsupported_router_axes = tuple(
                axis
                for axis, attribute in (("tensor", "tp"), ("context", "cp"))
                if (group := getattr(pg_collection, attribute, None)) is not None
                and group.size() > 1
            )
            if unsupported_router_axes:
                raise NotImplementedError(
                    "Router diagnostic metrics currently require tensor and context parallel "
                    f"size one; unsupported axes: {unsupported_router_axes}."
                )
        _validate_forward_observation_model(model, source_kinds)
        self._active_forward_metrics = forward_metrics
        self._active_forward_executor = TensorMetricExecutor(
            _tensor_metric_process_groups(pg_collection)
        )
        self._active_pg_collection = pg_collection
        self._active_module_names = self._get_module_names(model)
        self._prepared_forward_values = {
            id(scheduled.metric): [] for scheduled in forward_metrics
        }
        self._prepared_forward_iteration = iteration

        try:
            with capture_tensor_observations(self._observe_forward_tensor, source_kinds):
                yield
        except BaseException:
            self._prepared_forward_values = None
            self._prepared_forward_iteration = None
            raise
        finally:
            self._active_forward_metrics = ()
            self._active_forward_executor = None
            self._active_pg_collection = None
            self._active_module_names = None

    def _observe_forward_tensor(
        self,
        owner: object,
        name: str,
        source_kind: str,
        tensor: torch.Tensor,
        tp_shard_dim: int | None,
    ) -> None:
        """Prepare one forward tensor for every active metric accepting its site."""
        if (
            self._active_forward_executor is None
            or self._active_pg_collection is None
            or self._active_module_names is None
            or self._prepared_forward_values is None
        ):
            raise RuntimeError("A tensor observation arrived outside an active metric scope.")
        owner_name = self._active_module_names.get(id(owner))
        if owner_name is None:
            raise ValueError(
                f"Cannot resolve a canonical tensor metric name for {type(owner).__name__}."
            )
        site_name = f"{owner_name}.{name}" if owner_name else name
        site = MetricSite(site_name, source_kind)
        value = MetricTensor(
            tensor=tensor,
            sites=(site,),
            rank_relations=_forward_rank_relations(tp_shard_dim, self._active_pg_collection),
        )
        # Deliberately untimed: Megatron timers synchronize the device when they start and stop,
        # which would serialize the forward-backward pass once per observed tensor. The commit in
        # __call__ owns the "tensor-metrics" timer, where one synchronization is affordable.
        with get_nvtx_range()("tensor-metrics"):
            for scheduled in self._active_forward_metrics:
                if scheduled.metric.accepts(site):
                    self._prepared_forward_values[id(scheduled.metric)].extend(
                        self._active_forward_executor.prepare(scheduled.metric, (value,))
                    )

    @torch.no_grad()
    def __call__(
        self,
        *,
        model: Sequence[torch.nn.Module],
        optimizer: MegatronOptimizer,
        iteration: int | None,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        """Run metrics due on this training iteration.

        Args:
            model: Wrapped model chunks participating in the training step.
            optimizer: Optimizer exposing locally valid parameter and finalized-wgrad storage.
            iteration: Zero-based training iteration.
            pg_collection: Concrete model process groups.

        Raises:
            ValueError: If the iteration or process groups are unavailable.
            NotImplementedError: If a requested source or model topology is not yet supported.
        """
        if iteration is None:
            raise ValueError("Configured tensor metrics require a training iteration.")
        _validate_supported_topology(pg_collection)
        due_metrics = self._due_metrics(iteration)
        if not due_metrics:
            return
        self._validate_metric_sources(due_metrics)
        forward_metrics = tuple(
            scheduled
            for scheduled in due_metrics
            if _metric_source_kinds(scheduled.metric) & _FORWARD_SOURCE_KINDS
        )
        if forward_metrics and self._prepared_forward_iteration != iteration:
            raise RuntimeError(
                "Forward tensor metrics require observe_forward_backward() around the "
                "forward-backward execution."
            )

        try:
            with get_nvtx_range()("tensor-metrics", time=True):
                optimizer_metrics = tuple(
                    scheduled
                    for scheduled in due_metrics
                    if _metric_source_kinds(scheduled.metric) & _OPTIMIZER_SOURCE_KINDS
                )
                optimizer_values = []
                if optimizer_metrics:
                    parameter_names = self._get_parameter_names(model, pg_collection)
                    manifest = self._get_optimizer_parameter_manifest(
                        parameter_names, optimizer, pg_collection
                    )
                    optimizer_values = _optimizer_metric_tensors(
                        parameter_names,
                        optimizer,
                        pg_collection,
                        tuple(scheduled.metric for scheduled in optimizer_metrics),
                        manifest,
                    )
                executor = TensorMetricExecutor(_tensor_metric_process_groups(pg_collection))
                prepared_forward_values = self._prepared_forward_values or {}
                for scheduled in due_metrics:
                    prepared_values = list(
                        prepared_forward_values.get(id(scheduled.metric), ())
                    )
                    prepared_values.extend(executor.prepare(scheduled.metric, optimizer_values))
                    results = executor.complete(
                        scheduled.metric, executor.start(scheduled.metric, prepared_values)
                    )
                    self._result_sink(scheduled.metric, results, iteration + 1)
        finally:
            self._prepared_forward_values = None
            self._prepared_forward_iteration = None

    def _due_metrics(self, iteration: int) -> tuple[ScheduledMetric, ...]:
        """Return metrics due on ``iteration`` in configuration order."""
        return tuple(
            scheduled for scheduled in self.scheduled_metrics if scheduled.is_due(iteration)
        )

    @staticmethod
    def _validate_metric_sources(scheduled_metrics: Sequence[ScheduledMetric]) -> None:
        unsupported_sources = {
            source_kind
            for scheduled in scheduled_metrics
            for source_kind in scheduled.metric.source_kinds
            if source_kind not in _SUPPORTED_SOURCE_KINDS
        }
        if unsupported_sources:
            raise NotImplementedError(
                "The training tensor metric observer does not support sources "
                f"{sorted(unsupported_sources)}."
            )

    def _get_module_names(self, model: Sequence[torch.nn.Module]) -> dict[int, str]:
        """Return cached canonical module names used by forward observation sites."""
        key = tuple(id(model_chunk) for model_chunk in model)
        if self._module_names is None or self._module_names_key != key:
            names = {}
            use_chunk_prefix = len(model) > 1
            for chunk_index, model_chunk in enumerate(model):
                unwrapped = unwrap_model(model_chunk)
                chunk_prefix = f"model_chunk{chunk_index}." if use_chunk_prefix else ""
                for module_name, module in unwrapped.named_modules(remove_duplicate=False):
                    names.setdefault(id(module), f"{chunk_prefix}{module_name}")
            self._module_names = names
            self._module_names_key = key
        return self._module_names

    def _get_parameter_names(
        self,
        model: Sequence[torch.nn.Module],
        pg_collection: ProcessGroupCollection,
    ) -> CanonicalParameterNameMap:
        """Return the cached canonical parameter names for this model topology."""
        ep_group = getattr(pg_collection, "ep", None)
        ep_size = ep_group.size() if ep_group is not None else 1
        ep_rank = ep_group.rank() if ep_group is not None and ep_size > 1 else 0
        key = (tuple(id(model_chunk) for model_chunk in model), ep_rank, ep_size)
        if self._parameter_names is None or self._parameter_names_key != key:
            self._parameter_names = CanonicalParameterNameMap(
                model,
                expert_parallel_rank=ep_rank,
                expert_parallel_size=ep_size,
            )
            self._parameter_names_key = key
        return self._parameter_names

    def _get_optimizer_parameter_manifest(
        self,
        parameter_names: CanonicalParameterNameMap,
        optimizer: MegatronOptimizer,
        pg_collection: ProcessGroupCollection,
    ) -> tuple[_OptimizerParameterManifestEntry, ...]:
        """Return cached rank-symmetric optimizer parameter metadata."""
        key = (id(parameter_names), id(optimizer))
        if (
            self._optimizer_parameter_manifest is None
            or self._optimizer_parameter_manifest_key != key
        ):
            self._optimizer_parameter_manifest = _build_optimizer_parameter_manifest(
                parameter_names, optimizer, pg_collection
            )
            self._optimizer_parameter_manifest_key = key
        return self._optimizer_parameter_manifest


def _metric_source_kinds(metric: TensorMetric) -> frozenset[str]:
    """Return declared sources, treating an empty declaration as accepting every source."""
    return metric.source_kinds or _SUPPORTED_SOURCE_KINDS


def _forward_rank_relations(
    tp_shard_dim: int | None, pg_collection: ProcessGroupCollection
) -> tuple[RankRelation, ...]:
    """Describe a forward tensor over tensor and data/context parallel ranks."""
    tp_group = getattr(pg_collection, "tp", None)
    dp_cp_group = getattr(pg_collection, "dp_cp", None)
    if tp_group is None or dp_cp_group is None:
        raise ValueError("Forward tensor metrics require tp and dp_cp process groups.")
    tp_placement = (
        Shard(tp_shard_dim) if tp_shard_dim is not None and tp_group.size() > 1 else Replica()
    )
    dp_placement = Shard(None) if dp_cp_group.size() > 1 else Replica()
    return (RankRelation("tp", tp_placement), RankRelation("dp", dp_placement))


def _validate_forward_observation_model(
    model: Sequence[torch.nn.Module], source_kinds: frozenset[str]
) -> None:
    """Reject graph modes that cannot produce reliable Python observation notifications."""
    for model_chunk in model:
        config = getattr(unwrap_model(model_chunk), "config", None)
        cuda_graph_impl = getattr(config, "cuda_graph_impl", "none")
        if cuda_graph_impl == "full_iteration":
            raise NotImplementedError(
                "Forward tensor metrics do not yet support full-iteration CUDA graphs."
            )
        if source_kinds.isdisjoint(
            {"router_logits", "router_scores", "router_diagnostics"}
        ) or cuda_graph_impl not in {
            "local",
            "transformer_engine",
        }:
            continue
        graph_modules = tuple(getattr(config, "cuda_graph_modules", ()))
        if not graph_modules or any(
            graph_module
            in {
                CudaGraphModule.moe,
                CudaGraphModule.moe_router,
                CudaGraphModule.moe_preprocess,
            }
            for graph_module in graph_modules
        ):
            raise NotImplementedError(
                "Router tensor metrics require an eager MoE router; disable moe, "
                "moe_router, and moe_preprocess CUDA graph modules for now."
            )


def _tensor_metric_process_groups(
    pg_collection: ProcessGroupCollection,
) -> dict[str, torch.distributed.ProcessGroup]:
    """Bind abstract metric axes to caller-supplied process groups."""
    groups = {}
    for axis, attribute in (
        ("tp", "tp"),
        ("expert_tp", "expt_tp"),
        ("ep", "ep"),
        ("dp", "dp_cp"),
        ("expert_dp", "expt_dp"),
    ):
        group = getattr(pg_collection, attribute, None)
        if group is not None:
            groups[axis] = group
    return groups


def _validate_supported_topology(pg_collection: ProcessGroupCollection) -> None:
    unsupported_axes = []
    for axis, attribute in (
        ("pipeline parallelism", "pp"),
        ("GTP weight rematerialization", "gtp_remat"),
        ("expert GTP weight rematerialization", "expt_gtp_remat"),
    ):
        process_group = getattr(pg_collection, attribute, None)
        if process_group is not None and process_group.size() > 1:
            unsupported_axes.append(axis)
    if unsupported_axes:
        raise NotImplementedError(
            "The training tensor metric observer does not yet support "
            + ", ".join(unsupported_axes)
            + "."
        )


def _write_scalar_results(
    metric: TensorMetric, results: Sequence[MetricResult], iteration: int
) -> None:
    scalars = {}
    for index, result in enumerate(results):
        if result.value.tensor.numel() != 1:
            raise ValueError(
                f"The default tensor metric sink requires scalar results, but {metric.name!r} "
                f"produced shape {tuple(result.value.tensor.shape)}."
            )
        if result.label is not None:
            result_name = str(result.label)
        elif len(result.value.sites) == 1:
            result_name = result.value.sites[0].name
        else:
            result_name = f"result-{index}"
        tag = f"tensor-metrics/{metric.name}/{result_name}"
        if tag in scalars:
            raise ValueError(f"Tensor metric {metric.name!r} produced duplicate result tag {tag!r}.")
        scalars[tag] = result.value.tensor.detach()

    tensorboard_writer = get_tensorboard_writer()
    if tensorboard_writer is not None:
        for tag, value in scalars.items():
            tensorboard_writer.add_scalar(tag, value, iteration)
    wandb_writer = get_wandb_writer()
    if wandb_writer is not None and scalars:
        wandb_writer.log(scalars, iteration)
    if scalars and tensorboard_writer is None and wandb_writer is None and _is_last_rank():
        printable_scalars = {tag: value.item() for tag, value in scalars.items()}
        print(f"tensor metrics at iteration {iteration}: {printable_scalars}", flush=True)


def _is_last_rank() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1
    )
