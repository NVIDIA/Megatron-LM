# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoE metrics tracking and logging utilities.

This module provides a clean interface for collecting, reducing, and logging
MoE-related metrics across distributed training.

Usage:
    tracker = MoEMetricsTracker.get_instance()
    tracker.record("load_balancing_loss", loss, layer_number, num_layers, ...)
    log_string = tracker.track(
        loss_scale=1.0,
        iteration=100,
        writer=tb_writer,
        num_layers=32,
        ...
    )
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Union

import torch

from megatron.core import parallel_state
from megatron.core.utils import internal_api


@dataclass
class _MetricEntry:
    """Storage for a single metric across layers."""

    values: torch.Tensor
    reduce_group: Optional[torch.distributed.ProcessGroup] = None
    avg_group: Optional[torch.distributed.ProcessGroup] = None
    percentiles: Optional[List[float]] = None  # e.g., [0.5, 0.95] for p50, p95


class MoEMetricsTracker:
    """Singleton tracker for MoE layer-wise metrics.

    This class manages the collection, reduction, and logging of MoE metrics
    across distributed training. Metrics are lazily initialized on first record.

    Example:
        tracker = MoEMetricsTracker.get_instance()
        tracker.record("load_balancing_loss", loss, layer_number=1, num_layers=32)
        log_string = tracker.track(
            loss_scale=1.0,
            iteration=100,
            writer=tb_writer,
            num_layers=32,
        )
    """

    _instance: ClassVar[Optional['MoEMetricsTracker']] = None

    def __init__(self):
        self._metrics: Dict[str, _MetricEntry] = {}

    @classmethod
    def get_instance(cls) -> 'MoEMetricsTracker':
        """Get the singleton instance of the tracker."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record(
        self,
        name: str,
        value: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: Optional[torch.distributed.ProcessGroup] = None,
        avg_group: Optional[torch.distributed.ProcessGroup] = None,
        percentiles: Optional[List[float]] = None,
    ) -> None:
        """Record a metric value for a specific layer.

        Lazily creates the metric entry on first call for each metric name.

        Args:
            name: Name of the metric (e.g., "load_balancing_loss", "z_loss").
            value: The metric value to record (will be detached).
            layer_number: 1-based layer index.
            num_layers: Total number of layers.
            reduce_group: Process group for all-reduce operations.
            avg_group: Process group for averaging operations.
            percentiles: List of percentiles to compute (e.g., [0.5, 0.95] for p50, p95).
        """
        if layer_number is None:
            return

        if name not in self._metrics:
            self._metrics[name] = _MetricEntry(values=torch.zeros(num_layers, device=value.device))

        entry = self._metrics[name]
        entry.values[layer_number - 1] += value.detach()
        entry.reduce_group = reduce_group
        entry.avg_group = avg_group
        entry.percentiles = percentiles

    def track(
        self,
        loss_scale: float,
        iteration: int,
        writer,
        wandb_writer=None,
        per_layer_logging: bool = False,
        force_initialize: bool = False,
        names: Optional[Union[str, List[str]]] = None,
        num_layers: Optional[int] = None,
        moe_layer_freq: Optional[Union[int, List[int]]] = None,
        mtp_num_layers: Optional[int] = None,
    ) -> str:
        """Track MoE metrics: reduce, write to backends, and return log string.

        This is the main entry point for tracking MoE metrics during training.
        It performs: force initialization -> reduce across ranks -> aggregate ->
        write to TensorBoard/W&B -> get log string -> clear.

        Args:
            loss_scale: Scale factor to average metrics across batches,
                usually is 1/num_microbatches.
            iteration: Current training iteration.
            writer: TensorBoard writer.
            wandb_writer: Weights & Biases writer (optional).
            per_layer_logging: Whether to log per-layer values.
            force_initialize: Whether to force initialize metrics that don't exist.
            names: Metric name(s) to track. If None, tracks all metrics.
            num_layers: Total number of layers (required if force_initialize is True).
            moe_layer_freq: Frequency of MoE layers or list indicating MoE layer pattern.
            mtp_num_layers: Number of MTP (Multi-Token Prediction) layers to add.

        Returns:
            Formatted log string for console output.
        """
        names_list = self._resolve_names(names)

        # Force initialize metrics if needed
        if force_initialize and names_list is not None:
            for name in names_list:
                self.initialize_metric(name, num_layers, device=torch.cuda.current_device())

        # Reduce metrics across ranks
        self.reduce_across_ranks(names_list)

        # Compute number of MoE layers
        num_moe_layers = self._compute_num_moe_layers(num_layers, moe_layer_freq, mtp_num_layers)

        # Aggregate metrics to scalars
        aggregated = self._aggregate_metrics(loss_scale, num_moe_layers, names_list)

        # Write aggregated metrics to TensorBoard/W&B
        self.write(aggregated, iteration, writer, wandb_writer)

        # Write per-layer values if requested
        if per_layer_logging:
            self._write_per_layer_metrics(loss_scale, iteration, writer, wandb_writer, names_list)

        # Get log string for console
        log_string = self.get_log_string(aggregated)

        # Clear after tracking
        self.clear()

        return log_string

    def reduce_across_ranks(self, names: Optional[Union[str, List[str]]] = None) -> None:
        """Reduce metrics across distributed ranks.

        Performs multi-stage reduction:
        1. Collect across Pipeline Parallel (PP)
        2. Reduce using custom reduce_group (e.g., tp_cp_group)
        3. Average using custom avg_group
        4. Average across Data Parallel (DP) - except for global_load_balancing_loss

        Args:
            names: Metric name(s) to reduce. If None, reduces all metrics.
        """
        names_list = self._resolve_names(names)

        for name in names_list:
            if name not in self._metrics:
                continue

            entry = self._metrics[name]
            values = entry.values

            # Collect aux losses across PP
            torch.distributed.all_reduce(
                values, group=parallel_state.get_pipeline_model_parallel_group()
            )

            # Reduce aux losses across custom reduce_group
            if entry.reduce_group is not None:
                torch.distributed.all_reduce(values, group=entry.reduce_group)

            # Average aux losses across custom avg_group
            if entry.avg_group is not None:
                torch.distributed.all_reduce(
                    values, group=entry.avg_group, op=torch.distributed.ReduceOp.AVG
                )

            # Average across data parallel ranks
            # global_load_balancing_loss already uses tp_dp_cp_group in reduce_group
            if name != "global_load_balancing_loss":
                torch.distributed.all_reduce(
                    values,
                    group=parallel_state.get_data_parallel_group(with_context_parallel=False),
                    op=torch.distributed.ReduceOp.AVG,
                )

    def get_log_string(self, aggregated: Dict[str, float]) -> str:
        """Get formatted log string for console output.

        Args:
            aggregated: Dictionary of aggregated metric name -> value.

        Returns:
            Formatted log string for console output.
        """
        log_parts = [f" {name}: {value:.4f} |" for name, value in aggregated.items()]
        return "".join(log_parts)

    def write(
        self, aggregated: Dict[str, float], iteration: int, writer, wandb_writer=None
    ) -> None:
        """Write aggregated metrics to TensorBoard and/or Weights & Biases.

        Args:
            aggregated: Dictionary of aggregated metric name -> value.
            iteration: Current training iteration.
            writer: TensorBoard writer.
            wandb_writer: Weights & Biases writer (optional).
        """
        for name, value in aggregated.items():
            self._write_scalar_metric(name, value, iteration, writer, wandb_writer)

    def clear(self) -> None:
        """Clear metric values (zero out tensors)."""
        for name in self._metrics.keys():
            if name in self._metrics:
                self._metrics[name].values.zero_()

    def get_metrics(self) -> Dict[str, _MetricEntry]:
        """Get all recorded metrics."""
        return self._metrics

    def get_raw_tracker(self) -> Dict[str, dict]:
        """Get metrics in the legacy dict format for backward compatibility."""
        return {
            name: {
                "values": entry.values,
                "reduce_group": entry.reduce_group,
                "avg_group": entry.avg_group,
                "percentiles": entry.percentiles,
            }
            for name, entry in self._metrics.items()
        }

    def initialize_metric(
        self,
        name: str,
        num_layers: int,
        device: str = "cuda",
        percentiles: Optional[List[float]] = None,
    ) -> None:
        """Force initialize a metric entry."""
        if name not in self._metrics:
            self._metrics[name] = _MetricEntry(
                values=torch.zeros(num_layers, device=device), percentiles=percentiles
            )

    # =========================================================================
    # Private helper methods
    # =========================================================================

    def _resolve_names(self, names: Optional[Union[str, List[str]]]) -> List[str]:
        """Resolve names argument to a list of strings."""
        if names is None:
            return list(self._metrics.keys())
        if isinstance(names, str):
            return [names]
        return names

    def _compute_num_moe_layers(
        self,
        num_layers: Optional[int],
        moe_layer_freq: Optional[Union[int, List[int]]],
        mtp_num_layers: Optional[int],
    ) -> int:
        """Compute the number of MoE layers based on configuration."""
        if moe_layer_freq is None:
            num_moe_layers = num_layers
        elif isinstance(moe_layer_freq, int):
            assert isinstance(num_layers, int)
            moe_layer_pattern = [1 if (i % moe_layer_freq == 0) else 0 for i in range(num_layers)]
            num_moe_layers = sum(moe_layer_pattern)
        elif isinstance(moe_layer_freq, list):
            num_moe_layers = sum(moe_layer_freq)
        else:
            raise ValueError(f"Invalid moe_layer_freq: {moe_layer_freq}")

        if mtp_num_layers is not None:
            num_moe_layers += mtp_num_layers

        return num_moe_layers

    def _aggregate_metrics(
        self, loss_scale: float, num_moe_layers: int, names: List[str]
    ) -> Dict[str, float]:
        """Aggregate layer-wise metrics to scalar values.

        For all metrics: computes average across layers.
        If percentiles is set: also computes specified percentiles as additional scalars.

        Args:
            loss_scale: Scale factor to apply to losses.
            num_moe_layers: Number of MoE layers for averaging.
            names: List of metric names to aggregate.

        Returns:
            Dictionary of aggregated metric name -> scalar value.
        """
        aggregated = {}

        for name in names:
            if name not in self._metrics:
                continue

            entry = self._metrics[name]
            values = entry.values.float() * loss_scale

            # Compute percentiles if configured
            if entry.percentiles is not None:
                vals = values[values > 0]
                pct_values = torch.quantile(
                    vals.float(), torch.tensor(entry.percentiles, device=vals.device)
                ).tolist()
                for pct, pct_val in zip(entry.percentiles, pct_values):
                    aggregated[f"{name}_p{int(pct * 100)}"] = pct_val

            # Always compute mean
            aggregated[name] = (values.sum() / num_moe_layers).item()

        return aggregated

    def _write_scalar_metric(
        self, name: str, value: float, iteration: int, writer, wandb_writer
    ) -> None:
        """Write a single scalar metric to TensorBoard/W&B."""
        if writer is not None:
            writer.add_scalar(name, value, iteration)
        if wandb_writer is not None:
            wandb_writer.log({name: value}, iteration)

    def _write_per_layer_metrics(
        self, loss_scale: float, iteration: int, writer, wandb_writer, names: List[str]
    ) -> None:
        """Write per-layer metric values to TensorBoard/W&B."""
        for name in names:
            if name not in self._metrics:
                continue

            entry = self._metrics[name]
            values = entry.values.float() * loss_scale
            for i, val in enumerate(values.tolist()):
                # Skip zero values for sparse metrics (those with percentiles configured)
                if entry.percentiles is not None and val == 0:
                    continue
                self._write_scalar_metric(
                    f"moe/{name}_layer_{i}", val, iteration, writer, wandb_writer
                )


def _compute_load_balance_discrepancy(
    tokens_per_expert: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Compute Manhattan distance between token distribution and uniform distribution.

    Manhattan distance (L1) = Σ|actual_ratio - uniform_ratio|

    Range: [0, 2*(1 - 1/n)] where n = num_experts
        - 0: perfect uniform distribution
        - 2*(1 - 1/n): all tokens routed to one expert (e.g., 1.75 for 8 experts)
    """
    total_tokens = tokens_per_expert.sum()
    if total_tokens == 0:
        return torch.tensor(0.0, device=tokens_per_expert.device)

    actual_distribution = tokens_per_expert.float() / total_tokens
    uniform_distribution = torch.full_like(actual_distribution, 1.0 / num_experts)

    return torch.sum(torch.abs(actual_distribution - uniform_distribution))


# =============================================================================
# Public API functions - used by router.py, cuda_graphs.py, tests, etc.
# =============================================================================


def save_to_aux_losses_tracker(
    name: str,
    loss: torch.Tensor,
    layer_number: int,
    num_layers: int,
    reduce_group: torch.distributed.ProcessGroup = None,
    avg_group: torch.distributed.ProcessGroup = None,
    percentiles: Optional[List[float]] = None,
) -> None:
    """Save the auxiliary loss for logging."""
    MoEMetricsTracker.get_instance().record(
        name=name,
        value=loss,
        layer_number=layer_number,
        num_layers=num_layers,
        reduce_group=reduce_group,
        avg_group=avg_group,
        percentiles=percentiles,
    )


@internal_api
def save_load_balance_discrepancy(
    tokens_per_expert: torch.Tensor,
    num_experts: int,
    layer_number: int,
    num_layers: int,
    reduce_group: torch.distributed.ProcessGroup = None,
    percentiles: Optional[List[float]] = None,
) -> None:
    """Compute load balance discrepancy and save to tracker.

    Args:
        tokens_per_expert: Number of tokens routed to each expert.
        num_experts: Total number of experts.
        layer_number: 1-based layer index.
        num_layers: Total number of layers.
        reduce_group: Process group for all-reduce operations.
        percentiles: List of percentiles to compute (e.g., [0.5, 0.95] for p50, p95).
    """
    if layer_number is None:
        return

    # Default percentiles for load_balance_discrepancy
    if percentiles is None:
        percentiles = [0.5, 0.95]

    discrepancy = _compute_load_balance_discrepancy(tokens_per_expert, num_experts)
    save_to_aux_losses_tracker(
        "load_balance_discrepancy",
        discrepancy,
        layer_number,
        num_layers,
        reduce_group=reduce_group,
        percentiles=percentiles,
    )


def clear_aux_losses_tracker() -> None:
    """Clear the auxiliary losses."""
    MoEMetricsTracker.get_instance().clear()


def get_moe_layer_wise_logging_tracker() -> Dict[str, dict]:
    """Return the moe layer wise tracker in legacy dict format."""
    return MoEMetricsTracker.get_instance().get_raw_tracker()


# Deprecated: Use MoEMetricsTracker.get_instance().track() directly
def track_moe_metrics(
    loss_scale: float,
    iteration: int,
    writer,
    wandb_writer=None,
    total_loss_dict=None,  # Deprecated, ignored
    per_layer_logging=False,
    force_initialize: bool = False,
    track_names: Optional[Union[str, List[str]]] = None,
    num_layers: Optional[int] = None,
    moe_layer_freq: Optional[Union[int, List[int]]] = None,
    mtp_num_layers: Optional[int] = None,
) -> str:
    """Track the MoE metrics for logging.

    Deprecated: Use MoEMetricsTracker.get_instance().track() directly.
    """
    return MoEMetricsTracker.get_instance().track(
        loss_scale=loss_scale,
        iteration=iteration,
        writer=writer,
        wandb_writer=wandb_writer,
        per_layer_logging=per_layer_logging,
        force_initialize=force_initialize,
        names=track_names,
        num_layers=num_layers,
        moe_layer_freq=moe_layer_freq,
        mtp_num_layers=mtp_num_layers,
    )
