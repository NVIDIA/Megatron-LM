# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoE metrics tracking and logging.

Collects per-layer MoE metrics during forward passes, synchronizes them across
distributed ranks, and writes scalar summaries to TensorBoard / W&B.

Usage:
    tracker = MoEMetricsTracker.get_instance()

    # In router forward pass:
    tracker.record("load_balancing_loss", loss, layer_number=1, num_layers=32,
                   reduce_group=tp_cp_group)

    # At end of training step:
    log_str = tracker.report(
        loss_scale=1 / num_microbatches,
        iteration=step,
        writer=tb_writer,
        num_layers=32,
    )
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Union

import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection


@dataclass
class MetricEntry:
    """Per-layer metric with distributed reduction configuration."""

    values: torch.Tensor
    reduce_group: Optional[torch.distributed.ProcessGroup] = None
    avg_group: Optional[torch.distributed.ProcessGroup] = None
    needs_dp_avg: bool = True


class MoEMetricsTracker:
    """Singleton tracker for MoE layer-wise metrics.

    Lifecycle: ``record()`` per-layer values during forward → ``report()`` at
    step end (sync, aggregate, log, clear) → repeat.

    Example:
        tracker = MoEMetricsTracker.get_instance()
        tracker.record("load_balancing_loss", loss, layer_number=1, num_layers=32)
        log_str = tracker.report(loss_scale=1/8, iteration=100, writer=tb_writer,
                                 num_layers=32)
    """

    _instance: ClassVar[Optional['MoEMetricsTracker']] = None

    def __init__(self):
        self._metrics: Dict[str, MetricEntry] = {}

    @classmethod
    def get_instance(cls) -> 'MoEMetricsTracker':
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # =========================================================================
    # Public API
    # =========================================================================

    @property
    def metrics(self) -> Dict[str, MetricEntry]:
        """Read-only access to the underlying metric entries."""
        return self._metrics

    def record(
        self,
        name: str,
        value: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: Optional[torch.distributed.ProcessGroup] = None,
        avg_group: Optional[torch.distributed.ProcessGroup] = None,
        needs_dp_avg: bool = True,
    ) -> None:
        """Accumulate a metric value for a specific layer.

        Called during the router forward pass.  Lazily creates the metric entry
        on first call for each metric name.

        Args:
            name: Metric name (e.g. ``"load_balancing_loss"``).
            value: Scalar tensor to accumulate (will be detached).
            layer_number: 1-based layer index.
            num_layers: Total number of layers (determines tensor size).
            reduce_group: Process group for sum-reduction (e.g. tp_cp_group).
            avg_group: Process group for average-reduction.
            needs_dp_avg: Whether to average across DP ranks after other reductions.
        """
        if layer_number is None:
            return

        if name not in self._metrics:
            self._metrics[name] = MetricEntry(values=torch.zeros(num_layers, device=value.device))

        entry = self._metrics[name]
        entry.values[layer_number - 1] += value.detach()
        entry.reduce_group = reduce_group
        entry.avg_group = avg_group
        entry.needs_dp_avg = needs_dp_avg

    def report(
        self,
        loss_scale: float,
        iteration: int,
        writer=None,
        wandb_writer=None,
        per_layer_logging: bool = False,
        force_initialize: bool = False,
        names: Optional[Union[str, List[str]]] = None,
        num_layers: Optional[int] = None,
        moe_layer_freq: Optional[Union[int, List[int]]] = None,
        mtp_num_layers: Optional[int] = None,
        total_loss_dict: Optional[dict[str, torch.Tensor]] = None,
        percentiles: Optional[Dict[str, List[float]]] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> str:
        """Sync metrics across ranks, aggregate, log, and clear.

        This is the main entry point called once per training step.  It pairs
        with :meth:`record`: you *record* individual data points during forward,
        then *report* the summary at step end.

        Args:
            loss_scale: Scale factor for averaging across microbatches
                (usually ``1 / num_microbatches``).
            iteration: Current training iteration.
            writer: TensorBoard ``SummaryWriter`` (optional).
            wandb_writer: Weights & Biases run object (optional).
            per_layer_logging: Whether to also write per-layer values.
            force_initialize: If True, pre-create metric entries for *names*
                that don't exist yet.  Required for PP ranks without MoE layers
                whose tensor sizes must match ranks that do have MoE layers.
            names: Metric name(s) to report.  ``None`` reports all.
            num_layers: Total transformer layers (required when *force_initialize*).
            moe_layer_freq: MoE layer frequency or binary pattern list.
            mtp_num_layers: Extra layers from Multi-Token Prediction.
            total_loss_dict: Megatron training-loop accumulator.  Metrics whose
                names end with ``"loss"`` are accumulated here and excluded from
                the returned console log string.
            percentiles: Per-metric percentiles to compute, e.g.
                ``{"load_imbalance": [0.5, 0.95]}``.
            pg_collection: Custom process-group collection for reduction.

        Returns:
            Formatted log string for console output.
        """
        names_list = self._resolve_names(names)

        # Pre-create entries on PP ranks that lack MoE layers.
        # Tensor size must be (num_layers + mtp_num_layers) to match ranks that
        # recorded via record(), otherwise all_reduce across PP will hang.
        if force_initialize and names_list is not None:
            if num_layers is None:
                raise ValueError("num_layers must be provided when force_initialize=True.")
            init_size = num_layers + (mtp_num_layers or 0)
            for name in names_list:
                self.ensure_initialized(name, init_size)

        self._sync_metrics(names_list, pg_collection)

        num_moe_layers = self._count_moe_layers(num_layers, moe_layer_freq, mtp_num_layers)
        scalars = self._aggregate(loss_scale, num_moe_layers, names_list, percentiles)

        # Megatron integration: accumulate loss metrics into total_loss_dict
        console_scalars = dict(scalars)
        if total_loss_dict is not None:
            for k, v in scalars.items():
                if k.lower().endswith("loss"):
                    if k in total_loss_dict:
                        total_loss_dict[k] += v
                    else:
                        total_loss_dict[k] = v
                    console_scalars.pop(k)

        self._log_scalars(scalars, iteration, writer, wandb_writer)
        if per_layer_logging:
            self._log_per_layer(
                loss_scale, names_list, iteration, writer, wandb_writer, percentiles
            )

        log_string = self._format(console_scalars)
        self.clear()
        return log_string

    def clear(self) -> None:
        """Zero out all metric values (entries are kept for reuse)."""
        for entry in self._metrics.values():
            entry.values.zero_()

    def ensure_initialized(
        self, name: str, num_layers: int, device: Optional[Union[str, torch.device, int]] = None
    ) -> None:
        """Pre-create a metric entry if it does not already exist.

        This is needed for PP ranks that have no MoE layers -- their tensor
        size must match ranks that do, otherwise ``all_reduce`` across PP hangs.

        Args:
            name: Metric name.
            num_layers: Tensor size (should include MTP layers).
            device: Device for the zero tensor.  Defaults to current CUDA device.
        """
        if name not in self._metrics:
            if device is None:
                device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            self._metrics[name] = MetricEntry(values=torch.zeros(num_layers, device=device))

    # =========================================================================
    # Private implementation
    # =========================================================================

    def _resolve_names(self, names: Optional[Union[str, List[str]]]) -> List[str]:
        """Normalize *names* argument to a list of strings."""
        if names is None:
            return list(self._metrics.keys())
        if isinstance(names, str):
            return [names]
        return names

    def _sync_metrics(
        self, names: List[str], pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        """All-reduce metrics across distributed ranks.

        Reduction order: PP collect → reduce_group sum → avg_group avg → DP avg.
        """
        if pg_collection is None:
            pp_group = parallel_state.get_pipeline_model_parallel_group()
            dp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=False, partial_data_parallel=False
            )
        else:
            pp_group = pg_collection.pp
            dp_group = pg_collection.dp

        for name in names:
            if name not in self._metrics:
                continue

            entry = self._metrics[name]
            v = entry.values

            torch.distributed.all_reduce(v, group=pp_group)

            if entry.reduce_group is not None:
                torch.distributed.all_reduce(v, group=entry.reduce_group)

            if entry.avg_group is not None:
                torch.distributed.all_reduce(
                    v, group=entry.avg_group, op=torch.distributed.ReduceOp.AVG
                )

            if entry.needs_dp_avg:
                torch.distributed.all_reduce(v, group=dp_group, op=torch.distributed.ReduceOp.AVG)

    @staticmethod
    def _count_moe_layers(
        num_layers: Optional[int],
        moe_layer_freq: Optional[Union[int, List[int]]],
        mtp_num_layers: Optional[int],
    ) -> int:
        """Compute the effective number of MoE layers from configuration."""
        if moe_layer_freq is None:
            n = num_layers
        elif isinstance(moe_layer_freq, int):
            assert isinstance(num_layers, int)
            n = sum(1 for i in range(num_layers) if i % moe_layer_freq == 0)
        elif isinstance(moe_layer_freq, list):
            n = sum(moe_layer_freq)
        else:
            raise ValueError(f"Invalid moe_layer_freq: {moe_layer_freq}")

        if mtp_num_layers is not None:
            n += mtp_num_layers

        return n

    def _aggregate(
        self,
        loss_scale: float,
        num_moe_layers: int,
        names: List[str],
        percentiles: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """Aggregate per-layer values into scalar summaries.

        Always computes the mean across MoE layers.  If *percentiles* specifies
        quantiles for a metric, those are computed over non-zero layer values and
        added as ``"{name}_p{pct}"`` keys.
        """
        result: Dict[str, Union[float, torch.Tensor]] = {}

        for name in names:
            if name not in self._metrics:
                continue

            values = self._metrics[name].values.float() * loss_scale

            if percentiles and name in percentiles:
                nonzero = values[values > 0]
                if nonzero.numel() > 0:
                    pcts = percentiles[name]
                    pct_vals = torch.quantile(
                        nonzero, torch.tensor(pcts, device=nonzero.device)
                    ).tolist()
                    for pct, pct_val in zip(pcts, pct_vals):
                        result[f"{name}_p{int(pct * 100)}"] = pct_val

            result[name] = values.sum() / num_moe_layers

        return result

    def _log_scalars(
        self, scalars: Dict[str, Union[float, torch.Tensor]], iteration: int, writer, wandb_writer
    ) -> None:
        """Write scalar metrics to TensorBoard and/or W&B."""
        for name, value in scalars.items():
            if writer is not None:
                writer.add_scalar(name, value, iteration)
            if wandb_writer is not None:
                wandb_writer.log({name: value}, iteration)

    def _log_per_layer(
        self,
        loss_scale: float,
        names: List[str],
        iteration: int,
        writer,
        wandb_writer,
        percentiles: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """Write per-layer metric values to TensorBoard and/or W&B."""
        for name in names:
            if name not in self._metrics:
                continue

            values = self._metrics[name].values.float() * loss_scale
            is_sparse = percentiles is not None and name in percentiles
            for i, val in enumerate(values.tolist()):
                if is_sparse and val == 0:
                    continue
                if writer is not None:
                    writer.add_scalar(f"moe/{name}_layer_{i}", val, iteration)
                if wandb_writer is not None:
                    wandb_writer.log({f"moe/{name}_layer_{i}": val}, iteration)

    @staticmethod
    def _format(scalars: Dict[str, Union[float, torch.Tensor]]) -> str:
        """Format aggregated metrics as a console log string."""
        return "".join(f" {k}: {v:.2f} |" for k, v in scalars.items())
