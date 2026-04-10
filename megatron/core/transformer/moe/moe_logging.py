# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoE metrics tracking and logging.

Collects per-layer MoE metrics during forward passes, synchronizes them across
distributed ranks, and writes scalar summaries to TensorBoard / W&B.

Usage:
    tracker = get_moe_metrics_tracker()

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
from typing import Dict, List, Optional, Tuple, Union

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


# ---------------------------------------------------------------------------
# Module-level global tracker (follows parallel_state / global_vars pattern)
# ---------------------------------------------------------------------------
_MOE_METRICS_TRACKER: Optional['MoEMetricsTracker'] = None


def get_moe_metrics_tracker() -> 'MoEMetricsTracker':
    """Return the global MoE metrics tracker, creating it lazily if needed."""
    global _MOE_METRICS_TRACKER
    if _MOE_METRICS_TRACKER is None:
        _MOE_METRICS_TRACKER = MoEMetricsTracker()
    return _MOE_METRICS_TRACKER


def set_moe_metrics_tracker(tracker: 'MoEMetricsTracker') -> None:
    """Set the global MoE metrics tracker."""
    global _MOE_METRICS_TRACKER
    _MOE_METRICS_TRACKER = tracker


def destroy_moe_metrics_tracker() -> None:
    """Reset the global MoE metrics tracker to ``None``."""
    global _MOE_METRICS_TRACKER
    _MOE_METRICS_TRACKER = None


# ---------------------------------------------------------------------------
# MoE Overload Factor Tracker (same pattern as MoEMetricsTracker)
# ---------------------------------------------------------------------------
_MOE_OVERLOAD_FACTOR_TRACKER: Optional['MoEOverloadFactorTracker'] = None


def get_moe_overload_factor_tracker() -> 'MoEOverloadFactorTracker':
    """Return the global MoE overload factor tracker, creating it lazily if needed."""
    global _MOE_OVERLOAD_FACTOR_TRACKER
    if _MOE_OVERLOAD_FACTOR_TRACKER is None:
        _MOE_OVERLOAD_FACTOR_TRACKER = MoEOverloadFactorTracker()
    return _MOE_OVERLOAD_FACTOR_TRACKER


def set_moe_overload_factor_tracker(tracker: 'MoEOverloadFactorTracker') -> None:
    """Set the global MoE overload factor tracker."""
    global _MOE_OVERLOAD_FACTOR_TRACKER
    _MOE_OVERLOAD_FACTOR_TRACKER = tracker


def destroy_moe_overload_factor_tracker() -> None:
    """Reset the global MoE overload factor tracker to ``None``."""
    global _MOE_OVERLOAD_FACTOR_TRACKER
    _MOE_OVERLOAD_FACTOR_TRACKER = None


class MoEOverloadFactorTracker:
    """Track MoE overload-factor metrics.

    Recorded values
    ---------------
    - Per-layer **actual tokens on this rank** after dispatch:
      ``tokens_per_expert.sum()`` (0-dim tensor per microbatch entry), in
      ``_layer_fwd_tokens`` keyed by layer index; balanced counts in
      ``_layer_fwd_balanced``.
    - Per-layer **balanced token count** before overload is computed:
      ``num_local_tokens * moe_router_topk``, with ``num_local_tokens`` taken from
      ``MoELayer`` forward ``hidden_states`` (product of leading dimensions, same row
      count as ``view(-1, hidden_size)``). Both are attached via an autograd hook on
      ``dispatched_input`` (``RecordDispatchTokenCountsFunction`` in ``moe_utils``).
    - ``_cumulative_tokens_timeline`` and ``_cumulative_balanced_timeline`` store a
      time-ordered sequence of ``+fwd`` / ``-bwd`` events so cumulative peaks of
      actual vs balanced token counts can be compared for ``max_cum`` style metrics.

    How ``report()`` aggregates
    -----------------------------
    1. Over ``tp_ep_group``, ``all_reduce(MAX)`` on per-rank actual token totals per
       entry, then divide by balanced tokens per rank (summed local balanced counts /
       group size) to get **tp_ep overload** per entry.
    2. Over ``dp_group``, ``all_reduce(AVG)`` and ``all_reduce(MAX)`` on that overload
       for scalar summaries.
    3. Over the **pipeline-parallel** group, ``max`` and ``max_cum`` use
       ``all_reduce(MAX)`` so every stage agrees on the worst overload; ranks without
       MoE layers contribute ``0``.
    4. The **mean** overload scalar is not reduced across PP; each rank logs its local
       mean (``0`` if nothing was recorded).

    Lifecycle
    ---------
    ``set_process_groups()`` runs in ``MoELayer.__init__`` when ``log_overload_factor``
    is enabled. ``record_fwd()`` / ``record_bwd()`` run during forward (hooked from
    ``MoELayer``). ``report()`` runs at step end (sync, aggregate, log, then requests
    deferred clear).

    ``clear()`` behavior
    --------------------
    ``clear()`` does not immediately reset storage. It marks storage for reset on the
    next ``record_fwd()`` or ``record_bwd()`` so tensor handles stay valid until Python
    executes a recording hook again (for example across CUDA graph replay windows that
    skip those hooks).
    """

    def __init__(self) -> None:
        self._layer_fwd_tokens: Dict[int, List[torch.Tensor]] = {}
        # layer_idx -> list of 0-dim float (tokens on rank)
        self._layer_fwd_balanced: Dict[int, List[torch.Tensor]] = {}
        # same keys as _layer_fwd_tokens, balanced token count per entry
        self._cumulative_tokens_timeline: List[torch.Tensor] = []
        # +actual tokens on forward, - on backward (mirrors balanced timeline).
        self._cumulative_balanced_timeline: List[torch.Tensor] = []
        self._tp_ep_group: Optional[torch.distributed.ProcessGroup] = None
        self._dp_group: Optional[torch.distributed.ProcessGroup] = None
        self._pending_clear: bool = False

    def set_process_groups(
        self,
        tp_ep_group: Optional[torch.distributed.ProcessGroup] = None,
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        """Set process groups for reduction (``MoELayer.__init__`` when ``log_overload_factor``)."""
        if tp_ep_group is not None:
            self._tp_ep_group = tp_ep_group
        if dp_group is not None:
            self._dp_group = dp_group

    def _clear_storage(self) -> None:
        self._layer_fwd_tokens.clear()
        self._layer_fwd_balanced.clear()
        self._cumulative_tokens_timeline.clear()
        self._cumulative_balanced_timeline.clear()

    def _flush_pending_clear(self) -> None:
        if self._pending_clear:
            self._pending_clear = False
            self._clear_storage()

    def record_fwd(
        self,
        layer_number: Optional[int],
        tokens_on_rank: torch.Tensor,
        local_balanced_token_count: torch.Tensor,
    ) -> None:
        """Record forward token total on this rank (0-dim float) and balanced count scalar."""
        self._flush_pending_clear()
        if layer_number is None:
            return
        layer_idx = layer_number - 1
        if layer_idx not in self._layer_fwd_tokens:
            self._layer_fwd_tokens[layer_idx] = []
            self._layer_fwd_balanced[layer_idx] = []
        self._layer_fwd_tokens[layer_idx].append(tokens_on_rank.detach())
        self._layer_fwd_balanced[layer_idx].append(local_balanced_token_count.detach())
        self._cumulative_tokens_timeline.append(tokens_on_rank.detach())
        self._cumulative_balanced_timeline.append(local_balanced_token_count.detach())

    def record_bwd(
        self, tokens_on_rank: torch.Tensor, local_balanced_token_count: torch.Tensor
    ) -> None:
        """Record backward-pass (negated actual and balanced count) for paired cumsums."""
        self._flush_pending_clear()
        self._cumulative_tokens_timeline.append(-tokens_on_rank.detach())
        self._cumulative_balanced_timeline.append(-local_balanced_token_count.detach())

    def _pipeline_group_and_use_reduce(
        self,
    ) -> Tuple[Optional[torch.distributed.ProcessGroup], bool]:
        pp_group = (
            parallel_state.get_pipeline_model_parallel_group(check_initialized=False)
            if torch.distributed.is_initialized()
            else None
        )
        use_pp_reduce = (
            pp_group is not None and torch.distributed.get_world_size(group=pp_group) > 1
        )
        return pp_group, use_pp_reduce

    def _flatten_recorded_tokens(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        fwd_tensors: List[torch.Tensor] = []
        balanced_tensors: List[torch.Tensor] = []
        if self._layer_fwd_tokens:
            for layer_idx in sorted(self._layer_fwd_tokens.keys()):
                for t, b in zip(
                    self._layer_fwd_tokens[layer_idx], self._layer_fwd_balanced[layer_idx]
                ):
                    fwd_tensors.append(t)
                    balanced_tensors.append(b)
        return fwd_tensors, balanced_tensors

    def _pp_allreduce_empty_tracker(self, pp_group: torch.distributed.ProcessGroup) -> None:
        """Ranks without MoE still join PP collectives so peers do not hang."""
        device = (
            torch.device('cuda', torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device('cpu')
        )
        pp_buf = torch.zeros(2, device=device, dtype=torch.float32)
        torch.distributed.all_reduce(pp_buf, group=pp_group, op=torch.distributed.ReduceOp.MAX)

    def _validate_overload_tensor_lists(
        self, num_entries: int, num_layers: int, num_balanced: int
    ) -> None:
        if num_entries % num_layers != 0:
            raise ValueError(
                f"Overload factor tracker: num_entries ({num_entries}) must be "
                f"divisible by num_layers ({num_layers})."
            )
        if num_balanced != num_entries:
            raise ValueError(
                f"Overload factor tracker: balanced_tensors length ({num_balanced}) "
                f"must match fwd_tensors ({num_entries})."
            )

    def _max_cum_overload_if_timeline(
        self,
        tp_ep_group: Optional[torch.distributed.ProcessGroup],
        dp_group: Optional[torch.distributed.ProcessGroup],
    ) -> Optional[float]:
        """Cumulative actual vs balanced token count; ratio of peaks across ranks."""
        if not self._cumulative_tokens_timeline:
            return None
        if len(self._cumulative_balanced_timeline) != len(self._cumulative_tokens_timeline):
            raise ValueError(
                f"Overload tracker: _cumulative_tokens_timeline "
                f"({len(self._cumulative_tokens_timeline)}) and "
                f"_cumulative_balanced_timeline "
                f"({len(self._cumulative_balanced_timeline)}) length mismatch."
            )
        fwd_bwd_stacked = torch.stack(
            [t.float() for t in self._cumulative_tokens_timeline], dim=0
        )  # [num_events]
        balanced_fwd_bwd_stacked = torch.stack(
            [t.float() for t in self._cumulative_balanced_timeline], dim=0
        )
        cum_actual = fwd_bwd_stacked.cumsum(dim=0)
        cum_balanced = balanced_fwd_bwd_stacked.cumsum(dim=0)
        local_actual_peak = cum_actual.max()
        local_balanced_peak = cum_balanced.max()
        cum_overload_ratio = torch.where(
            local_balanced_peak > 0,
            local_actual_peak / (local_balanced_peak + 1e-8),
            local_actual_peak.new_zeros(()),
        ).unsqueeze(0)
        if tp_ep_group is not None:
            torch.distributed.all_reduce(
                cum_overload_ratio, group=tp_ep_group, op=torch.distributed.ReduceOp.MAX
            )
        if dp_group is not None:
            torch.distributed.all_reduce(
                cum_overload_ratio, group=dp_group, op=torch.distributed.ReduceOp.MAX
            )
        return cum_overload_ratio.item()

    def _tp_ep_overload_from_lists(
        self,
        fwd_tensors: List[torch.Tensor],
        balanced_tensors: List[torch.Tensor],
        tp_ep_group: Optional[torch.distributed.ProcessGroup],
    ) -> Tuple[torch.Tensor, torch.device]:
        """Max actual per entry over tp_ep, balanced sum per entry, then overload ratio."""
        actual_tokens_stacked = torch.stack([t.float() for t in fwd_tensors], dim=0)
        device = actual_tokens_stacked.device
        if tp_ep_group is not None:
            tp_ep_world = float(tp_ep_group.size())
            max_actual = actual_tokens_stacked.clone()
            torch.distributed.all_reduce(
                max_actual, group=tp_ep_group, op=torch.distributed.ReduceOp.MAX
            )
        else:
            tp_ep_world = 1.0
            max_actual = actual_tokens_stacked

        balanced_stacked = torch.stack(
            [b.to(device=device, dtype=torch.float32) for b in balanced_tensors], dim=0
        )
        if tp_ep_group is not None:
            torch.distributed.all_reduce(balanced_stacked, group=tp_ep_group)
        balanced_per_rank = balanced_stacked / tp_ep_world
        tp_ep_overload = max_actual / (balanced_per_rank + 1e-8)
        return tp_ep_overload, device

    def _dp_reduce_overload(
        self, tp_ep_overload: torch.Tensor, dp_group: Optional[torch.distributed.ProcessGroup]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Average and worst-case overload across DP replicas (per entry)."""
        if dp_group is not None:
            overload_avg = tp_ep_overload.clone()
            torch.distributed.all_reduce(
                overload_avg, group=dp_group, op=torch.distributed.ReduceOp.AVG
            )
            overload_max = tp_ep_overload.clone()
            torch.distributed.all_reduce(
                overload_max, group=dp_group, op=torch.distributed.ReduceOp.MAX
            )
        else:
            overload_avg = tp_ep_overload
            overload_max = tp_ep_overload
        return overload_avg, overload_max

    def _pp_reduce_max_overload_scalars(
        self,
        max_overload_factor: float,
        max_cum_overload_factor: Optional[float],
        device: torch.device,
        pp_group: torch.distributed.ProcessGroup,
    ) -> Tuple[float, float]:
        max_cum_value = (
            float(max_cum_overload_factor) if max_cum_overload_factor is not None else 0.0
        )
        pp_buf = torch.tensor(
            [max_overload_factor, max_cum_value], device=device, dtype=torch.float32
        )
        torch.distributed.all_reduce(pp_buf, group=pp_group, op=torch.distributed.ReduceOp.MAX)
        return pp_buf[0].item(), pp_buf[1].item()

    def _log_overload_metrics(
        self,
        iteration: int,
        writer,
        wandb_writer,
        avg_overload_factor: float,
        max_overload_factor: float,
        max_cum_overload_factor: Optional[float],
        per_layer_logging: bool,
        overload_avg: torch.Tensor,
        overload_max: torch.Tensor,
        num_layers: int,
        num_entries: int,
    ) -> None:
        if writer is not None:
            writer.add_scalar("moe/avg_overload_factor", avg_overload_factor, iteration)
            writer.add_scalar("moe/max_overload_factor", max_overload_factor, iteration)
            if max_cum_overload_factor is not None:
                writer.add_scalar("moe/max_cum_overload_factor", max_cum_overload_factor, iteration)
        if wandb_writer is not None:
            wandb_writer.log({"moe/avg_overload_factor": avg_overload_factor}, iteration)
            wandb_writer.log({"moe/max_overload_factor": max_overload_factor}, iteration)
            if max_cum_overload_factor is not None:
                wandb_writer.log(
                    {"moe/max_cum_overload_factor": max_cum_overload_factor}, iteration
                )

        if per_layer_logging:
            entries_per_layer = num_entries // num_layers
            layer_avg = overload_avg.view(num_layers, entries_per_layer).mean(dim=1)
            layer_max = overload_max.view(num_layers, entries_per_layer).max(dim=1).values
            for i in range(num_layers):
                avg_val, max_val = layer_avg[i].item(), layer_max[i].item()
                if writer is not None:
                    writer.add_scalar(f"moe/avg_overload_factor_layer_{i}", avg_val, iteration)
                    writer.add_scalar(f"moe/max_overload_factor_layer_{i}", max_val, iteration)
                if wandb_writer is not None:
                    wandb_writer.log(
                        {
                            f"moe/avg_overload_factor_layer_{i}": avg_val,
                            f"moe/max_overload_factor_layer_{i}": max_val,
                        },
                        iteration,
                    )

    def report(
        self, iteration: int, writer=None, wandb_writer=None, per_layer_logging: bool = False
    ) -> str:
        """Reduce data, overload factors, log to TB/W&B, defer clear, return log string."""
        pp_group, use_pp_reduce = self._pipeline_group_and_use_reduce()
        tp_ep_group = self._tp_ep_group
        dp_group = self._dp_group

        fwd_tensors, balanced_tensors = self._flatten_recorded_tokens()

        if not fwd_tensors:
            if use_pp_reduce:
                assert pp_group is not None
                self._pp_allreduce_empty_tracker(pp_group)
            self.clear()
            return ""

        num_entries = len(fwd_tensors)
        num_layers = len(self._layer_fwd_tokens)
        self._validate_overload_tensor_lists(num_entries, num_layers, len(balanced_tensors))

        max_cum_overload_factor = self._max_cum_overload_if_timeline(tp_ep_group, dp_group)
        tp_ep_overload, device = self._tp_ep_overload_from_lists(
            fwd_tensors, balanced_tensors, tp_ep_group
        )
        overload_avg, overload_max = self._dp_reduce_overload(tp_ep_overload, dp_group)

        avg_overload_factor = overload_avg.mean().item()
        max_overload_factor = overload_max.max().item()

        if use_pp_reduce:
            assert pp_group is not None
            max_overload_factor, max_cum_reduced = self._pp_reduce_max_overload_scalars(
                max_overload_factor, max_cum_overload_factor, device, pp_group
            )
            max_cum_overload_factor = max_cum_reduced

        self._log_overload_metrics(
            iteration,
            writer,
            wandb_writer,
            avg_overload_factor,
            max_overload_factor,
            max_cum_overload_factor,
            per_layer_logging,
            overload_avg,
            overload_max,
            num_layers,
            num_entries,
        )

        self.clear()

        parts = [
            f" avg overload factor: {avg_overload_factor:.3f} |",
            f" max overload factor: {max_overload_factor:.3f} |",
        ]
        if max_cum_overload_factor is not None:
            parts.append(f" max cum overload factor: {max_cum_overload_factor:.3f} |")
        return "".join(parts)

    def clear(self) -> None:
        """Mark stored tensors for reset on the next :meth:`record_fwd` or :meth:`record_bwd`.

        Does not drop list contents yet, so captured tensor references stay valid
        until the next recording hook runs. Process groups are kept.
        """
        self._pending_clear = True


class MoEMetricsTracker:
    """Tracker for MoE layer-wise metrics.

    Lifecycle: ``record()`` per-layer values during forward → ``report()`` at
    step end (sync, aggregate, log, clear) → repeat.

    Example:
        tracker = get_moe_metrics_tracker()
        tracker.record("load_balancing_loss", loss, layer_number=1, num_layers=32)
        log_str = tracker.report(loss_scale=1/8, iteration=100, writer=tb_writer,
                                 num_layers=32)
    """

    def __init__(self):
        self._metrics: Dict[str, MetricEntry] = {}

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
        track_names: Optional[Union[str, List[str]]] = None,
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
            force_initialize: If True, pre-create metric entries for *track_names*
                that don't exist yet.  Required for PP ranks without MoE layers
                whose tensor sizes must match ranks that do have MoE layers.
            track_names: Metric name(s) to report.  ``None`` reports all.
            num_layers: Total transformer layers (required when *force_initialize*).
            moe_layer_freq: MoE layer frequency or binary pattern list.
            mtp_num_layers: Extra layers from Multi-Token Prediction.
            total_loss_dict: Megatron training-loop accumulator.  Metrics
                ending with ``"loss"`` are accumulated here and excluded from
                the returned console log string.
            percentiles: Per-metric percentiles to compute, e.g.
                ``{"load_imbalance": [0.5, 0.95]}``.
            pg_collection: Custom process-group collection for reduction.

        Returns:
            Formatted log string for console output.
        """
        metric_names = self._resolve_names(track_names)

        # Pre-create entries on PP ranks that lack MoE layers.
        # Tensor size must be (num_layers + mtp_num_layers) to match ranks that
        # recorded via record(), otherwise all_reduce across PP will hang.
        if force_initialize:
            if num_layers is None:
                raise ValueError("num_layers must be provided when force_initialize=True.")
            init_size = num_layers + (mtp_num_layers or 0)
            for name in metric_names:
                self.ensure_initialized(name, init_size)

        self._sync_metrics(metric_names, pg_collection)

        num_moe_layers = self._count_moe_layers(num_layers, moe_layer_freq, mtp_num_layers)
        scalars = self._aggregate(loss_scale, num_moe_layers, metric_names, percentiles)

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
                loss_scale, metric_names, iteration, writer, wandb_writer, percentiles
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

    def _resolve_names(self, track_names: Optional[Union[str, List[str]]]) -> List[str]:
        """Normalize *track_names* argument to a list of strings."""
        if track_names is None:
            return list(self._metrics.keys())
        if isinstance(track_names, str):
            return [track_names]
        return track_names

    def _sync_metrics(
        self, metric_names: List[str], pg_collection: Optional[ProcessGroupCollection] = None
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

        for name in metric_names:
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
        metric_names: List[str],
        percentiles: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """Aggregate per-layer values into scalar summaries.

        Always computes the mean across MoE layers.  If *percentiles* specifies
        quantiles for a metric, those are computed over non-zero layer values and
        added as ``"{name}_p{pct}"`` keys.
        """
        result: Dict[str, Union[float, torch.Tensor]] = {}

        for name in metric_names:
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
        metric_names: List[str],
        iteration: int,
        writer,
        wandb_writer,
        percentiles: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """Write per-layer metric values to TensorBoard and/or W&B."""
        for name in metric_names:
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
