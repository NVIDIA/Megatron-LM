# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSA and CSA indexer-loss metric tracking helpers."""

from typing import Optional, Tuple, Union

import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.core.utils import get_pg_size


def _get_process_group_ranks(group) -> Optional[Tuple[int, ...]]:
    """Return stable rank provenance for a process group when distributed is initialized."""
    if not torch.distributed.is_initialized():
        return None
    try:
        if group is None:
            return tuple(range(torch.distributed.get_world_size()))
        return tuple(torch.distributed.get_process_group_ranks(group))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None


def _tracker_pp_group_matches(tracker, prefix: str, pp_group) -> bool:
    """Return whether a cached group is the same object or spans the same ordered ranks."""
    cached_group = tracker.get(f"{prefix}_pp_group")
    if cached_group is pp_group:
        return True
    cached_ranks = tracker.get(f"{prefix}_pp_ranks")
    return cached_ranks is not None and cached_ranks == _get_process_group_ranks(pp_group)


def _record_tracker_pp_group(tracker, prefix: str, pp_group) -> None:
    """Record both immediate and reinitialization-stable process-group provenance."""
    tracker[f"{prefix}_pp_group"] = pp_group
    ranks = _get_process_group_ranks(pp_group)
    if ranks is None:
        tracker.pop(f"{prefix}_pp_ranks", None)
    else:
        tracker[f"{prefix}_pp_ranks"] = ranks


class DSAIndexerLossLoggingHelper:
    """Helper class for logging sparse attention indexer losses."""

    tracker = {}

    @staticmethod
    def ensure_tracker_size(num_layers: int) -> None:
        """Ensure the tracker can hold every indexed layer.

        Standard training initializes a fixed-capacity tracker after model construction. Once
        that capacity is agreed across pipeline ranks, the tensor must never be rebound because
        a CUDA Graph may retain its address. Lazy allocation remains available for standalone
        modules and other legacy callers that do not use the training setup path.
        """
        tracker = DSAIndexerLossLoggingHelper.tracker
        agreed_size = tracker.get("agreed_size")
        if agreed_size is not None:
            if num_layers > agreed_size:
                raise RuntimeError(
                    "DSA metric tracker fixed capacity is too small for layer index "
                    f"{num_layers} (capacity: {agreed_size})."
                )
            values = tracker.get("values")
            if agreed_size > 0 and (values is None or values.shape[0] != agreed_size):
                raise RuntimeError("DSA metric tracker storage does not match its fixed capacity.")
            if agreed_size == 0 and values is not None:
                raise RuntimeError("Zero-capacity DSA metric tracker unexpectedly owns storage.")
            return
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        elif tracker["values"].shape[0] < num_layers:
            grown = torch.zeros(
                num_layers, device=tracker["values"].device, dtype=tracker["values"].dtype
            )
            grown[: tracker["values"].shape[0]] = tracker["values"]
            tracker["values"] = grown

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the indexer loss for logging.

        Args:
            loss: The loss tensor.
            layer_number: Layer index of the loss, 1-indexed.
            num_layers: The number of total layers.
            reduce_group: The group for reducing the loss.
            avg_group: The group for averaging the loss.
        """
        # Skip indexer loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        # Tracker must be at least max(num_layers, layer_number) so hybrid MTP layers
        # (whose layer_number can exceed config.num_layers + config.mtp_num_layers when
        # each MTP depth contains multiple hybrid layers) don't index out of bounds.
        # Standard training has already fixed this capacity. Standalone modules that bypass
        # lifecycle initialization retain lazy growth for backward compatibility.
        needed = max(num_layers, layer_number)
        DSAIndexerLossLoggingHelper.ensure_tracker_size(needed)
        tracker["values"][layer_number - 1] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker(preserve_groups: bool = False):
        """Clear indexer losses, optionally retaining reduction-group metadata."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        reduce_group = tracker.get("reduce_group") if preserve_groups else None
        avg_group = tracker.get("avg_group") if preserve_groups else None
        if "values" in tracker:
            tracker["values"].zero_()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def reduce_loss_in_tracker(
        num_layers: Optional[int] = None,
        dynamic_cp_parent_group: Optional[torch.distributed.ProcessGroup] = None,
        configured_cp_size: Optional[int] = None,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        """Collect and reduce the indexer losses across ranks.

        Cross-PP `all_reduce` must be invoked on every rank in the pipeline-parallel group,
        otherwise ranks without any indexer layer would skip the collective and cause a hang.
        Pass `num_layers` to lazily initialize the tracker on such ranks so they participate
        with a zero-filled tensor.

        Args:
            num_layers: Total number of decoder layers; required to lazily initialize the
                tracker on ranks where no indexer layer ran.
            dynamic_cp_parent_group: Stable physical DP x CP group for Dynamic-CP metrics.
                Supplying it on every PP rank makes stages without indexer layers use the
                same reduction domain after the PP reduction.
            configured_cp_size: Configured/static CP width. Dynamic CP accumulates raw local
                token sums, so multiplying by this value before the physical DP x CP average
                preserves the nominal global-batch weighting of static CP-SUM then DP-AVG.
            pp_group: Pipeline-parallel group that owns the language model. Defaults to the
                legacy global process-group registry when not supplied.
            dp_group: Data-parallel group that owns the language model. Used by static CP
                logging and to validate the Dynamic-CP parent domain; defaults to the legacy
                global process-group registry when omitted.
        """
        tracker = DSAIndexerLossLoggingHelper.tracker
        if pp_group is None:
            # Legacy callers may omit the collection. Training passes the owning language-model
            # group explicitly so multi-module ranks never consult unrelated global groups.
            pp_group = parallel_state.get_pipeline_model_parallel_group()
        if dynamic_cp_parent_group is not None and (
            configured_cp_size is None or configured_cp_size < 1
        ):
            raise ValueError("configured_cp_size must be positive for Dynamic CP logging.")
        if dynamic_cp_parent_group is not None and torch.distributed.is_initialized():
            parent_size = get_pg_size(dynamic_cp_parent_group)
            if parent_size % configured_cp_size != 0:
                raise ValueError(
                    "Dynamic CP metric parent group size must be divisible by configured_cp_size."
                )
            if dp_group is not None:
                expected_parent_size = get_pg_size(dp_group) * configured_cp_size
                if parent_size != expected_parent_size:
                    raise ValueError(
                        "Dynamic CP metric parent group must span the language-model DP x CP "
                        f"domain (expected size {expected_parent_size}, got {parent_size})."
                    )

        # Standard training has already agreed on a fixed capacity after model construction.
        # Keep lazy PP negotiation as a compatibility path for standalone modules that bypass
        # setup_model_and_optimizer().
        agreed_size = tracker.get("agreed_size")
        if agreed_size is not None:
            if not _tracker_pp_group_matches(tracker, "agreed_size", pp_group):
                raise RuntimeError(
                    "DSA metric tracker cached size belongs to a different PP group."
                )
            size = agreed_size
            if size == 0 and "values" in tracker:
                raise RuntimeError("Zero-capacity DSA metric tracker unexpectedly owns storage.")
        else:
            local_size = tracker["values"].shape[0] if "values" in tracker else (num_layers or 0)
            size_t = torch.tensor(
                [local_size], device=torch.cuda.current_device(), dtype=torch.long
            )
            torch.distributed.all_reduce(size_t, op=torch.distributed.ReduceOp.MAX, group=pp_group)
            size = int(size_t.item())
            if size > 0:
                if "values" not in tracker:
                    tracker["values"] = torch.zeros(size, device=torch.cuda.current_device())
                elif tracker["values"].shape[0] < size:
                    grown = torch.zeros(
                        size, device=tracker["values"].device, dtype=tracker["values"].dtype
                    )
                    grown[: tracker["values"].shape[0]] = tracker["values"]
                    tracker["values"] = grown
                tracker["agreed_size"] = size
                _record_tracker_pp_group(tracker, "agreed_size", pp_group)
        if size == 0:
            return
        if agreed_size is not None:
            values = tracker.get("values")
            if values is None or values.shape[0] != size:
                raise RuntimeError("DSA metric tracker storage does not match its fixed capacity.")
        values = tracker["values"]

        if dynamic_cp_parent_group is not None:
            torch.distributed.all_reduce(values, group=pp_group)
            values.mul_(configured_cp_size)
            torch.distributed.all_reduce(
                values, group=dynamic_cp_parent_group, op=torch.distributed.ReduceOp.AVG
            )
        else:
            # Apply static-CP normalization on the PP stage that owns each indexer before
            # broadcasting its layer slots across PP. Empty PP stages have no local group
            # metadata and can safely skip CP because their values are still zero here.
            if tracker.get('reduce_group') is not None:
                torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
            if tracker.get('avg_group') is not None:
                torch.distributed.all_reduce(
                    values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
                )
            torch.distributed.all_reduce(values, group=pp_group)
            if dp_group is None:
                # Legacy fallback; explicit process-group collections are preferred.
                dp_group = parallel_state.get_data_parallel_group(with_context_parallel=False)
            torch.distributed.all_reduce(values, group=dp_group, op=torch.distributed.ReduceOp.AVG)

    @staticmethod
    def track_indexer_metrics(
        loss_scale: float,
        iteration: int,
        writer,
        wandb_writer=None,
        total_loss_dict=None,
        per_layer_logging: bool = False,
        num_layers: Optional[int] = None,
        num_indexer_layers: Optional[int] = None,
        preserve_groups: bool = False,
        dynamic_cp_parent_group: Optional[torch.distributed.ProcessGroup] = None,
        configured_cp_size: Optional[int] = None,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        """Track the sparse attention indexer metrics for logging.

        Args:
            loss_scale: Scale factor for the loss.
            iteration: Current training iteration.
            writer: TensorBoard writer.
            wandb_writer: Weights & Biases writer.
            total_loss_dict: Dictionary to accumulate total losses.
            per_layer_logging: Whether to log per-layer losses.
            num_layers: Total decoder layer count used to initialize empty PP ranks.
            num_indexer_layers: Number of indexer executions contributing to the
                metric. Repeated MTP executions may share one tracker slot. Defaults
                to the tracker size when every tracked layer executes once.
            preserve_groups: Keep reduction groups after logging for CUDA Graph runs.
            dynamic_cp_parent_group: Stable physical DP x CP group used by every PP rank
                when Dynamic CP is enabled.
            configured_cp_size: Configured/static CP width used to normalize Dynamic-CP
                raw local token sums.
            pp_group: Pipeline-parallel group that owns the language model.
            dp_group: Data-parallel group that owns the language model.
        """
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(
            num_layers=num_layers,
            dynamic_cp_parent_group=dynamic_cp_parent_group,
            configured_cp_size=configured_cp_size,
            pp_group=pp_group,
            dp_group=dp_group,
        )
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return

        indexer_loss_values = tracker["values"] * loss_scale
        if num_indexer_layers is None:
            num_indexer_layers = indexer_loss_values.shape[0]

        # Average across actual indexer executions. Layers without an indexer contribute
        # zero, while repeated MTP accumulates multiple executions in one tracker slot.
        avg_indexer_loss = indexer_loss_values.sum() / max(num_indexer_layers, 1)

        # Log average loss
        if total_loss_dict is not None:
            if "indexer loss" in total_loss_dict:
                total_loss_dict["indexer loss"] += avg_indexer_loss
            else:
                total_loss_dict["indexer loss"] = avg_indexer_loss

        if writer is not None:
            writer.add_scalar("indexer loss", avg_indexer_loss, iteration)

        if wandb_writer is not None:
            wandb_writer.log({"indexer loss": avg_indexer_loss}, iteration)

        DSAIndexerLossLoggingHelper.clean_loss_in_tracker(preserve_groups=preserve_groups)


def resolve_dsa_metric_pg_collection(
    pg_collection: Optional[Union[ProcessGroupCollection, MultiModuleProcessGroupCollection]],
    *,
    schedule_pg_collection: Optional[
        Union[ProcessGroupCollection, MultiModuleProcessGroupCollection]
    ] = None,
) -> Tuple[bool, Optional[ProcessGroupCollection]]:
    """Resolve the language-model process groups used for DSA metric collection.

    Args:
        pg_collection: Process groups associated with the model passed by the caller.
            Full-iteration CUDA Graph callers may pass a multi-module collection here.
        schedule_pg_collection: Optional schedule-wide process groups. A multi-module
            collection here is authoritative for deciding whether this rank owns the
            language model.

    Returns:
        A pair ``(should_track, metric_pg_collection)``. ``should_track`` is false only
        when a multi-module collection explicitly says this rank does not own a language
        model. ``metric_pg_collection`` is the owning language model's collection, or the
        model collection for a single-module caller. In particular, ``(True, None)`` means
        that a legacy caller should use the global process-group registry, while
        ``(False, None)`` means this rank must skip DSA metric collection.
    """
    metric_pg_collection = (
        schedule_pg_collection
        if isinstance(schedule_pg_collection, MultiModuleProcessGroupCollection)
        else pg_collection
    )
    if isinstance(metric_pg_collection, MultiModuleProcessGroupCollection):
        if not metric_pg_collection.has_language_model():
            return False, None
        return True, metric_pg_collection.get_language_model_collection()
    return True, pg_collection


def get_dsa_metric_tracker_size(model) -> int:
    """Return the local tracker size required by active DSA/CSA writers in this model."""
    if isinstance(model, (list, tuple)):
        required_size = 0
        model_chunks = model
    else:
        required_size = 0
        model_chunks = (model,)
    for model_chunk in model_chunks:
        for module in model_chunk.modules():
            if not getattr(module, "logs_dsa_indexer_loss", False):
                continue
            config = getattr(module, "config", None)
            if (
                config is not None
                and hasattr(config, "dsa_indexer_loss_coeff")
                and (config.dsa_indexer_loss_coeff or 0.0) <= 0
            ):
                continue
            layer_number = getattr(module, "layer_number", None)
            if layer_number is not None:
                required_size = max(required_size, layer_number)
            if config is not None:
                required_size = max(required_size, config.num_layers + (config.mtp_num_layers or 0))
    return required_size


def initialize_dsa_metric_tracker(
    model,
    pg_collection: Optional[
        Union[ProcessGroupCollection, MultiModuleProcessGroupCollection]
    ] = None,
) -> int:
    """Initialize fixed DSA tracker storage for one model lifecycle.

    Args:
        model: Local model chunk or chunks whose DSA/CSA writers may run.
        pg_collection: Model- or schedule-level process groups. Multi-module collections
            select their language-model collection; legacy callers may pass ``None``.

    Returns:
        The PP-agreed fixed capacity, or zero when this rank has no language model or the
        language-model PP group contains no DSA/CSA metric writers. Zero is also finalized,
        so a later metric write reports an invalid model scan instead of renegotiating.
    """
    should_track, metric_pg_collection = resolve_dsa_metric_pg_collection(pg_collection)
    if not should_track:
        return 0

    pp_group = metric_pg_collection.pp if metric_pg_collection is not None else None
    tracker = DSAIndexerLossLoggingHelper.tracker
    local_required_size = get_dsa_metric_tracker_size(model)
    if (
        torch.distributed.is_initialized()
        and pp_group is None
        and parallel_state.model_parallel_is_initialized()
    ):
        # Legacy callers may omit the collection. Internal capture paths pass the owning
        # language-model process-group collection explicitly.
        pp_group = parallel_state.get_pipeline_model_parallel_group()

    agreed_size = tracker.get("agreed_size")
    if agreed_size is not None:
        if not _tracker_pp_group_matches(tracker, "agreed_size", pp_group):
            raise RuntimeError("DSA metric tracker was initialized with a different PP group.")
        if local_required_size > agreed_size:
            raise RuntimeError(
                "DSA metric tracker fixed capacity is smaller than the model's required "
                f"layer index (capacity: {agreed_size}, required: {local_required_size})."
            )
        values = tracker.get("values")
        if agreed_size > 0 and (values is None or values.shape[0] != agreed_size):
            raise RuntimeError("DSA metric tracker storage does not match its fixed capacity.")
        if agreed_size == 0 and values is not None:
            raise RuntimeError("Zero-capacity DSA metric tracker unexpectedly owns storage.")
        return agreed_size

    if torch.distributed.is_initialized():
        size_tensor = torch.tensor(
            [local_required_size], device=torch.cuda.current_device(), dtype=torch.long
        )
        torch.distributed.all_reduce(size_tensor, op=torch.distributed.ReduceOp.MAX, group=pp_group)
        local_required_size = int(size_tensor.item())

    size = local_required_size
    if "values" in tracker:
        raise RuntimeError("DSA metric tracker must be initialized before its first metric write.")
    if size > 0:
        tracker["values"] = torch.zeros(size, device=torch.cuda.current_device())

    tracker["agreed_size"] = size
    _record_tracker_pp_group(tracker, "agreed_size", pp_group)
    return size
