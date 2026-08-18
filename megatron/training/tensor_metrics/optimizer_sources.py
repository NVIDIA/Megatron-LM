# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optimizer-backed parameter and weight-gradient sources for training tensor metrics."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from megatron.core.optimizer import (
    ChainedOptimizer,
    DistributedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    LayerWiseDistributedOptimizer,
    MegatronOptimizer,
)
from megatron.core.parameter_names import CanonicalParameterNameMap
from megatron.core.process_groups_config import ProcessGroupCollection

from .core import (
    FlatShard,
    MetricSite,
    MetricTensor,
    Owned,
    RankRelation,
    Replica,
    Shard,
    TensorMetric,
)


@dataclass(frozen=True)
class _OptimizerTensorView:
    """Locally valid optimizer tensors anchored to an original model parameter."""

    model_parameter: torch.nn.Parameter
    parameter: torch.Tensor
    wgrad: torch.Tensor | None
    storage_relations: tuple[RankRelation, ...]
    is_placeholder: bool = False


@dataclass(frozen=True)
class _OptimizerParameterManifestEntry:
    """Rank-symmetric metadata for one logical optimizer parameter slot."""

    name: str
    logical_shape: tuple[int, ...]
    parameter_dtype: torch.dtype
    wgrad_dtype: torch.dtype
    ep_owner: int | None
    model_relations: tuple[RankRelation, ...]
    storage_relations: tuple[RankRelation, ...]

    @property
    def rank_relations(self) -> tuple[RankRelation, ...]:
        """Compose model and optimizer-storage placement metadata."""
        return self.model_relations + self.storage_relations


def _build_optimizer_parameter_manifest(
    parameter_names: CanonicalParameterNameMap,
    optimizer: MegatronOptimizer,
    pg_collection: ProcessGroupCollection,
) -> tuple[_OptimizerParameterManifestEntry, ...]:
    """Build one ordered parameter manifest shared across expert-parallel ranks.

    The payload extends the old canonical-name all-gather with the logical shape, observed dtypes,
    and complete placement metadata needed to materialize neutral remote-owner slots. Dense and
    shared parameters appear on every EP rank and are deduplicated after their metadata is checked.
    Separately named expert parameters contribute one entry from their owning EP rank.
    """
    local_entries = tuple(
        _optimizer_parameter_manifest_entry(parameter_names, view, pg_collection)
        for view in _optimizer_tensor_views(parameter_names, optimizer, pg_collection)
    )
    ep_group = getattr(pg_collection, "ep", None)
    if ep_group is None or ep_group.size() == 1:
        return tuple(sorted(local_entries, key=lambda entry: entry.name))
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "An expert-parallel optimizer manifest requires initialized distributed communication."
        )

    gathered_entries: list[tuple[_OptimizerParameterManifestEntry, ...] | None] = [
        None
    ] * ep_group.size()
    torch.distributed.all_gather_object(gathered_entries, local_entries, group=ep_group)

    local_entries_by_name = {entry.name: entry for entry in local_entries}
    templates_by_name = {}
    for ep_rank, rank_entries in enumerate(gathered_entries):
        if rank_entries is None:
            raise RuntimeError("An expert-parallel rank did not contribute parameter metadata.")
        for entry in rank_entries:
            if entry.ep_owner is not None and entry.ep_owner != ep_rank:
                raise ValueError(
                    f"Parameter {entry.name!r} was reported by EP rank {ep_rank}, "
                    f"but names owner {entry.ep_owner}."
                )
            previous = templates_by_name.setdefault(entry.name, entry)
            if (
                previous.logical_shape != entry.logical_shape
                or previous.ep_owner != entry.ep_owner
                or previous.model_relations != entry.model_relations
            ):
                raise ValueError(
                    f"Expert-parallel ranks disagree about parameter metadata for {entry.name!r}."
                )
    missing_replicas = [
        name
        for name, entry in templates_by_name.items()
        if entry.ep_owner is None and name not in local_entries_by_name
    ]
    if missing_replicas:
        raise ValueError(
            "Expert-parallel replicas are missing from the local parameter catalog: "
            f"{sorted(missing_replicas)}."
        )
    return tuple(
        local_entries_by_name.get(name, templates_by_name[name])
        for name in sorted(templates_by_name)
    )


def _optimizer_metric_tensors(
    parameter_names: CanonicalParameterNameMap,
    optimizer: MegatronOptimizer,
    pg_collection: ProcessGroupCollection,
    metrics: Sequence[TensorMetric],
    manifest: tuple[_OptimizerParameterManifestEntry, ...] | None = None,
) -> list[MetricTensor]:
    """Build rank-symmetric metric inputs from optimizer-backed tensor views.

    Parameter observations use the optimizer/master representation selected by the storage
    adapter. Wgrad observations use finalized ``main_grad`` or ``grad`` storage. The manifest
    supplies empty identity slots for expert parameters owned by other EP ranks. Site predicates
    are evaluated before metric tensors are constructed; the executor repeats the filtering for
    each individual metric.
    """
    local_views = {
        parameter_names[view.model_parameter]: view
        for view in _optimizer_tensor_views(parameter_names, optimizer, pg_collection)
    }
    if manifest is None:
        manifest = _build_optimizer_parameter_manifest(
            parameter_names, optimizer, pg_collection
        )
    if not local_views and manifest:
        raise ValueError("Optimizer parameter metadata has no local tensor device.")
    local_device = next(iter(local_views.values())).parameter.device if local_views else None

    values = []
    for entry in manifest:
        parameter_site = MetricSite(entry.name, "parameter")
        wgrad_site = MetricSite(entry.name, "wgrad")
        include_parameter = any(metric.accepts(parameter_site) for metric in metrics)
        include_wgrad = any(metric.accepts(wgrad_site) for metric in metrics)
        if not include_parameter and not include_wgrad:
            continue

        view = local_views.get(entry.name)
        if view is None:
            parameter = torch.empty(
                0, dtype=entry.parameter_dtype, device=local_device
            )
            wgrad = torch.empty(0, dtype=entry.wgrad_dtype, device=local_device)
        else:
            parameter = view.parameter
            wgrad = view.wgrad
            if wgrad is None:
                wgrad = torch.empty(
                    0, dtype=entry.wgrad_dtype, device=parameter.device
                )

        if include_parameter:
            values.append(
                MetricTensor(
                    tensor=parameter,
                    sites=(parameter_site,),
                    rank_relations=entry.rank_relations,
                    is_placeholder=view is None or view.is_placeholder,
                )
            )
        if include_wgrad:
            values.append(
                MetricTensor(
                    tensor=wgrad,
                    sites=(wgrad_site,),
                    rank_relations=entry.rank_relations,
                    is_placeholder=view is None or view.is_placeholder,
                )
            )
    return values


def _optimizer_parameter_manifest_entry(
    parameter_names: CanonicalParameterNameMap,
    view: _OptimizerTensorView,
    pg_collection: ProcessGroupCollection,
) -> _OptimizerParameterManifestEntry:
    model_parameter = view.model_parameter
    wgrad_dtype = view.wgrad.dtype if view.wgrad is not None else view.parameter.dtype
    model_relations = _model_parallel_relations(model_parameter, pg_collection)
    ep_relation = next(
        (relation for relation in model_relations if relation.axis == "ep"), None
    )
    ep_owner = (
        ep_relation.placement.rank
        if ep_relation is not None and isinstance(ep_relation.placement, Owned)
        else None
    )
    return _OptimizerParameterManifestEntry(
        name=parameter_names[model_parameter],
        logical_shape=_logical_local_shape(model_parameter, pg_collection),
        parameter_dtype=view.parameter.dtype,
        wgrad_dtype=wgrad_dtype,
        ep_owner=ep_owner,
        model_relations=model_relations,
        storage_relations=view.storage_relations,
    )


def _model_parallel_relations(
    model_parameter: torch.nn.Parameter, pg_collection: ProcessGroupCollection
) -> tuple[RankRelation, ...]:
    """Describe model-parallel placement for an original model parameter."""
    is_expert = not getattr(model_parameter, "allreduce", True)
    tensor_axis = "expert_tp" if is_expert else "tp"
    tensor_group_attribute = "expt_tp" if is_expert else "tp"
    tensor_group = getattr(pg_collection, tensor_group_attribute, None)
    if tensor_group is None:
        raise ValueError(
            f"Tensor metric observation requires the {tensor_axis} process group."
        )
    if getattr(model_parameter, "tensor_model_parallel", False):
        tensor_placement = Shard(getattr(model_parameter, "partition_dim", None))
    else:
        tensor_placement = Replica()
    relations = [RankRelation(tensor_axis, tensor_placement)]

    ep_group = getattr(pg_collection, "ep", None)
    if ep_group is not None:
        ep_placement = (
            Owned(ep_group.rank()) if is_expert and ep_group.size() > 1 else Replica()
        )
        relations.append(RankRelation("ep", ep_placement))

    gtp_axis = "expert_gtp" if is_expert else "gtp"
    gtp_group_attribute = "expt_gtp_remat" if is_expert else "gtp_remat"
    gtp_group = getattr(pg_collection, gtp_group_attribute, None)
    is_gtp_shard = getattr(model_parameter, "is_gtp_weight_remat", False)
    if is_gtp_shard and gtp_group is None:
        raise ValueError(
            f"Tensor metric observation requires the {gtp_group_attribute} process group for "
            "a GTP-rematerialized parameter."
        )
    if gtp_group is not None:
        gtp_placement = Shard(0) if is_gtp_shard else Replica()
        relations.append(RankRelation(gtp_axis, gtp_placement))
    return tuple(relations)


def _local_optimizer_tensor_views(
    optimizer: MegatronOptimizer,
    pg_collection: ProcessGroupCollection,
) -> list[_OptimizerTensorView]:
    """Return locally valid parameter and wgrad views for a Megatron optimizer.

    The adapters deliberately dispatch on Megatron storage wrappers. They do not execute
    communication or prepare optimizer gradients; returned wgrads are the finalized values
    available at the observer's pre-step hook.
    """
    if isinstance(optimizer, LayerWiseDistributedOptimizer):
        return [
            _trim_gtp_padding(view, pg_collection)
            for view in _layerwise_optimizer_tensor_views(optimizer, pg_collection)
        ]
    if isinstance(optimizer, DistributedOptimizer):
        return [
            _trim_gtp_padding(view, pg_collection)
            for view in _distributed_optimizer_tensor_views(optimizer)
        ]
    if isinstance(optimizer, ChainedOptimizer):
        views = []
        seen_parameters = set()
        for child in optimizer.chained_optimizers:
            for view in _local_optimizer_tensor_views(child, pg_collection):
                parameter_id = id(view.model_parameter)
                if parameter_id in seen_parameters:
                    continue
                seen_parameters.add(parameter_id)
                views.append(view)
        return views
    if isinstance(optimizer, (Float16OptimizerWithFloat16Params, FP32Optimizer)):
        return [
            _trim_gtp_padding(view, pg_collection)
            for view in _replicated_optimizer_tensor_views(optimizer)
        ]
    raise NotImplementedError(
        f"Tensor metric observation does not support optimizer type {type(optimizer).__name__}."
    )


def _optimizer_tensor_views(
    parameter_names: CanonicalParameterNameMap,
    optimizer: MegatronOptimizer,
    pg_collection: ProcessGroupCollection,
) -> list[_OptimizerTensorView]:
    """Expand sparse local optimizer views into deterministic logical parameter slots."""
    local_views = {}
    for view in _local_optimizer_tensor_views(optimizer, pg_collection):
        if view.model_parameter in local_views:
            raise ValueError("An optimizer produced multiple tensor views for one model parameter.")
        local_views[view.model_parameter] = view

    layerwise_owner_maps = _layerwise_owner_maps_by_optimizer(optimizer)
    materialized_views = []
    for model_parameter, _ in sorted(parameter_names.items(), key=lambda item: item[1]):
        if not model_parameter.requires_grad:
            continue
        view = local_views.pop(model_parameter, None)
        if view is None:
            empty = model_parameter.detach().new_empty(0)
            storage_relations = _missing_optimizer_storage_relations(
                optimizer, model_parameter, pg_collection, layerwise_owner_maps
            )
            view = _OptimizerTensorView(
                model_parameter=model_parameter,
                parameter=empty,
                wgrad=empty,
                storage_relations=storage_relations,
                is_placeholder=_has_remote_owner(storage_relations, pg_collection),
            )
        materialized_views.append(view)

    if local_views:
        raise ValueError("An optimizer tensor view has no canonical model parameter name.")
    return materialized_views


def _replicated_optimizer_tensor_views(
    optimizer: Float16OptimizerWithFloat16Params | FP32Optimizer,
) -> list[_OptimizerTensorView]:
    views = _unplaced_optimizer_tensor_views(optimizer)
    return [
        _with_storage_relation(view, _data_parallel_axis(view.model_parameter), Replica())
        for view in views
    ]


def _unplaced_optimizer_tensor_views(
    optimizer: Float16OptimizerWithFloat16Params | FP32Optimizer,
) -> list[_OptimizerTensorView]:
    if getattr(optimizer, "is_stub_optimizer", False):
        return []
    if isinstance(optimizer, Float16OptimizerWithFloat16Params):
        return _float16_optimizer_tensor_views(optimizer)
    if isinstance(optimizer, FP32Optimizer):
        views = []
        for param_group in optimizer.optimizer.param_groups:
            for model_parameter in param_group["params"]:
                views.append(
                    _OptimizerTensorView(
                        model_parameter=model_parameter,
                        parameter=model_parameter.detach(),
                        wgrad=_model_parameter_wgrad(model_parameter),
                        storage_relations=(),
                    )
                )
        return views
    raise NotImplementedError(
        "LayerWise tensor metric observation supports only FP32 and Float16 child optimizers; "
        f"got {type(optimizer).__name__}."
    )


def _float16_optimizer_tensor_views(
    optimizer: Float16OptimizerWithFloat16Params,
) -> list[_OptimizerTensorView]:
    if len(optimizer.float16_groups) != len(optimizer.fp32_from_float16_groups):
        raise ValueError("Float16 optimizer model and main parameter group counts must align.")
    views = []
    for model_group, main_group in zip(
        optimizer.float16_groups, optimizer.fp32_from_float16_groups
    ):
        if len(model_group) != len(main_group):
            raise ValueError("Float16 optimizer model and main parameter groups must align.")
        for model_parameter, main_parameter in zip(model_group, main_group):
            views.append(
                _OptimizerTensorView(
                    model_parameter=model_parameter,
                    parameter=main_parameter.detach(),
                    wgrad=_model_parameter_wgrad(model_parameter),
                    storage_relations=(),
                )
            )
    for model_group in optimizer.fp32_from_fp32_groups:
        for model_parameter in model_group:
            views.append(
                _OptimizerTensorView(
                    model_parameter=model_parameter,
                    parameter=model_parameter.detach(),
                    wgrad=_model_parameter_wgrad(model_parameter),
                    storage_relations=(),
                )
            )
    return views


def _distributed_optimizer_tensor_views(
    optimizer: DistributedOptimizer,
) -> list[_OptimizerTensorView]:
    if getattr(optimizer, "is_stub_optimizer", False):
        return []

    if not (
        len(optimizer.model_float16_groups)
        == len(optimizer.shard_fp32_from_float16_groups)
        == len(optimizer.shard_float16_groups)
    ):
        raise ValueError("Distributed optimizer model and parameter shard group counts must align.")
    if len(optimizer.model_fp32_groups) != len(optimizer.shard_fp32_groups):
        raise ValueError("Distributed optimizer FP32 model and shard group counts must align.")

    views = []
    for model_group, main_group, model_shard_group in zip(
        optimizer.model_float16_groups,
        optimizer.shard_fp32_from_float16_groups,
        optimizer.shard_float16_groups,
    ):
        if not (len(model_group) == len(main_group) == len(model_shard_group)):
            raise ValueError("Distributed optimizer model and parameter shard groups must align.")
        for model_parameter, main_parameter, model_shard in zip(
            model_group, main_group, model_shard_group
        ):
            parameter = main_parameter if main_parameter is not None else model_shard
            if parameter is None:
                raise NotImplementedError(
                    "Tensor metric observation requires an addressable local parameter shard."
                )
            views.append(_distributed_optimizer_tensor_view(optimizer, model_parameter, parameter))

    for model_group, model_shard_group in zip(
        optimizer.model_fp32_groups, optimizer.shard_fp32_groups
    ):
        if len(model_group) != len(model_shard_group):
            raise ValueError("Distributed optimizer FP32 model and shard groups must align.")
        for model_parameter, model_shard in zip(model_group, model_shard_group):
            views.append(
                _distributed_optimizer_tensor_view(optimizer, model_parameter, model_shard)
            )
    return views


def _distributed_optimizer_tensor_view(
    optimizer: DistributedOptimizer,
    model_parameter: torch.nn.Parameter,
    parameter: torch.Tensor,
) -> _OptimizerTensorView:
    param_range = optimizer._get_model_param_range_map(model_parameter)["param"]
    model_wgrad = _model_parameter_wgrad(model_parameter)
    local_wgrad = (
        None
        if model_wgrad is None
        else model_wgrad.detach().view(-1)[param_range.start : param_range.end]
    )
    local_numel = param_range.end - param_range.start
    if parameter.numel() != local_numel:
        raise ValueError("Distributed optimizer parameter shard length does not match its range.")
    if local_wgrad is not None and local_wgrad.numel() != local_numel:
        raise ValueError("Distributed optimizer wgrad shard length does not match its range.")
    return _OptimizerTensorView(
        model_parameter=model_parameter,
        parameter=parameter.detach(),
        wgrad=local_wgrad,
        storage_relations=(
            RankRelation(
                _data_parallel_axis(model_parameter),
                FlatShard(
                    logical_shape=tuple(model_parameter.shape),
                    start=param_range.start,
                    end=param_range.end,
                ),
            ),
        ),
    )


def _layerwise_optimizer_tensor_views(
    optimizer: LayerWiseDistributedOptimizer,
    pg_collection: ProcessGroupCollection,
) -> list[_OptimizerTensorView]:
    owner_maps = _layerwise_parameter_owner_maps(optimizer)
    views = []
    for child in optimizer.chained_optimizers:
        for view in _unplaced_optimizer_tensor_views(child):
            relation = _layerwise_storage_relation(
                view.model_parameter, pg_collection, owner_maps
            )
            group = _data_parallel_group(pg_collection, relation.axis)
            placement = relation.placement
            if isinstance(placement, Owned) and placement.rank != group.rank():
                raise ValueError(
                    "A LayerWise optimizer exposed a parameter on a rank other than its owner."
                )
            views.append(_with_storage_relation(view, relation.axis, placement))
    return views


def _missing_optimizer_storage_relations(
    optimizer: MegatronOptimizer,
    model_parameter: torch.nn.Parameter,
    pg_collection: ProcessGroupCollection,
    layerwise_owner_maps: dict[int, dict[str, dict[int, int]]],
) -> tuple[RankRelation, ...]:
    if isinstance(optimizer, LayerWiseDistributedOptimizer):
        return (
            _layerwise_storage_relation(
                model_parameter,
                pg_collection,
                layerwise_owner_maps[id(optimizer)],
            ),
        )
    if isinstance(optimizer, DistributedOptimizer):
        return (_empty_flat_shard_relation(model_parameter, pg_collection),)
    if isinstance(optimizer, ChainedOptimizer):
        layerwise_children = [
            child
            for child in optimizer.chained_optimizers
            if isinstance(child, LayerWiseDistributedOptimizer)
        ]
        for child in layerwise_children:
            owner_maps = layerwise_owner_maps[id(child)]
            if _layerwise_owner_rank(model_parameter, owner_maps) is not None:
                return (
                    _layerwise_storage_relation(
                        model_parameter, pg_collection, owner_maps
                    ),
                )
        if layerwise_children and getattr(
            model_parameter, "is_managed_by_layer_wise_optimizer", False
        ):
            return (
                _layerwise_storage_relation(
                    model_parameter,
                    pg_collection,
                    layerwise_owner_maps[id(layerwise_children[0])],
                ),
            )
        if any(
            isinstance(child, DistributedOptimizer) for child in optimizer.chained_optimizers
        ):
            return (_empty_flat_shard_relation(model_parameter, pg_collection),)
        if len(layerwise_children) == 1:
            return (
                _layerwise_storage_relation(
                    model_parameter,
                    pg_collection,
                    layerwise_owner_maps[id(layerwise_children[0])],
                ),
            )
    raise ValueError(
        f"Optimizer type {type(optimizer).__name__} does not describe storage for a missing "
        "model parameter."
    )


def _empty_flat_shard_relation(
    model_parameter: torch.nn.Parameter, pg_collection: ProcessGroupCollection
) -> RankRelation:
    return RankRelation(
        _data_parallel_axis(model_parameter),
        FlatShard(
            logical_shape=_logical_local_shape(model_parameter, pg_collection), start=0, end=0
        ),
    )


def _trim_gtp_padding(
    view: _OptimizerTensorView, pg_collection: ProcessGroupCollection
) -> _OptimizerTensorView:
    """Remove this GTP rank's physical padding from optimizer-backed tensors."""
    model_parameter = view.model_parameter
    physical_shape = tuple(model_parameter.shape)
    logical_shape = _logical_local_shape(model_parameter, pg_collection)
    if logical_shape == physical_shape:
        return view

    logical_numel = 1
    for size in logical_shape:
        logical_numel *= size
    flat_shard_indices = [
        index
        for index, relation in enumerate(view.storage_relations)
        if isinstance(relation.placement, FlatShard)
    ]
    if len(flat_shard_indices) > 1:
        raise ValueError("Tensor metric observation supports at most one optimizer FlatShard.")

    if flat_shard_indices:
        relation_index = flat_shard_indices[0]
        relation = view.storage_relations[relation_index]
        placement = relation.placement
        if not isinstance(placement, FlatShard):
            raise TypeError("A selected optimizer storage relation must contain a FlatShard.")
        clipped_start = min(placement.start, logical_numel)
        clipped_end = min(placement.end, logical_numel)
        local_numel = clipped_end - clipped_start
        parameter = view.parameter.reshape(-1)[:local_numel]
        wgrad = None if view.wgrad is None else view.wgrad.reshape(-1)[:local_numel]
        storage_relations = list(view.storage_relations)
        storage_relations[relation_index] = RankRelation(
            relation.axis, FlatShard(logical_shape, clipped_start, clipped_end)
        )
    else:
        physical_numel = model_parameter.numel()
        if view.parameter.numel() != physical_numel:
            raise ValueError(
                "A padded GTP optimizer parameter must match its physical model shard size."
            )
        parameter = view.parameter.reshape(physical_shape)[: logical_shape[0]]
        if view.wgrad is not None:
            if view.wgrad.numel() != physical_numel:
                raise ValueError(
                    "A padded GTP wgrad must match its physical model shard size."
                )
            wgrad = view.wgrad.reshape(physical_shape)[: logical_shape[0]]
        else:
            wgrad = None
        storage_relations = list(view.storage_relations)

    return _OptimizerTensorView(
        model_parameter=model_parameter,
        parameter=parameter,
        wgrad=wgrad,
        storage_relations=tuple(storage_relations),
        is_placeholder=view.is_placeholder,
    )


def _logical_local_shape(
    model_parameter: torch.nn.Parameter, pg_collection: ProcessGroupCollection
) -> tuple[int, ...]:
    """Return this rank's unpadded logical shape for a dim-0 GTP shard."""
    physical_shape = tuple(model_parameter.shape)
    if not getattr(model_parameter, "is_gtp_weight_remat", False):
        return physical_shape
    pad_length = getattr(model_parameter, "pad_length", 0)
    if not pad_length:
        return physical_shape
    if not physical_shape:
        raise ValueError("A padded GTP-rematerialized parameter must not be scalar.")

    is_expert = not getattr(model_parameter, "allreduce", True)
    group_attribute = "expt_gtp_remat" if is_expert else "gtp_remat"
    group = getattr(pg_collection, group_attribute, None)
    if group is None:
        raise ValueError(
            f"Tensor metric observation requires the {group_attribute} process group for "
            "a GTP-rematerialized parameter."
        )
    padded_rows = physical_shape[0] * group.size()
    logical_rows = padded_rows - pad_length
    if logical_rows < 0:
        raise ValueError("GTP padding exceeds the globally padded parameter dimension.")
    local_start = group.rank() * physical_shape[0]
    local_rows = max(0, min(physical_shape[0], logical_rows - local_start))
    return (local_rows,) + physical_shape[1:]


def _layerwise_storage_relation(
    model_parameter: torch.nn.Parameter,
    pg_collection: ProcessGroupCollection,
    owner_maps: dict[str, dict[int, int]],
) -> RankRelation:
    axis = _data_parallel_axis(model_parameter)
    group = _data_parallel_group(pg_collection, axis)
    if group.size() == 1:
        return RankRelation(axis, Replica())
    owner_rank = _layerwise_owner_rank(model_parameter, owner_maps)
    if owner_rank is None:
        raise ValueError("A LayerWise optimizer parameter has no owner in its data-parallel layout.")
    return RankRelation(axis, Owned(owner_rank))


def _layerwise_owner_rank(
    model_parameter: torch.nn.Parameter,
    owner_maps: dict[str, dict[int, int]],
) -> int | None:
    return owner_maps[_data_parallel_axis(model_parameter)].get(id(model_parameter))


def _layerwise_owner_maps_by_optimizer(
    optimizer: MegatronOptimizer,
) -> dict[int, dict[str, dict[int, int]]]:
    if isinstance(optimizer, LayerWiseDistributedOptimizer):
        layerwise_optimizers = [optimizer]
    elif isinstance(optimizer, ChainedOptimizer):
        layerwise_optimizers = [
            child
            for child in optimizer.chained_optimizers
            if isinstance(child, LayerWiseDistributedOptimizer)
        ]
    else:
        layerwise_optimizers = []
    return {
        id(layerwise_optimizer): _layerwise_parameter_owner_maps(layerwise_optimizer)
        for layerwise_optimizer in layerwise_optimizers
    }


def _layerwise_parameter_owner_maps(
    optimizer: LayerWiseDistributedOptimizer,
) -> dict[str, dict[int, int]]:
    return {
        "dp": _parameter_owner_map(getattr(optimizer, "dp_cp_params_list", None)),
        "expert_dp": _parameter_owner_map(
            getattr(optimizer, "expt_dp_params_list", None)
        ),
    }


def _data_parallel_group(
    pg_collection: ProcessGroupCollection, axis: str
) -> torch.distributed.ProcessGroup:
    attribute = "expt_dp" if axis == "expert_dp" else "dp_cp"
    group = getattr(pg_collection, attribute, None)
    if group is None:
        raise ValueError(f"LayerWise tensor metric observation requires the {axis} group.")
    return group


def _parameter_owner_map(
    parameters_by_rank: Sequence[Sequence[torch.nn.Parameter]] | None,
) -> dict[int, int]:
    owners = {}
    if parameters_by_rank is None:
        return owners
    for owner_rank, parameters in enumerate(parameters_by_rank):
        for parameter in parameters:
            parameter_id = id(parameter)
            if parameter_id in owners:
                raise ValueError("A LayerWise optimizer parameter is assigned to multiple ranks.")
            owners[parameter_id] = owner_rank
    return owners


def _with_storage_relation(
    view: _OptimizerTensorView,
    axis: str,
    placement: Replica | Shard | Owned,
) -> _OptimizerTensorView:
    return _OptimizerTensorView(
        model_parameter=view.model_parameter,
        parameter=view.parameter,
        wgrad=view.wgrad,
        storage_relations=view.storage_relations + (RankRelation(axis, placement),),
        is_placeholder=view.is_placeholder,
    )


def _has_remote_owner(
    relations: Sequence[RankRelation], pg_collection: ProcessGroupCollection
) -> bool:
    """Return whether any owned storage axis names a different local group rank."""
    return any(
        isinstance(relation.placement, Owned)
        and relation.placement.rank
        != _data_parallel_group(pg_collection, relation.axis).rank()
        for relation in relations
    )


def _model_parameter_wgrad(model_parameter: torch.nn.Parameter) -> torch.Tensor | None:
    main_grad = getattr(model_parameter, "main_grad", None)
    wgrad = main_grad if main_grad is not None else model_parameter.grad
    return wgrad.detach() if wgrad is not None else None


def _data_parallel_axis(model_parameter: torch.nn.Parameter) -> str:
    return "dp" if getattr(model_parameter, "allreduce", True) else "expert_dp"
