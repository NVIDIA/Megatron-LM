# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Dataclasses used by the planner
# -----------------------------------------------------------------------------


@dataclass
class TransferOp:
    """Single logical send/recv operation used in a reshard plan."""

    param_name: str
    peer_rank: int  # Who to send to / receive from
    is_send: bool  # True=send, False=recv

    # Slice information (for when we execute the plan)
    my_slice: tuple[slice, ...]  # My tensor slice
    peer_slice: tuple[slice, ...]  # Peer's tensor slice (for reference)

    # Optional global task identifier for advanced backends (e.g., NVSHMEM)
    # When present, this ID is shared between the matching send/recv ops
    # across ranks and can be used to build richer communication schedules.
    task_id: int | None = None


@dataclass
class ParameterMetadata:
    """Metadata for a parameter (used when param is on different rank)."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    element_size: int

    # TP sharding info
    is_tp: bool = False
    partition_dim: int = 0
    partition_stride: int = 1

    # EP sharding info (fused/grouped MoE)
    is_ep: bool = False
    num_experts: Optional[int] = None

    # Which rank owns this param
    owner_rank: int = -1

    tensor_parallel_group_ranks: list[int] | None = None
    expert_parallel_group_ranks: list[int] | None = None
    data_parallel_group_ranks: list[int] | None = None
    pipeline_parallel_group_ranks: list[int] | None = None

    # Canonical name for matching parameters across models with different EP/PP configurations.
    #
    # - EP (expert parallel): each rank owns a subset of experts with local indices
    #   (e.g., rank 1 has "weight0" locally, but it's actually global expert 4). The raw param
    #   name can't be used to match across source/destination because the same local name refers
    #   to different global experts on different ranks. `resolved_name` remaps local expert indices
    #   to global indices (e.g., "layer.experts.weight0" on rank 1 → "layer.experts.weight4").
    #
    # - PP (pipeline parallel): transformer blocks are often named with rank-local indices
    #   (e.g., PP stage 1 may have "decoder.layers.0" even though that corresponds to global
    #   layer 16). For reshard/refit across different PP partitionings (e.g., PP2 ↔ PP1),
    #   `resolved_name` may be further canonicalized to global layer indices.
    #
    # For non-EP and non-PP cases, resolved_name == name.
    resolved_name: Optional[str] = None
    # The global expert index this parameter belongs to (e.g., 4 for global expert 4).
    # Computed alongside resolved_name; None for non-EP or fused expert tensors.
    global_expert_index: Optional[int] = None


@dataclass
class ShardingDescriptor:
    """Descriptor for a sharded dimension for a parameter."""

    name: str  # "tp" | "ep" | custom label
    dim: int
    src_stride: int
    dst_stride: int
    src_dim_ranks: list[int]
    dst_dim_ranks: list[int]


@dataclass
class ReshardPlan:
    """Reshard plan - operations for this rank."""

    send_ops: list[TransferOp]
    recv_ops: list[TransferOp]

    def __str__(self):
        return f"ReshardPlan(sends={len(self.send_ops)}, recvs={len(self.recv_ops)})"


# -----------------------------------------------------------------------------
# EP + Metadata helpers
# -----------------------------------------------------------------------------


def _get_rank_in_group(global_rank: int, group_ranks: list[int]) -> int:
    try:
        return group_ranks.index(global_rank)
    except ValueError:
        raise ValueError(
            f"Rank {global_rank} not found in process group {group_ranks}. "
            f"This likely indicates a configuration mismatch."
        )


def _detect_expert_index_from_param_name(param_name: str) -> Optional[int]:
    """Extract expert index from parameter name for TEGroupedMLP per-expert tensors."""
    for part in param_name.split('.'):
        if (
            part.startswith('weight')
            and len(part) > len('weight')
            and part[len('weight') :].isdigit()
        ):
            return int(part[len('weight') :])
        if part.startswith('bias') and len(part) > len('bias') and part[len('bias') :].isdigit():
            return int(part[len('bias') :])
    return None


def assign_ep_resolved_name_inplace(
    meta: ParameterMetadata, *, base_name: str | None = None
) -> None:
    """
    EP-only canonicalization for per-expert parameters.

    Under Expert Parallelism (EP), each rank owns a subset of experts with local indices
    (e.g., rank 1 has "weight0" locally, but it's actually global expert 4). The raw param
    name can't be used to match across source/destination because the same local name refers
    to different global experts on different ranks. This function remaps local expert indices
    to global indices in `resolved_name` and sets `global_expert_index`.

    Effects:
    - Sets meta.resolved_name (defaults to base_name/meta.name for non-EP).
    - Sets meta.global_expert_index for per-expert parameters; otherwise leaves it as None.
    """
    base = meta.name if base_name is None else base_name
    meta.resolved_name = base
    meta.global_expert_index = None
    if not meta.is_ep:
        return

    local_idx = _detect_expert_index_from_param_name(base)
    if local_idx is None:
        # Fused experts tensor: leave name as-is; TP planner will handle slicing
        return
    ep_group = meta.expert_parallel_group_ranks
    ep_size = len(ep_group)
    ep_local_rank = ep_group.index(meta.owner_rank)
    experts_per_rank = meta.num_experts // ep_size
    global_idx = ep_local_rank * experts_per_rank + local_idx
    meta.global_expert_index = global_idx

    # Replace trailing integer in "weightK"/"biasK" with global_idx
    parts = base.split('.')
    new_parts = []
    for p in parts:
        if p.startswith('weight') and len(p) > len('weight') and p[len('weight') :].isdigit():
            new_parts.append('weight' + str(global_idx))
        elif p.startswith('bias') and len(p) > len('bias') and p[len('bias') :].isdigit():
            new_parts.append('bias' + str(global_idx))
        else:
            new_parts.append(p)
    meta.resolved_name = '.'.join(new_parts)


def assign_resolved_name_inplace(
    meta: ParameterMetadata,
    *,
    layer_module_prefix_map: Mapping[str, str] | None = None,
    base_name: str | None = None,
) -> None:
    """Set meta.resolved_name so the planner can match the same weights across models.

    It rewrites PP layer indices to global layer indices (when layer_module_prefix_map is
    provided) and
    rewrites EP per-expert indices (weightK/biasK) to global expert indices.
    """
    name = meta.name if base_name is None else base_name
    if layer_module_prefix_map:
        name = _resolve_global_layer_number_in_name(name, layer_module_prefix_map)
    assign_ep_resolved_name_inplace(meta, base_name=name)


def _build_layer_module_prefix_map(module: torch.nn.Module) -> dict[str, str]:
    """Build a mapping local_module_prefix -> global_module_prefix for PP layer modules.

    Megatron assigns a global, 1-indexed layer_number to each transformer layer module at
    construction time (including PP/VPP/layout offsets). We convert that to the 0-indexed naming
    convention used in parameter names and build a map such as:

    - "decoder.layers.0" → "decoder.layers.16"  (if layer_number == 17)
    """
    prefix_map: dict[str, str] = {}
    for module_name, submodule in module.named_modules():
        if not module_name:
            continue
        layer_number = getattr(submodule, 'layer_number', None)
        if not isinstance(layer_number, int):
            continue
        parts = module_name.split('.')
        if not parts[-1].isdigit():
            continue
        parts[-1] = str(layer_number - 1)  # convert 1-indexed to 0-indexed
        prefix_map[module_name] = '.'.join(parts)
    return prefix_map


def _resolve_global_layer_number_in_name(
    name: str, layer_module_prefix_map: Mapping[str, str]
) -> str:
    """Rewrite a parameter name to use global layer indices (PP-aware).

    Given a parameter name like decoder.layers.0.self_attention..., this function rewrites
    the decoder.layers.0 prefix to the corresponding global layer index using the owning
    layer module's layer_number.

    Implementation:
    - Build a {local_prefix -> global_prefix} map once (outside the per-parameter loop).
    - Perform a longest-prefix match replacement so we only rewrite the module path portion.
    """
    if not layer_module_prefix_map:
        return name

    parts = name.split('.')
    for i in range(len(parts), 0, -1):
        prefix = '.'.join(parts[:i])
        mapped = layer_module_prefix_map.get(prefix)
        if mapped is None:
            continue
        rest = '.'.join(parts[i:])
        return mapped if not rest else mapped + '.' + rest
    return name


def extract_param_metadata(
    param: torch.nn.Parameter,
    param_name: str,
    owner_rank: int,
    pg_collection,
    num_experts: Optional[int] = None,
    layer_module_prefix_map: Mapping[str, str] | None = None,
) -> ParameterMetadata:
    """Extract metadata from a parameter for cross-rank communication."""
    # TP flags from attributes (set by Megatron linear layers)
    is_tp = bool(getattr(param, 'tensor_model_parallel', False))
    partition_dim = int(getattr(param, 'partition_dim', 0))
    partition_stride = int(getattr(param, 'partition_stride', 1))

    # SwiGLU/GLU compatibility: For gated linear units, fc1 stores interleaved [gate, up] portions
    # and requires partition_stride=2 for correct resharding. New models set this at construction
    # time (MLP sets partition_stride=2 on weight when gated_linear_unit=True). For legacy models
    # where stride=1 was left as default, we apply stride=2 as a fallback for fc1 parameters.
    # This is safe because: (1) gated models need it, and (2) non-gated models have smaller fc1
    # and stride doesn't affect single-block transfers.
    # if 'mlp.linear_fc1' in param_name and is_tp and partition_stride == 1:
    #     partition_stride = 2

    # EP detection: Megatron convention - expert params are not allreduced
    is_ep = not bool(getattr(param, 'allreduce', True))

    tensor_parallel_group_ranks: list[int] | None = None
    expert_parallel_group_ranks: list[int] | None = None
    data_parallel_group_ranks: list[int] | None = None
    pipeline_parallel_group_ranks: list[int] | None = None

    if is_ep:
        expert_parallel_group_ranks = dist.get_process_group_ranks(pg_collection.ep)
        # For MoE params, prefer expert TP group when available, else regular TP
        if is_tp and hasattr(pg_collection, 'expt_tp') and pg_collection.expt_tp is not None:
            tensor_parallel_group_ranks = dist.get_process_group_ranks(pg_collection.expt_tp)
        elif is_tp and hasattr(pg_collection, 'tp') and pg_collection.tp is not None:
            tensor_parallel_group_ranks = dist.get_process_group_ranks(pg_collection.tp)
        data_parallel_group_ranks = dist.get_process_group_ranks(pg_collection.dp)
    elif is_tp:
        # Non-EP: use regular TP group
        if hasattr(pg_collection, 'tp') and pg_collection.tp is not None:
            tensor_parallel_group_ranks = dist.get_process_group_ranks(pg_collection.tp)
        data_parallel_group_ranks = dist.get_process_group_ranks(pg_collection.dp)
    else:
        data_parallel_group_ranks = dist.get_process_group_ranks(pg_collection.dp)

    if hasattr(pg_collection, 'pp') and pg_collection.pp is not None:
        pipeline_parallel_group_ranks = dist.get_process_group_ranks(pg_collection.pp)
    else:
        pipeline_parallel_group_ranks = list(range(dist.get_world_size()))

    meta = ParameterMetadata(
        name=param_name,
        shape=tuple(param.shape),
        dtype=param.dtype,
        element_size=param.element_size(),
        is_tp=is_tp,
        partition_dim=partition_dim,
        partition_stride=partition_stride,
        is_ep=is_ep,
        num_experts=num_experts,
        owner_rank=owner_rank,
        tensor_parallel_group_ranks=tensor_parallel_group_ranks,
        expert_parallel_group_ranks=expert_parallel_group_ranks,
        data_parallel_group_ranks=data_parallel_group_ranks,
        pipeline_parallel_group_ranks=pipeline_parallel_group_ranks,
    )
    assign_resolved_name_inplace(
        meta, layer_module_prefix_map=layer_module_prefix_map, base_name=param_name
    )

    return meta


def select_src_metadata_balanced(
    src_meta_list: list[ParameterMetadata], dst_metadata: ParameterMetadata, dst_rank: int
) -> ParameterMetadata:
    """Choose a representative source `ParameterMetadata` for a destination rank.

    Multiple source data-parallel (DP) groups may hold the same logical parameter.
    To avoid always reading from the same group, we:
      - bucket `src_meta_list` by their DP group (tuple of ranks)
      - if there is only one bucket, just return the first entry
      - otherwise, use the destination rank's global rank to select a source
        DP group in a round-robin fashion, ensuring even distribution of load
        across all source DP groups.
    """
    if not src_meta_list:
        raise ValueError("src_meta_list must be non-empty")

    # Group source metadata by their DP group layout so we can balance across groups.
    #   (dp_rank0, dp_rank1, ...) -> [ParameterMetadata for that DP group]
    grouped_by_dp: dict[tuple[int, ...], list[ParameterMetadata]] = {}
    for meta in src_meta_list:
        dp_group = tuple(meta.data_parallel_group_ranks or [])
        grouped_by_dp.setdefault(dp_group, []).append(meta)

    # Fast path: only one DP layout present; no balancing necessary.
    if len(grouped_by_dp) == 1:
        return src_meta_list[0]

    # Use the destination rank's global rank to select a source DP group in a
    # round-robin fashion. This ensures that even when multiple destination ranks
    # have the same DP index (e.g., ranks 0,1,2,3 all being at position 0 in their
    # respective DP groups), they still get distributed across different source
    # DP groups based on their global rank.
    sorted_dp_groups = sorted(grouped_by_dp.keys())
    chosen_group = sorted_dp_groups[dst_rank % len(sorted_dp_groups)]

    # Within the chosen group, any representative metadata works; use the first.
    return grouped_by_dp[chosen_group][0]


logger = logging.getLogger(__name__)
