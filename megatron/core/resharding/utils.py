# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from .transforms import ReshardTransform

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
    # For parameters that pack multiple independently-sharded components of
    # different sizes (e.g. Mamba in_proj packs z, x, B, C, dt).  When present,
    # lists the per-TP-rank block sizes along partition_dim.  The refit planner
    # interleaves these blocks rather than doing a simple contiguous concat.
    partition_sizes: list[int] | None = None

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
    transform: Optional["ReshardTransform"] = None
    # Cache of canonical persistent-buffer dtypes keyed by raw module path.
    # Populated by _harmonize_buffer_dtypes on first call; reused thereafter to
    # skip the all_gather_object + named_modules() walks on the hot path.
    buffer_dtypes: Optional[dict[str, torch.dtype]] = None

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


_EXPERT_PARAM_RE = re.compile(r"^(weight|bias)(\d+)$")


def _detect_expert_index_from_param_name(param_name: str) -> Optional[int]:
    """Extract expert index from parameter name for TEGroupedMLP per-expert tensors."""
    for part in param_name.split('.'):
        m = _EXPERT_PARAM_RE.match(part)
        if m is not None:
            return int(m.group(2))
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
    new_parts = []
    for p in base.split('.'):
        m = _EXPERT_PARAM_RE.match(p)
        new_parts.append(f"{m.group(1)}{global_idx}" if m else p)
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


def named_persistent_buffers(module: torch.nn.Module):
    """Yield ``(full_name, parent_module, buf_name, tensor)`` for every
    persistent buffer in ``module``.  Skips ``_non_persistent_buffers_set``.

    Persistent buffers (those saved in ``state_dict``) carry training state that
    must travel with the weights during refit/resharding — e.g. the MoE
    router's ``expert_bias``, which is updated each step by aux-loss-free load
    balancing.  Non-persistent buffers are excluded since they hold ephemeral
    state (e.g. accumulators reset at the next train step).
    """
    for module_prefix, sub_module in module.named_modules():
        non_persistent = sub_module._non_persistent_buffers_set
        for buf_name, buf in sub_module._buffers.items():
            if buf is None or buf_name in non_persistent:
                continue
            full_name = f"{module_prefix}.{buf_name}" if module_prefix else buf_name
            yield full_name, sub_module, buf_name, buf


def named_refit_tensors(module: torch.nn.Module):
    """Yield ``(name, tensor)`` pairs for every parameter and persistent buffer.

    Used by the refit planner and executor to enumerate which tensors should
    travel during resharding.  Persistent buffers are included alongside
    parameters because they may carry training state (see
    ``named_persistent_buffers``).
    """
    yield from module.named_parameters(recurse=True)
    for full_name, _sub, _buf_name, buf in named_persistent_buffers(module):
        yield full_name, buf


_REFIT_TENSOR_CACHE_ATTR = "_refit_tensor_cache"


def get_refit_tensor_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return the cached ``{name: tensor}`` dict for ``module``, building it if needed.

    Walking ``named_modules()`` is hundreds of ms for multi-B-parameter models,
    and the parameter/persistent-buffer set is stable across refits — so we
    cache the dict on the module itself.  ``invalidate_refit_tensor_cache``
    must be called whenever a persistent buffer is replaced (e.g. by
    ``_harmonize_buffer_dtypes``) so the cache picks up the new tensor.
    """
    cached = getattr(module, _REFIT_TENSOR_CACHE_ATTR, None)
    if cached is None:
        cached = dict(named_refit_tensors(module))
        setattr(module, _REFIT_TENSOR_CACHE_ATTR, cached)
    return cached


def invalidate_refit_tensor_cache(module: torch.nn.Module) -> None:
    """Drop the cached refit tensor dict so the next call rebuilds it."""
    if hasattr(module, _REFIT_TENSOR_CACHE_ATTR):
        delattr(module, _REFIT_TENSOR_CACHE_ATTR)


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
    rank_offset: int = 0,
    _rank_list_cache: dict | None = None,
) -> ParameterMetadata:
    """Extract metadata from a parameter for cross-rank communication.

    Args:
        _rank_list_cache: Optional dict used to deduplicate rank lists so
            that params sharing the same process group reuse one object.
            This dramatically shrinks pickle size when metadata is gathered
            across many ranks (pickle uses backreferences for same-``id()``
            objects, avoiding re-serialization of identical group lists).
    """
    # TP flags from attributes (set by Megatron linear layers)
    is_tp = bool(getattr(param, 'tensor_model_parallel', False))
    partition_dim = int(getattr(param, 'partition_dim', 0))
    partition_stride = int(getattr(param, 'partition_stride', 1))
    partition_sizes = getattr(param, 'partition_sizes', None)
    if partition_sizes is not None:
        partition_sizes = list(partition_sizes)

    # EP detection: Megatron convention - expert params are not allreduced
    is_ep = not bool(getattr(param, 'allreduce', True))

    # Expert-param detection for TP inference.  When explicit_expert_comm is
    # active (is_expert and (tp_size>1 or ep)), TE clears parallel_mode so
    # tensor_model_parallel is never stamped — yet the weight IS TP-sharded
    # when tp_size > 1.  We detect expert params via num_experts + the
    # per-expert naming convention (weightK / biasK in TEGroupedLinear).
    is_expert_param = (
        num_experts is not None and _detect_expert_index_from_param_name(param_name) is not None
    )

    tensor_parallel_group_ranks: list[int] | None = None
    expert_parallel_group_ranks: list[int] | None = None
    data_parallel_group_ranks: list[int] | None = None
    pipeline_parallel_group_ranks: list[int] | None = None

    # Deduplicate rank lists: params sharing the same TP/DP/EP/PP group get
    # one shared list object instead of separate copies.  This shrinks pickle
    # size ~75% when metadata is gathered across many ranks (pickle uses
    # backreferences for same-id() objects).
    if _rank_list_cache is None:
        _rank_list_cache = {}

    def _dedup_ranks(ranks: list[int]) -> list[int]:
        key = tuple(ranks)
        if key not in _rank_list_cache:
            _rank_list_cache[key] = list(key)
        return _rank_list_cache[key]

    def _offset_ranks(ranks: list[int]) -> list[int]:
        result = [r + rank_offset for r in ranks] if rank_offset else ranks
        return _dedup_ranks(result)

    if is_ep or is_expert_param:
        if is_ep:
            expert_parallel_group_ranks = _offset_ranks(
                dist.get_process_group_ranks(pg_collection.ep)
            )
        # For expert params, always provide TP group ranks so the planner can
        # handle TP size transitions (e.g., TP2→TP1).  When explicit_expert_comm
        # clears TE's parallel_mode, tensor_model_parallel may not be set even
        # though the weight IS TP-sharded.  Detect TP via group size instead.
        expt_tp = getattr(pg_collection, 'expt_tp', None)
        tp_grp = expt_tp if expt_tp is not None else getattr(pg_collection, 'tp', None)
        if tp_grp is not None:
            tp_ranks = _offset_ranks(dist.get_process_group_ranks(tp_grp))
            tensor_parallel_group_ranks = tp_ranks
            if not is_tp and len(tp_ranks) > 1:
                is_tp = True
        data_parallel_group_ranks = _offset_ranks(dist.get_process_group_ranks(pg_collection.dp))
    elif is_tp:
        # Non-EP: use regular TP group
        if hasattr(pg_collection, 'tp') and pg_collection.tp is not None:
            tensor_parallel_group_ranks = _offset_ranks(
                dist.get_process_group_ranks(pg_collection.tp)
            )
        data_parallel_group_ranks = _offset_ranks(dist.get_process_group_ranks(pg_collection.dp))
    else:
        data_parallel_group_ranks = _offset_ranks(dist.get_process_group_ranks(pg_collection.dp))

    # Always provide TP group ranks so the planner can handle TP size transitions
    # (e.g., TP2→TP1).  When is_tp=False the param is replicated across the TP group,
    # but the planner still needs to know the TP topology to plan gather/scatter ops
    # when the *other* side of the reshard IS TP-sharded.
    if (
        tensor_parallel_group_ranks is None
        and hasattr(pg_collection, 'tp')
        and pg_collection.tp is not None
    ):
        tensor_parallel_group_ranks = _offset_ranks(dist.get_process_group_ranks(pg_collection.tp))

    if hasattr(pg_collection, 'pp') and pg_collection.pp is not None:
        pipeline_parallel_group_ranks = _offset_ranks(
            dist.get_process_group_ranks(pg_collection.pp)
        )
    else:
        pipeline_parallel_group_ranks = _dedup_ranks(
            list(range(rank_offset, rank_offset + dist.get_world_size()))
        )

    meta = ParameterMetadata(
        name=param_name,
        shape=tuple(param.shape),
        dtype=param.dtype,
        element_size=param.element_size(),
        is_tp=is_tp,
        partition_dim=partition_dim,
        partition_stride=partition_stride,
        partition_sizes=partition_sizes,
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


def _filter_by_ep_local_rank(
    src_meta_list: list[ParameterMetadata], dst_metadata: ParameterMetadata, dst_rank: int
) -> list[ParameterMetadata]:
    """In non-collocated mode with matching EP size, restrict candidates to the
    source rank holding the same global experts as ``dst_rank``.

    When EP sizes differ (resharding), expert matching is handled via
    ``resolved_name`` and no filter is applied here.

    Why size check matters:
      - Same size (EP=8→EP=8): local ranks 0-7 exist in both src and dst →
        filter ensures dst EP local 0 uses src EP local 0 (same global experts).
      - Different size (EP=8→EP=16): dst EP local 8 has no corresponding src
        EP local → skip filter; expert reassignment is handled by resolved_name
        matching, and the LCM/TP planner handles any TP dimension changes.
    """
    dst_ep_group = dst_metadata.expert_parallel_group_ranks
    if dst_ep_group is None:
        return src_meta_list

    dst_ep_local = dst_ep_group.index(dst_rank)
    src_ep_size = (
        len(src_meta_list[0].expert_parallel_group_ranks)
        if src_meta_list[0].expert_parallel_group_ranks
        else None
    )
    # EP resharding (sizes differ) — skip filter; keep all source candidates.
    if src_ep_size != len(dst_ep_group):
        return src_meta_list

    matching = [
        m
        for m in src_meta_list
        if m.expert_parallel_group_ranks
        and m.expert_parallel_group_ranks.index(m.owner_rank) == dst_ep_local
    ]
    if not matching:
        # Sizes match but no local rank match — configuration bug.
        available = [
            (
                m.owner_rank,
                (
                    m.expert_parallel_group_ranks.index(m.owner_rank)
                    if m.expert_parallel_group_ranks
                    else None
                ),
            )
            for m in src_meta_list
        ]
        raise ValueError(
            f"No source metadata with EP local rank {dst_ep_local}"
            f" found for dst rank {dst_rank}. Available: {available}"
        )
    return matching


def _round_robin_dp(src_meta_list: list[ParameterMetadata], dst_rank: int) -> ParameterMetadata:
    """Round-robin across source DP groups so transfer load spreads evenly.

    Each DP group holds a complete copy of the model, so we can read from any
    DP group; choosing via ``dst_rank % num_src_dp_groups`` ensures even
    distribution even when destination has different DP configuration.  E.g.
    with 4 src DP groups and 128 dst ranks, each src DP group is selected by
    32 dst ranks (128/4=32).  Within the chosen DP group we further cycle
    across available metadata entries to balance load across TP groups in the
    DP replica.
    """
    grouped_by_dp: dict[tuple[int, ...], list[ParameterMetadata]] = {}
    for meta in src_meta_list:
        dp_group = tuple(meta.data_parallel_group_ranks or [])
        grouped_by_dp.setdefault(dp_group, []).append(meta)

    # Fast path: only one DP group present; no balancing necessary.
    if len(grouped_by_dp) == 1:
        return src_meta_list[0]

    # Round-robin selection across source DP groups based on destination global rank.
    # This ensures even distribution: if we have 4 src DP groups and 128 dst ranks,
    # each src DP group will be selected by 32 dst ranks (128 / 4 = 32).
    sorted_dp_groups = sorted(grouped_by_dp.keys())
    chosen_group = sorted_dp_groups[dst_rank % len(sorted_dp_groups)]
    # Within the chosen DP group, distribute across available metadata entries
    # to balance load across all TP groups in the DP replica.
    # Example: With 4 TP groups in a DP group, dst_ranks will cycle through all 4
    # instead of always using the first one, better distributing transfer load.
    group_metadata = grouped_by_dp[chosen_group]
    within_group_idx = (dst_rank // len(sorted_dp_groups)) % len(group_metadata)
    return group_metadata[within_group_idx]


def select_src_metadata_balanced(
    src_meta_list: list[ParameterMetadata], dst_metadata: ParameterMetadata, dst_rank: int
) -> ParameterMetadata:
    """Choose a representative source `ParameterMetadata` for a destination rank.

    The selected metadata supplies topology (TP/EP/DP group ranks) to the LCM
    planner.  Selection prefers a local copy when ``dst_rank`` itself owns a
    source replica, then round-robins across source DP groups to balance load.
    A local copy is essentially free (``tensor.copy_()`` on same GPU), while
    any remote transfer incurs significant overhead even within the same node.
    """
    if not src_meta_list:
        raise ValueError("src_meta_list must be non-empty")

    candidates = _filter_by_ep_local_rank(src_meta_list, dst_metadata, dst_rank)

    # Local copy optimization (collocated mode): if dst_rank owns a source
    # replica after EP filtering, use it directly to skip the network entirely.
    for meta in candidates:
        if meta.owner_rank == dst_rank:
            return meta

    return _round_robin_dp(candidates, dst_rank)
