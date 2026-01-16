# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
import math

import torch
import torch.distributed as dist

from .utils import (
    ParameterMetadata,
    ReshardPlan,
    ShardingDescriptor,
    TransferOp,
    _build_layer_module_prefix_map,
    _get_rank_in_group,
    extract_param_metadata,
    select_src_metadata_balanced,
)

logger = logging.getLogger(__name__)


def _build_descriptors_for_param(
    src_metadata: ParameterMetadata, dst_metadata: ParameterMetadata
) -> list[ShardingDescriptor]:
    """Construct sharding descriptors (currently TP) for this parameter based on actual layout.
    Guard TP descriptor with size conservation so we don't mis-classify replicated tensors.
    """
    descriptors: list[ShardingDescriptor] = []

    # TP descriptor: allow when either side participates in TP
    if src_metadata.is_tp or dst_metadata.is_tp:
        # Prefer destination partition_dim, else source
        tp_dim = dst_metadata.partition_dim if dst_metadata.is_tp else src_metadata.partition_dim
        src_tp_ranks = src_metadata.tensor_parallel_group_ranks
        dst_tp_ranks = dst_metadata.tensor_parallel_group_ranks
        if src_tp_ranks is None or dst_tp_ranks is None:
            # Not enough context to build TP descriptor
            return descriptors
        src_stride = src_metadata.partition_stride if src_metadata.is_tp else 1
        dst_stride = dst_metadata.partition_stride if dst_metadata.is_tp else 1

        # Size conservation check on partition dim
        src_world = len(src_tp_ranks)
        dst_world = len(dst_tp_ranks)
        src_local = src_metadata.shape[tp_dim]
        dst_local = dst_metadata.shape[tp_dim]
        if src_world * src_local != dst_world * dst_local:
            raise RuntimeError(
                f"Cannot build TP descriptor for {dst_metadata.name} dim{tp_dim}: "
                f"src_world*src_local={src_world}*{src_local} != {dst_world}*{dst_local}. "
                "This usually means the param is marked TP but is effectively replicated on that "
                "dim or partition_dim/metadata is inconsistent between source and destination."
            )

        descriptors.append(
            ShardingDescriptor(
                name="tp",
                dim=tp_dim,
                src_stride=src_stride,
                dst_stride=dst_stride,
                src_dim_ranks=src_tp_ranks,
                dst_dim_ranks=dst_tp_ranks,
            )
        )
    return descriptors


def _plan_multi_dim_lcm(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    descriptors: list[ShardingDescriptor],
    my_global_rank: int,
) -> list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]:
    """
    TP-only planner using LCM tiling to support strides on source/destination.
    - Requires exactly one TP descriptor
    - Supports arbitrary integer strides (contiguous micro-tiles)
    """
    if not descriptors:
        return []
    if len(descriptors) != 1:
        raise NotImplementedError(
            f"{param_name}: _plan_multi_dim_lcm supports TP-only (one descriptor)"
        )
    if descriptors[0].name != "tp":
        raise NotImplementedError(f"{param_name}: _plan_multi_dim_lcm expects TP descriptor")
    d = descriptors[0]
    if my_global_rank not in d.dst_dim_ranks:
        return []
    src_shape = tuple(src_metadata.shape)
    dst_shape = tuple(dst_metadata.shape)
    dim = d.dim
    src_world = len(d.src_dim_ranks)
    dst_world = len(d.dst_dim_ranks)
    src_local = src_shape[dim]
    dst_local = dst_shape[dim]
    if src_world * src_local != dst_world * dst_local:
        raise RuntimeError(
            f"{param_name}: size mismatch on TP dim{dim} "
            f"(src_world={src_world}, src_local={src_local}, "
            f"dst_world={dst_world}, dst_local={dst_local})"
        )
    # LCM tiling with strides
    Ns = src_world * max(1, d.src_stride)
    Nd = dst_world * max(1, d.dst_stride)
    full_len = dst_local * dst_world
    g = math.gcd(Ns, Nd)
    L = (Ns // g) * Nd
    if full_len % L != 0:
        raise RuntimeError(
            f"{param_name}: TP dim{dim} full_len {full_len} not divisible by LCM {L} "
            f"(Ns={Ns}, Nd={Nd})"
        )
    unit = full_len // L  # micro-tile length
    cps = L // Ns  # micro-tiles per source segment
    cpd = L // Nd  # micro-tiles per destination segment
    seg_src = cps * unit  # contiguous length per source segment
    seg_dst = cpd * unit  # contiguous length per destination segment
    dst_local_rank = _get_rank_in_group(my_global_rank, d.dst_dim_ranks)
    ops: list[tuple[int, tuple[slice, ...], tuple[slice, ...]]] = []
    # Sweep destination segments owned by this rank (handle destination stride)
    for k in range(max(1, d.dst_stride)):
        g_dst_seg = dst_local_rank + k * dst_world
        # Within this segment, enumerate the cpd micro-tiles
        for off in range(cpd):
            g_micro = g_dst_seg * cpd + off
            s_idx = g_micro // cps
            in_seg = g_micro % cps
            src_owner_in_dim = s_idx % src_world
            src_global_rank = d.src_dim_ranks[src_owner_in_dim]
            src_local_seg_idx = s_idx // src_world
            src_start = src_local_seg_idx * seg_src + in_seg * unit
            dst_start = k * seg_dst + off * unit
            # Build full N-D slices
            src_slice = [slice(None)] * len(src_shape)
            dst_slice = [slice(None)] * len(dst_shape)
            src_slice[dim] = slice(src_start, src_start + unit)
            dst_slice[dim] = slice(dst_start, dst_start + unit)
            ops.append((src_global_rank, tuple(src_slice), tuple(dst_slice)))

    # Stable order by destination offset
    def dst_key(op):
        _, _, dsl = op
        s = dsl[dim]
        return s.start if isinstance(s, slice) else 0

    ops.sort(key=dst_key)
    return ops


def _finalize_dp_transfers(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    my_global_rank: int,
) -> list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]:
    """Return receiver-side transfer for a parameter that is not TP-sharded.

    This is reached when we cannot build a TP sharding descriptor for the parameter
    (i.e., it is effectively replicated with respect to sharding).  We use this when the
    destination and source mode have no TP or the parameter is replicted on all ranks
    such as layernorm. If the source and destination DP groups match, we return a local
    full-tensor copy; otherwise we pick a source rank from the source DP group in a
    deterministic round-robin manner based on the receiver's global rank for better load
    distribution.
    """
    dst_dp_ranks = dst_metadata.data_parallel_group_ranks
    src_dp_ranks = src_metadata.data_parallel_group_ranks
    if my_global_rank not in dst_dp_ranks:
        return []

    dst_shape = dst_metadata.shape

    # Same DP layout - local copy
    if src_dp_ranks == dst_dp_ranks:
        full_slice = tuple(slice(None) for _ in range(len(dst_shape)))
        return [(my_global_rank, full_slice, full_slice)]

    # Different DP groups - use round-robin based on destination global rank for
    # better load balancing across source ranks. This ensures that destination
    # ranks are distributed across source ranks even when they have the same
    # position within their respective DP groups.
    src_global_rank = src_dp_ranks[my_global_rank % len(src_dp_ranks)]
    full_slice = tuple(slice(None) for _ in range(len(dst_shape)))
    return [(src_global_rank, full_slice, full_slice)]


def _determine_source_ranks_for_dst_param(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    my_global_rank: int,
) -> list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]:
    """Route to dimension-specific planner based on parameter sharding type."""

    # Regular TP/DP planning with EP-resolved metadata
    descriptors = _build_descriptors_for_param(src_metadata=src_metadata, dst_metadata=dst_metadata)
    if descriptors:
        return _plan_multi_dim_lcm(
            param_name=param_name,
            src_metadata=src_metadata,
            dst_metadata=dst_metadata,
            descriptors=descriptors,
            my_global_rank=my_global_rank,
        )
    # DP / replicated fallback
    return _finalize_dp_transfers(param_name, src_metadata, dst_metadata, my_global_rank)


def build_centralized_reshard_plan(
    src_module: torch.nn.Module, dst_module: torch.nn.Module, num_experts: int = None
) -> ReshardPlan:
    """
    Centralized planning: Rank 0 builds complete plan for all ranks, then scatters.
    """
    my_global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Get process groups
    src_pg = getattr(src_module, "pg_collection", None)
    dst_pg = getattr(dst_module, "pg_collection", None)
    if src_pg is None or dst_pg is None:
        raise ValueError("Both modules must have pg_collection")

    # Gather param metadata from all ranks
    my_src_params = {name: p for name, p in src_module.named_parameters(recurse=True)}
    my_dst_params = {name: p for name, p in dst_module.named_parameters(recurse=True)}

    # Build PP layer prefix maps to be used for parameter name rewriting
    src_layer_prefix_map = _build_layer_module_prefix_map(src_module)
    dst_layer_prefix_map = _build_layer_module_prefix_map(dst_module)

    my_src_metadata = [
        extract_param_metadata(
            p,
            name,
            my_global_rank,
            src_pg,
            num_experts=num_experts,
            layer_module_prefix_map=src_layer_prefix_map,
        )
        for name, p in my_src_params.items()
    ]
    my_dst_metadata = [
        extract_param_metadata(
            p,
            name,
            my_global_rank,
            dst_pg,
            num_experts=num_experts,
            layer_module_prefix_map=dst_layer_prefix_map,
        )
        for name, p in my_dst_params.items()
    ]

    all_src_metadata_by_rank = [None] * world_size
    all_dst_metadata_by_rank = [None] * world_size
    dist.all_gather_object(all_src_metadata_by_rank, my_src_metadata)
    dist.all_gather_object(all_dst_metadata_by_rank, my_dst_metadata)

    # Parameter to metadata maps keyed by resolved_name
    src_param_metadata_by_rank = {}
    dst_param_metadata_by_rank = {}
    src_param_metadata: dict[str, list[ParameterMetadata]] = {}

    for rank_id, rank_metadata_list in enumerate(all_src_metadata_by_rank):
        src_param_metadata_by_rank[rank_id] = {m.resolved_name: m for m in rank_metadata_list}
    for rank_id, rank_metadata_list in enumerate(all_dst_metadata_by_rank):
        dst_param_metadata_by_rank[rank_id] = {m.resolved_name: m for m in rank_metadata_list}
    for rank_metadata_list in all_src_metadata_by_rank:
        for metadata in rank_metadata_list:
            key = metadata.resolved_name
            if key not in src_param_metadata:
                src_param_metadata[key] = []
            src_param_metadata[key].append(metadata)

    # Build the plan on global rank 0 and broadcast to all ranks
    if my_global_rank == 0:
        plans_for_all_ranks = {r: ReshardPlan([], []) for r in range(world_size)}
        # Global monotonically increasing ID for non-local transfers.
        # This is shared between the corresponding send/recv ops so that
        # NVSHMEM can build schedule.
        next_task_id = 0

        # Pipeline-parallel (PP) "mapping" is handled implicitly.
        # Each rank contributes metadata only for the parameters it actually owns
        # (i.e., the module partitioning for its PP stage). When PP sizes differ
        # between source and destination, we don't compute an explicit stage-to-stage
        # mapping here; instead, we iterate destination ranks and plan copies for the
        # parameters present on those ranks. Any source rank that has the same logical
        # parameter (matched by resolved_name) can serve as a sender (with DP balancing),
        # and TP slicing is applied when applicable.
        for dst_rank in range(world_size):
            dst_rank_params = dst_param_metadata_by_rank.get(dst_rank, {})
            for resolved_name, dst_metadata in dst_rank_params.items():
                src_meta_list = src_param_metadata.get(resolved_name)
                if not src_meta_list:
                    raise RuntimeError(
                        f"Destination parameter '{resolved_name}' on rank {dst_rank} "
                        "not found in source model."
                    )
                # Choose a representative source metadata with DP round-robin balancing
                src_metadata = select_src_metadata_balanced(src_meta_list, dst_metadata, dst_rank)
                sources = _determine_source_ranks_for_dst_param(
                    resolved_name, src_metadata, dst_metadata, dst_rank
                )
                for src_rank, src_slice, dst_slice in sources:
                    task_id = next_task_id
                    next_task_id += 1

                    plans_for_all_ranks[dst_rank].recv_ops.append(
                        TransferOp(
                            param_name=dst_metadata.name,
                            peer_rank=src_rank,
                            is_send=False,
                            my_slice=dst_slice,
                            peer_slice=src_slice,
                            task_id=task_id,
                        )
                    )
                    plans_for_all_ranks[src_rank].send_ops.append(
                        TransferOp(
                            param_name=src_metadata.name,
                            peer_rank=dst_rank,
                            is_send=True,
                            my_slice=src_slice,
                            peer_slice=dst_slice,
                            task_id=task_id,
                        )
                    )
        plans_list = [plans_for_all_ranks[r] for r in range(world_size)]
    else:
        plans_list = [None] * world_size
    torch.distributed.broadcast_object_list(plans_list, src=0)
    my_plan = plans_list[my_global_rank]

    logger.info(
        f"Rank {my_global_rank}: Received plan - {len(my_plan.recv_ops)} recvs, "
        f"{len(my_plan.send_ops)} sends"
    )

    return my_plan
