from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


# -----------------------------------------------------------------------------
# Dataclasses used by the planner
# -----------------------------------------------------------------------------


@dataclass
class TransferOp:
    param_name: str
    peer_rank: int  # Who to send to / receive from
    is_send: bool  # True=send, False=recv

    # Slice information (for when we execute the plan)
    my_slice: tuple[slice, ...]  # My tensor slice
    peer_slice: tuple[slice, ...]  # Peer's tensor slice (for reference)


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

    # Canonicalization for EP per-expert params
    resolved_name: Optional[str] = None
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
    local_copy_ops: list[
        tuple[str, torch.nn.Parameter | None, torch.nn.Parameter | None, tuple[slice, ...], tuple[slice, ...]]
    ]  # (name, src_param, dst_param, src_slice, dst_slice)

    def __str__(self):
        return (
            f"ReshardPlan(sends={len(self.send_ops)}, recvs={len(self.recv_ops)}, "
            f"local_copies={len(self.local_copy_ops)})"
        )


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
        if part.startswith('weight') and len(part) > len('weight') and part[len('weight'):].isdigit():
            return int(part[len('weight'):])
        if part.startswith('bias') and len(part) > len('bias') and part[len('bias'):].isdigit():
            return int(part[len('bias'):])
    return None


def assign_resolved_name_inplace(meta: ParameterMetadata) -> None:
    """
    Compute a canonical resolved_name for EP per-expert parameters, and set global_expert_index.
    For non-EP or non-per-expert params, resolved_name defaults to original name.
    """
    meta.resolved_name = meta.name
    meta.global_expert_index = None
    if not meta.is_ep:
        return

    local_idx = _detect_expert_index_from_param_name(meta.name)
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
    parts = meta.name.split('.')
    new_parts = []
    for p in parts:
        if (p.startswith('weight') and len(p) > len('weight') and p[len('weight'):].isdigit()):
            new_parts.append('weight' + str(global_idx))
        elif (p.startswith('bias') and len(p) > len('bias') and p[len('bias'):].isdigit()):
            new_parts.append('bias' + str(global_idx))
        else:
            new_parts.append(p)
    meta.resolved_name = '.'.join(new_parts)


def extract_param_metadata(
    param: torch.nn.Parameter,
    param_name: str,
    owner_rank: int,
    pg_collection,
    num_experts: Optional[int] = None,
) -> ParameterMetadata:
    """Extract metadata from a parameter for cross-rank communication."""
    # TP flags from attributes (set by Megatron linear layers)
    is_tp = bool(getattr(param, 'tensor_model_parallel', False))
    partition_dim = int(getattr(param, 'partition_dim', 0))
    partition_stride = int(getattr(param, 'partition_stride', 1))
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
    assign_resolved_name_inplace(meta)
    return meta


def select_src_metadata_balanced(
    src_meta_list: list[ParameterMetadata], dst_metadata: ParameterMetadata, dst_rank: int
) -> ParameterMetadata:
    """Choose representative source metadata using DP round-robin across source DP groups."""
    if not src_meta_list:
        raise ValueError("src_meta_list must be non-empty")
    groups: dict[tuple[int, ...], list[ParameterMetadata]] = {}
    for m in src_meta_list:
        key = tuple(m.data_parallel_group_ranks or [])
        groups.setdefault(key, []).append(m)
    if len(groups) == 1:
        return src_meta_list[0]
    dst_dp = dst_metadata.data_parallel_group_ranks or []
    if dst_rank in dst_dp and len(dst_dp) > 0:
        my_dst_dp_idx = dst_dp.index(dst_rank)
    else:
        my_dst_dp_idx = 0
    keys_sorted = sorted(groups.keys())
    chosen_key = keys_sorted[my_dst_dp_idx % len(keys_sorted)]
    return groups[chosen_key][0]


logger = logging.getLogger(__name__)


