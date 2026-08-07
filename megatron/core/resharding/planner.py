# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
import warnings

import torch
import torch.distributed as dist

from .shard_planner import plan_sharded_transfer
from .utils import (
    ParameterMetadata,
    ReshardPlan,
    TransferOp,
    _build_layer_module_prefix_map,
    extract_param_metadata,
    named_refit_tensors,
    select_src_metadata_balanced,
)

logger = logging.getLogger(__name__)


def _iter_global_transfer_ops(
    dst_param_metadata_by_rank: dict[int, dict[str, ParameterMetadata]],
    src_param_metadata: dict[str, list[ParameterMetadata]],
):
    """Yield the whole reshard schedule in a deterministic order.

    The iteration order (dst rank ascending, then that rank's dst params in
    gathered order, then per-source sub-ops) depends only on the rosters, so
    replaying this on any rank produces the same sequence and assigns the same
    task_id to the same transfer. That's what lets each rank build its own
    send/recv ops while sender and receiver still agree on task_id.

    Ranks are taken from the roster keys rather than range(world_size), so a
    sparse or growing rank set (nodes added later) rebuilds identically.

    Yields (task_id, dst_rank, src_rank, src_slice, dst_slice, src_metadata,
    dst_metadata). PP is handled implicitly: each rank contributes metadata only
    for the params it owns, and any source holding the same resolved_name can
    serve as sender (with DP balancing).
    """
    # Shared between a send and its recv; NVSHMEM builds a schedule from it and
    # local copies match on it.
    next_task_id = 0
    for dst_rank in sorted(dst_param_metadata_by_rank):
        dst_rank_params = dst_param_metadata_by_rank[dst_rank]
        for resolved_name, dst_metadata in dst_rank_params.items():
            src_meta_list = src_param_metadata.get(resolved_name)
            if not src_meta_list and resolved_name.endswith("output_layer.weight"):
                # Tied embeddings: the source shares the output projection with
                # the input embedding, so it has no separate output_layer.weight.
                # A pp>1 destination materializes one (embedding and output land
                # on different stages), e.g. pp=1 (tied) -> pp=2. Source it from
                # the embedding weight (same shape + vocab/TP shard); that tensor
                # then feeds both the destination embedding and output_layer.
                for emb_name in ("embedding.word_embeddings.weight", "word_embeddings.weight"):
                    src_meta_list = src_param_metadata.get(emb_name)
                    if src_meta_list:
                        break
            if not src_meta_list:
                raise RuntimeError(
                    f"Destination parameter '{resolved_name}' on rank {dst_rank} "
                    "not found in source model."
                )
            # Choose a representative source metadata with DP round-robin balancing.
            src_metadata = select_src_metadata_balanced(src_meta_list, dst_metadata, dst_rank)
            sources = plan_sharded_transfer(
                resolved_name, src_meta_list, src_metadata, dst_metadata
            )
            for src_rank, src_slice, dst_slice in sources:
                task_id = next_task_id
                next_task_id += 1
                yield task_id, dst_rank, src_rank, src_slice, dst_slice, src_metadata, dst_metadata


def _extract_module_metadata(
    module, owner_rank, num_experts, rank_offset, rank_list_cache
) -> list[ParameterMetadata]:
    """Metadata for a module's params and persistent buffers, or [] if None.

    Persistent buffers travel too so training state (e.g. MoE router expert_bias)
    refits with the weights.
    """
    if module is None:
        return []
    pg = getattr(module, "pg_collection", None)
    if pg is None:
        raise ValueError("Module must have pg_collection")
    layer_prefix_map = _build_layer_module_prefix_map(module)
    return [
        extract_param_metadata(
            p,
            name,
            owner_rank,
            pg,
            num_experts=num_experts,
            layer_module_prefix_map=layer_prefix_map,
            rank_offset=rank_offset,
            _rank_list_cache=rank_list_cache,
        )
        for name, p in named_refit_tensors(module)
    ]


def index_metadata_rosters(gathered_pairs: list):
    """Turn a rank-ordered list of ``(src_meta, dst_meta)`` (index == rank) into the
    two rosters the plan builder consumes: dst params keyed by rank, and src params
    keyed by resolved_name. The list may come from the all-gather, or be reassembled
    in rank order as nodes are added, before calling build_plan_from_rosters.
    """
    dst_param_metadata_by_rank: dict[int, dict[str, ParameterMetadata]] = {}
    src_param_metadata: dict[str, list[ParameterMetadata]] = {}
    for rank_id, (src_meta_list, dst_meta_list) in enumerate(gathered_pairs):
        dst_param_metadata_by_rank[rank_id] = {m.resolved_name: m for m in dst_meta_list}
        for metadata in src_meta_list:
            src_param_metadata.setdefault(metadata.resolved_name, []).append(metadata)
    return dst_param_metadata_by_rank, src_param_metadata


def build_plan_from_rosters(
    dst_param_metadata_by_rank: dict[int, dict[str, ParameterMetadata]],
    src_param_metadata: dict[str, list[ParameterMetadata]],
    my_global_rank: int,
) -> ReshardPlan:
    """Replay the deterministic global schedule and keep only this rank's ops.

    Pure and collective-free, so it can be tested or reused with preassembled
    rosters without touching the process group. Live membership orchestration
    is intentionally outside this module.
    """
    my_plan = ReshardPlan([], [])
    for (
        task_id,
        dst_rank,
        src_rank,
        src_slice,
        dst_slice,
        src_metadata,
        dst_metadata,
    ) in _iter_global_transfer_ops(dst_param_metadata_by_rank, src_param_metadata):
        if dst_rank == my_global_rank:
            my_plan.recv_ops.append(
                TransferOp(
                    param_name=dst_metadata.name,
                    peer_rank=src_rank,
                    is_send=False,
                    my_slice=dst_slice,
                    peer_slice=src_slice,
                    task_id=task_id,
                )
            )
        if src_rank == my_global_rank:
            my_plan.send_ops.append(
                TransferOp(
                    param_name=src_metadata.name,
                    peer_rank=dst_rank,
                    is_send=True,
                    my_slice=src_slice,
                    peer_slice=dst_slice,
                    task_id=task_id,
                )
            )

    logger.info(
        f"Rank {my_global_rank}: Built plan locally - {len(my_plan.recv_ops)} recvs, "
        f"{len(my_plan.send_ops)} sends"
    )
    return my_plan


def build_local_reshard_plan(
    src_module: torch.nn.Module,
    dst_module: torch.nn.Module,
    num_experts: int | None = None,
    group=None,
    src_rank_offset: int = 0,
    dst_rank_offset: int = 0,
) -> ReshardPlan:
    """
    Build this rank's reshard plan locally: all-gather the parameter metadata,
    replay the global schedule (see _iter_global_transfer_ops), and keep only the
    ops where this rank is the sender or receiver. No rank-0 bottleneck and no
    scatter, since sender and receiver derive matching task_ids from the same
    metadata.

    The metadata gather (the one collective) and the plan build are split into
    index_metadata_rosters + build_plan_from_rosters, so the deterministic build
    can also be tested against preassembled rosters without a process group.

    src_module/dst_module may be None for non-collocated ranks (destination-only,
    source-only, or idle). Each rank contributes metadata only for the models it
    owns, including its parallel-group membership.
    """
    # group.rank()/size() (not dist.get_rank(group)) support cross-cluster PGs
    # whose members have independent default PGs.
    my_global_rank = group.rank() if group is not None else dist.get_rank()
    world_size = group.size() if group is not None else dist.get_world_size()

    # Dedup rank lists so params sharing a group reuse one list object; shrinks
    # the pickled all-gather ~75%.
    rank_list_cache: dict = {}
    my_src_metadata = _extract_module_metadata(
        src_module, my_global_rank, num_experts, src_rank_offset, rank_list_cache
    )
    my_dst_metadata = _extract_module_metadata(
        dst_module, my_global_rank, num_experts, dst_rank_offset, rank_list_cache
    )

    # One all-gather gives every rank the full (src, dst) picture, replacing the
    # gather-to-0 + scatter.
    gathered_pairs = [None] * world_size
    dist.all_gather_object(gathered_pairs, (my_src_metadata, my_dst_metadata), group=group)
    del my_src_metadata, my_dst_metadata

    dst_param_metadata_by_rank, src_param_metadata = index_metadata_rosters(gathered_pairs)
    del gathered_pairs
    return build_plan_from_rosters(dst_param_metadata_by_rank, src_param_metadata, my_global_rank)


def build_centralized_reshard_plan(
    src_module: torch.nn.Module,
    dst_module: torch.nn.Module,
    num_experts: int | None = None,
    group=None,
    src_rank_offset: int = 0,
    dst_rank_offset: int = 0,
) -> ReshardPlan:
    """Deprecated compatibility wrapper for :func:`build_local_reshard_plan`."""
    warnings.warn(
        "build_centralized_reshard_plan is deprecated; use build_local_reshard_plan instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_local_reshard_plan(
        src_module,
        dst_module,
        num_experts=num_experts,
        group=group,
        src_rank_offset=src_rank_offset,
        dst_rank_offset=dst_rank_offset,
    )
