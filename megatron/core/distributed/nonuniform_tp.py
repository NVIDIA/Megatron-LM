# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""
Nonuniform Tensor Parallelism (NTP) - Non-intrusive implementation.

This module provides fault tolerance for tensor-parallel training by allowing
a subset of TP ranks ("spares") to handle failures while "core" ranks continue computation.

All NTP logic is contained in this module as subclasses of core components,
making it non-intrusive to the main codebase.

Usage:
    Instead of using the standard classes, use the NTP variants:
    - NonuniformTPDistributedDataParallel instead of DistributedDataParallel
    - Call initialize_nonuniform_tp_process_groups() after initialize_model_parallel()
"""

import fnmatch
import functools
import inspect
import logging
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .. import parallel_state
from ..process_groups_config import ProcessGroupCollection
from ..transformer.cuda_graphs import is_graph_capturing
from ..transformer.transformer_config import TransformerConfig
from ..utils import log_on_each_pipeline_stage
from . import distributed_data_parallel as ddp_module
from . import param_and_grad_buffer as pgb
from .distributed_data_parallel import DistributedDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import _ParamAndGradBucketGroup, _ParamAndGradBuffer

logger = logging.getLogger(__name__)


def pad_to_divisor(value: int, divisor: int) -> int:
    """Round up ``value`` to the nearest multiple of ``divisor``."""
    return int(math.ceil(value / divisor) * divisor)


def pad_param_start(param_start_index: int) -> int:
    """Align parameter start index to a 64-element boundary."""
    return pad_to_divisor(param_start_index, 64)


def pad_bucket_end(
    bucket_end_index: int, data_parallel_world_size: int, pad_for_high_nccl_busbw: bool
) -> int:
    """Pad bucket end for DP divisibility and optionally high NCCL bus bandwidth."""
    if pad_for_high_nccl_busbw:
        divisor = math.lcm(data_parallel_world_size, 128, 2**16)
    else:
        divisor = math.lcm(data_parallel_world_size, 128)
    return pad_to_divisor(bucket_end_index, divisor)


@dataclass
class PerBufferParamLayout:
    """Layout for parameters within one NTP-owned contiguous DDP buffer."""

    param_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = field(default_factory=dict)
    side_grad_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = field(
        default_factory=dict
    )
    bucket_indices: List[Tuple[int, int]] = field(default_factory=list)
    per_bucket_numel_unpadded: List[int] = field(default_factory=list)
    param_indices: List[int] = field(default_factory=list)


@dataclass
class FullParamLayout:
    """Compatibility placeholder for callers that pass precomputed layouts."""

    layouts: Dict[object, PerBufferParamLayout] = field(default_factory=dict)


class _NTPAllToAllHandle:
    """Async all_to_all handle that copies temporary contiguous outputs back into views."""

    def __init__(self, handle, output_copies):
        self.handle = handle
        self.output_copies = output_copies

    def wait(self):
        """Wait for the collective and copy temporary outputs into their target views."""
        self.handle.wait()
        for dst, src in self.output_copies:
            dst.copy_(src)
        self.output_copies = []


def _ntp_all_to_all(output_tensors, input_tensors, group, async_op: bool = False):
    """Run all_to_all, preserving non-contiguous output views via temporary buffers."""
    output_list = []
    output_copies = []
    for tensor in output_tensors:
        if tensor.is_contiguous():
            output_list.append(tensor)
        else:
            contiguous = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            output_list.append(contiguous)
            output_copies.append((tensor, contiguous))

    handle = dist.all_to_all(output_list, input_tensors, group=group, async_op=async_op)
    if async_op:
        return _NTPAllToAllHandle(handle, output_copies)

    for dst, src in output_copies:
        dst.copy_(src)
    return None


def _ntp_get_non_active_ranks(
    ntp_config: "NonuniformTPConfig", dp_rank: int, cp_rank: int = 0, pp_rank: int = 0
) -> Optional[List[int]]:
    """Return configured inactive local TP ranks, accepting both legacy and tuple keys."""
    if not ntp_config.non_active_ranks_per_dp:
        return None

    rank_key = (dp_rank, cp_rank, pp_rank)
    if rank_key in ntp_config.non_active_ranks_per_dp:
        return ntp_config.non_active_ranks_per_dp[rank_key]
    if dp_rank in ntp_config.non_active_ranks_per_dp:
        return ntp_config.non_active_ranks_per_dp[dp_rank]
    return None


def _ntp_current_rank_is_reduced_dp(ntp_config: "NonuniformTPConfig") -> bool:
    """Return True if this rank belongs to a DP replica configured with reduced TP."""
    if ntp_config.tp_spares == 0:
        return False

    dp_rank = parallel_state.get_data_parallel_rank()
    if ntp_config.non_active_ranks_per_dp:
        cp_rank = parallel_state.get_context_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        return _ntp_get_non_active_ranks(ntp_config, dp_rank, cp_rank, pp_rank) is not None
    return dp_rank < ntp_config.num_reduced_tp_dp_ranks


def _ntp_current_rank_should_dp_sync(ntp_config: "NonuniformTPConfig") -> bool:
    """Return True if this rank should participate in data-parallel grad sync."""
    if ntp_config.tp_spares == 0:
        return True

    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    reduced_tp_size = ntp_config.tp_base - ntp_config.tp_spares

    # Reduced DP replicas only contain active TP ranks after NTP group reconfiguration.
    if tp_size != ntp_config.tp_base:
        return True

    # In healthy full-TP replicas, ranks beyond reduced_tp_size are folded into core ranks by
    # NTP resharding and must not wait for a DP peer from the reduced replica.
    return tp_rank < reduced_tp_size


def _ntp_can_query_parallel_state() -> bool:
    """Return True when distributed/model-parallel state is initialized enough for NTP."""
    if not dist.is_available() or not dist.is_initialized():
        return False
    try:
        parallel_state.get_tensor_model_parallel_world_size()
        parallel_state.get_tensor_model_parallel_rank()
        parallel_state.get_data_parallel_rank()
        parallel_state.get_context_parallel_rank()
        parallel_state.get_pipeline_model_parallel_rank()
    except Exception:
        return False
    return True


def _ntp_param_can_reshard(param: torch.nn.Parameter) -> bool:
    """Return True for tensor-parallel params initialized with NTP split metadata."""
    return (
        hasattr(param, 'tensor_model_parallel')
        and param.tensor_model_parallel
        and hasattr(param, 'partition_dim')
        and hasattr(param, 'send_splits')
        and hasattr(param, 'recv_splits')
    )


def _ntp_should_expand_param_grad(
    param: torch.nn.Parameter, ntp_config: "NonuniformTPConfig"
) -> bool:
    """Return True if healthy core rank needs side_grad storage for this TP parameter."""
    if ntp_config.tp_spares == 0 or not _ntp_param_can_reshard(param):
        return False
    if _ntp_current_rank_is_reduced_dp(ntp_config):
        return False
    if parallel_state.get_tensor_model_parallel_world_size() != ntp_config.tp_base:
        return False
    return parallel_state.get_tensor_model_parallel_rank() < (
        ntp_config.tp_base - ntp_config.tp_spares
    )


def _ntp_extra_partition_dim(param: torch.nn.Parameter, ntp_config: "NonuniformTPConfig") -> int:
    """Return side_grad extent along partition_dim for this healthy core rank."""
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    return int(sum(param.recv_splits[tp_rank][-ntp_config.tp_spares :]))


def _ntp_param_numel(param: torch.nn.Parameter, ntp_config: "NonuniformTPConfig") -> int:
    """Return main grad plus any NTP side grad storage needed for this param."""
    numel = param.data.nelement()
    if _ntp_should_expand_param_grad(param, ntp_config):
        side_shape = list(param.data.shape)
        side_shape[param.partition_dim] = _ntp_extra_partition_dim(param, ntp_config)
        numel += torch.Size(side_shape).numel()
    return numel


def _ntp_empty_like_partition(
    param: torch.nn.Parameter, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Create a zero-width tensor matching a TP param's partition dimension."""
    empty_shape = list(param.shape)
    empty_shape[param.partition_dim] = 0
    return torch.empty(empty_shape, device=param.device, dtype=dtype or param.dtype).contiguous()


def _ntp_split_for_all_to_all(tensor: torch.Tensor, splits: List[int], dim: int):
    """Split tensor for all_to_all, preserving zero-sized entries."""
    return [piece.contiguous() for piece in torch.split(tensor, splits, dim=dim)]


def _ntp_split_views_for_all_to_all(tensor: torch.Tensor, splits: List[int], dim: int):
    """Split tensor into output views for all_to_all receive paths."""
    return list(torch.split(tensor, splits, dim=dim))


def _compute_ntp_per_buffer_param_layout(
    params: List[torch.nn.Parameter],
    bucket_size: Optional[int],
    data_parallel_world_size: int,
    ddp_config: DistributedDataParallelConfig,
    ntp_config: "NonuniformTPConfig",
    param_indices: Optional[List[int]] = None,
) -> PerBufferParamLayout:
    """Compute a buffer layout that includes side_grad storage for healthy core ranks."""

    def _does_param_require_new_bucket(param):
        return getattr(param, "shared_embedding", False)

    param_index_map = {}
    side_grad_index_map = {}
    bucket_indices = []
    per_bucket_numel_unpadded = []

    param_start_index = 0
    bucket_start_index = 0
    bucket_params = set()
    bucket_id = 0

    def _finalize_bucket(param_end_index, bucket_start_index, bucket_id):
        per_bucket_numel_unpadded.append(param_end_index - bucket_start_index)
        if ddp_config.use_distributed_optimizer:
            bucket_end_index = pad_bucket_end(
                param_end_index,
                data_parallel_world_size,
                ddp_config.pad_buckets_for_high_nccl_busbw,
            )
        else:
            bucket_end_index = param_end_index
        bucket_indices.append((bucket_start_index, bucket_end_index))
        return bucket_end_index, bucket_id + 1

    for param in params[::-1]:
        if ddp_config.use_distributed_optimizer:
            param_start_index = pad_param_start(param_start_index)

        if _does_param_require_new_bucket(param) and len(bucket_params) > 0:
            bucket_start_index, bucket_id = _finalize_bucket(
                param_start_index, bucket_start_index, bucket_id
            )
            bucket_params = set()
            param_start_index = bucket_start_index

        main_numel = param.data.nelement()
        param_main_end_index = param_start_index + main_numel
        param_end_index = param_start_index + _ntp_param_numel(param, ntp_config)
        param_index_map[param] = (param_start_index, param_main_end_index, bucket_id)
        if param_end_index > param_main_end_index:
            side_grad_index_map[param] = (param_main_end_index, param_end_index, bucket_id)
        bucket_params.add(param)

        if (
            bucket_size is not None and (param_end_index - bucket_start_index) >= bucket_size
        ) or _does_param_require_new_bucket(param):
            bucket_start_index, bucket_id = _finalize_bucket(
                param_end_index, bucket_start_index, bucket_id
            )
            bucket_params = set()
            param_start_index = bucket_start_index
        else:
            param_start_index = param_end_index

    if len(bucket_params) > 0:
        _finalize_bucket(param_end_index, bucket_start_index, bucket_id)

    return PerBufferParamLayout(
        param_index_map=param_index_map,
        side_grad_index_map=side_grad_index_map,
        bucket_indices=bucket_indices,
        per_bucket_numel_unpadded=per_bucket_numel_unpadded,
        param_indices=param_indices if param_indices is not None else [],
    )


# ======================================================================================
# NTP Configuration
# ======================================================================================


@dataclass
class NonuniformTPConfig:
    """Configuration for Nonuniform Tensor Parallelism (NTP).

    NTP provides fault tolerance for tensor-parallel training by designating
    a subset of TP ranks as "spares" that can handle GPU failures.
    """

    tp_base: int = 8
    """Base for tensor parallelism. This is the number of ranks in healthy tensor parallel groups.
       Used for nonuniform tensor parallelism."""

    tp_spares: int = 0
    """Number of spares for nonuniform tensor parallelism.

       When > 0, (tp_base - tp_spares) ranks handle computation and tp_spares ranks
       provide fault tolerance.
    """

    num_reduced_tp_dp_ranks: int = 1
    """Number of DP ranks that use reduced TP (tp_base - tp_spares). The remaining DP ranks use
       full tp_base. Reduced TP ranks are assumed to come first in the global rank ordering."""

    non_active_ranks_per_dp: Optional[Dict[Tuple[int, int, int], List[int]]] = None
    """Mapping of (DP rank, CP rank, PP rank) to list of non-active (spare) local TP rank IDs.
       This allows specifying arbitrary GPU failures across all parallelism dimensions.
       Example: {(0,0,0): [0,3], (0,1,0): [1,2], (1,0,0): [0,3]} means:
         - DP rank 0, CP rank 0, PP rank 0 has local TP ranks 0,3 as spares
         - DP rank 0, CP rank 1, PP rank 0 has local TP ranks 1,2 as spares
         - DP rank 1, CP rank 0, PP rank 0 has local TP ranks 0,3 as spares
       The number of non-active ranks must be consistent across CP replicas within each DP rank.
       If None, defaults to last tp_spares ranks as non-active."""


# ======================================================================================
# Utility Functions for NTP Configuration
# ======================================================================================


def compute_uniform_tp_spares_with_parity(
    faulty_gpu_map: Dict[int, List[int]], tp_base: int
) -> Tuple[int, Dict[int, List[int]]]:
    """
    Compute uniform tp_spares across all faulty DP ranks and add additional
    non-active ranks to achieve parity.

    Strategy:
    1. Find the maximum number of failed GPUs across all affected DP ranks
    2. Use this as tp_spares (smallest reduced_tp that works for all)
    3. For DP ranks with fewer failures, pad with additional healthy GPUs
       to reach uniform tp_spares

    Args:
        faulty_gpu_map: Mapping of DP rank -> list of failed GPU IDs
        tp_base: Base tensor parallel size

    Returns:
        Tuple of (tp_spares, non_active_ranks_per_dp)
        where non_active_ranks_per_dp includes both failed and padded GPUs

    Example:
        Input:  {0: [2, 5], 1: [1]}  # DP rank 0 has 2 failures, DP rank 1 has 1
        Output: (2, {0: [2, 5], 1: [1, 7]})  # Pad DP rank 1 with GPU 7 to reach 2
    """
    if not faulty_gpu_map:
        return 0, {}

    # Find maximum number of failures
    max_failures = max(len(gpu_ids) for gpu_ids in faulty_gpu_map.values())
    tp_spares = max_failures

    non_active_ranks_per_dp = {}

    for dp_rank, failed_gpus in faulty_gpu_map.items():
        non_active = list(failed_gpus)  # Start with actually failed GPUs
        num_to_pad = tp_spares - len(failed_gpus)

        if num_to_pad > 0:
            # Need to add more non-active ranks for parity
            # Find healthy GPUs to mark as non-active
            failed_set = set(failed_gpus)
            healthy_gpus = [i for i in range(tp_base) if i not in failed_set]

            # Take from the end of healthy GPUs (prefer keeping lower ranks active)
            gpus_to_deactivate = healthy_gpus[-num_to_pad:]
            non_active.extend(gpus_to_deactivate)

        non_active_ranks_per_dp[dp_rank] = sorted(non_active)

    return tp_spares, non_active_ranks_per_dp


def get_active_ranks_for_dp(
    dp_rank: int, tp_base: int, ntp_config: NonuniformTPConfig, cp_rank: int = 0, pp_rank: int = 0
) -> List[int]:
    """
    Get list of active (non-spare) local rank IDs for a given DP rank.

    Args:
        dp_rank: Data parallel rank
        tp_base: Base tensor parallel size
        ntp_config: NTP configuration

    Returns:
        List of local rank IDs that are active (not spare)
    """
    non_active = _ntp_get_non_active_ranks(ntp_config, dp_rank, cp_rank, pp_rank)
    if non_active is not None:
        # Use explicitly specified non-active ranks
        non_active_set = set(non_active)
        active_ranks = [i for i in range(tp_base) if i not in non_active_set]
    else:
        # Default: first (tp_base - tp_spares) ranks are active
        red_tp = tp_base - ntp_config.tp_spares
        active_ranks = list(range(red_tp))

    return active_ranks


# ======================================================================================
# Process Group Initialization for NTP
# ======================================================================================


def initialize_nonuniform_tp_process_groups(
    ntp_config: NonuniformTPConfig, exit_spares: bool = True
) -> bool:
    """
    Reconfigure TP and CP process groups for nonuniform tensor parallelism.

    Call this function after initialize_model_parallel() to enable NTP.
    Non-active (spare) ranks will exit after group creation.

    Args:
        ntp_config: NTP configuration containing tp_base, tp_spares, num_reduced_tp_dp_ranks,
                    and optionally non_active_ranks_per_dp
    """
    if ntp_config.tp_spares == 0:
        # No nonuniform TP, nothing to reconfigure
        return True

    tp_base = ntp_config.tp_base
    cp_size = parallel_state.get_context_parallel_world_size()
    rank = dist.get_rank()

    # Calculate which DP replicas use reduced TP
    dp_replica_size = tp_base * cp_size
    num_reduced_dp_ranks = ntp_config.num_reduced_tp_dp_ranks

    # Determine if current rank is in a reduced TP DP replica
    dp_replica_id = rank // dp_replica_size
    if dp_replica_id >= num_reduced_dp_ranks:
        # This rank is in a normal TP DP replica, no reconfiguration needed
        logger.info(
            "[NTP] Rank %s is in normal TP DP replica %s, skipping reconfiguration",
            rank,
            dp_replica_id,
        )
        return True

    local_rank_in_dp = rank % dp_replica_size
    cp_rank_in_dp = local_rank_in_dp // tp_base if cp_size > 1 else 0

    # This rank is in a reduced TP DP replica - need to reconfigure
    # Get active ranks for this DP replica (supports non-contiguous)
    active_local_ranks = get_active_ranks_for_dp(
        dp_replica_id, tp_base, ntp_config, cp_rank=cp_rank_in_dp
    )

    logger.info(
        "[NTP] Rank %s in DP replica %s: active_local_ranks=%s",
        rank,
        dp_replica_id,
        active_local_ranks,
    )

    if cp_size > 1:
        # With CP enabled: recreate TP, CP, and TP-CP groups
        dp_replica_start = dp_replica_id * dp_replica_size

        # Create new TP groups (one per CP slice in this DP replica)
        for cp_rank in range(cp_size):
            cp_slice_start = dp_replica_start + cp_rank * tp_base
            tp_group_ranks = [cp_slice_start + local_tp for local_tp in active_local_ranks]
            tp_group = dist.new_group(ranks=tp_group_ranks)

            if rank in tp_group_ranks:
                parallel_state._TENSOR_MODEL_PARALLEL_GROUP = tp_group
                parallel_state._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = tp_group_ranks
                parallel_state._MODEL_PARALLEL_GROUP = tp_group
                parallel_state._MODEL_PARALLEL_GLOBAL_RANKS = tp_group_ranks
                logger.info("[NTP] Rank %s created TP group: %s", rank, tp_group_ranks)

        # Create new CP groups (one per active TP position)
        for tp_rank_in_slice in active_local_ranks:
            cp_group_ranks = [
                dp_replica_start + tp_rank_in_slice + i * tp_base for i in range(cp_size)
            ]
            cp_group = dist.new_group(ranks=cp_group_ranks)

            if rank in cp_group_ranks:
                parallel_state._CONTEXT_PARALLEL_GROUP = cp_group
                parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS = cp_group_ranks
                logger.info("[NTP] Rank %s created CP group: %s", rank, cp_group_ranks)

        # Update TENSOR_AND_CONTEXT_PARALLEL_GROUP
        tp_rank_in_slice = local_rank_in_dp % tp_base
        if tp_rank_in_slice in active_local_ranks:
            tp_cp_group_ranks = []
            for cp_r in range(cp_size):
                for active_tp in active_local_ranks:
                    tp_cp_group_ranks.append(dp_replica_start + cp_r * tp_base + active_tp)
            tp_cp_group = dist.new_group(ranks=tp_cp_group_ranks)
            parallel_state._TENSOR_AND_CONTEXT_PARALLEL_GROUP = tp_cp_group
            logger.info("[NTP] Rank %s created TP-CP group: %s", rank, tp_cp_group_ranks)
        else:
            # Non-active (spare) rank - exit
            logger.info("[NTP] Rank %s is a spare rank with CP, exiting", rank)
            if exit_spares:
                sys.exit(0)
            return False
    else:
        # No CP: simpler case
        dp_replica_start = dp_replica_id * dp_replica_size
        tp_group_ranks = [dp_replica_start + local_tp for local_tp in active_local_ranks]

        if rank in tp_group_ranks:
            tp_group = dist.new_group(ranks=tp_group_ranks)
            parallel_state._TENSOR_MODEL_PARALLEL_GROUP = tp_group
            parallel_state._MODEL_PARALLEL_GROUP = tp_group
            parallel_state._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = tp_group_ranks
            parallel_state._MODEL_PARALLEL_GLOBAL_RANKS = tp_group_ranks
            logger.info("[NTP] Rank %s created TP group: %s", rank, tp_group_ranks)
        else:
            # Non-active (spare) rank - exit
            logger.info("[NTP] Rank %s is a spare rank, exiting", rank)
            if exit_spares:
                sys.exit(0)
            return False

    return True


# ======================================================================================
# Parameter Resharding for NTP
# ======================================================================================


def ntp_map(module: torch.nn.Module, ntp_config: NonuniformTPConfig, num_shards: int):
    """
    Initialize TP-sharded params with mapping between healthy and unhealthy TP sizes.

    Only healthy (full TP) ranks need send_splits and recv_splits to know how to reshard
    parameters when synchronizing with unhealthy (reduced TP) ranks.
    Unhealthy ranks synchronize directly without resharding.

    Args:
        module: Module containing parameters to initialize (e.g., self_attention or mlp)
        ntp_config: NTP configuration containing tp_base and tp_spares
        num_shards: Number of shards (e.g., num_attention_heads or ffn_hidden_size)
    """
    if ntp_config.tp_spares == 0:
        # No nonuniform TP, skip initialization
        return

    # Determine which ranks are active (non-spare) for the current DP rank
    rank = dist.get_rank()
    dp_rank = parallel_state.get_data_parallel_rank()
    cp_rank = parallel_state.get_context_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    logger.debug(
        f"[NTP] Rank {rank} [DP {dp_rank}, CP {cp_rank}, PP {pp_rank}] "
        f"ntp_map called with module={type(module).__name__}, num_shards={num_shards}"
    )

    # Check if this (dp, cp, pp) combination has non-active ranks specified
    # If it does, it's an unhealthy rank that uses reduced TP
    if _ntp_get_non_active_ranks(ntp_config, dp_rank, cp_rank, pp_rank) is not None:
        # This is an unhealthy rank with reduced TP - skip
        logger.debug(
            "[NTP] Rank %s [DP %s, CP %s, PP %s] Unhealthy rank, skipping",
            rank,
            dp_rank,
            cp_rank,
            pp_rank,
        )
        return

    # This is a healthy rank (full TP) - it needs send/recv splits to communicate
    # with unhealthy ranks that have reduced TP
    logger.debug(
        "[NTP] Rank %s [DP %s] Setting up send/recv splits for healthy rank", rank, dp_rank
    )

    for param in module.parameters():
        # Handle both tensor parallel parameters and vocabulary-parallel parameters that only
        # carry partition_dim metadata.
        if (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel) or (
            hasattr(param, 'partition_dim') and not hasattr(param, 'tensor_model_parallel')
        ):
            # For healthy ranks, compute send/recv splits for communication with unhealthy ranks
            # We need to know how to reshard to match the reduced TP size
            reduced_tp_size = ntp_config.tp_base - ntp_config.tp_spares

            shard_ids = torch.arange(num_shards)
            # Partitions for reduced TP (what unhealthy ranks have)
            sync_partitions = list(shard_ids.chunk(reduced_tp_size))

            # Full partitions for healthy ranks (tp_base ranks)
            comp_partitions = sync_partitions + [
                torch.empty(int(len(shard_ids) / ntp_config.tp_base), dtype=torch.int)
                for _ in range(ntp_config.tp_spares)
            ]

            # Build comp_2_sync: for spare positions, which reduced TP ranks do they map to
            comp_2_sync = [[] for _ in range(ntp_config.tp_base)]
            sync_part_idx = 0

            for spare_part_idx in range(reduced_tp_size, ntp_config.tp_base):
                for shard_part_idx in range(len(comp_partitions[spare_part_idx])):
                    # Take the last shard from the current reduced TP rank
                    comp_partitions[spare_part_idx][shard_part_idx] = comp_partitions[
                        sync_part_idx
                    ][-1]
                    comp_partitions[sync_part_idx] = comp_partitions[sync_part_idx][:-1]
                    comp_2_sync[spare_part_idx].append(sync_part_idx)
                    sync_part_idx = (sync_part_idx + 1) % reduced_tp_size

            # Compute param_splits: how many shards each rank sends to each other rank
            param_splits = [
                torch.bincount(torch.tensor(c2s, dtype=torch.int), minlength=ntp_config.tp_base)
                for c2s in comp_2_sync
            ]

            shard_size = int(param.shape[param.partition_dim] * ntp_config.tp_base / len(shard_ids))
            send_splits = [(p_split * shard_size).tolist() for p_split in param_splits]
            recv_splits = [
                [send_splits[send_idx][recv_idx] for send_idx in range(len(send_splits))]
                for recv_idx in range(ntp_config.tp_base)
            ]
            param.send_splits = send_splits
            param.recv_splits = recv_splits
            logger.debug(
                f"[NTP] Rank {rank} [DP {dp_rank}] Set send_splits and recv_splits "
                f"on parameter id={id(param)}, shape={param.shape}"
            )


def ntp_init(layer: torch.nn.Module, ntp_config: NonuniformTPConfig):
    """
    Initialize nonuniform TP mappings for a TransformerLayer.

    This should be called after the layer is created to set up the send_splits
    and recv_splits attributes on tensor-parallel parameters.

    Args:
        layer: TransformerLayer instance
        ntp_config: NTP configuration containing tp_base and tp_spares
    """
    if ntp_config.tp_spares == 0:
        # No nonuniform TP, skip initialization
        return

    # Initialize self-attention parameters
    if hasattr(layer, 'self_attention'):
        ntp_map(layer.self_attention, ntp_config, layer.self_attention.config.num_attention_heads)

    # Initialize MLP parameters
    if hasattr(layer, 'mlp'):
        ntp_map(layer.mlp, ntp_config, layer.mlp.config.ffn_hidden_size)


def _ntp_start_post_sync_grad_reshard(
    params: List[torch.nn.Parameter], ntp_config: NonuniformTPConfig
):
    """Launch async all-to-all that scatters reduced side grads back to extra ranks."""
    if ntp_config.tp_spares == 0 or not _ntp_can_query_parallel_state():
        return []
    if _ntp_current_rank_is_reduced_dp(ntp_config):
        return []
    if parallel_state.get_tensor_model_parallel_world_size() != ntp_config.tp_base:
        return []

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    reduced_tp_size = ntp_config.tp_base - ntp_config.tp_spares
    handles = []

    for param in params:
        if not _ntp_param_can_reshard(param):
            continue

        if tp_rank < reduced_tp_size:
            if not hasattr(param, 'side_grad') or param.side_grad is None:
                raise RuntimeError(
                    "NTP core rank is missing side_grad storage for a tensor-parallel param"
                )
            input_tensors = [
                _ntp_empty_like_partition(param, dtype=param.side_grad.dtype)
                for _ in range(reduced_tp_size)
            ] + _ntp_split_for_all_to_all(
                param.side_grad,
                param.recv_splits[tp_rank][-ntp_config.tp_spares :],
                param.partition_dim,
            )
            output_tensors = [
                _ntp_empty_like_partition(param, dtype=param.side_grad.dtype)
                for _ in range(ntp_config.tp_base)
            ]
        else:
            input_tensors = [
                _ntp_empty_like_partition(param, dtype=param.main_grad.dtype)
                for _ in range(ntp_config.tp_base)
            ]
            output_tensors = _ntp_split_views_for_all_to_all(
                param.main_grad, param.send_splits[tp_rank], param.partition_dim
            )

        handles.append(
            _ntp_all_to_all(output_tensors, input_tensors, group=tp_group, async_op=True)
        )

    return handles


# ======================================================================================
# NTP-aware ParamAndGradBuffer
# ======================================================================================


class NonuniformTPParamAndGradBucketGroup(_ParamAndGradBucketGroup):
    """
    NTP-aware version of _ParamAndGradBucketGroup.
    Skips gradient synchronization for spare GPUs.
    """

    def __init__(self, *args, ntp_config: Optional[NonuniformTPConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ntp_config = ntp_config or NonuniformTPConfig()
        self.ntp_post_sync_state = None

    def _wait_ntp_reshard_handles(self):
        """Wait for NTP all-to-all reshard work touching params in this bucket group."""
        for bucket in self.buckets:
            for param in bucket.params:
                handle = getattr(param, 'ntp_reshard_handle', None)
                if handle is not None:
                    handle.wait()
                    param.ntp_reshard_handle = None

    def _start_ntp_post_sync_reshard(self):
        """Start async post-DP-sync gradient reshard for this bucket group."""
        handles = []
        for bucket in self.buckets:
            handles.extend(_ntp_start_post_sync_grad_reshard(bucket.params_list, self.ntp_config))
        return handles

    def _record_ntp_post_sync_handles(self, handles):
        """Track post-sync handles and wait for all of them at the last bucket group."""
        state = self.ntp_post_sync_state
        if state is None:
            for handle in handles:
                handle.wait()
            return

        state['handles'].extend(handles)
        if self is state['last_bucket_group']:
            try:
                for handle in state['handles']:
                    handle.wait()
            finally:
                state['handles'] = []

    def start_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Start DP grad sync after any pending NTP reshard for this bucket is complete."""
        self._wait_ntp_reshard_handles()
        if not _ntp_current_rank_should_dp_sync(self.ntp_config):
            self.grad_reduce_handle = None
            return
        return super().start_grad_sync(force_all_reduce=force_all_reduce)

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Finish DP grad sync and launch async post-sync NTP gradient reshard."""
        self.param_gather_dispatched = False
        self._wait_ntp_reshard_handles()
        if not _ntp_current_rank_should_dp_sync(self.ntp_config):
            handles = self._start_ntp_post_sync_reshard()
            self._record_ntp_post_sync_handles(handles)
            return
        result = super().finish_grad_sync(force_all_reduce=force_all_reduce)
        handles = self._start_ntp_post_sync_reshard()
        self._record_ntp_post_sync_handles(handles)
        return result

    def register_grad_ready(
        self, param: torch.nn.Parameter, force_all_reduce: Optional[bool] = False
    ):
        """Skip DP-ready bookkeeping on ranks that are folded into core TP ranks."""
        if not _ntp_current_rank_should_dp_sync(self.ntp_config):
            return
        return super().register_grad_ready(param, force_all_reduce=force_all_reduce)


def _compute_default_per_buffer_param_layout(
    params: List[torch.nn.Parameter],
    bucket_size: Optional[int],
    ddp_config: DistributedDataParallelConfig,
    data_parallel_world_size: int,
) -> PerBufferParamLayout:
    """Compute the default DDP layout locally so generic buffer code is untouched."""

    def _does_param_require_new_bucket(param):
        return getattr(param, "shared_embedding", False) and ddp_config.use_distributed_optimizer

    param_index_map = {}
    bucket_indices = []
    per_bucket_numel_unpadded = []

    param_start_index = 0
    bucket_start_index = 0
    bucket_params = set()
    bucket_id = 0

    def _finalize_bucket(param_end_index, current_bucket_start_index):
        nonlocal bucket_params, bucket_id
        per_bucket_numel_unpadded.append(param_end_index - current_bucket_start_index)
        if ddp_config.use_distributed_optimizer:
            bucket_end_index = pad_bucket_end(
                param_end_index,
                data_parallel_world_size,
                ddp_config.pad_buckets_for_high_nccl_busbw,
            )
        else:
            bucket_end_index = param_end_index
        bucket_indices.append((current_bucket_start_index, bucket_end_index))
        bucket_params = set()
        bucket_id += 1
        return bucket_end_index

    for param in params[::-1]:
        if ddp_config.use_distributed_optimizer:
            param_start_index = pad_param_start(param_start_index)

        if _does_param_require_new_bucket(param) and len(bucket_params) > 0:
            bucket_start_index = _finalize_bucket(param_start_index, bucket_start_index)
            param_start_index = bucket_start_index

        param_end_index = param_start_index + param.data.nelement()
        param_index_map[param] = (param_start_index, param_end_index, bucket_id)
        bucket_params.add(param)

        if (
            bucket_size is not None and (param_end_index - bucket_start_index) >= bucket_size
        ) or _does_param_require_new_bucket(param):
            bucket_start_index = _finalize_bucket(param_end_index, bucket_start_index)
            param_start_index = bucket_start_index
        else:
            param_start_index = param_end_index

    if len(bucket_params) > 0:
        _finalize_bucket(param_start_index, bucket_start_index)

    return PerBufferParamLayout(
        param_index_map=param_index_map,
        bucket_indices=bucket_indices,
        per_bucket_numel_unpadded=per_bucket_numel_unpadded,
    )


class NonuniformTPParamAndGradBuffer(_ParamAndGradBuffer):
    """
    NTP-aware version of _ParamAndGradBuffer.
    Adjusts buffer sizes and keeps optimizer-visible main_grad ranges contiguous.
    """

    def __init__(self, *args, ntp_config: Optional[NonuniformTPConfig] = None, **kwargs):
        self.ntp_config = ntp_config or NonuniformTPConfig()
        ddp_config = args[0] if len(args) > 0 else kwargs['ddp_config']
        params_with_names = args[3] if len(args) > 3 else kwargs['params_with_names']
        data_parallel_group = args[4] if len(args) > 4 else kwargs['data_parallel_group']
        bucket_size = args[5] if len(args) > 5 else kwargs['bucket_size']
        param_indices = args[8] if len(args) > 8 else kwargs['param_indices']
        params = [param for param, _ in params_with_names]

        if self.ntp_config.tp_spares > 0:
            param_layout = _compute_ntp_per_buffer_param_layout(
                params,
                bucket_size,
                data_parallel_group.size(),
                ddp_config,
                self.ntp_config,
                param_indices,
            )
        else:
            param_layout = _compute_default_per_buffer_param_layout(
                params, bucket_size, ddp_config, data_parallel_group.size()
            )
        self._ntp_side_grad_index_map = param_layout.side_grad_index_map

        self._init_with_param_layout(*args, param_layout=param_layout, **kwargs)

        if self.ntp_config.tp_spares > 0:
            for param in self.params:
                if not _ntp_should_expand_param_grad(param, self.ntp_config):
                    continue
                side_range = self._ntp_side_grad_index_map.get(param)
                if side_range is None:
                    continue
                side_start, side_end, _ = side_range
                side_shape = list(param.data.shape)
                side_shape[param.partition_dim] = _ntp_extra_partition_dim(param, self.ntp_config)
                assert torch.Size(side_shape).numel() == side_end - side_start
                param.side_grad = self.grad_data[side_start:side_end].view(side_shape)

    def _init_with_param_layout(self, *args, param_layout: PerBufferParamLayout, **kwargs):
        ddp_config = args[0] if len(args) > 0 else kwargs['ddp_config']
        param_dtype = args[1] if len(args) > 1 else kwargs['param_dtype']
        grad_dtype = args[2] if len(args) > 2 else kwargs['grad_dtype']
        params_with_names = args[3] if len(args) > 3 else kwargs['params_with_names']
        data_parallel_group = args[4] if len(args) > 4 else kwargs['data_parallel_group']
        param_to_name = args[6] if len(args) > 6 else kwargs['param_to_name']
        gradient_scaling_factor = args[7] if len(args) > 7 else kwargs['gradient_scaling_factor']
        param_indices = args[8] if len(args) > 8 else kwargs['param_indices']
        nccl_ub = args[9] if len(args) > 9 else kwargs['nccl_ub']
        pg_collection = args[10] if len(args) > 10 else kwargs.get('pg_collection')

        if pg_collection is None:
            self.dp_cp_group = parallel_state.get_data_and_context_parallel_group(
                with_context_parallel=True
            )
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            assert hasattr(pg_collection, 'tp') and hasattr(pg_collection, 'dp_cp')
            self.dp_cp_group = pg_collection.dp_cp
            self.tp_group = pg_collection.tp

        self.ddp_config = ddp_config
        self.params = [param for (param, _) in params_with_names]
        self.param_indices = param_indices

        unique_params = set()
        for param, _ in params_with_names:
            assert param not in unique_params
            unique_params.add(param)

        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = self.data_parallel_group.size()
        self.gradient_scaling_factor = gradient_scaling_factor
        self.nccl_ub = nccl_ub

        self.buckets = []
        self.param_to_bucket = {}
        self.param_index_map = param_layout.param_index_map
        self.bucket_indices = param_layout.bucket_indices
        per_bucket_numel_unpadded = param_layout.per_bucket_numel_unpadded

        self.has_nvfp4_params = any(pgb.is_nvfp4tensor(p) for p in self.params)
        self.nvfp4_packed_param_index_map = None
        self.nvfp4_packed_bucket_indices = None
        if self.has_nvfp4_params:
            self._compute_nvfp4_packed_layout(params_with_names)

        self.numel = self.bucket_indices[-1][1]
        self.numel_unpadded = sum(per_bucket_numel_unpadded)
        if self.has_nvfp4_params:
            self.nvfp4_packed_numel = self.nvfp4_packed_bucket_indices[-1][1]

        assert self.numel_unpadded <= self.numel
        if self.has_nvfp4_params:
            assert self.nvfp4_packed_numel_unpadded <= self.nvfp4_packed_numel
        if self.ddp_config.use_distributed_optimizer:
            assert self.numel % self.data_parallel_world_size == 0
            if self.has_nvfp4_params:
                assert self.nvfp4_packed_numel % self.data_parallel_world_size == 0
        else:
            assert self.numel == self.numel_unpadded

        self.param_data = None
        self.grad_data = None
        self.extra_main_grads = []

        if self.nccl_ub:
            pgb.nccl_allocator.init()
            pool = pgb.nccl_allocator.create_nccl_mem_pool(
                symmetric=not self.ddp_config.disable_symmetric_registration
            )
            mem_alloc_context = functools.partial(
                pgb.nccl_allocator.nccl_mem,
                pool,
                group=self.data_parallel_group,
                symmetric=not self.ddp_config.disable_symmetric_registration,
            )
            torch.distributed.barrier()
            tmp_warmup_tensor = torch.zeros([1], device="cuda")
            torch.distributed.all_reduce(tmp_warmup_tensor, group=self.data_parallel_group)
            torch.distributed.barrier()
        else:
            mem_alloc_context = nullcontext

        with mem_alloc_context():
            if self.ddp_config.use_distributed_optimizer and any(
                pgb.is_mxfp8tensor(p) for p in self.params
            ):
                self.shared_buffer = torch.zeros(
                    self.numel,
                    dtype=self.grad_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
                if self.grad_dtype == torch.float32:
                    self.param_data = self.shared_buffer[: math.ceil(self.numel / 2)].view(
                        torch.bfloat16
                    )
                else:
                    self.param_data = self.shared_buffer
                self.grad_data = self.shared_buffer
            else:
                if self.ddp_config.use_distributed_optimizer:
                    numel = self.nvfp4_packed_numel if self.has_nvfp4_params else self.numel
                    self.param_data = torch.zeros(
                        numel,
                        dtype=self.param_dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                self.grad_data = torch.zeros(
                    self.numel,
                    dtype=self.grad_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )

        self.grad_data_size = 0
        self.param_data_size = 0
        self.param_data_cpu = None

        def _create_bucket(bucket_id, bucket_params, bucket_params_with_extra_main_grads):
            bucket_start_index, bucket_end_index = self.bucket_indices[bucket_id]
            if self.has_nvfp4_params:
                nvfp4_packed_start_index, nvfp4_packed_end_index = self.nvfp4_packed_bucket_indices[
                    bucket_id
                ]
            else:
                nvfp4_packed_start_index, nvfp4_packed_end_index = None, None
            return self._new_bucket(
                bucket_params=bucket_params,
                start_index=bucket_start_index,
                end_index=bucket_end_index,
                numel_unpadded=per_bucket_numel_unpadded[bucket_id],
                bucket_id=bucket_id,
                bucket_params_with_extra_main_grads=bucket_params_with_extra_main_grads,
                nvfp4_packed_start_index=nvfp4_packed_start_index,
                nvfp4_packed_end_index=nvfp4_packed_end_index,
            )

        bucket_params = []
        bucket_params_with_extra_main_grads = []
        cur_bucket_id = 0
        for param, param_name in params_with_names[::-1]:
            param_start_index, _, bucket_id = self.param_index_map[param]
            nvfp4_packed_param_start_index = None
            if self.has_nvfp4_params:
                nvfp4_packed_param_start_index, _, _ = self.nvfp4_packed_param_index_map[param]

            if not self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag or not pgb.is_mxfp8tensor(
                param
            ):
                if self.param_data is not None:
                    if pgb.is_nvfp4tensor(param):
                        from ..fp4_utils import modify_nvfp4_rowwise_storage

                        packed_shape = pgb.get_nvfp4_rowwise_packed_shape(param.data.shape)
                        rowwise_bytes_view = self._get(
                            packed_shape,
                            nvfp4_packed_param_start_index,
                            buffer_type=pgb.BufferType.PARAM,
                        )
                        modify_nvfp4_rowwise_storage(param, rowwise_bytes_view)
                    elif pgb.is_float8tensor(param):
                        new_param_data = self._get(
                            param.data.shape,
                            (
                                nvfp4_packed_param_start_index
                                if self.has_nvfp4_params
                                else param_start_index
                            ),
                            buffer_type=pgb.BufferType.PARAM,
                        )
                        pgb.modify_underlying_storage(param, new_param_data)
                    else:
                        new_param_data = self._get(
                            param.data.shape,
                            (
                                nvfp4_packed_param_start_index
                                if self.has_nvfp4_params
                                else param_start_index
                            ),
                            buffer_type=pgb.BufferType.PARAM,
                        )
                        old_param_data = param.data
                        param.data = new_param_data
                        assert old_param_data._base is None
                        param.data.detach().copy_(old_param_data)
                        del old_param_data

            param.main_grad = self._get(
                param.data.shape, param_start_index, buffer_type=pgb.BufferType.GRAD
            )

            promote_main_grads_to_higher_precision = False
            for param_name_pattern in ddp_config.param_name_patterns_for_fp32_local_accumulation:
                if fnmatch.fnmatch(param_name, param_name_pattern) or param_name_pattern == 'all':
                    log_on_each_pipeline_stage(
                        logger,
                        logging.INFO,
                        (
                            f"Matched {param_name} with '{param_name_pattern}'; promoting "
                            f"main_grad.type from {param.main_grad.dtype} to torch.float32!"
                        ),
                        tp_group=self.tp_group,
                        dp_cp_group=self.dp_cp_group,
                    )
                    promote_main_grads_to_higher_precision = True
                    break
            if promote_main_grads_to_higher_precision:
                param.main_grad_copy_in_grad_buffer = param.main_grad
                param.main_grad = torch.empty_like(param.main_grad, dtype=torch.float32)
                self.extra_main_grads.append(param.main_grad)

            if bucket_id != cur_bucket_id:
                self.buckets.append(
                    _create_bucket(
                        cur_bucket_id, bucket_params, bucket_params_with_extra_main_grads
                    )
                )
                bucket_params = []
                bucket_params_with_extra_main_grads = []
                assert cur_bucket_id + 1 == len(self.buckets)
                assert bucket_id == cur_bucket_id + 1
                cur_bucket_id = bucket_id

            bucket_params.append(param)
            if promote_main_grads_to_higher_precision:
                bucket_params_with_extra_main_grads.append(param)

        if len(bucket_params) > 0:
            self.buckets.append(
                _create_bucket(cur_bucket_id, bucket_params, bucket_params_with_extra_main_grads)
            )

        log_strs = [
            f"Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}"
        ]
        for index, bucket in enumerate(self.buckets):
            numel = sum(param.data.nelement() for param in bucket.params_list)
            log_strs.append(
                f"Params for bucket {index + 1} ({numel} elements, "
                f"{bucket.grad_data.nelement()} padded size, "
                f"{len(bucket.params_with_extra_main_grads)} param(s) with extra main_grads):"
            )
            for param in bucket.params_list:
                log_strs.append(f"\t{param_to_name[param]} ({param.main_grad.dtype=})")
        log_on_each_pipeline_stage(
            logger,
            logging.INFO,
            "\n".join(log_strs),
            tp_group=self.tp_group,
            dp_cp_group=self.dp_cp_group,
        )

    def _compute_nvfp4_packed_layout(self, params_with_names):
        def _pad_start_of_param(param_start_index: int) -> int:
            if self.ddp_config.use_distributed_optimizer:
                return pad_param_start(param_start_index)
            return param_start_index

        def _pad_end_of_bucket(bucket_end_index: int) -> int:
            if self.ddp_config.use_distributed_optimizer:
                return pad_bucket_end(
                    bucket_end_index,
                    self.data_parallel_world_size,
                    self.ddp_config.pad_buckets_for_high_nccl_busbw,
                )
            return bucket_end_index

        self.nvfp4_packed_param_index_map = {}
        self.nvfp4_packed_bucket_indices = []
        nvfp4_packed_per_bucket_numel_unpadded = []

        packed_param_start = 0
        packed_bucket_start = 0
        cur_bucket_id = 0

        for param, _ in params_with_names[::-1]:
            _, _, bucket_id = self.param_index_map[param]
            param_numel = param.data.nelement()
            packed_param_start = _pad_start_of_param(packed_param_start)

            if bucket_id != cur_bucket_id:
                nvfp4_packed_per_bucket_numel_unpadded.append(
                    packed_param_start - packed_bucket_start
                )
                packed_bucket_end = _pad_end_of_bucket(packed_param_start)
                self.nvfp4_packed_bucket_indices.append((packed_bucket_start, packed_bucket_end))
                packed_bucket_start = packed_bucket_end
                packed_param_start = packed_bucket_start
                cur_bucket_id = bucket_id

            if pgb.is_nvfp4tensor(param):
                assert (
                    param_numel % 2 == 0
                ), f"NVFP4 requires even numel for packing, got {param_numel}"
                packed_numel = param_numel // 2
            else:
                packed_numel = param_numel

            packed_param_end = packed_param_start + packed_numel
            self.nvfp4_packed_param_index_map[param] = (
                packed_param_start,
                packed_param_end,
                bucket_id,
            )
            packed_param_start = packed_param_end

        if packed_param_start > packed_bucket_start:
            nvfp4_packed_per_bucket_numel_unpadded.append(packed_param_start - packed_bucket_start)
            packed_bucket_end = _pad_end_of_bucket(packed_param_start)
            self.nvfp4_packed_bucket_indices.append((packed_bucket_start, packed_bucket_end))

        assert len(self.nvfp4_packed_bucket_indices) == len(self.bucket_indices), (
            f"Packed bucket count ({len(self.nvfp4_packed_bucket_indices)}) != "
            f"primary bucket count ({len(self.bucket_indices)})"
        )
        self.nvfp4_packed_numel_unpadded = sum(nvfp4_packed_per_bucket_numel_unpadded)


# ======================================================================================
# NTP-aware DistributedDataParallel
# ======================================================================================


class NonuniformTPDistributedDataParallel(DistributedDataParallel):
    """
    NTP-aware version of DistributedDataParallel.
    Adds gradient synchronization logic for spare GPUs.
    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        disable_bucketing: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
        ntp_config: Optional[NonuniformTPConfig] = None,
        full_param_layout: Optional[FullParamLayout] = None,
    ):
        self.ntp_config = ntp_config or NonuniformTPConfig()

        def _call_parent_init():
            parent_kwargs = {
                'config': config,
                'ddp_config': ddp_config,
                'module': module,
                'disable_bucketing': disable_bucketing,
                'pg_collection': pg_collection,
            }
            if (
                'full_param_layout'
                in inspect.signature(DistributedDataParallel.__init__).parameters
            ):
                parent_kwargs['full_param_layout'] = full_param_layout
            elif full_param_layout is not None:
                logger.warning(
                    "Ignoring full_param_layout because this DDP base does not accept it"
                )
            super(NonuniformTPDistributedDataParallel, self).__init__(**parent_kwargs)

        # Use NTP-aware buffer class
        if self.ntp_config.tp_spares > 0:
            # DDP imports _ParamAndGradBuffer into its module namespace, so patch that binding
            # while the parent constructor allocates buffers.
            original_buffer_class = ddp_module._ParamAndGradBuffer
            ddp_module._ParamAndGradBuffer = functools.partial(
                NonuniformTPParamAndGradBuffer, ntp_config=self.ntp_config
            )
            try:
                _call_parent_init()
            finally:
                ddp_module._ParamAndGradBuffer = original_buffer_class
            self._wrap_bucket_groups_for_ntp()
        else:
            _call_parent_init()

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Ensure all first-batch reductions are launched before post-sync reshards."""
        if self.ntp_config.tp_spares > 0 and self.ddp_config.overlap_grad_reduce:
            for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
                if bucket_group.is_first_batch and bucket_group.grad_reduce_handle is None:
                    bucket_group.start_grad_sync(force_all_reduce=force_all_reduce)
        return super().finish_grad_sync(force_all_reduce=force_all_reduce)

    def _wrap_bucket_groups_for_ntp(self):
        """Replace DDP bucket groups with NTP-aware groups and rebuild param lookup."""

        def wrap_groups(bucket_groups):
            wrapped_groups = []
            old_to_new = {}
            for bucket_group in bucket_groups:
                if (
                    self.ddp_config.use_distributed_optimizer
                    or self.ddp_config.overlap_param_gather
                ):
                    collective_group = bucket_group.intra_distributed_optimizer_instance_group
                    collective_group_size = bucket_group.intra_distributed_optimizer_instance_size
                else:
                    collective_group = bucket_group.data_parallel_group
                    collective_group_size = bucket_group.data_parallel_group.size()

                wrapped_group = NonuniformTPParamAndGradBucketGroup(
                    bucket_group.buckets,
                    bucket_group.ddp_config,
                    collective_group,
                    collective_group_size,
                    ntp_config=self.ntp_config,
                )
                if hasattr(bucket_group, 'inter_distributed_optimizer_instance_group'):
                    wrapped_group.inter_distributed_optimizer_instance_group = (
                        bucket_group.inter_distributed_optimizer_instance_group
                    )
                if hasattr(bucket_group, 'communication_stream'):
                    wrapped_group.communication_stream = bucket_group.communication_stream
                old_to_new[bucket_group] = wrapped_group
                wrapped_groups.append(wrapped_group)

            for bucket_group, wrapped_group in old_to_new.items():
                next_group = getattr(bucket_group, 'next_param_gather_bucket_group', None)
                if next_group is not None:
                    wrapped_group.next_param_gather_bucket_group = old_to_new[next_group]

            return wrapped_groups

        self.bucket_groups = wrap_groups(self.bucket_groups)
        self.expert_parallel_bucket_groups = wrap_groups(self.expert_parallel_bucket_groups)
        self.param_to_bucket_group = {}
        for bucket_groups in [self.bucket_groups, self.expert_parallel_bucket_groups]:
            for bucket_group in bucket_groups:
                for bucket in bucket_group.buckets:
                    for param in bucket.params_list:
                        self.param_to_bucket_group[param] = bucket_group

        all_bucket_groups = self.bucket_groups + self.expert_parallel_bucket_groups
        if all_bucket_groups:
            post_sync_state = {'handles': [], 'last_bucket_group': all_bucket_groups[-1]}
            for bucket_group in all_bucket_groups:
                bucket_group.ntp_post_sync_state = post_sync_state

    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        """
        Override to add NTP gradient synchronization between spare and core GPUs.
        """

        def ntp_hook(*unused):
            if is_graph_capturing():
                return

            bucket_group = self.param_to_bucket_group.get(param)
            is_last_microbatch = bucket_group is None or bucket_group.is_last_microbatch
            if param in self.param_to_bucket_group:
                assert param.requires_grad
                if self.ddp_config.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data)
                param.grad = None

            # Add NTP-specific logic
            if (
                self.ntp_config.tp_spares > 0
                and _ntp_param_can_reshard(param)
                and is_last_microbatch
                and not _ntp_current_rank_is_reduced_dp(self.ntp_config)
                and parallel_state.get_tensor_model_parallel_world_size() == self.ntp_config.tp_base
            ):
                tp_rank = parallel_state.get_tensor_model_parallel_rank()
                reduced_tp_size = self.ntp_config.tp_base - self.ntp_config.tp_spares

                if tp_rank < reduced_tp_size:
                    # Core GPU: receive grads from spare GPUs
                    input = [
                        _ntp_empty_like_partition(param, dtype=param.side_grad.dtype)
                        for _ in range(parallel_state.get_tensor_model_parallel_world_size())
                    ]
                    # Split side_grad and send to core GPUs
                    output = [
                        _ntp_empty_like_partition(param, dtype=param.side_grad.dtype)
                        for _ in range(reduced_tp_size)
                    ] + _ntp_split_views_for_all_to_all(
                        param.side_grad, param.recv_splits[tp_rank], dim=param.partition_dim
                    )[
                        -self.ntp_config.tp_spares :
                    ]
                else:
                    # Spare GPU: send grads to core GPUs
                    input = _ntp_split_for_all_to_all(
                        param.main_grad, param.send_splits[tp_rank], dim=param.partition_dim
                    )
                    output = [
                        _ntp_empty_like_partition(param, dtype=param.main_grad.dtype)
                        for _ in range(parallel_state.get_tensor_model_parallel_world_size())
                    ]

                try:
                    handle = _ntp_all_to_all(
                        output,
                        input,
                        group=parallel_state.get_tensor_model_parallel_group(),
                        async_op=True,
                    )
                    param.ntp_reshard_handle = handle
                except Exception as e:
                    logger.error("[NTP] Rank %s all_to_all error: %s", tp_rank, e)
                    input_contiguity = [i.is_contiguous() for i in input]
                    output_contiguity = [o.is_contiguous() for o in output]
                    logger.error(
                        "[NTP] Rank %s input element contiguity: %s", tp_rank, input_contiguity
                    )
                    logger.error(
                        "[NTP] Rank %s output element contiguity: %s", tp_rank, output_contiguity
                    )
                    raise e

            if param in self.param_to_bucket_group and self.ddp_config.overlap_grad_reduce:
                self.param_to_bucket_group[param].register_grad_ready(param, self.force_all_reduce)

        return ntp_hook
