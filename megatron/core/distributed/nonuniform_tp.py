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
    - NonuniformTPOptimizer to wrap your optimizer
    - Call initialize_nonuniform_tp_process_groups() after initialize_model_parallel()
"""

import functools
import logging
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .. import parallel_state
from ..optimizer.param_layout import (
    FullParamLayout,
    PerBufferParamLayout,
    pad_bucket_end,
    pad_param_start,
)
from ..process_groups_config import ProcessGroupCollection
from ..transformer.cuda_graphs import is_graph_capturing
from ..transformer.transformer_config import TransformerConfig
from . import distributed_data_parallel as ddp_module
from .distributed_data_parallel import DistributedDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import _ParamAndGradBucketGroup, _ParamAndGradBuffer

logger = logging.getLogger(__name__)


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

        param_end_index = param_start_index + _ntp_param_numel(param, ntp_config)
        param_index_map[param] = (param_start_index, param_end_index, bucket_id)
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

    def _wait_ntp_reshard_handles(self):
        """Wait for NTP all-to-all reshard work touching params in this bucket group."""
        for bucket in self.buckets:
            for param in bucket.params:
                handle = getattr(param, 'ntp_reshard_handle', None)
                if handle is not None:
                    handle.wait()
                    param.ntp_reshard_handle = None

    def start_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Start DP grad sync after any pending NTP reshard for this bucket is complete."""
        self._wait_ntp_reshard_handles()
        if not _ntp_current_rank_should_dp_sync(self.ntp_config):
            self.grad_reduce_handle = None
            return
        return super().start_grad_sync(force_all_reduce=force_all_reduce)

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Finish DP grad sync, treating folded-away healthy spare ranks as no-ops."""
        self.param_gather_dispatched = False
        self._wait_ntp_reshard_handles()
        if not _ntp_current_rank_should_dp_sync(self.ntp_config):
            self._copy_back_extra_main_grads()
            return
        return super().finish_grad_sync(force_all_reduce=force_all_reduce)

    def register_grad_ready(
        self, param: torch.nn.Parameter, force_all_reduce: Optional[bool] = False
    ):
        """Skip DP-ready bookkeeping on ranks that are folded into core TP ranks."""
        if not _ntp_current_rank_should_dp_sync(self.ntp_config):
            return
        return super().register_grad_ready(param, force_all_reduce=force_all_reduce)


class NonuniformTPParamAndGradBuffer(_ParamAndGradBuffer):
    """
    NTP-aware version of _ParamAndGradBuffer.
    Adjusts buffer sizes and splits gradients for NTP.
    """

    def __init__(self, *args, ntp_config: Optional[NonuniformTPConfig] = None, **kwargs):
        self.ntp_config = ntp_config or NonuniformTPConfig()
        if self.ntp_config.tp_spares > 0:
            ddp_config = args[0] if len(args) > 0 else kwargs['ddp_config']
            params_with_names = args[3] if len(args) > 3 else kwargs['params_with_names']
            data_parallel_group = args[4] if len(args) > 4 else kwargs['data_parallel_group']
            bucket_size = args[5] if len(args) > 5 else kwargs['bucket_size']
            param_indices = args[8] if len(args) > 8 else kwargs['param_indices']
            params = [param for param, _ in params_with_names]
            kwargs['param_layout'] = _compute_ntp_per_buffer_param_layout(
                params,
                bucket_size,
                data_parallel_group.size(),
                ddp_config,
                self.ntp_config,
                param_indices,
            )

        super().__init__(*args, **kwargs)

        if self.ntp_config.tp_spares > 0:
            for param in self.params:
                if not _ntp_should_expand_param_grad(param, self.ntp_config):
                    continue
                param_start_index, param_end_index, _ = self.param_index_map[param]
                main_numel = param.data.nelement()
                side_numel = param_end_index - param_start_index - main_numel
                if side_numel <= 0:
                    continue

                side_shape = list(param.data.shape)
                side_shape[param.partition_dim] = _ntp_extra_partition_dim(param, self.ntp_config)
                assert torch.Size(side_shape).numel() == side_numel
                side_start = param_start_index + main_numel
                param.side_grad = self.grad_data[side_start:param_end_index].view(side_shape)


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

        # Use NTP-aware buffer class
        if self.ntp_config.tp_spares > 0:
            # DDP imports _ParamAndGradBuffer into its module namespace, so patch that binding
            # while the parent constructor allocates buffers.
            original_buffer_class = ddp_module._ParamAndGradBuffer
            ddp_module._ParamAndGradBuffer = functools.partial(
                NonuniformTPParamAndGradBuffer, ntp_config=self.ntp_config
            )
            try:
                super().__init__(
                    config=config,
                    ddp_config=ddp_config,
                    module=module,
                    disable_bucketing=disable_bucketing,
                    pg_collection=pg_collection,
                    full_param_layout=full_param_layout,
                )
            finally:
                ddp_module._ParamAndGradBuffer = original_buffer_class
            self._wrap_bucket_groups_for_ntp()
        else:
            super().__init__(
                config=config,
                ddp_config=ddp_config,
                module=module,
                disable_bucketing=disable_bucketing,
                pg_collection=pg_collection,
                full_param_layout=full_param_layout,
            )

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
                empty_shape = list(param.shape)
                empty_shape[param.partition_dim] = 0
                tp_rank = parallel_state.get_tensor_model_parallel_rank()

                if tp_rank < self.ntp_config.tp_base - self.ntp_config.tp_spares:
                    # Core GPU: receive grads from spare GPUs
                    input = [
                        torch.empty(
                            empty_shape, device=param.device, dtype=param.side_grad.dtype
                        ).contiguous()
                        for _ in range(parallel_state.get_tensor_model_parallel_world_size())
                    ]
                    # Split side_grad and send to core GPUs
                    output = [
                        torch.empty(
                            empty_shape, device=param.device, dtype=param.side_grad.dtype
                        ).contiguous()
                        for _ in range(self.ntp_config.tp_base - self.ntp_config.tp_spares)
                    ] + [
                        t.contiguous()
                        for t in torch.split(
                            param.side_grad, param.recv_splits[tp_rank], dim=param.partition_dim
                        )
                    ][
                        -self.ntp_config.tp_spares :
                    ]
                else:
                    # Spare GPU: send grads to core GPUs
                    input = [
                        t.contiguous()
                        for t in torch.split(
                            param.main_grad, param.send_splits[tp_rank], dim=param.partition_dim
                        )
                    ]
                    output = [
                        torch.empty(
                            empty_shape, device=param.device, dtype=param.main_grad.dtype
                        ).contiguous()
                        for _ in range(parallel_state.get_tensor_model_parallel_world_size())
                    ]

                try:
                    handle = dist.all_to_all(
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


# ======================================================================================
# NTP-aware Optimizer Wrapper
# ======================================================================================


class NonuniformTPOptimizer:
    """
    Wrapper for optimizers to make gradients contiguous for NTP.
    """

    def __init__(self, optimizer, ntp_config: NonuniformTPConfig):
        self.optimizer = optimizer
        self.ntp_config = ntp_config

    def __getattr__(self, name):
        """Delegate attribute access to wrapped optimizer."""
        return getattr(self.optimizer, name)

    def prepare_grads(self, *args, **kwargs):
        """
        Override prepare_grads to make gradients contiguous for NTP.
        """
        # Call original prepare_grads if it exists
        if hasattr(self.optimizer, 'prepare_grads'):
            result = self.optimizer.prepare_grads(*args, **kwargs)
        else:
            result = False

        # Make gradients contiguous for NTP
        if self.ntp_config.tp_spares > 0:
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    if hasattr(param, 'main_grad') and param.main_grad is not None:
                        if not param.main_grad.is_contiguous():
                            param.grad = param.main_grad.contiguous()
                        else:
                            param.grad = param.main_grad

        return result
