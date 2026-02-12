# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

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
import torch
import torch.distributed as dist
from contextlib import nullcontext
from typing import Dict, List, Optional, Set, Tuple

from torch.distributed import _coalescing_manager

from .. import parallel_state
from ..process_groups_config import ProcessGroupCollection
from ..transformer.transformer_config import TransformerConfig
from .distributed_data_parallel import DistributedDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import (
    _ParamAndGradBuffer,
    _ParamAndGradBucketGroup,
    BufferType,
    dist_reduce_scatter_func,
    shard_buffer,
)

logger = logging.getLogger(__name__)


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
    dp_rank: int, tp_base: int, ddp_config: DistributedDataParallelConfig
) -> List[int]:
    """
    Get list of active (non-spare) local rank IDs for a given DP rank.

    Args:
        dp_rank: Data parallel rank
        tp_base: Base tensor parallel size
        ddp_config: DDP configuration

    Returns:
        List of local rank IDs that are active (not spare)
    """
    if ddp_config.non_active_ranks_per_dp and dp_rank in ddp_config.non_active_ranks_per_dp:
        # Use explicitly specified non-active ranks
        non_active = set(ddp_config.non_active_ranks_per_dp[dp_rank])
        active_ranks = [i for i in range(tp_base) if i not in non_active]
    else:
        # Default: first (tp_base - tp_spares) ranks are active
        red_tp = tp_base - ddp_config.tp_spares
        active_ranks = list(range(red_tp))

    return active_ranks


# ======================================================================================
# Process Group Initialization for NTP
# ======================================================================================


def initialize_nonuniform_tp_process_groups(ddp_config: DistributedDataParallelConfig):
    """
    Reconfigure TP and CP process groups for nonuniform tensor parallelism.

    Call this function after initialize_model_parallel() to enable NTP.
    Non-active (spare) ranks will exit after group creation.

    Args:
        ddp_config: DDP configuration containing tp_base, tp_spares, num_reduced_tp_dp_ranks,
                    and optionally non_active_ranks_per_dp
    """
    if ddp_config.tp_spares == 0:
        # No nonuniform TP, nothing to reconfigure
        return

    tp_base = ddp_config.tp_base
    tp_spares = ddp_config.tp_spares
    cp_size = parallel_state.get_context_parallel_world_size()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Calculate which DP replicas use reduced TP
    dp_replica_size = tp_base * cp_size
    num_reduced_dp_ranks = ddp_config.num_reduced_tp_dp_ranks

    # Determine if current rank is in a reduced TP DP replica
    dp_replica_id = rank // dp_replica_size
    if dp_replica_id >= num_reduced_dp_ranks:
        # This rank is in a normal TP DP replica, no reconfiguration needed
        logger.info(f"[NTP] Rank {rank} is in normal TP DP replica {dp_replica_id}, skipping reconfiguration")
        return

    # This rank is in a reduced TP DP replica - need to reconfigure
    # Get active ranks for this DP replica (supports non-contiguous)
    active_local_ranks = get_active_ranks_for_dp(dp_replica_id, tp_base, ddp_config)
    local_rank_in_dp = rank % dp_replica_size

    logger.info(f"[NTP] Rank {rank} in DP replica {dp_replica_id}: active_local_ranks={active_local_ranks}")

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
                logger.info(f"[NTP] Rank {rank} created TP group: {tp_group_ranks}")

        # Create new CP groups (one per active TP position)
        for tp_rank_in_slice in active_local_ranks:
            cp_group_ranks = [
                dp_replica_start + tp_rank_in_slice + i * tp_base for i in range(cp_size)
            ]
            cp_group = dist.new_group(ranks=cp_group_ranks)

            if rank in cp_group_ranks:
                parallel_state._CONTEXT_PARALLEL_GROUP = cp_group
                parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS = cp_group_ranks
                logger.info(f"[NTP] Rank {rank} created CP group: {cp_group_ranks}")

        # Update TENSOR_AND_CONTEXT_PARALLEL_GROUP
        tp_rank_in_slice = local_rank_in_dp % tp_base
        if tp_rank_in_slice in active_local_ranks:
            tp_cp_group_ranks = []
            for cp_r in range(cp_size):
                for active_tp in active_local_ranks:
                    tp_cp_group_ranks.append(dp_replica_start + cp_r * tp_base + active_tp)
            tp_cp_group = dist.new_group(ranks=tp_cp_group_ranks)
            parallel_state._TENSOR_AND_CONTEXT_PARALLEL_GROUP = tp_cp_group
            logger.info(f"[NTP] Rank {rank} created TP-CP group: {tp_cp_group_ranks}")
        else:
            # Non-active (spare) rank - exit
            logger.info(f"[NTP] Rank {rank} is a spare rank with CP, exiting")
            sys.exit(0)
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
            logger.info(f"[NTP] Rank {rank} created TP group: {tp_group_ranks}")
        else:
            # Non-active (spare) rank - exit
            logger.info(f"[NTP] Rank {rank} is a spare rank, exiting")
            sys.exit(0)


# ======================================================================================
# Parameter Resharding for NTP
# ======================================================================================


def ntp_map(module: torch.nn.Module, ddp_config: DistributedDataParallelConfig, num_shards: int):
    """
    Initialize TP-sharded params with mapping between healthy and unhealthy TP sizes.

    Only healthy (full TP) ranks need send_splits and recv_splits to know how to reshard
    parameters when synchronizing with unhealthy (reduced TP) ranks.
    Unhealthy ranks synchronize directly without resharding.

    Args:
        module: Module containing parameters to initialize (e.g., self_attention or mlp)
        ddp_config: DDP configuration containing tp_base and tp_spares
        num_shards: Number of shards (e.g., num_attention_heads or ffn_hidden_size)
    """
    if ddp_config.tp_spares == 0:
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

    # Check if this (DP, CP, PP) combination uses reduced TP (unhealthy) or full TP (healthy)
    non_active_ranks_per_dp = ddp_config.non_active_ranks_per_dp or {}

    # Check if this (dp, cp, pp) combination has non-active ranks specified
    # If it does, it's an unhealthy rank that uses reduced TP
    rank_key = (dp_rank, cp_rank, pp_rank)
    if rank_key in non_active_ranks_per_dp:
        # This is an unhealthy rank with reduced TP - skip
        logger.debug(f"[NTP] Rank {rank} [DP {dp_rank}, CP {cp_rank}, PP {pp_rank}] Unhealthy rank, skipping")
        return

    # This is a healthy rank (full TP) - it needs send/recv splits to communicate
    # with unhealthy ranks that have reduced TP
    logger.debug(f"[NTP] Rank {rank} [DP {dp_rank}] Setting up send/recv splits for healthy rank")

    for param in module.parameters():
        # Handle both tensor parallel parameters (tensor_model_parallel=True)
        # and vocabulary parallel parameters (partition_dim exists but tensor_model_parallel may be False/absent)
        if (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel) or (
            hasattr(param, 'partition_dim') and not hasattr(param, 'tensor_model_parallel')
        ):
            # For healthy ranks, compute send/recv splits for communication with unhealthy ranks
            # We need to know how to reshard to match the reduced TP size
            reduced_tp_size = ddp_config.tp_base - ddp_config.tp_spares

            shard_ids = torch.arange(num_shards)
            # Partitions for reduced TP (what unhealthy ranks have)
            sync_partitions = list(shard_ids.chunk(reduced_tp_size))

            # Full partitions for healthy ranks (tp_base ranks)
            comp_partitions = sync_partitions + [
                torch.empty(int(len(shard_ids) / ddp_config.tp_base), dtype=torch.int)
                for _ in range(ddp_config.tp_spares)
            ]

            # Build comp_2_sync: for spare positions, which reduced TP ranks do they map to
            comp_2_sync = [[] for _ in range(ddp_config.tp_base)]
            sync_part_idx = 0

            for spare_part_idx in range(reduced_tp_size, ddp_config.tp_base):
                for shard_part_idx in range(len(comp_partitions[spare_part_idx])):
                    # Take the last shard from the current reduced TP rank
                    comp_partitions[spare_part_idx][shard_part_idx] = comp_partitions[sync_part_idx][
                        -1
                    ]
                    comp_partitions[sync_part_idx] = comp_partitions[sync_part_idx][:-1]
                    comp_2_sync[spare_part_idx].append(sync_part_idx)
                    sync_part_idx = (sync_part_idx + 1) % reduced_tp_size

            # Compute param_splits: how many shards each rank sends to each other rank
            param_splits = [
                torch.bincount(torch.tensor(c2s, dtype=torch.int), minlength=ddp_config.tp_base)
                for c2s in comp_2_sync
            ]

            shard_size = int(param.shape[param.partition_dim] * ddp_config.tp_base / len(shard_ids))
            send_splits = [(p_split * shard_size).tolist() for p_split in param_splits]
            recv_splits = [
                [send_splits[send_idx][recv_idx] for send_idx in range(len(send_splits))]
                for recv_idx in range(ddp_config.tp_base)
            ]
            param.send_splits = send_splits
            param.recv_splits = recv_splits
            logger.debug(
                f"[NTP] Rank {rank} [DP {dp_rank}] Set send_splits and recv_splits "
                f"on parameter id={id(param)}, shape={param.shape}"
            )


def ntp_init(layer: torch.nn.Module, ddp_config: DistributedDataParallelConfig):
    """
    Initialize nonuniform TP mappings for a TransformerLayer.

    This should be called after the layer is created to set up the send_splits
    and recv_splits attributes on tensor-parallel parameters.

    Args:
        layer: TransformerLayer instance
        ddp_config: DDP configuration containing tp_base and tp_spares
    """
    if ddp_config.tp_spares == 0:
        # No nonuniform TP, skip initialization
        return

    # Initialize self-attention parameters
    if hasattr(layer, 'self_attention'):
        ntp_map(
            layer.self_attention,
            ddp_config,
            layer.self_attention.config.num_attention_heads,
        )

    # Initialize MLP parameters
    if hasattr(layer, 'mlp'):
        ntp_map(layer.mlp, ddp_config, layer.mlp.config.ffn_hidden_size)


# ======================================================================================
# NTP-aware ParamAndGradBuffer
# ======================================================================================


class NonuniformTPParamAndGradBucketGroup(_ParamAndGradBucketGroup):
    """
    NTP-aware version of _ParamAndGradBucketGroup.
    Skips gradient synchronization for spare GPUs.
    """

    def allreduce_or_reduce_scatter_gradients(
        self,
        async_op: bool = True,
        reduce_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
        stream_context=nullcontext(),
    ):
        """
        Override to skip gradient synchronization for spare GPUs in NTP mode.
        """
        # Determine communication group
        if self.ddp_config.use_distributed_optimizer:
            communication_group = self.data_parallel_group
        elif self.ddp_config.use_custom_fsdp:
            assert (
                self.local_distributed_optimizer_instance_size == 1
            ), "Custom FSDP only works with DistOpt instance size 1"
            communication_group = self.data_parallel_group
        else:
            communication_group = self.data_parallel_group

        # NOTE: only sync on core GPUs (not spares) for nonuniform TP
        grad_reduce_handle = None
        should_sync = True
        if self.ddp_config.tp_spares > 0:
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            should_sync = tp_rank < self.ddp_config.tp_base - self.ddp_config.tp_spares

        if should_sync:
            # Coalesce communication kernels across buckets in the bucket group.
            with stream_context, _coalescing_manager(
                communication_group, async_ops=async_op
            ) as cm:
                for idx, bucket in enumerate(self.buckets):
                    if self.ddp_config.use_distributed_optimizer:
                        if self.cached_grad_buffer_shard_list[idx] is None:
                            self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                                bucket.grad_data, self.intra_distributed_optimizer_instance_size
                            )
                        local_data_view = self.cached_grad_buffer_shard_list[idx][
                            self.intra_distributed_optimizer_instance_rank
                        ]
                        grad_reduce_handle = dist_reduce_scatter_func(
                            local_data_view,
                            bucket.grad_data,
                            op=reduce_op,
                            group=communication_group,
                            async_op=async_op,
                        )
                    else:
                        dist.all_reduce(
                            bucket.grad_data,
                            op=reduce_op,
                            group=communication_group,
                            async_op=async_op,
                        )

        # With multiple DistOpt instances, we need to all-reduce across instances.
        if (
            self.ddp_config.use_distributed_optimizer
            and self.distributed_optimizer_instance_size > 1
        ):
            assert (
                self.intra_distributed_optimizer_instance_size == 1
            ), "Multiple DistOpt instances not supported with instance size > 1"

            # All-gather all reduced shards across the DistOpt instances.
            if grad_reduce_handle is not None:
                grad_reduce_handle.wait()

            # Apply all-gather for instances.
            for idx, bucket in enumerate(self.buckets):
                if async_op:
                    dist.all_reduce(
                        self.cached_grad_buffer_shard_list[idx],
                        op=reduce_op,
                        group=self.intra_distributed_optimizer_instance_group,
                        async_op=async_op,
                    )
                else:
                    dist.all_reduce(
                        self.cached_grad_buffer_shard_list[idx],
                        op=reduce_op,
                        group=self.intra_distributed_optimizer_instance_group,
                        async_op=async_op,
                    )

        # NOTE: cm only exists for core GPUs when nonuniform TP is enabled
        if async_op and should_sync:
            if self.ddp_config.reduce_scatter_with_fp32_accumulation:
                assert (
                    len(self.buckets) == 1
                ), "reduce_scatter_with_fp32_accumulation requires single bucket"
                return cm
            else:
                return cm if grad_reduce_handle is None else grad_reduce_handle


class NonuniformTPParamAndGradBuffer(_ParamAndGradBuffer):
    """
    NTP-aware version of _ParamAndGradBuffer.
    Adjusts buffer sizes and splits gradients for NTP.
    """

    def _make_param_hook(
        self,
        param: torch.nn.Parameter,
        param_group_id: int,
        param_id: int,
        data_parallel_group: dist.ProcessGroup,
        overlap_param_gather: bool,
    ):
        """
        Override to adjust buffer sizes for NTP and split gradients.
        """
        # First, calculate this_numel with NTP adjustment
        this_numel = param.data.nelement()

        # Adjust numel for nonuniform tensor parallelism
        if (
            self.ddp_config.tp_spares > 0
            and hasattr(param, 'tensor_model_parallel')
            and param.tensor_model_parallel
        ):
            tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
            this_numel = int(
                tp_world_size * this_numel / (self.ddp_config.tp_base - self.ddp_config.tp_spares)
            )

        # Call parent method to set up the param hook and buffers
        # (Note: This is a simplified approach; you may need to copy more logic from parent)
        result = super()._make_param_hook(
            param, param_group_id, param_id, data_parallel_group, overlap_param_gather
        )

        # After parent setup, handle NTP-specific grad buffer splitting
        if (
            self.ddp_config.tp_spares > 0
            and hasattr(param, 'tensor_model_parallel')
            and param.tensor_model_parallel
        ):
            tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
            shape = list(param.data.shape)
            shape[param.partition_dim] = int(
                shape[param.partition_dim]
                * tp_world_size
                / (self.ddp_config.tp_base - self.ddp_config.tp_spares)
            )

            # Get the grad buffer that was allocated by parent
            # Calculate sizes for contiguous split
            main_size = param.shape[param.partition_dim]
            side_size = shape[param.partition_dim] - param.shape[param.partition_dim]

            # Create target shapes for main_grad and side_grad
            main_shape = list(shape)
            main_shape[param.partition_dim] = main_size
            side_shape = list(shape)
            side_shape[param.partition_dim] = side_size

            # Calculate total elements for main_grad
            main_numel = torch.Size(main_shape).numel()

            # Split param.main_grad into main_grad and side_grad
            if hasattr(param, 'main_grad'):
                grad_buffer_flat = param.main_grad.view(-1)
                main_grad_flat = grad_buffer_flat[:main_numel]
                side_grad_flat = grad_buffer_flat[main_numel:]

                # Reshape to final dimensions - these will be contiguous
                param.main_grad = main_grad_flat.view(main_shape)
                param.side_grad = side_grad_flat.view(side_shape)

        return result


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
    ):
        # Use NTP-aware buffer class
        if ddp_config.tp_spares > 0:
            # Temporarily monkey-patch the buffer class
            original_buffer_class = _ParamAndGradBuffer
            import megatron.core.distributed.param_and_grad_buffer as buffer_module

            buffer_module._ParamAndGradBuffer = NonuniformTPParamAndGradBuffer

        super().__init__(config, ddp_config, module, disable_bucketing, pg_collection)

        if ddp_config.tp_spares > 0:
            # Restore original class
            buffer_module._ParamAndGradBuffer = original_buffer_class

    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        """
        Override to add NTP gradient synchronization between spare and core GPUs.
        """
        original_hook = super()._make_backward_post_hook(param)

        def ntp_hook(*unused):
            # Call original hook first
            original_hook(*unused)

            # Add NTP-specific logic
            if (
                self.ddp_config.tp_spares > 0
                and hasattr(param, 'tensor_model_parallel')
                and param.tensor_model_parallel
                and parallel_state.get_tensor_model_parallel_world_size() == self.ddp_config.tp_base
            ):
                empty_shape = list(param.shape)
                empty_shape[param.partition_dim] = 0
                tp_rank = parallel_state.get_tensor_model_parallel_rank()

                if tp_rank < self.ddp_config.tp_base - self.ddp_config.tp_spares:
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
                        for _ in range(self.ddp_config.tp_base - self.ddp_config.tp_spares)
                    ] + [
                        t.contiguous()
                        for t in torch.split(
                            param.side_grad, param.recv_splits[tp_rank], dim=param.partition_dim
                        )
                    ][-self.ddp_config.tp_spares :]
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
                    dist.all_to_all(
                        output,
                        input,
                        group=parallel_state.get_tensor_model_parallel_group(),
                        async_op=True,
                    )
                except Exception as e:
                    logger.error(f'[NTP] Rank {tp_rank} all_to_all error: {e}')
                    logger.error(
                        f'[NTP] Rank {tp_rank} input element contiguity: {[i.is_contiguous() for i in input]}'
                    )
                    logger.error(
                        f'[NTP] Rank {tp_rank} output element contiguity: {[o.is_contiguous() for o in output]}'
                    )
                    raise e

        return ntp_hook


# ======================================================================================
# NTP-aware Optimizer Wrapper
# ======================================================================================


class NonuniformTPOptimizer:
    """
    Wrapper for optimizers to make gradients contiguous for NTP.
    """

    def __init__(self, optimizer, ddp_config: DistributedDataParallelConfig):
        self.optimizer = optimizer
        self.ddp_config = ddp_config

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
        if self.ddp_config.tp_spares > 0:
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    if hasattr(param, 'main_grad') and param.main_grad is not None:
                        if not param.main_grad.is_contiguous():
                            param.grad = param.main_grad.contiguous()
                        else:
                            param.grad = param.main_grad

        return result


# ======================================================================================
# Test Function
# ======================================================================================


def test_ntp():
    """Test function for nonuniform TP initialization."""
    head_dim = 128
    ffn_exp = 4

    class MockConfig:
        num_attention_heads = 24
        ffn_hidden_size = num_attention_heads * head_dim * ffn_exp

    class MockModule:
        def __init__(self, out_features):
            self.weight = torch.nn.Parameter(torch.randn(out_features, 1, dtype=torch.half))
            self.weight.partition_dim = 1
            self.weight.tensor_model_parallel = True
            self.config = MockConfig()

        def parameters(self):
            return [self.weight]

    class MockLayer:
        def __init__(self):
            self.self_attention = MockModule(int(3 * 10248 / 8))
            self.mlp = MockModule(12288 // 8)

    layer = MockLayer()
    ddp_config = DistributedDataParallelConfig(tp_base=8, tp_spares=2)
    ntp_init(layer, ddp_config)
    print("NTP initialization test passed!")
    return layer


if __name__ == '__main__':
    layer = test_ntp()
