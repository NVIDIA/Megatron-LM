# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Nonuniform Tensor Parallelism utilities.

This module provides functionality for nonuniform tensor parallelism (NTP),
where a subset of TP ranks (called "spares") provide fault tolerance while
the remaining "core" ranks handle the actual computation.

Extended to support non-contiguous active ranks, allowing arbitrary GPU failures.
"""

import torch
from typing import Dict, List, Set, Tuple

from .. import parallel_state
from .distributed_data_parallel_config import DistributedDataParallelConfig


def compute_uniform_tp_spares_with_parity(
    faulty_gpu_map: Dict[int, List[int]],
    tp_base: int
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
    dp_rank: int,
    tp_base: int,
    ddp_config: DistributedDataParallelConfig
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


def reconfigure_tp_cp_groups_for_ntp(ddp_config: DistributedDataParallelConfig):
    """
    Reconfigure TP and CP process groups for nonuniform tensor parallelism.
    
    This function is now a lightweight wrapper that leverages the NTP support
    built into initialize_model_parallel. It recreates the TP/CP groups with
    the NTP configuration.
    
    Args:
        ddp_config: DDP configuration containing tp_base, tp_spares, num_reduced_tp_dp_ranks,
                    and optionally non_active_ranks_per_dp
        
    Note:
        This function directly modifies the parallel_state module's process groups.
        Non-active (spare) ranks will exit after group creation.
        
    For new code: If you know NTP configuration from the start, prefer passing
    ntp_config to initialize_model_parallel directly instead of calling this function.
    """
    if ddp_config.tp_spares == 0:
        # No nonuniform TP, nothing to reconfigure
        return
    
    tp_base = ddp_config.tp_base
    tp_spares = ddp_config.tp_spares
    cp_size = parallel_state.get_context_parallel_world_size()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    # Calculate which DP replicas use reduced TP
    dp_replica_size = tp_base * cp_size
    num_reduced_dp_ranks = ddp_config.num_reduced_tp_dp_ranks
    
    # Determine if current rank is in a reduced TP DP replica
    dp_replica_id = rank // dp_replica_size
    if dp_replica_id >= num_reduced_dp_ranks:
        # This rank is in a normal TP DP replica, no reconfiguration needed
        return
    
    # This rank is in a reduced TP DP replica - need to reconfigure
    # Get active ranks for this DP replica (supports non-contiguous)
    active_local_ranks = get_active_ranks_for_dp(dp_replica_id, tp_base, ddp_config)
    local_rank_in_dp = rank % dp_replica_size
    
    if cp_size > 1:
        # With CP enabled: recreate TP, CP, and TP-CP groups
        dp_replica_start = dp_replica_id * dp_replica_size
        
        # Create new TP groups (one per CP slice in this DP replica)
        for cp_rank in range(cp_size):
            cp_slice_start = dp_replica_start + cp_rank * tp_base
            tp_group_ranks = [cp_slice_start + local_tp for local_tp in active_local_ranks]
            tp_group = torch.distributed.new_group(ranks=tp_group_ranks)
            
            if rank in tp_group_ranks:
                parallel_state._TENSOR_MODEL_PARALLEL_GROUP = tp_group
                parallel_state._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = tp_group_ranks
                parallel_state._MODEL_PARALLEL_GROUP = tp_group
                parallel_state._MODEL_PARALLEL_GLOBAL_RANKS = tp_group_ranks
        
        # Create new CP groups (one per active TP position)
        for tp_rank_in_slice in active_local_ranks:
            cp_group_ranks = [
                dp_replica_start + tp_rank_in_slice + i * tp_base for i in range(cp_size)
            ]
            cp_group = torch.distributed.new_group(ranks=cp_group_ranks)
            
            if rank in cp_group_ranks:
                parallel_state._CONTEXT_PARALLEL_GROUP = cp_group
                parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS = cp_group_ranks
        
        # Update TENSOR_AND_CONTEXT_PARALLEL_GROUP
        tp_rank_in_slice = local_rank_in_dp % tp_base
        if tp_rank_in_slice in active_local_ranks:
            tp_cp_group_ranks = []
            for cp_r in range(cp_size):
                for active_tp in active_local_ranks:
                    tp_cp_group_ranks.append(dp_replica_start + cp_r * tp_base + active_tp)
            tp_cp_group = torch.distributed.new_group(ranks=tp_cp_group_ranks)
            parallel_state._TENSOR_AND_CONTEXT_PARALLEL_GROUP = tp_cp_group
        else:
            # Non-active (spare) rank - exit
            import sys
            sys.exit(0)
    else:
        # No CP: simpler case
        dp_replica_start = dp_replica_id * dp_replica_size
        tp_group_ranks = [dp_replica_start + local_tp for local_tp in active_local_ranks]
        
        if rank in tp_group_ranks:
            tp_group = torch.distributed.new_group(ranks=tp_group_ranks)
            parallel_state._TENSOR_MODEL_PARALLEL_GROUP = tp_group
            parallel_state._MODEL_PARALLEL_GROUP = tp_group
            parallel_state._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = tp_group_ranks
            parallel_state._MODEL_PARALLEL_GLOBAL_RANKS = tp_group_ranks
        else:
            # Non-active (spare) rank - exit
            import sys
            sys.exit(0)


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
    from megatron.core import parallel_state
    rank = torch.distributed.get_rank()
    dp_rank = parallel_state.get_data_parallel_rank()
    cp_rank = parallel_state.get_context_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    
    print(f"[Rank {rank}] [DP {dp_rank}, CP {cp_rank}, PP {pp_rank}] ntp_map called with module={type(module).__name__}, num_shards={num_shards}")
    
    # Check if this (DP, CP, PP) combination uses reduced TP (unhealthy) or full TP (healthy)
    non_active_ranks_per_dp = ddp_config.non_active_ranks_per_dp or {}
    
    print(f"[Rank {rank}] [DP {dp_rank}, CP {cp_rank}, PP {pp_rank}] non_active_ranks_per_dp = {non_active_ranks_per_dp}")
    
    # Check if this (dp, cp, pp) combination has non-active ranks specified
    # If it does, it's an unhealthy rank that uses reduced TP
    rank_key = (dp_rank, cp_rank, pp_rank)
    if rank_key in non_active_ranks_per_dp:
        # This is an unhealthy rank with reduced TP - skip
        print(f"[Rank {rank}] [DP {dp_rank}, CP {cp_rank}, PP {pp_rank}] ntp_map: Unhealthy rank, skipping")
        return
    
    # This is a healthy rank (full TP) - it needs send/recv splits to communicate
    # with unhealthy ranks that have reduced TP
    
    import torch.distributed as dist
    print(f"[Rank {rank}] [DP {dp_rank}] ntp_map: Setting up send/recv splits for healthy rank")
        
    for param in module.parameters():
        has_tp_attr = hasattr(param, 'tensor_model_parallel')
        is_tp = param.tensor_model_parallel if has_tp_attr else False
        has_partition_dim = hasattr(param, 'partition_dim')
        print(f"[Rank {rank}] [DP {dp_rank}] ntp_map: param has_tp_attr={has_tp_attr}, is_tp={is_tp}, has_partition_dim={has_partition_dim}")
        
        # Handle both tensor parallel parameters (tensor_model_parallel=True)
        # and vocabulary parallel parameters (partition_dim exists but tensor_model_parallel may be False/absent)
        if (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel) or \
           (hasattr(param, 'partition_dim') and not has_tp_attr):
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
                    comp_partitions[spare_part_idx][shard_part_idx] = comp_partitions[sync_part_idx][-1]
                    comp_partitions[sync_part_idx] = comp_partitions[sync_part_idx][:-1]
                    comp_2_sync[spare_part_idx].append(sync_part_idx)
                    sync_part_idx = (sync_part_idx + 1) % reduced_tp_size
            
            # Compute param_splits: how many shards each rank sends to each other rank
            param_splits = [
                torch.bincount(
                    torch.tensor(c2s, dtype=torch.int), minlength=ddp_config.tp_base
                )
                for c2s in comp_2_sync
            ]
            
            shard_size = int(
                param.shape[param.partition_dim] * ddp_config.tp_base / len(shard_ids)
            )
            send_splits = [(p_split * shard_size).tolist() for p_split in param_splits]
            recv_splits = [
                [send_splits[send_idx][recv_idx] for send_idx in range(len(send_splits))]
                for recv_idx in range(ddp_config.tp_base)
            ]
            param.send_splits = send_splits
            param.recv_splits = recv_splits
            print(f"[Rank {rank}] [DP {dp_rank}] ntp_map: Set send_splits and recv_splits on parameter id={id(param)}, shape={param.shape}")


def initialize_nonuniform_tp(ddp_config: DistributedDataParallelConfig):
    """
    Initialize nonuniform tensor parallelism.
    
    This function should be called after initialize_megatron() but before creating
    the model. It reconfigures TP and CP process groups for ranks that should use
    reduced TP.
    
    Args:
        ddp_config: DDP configuration containing tp_base, tp_spares, and num_reduced_tp_dp_ranks
        
    Example:
        ```python
        # After initialize_megatron()
        initialize_megatron()
        
        # Create DDP config
        ddp_config = DistributedDataParallelConfig(
            tp_base=8,
            tp_spares=2,
            num_reduced_tp_dp_ranks=1,
        )
        
        # Initialize nonuniform TP (reconfigures process groups)
        initialize_nonuniform_tp(ddp_config)
        
        # Now create model
        model = build_model(...)
        ```
    """
    if ddp_config.tp_spares > 0:
        reconfigure_tp_cp_groups_for_ntp(ddp_config)


def ntp_init(
    layer: torch.nn.Module, ddp_config: DistributedDataParallelConfig
):
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


def test_ntp():
    """Test function for nonuniform TP initialization."""
    head_dim = 128
    ffn_exp = 4

    class MockConfig:
        num_attention_heads = 24
        ffn_hidden_size = num_attention_heads * head_dim * ffn_exp

    class MockModule:
        def __init__(self, out_features):
            self.weight = torch.nn.Parameter(
                torch.randn(out_features, 1, dtype=torch.half)
            )
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
    return layer


if __name__ == '__main__':
    layer = test_ntp()
    print("NTP initialization test passed!")

