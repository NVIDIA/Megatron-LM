import heapq
import logging
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Dict, List, Optional, TypeVar

import numpy as np
import torch
import torch.distributed as dist

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, is_main_replica
from megatron.core.dist_checkpointing.strategies.base import SaveShardedStrategy

logger = logging.getLogger(__name__)


class FullyParallelSaveStrategyWrapper(SaveShardedStrategy):
    def __init__(
        self,
        strategy: SaveShardedStrategy,
        parallelization_group: Optional[torch.distributed.group] = None,
        do_cache_distribution: bool = True,
    ):
        super().__init__(strategy.backend, strategy.version)
        self.base_strategy = strategy
        self.parallelization_group = parallelization_group
        self.do_cache_distribution = do_cache_distribution

        self.cached_distribution = None

    def save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        self.apply_saving_parallelization(sharded_state_dict)
        return self.base_strategy.save(sharded_state_dict, checkpoint_dir)

    def apply_saving_parallelization(self, sharded_state_dict: ShardedStateDict) -> None:
        if self.do_cache_distribution and self.cached_distribution is not None:
            logger.debug(f'Apply *cached* save parallelization')
            precomputed_distribution = self.cached_distribution
        else:
            logger.debug(f'Apply save parallelization')
            precomputed_distribution = determine_main_replica_uniform_distribution(
                sharded_state_dict, self.parallelization_group
            )
            if self.do_cache_distribution:
                self.cached_distribution = precomputed_distribution

        distribute_main_replicas_with_precomputed_distribution(
            sharded_state_dict, self.parallelization_group, precomputed_distribution
        )

    @property
    def can_handle_sharded_objects(self):
        return self.base_strategy.can_handle_sharded_objects


def sharded_tensor_chunk_id(sharded_tensor: ShardedTensor):
    f_range = sharded_tensor.flattened_range
    return (
        sharded_tensor.key,
        sharded_tensor.global_offset,
        None if f_range is None else (f_range.start, f_range.stop),
    )


def _shard_size(sh_ten: ShardedTensor):
    if sh_ten.flattened_range is None:
        numel = np.product(sh_ten.local_shape)
    else:
        numel = sh_ten.flattened_range.stop - sh_ten.flattened_range.start
    return numel * torch._utils._element_size(sh_ten.dtype)


T = TypeVar('T')


def determine_main_replica_uniform_distribution(sharded_state_dict, parallelization_group):
    group_size = torch.distributed.get_world_size(group=parallelization_group)
    if group_size <= 1:
        return
    local_shards = list(
        sh_base
        for sh_base in nested_values(sharded_state_dict)
        if isinstance(sh_base, ShardedTensor)
    )
    local_shards_no_data = [ten.without_data() for ten in local_shards]

    all_shards = [None] * torch.distributed.get_world_size(group=parallelization_group)
    torch.distributed.all_gather_object(
        all_shards, local_shards_no_data, group=parallelization_group
    )

    shard_to_ranks = defaultdict(list)
    shard_to_size = {}
    is_saved_by_this_distributed_group = {}
    for rank, rank_shards in enumerate(all_shards):
        for sh_ten in rank_shards:
            shard_id = sharded_tensor_chunk_id(sh_ten)
            shard_to_ranks[shard_id].append(rank)
            if shard_id not in shard_to_size:
                shard_to_size[shard_id] = _shard_size(sh_ten)
            if is_main_replica(sh_ten.replica_id):
                is_saved_by_this_distributed_group[shard_id] = True

    shard_to_ranks = {
        k: v for k, v in shard_to_ranks.items() if is_saved_by_this_distributed_group.get(k, False)
    }

    shard_to_saving_rank = distribute_chunks_to_ranks(
        shard_to_ranks, shard_to_size, len(all_shards)
    )

    return shard_to_saving_rank, is_saved_by_this_distributed_group


def distribute_main_replicas_with_precomputed_distribution(
    sharded_state_dict, data_parallel_group, precomputed_distribution
):
    group_size = torch.distributed.get_world_size(group=data_parallel_group)
    if group_size <= 1:
        return
    local_shards = list(
        sh_base
        for sh_base in nested_values(sharded_state_dict)
        if isinstance(sh_base, ShardedTensor)
    )

    shard_to_saving_rank, is_saved_by_this_distributed_group = precomputed_distribution

    rank_within_dp_group = torch.distributed.get_rank(data_parallel_group)
    for sh_ten in local_shards:
        shard_id = sharded_tensor_chunk_id(sh_ten)
        if (
            is_saved_by_this_distributed_group.get(shard_id, False)
            and rank_within_dp_group == shard_to_saving_rank[shard_id]
        ):
            sh_ten.replica_id = 0
        else:
            sh_ten.replica_id = 1  # TODO: consider something more informative


def distribute_chunks_to_ranks_heapq(
    shard_to_ranks: Dict[T, List[int]], shard_to_size: Dict[T, int], num_ranks: int
) -> Dict[T, int]:
    shard_to_ranks = {k: tuple(v) for k, v in shard_to_ranks.items()}
    shard_to_saving_rank = {}
    rank_sizes = [(0, rank) for rank in range(num_ranks)]
    heapq.heapify(rank_sizes)

    # start from tensors with lowest coverage, then go by tensor size from largest
    for shard_id, shard_ranks in sorted(
        shard_to_ranks.items(),
        key=lambda sh_id_ranks: (
            len(sh_id_ranks[1]),
            shard_to_size[sh_id_ranks[0]],
            sh_id_ranks[0],
        ),
    ):
        # assign greedily to the least occupied rank
        popped = []
        while True:
            size, rank = heapq.heappop(rank_sizes)
            if rank in shard_ranks:
                break
            popped.append((size, rank))

        shard_to_saving_rank[shard_id] = rank
        for p in popped:
            heapq.heappush(rank_sizes, p)

        heapq.heappush(rank_sizes, (size + shard_to_size[shard_id], rank))

    return shard_to_saving_rank


def distribute_chunks_to_ranks(
    shard_to_ranks: Dict[T, List[int]], shard_to_size: Dict[T, int], num_ranks: int
) -> Dict[T, int]:
    shard_to_ranks = {k: tuple(v) for k, v in shard_to_ranks.items()}
    shard_to_saving_rank = {}
    rank_sizes = [(0, rank) for rank in range(num_ranks)]

    # start from tensors with lowest coverage, then go by tensor size from largest (hence minus size)
    for shard_id, shard_ranks in sorted(
        shard_to_ranks.items(),
        key=lambda sh_id_ranks: (
            len(sh_id_ranks[1]),
            -shard_to_size[sh_id_ranks[0]],
            sh_id_ranks[0],
        ),
    ):
        # assign greedily to the least occupied rank

        size, rank = min((size, rank) for size, rank in rank_sizes if rank in shard_ranks)

        shard_to_saving_rank[shard_id] = rank
        rank_sizes[rank] = (size + shard_to_size[shard_id], rank)

    logger.debug(f'distribute_chunks_to_ranks distribution: {rank_sizes}')

    return shard_to_saving_rank
