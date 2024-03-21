import heapq
import logging
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple, TypeVar

import numpy as np
import torch
import torch.distributed as dist

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, is_main_replica
from megatron.core.dist_checkpointing.strategies.base import SaveShardedStrategy

logger = logging.getLogger(__name__)


SaveDistributionT = Tuple[dict, dict]


class FullyParallelSaveStrategyWrapper(SaveShardedStrategy):
    """ Wraps arbitrary strategy and distributes the save during `save`.

    The save distribution happens without any *data* communication.
    Only the *metadata* is exchanged and based on data replication on different
    ranks, we try to distribute the save as uniformly as possible.

    This wrapper assumes, that setting `replica_id` to 0 will make the
    underlying strategy do the saving on current rank. All the other `replica_id`s
    are set to 1.

    Currently, the save distribution is realized with a greedy algorithm
    described in `distribute_chunks_to_ranks`.
    """

    def __init__(
        self,
        strategy: SaveShardedStrategy,
        parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
        do_cache_distribution: bool = True,
    ):
        """ Initializes the wrapper.

        Args:
            strategy (SaveShardedStrategy): base strategy to wrap
            parallelization_group (ProcessGroup, optional): process group to use for save
                distribution. Note that this doesn't have to match exactly the
                data distribution, but should cover the replication pattern
                to maximize performance. Defaults to the whole world.
            do_cache_distribution (bool, optional): whether to cache the save distribution
                from previous calls. Should be set to True only if the state dict
                structure between the calls is always the same. Defaults to True.
        """
        super().__init__(strategy.backend, strategy.version)
        self.base_strategy = strategy
        self.parallelization_group = parallelization_group
        self.do_cache_distribution = do_cache_distribution

        self.cached_distribution: Optional[SaveDistributionT] = None

    def save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        self.apply_saving_parallelization(sharded_state_dict)
        return self.base_strategy.save(sharded_state_dict, checkpoint_dir)

    def apply_saving_parallelization(self, sharded_state_dict: ShardedStateDict) -> None:
        """ Distributes the save across ranks by exchanging metadata.

        Exchanges metadata from the state dict and computes the uniform
        (as close as possible) distribution of saves among the ranks.

        If `self.do_cache_distribution` is True, caches the distribution between
        the calls and subsequent distributions happen without any inter-rank
        communication.

        Args:
            sharded_state_dict (ShardedStateDict): state dict to distribute the saving

        Returns: None
        """
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


def _sharded_tensor_chunk_id(sharded_tensor: ShardedTensor) -> tuple:
    """ Unique id of the sharded tensor data.

    Should yield the same value for same data replicated on different ranks.

    Args:
        sharded_tensor (ShardedTensor): sharded tensor representing the data chunk

    Returns (tuple): unique id of a data chunk
    """
    f_range = sharded_tensor.flattened_range
    return (
        sharded_tensor.key,
        sharded_tensor.global_offset,
        None if f_range is None else (f_range.start, f_range.stop),
    )


def _shard_size(sh_ten: ShardedTensor):
    """ Returns size in bytes of a given sharded tensor. """
    if sh_ten.flattened_range is None:
        numel = np.product(sh_ten.local_shape)
    else:
        numel = sh_ten.flattened_range.stop - sh_ten.flattened_range.start
    return numel * torch._utils._element_size(sh_ten.dtype)


def determine_main_replica_uniform_distribution(
    sharded_state_dict: ShardedStateDict, parallelization_group: torch.distributed.ProcessGroup
) -> Optional[SaveDistributionT]:
    """ Computes the save distribution.

    Should be used in conjunction with `distribute_main_replicas_with_precomputed_distribution`
    which applies the computed save distribution.

    We rely on the fact that the assignment algorithm is deterministic on all ranks,
    so there is no extra communication needed after metadata exchange.

    Args:
        sharded_state_dict (ShardedStateDict): state dict to compute the distribution of
        parallelization_group (ProcessGroup): distribution will be computed
            within this process group

    Returns (SaveDistributionT, optional): distribution that can be used to apply the
        parallelization. Returns None if the process_group is trivial (1 rank)

    """
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
            shard_id = _sharded_tensor_chunk_id(sh_ten)
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
    sharded_state_dict: ShardedStateDict,
    parallelization_group: torch.distributed.ProcessGroup,
    precomputed_distribution: Optional[SaveDistributionT],
):
    """ Applies the save distribution computed with `determine_main_replica_uniform_distribution`

    Args:
        sharded_state_dict (ShardedStateDict): state dict to apply the save distribution to
        parallelization_group (ProcessGroup): distribution will be applied within this
            process group. Must match with the process group passed to
            `determine_main_replica_uniform_distribution`.
        precomputed_distribution (DistributionT): distribution computed with
            `determine_main_replica_uniform_distribution`

    Returns: None
    """
    group_size = torch.distributed.get_world_size(group=parallelization_group)
    if group_size <= 1:
        return
    if precomputed_distribution is None:
        raise ValueError(
            'precomputed_distribution must be not None for non-trivial parallelization group'
        )

    local_shards = list(
        sh_base
        for sh_base in nested_values(sharded_state_dict)
        if isinstance(sh_base, ShardedTensor)
    )

    shard_to_saving_rank, is_saved_by_this_distributed_group = precomputed_distribution

    rank_within_dp_group = torch.distributed.get_rank(parallelization_group)
    for sh_ten in local_shards:
        shard_id = _sharded_tensor_chunk_id(sh_ten)
        if (
            is_saved_by_this_distributed_group.get(shard_id, False)
            and rank_within_dp_group == shard_to_saving_rank[shard_id]
        ):
            sh_ten.replica_id = 0
        else:
            sh_ten.replica_id = 1  # TODO: consider something more informative


T = TypeVar('T')


def distribute_chunks_to_ranks(
    shard_to_ranks: Dict[T, List[int]], shard_to_size: Dict[T, int], num_ranks: int
) -> Dict[T, int]:
    """ Computes uniform distribution of workload across ranks, based on sizes.

    Currently, the assignment is greedy, based on:
    1. Firstly, the coverage of each shard (lower coverage is assigned first)
    2. Secondly, the size of each shard (larger size is assigned first)
    3. Finally, shard id for differentiation.

    Third step is added because we rely on the fact that the assignment is deterministic on all ranks.

    Args:
        shard_to_ranks (Dict[T, List[int]]): mapping which tells which rank have access to which shards
        shard_to_size (Dict[T, int]): sizes of each shard
        num_ranks (int): number of ranks in the parallelization group

    Returns (Dict[T, int]): assignment of shard to rank (which rank should do the work
        to achieve maximal uniformity)
    """
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


def distribute_chunks_to_ranks_heapq(
    shard_to_ranks: Dict[T, List[int]], shard_to_size: Dict[T, int], num_ranks: int
) -> Dict[T, int]:
    """ Heapq implementation of `distribute_chunks_to_ranks`. *Not* required for efficiency now. """
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
