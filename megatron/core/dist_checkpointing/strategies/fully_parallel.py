import heapq
import logging
from collections import defaultdict
from functools import reduce
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Set, Tuple, TypeVar, cast

import numpy as np
import torch
import torch.distributed as dist

from megatron.core.dist_checkpointing import LocalNonpersitentObject, ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.dict_utils import (
    dict_list_map_inplace,
    extract_matching_values,
    merge,
    nested_values,
)
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, is_main_replica
from megatron.core.dist_checkpointing.serialization import validate_sharding_integrity
from megatron.core.dist_checkpointing.strategies.base import (
    LoadShardedStrategy,
    SaveShardedStrategy,
)

logger = logging.getLogger(__name__)


ChunkId = Tuple[str, tuple, Optional[tuple]]
SaveDistribution = Tuple[Dict[ChunkId, int], Set[ChunkId], Dict[ChunkId, ShardedTensor]]


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

    def __init__(
        self,
        strategy: SaveShardedStrategy,
        parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
        do_cache_distribution: bool = False,
    ):
        super().__init__(strategy.backend, strategy.version)
        self.base_strategy = strategy
        self.parallelization_group = parallelization_group
        self.do_cache_distribution = do_cache_distribution

        self.cached_distribution: Optional[SaveDistribution] = None

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

        distribute_main_replicas_with_precomputed_distribution(
            sharded_state_dict, self.parallelization_group, precomputed_distribution
        )
        if self.cached_distribution is None:
            # First time applying the parallelization
            validate_sharding_integrity(nested_values(sharded_state_dict))
        if self.do_cache_distribution:
            self.cached_distribution = precomputed_distribution

    @property
    def can_handle_sharded_objects(self):
        return self.base_strategy.can_handle_sharded_objects


class FullyParallelLoadStrategyWrapper(LoadShardedStrategy):
    def __init__(
        self,
        strategy: LoadShardedStrategy,
        parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
        do_cache_distribution: bool = False,
        gather_algo: str = 'rounds'  # or 'object'
    ):
        super().__init__()
        self.base_strategy = strategy
        self.parallelization_group = parallelization_group
        self.do_cache_distribution = do_cache_distribution
        self.gather_algo = gather_algo

        self.cached_distribution: Optional[SaveDistribution] = None

    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        if torch.distributed.get_world_size(self.parallelization_group) <= 1:
            return self.base_strategy.load(sharded_state_dict, checkpoint_dir)

        start = time()
        precomputed_distribution = self.apply_loading_parallelization(sharded_state_dict)
        end = time()
        logger.debug(f'self.apply_loading_parallelization took {end - start}s')
        start = end
        (
            sharded_tensors,
            sharded_state_dict,
            to_load_shards,
            unloaded_shards,
        ) = self.defer_loading_sharded_tensors(sharded_state_dict)
        # Load only sharded objects
        loaded_state_dict = self.base_strategy.load(sharded_state_dict, checkpoint_dir)

        end = time()
        logger.debug(f'Base load of ShardedObjects took {end - start}s')
        start = end

        # Load sharded tensors separately
        loaded_tensors = self.base_strategy.load(to_load_shards, checkpoint_dir)

        end = time()
        logger.debug(f'Base load of ShardedTensors took {end - start}s')
        start = end

        logger.debug(f'Applying parallel load with algo {self.gather_algo}')
        if self.gather_algo == 'object':
            all_loaded_tensors = self.exchange_loaded_tensors_gather_object(
                loaded_tensors, unloaded_shards, precomputed_distribution, self.parallelization_group
            )
        elif self.gather_algo == 'rounds':
            all_loaded_tensors = self.exchange_loaded_tensors_gather_rounds(
                loaded_tensors, unloaded_shards, precomputed_distribution, self.parallelization_group
            )
        else:
            raise NotImplementedError(f'Unrecognized gather algorithm: {self.gather_algo}')

        sync_start = time()
        torch.cuda.synchronize()
        end = time()
        logger.debug(f'torch.cuda.synchronize took {end - sync_start}s')
        logger.debug(f'self.exchange_loaded_tensors took {end - start}s')

        self.fill_in_deferred_sharded_tensors(sharded_tensors, all_loaded_tensors)
        merge(loaded_state_dict, sharded_tensors)
        return loaded_state_dict

    def defer_loading_sharded_tensors(
        self, sharded_state_dict: ShardedStateDict
    ) -> Tuple[
        ShardedStateDict,
        ShardedStateDict,
        Dict[ChunkId, ShardedTensor],
        Dict[ChunkId, ShardedTensor],
    ]:
        """ Wrap non-main ShardedTenors with LocalNonpersitentObject """
        to_load_shards = {}
        unloaded_shards = {}

        sharded_tensors, sharded_state_dict = extract_matching_values(
            sharded_state_dict, lambda v: isinstance(v, ShardedTensor)
        )

        def wrap_non_main_replicas(x):
            if isinstance(x, ShardedTensor):
                # Assign shard to be loaded or not
                if is_main_replica(x.replica_id):
                    to_load_shards[_sharded_tensor_chunk_id(x)] = x
                else:
                    unloaded_shards[_sharded_tensor_chunk_id(x)] = x
            return x

        dict_list_map_inplace(wrap_non_main_replicas, sharded_tensors)
        return sharded_tensors, sharded_state_dict, to_load_shards, unloaded_shards

    def apply_loading_parallelization(
        self, sharded_state_dict: ShardedStateDict
    ) -> Optional[SaveDistribution]:
        precomputed_distribution = determine_main_replica_uniform_distribution(
            sharded_state_dict, self.parallelization_group, True
        )
        distribute_main_replicas_with_precomputed_distribution(
            sharded_state_dict, self.parallelization_group, precomputed_distribution
        )
        if self.do_cache_distribution:
            self.cached_distribution = precomputed_distribution

        return precomputed_distribution

    def exchange_loaded_tensors_gather_object(
        self,
        loaded_tensors: Dict[ChunkId, torch.Tensor],
        unloaded_shards: Dict[ChunkId, ShardedTensor],
        precomputed_distribution: SaveDistribution = None,
        parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Dict[ChunkId, torch.Tensor]:
        """  """
        all_loaded_tensors_list = [None] * torch.distributed.get_world_size(
            group=parallelization_group
        )
        torch.distributed.all_gather_object(
            all_loaded_tensors_list, loaded_tensors, group=parallelization_group
        )
        all_loaded_tensors_list = cast(List[Dict[ChunkId, torch.Tensor]], all_loaded_tensors_list)
        all_loaded_tensors = reduce(lambda x, y: {**x, **y}, all_loaded_tensors_list)

        # Error checks
        if len(all_loaded_tensors) != sum(map(len, all_loaded_tensors_list)):
            err_msg = 'Duplicate chunk ids loaded by different ranks'
            if torch.distributed.get_rank() == 0:
                logger.error(
                    f'{err_msg}. Chunks ids by rank: {[lt.keys() for lt in all_loaded_tensors_list]}'
                )
            raise CheckpointingException(err_msg)
        if not set(unloaded_shards.keys()).issubset(all_loaded_tensors.keys()):
            missing_shards = set(unloaded_shards.keys()) - all_loaded_tensors.keys()
            raise CheckpointingException(
                f'Missing shards after fully parallel loading: {missing_shards}'
            )

        return all_loaded_tensors

    def exchange_loaded_tensors_gather_rounds(
        self,
        loaded_tensors: Dict[ChunkId, torch.Tensor],
        unloaded_shards: Dict[ChunkId, ShardedTensor],
        precomputed_distribution: SaveDistribution = None,
        parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Dict[ChunkId, torch.Tensor]:
        """  """
        # local_sh_tens = list(nested_values(sharded_state_dict))
        # local_sh_tens_by_id = {_sharded_tensor_chunk_id(sh_ten): sh_ten for sh_ten in local_sh_tens}
        shard_to_saving_rank, _, shard_to_metadata = precomputed_distribution
        local_rank = torch.distributed.get_rank(group=self.parallelization_group)

        all_loaded_tensors = dict(loaded_tensors)

        for dtype in sorted(set(map(lambda sh_ten: sh_ten.dtype, shard_to_metadata.values())), key=str):

            start = time()
            shards_by_rank: List[List[torch.Tensor]] = [
                []
                for _ in range(torch.distributed.get_world_size(group=parallelization_group))
            ]
            for shard_id, rank in shard_to_saving_rank.items():
                if shard_to_metadata[shard_id].dtype != dtype:
                    continue
                if rank == local_rank:
                    assert shard_id in loaded_tensors, (shard_id, loaded_tensors.keys())
                    shards_by_rank[rank].append(loaded_tensors[shard_id])
                else:
                    local_unloaded_sh_ten = unloaded_shards.get(shard_id)
                    if local_unloaded_sh_ten is None:
                        sh_ten = shard_to_metadata[shard_id]
                        _ten = torch.empty(
                            sh_ten.local_shape,
                            dtype=sh_ten.dtype,
                            device='cuda',
                        )
                    else:
                        local_unloaded_sh_ten.init_data('cuda')
                        _ten = local_unloaded_sh_ten.data
                        all_loaded_tensors[shard_id] = _ten
                    shards_by_rank[rank].append(_ten)

            num_rounds = max(map(len, shards_by_rank))
            for rank_shards in shards_by_rank:
                rank_shards.extend(
                    [
                        torch.empty(0, dtype=dtype, device='cuda')
                        for _ in range(num_rounds - len(rank_shards))
                    ]
                )

            torch.distributed.barrier()
            end = time()
            if torch.distributed.get_rank() == 0:
                logger.debug(f'{dtype} exchange rounds prep time took {end - start}s')
            start = time()

            for round_idx, round_tensors in enumerate(zip(*shards_by_rank)):
                torch.distributed.all_gather(
                    list(round_tensors),
                    round_tensors[local_rank],
                    group=self.parallelization_group,
                    async_op=True,
                )
            end = time()
            if torch.distributed.get_rank() == 0:
                logger.debug(
                    f'{dtype} exchange rounds all_gather schedule took {end - start}s')

        # Error checks
        if not set(unloaded_shards.keys()).issubset(all_loaded_tensors.keys()):
            missing_shards = set(unloaded_shards.keys()) - all_loaded_tensors.keys()
            raise CheckpointingException(
                f'Missing shards after fully parallel loading: {missing_shards}'
            )

        return all_loaded_tensors

    def fill_in_deferred_sharded_tensors(
        self, sharded_state_dict: ShardedStateDict, loaded_tensors: Dict[ChunkId, torch.Tensor]
    ) -> None:
        def fill_in_sharded_tensor(x):
            if isinstance(x, ShardedTensor):
                try:
                    x = loaded_tensors[_sharded_tensor_chunk_id(x)]
                except KeyError as e:
                    raise CheckpointingException(
                        f'Missing loaded tensor shard: {_sharded_tensor_chunk_id(x)}'
                    ) from e

            return x

        dict_list_map_inplace(fill_in_sharded_tensor, sharded_state_dict)

    @property
    def can_handle_sharded_objects(self):
        return self.base_strategy.can_handle_sharded_objects

    def load_tensors_metadata(self, checkpoint_dir: Path):
        self.base_strategy.load_tensors_metadata(checkpoint_dir)

    def check_backend_compatibility(self, loaded_version):
        self.base_strategy.check_backend_compatibility(loaded_version)

    def check_version_compatibility(self, loaded_version):
        self.base_strategy.check_version_compatibility(loaded_version)


def _sharded_tensor_chunk_id(sharded_tensor: ShardedTensor) -> ChunkId:
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
    sharded_state_dict: ShardedStateDict,
    parallelization_group: torch.distributed.ProcessGroup,
    is_loading: bool = False,
) -> Optional[SaveDistribution]:
    """ Computes the save distribution.

    Should be used in conjunction with `distribute_main_replicas_with_precomputed_distribution`
    which applies the computed save distribution.

    We rely on the fact that the assignment algorithm is deterministic on all ranks,
    so there is no extra communication needed after metadata exchange.

    Args:
        sharded_state_dict (ShardedStateDict): state dict to compute the distribution of
        parallelization_group (ProcessGroup): distribution will be computed
            within this process group

    Returns (SaveDistribution, optional): distribution that can be used to apply the
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
    shard_to_metadata = {}
    shards_saved_by_this_parallelization_group: Set[ChunkId] = set()
    for rank, rank_shards in enumerate(all_shards):
        for sh_ten in rank_shards:
            shard_id = _sharded_tensor_chunk_id(sh_ten)
            shard_to_ranks[shard_id].append(rank)
            if shard_id not in shard_to_size:
                shard_to_size[shard_id] = _shard_size(sh_ten)
                shard_to_metadata[shard_id] = sh_ten
            if is_main_replica(sh_ten.replica_id) or is_loading:
                shards_saved_by_this_parallelization_group.add(shard_id)

    shard_to_ranks = {
        k: v for k, v in shard_to_ranks.items() if k in shards_saved_by_this_parallelization_group
    }

    shard_to_saving_rank = distribute_chunks_to_ranks(
        shard_to_ranks, shard_to_size, len(all_shards)
    )

    return shard_to_saving_rank, shards_saved_by_this_parallelization_group, shard_to_metadata


def distribute_main_replicas_with_precomputed_distribution(
    sharded_state_dict: ShardedStateDict,
    parallelization_group: torch.distributed.ProcessGroup,
    precomputed_distribution: Optional[SaveDistribution],
):
    """ Applies the save distribution computed with `determine_main_replica_uniform_distribution`.

    Based on rank assignment, sets replica ids of the shards saved by current rank to 0
    and all the other replica ids to 1.

    Args:
        sharded_state_dict (ShardedStateDict): state dict to apply the save distribution to
        parallelization_group (ProcessGroup): distribution will be applied within this
            process group. Must match with the process group passed to
            `determine_main_replica_uniform_distribution`.
        precomputed_distribution (DistributionT): distribution computed with
            `determine_main_replica_uniform_distribution`

    Returns: None

    Example replica ids of tensors A, B, C before distribution:
    rank0: A: (0, 0, 0), B: (0, 0, 0), C: (0, 0, 0)
    rank1: A: (0, 0, 1), B: (0, 0, 1), C: (0, 0, 1)
    rank2: A: (0, 0, 2), B: (0, 0, 2), C: (0, 0, 2)

    Replicas after distribution for the example above:
    rank0: A: 0, B: 1, C: 1
    rank0: A: 1, B: 0, C: 1
    rank0: A: 1, B: 1, C: 0
    """
    if torch.distributed.get_world_size(group=parallelization_group) <= 1:
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

    shard_to_saving_rank, shards_saved_by_this_parallelization_group, _ = precomputed_distribution

    rank_within_dp_group = torch.distributed.get_rank(parallelization_group)
    for sh_ten in local_shards:
        shard_id = _sharded_tensor_chunk_id(sh_ten)
        if (
            shard_id in shards_saved_by_this_parallelization_group
            and rank_within_dp_group == shard_to_saving_rank[shard_id]
        ):
            sh_ten.replica_id = 0
        else:
            sh_ten.replica_id = 1


T = TypeVar('T')


def distribute_chunks_to_ranks(
    shard_to_ranks: Dict[T, List[int]], shard_to_size: Dict[T, int], num_ranks: int
) -> Dict[T, int]:
    """ Computes uniform distribution of workload across ranks, based on sizes.

    Currently, the assignment is greedy, based on:
    1. Firstly, the coverage of each shard
        (how many ranks the shard is available on; lower coverage is assigned first)
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
