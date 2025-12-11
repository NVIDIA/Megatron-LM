# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
from pathlib import Path
from time import time
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import Metadata

from megatron.core.dist_checkpointing import ShardedObject, ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.dict_utils import (
    dict_list_map_inplace,
    extract_matching_values,
    merge,
    nested_values,
)
from megatron.core.dist_checkpointing.exchange_utils import (
    ShardDistribution,
    determine_main_replica_uniform_distribution,
    exchange_by_distribution,
    exchange_loaded_objects_gather_object,
)
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, StateDict, is_main_replica
from megatron.core.dist_checkpointing.strategies.base import (
    AsyncSaveShardedStrategy,
    LoadShardedStrategy,
    SaveShardedStrategy,
)
from megatron.core.dist_checkpointing.utils import (
    _sharded_object_id,
    _sharded_tensor_shard_id,
    _ShardId,
    debug_time,
)
from megatron.core.dist_checkpointing.validation import (
    determine_global_metadata,
    validate_sharding_integrity,
)
from megatron.core.utils import get_pg_rank, get_pg_size

logger = logging.getLogger(__name__)

T = TypeVar('T', ShardedObject, ShardedTensor)


class FullyParallelSaveStrategyWrapper(AsyncSaveShardedStrategy):
    """Wraps arbitrary strategy and distributes the save during `save`.

    The save distribution happens without any *data* communication.
    Only the *metadata* is exchanged and based on data replication on different
    ranks, we try to distribute the save as uniformly as possible.

    This wrapper assumes, that setting `replica_id` to 0 will make the
    underlying strategy do the saving on current rank. All the other `replica_id`s
    are set to 1.

    Currently, the save distribution is realized with a greedy algorithm
    described in `distribute_shards_to_ranks`.

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
        if parallelization_group is None:
            parallelization_group = torch.distributed.group.WORLD
        self.parallelization_group = parallelization_group
        self.do_cache_distribution = do_cache_distribution

        self.cached_distribution: Optional[ShardDistribution] = None

    def async_save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        if not isinstance(self.base_strategy, AsyncSaveShardedStrategy):
            raise CheckpointingException(
                f'Cannot apply async_save to non-async base strategy {self.base_strategy}'
            )
        self.apply_saving_parallelization(sharded_state_dict)
        return self.base_strategy.async_save(sharded_state_dict, checkpoint_dir)

    def save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        self.apply_saving_parallelization(sharded_state_dict)
        return self.base_strategy.save(sharded_state_dict, checkpoint_dir)

    def apply_saving_parallelization(self, sharded_state_dict: ShardedStateDict) -> None:
        """Distributes the save across ranks by exchanging metadata.

        Exchanges metadata from the state dict and computes the uniform
        (as close as possible) distribution of saves among the ranks.

        If `self.do_cache_distribution` is True, caches the distribution between
        the calls and subsequent distributions happen without any inter-rank
        communication.

        Args:
            sharded_state_dict (ShardedStateDict): state dict to distribute the saving

        Returns: None
        """
        start = time()
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
            validate_sharding_integrity(determine_global_metadata(sharded_state_dict)[1])
        if self.do_cache_distribution:
            self.cached_distribution = precomputed_distribution
        end = time()
        logger.debug(f"parallel save sharding, time: {end - start}")

    @property
    def can_handle_sharded_objects(self):
        return self.base_strategy.can_handle_sharded_objects


class FullyParallelLoadStrategyWrapper(LoadShardedStrategy):
    """Wraps arbitrary load strategy and distributes the load during `load`.

    See `load` method docs for details.

    Args:
        strategy (LoadShardedStrategy): base strategy to wrap
        parallelization_group (ProcessGroup, optional): process group to use for load
            distribution. Note that this doesn't have to match exactly the
            data distribution, but should cover the replication pattern
            to maximize performance. Defaults to the whole world.
            In most cases, it's recommended to set it to the DP group.
        do_cache_distribution (bool, optional): whether to cache the load distribution
            from previous calls. Should be set to True only if the state dict
            structure between the calls is always the same. Defaults to False,
            since the loading in general happens only once during training.
            Note that the load distribution *cannot* be reused as a save distribution,
            because save/load is not fully symmetrical.
        exchange_algo (str): algorithm to use for exchanging the data.
            Options:
            - broadcast - each rank broadcasts individual tensors to others
            - gather_object (default) - ranks all_gather_object the whole loaded state dicts
            - gather_rounds (default) - ranks all gather individual tensors in rounds
            See method docs for more details.
    """

    def __init__(
        self,
        strategy: LoadShardedStrategy,
        parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
        do_cache_distribution: bool = False,
        exchange_algo: str = 'broadcast',
    ):
        super().__init__()
        self.base_strategy = strategy
        if parallelization_group is None:
            parallelization_group = (
                dist.GroupMember.WORLD
            )  # explicit group needed for torch.distributed.get_global_rank call
        self.parallelization_group = parallelization_group
        self.do_cache_distribution = do_cache_distribution
        self.exchange_algo = exchange_algo

        self.cached_distribution: Optional[ShardDistribution] = None
        self.cached_global_metadata: Optional[Metadata] = None

    @debug_time("FullyParallelLoadStrategyWrapper.load", logger)
    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Distributes the load and calls underlying strategy only for parts of the state dict.

        Steps:
        1. Load metadata is exchanged between the ranks in the parallelization group.
        2. Each rank deterministically plans the load for the whole workload
            so that the loads are as uniform as possible.
        3. Each ranks loads its planned shard of the checkpoint.
        4. All ranks exchange the loaded shards.

        Internode communication is involved in steps (1) (with metadata)
        and (4) (with actual data). Storage interaction is involved in step (3).

        Currently, the load distribution (step 2) is realized with a greedy algorithm
        described in `distribute_shards_to_ranks` (same as for saving distribution).

        Currently, the shards are all gathered between all ranks in the parallelization
        group. This might not be optimal (some ranks do not need all tensors),
        but it's a reasonable approximation for an optimal exchange in most scenarios.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict to load
            checkpoint_dir (Path): checkpoint directory to load from

        Returns:
            StateDict: loaded state dict. The state dict should be equivalent to
            a state dict that would be loaded with the underlying strategy
            without this wrapper.
        """

        loaded_state_dict = {}

        if get_pg_size(self.parallelization_group) <= 1:
            return self.base_strategy.load(sharded_state_dict, checkpoint_dir)

        # Step 1 and 2: exchange load metadata and distribute the load
        with debug_time("self.apply_loading_parallelization", logger):
            precomputed_distribution: ShardDistribution | None = self.apply_loading_parallelization(
                sharded_state_dict
            )
            assert (
                precomputed_distribution is not None
            ), 'Expecting non-trivial distribution for non-trivial parallelization group'

        # Step 3: load part of the checkpoint.
        # Load only sharded objects first. ShardedTensors will be loaded separately
        # so that we can keep track of sharded tensors loaded by this rank
        (sharded_tensors, sharded_state_dict, to_load_shards, unloaded_shards) = (
            self._defer_loading_sharded_tensors(sharded_state_dict)
        )

        (sharded_objects, sharded_state_dict, to_load_objects, unloaded_objects) = (
            self._defer_loading_sharded_objects(sharded_state_dict)
        )

        assert (
            len(sharded_state_dict) == 0
        ), "sharded_state_dict is not empty after deferring tensors and objects"
        with debug_time("base_load_ShardedObjects", logger):
            # Load sharded objects first
            loaded_objects = self.base_strategy.load(to_load_objects, checkpoint_dir)

        with debug_time("base_load_ShardedTensors", logger):
            # Load sharded tensors separately
            loaded_tensors = self.base_strategy.load(to_load_shards, checkpoint_dir)

        with debug_time("self.exchange_loaded_tensors", logger):

            # Step 4: exchange data between ranks
            logger.debug(f'Applying parallel load with algo {self.exchange_algo}')
            all_loaded_tensors = exchange_by_distribution(
                loaded_tensors,
                unloaded_shards,
                precomputed_distribution,
                self.parallelization_group,
                self.exchange_algo,
            )
            if not set(unloaded_shards.keys()).issubset(all_loaded_tensors.keys()):
                missing_shards = set(unloaded_shards.keys()) - all_loaded_tensors.keys()
                raise CheckpointingException(
                    f'Missing shards after fully parallel loading: {missing_shards}'
                )

            with debug_time("torch.cuda.synchronize", logger):
                torch.cuda.synchronize()

        all_loaded_objects = exchange_loaded_objects_gather_object(loaded_objects)

        if not set(unloaded_objects.keys()).issubset(all_loaded_objects.keys()):
            missing_object_shards = set(unloaded_objects.keys()) - all_loaded_objects.keys()
            raise CheckpointingException(
                f'Missing object shards after fully parallel loading: {missing_object_shards}'
            )
        torch.cuda.synchronize()

        self.fill_in_deferred_sharded_tensors(sharded_tensors, all_loaded_tensors)
        self.fill_in_deferred_sharded_objects(sharded_objects, all_loaded_objects)

        merge(loaded_state_dict, sharded_objects)
        merge(loaded_state_dict, sharded_tensors)
        if hasattr(self.base_strategy, "cached_global_metadata"):
            self.cached_global_metadata = self.base_strategy.cached_global_metadata
        return loaded_state_dict

    @staticmethod
    def _defer_loading_sharded_objects(
        sharded_state_dict: ShardedStateDict,
    ) -> Tuple[
        ShardedStateDict,
        ShardedStateDict,
        Dict[_ShardId, ShardedObject],
        Dict[_ShardId, ShardedObject],
    ]:
        return _defer_loading_sharded_items(sharded_state_dict, ShardedObject, _sharded_object_id)

    @staticmethod
    def _defer_loading_sharded_tensors(
        sharded_state_dict: ShardedStateDict,
    ) -> Tuple[
        ShardedStateDict,
        ShardedStateDict,
        Dict[_ShardId, ShardedTensor],
        Dict[_ShardId, ShardedTensor],
    ]:
        return _defer_loading_sharded_items(
            sharded_state_dict, ShardedTensor, _sharded_tensor_shard_id
        )

    @staticmethod
    def fill_in_deferred_sharded_objects(
        sharded_state_dict: ShardedStateDict, loaded_objects: Dict[_ShardId, Any]
    ) -> None:
        """Fill in objects not loaded by current rank with objects from `loaded_objects` map.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict to fill in.
                ShardedObjects are completely replaced with corresponding objects.
            loaded_objects (Dict[_ShardId, Any]): dict allowing to map
                ShardedObject from the sharded_state_dict to loaded objects.

        Returns:
            None
        """
        _fill_in_deferred_sharded_items(
            sharded_state_dict, loaded_objects, ShardedObject, _sharded_object_id
        )

    @staticmethod
    def fill_in_deferred_sharded_tensors(
        sharded_state_dict: ShardedStateDict, loaded_tensors: Dict[_ShardId, torch.Tensor]
    ) -> None:
        """Fill in tensors not loaded by current rank with tensors from `loaded_tensors` map.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict to fill in.
                ShardedTensors are completely replaced with corresponding torch.Tensors.
            loaded_tensors (Dict[_ShardId, torch.Tensor]): dict allowing to map
                ShardedTensor from the sharded_state_dict to loaded tensors.

        Returns:
            None
        """
        _fill_in_deferred_sharded_items(
            sharded_state_dict, loaded_tensors, ShardedTensor, _sharded_tensor_shard_id
        )

    def apply_loading_parallelization(
        self, sharded_state_dict: ShardedStateDict
    ) -> Optional[ShardDistribution]:
        """Distributes the load across ranks by exchanging metadata.

        Exchanges metadata from the state dict and computes the uniform
        (as close as possible) distribution of loads among the ranks.
        Marks ShardedTensors to be loaded by the current rank with replica_id 0
        (and others with non 0 values).

        If `self.do_cache_distribution` is True, caches the distribution between
        the calls and subsequent distributions happen without any inter-rank
        communication.

        Args:
            sharded_state_dict (ShardedStateDict): state dict to distribute the loading

        Returns:
            ShardDistribution (optional): the computed loading distribution
        """
        if self.do_cache_distribution and self.cached_distribution is not None:
            logger.debug(f'Apply *cached* load parallelization')
            precomputed_distribution = self.cached_distribution
        else:
            logger.debug(f'Apply load parallelization')
            precomputed_distribution = determine_main_replica_uniform_distribution(
                sharded_state_dict, self.parallelization_group, True
            )

        distribute_main_replicas_with_precomputed_distribution(
            sharded_state_dict, self.parallelization_group, precomputed_distribution
        )
        if self.do_cache_distribution:
            self.cached_distribution = precomputed_distribution

        return precomputed_distribution

    @property
    def can_handle_sharded_objects(self):
        return self.base_strategy.can_handle_sharded_objects

    def load_tensors_metadata(self, checkpoint_dir: Path):
        return self.base_strategy.load_tensors_metadata(checkpoint_dir)

    def load_sharded_metadata(self, checkpoint_dir: Path):
        return self.base_strategy.load_sharded_metadata(checkpoint_dir)

    def check_backend_compatibility(self, loaded_version):
        return self.base_strategy.check_backend_compatibility(loaded_version)

    def check_version_compatibility(self, loaded_version):
        return self.base_strategy.check_version_compatibility(loaded_version)


def distribute_main_replicas_with_precomputed_distribution(
    sharded_state_dict: ShardedStateDict,
    parallelization_group: torch.distributed.ProcessGroup,
    precomputed_distribution: Optional[ShardDistribution],
):
    """Applies the save distribution computed with `determine_main_replica_uniform_distribution`.

    Based on rank assignment, sets replica ids of the shards saved by current rank to 0
    and all the other replica ids to 1.

    Args:
        sharded_state_dict (ShardedStateDict): state dict to apply the save distribution to
        parallelization_group (ProcessGroup): distribution will be applied within this
            process group. Must match with the process group passed to
            `determine_main_replica_uniform_distribution`.
        precomputed_distribution (ShardDistribution): distribution computed with
            `determine_main_replica_uniform_distribution`

    Returns: None

    Example replica ids of tensors A, B, C before distribution:
    rank0: A: (0, 0, 0), B: (0, 0, 0), C: (0, 0, 0)
    rank1: A: (0, 0, 1), B: (0, 0, 1), C: (0, 0, 1)
    rank2: A: (0, 0, 2), B: (0, 0, 2), C: (0, 0, 2)

    Replicas after distribution for the example above:
    rank0: A: 0, B: 1, C: 1
    rank1: A: 1, B: 0, C: 1
    rank2: A: 1, B: 1, C: 0
    """
    if parallelization_group is None:
        parallelization_group = torch.distributed.group.WORLD
    if get_pg_size(group=parallelization_group) <= 1:
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

    rank_within_dp_group = get_pg_rank(group=parallelization_group)
    for sh_ten in local_shards:
        shard_id = _sharded_tensor_shard_id(sh_ten)
        if (
            shard_id in precomputed_distribution.shards_in_this_group
            and rank_within_dp_group == precomputed_distribution.main_rank_for_shard[shard_id]
        ):
            sh_ten.replica_id = 0
        else:
            sh_ten.replica_id = 1


def _defer_loading_sharded_items(
    sharded_state_dict: ShardedStateDict, item_type: type, shard_id_func: Callable[[T], _ShardId]
) -> Tuple[ShardedStateDict, ShardedStateDict, Dict[_ShardId, T], Dict[_ShardId, T]]:
    """Divides state dict into parts loaded by this vs other ranks.

    Args:
        sharded_state_dict (ShardedStateDict): state dict with sharded items
            that will be divided.
        item_type: The type of sharded item (ShardedObject or ShardedTensor)
        shard_id_func: Function to get the shard ID for the item type

    Returns: a tuple of:
        - ShardedStateDict: sub-state dict only with sharded items
        - ShardedStateDict: sub-state dict with non-sharded items
        - Dict[_ShardId, T]: mapping from shard id to items loaded by *this* rank
        - Dict[_ShardId, T]: mapping from shard id to items loaded by *other* ranks
    """
    to_load_shards = {}
    unloaded_shards = {}

    sharded_items, remaining_state_dict = extract_matching_values(
        sharded_state_dict, lambda v: isinstance(v, item_type)
    )

    def wrap_non_main_replicas(x: Any) -> Any:
        if isinstance(x, item_type):
            shard_id = shard_id_func(x)
            if is_main_replica(x.replica_id):
                to_load_shards[shard_id] = x
            else:
                unloaded_shards[shard_id] = x
        return x

    dict_list_map_inplace(wrap_non_main_replicas, sharded_items)
    return sharded_items, remaining_state_dict, to_load_shards, unloaded_shards


def _fill_in_deferred_sharded_items(
    sharded_state_dict: ShardedStateDict,
    loaded_items: Dict[_ShardId, Any],
    item_type: type,
    shard_id_func: Callable[[T], _ShardId],
) -> None:
    """Helper function to fill in items not loaded by current rank."""

    def fill_in_sharded_item(x: Any) -> Any:
        if isinstance(x, item_type):
            try:
                x = loaded_items[shard_id_func(x)]
            except KeyError as e:
                raise CheckpointingException(
                    f'Missing loaded item shard: {shard_id_func(x)}'
                ) from e
        return x

    dict_list_map_inplace(fill_in_sharded_item, sharded_state_dict)
