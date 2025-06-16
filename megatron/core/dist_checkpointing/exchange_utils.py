# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Utilities for exchanging data between ranks."""

import logging
from collections import defaultdict
from functools import reduce
from itertools import zip_longest
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, TypeVar, cast

import numpy as np
import torch

from ..utils import get_pg_rank, get_pg_size
from .core import CheckpointingException
from .dict_utils import nested_values
from .mapping import ShardedStateDict, ShardedTensor, is_main_replica
from .utils import _sharded_tensor_shard_id, _ShardId, debug_time

# TODO: remove TE references once the TE bug is fixed
# Check if Transformer Engine has Float8Tensor class
HAVE_TE_FLOAT8TENSOR = False
try:
    from transformer_engine.pytorch.float8_tensor import Float8Tensor

    HAVE_TE_FLOAT8TENSOR = True
except (ImportError, ModuleNotFoundError):
    # Float8Tensor not found
    pass


def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine Float8Tensor"""
    return HAVE_TE_FLOAT8TENSOR and isinstance(tensor, Float8Tensor)


logger = logging.getLogger(__name__)


class ShardDistribution(NamedTuple):
    """Represents a distribution of ShardedTensors.

    Given distribution is valid only for a specific parallelization group,
    which is implicit here (not referenced by this class).

    Args:
        main_rank_for_shard (Dict[_ShardId, int]): specifies which rank should hold
            the main replica for a given shard
        shards_in_this_group (Set[_ShardId]): which shards have a main replica
            in this parallelization group
        shard_to_metadata (Dict[_ShardId, ShardedTensor]): maps ShardedTensor
            identifier to the original ShardedTensor
        all_ranks_for_shard (Dict[_ShardId, List[int]]): specifies which ranks
            need a given shard in a given parallelization group
    """

    main_rank_for_shard: Dict[_ShardId, int]
    shards_in_this_group: Set[_ShardId]
    shard_to_metadata: Dict[_ShardId, ShardedTensor]
    all_ranks_for_shard: Dict[_ShardId, List[int]]


def _shard_size(sh_ten: ShardedTensor):
    """Returns size in bytes of a given sharded tensor."""
    if sh_ten.flattened_range is None:
        numel = np.product(sh_ten.local_shape)
    else:
        numel = sh_ten.flattened_range.stop - sh_ten.flattened_range.start
    return numel * torch._utils._element_size(sh_ten.dtype)


def _get_empty_tensor_for_exchange(
    shard_id: _ShardId,
    needed_shards: Dict[_ShardId, ShardedTensor],
    unneeded_shards: Dict[_ShardId, ShardedTensor],
    loaded_tensors: Dict[_ShardId, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.device]]:
    """Determines the empty tensor to use for exchange.

    If shard_id is needed by this rank, it will be in the `unloaded_shards`.
    Otherwise, the metadata for this tensor can be found in `shard_to_metadata`

    Args:
        shard_id (_ShardId): shard_id that will be exchanged
        needed_shards (Dict[_ShardId, ShardedTensor]): mapping from shard ids
            to metadata for shards needed by this rank
        unneeded_shards (Dict[_ShardId, ShardedTensor]): mapping from shard ids
            to metadata for shards that can be discarded after exchange
        loaded_tensors (Dict[_ShardId, torch.Tensor]): mapping where useful tensors
            are placed in

    Returns:
        Tuple[torch.Tensor, Optional[torch.device]]: empty CUDA tensor to be exchanged,
            and the device of the original state dict tensor (if there was any)
    """
    local_unloaded_sh_ten = needed_shards.get(shard_id)
    if local_unloaded_sh_ten is None:
        orig_device = None  # this tensor will be discarded anyway
        sh_ten = unneeded_shards[shard_id]
        if sh_ten.data is None:
            sh_ten.init_data('cuda')
            tensor = sh_ten.data
            sh_ten.data = None  # won't be used. free memory
        else:
            tensor = sh_ten.data
            if tensor.device.type == 'cpu':
                tensor = torch.empty_like(tensor, device='cuda')
    else:
        local_unloaded_sh_ten.init_data('cuda')
        orig_device = local_unloaded_sh_ten.data.device
        tensor = local_unloaded_sh_ten.data
        if tensor.device.type == 'cpu':
            tensor = torch.empty_like(tensor, device='cuda')
        loaded_tensors[shard_id] = tensor
    return tensor, orig_device


T = TypeVar('T')


def distribute_shards_to_ranks(
    shard_to_ranks: Dict[T, List[int]],
    shard_to_size: Dict[T, int],
    num_ranks: int,
    cross_parallelization_group_loads: Set[T],
) -> Dict[T, int]:
    """Computes uniform distribution of workload across ranks, based on sizes.

    Currently, the assignment is greedy, based on:
    1. Cross-parallelization group dependencies (shards with main rank in another group
       are assigned at the end to make sure the distribution for load and save
       is as similar as possible).
    2. Secondly, the coverage of each shard
        (how many ranks the shard is available on; lower coverage is assigned first)
    3. Then, the size of each shard (larger size is assigned first)
    4. Finally, shard id for differentiation.

    Last step is added because we rely on the fact that
    the assignment is deterministic on all ranks.

    Args:
        shard_to_ranks (Dict[T, List[int]]): mapping of rank access to shards
        shard_to_size (Dict[T, int]): sizes of each shard
        num_ranks (int): number of ranks in the parallelization group
        cross_parallelization_group_loads (Set[T]): Shards to load that are not in the main replica

    Returns (Dict[T, int]): assignment of shard to rank (which rank should do the work
        to achieve maximal uniformity)
    """
    shard_to_ranks = {k: tuple(v) for k, v in shard_to_ranks.items()}
    shard_to_saving_rank = {}
    rank_sizes = [(0, rank) for rank in range(num_ranks)]

    # start from tensors of lowest coverage, then go by tensor size from largest (hence minus size)
    for shard_id, shard_ranks in sorted(
        shard_to_ranks.items(),
        key=lambda sh_id_ranks: (
            # 0 if rank is not in cross_parallelization_group_loads
            # which means it has higher priority
            int(sh_id_ranks[0] in cross_parallelization_group_loads),
            len(sh_id_ranks[1]),
            -shard_to_size[sh_id_ranks[0]],
            sh_id_ranks[0],
        ),
    ):
        # assign greedily to the least occupied rank
        size, rank = min((size, rank) for size, rank in rank_sizes if rank in shard_ranks)

        shard_to_saving_rank[shard_id] = rank
        rank_sizes[rank] = (size + shard_to_size[shard_id], rank)

    logger.debug(f'distribute_shards_to_ranks distribution: {rank_sizes}')

    return shard_to_saving_rank


def determine_main_replica_uniform_distribution(
    sharded_state_dict: ShardedStateDict,
    parallelization_group: torch.distributed.ProcessGroup,
    ignore_groups: bool = False,
) -> Optional[ShardDistribution]:
    """Computes the save distribution.

    Should be used in conjunction with `distribute_main_replicas_with_precomputed_distribution`
    which applies the computed save distribution.

    We rely on the fact that the assignment algorithm is deterministic on all ranks,
    so there is no extra communication needed after metadata exchange.

    Args:
        sharded_state_dict (ShardedStateDict): state dict to compute the distribution of
        parallelization_group (ProcessGroup): distribution will be computed
            within this process group
        ignore_groups (bool, optional): whether the distribution defines groups.
            This option is primarily used during loading, as it ensures that all replicas,
            including non-main ones, are loaded by this parallelization group
            Defaults to False.

    Returns (ShardDistribution, optional): distribution that can be used to apply the
        parallelization. Returns None if the process_group is trivial (1 rank)

    """
    if parallelization_group is None:
        parallelization_group = torch.distributed.group.WORLD
    group_size = get_pg_size(group=parallelization_group)
    if group_size <= 1:
        return
    local_shards = list(
        sh_base
        for sh_base in nested_values(sharded_state_dict)
        if isinstance(sh_base, ShardedTensor)
    )
    local_shards_no_data = [ten.without_data() for ten in local_shards]

    all_shards = [None] * get_pg_size(group=parallelization_group)
    torch.distributed.all_gather_object(
        all_shards, local_shards_no_data, group=parallelization_group
    )

    shard_to_ranks = defaultdict(list)
    shard_to_size = {}
    shard_to_metadata = {}
    group_has_main_replica: Set[_ShardId] = set()
    group_has_non_main_replica: Set[_ShardId] = set()

    for rank, rank_shards in enumerate(all_shards):
        for sh_ten in rank_shards:
            shard_id = _sharded_tensor_shard_id(sh_ten)
            shard_to_ranks[shard_id].append(rank)
            if shard_id not in shard_to_size:
                shard_to_size[shard_id] = _shard_size(sh_ten)
                shard_to_metadata[shard_id] = sh_ten
            if is_main_replica(sh_ten.replica_id):
                group_has_main_replica.add(shard_id)
            else:
                group_has_non_main_replica.add(shard_id)

    # we always include all main replicas, and non-main only if `ignore_groups`
    shards_in_this_group: Set[_ShardId] = group_has_main_replica
    if ignore_groups:
        shards_in_this_group = shards_in_this_group | group_has_non_main_replica
    # cross-parallel-group references are empty if `not ignore_groups`,
    # otherwise it's `group_has_non_main_replica - group_has_main_replica`
    cross_parallelization_group_loads = shards_in_this_group - group_has_main_replica

    # Filter out shards that don't belong to this group
    shard_to_ranks = {k: v for k, v in shard_to_ranks.items() if k in shards_in_this_group}

    shard_to_saving_rank = distribute_shards_to_ranks(
        shard_to_ranks, shard_to_size, len(all_shards), cross_parallelization_group_loads
    )

    return ShardDistribution(
        shard_to_saving_rank, shards_in_this_group, shard_to_metadata, shard_to_ranks
    )


@torch.no_grad()
@debug_time(f"exchange_loaded_tensors_gather_rounds", logger)
def exchange_loaded_tensors_gather_rounds(
    loaded_tensors: Dict[_ShardId, torch.Tensor],
    unloaded_shards: Dict[_ShardId, ShardedTensor],
    shard_distribution: ShardDistribution = None,
    parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Dict[_ShardId, torch.Tensor]:
    """Exchange the tensors loaded by different ranks with several all_gather calls.

    Groups tensors by dtype, divide tensors that will be exchanged into rounds
    and execute all_gather for tensors from each round.

    Note: the loading is distributed across ranks based on total loaded size
    in bytes, so there is no guarantee that number of rounds needed for each
    rank will be similar, which might result in a lot of almost empty
    all_gathers. The solution would be to group all tensors into a one
    bytes tensor and do a single all_gather (with similarly sized messages).

    Args:
        loaded_tensors (Dict[_ShardId, torch.Tensor]): mapping from ShardedTensor
            shard ids to tensors already loaded by this rank.
        unloaded_shards (Dict[_ShardId, torch.Tensor]): mapping from ShardedTensor
            shard ids to ShardedTensors that aren't loaded yet.
        shard_distribution (ShardDistribution): distribution of all shards
        parallelization_group (ProcessGroup, optional): process group used for load
            distribution. Tensors will be exchanged within this group

    Returns:
        Dict[_ShardId, torch.Tensor]: dictionary mapping shard ids to tensors
            needed by this rank to load a given state dict. Includes
            previously loaded tensors (from `loaded_tensors` input)
    """
    if parallelization_group is None:
        parallelization_group = torch.distributed.group.WORLD
    main_rank_for_shard, _, shard_to_metadata, all_ranks_for_shard = shard_distribution
    local_rank = get_pg_rank(group=parallelization_group)

    all_loaded_tensors = dict(loaded_tensors)

    # Group by dtype so that we all_gather tensors of the same dtype
    for dtype in sorted(set(map(lambda sh_ten: sh_ten.dtype, shard_to_metadata.values())), key=str):

        with debug_time(f"dtype_{dtype}"):
            # shards_by_rank maps rank to tensors loaded by this rank
            shards_by_rank: List[List[torch.Tensor]] = [
                [] for _ in range(get_pg_size(group=parallelization_group))
            ]
            for shard_id, rank in main_rank_for_shard.items():
                if len(all_ranks_for_shard[shard_id]) == 1:
                    assert all_ranks_for_shard[shard_id][0] == main_rank_for_shard[shard_id], (
                        f'When there is only 1 ranks that needs a given shard,'
                        f' it should be the loading rank.'
                        f' Got: needs [{all_ranks_for_shard[shard_id][0]}]'
                        f' vs loads [{main_rank_for_shard[shard_id]}]'
                    )
                    # Skipping the exchange since only the loading rank needs this tensor
                    # TODO: we can employ some optimizations even for `len(shard_to_ranks) > 1`
                    #  case, e.g. P2P exchange. Currently handling this case saves most of the
                    #  work though.
                    continue
                if shard_to_metadata[shard_id].dtype == dtype:
                    shards_by_rank[rank].append(shard_id)

            # Transpose `shards_by_rank` to form exchange rounds
            shards_by_round = zip_longest(*shards_by_rank, fillvalue=None)
            for round_idx, round_shard_ids in enumerate(shards_by_round):
                round_tensors = []
                orig_devices = {}
                for rank, shard_id in enumerate(round_shard_ids):
                    if shard_id is None:
                        # if no more useful data, the given rank will exchange empty tensor
                        local_ten = torch.empty(0, dtype=dtype, device='cuda')
                        orig_device = None
                    else:
                        assert isinstance(shard_id, tuple), type(shard_id)
                        if rank == local_rank:
                            assert shard_id in all_loaded_tensors, (
                                shard_id,
                                all_loaded_tensors.keys(),
                            )
                            orig_device = all_loaded_tensors[shard_id]
                            all_loaded_tensors[shard_id] = all_loaded_tensors[shard_id].cuda()
                            local_ten = all_loaded_tensors[shard_id]
                        else:
                            local_ten, orig_device = _get_empty_tensor_for_exchange(
                                shard_id, unloaded_shards, shard_to_metadata, all_loaded_tensors
                            )
                        # Because of a TE bug, we have to exchange a nominal dtype instead of FP8
                        # It's ok to keep the nominal dtype after exchange, because TE will handle
                        # this during state dict load.
                        # TODO: remove it once the bug is fixed
                        if is_float8tensor(local_ten):
                            try:
                                local_ten = local_ten.from_float8()
                            except Exception as e:
                                local_ten = local_ten.dequantize()
                            all_loaded_tensors[shard_id] = local_ten

                    round_tensors.append(local_ten)
                    if orig_device is not None:
                        orig_devices[shard_id] = orig_device

                torch.distributed.all_gather(
                    list(round_tensors),
                    round_tensors[local_rank],
                    group=parallelization_group,
                    async_op=False,
                )

                # Move tensors back to CPU if originally was on CPU
                for shard_id, orig_device in orig_devices.items():
                    all_loaded_tensors[shard_id] = all_loaded_tensors[shard_id].to(orig_device)

                del round_tensors  # remove tensor references

    return all_loaded_tensors


def exchange_loaded_tensors_gather_object(
    loaded_tensors: Dict[_ShardId, torch.Tensor],
    unloaded_shards: Dict[_ShardId, ShardedTensor],
    shard_distribution: ShardDistribution,
    parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Dict[_ShardId, torch.Tensor]:
    """Exchange the tensors loaded by different ranks with a simple all_gather_object call.

    This version can be used for debugging purposes do to its simplistic
    implementation. Shouldn't be used if performance is important.

    Args:
        loaded_tensors (Dict[_ShardId, torch.Tensor]): mapping from ShardedTensor
            shard ids to tensors already loaded by this rank.
        unloaded_shards (Dict[_ShardId, torch.Tensor]): mapping from ShardedTensor
            shard ids to ShardedTensors that aren't loaded yet.
        shard_distribution (ShardDistribution): distribution of all shards
        parallelization_group (ProcessGroup, optional): process group used for load
            distribution. Tensors will be exchanged within this group

    Returns:
        Dict[_ShardId, torch.Tensor]: dictionary mapping shard ids to tensors
            needed by this rank to load a given state dict. Includes
            previously loaded tensors (from `loaded_tensors` input)

    """
    all_loaded_tensors_list = [None] * torch.distributed.get_world_size(group=parallelization_group)
    torch.distributed.all_gather_object(
        all_loaded_tensors_list, loaded_tensors, group=parallelization_group
    )
    all_loaded_tensors_list = cast(List[Dict[_ShardId, torch.Tensor]], all_loaded_tensors_list)
    all_loaded_tensors = reduce(lambda x, y: {**x, **y}, all_loaded_tensors_list)

    # Error checks
    if len(all_loaded_tensors) != sum(map(len, all_loaded_tensors_list)):
        err_msg = 'Duplicate shard ids loaded by different ranks'
        if torch.distributed.get_rank() == 0:
            logger.error(
                f'{err_msg}. Shards ids by rank:'
                f' {[lt.keys() for lt in all_loaded_tensors_list]}'
            )
        raise CheckpointingException(err_msg)

    return all_loaded_tensors


def exchange_loaded_objects_gather_object(
    loaded_objects: Dict[_ShardId, Any]
) -> Dict[_ShardId, Any]:
    """Exchange the objects loaded by different ranks with a simple all_gather_object call.

    Args:
        loaded_objects (Dict[_ShardId, Any]): mapping from shard ids to objects
          already loaded by this rank.

    Returns:
        Dict[_ShardId, Any]: dictionary mapping shard ids to objects needed by this rank to
         load a given state dict.
    """
    all_loaded_objects_list = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(all_loaded_objects_list, loaded_objects, group=None)
    all_loaded_objects_list = cast(List[Dict[_ShardId, Any]], all_loaded_objects_list)
    all_loaded_objects = reduce(lambda x, y: {**x, **y}, all_loaded_objects_list)

    # Error checks
    if len(all_loaded_objects) != sum(map(len, all_loaded_objects_list)):
        err_msg = 'Duplicate shard ids loaded by different ranks'
        if torch.distributed.get_rank() == 0:
            logger.error(
                f'{err_msg}. Shards ids by rank:'
                f' {[lt.keys() for lt in all_loaded_objects_list]}'
            )
        raise CheckpointingException(err_msg)

    return all_loaded_objects


@torch.no_grad()
@debug_time("exchange_loaded_tensors_broadcast", logger)
def exchange_loaded_tensors_broadcast(
    loaded_tensors: Dict[_ShardId, torch.Tensor],
    unloaded_shards: Dict[_ShardId, ShardedTensor],
    shard_distribution: ShardDistribution,
    parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Dict[_ShardId, torch.Tensor]:
    """Exchange the tensors loaded by different ranks by a series of broadcasts.

    For each rank for each loaded tensor do a broadcast to the whole group.
    A reasonable tradeoff in terms of performance and simplicity.

    Args:
        loaded_tensors (Dict[_ShardId, torch.Tensor]): mapping from ShardedTensor
            shard ids to tensors already loaded by this rank.
        unloaded_shards (Dict[_ShardId, ShardedTensor]): mapping from ShardedTensor
            shard ids to ShardedTensors that aren't loaded yet.
        shard_distribution (ShardDistribution): distribution of all shards
        parallelization_group (ProcessGroup, optional): process group used for load
            distribution. Tensors will be exchanged within this group

    Returns:
        Dict[_ShardId, torch.Tensor]: dictionary mapping shard ids to tensors
            needed by this rank to load a given state dict. Includes
            previously loaded tensors (from `loaded_tensors` input)
    """
    main_rank_for_shard, _, shard_to_metadata, all_ranks_for_shard = shard_distribution
    local_rank = torch.distributed.get_rank(group=parallelization_group)

    all_loaded_tensors = dict(loaded_tensors)

    for idx, (shard_id, rank) in enumerate(main_rank_for_shard.items()):
        if len(all_ranks_for_shard[shard_id]) == 1:
            assert all_ranks_for_shard[shard_id][0] == main_rank_for_shard[shard_id], (
                f'When there is only 1 ranks that needs a given shard,'
                f' it should be the loading rank.'
                f'Got: needs [{all_ranks_for_shard[shard_id][0]}]'
                f' vs loads [{main_rank_for_shard[shard_id]}]'
            )
            # Skipping the exchange since only the loading rank needs this tensor
            # TODO: we can employ some optimizations even for `len(shard_to_ranks) > 1` case,
            #  e.g. P2P exchange. Currently handling this case saves most of the work though.
            continue
        if rank == local_rank:
            assert shard_id in all_loaded_tensors, (shard_id, all_loaded_tensors.keys())
            orig_device = all_loaded_tensors[shard_id].device
            local_ten = all_loaded_tensors[shard_id].cuda()
        else:
            local_ten, orig_device = _get_empty_tensor_for_exchange(
                shard_id, unloaded_shards, shard_to_metadata, all_loaded_tensors
            )

        # Because of a TE bug, we have to exchange a nominal dtype instead of FP8
        # It's ok to keep the nominal dtype after exchange, because TE will handle
        # this during state dict load.
        # TODO: remove it once the bug is fixed
        if is_float8tensor(local_ten):
            try:
                local_ten = local_ten.from_float8()
            except Exception as e:
                local_ten = local_ten.dequantize()
            all_loaded_tensors[shard_id] = local_ten

        global_src_rank = (
            rank
            if parallelization_group == None
            else torch.distributed.get_global_rank(parallelization_group, rank)
        )
        # We can do async_op=True only if there is no CPU-copy follow-up
        torch.distributed.broadcast(
            local_ten,
            src=global_src_rank,
            group=parallelization_group,
            async_op=orig_device is None,
        )
        # Move tensor back to CPU if originally was on CPU
        if orig_device is not None:
            all_loaded_tensors[shard_id] = local_ten.to(orig_device)
        del local_ten

    return all_loaded_tensors


def exchange_by_distribution(
    loaded_tensors: Dict[_ShardId, torch.Tensor],
    unloaded_shards: Dict[_ShardId, ShardedTensor],
    shard_distribution: ShardDistribution,
    parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
    exchange_algo='broadcast',
) -> Dict[_ShardId, torch.Tensor]:
    """Exchange tensors loaded by different ranks using the specified exchange_algo.

    Args:
        loaded_tensors (Dict[_ShardId, torch.Tensor]): mapping from ShardedTensor
            shard ids to tensors already loaded by this rank.
        unloaded_shards (Dict[_ShardId, ShardedTensor]): mapping from ShardedTensor
            shard ids to ShardedTensors that aren't loaded yet.
        shard_distribution (ShardDistribution): distribution of all shards
        parallelization_group (ProcessGroup, optional): process group used for load
            distribution. Tensors will be exchanged within this group
        exchange_algo (str): The algorithm used for performing exchanges.
            Defaults to 'broadcast'.

    Returns:
        Dict[_ShardId, torch.Tensor]: dictionary mapping shard ids to tensors
            needed by this rank to load a given state dict. Includes
            previously loaded tensors (from `loaded_tensors` input)
    """

    assert shard_distribution is not None, 'Expecting distribution to perform exchange'
    if exchange_algo == 'gather_object':
        exchange_fn = exchange_loaded_tensors_gather_object
    elif exchange_algo == 'gather_rounds':
        exchange_fn = exchange_loaded_tensors_gather_rounds
    elif exchange_algo == 'broadcast':
        exchange_fn = exchange_loaded_tensors_broadcast
    else:
        raise NotImplementedError(f'Unrecognized gather algorithm: {exchange_algo}')
    return exchange_fn(loaded_tensors, unloaded_shards, shard_distribution, parallelization_group)
