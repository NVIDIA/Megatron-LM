# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Utilities for transforming state_dict, including a tensor-aware implementation."""

import logging
from time import time
from typing import Any, Callable, Optional

import torch

from .dict_utils import dict_list_map_inplace, extract_matching_values, merge, nested_values
from .exchange_utils import determine_main_replica_uniform_distribution, exchange_by_distribution
from .mapping import (
    CommonStateDict,
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
    StateDict,
    apply_factories,
    apply_factory_merges,
)
from .utils import (
    _sharded_object_id,
    _sharded_tensor_shard_id,
    extract_nonpersistent,
    extract_sharded_base,
)
from .validation import determine_global_metadata, validate_sharding_integrity

logger = logging.getLogger(__name__)


def save_preprocess(
    sharded_state_dict: ShardedStateDict,
    validate_access_integrity: bool = True,
    preprocess_common_before_consistancy_check: Callable[[CommonStateDict], StateDict] = None,
):
    """Preprocesses the given state dictionary by applying factories,
    discarding non-persistent data and extracting the common state dictionary.
    Optionally, it can validate sharding integrity.

    Args:
        sharded_state_dict (ShardedStateDict): The initial state dictionary to be preprocessed.
        validate_access_integrity (bool): If True, triggers validation of sharding integrity.
        preprocess_common_before_consistancy_check (callable, None): A callable function
            that will preprocess the common state dict (i.e can be used  to remove keys
            that we expect to be different in the state dict)

    Returns:
        Tuple[ShardedStateDict, dict]:
            The preprocessed sharded state dictionary and the common state dictionary.
    """
    apply_factories(sharded_state_dict)
    _, sharded_state_dict = extract_nonpersistent(sharded_state_dict)
    sharded_part, common_state_dict = extract_sharded_base(sharded_state_dict)
    if validate_access_integrity:
        preprocessed_common_state_dict = common_state_dict
        if preprocess_common_before_consistancy_check:
            preprocessed_common_state_dict = preprocess_common_before_consistancy_check(
                common_state_dict
            )
        validate_sharding_integrity(
            determine_global_metadata(sharded_part)[1],
            common_state_dict=preprocessed_common_state_dict,
        )
    return sharded_part, common_state_dict


def load_preprocess(sharded_state_dict: ShardedStateDict):
    """Preprocesses the given state dictionary by applying factories
    and extracting non-persistent data, without modifying the original dictionary.

    Args:
        sharded_state_dict (ShardedStateDict):
            The initial state dictionary to be processed (remains unchanged).

    Returns:
        Tuple[ShardedStateDict, dict, dict]:
            - A preprocessed copy of the sharded state dictionary.
            - A dictionary containing non-persistent state data.
            - A dictionary of `ShardedTensorFactory` instances.
    """
    # Create a copy of sharded_state_dict as the passed in state dict may have
    # references that prevent tensors from being deallocated
    sharded_state_dict, _ = extract_matching_values(sharded_state_dict, lambda x: True)

    sh_ten_factories, _ = extract_matching_values(
        sharded_state_dict,
        lambda x: isinstance(x, ShardedTensorFactory),
        return_lists_as_dicts=True,
    )
    apply_factories(sharded_state_dict)

    # Data inside sh_ten_factories no longer needed so delete them to reduce memory usage
    dict_list_map_inplace(ShardedTensorFactory.without_data, sh_ten_factories)
    # Non-persistent objects
    nonpersistent_state_dict, sharded_state_dict = extract_nonpersistent(sharded_state_dict)
    dict_list_map_inplace(lambda o: o.unwrap(), nonpersistent_state_dict)
    return sharded_state_dict, nonpersistent_state_dict, sh_ten_factories


def prepare_state_dict_for_save(
    sharded_state_dict: ShardedStateDict,
    async_prepare: bool = False,
    algo: str = 'atomic',
    validate_access_integrity: bool = True,
    parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
    to_cpu: bool = True,
):
    """Creates a tensor-aware state dictionary that can be saved using the Local Checkpoint Manager.

    Args:
        sharded_state_dict (ShardedStateDict): The initial state dictionary.
        async_prepare (bool): If True, enables asynchronous preparation.
        algo (str): The algorithm used to create the tensor-aware state dictionary.
        validate_access_integrity (bool): If True, validates sharding integrity.
        parallelization_group (torch.distributed.ProcessGroup):
            The process group used for exchanges to avoid duplications.
        to_cpu (bool): If True, moves all tensors from device to CPU.

    Returns:
        ShardedStateDict: The tensor-aware state dictionary.
    """

    _start = time()

    if async_prepare:
        raise NotImplementedError('Async state_dict preparation is not yet implemented')
    if algo != 'atomic' and algo != 'fully_parallel':
        raise NotImplementedError(
            'Only "atomic" and "fully_parallel" sharding algorithms are supported.'
        )
    fully_parallel = algo == 'fully_parallel'

    sharded_part, common_state_dict = save_preprocess(sharded_state_dict, validate_access_integrity)
    sharded_tensors = []
    sharded_objects = []
    for sh_base in nested_values(sharded_part):
        if isinstance(sh_base, ShardedTensor):
            sharded_tensors.append(sh_base)
        else:
            assert isinstance(sh_base, ShardedObject)
            sharded_objects.append(sh_base)
    if fully_parallel:
        shard_to_saving_rank, _, shard_to_metadata = determine_main_replica_uniform_distribution(
            sharded_part, parallelization_group, True
        )

    raw_tensors, raw_objects = {}, {}
    for ten in sharded_tensors:
        shard_id = _sharded_tensor_shard_id(ten)
        if not fully_parallel or shard_to_saving_rank[shard_id] == torch.distributed.get_rank():
            # TODO cover creating copies on host in CheckpointManager.save()
            if to_cpu:
                raw_tensors[shard_id] = ten.data.to("cpu", non_blocking=True)
            else:
                raw_tensors[shard_id] = ten.data
        ten.data = None
    for obj in sharded_objects:
        raw_objects[_sharded_object_id(obj)] = obj.data
        obj.data = None

    logger.debug(f'prepare_state_dict_for_save took {time() - _start}')

    state_dict_for_save = {
        'raw_tensors': raw_tensors,
        'raw_objects': raw_objects,
        'common': common_state_dict,
        'sharded_state_dict': sharded_part,
    }
    if fully_parallel:
        state_dict_for_save['shard_to_rank'] = shard_to_saving_rank
        state_dict_for_save['shard_to_metadata'] = shard_to_metadata
    return state_dict_for_save


def recreate_state_dict_after_load(
    sharded_state_dict: ShardedStateDict,
    loaded_state_dict: ShardedStateDict,
    algo: str = 'atomic',
    exchange_algo: str = 'broadcast',
    validate_access_integrity: bool = True,
    parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """Creates a final sharded state dictionary from a tensor-aware state dictionary.

    Args:
        sharded_state_dict (ShardedStateDict):
            The initial sharded state dictionary generated from the model.
        loaded_state_dict (ShardedStateDict):
            Tensor-aware state dictionary used to fill in missing data in the sharded state.
        algo (str): The algorithm used to reconstruct the state dictionary
            from the tensor-aware state dictionary.
        exchange_algo (str): The algorithm used for tensor exchanges during retrieval.
        validate_access_integrity (bool): If True, performs validation of sharding integrity.
        parallelization_group (torch.distributed.ProcessGroup):
            The process group used for efficient exchanges during retrieval.

    Returns:
        ShardedStateDict: The finalized sharded state dictionary.
    """

    if algo != 'atomic' and algo != 'fully_parallel':
        raise NotImplementedError(
            'Only "atomic" and "fully_parallel" sharding algorithms are supported.'
        )
    fully_parallel = algo == 'fully_parallel'

    # __adding__ common part
    recreated_state_dict, _ = extract_matching_values(loaded_state_dict["common"], lambda x: True)

    if not sharded_state_dict:
        return recreated_state_dict
    # TODO validate laoded_state_dict["sharded_state_dict"] and sharded_state_dict are compatible

    sharded_state_dict, nonpersistent_state_dict, sh_ten_factories = load_preprocess(
        sharded_state_dict
    )
    # __adding__ nonpersistent part
    merge(recreated_state_dict, nonpersistent_state_dict)

    sharded_part, _ = extract_sharded_base(sharded_state_dict)
    if validate_access_integrity:
        validate_sharding_integrity(determine_global_metadata(sharded_part)[1])

    # load sharded tensors and sharded objects to sharded_part
    loaded_tensors = loaded_state_dict['raw_tensors']
    # TODO cover restoring the original device (H2D) in CheckpointManager.load()
    for k, v in loaded_tensors.items():
        loaded_tensors[k] = v.cuda()  # H2D
    if fully_parallel:
        distribution = (
            loaded_state_dict['shard_to_rank'],
            None,
            loaded_state_dict['shard_to_metadata'],
        )
        unloaded_shards = {}
        for sh_base in nested_values(sharded_part):
            if isinstance(sh_base, ShardedTensor):
                shard_id = _sharded_tensor_shard_id(sh_base)
                if shard_id not in loaded_tensors:
                    unloaded_shards[shard_id] = sh_base
        loaded_tensors = exchange_by_distribution(
            loaded_tensors, unloaded_shards, distribution, parallelization_group, exchange_algo
        )
    loaded_objects = loaded_state_dict['raw_objects']

    def load_sharded_base(x: Any):
        if isinstance(x, ShardedTensor):
            shard_id = _sharded_tensor_shard_id(x)
            if shard_id not in loaded_tensors:
                raise Exception(
                    'The current local checkpoint implementation assumes'
                    'consistent tensor sharding during load and save operations.'
                    f'However, the expected shard {x} (ID: {shard_id})'
                    f'was not found in the checkpoint. (IDs: {loaded_tensors.keys()})'
                )
            x = loaded_tensors[shard_id]
        if isinstance(x, ShardedObject):
            object_id = _sharded_object_id(x)
            assert object_id in loaded_objects, (x, object_id, loaded_objects.keys())
            x = loaded_objects[object_id]
        return x

    dict_list_map_inplace(load_sharded_base, sharded_part)
    sharded_part = apply_factory_merges(sharded_part, sh_ten_factories)
    # __adding__ sharded_part
    merge(recreated_state_dict, sharded_part)
    return recreated_state_dict
