# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Utilities for transforming state_dict."""

from typing import Callable, Optional, Sequence, Union

import torch
import torch.distributed as dist

from .dict_utils import dict_list_map_inplace, extract_matching_values, nested_values
from .mapping import (
    CommonStateDict,
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
    StateDict,
    apply_factories,
    is_main_replica,
)
from .utils import (
    _clean_metadata_for_serialization,
    _sharded_object_id,
    _sharded_tensor_shard_id,
    extract_nonpersistent,
    extract_sharded_base,
)
from .validation import determine_global_metadata, validate_sharding_integrity


def save_preprocess(
    sharded_state_dict: ShardedStateDict,
    validate_access_integrity: bool = True,
    preprocess_common_before_consistancy_check: Callable[[CommonStateDict], StateDict] = None,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
    preferred_writer_ranks: Optional[Sequence[int]] = None,
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
        preferred_writer_ranks (Sequence[int], optional): global ranks preferred when
            electing one writer for each tensor or object shard in ``process_group``.
            Shards absent from these ranks are written by another rank in the group.
            If omitted, existing replica IDs determine the writers.

    Returns:
        Tuple[ShardedStateDict, dict]:
            The preprocessed sharded state dictionary and the common state dictionary.
    """
    apply_factories(sharded_state_dict)
    _, sharded_state_dict = extract_nonpersistent(sharded_state_dict)
    sharded_part, common_state_dict = extract_sharded_base(sharded_state_dict)
    sharded_part = filter_out_empty_flatten_tensor(sharded_part)
    global_metadata = None
    if preferred_writer_ranks is not None:
        global_metadata = determine_global_metadata(sharded_part, process_group=process_group)[1]
        elect_checkpoint_replicas(
            sharded_part, global_metadata, process_group, preferred_writer_ranks
        )
    if validate_access_integrity:
        preprocessed_common_state_dict = common_state_dict
        if "content_metadata" in preprocessed_common_state_dict:
            preprocessed_common_state_dict["content_metadata"] = _clean_metadata_for_serialization(
                preprocessed_common_state_dict["content_metadata"]
            )

        if preprocess_common_before_consistancy_check:
            preprocessed_common_state_dict = preprocess_common_before_consistancy_check(
                common_state_dict
            )
        validate_sharding_integrity(
            (
                global_metadata
                if global_metadata is not None
                else determine_global_metadata(sharded_part, process_group=process_group)[1]
            ),
            common_state_dict=preprocessed_common_state_dict,
            process_group=process_group,
        )
    return sharded_part, common_state_dict


def _checkpoint_shard_id(shard):
    """Identify one data shard independently of its replica ID."""
    if isinstance(shard, ShardedTensor):
        return ('tensor', _sharded_tensor_shard_id(shard))
    if isinstance(shard, ShardedObject):
        return ('object', _sharded_object_id(shard))
    raise TypeError(f'Unsupported checkpoint shard type: {type(shard)}')


def elect_checkpoint_replicas(sharded_part, global_metadata, process_group, preferred_ranks):
    """Select one group-local main replica per shard, including ShardedObjects."""
    group_size = dist.get_world_size(group=process_group)
    global_ranks = (
        list(range(group_size))
        if process_group is None
        else [dist.get_global_rank(process_group, rank) for rank in range(group_size)]
    )
    preferred = set(preferred_ranks)
    if not preferred.issubset(global_ranks):
        raise ValueError('Preferred ranks must belong to the checkpoint process group')

    holders = {}
    for rank, shards in enumerate(global_metadata):
        for shard in shards:
            holders.setdefault(_checkpoint_shard_id(shard), []).append(
                (rank, is_main_replica(shard.replica_id))
            )

    writers = {}
    for shard_id, candidates in holders.items():
        preferred_candidates = [c for c in candidates if global_ranks[c[0]] in preferred]
        candidates = preferred_candidates or candidates
        writers[shard_id] = min(candidates, key=lambda c: (not c[1], c[0]))[0]

    def set_replica(shard, rank):
        shard.replica_id = 0 if writers[_checkpoint_shard_id(shard)] == rank else 1

    for rank, shards in enumerate(global_metadata):
        for shard in shards:
            set_replica(shard, rank)
    local_rank = dist.get_rank(group=process_group)
    for shard in nested_values(sharded_part):
        set_replica(shard, local_rank)


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
    sharded_state_dict = filter_out_empty_flatten_tensor(sharded_state_dict)

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


def filter_out_empty_flatten_tensor(sharded_state_dict: Union[dict, list]):
    """
    Filter out ShardedTensors with empty flatten_range.
    These tensors can cause the PyTorch check in failure.

    Args:
        sharded_state_dict: state dict possibly containing ShardedTensor objects
    """
    # Filter out ShardedTensors with empty flatten_range.
    # These tensors can cause the PyTorch check in
    # `TorchShardedTensor._init_from_local_shards_and_global_metadata` to fail.
    # This situation may occur in custom Fully Sharded Data Parallel (FSDP) cases.
    sharded_state_dict, _ = extract_matching_values(
        sharded_state_dict,
        lambda v: not (
            isinstance(v, ShardedTensor)
            and v.flattened_range
            and v.flattened_range.start == v.flattened_range.stop
        ),
    )

    return sharded_state_dict
