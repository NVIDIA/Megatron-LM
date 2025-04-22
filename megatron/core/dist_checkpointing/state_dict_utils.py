# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Utilities for transforming state_dict."""

from typing import Callable, Union

from .dict_utils import dict_list_map_inplace, extract_matching_values
from .mapping import (
    CommonStateDict,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
    StateDict,
    apply_factories,
)
from .utils import extract_nonpersistent, extract_sharded_base
from .validation import determine_global_metadata, validate_sharding_integrity


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
    sharded_part = filter_out_empty_flatten_tensor(sharded_part)
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
