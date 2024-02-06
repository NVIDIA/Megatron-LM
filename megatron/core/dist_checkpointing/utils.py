# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Helpers for manipulating sharded tensors and sharded state dicts. """

from typing import Dict, Tuple

from .dict_utils import dict_list_map_inplace, extract_matching_values
from .mapping import (
    LocalNonpersitentObject,
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
    StateDict,
)


def extract_sharded_tensors(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """ Extract a dict consisting of only ShardedTensor objects from a given state dict with any objects.

    Args:
        sharded_state_dict: state dict possibly containing ShardedTensor objects

    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all ShardedTensor (keeping the original state dict structure)
            - state dict with all objects other than ShardedTensor (keeping the original state dict structure)
    """
    return extract_matching_values(sharded_state_dict, lambda v: isinstance(v, ShardedTensor))


def extract_sharded_tensors_and_factories(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """ Extract a dict consisting of only ShardedTensor and ShardedTensorFactory objects from a given state dict with any objects.

    Args:
        sharded_state_dict: state dict possibly containing ShardedTensor and ShardedTensorFactory objects

    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all ShardedTensor and ShardedTensorFactory (keeping the original state dict structure)
            - state dict with all other objects (keeping the original state dict structure)
    """
    return extract_matching_values(
        sharded_state_dict, lambda v: isinstance(v, (ShardedTensor, ShardedTensorFactory))
    )


def extract_sharded_tensors_or_nonpersistent(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """ Extract a dict consisting of only ShardedTensor, ShardedTensorFactory and LocalNonpersitentObject
    objects from a given state dict with any objects.

    Args:
        sharded_state_dict: state dict possibly containing ShardedTensor, ShardedTensorFactory and LocalNonpersitentObject objects

    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all ShardedTensor, ShardedTensorFactory and LocalNonpersitentObject (keeping the original state dict structure)
            - state dict with all other objects (keeping the original state dict structure)
    """
    return extract_matching_values(
        sharded_state_dict,
        lambda v: isinstance(v, (ShardedTensor, LocalNonpersitentObject, ShardedTensorFactory)),
    )


def add_prefix_for_sharding(sharded_state_dict: ShardedStateDict, prefix: str):
    """ Prepend a given prefix to all ShardedTensor objects in a given state dict *in-place*.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict
        prefix (str): prefix to be prepended

    Returns:
        None: state dict is modified in-place
    """

    def add_prefix(t):
        if isinstance(t, ShardedTensor):
            t.key = f'{prefix}.{t.key}'
        return t

    dict_list_map_inplace(add_prefix, sharded_state_dict)


def replace_prefix_for_sharding(
    sharded_state_dict: ShardedStateDict, old_prefix: str, new_prefix: str
):
    """ Replaces the given prefix in *all* sharded keys in a given state dict.

    Errors out if some key does not begin with a given prefix.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict to replace keys in
        old_prefix (str): prefix to be replaced in each key
        new_prefix (str): new prefix

    Returns:
        None: state dict is modified in place
    """

    def _replace_prefix(x):
        if isinstance(x, (ShardedTensor, ShardedTensorFactory, ShardedObject)):
            if not x.key.startswith(old_prefix):
                raise ValueError(f'Expected {x.key} to begin with prefix {old_prefix}')
            x.key = f'{new_prefix}{x.key[len(old_prefix):]}'  # str.removeprefix in Python >= 3.9
        return x

    dict_list_map_inplace(_replace_prefix, sharded_state_dict)


def apply_prefix_mapping(sharded_state_dict: ShardedStateDict, prefix_map: Dict[str, str]):
    """ Replaces prefixes *only in keys matching* with one of prefixes in the map.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict to replace keys in
        prefix_map (Dict[str, str]): map of old->new prefixes. The first matching prefix for each key is used

    Returns:
        None: state dict is modified in place
    """

    def _replace_prefixes(x):
        if not isinstance(x, (ShardedTensor, ShardedTensorFactory, ShardedObject)):
            return x
        for old_prefix, new_prefix in prefix_map.items():
            if x.key.startswith(old_prefix):
                x.key = (
                    f'{new_prefix}{x.key[len(old_prefix):]}'  # str.removeprefix in Python >= 3.9
                )
                break
        return x

    dict_list_map_inplace(_replace_prefixes, sharded_state_dict)
