# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Helpers for manipulating sharded tensors and sharded state dicts. """
import logging
from contextlib import contextmanager
from time import time
from typing import Dict, Optional, Tuple

from .dict_utils import dict_list_map_inplace, extract_matching_values
from .mapping import (
    LocalNonpersistentObject,
    ShardedBase,
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
    StateDict,
)

# _ShardId uniquely identifies a ShardedTensor. This is a subset of ShardedTensor
# attributes: key (str), global_offset (tuple) and flattened_range (optional tuple)
_ShardId = Tuple[str, tuple, Optional[tuple]]


def zip_strict(*args):
    """
    Alternative to Python's builtin zip(..., strict=True) (available in 3.10+).
    Apart from providing functionality in earlier versions of Python is also more verbose.
    (Python's zip does not print lengths, only which iterable has finished earlier)
    """
    args = [list(a) for a in args]
    lens = [len(a) for a in args]
    assert len(set(lens)) <= 1, f"Tried to zip iterables of unequal lengths: {lens}!"
    return zip(*args)


def _sharded_tensor_shard_id(sharded_tensor: ShardedTensor) -> _ShardId:
    """Unique id of the sharded tensor data.

    Should yield the same value for same data replicated on different ranks.

    Args:
        sharded_tensor (ShardedTensor): sharded tensor representing the data shard

    Returns (tuple): unique id of a data shard
    """
    f_range = sharded_tensor.flattened_range
    return (
        sharded_tensor.key,
        sharded_tensor.global_offset,
        None if f_range is None else (f_range.start, f_range.stop),
    )


def _sharded_object_id(sharded_object: ShardedObject) -> _ShardId:
    """Unique id of the sharded object data.

    Should yield the same value for same data replicated on different ranks.

    Args:
        sharded_object (ShardedObject): sharded object representing the data shard

    Returns (tuple): unique id of a data shard
    """
    return (sharded_object.key, sharded_object.global_offset, sharded_object.global_shape)


def extract_sharded_tensors(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """Extract a dict consisting of only ShardedTensor objects
    from a given state dict with any objects.

    Args:
        sharded_state_dict: state dict possibly containing ShardedTensor objects

    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all ShardedTensor (keeping the original state dict structure)
            - state dict with all objects other than ShardedTensor
              (keeping the original state dict structure)
    """
    return extract_matching_values(sharded_state_dict, lambda v: isinstance(v, ShardedTensor))


def extract_sharded_tensors_and_factories(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """Extract a dict consisting of only ShardedTensor and ShardedTensorFactory objects
    from a given state dict with any objects.

    Args:
        sharded_state_dict:
            state dict possibly containing ShardedTensor and ShardedTensorFactory objects

    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all ShardedTensor and ShardedTensorFactory
              (keeping the original state dict structure)
            - state dict with all other objects (keeping the original state dict structure)
    """
    return extract_matching_values(
        sharded_state_dict, lambda v: isinstance(v, (ShardedTensor, ShardedTensorFactory))
    )


def extract_sharded_tensors_or_nonpersistent(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """Extract a dict consisting of only ShardedTensor, ShardedTensorFactory
    and LocalNonpersistentObject objects from a given state dict with any objects.

    Args:
        sharded_state_dict: state dict possibly containing ShardedTensor, ShardedTensorFactory
        and LocalNonpersistentObject objects

    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all ShardedTensor, ShardedTensorFactory and LocalNonpersistentObject
              (keeping the original state dict structure)
            - state dict with all other objects (keeping the original state dict structure)
    """
    return extract_matching_values(
        sharded_state_dict,
        lambda v: isinstance(v, (ShardedTensor, LocalNonpersistentObject, ShardedTensorFactory)),
    )


def extract_sharded_base(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """Extract a dict consisting of only ShardedBase from a given state dict with any objects.

    Args:
        sharded_state_dict: state dict possibly containing ShardedBase objects

    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all ShardedBase objects (keeping the original state dict structure)
            - state dict with all other objects (keeping the original state dict structure)
    """
    return extract_matching_values(sharded_state_dict, lambda v: isinstance(v, ShardedBase))


def extract_nonpersistent(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """Extract a dict consisting of only LocalNonpersistentObjects from a given state dict.

    Args:
        sharded_state_dict: state dict possibly containing LocalNonpersistentObjects

    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all LocalNonpersistentObjects
              (keeping the original state dict structure)
            - state dict with all other objects (keeping the original state dict structure)
    """

    return extract_matching_values(
        sharded_state_dict, lambda v: isinstance(v, LocalNonpersistentObject)
    )


def add_prefix_for_sharding(sharded_state_dict: ShardedStateDict, prefix: str):
    """Prepend a given prefix to all ShardedBase objects in a given state dict *in-place*.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict
        prefix (str): prefix to be prepended

    Returns:
        None: state dict is modified in-place
    """

    def add_prefix(t):
        if isinstance(t, ShardedBase):
            t.key = f'{prefix}{t.key}'
        return t

    dict_list_map_inplace(add_prefix, sharded_state_dict)


def replace_prefix_for_sharding(
    sharded_state_dict: ShardedStateDict, old_prefix: str, new_prefix: str
):
    """Replaces the given prefix in *all* sharded keys in a given state dict.

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
    """Replaces prefixes *only in keys matching* with one of prefixes in the map.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict to replace keys in
        prefix_map (Dict[str, str]):
            map of old->new prefixes. The first matching prefix for each key is used

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


fallback_logger = logging.getLogger(__name__)
__LOGGER_NAME_STACK = []
__LOGGER_STACK = []


@contextmanager
def logger_stack(name: Optional[str] = None, current_logger: Optional[logging.Logger] = None):
    """Context manager for managing logger and name stack.

    Temporarily pushes a logger and/or name onto their respective stacks, allowing hierarchical
    logging and contextual logger usage. Ensures the logger stack is restored afterward.

    Args:
        name (str, optional): Name to add to the logger stack. Defaults to None.
        current_logger (logging.Logger, optional): Logger to use. Defaults to the last logger in
                                                  the stack or a fallback if none exist.

    Yields:
        Tuple[str, logging.Logger]: A tuple with the concatenated logger name stack and
                                    the current logger for the block.

    Example:
        with logger_stack("scope", logger):
            logger.info("Log within 'scope'")
    """
    if name:
        __LOGGER_NAME_STACK.append(name)
    if current_logger:
        __LOGGER_STACK.append(current_logger)
        last_logger = current_logger
    elif __LOGGER_STACK:
        last_logger = __LOGGER_STACK[-1]
    else:
        last_logger = fallback_logger
    try:
        yield ".".join(__LOGGER_NAME_STACK), last_logger
    finally:
        if name and __LOGGER_NAME_STACK:
            __LOGGER_NAME_STACK.pop(-1)
        if current_logger and __LOGGER_STACK:
            __LOGGER_STACK.pop(-1)


@contextmanager
def debug_time(
    name: str, logger: Optional[logging.Logger] = None, threshold: float = float("-inf"), level=None
):
    """Simple context manager for timing functions/code blocks.

    Args:
        name (str): Label describing the code being measured.
        logger (logging.Logger, optional): Logger for output. Defaults to the lowest logger.
        threshold (float, optional): Minimum time (seconds) to log. Skips logging if faster.
        level (int, optional): Logging level. Defaults to DEBUG if `threshold` is unset;
                               WARNING otherwise.
    """
    with logger_stack(name, logger) as (stacked_name, last_logger):
        start = time()
        try:
            yield
        finally:
            result = time() - start
            if result < threshold:
                return
            if level is None:
                level = logging.DEBUG if threshold == float("-inf") else logging.WARNING
            last_logger.log(level, f"{stacked_name} took {result:.4f}s")


def debug_msg(msg: str):
    """Logs a debug message using the current logger stack.

    This function formats and logs a debug message with the current logger
    and name stack, preserving context from the logger_stack context manager.

    Args:
        msg (str): The message to be logged at the debug level.

    Example:
        debug_msg("Checkpoint initialized")
        # Logs: "scope_name Checkpoint initialized" if called within logger_stack("scope_name")
    """
    with logger_stack(None, None) as (stacked_name, last_logger):
        last_logger.debug(f"{stacked_name} {msg}")
