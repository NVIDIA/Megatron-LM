# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Helpers for manipulating sharded tensors and sharded state dicts."""

import logging
from contextlib import contextmanager
from time import time
from typing import Dict, Optional, Tuple

import torch

from .core import CheckpointingException
from .dict_utils import dict_list_map_inplace, extract_matching_values, nested_values
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


def force_all_tensors_to_non_fp8(sharded_state_dict: ShardedStateDict):
    """Force all tensors in state dict to be non-fp8.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict.
    """
    from ..fp8_utils import dequantize_fp8_tensor, is_float8tensor  # Avoid circular import

    for v in nested_values(sharded_state_dict):
        if hasattr(v, "data") and is_float8tensor(v.data):
            v.data = dequantize_fp8_tensor(v.data)


def is_delayed_scaling_fp8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine FP8 tensor quantized with delayed scaling.

    Delayed scaling is the only recipe whose scaling factor does not come from the values being
    quantized: it comes from the module's fp8 metadata (amax history), which a checkpoint load
    restores only in `load_state_dict`. All other recipes (current scaling, MXFP8, blockwise FP8,
    NVFP4) derive their scales from the values passed to `copy_`.
    """
    from ..fp8_utils import is_float8tensor  # Avoid circular import
    from ..utils import is_te_min_version

    if not is_float8tensor(tensor):
        return False
    if not is_te_min_version("2.0"):
        # TE1.x only supports delayed scaling.
        return True
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

    data = tensor.data if isinstance(tensor, torch.nn.Parameter) else tensor
    return isinstance(getattr(data, "_quantizer", None), Float8Quantizer)


def _quantized_storage_tensors(tensor: torch.Tensor):
    """Yields the raw storage tensors (element codes and scales) of a TE quantized tensor."""
    for attr in (
        "_data",
        "_scale_inv",
        "_rowwise_data",
        "_rowwise_scale_inv",
        "_columnwise_data",
        "_columnwise_scale_inv",
        "_amax_rowwise",
        "_amax_columnwise",
    ):
        value = getattr(tensor, attr, None)
        if torch.is_tensor(value):
            yield value


def _factory_builds_in_place_quantized_views(factory: ShardedTensorFactory) -> bool:
    """Whether the factory splits its quantized tensor into views that can be quantized in place.

    That requires every view to be a quantized tensor whose codes and scales alias the original
    tensor's storage (TE dequantizes on unsupported splits, e.g. NVFP4 and blockwise FP8, and
    pads MXFP8 scale slices that are not aligned to 128 rows, both of which produce copies), and
    no per-tensor scale shared between views (Float8 current scaling), since a view quantized on
    its own would overwrite the scale of the others.
    """
    from ..fp8_utils import is_float8tensor  # Avoid circular import

    original_storages = {
        t.untyped_storage().data_ptr() for t in _quantized_storage_tensors(factory.data)
    }
    views = [
        sh_ten.data
        for sh_ten in nested_values(factory.build())
        if isinstance(sh_ten, ShardedTensor)
    ]
    if not views or not all(is_float8tensor(view) for view in views):
        return False
    for view in views:
        if any(
            t.untyped_storage().data_ptr() not in original_storages
            for t in _quantized_storage_tensors(view)
        ):
            return False
    per_tensor_scales = [
        view._scale_inv.data_ptr()
        for view in views
        if torch.is_tensor(getattr(view, "_scale_inv", None))
    ]
    return len(per_tensor_scales) == len(set(per_tensor_scales))


def prepare_quantized_tensors_for_streaming_load(sharded_state_dict: ShardedStateDict):
    """Prepare the quantized tensors of a state dict for the streaming dequantize load.

    The streaming load (`MCoreLoadPlanner` with `stream_ckpt_dequant`) quantizes each quantized
    destination in place from a per-tensor high-precision scratch. This keeps quantized:
      - tensors whose scales are derived from the quantized values (current scaling, MXFP8,
        blockwise FP8, NVFP4);
      - ShardedTensorFactories (e.g. the SwiGLU split of linear_fc1) whose views alias the
        tensor's quantized storage: the views are loaded in place and the merge returns the
        tensor itself instead of concatenating high-precision copies of the views.
    Dequantizes up front, as `force_all_tensors_to_non_fp8` does for every quantized tensor,
    the factories whose views are high-precision copies (NVFP4 and blockwise FP8), padded copies
    (MXFP8 scale slices not aligned to 128 rows) or share one per-tensor scale (Float8 current
    scaling), since quantizing such views independently would corrupt the tensor.

    Rejects delayed-scaling FP8 tensors: their scale is only known once `load_state_dict`
    restores the fp8 metadata, so they cannot be quantized during the load.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict, modified in place.

    Raises:
        CheckpointingException: if the state dict holds a delayed-scaling FP8 tensor.
    """
    from ..fp8_utils import dequantize_fp8_tensor, is_float8tensor  # Avoid circular import

    for v in nested_values(sharded_state_dict):
        if not hasattr(v, "data") or not is_float8tensor(v.data):
            continue
        if is_delayed_scaling_fp8tensor(v.data):
            raise CheckpointingException(
                f'The streaming dequantize load does not support delayed-scaling FP8 tensors'
                f' ({getattr(v, "key", v)}): their scale is only restored by `load_state_dict`.'
            )
        if isinstance(v, ShardedTensorFactory):
            if _factory_builds_in_place_quantized_views(v):
                # The views are loaded in place, so the merged result is the tensor itself. Hand
                # back a view rather than the object: `load_state_dict` copies the loaded tensor
                # into the param, and TE's generic quantized `copy_` cannot copy a tensor onto
                # itself.
                quantized = v.data
                v.merge_fn = lambda sub_state_dict, quantized=quantized: quantized.view(
                    quantized.shape
                )
            else:
                v.data = dequantize_fp8_tensor(v.data)


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


def _clean_metadata_for_serialization(metadata: dict) -> dict:
    """Create a clean copy of metadata for serialization by removing non-serializable objects.

    Args:
        metadata: Original metadata dict

    Returns:
        Clean metadata dict suitable for serialization
    """
    if metadata is None:
        return None
    clean_metadata = metadata.copy()
    # Remove dp_cp_group as it's not serializable
    clean_metadata.pop('dp_cp_group', None)
    return clean_metadata
