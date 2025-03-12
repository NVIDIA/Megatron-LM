# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Entrypoints for saving and loading the distributed checkpoints.

Functions `load` and `save` are equivalents of `torch.load` and `torch.save`
but expect torch.Tensors to be wrapped with classes from the `mapping module`.
Additionally, `load` expects the sharded state dict argument as a guidance for
loading the sharded tensors.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Tuple, Union

import torch

from . import ShardedTensor
from .core import CheckpointingConfig, save_config
from .dict_utils import extract_matching_values, merge
from .mapping import (
    CheckpointingException,
    CommonStateDict,
    ShardedObject,
    ShardedStateDict,
    StateDict,
    apply_factory_merges,
)
from .state_dict_utils import load_preprocess, save_preprocess
from .strategies.async_utils import AsyncRequest
from .strategies.base import (
    AsyncSaveShardedStrategy,
    LoadCommonStrategy,
    LoadShardedStrategy,
    SaveCommonStrategy,
    SaveShardedStrategy,
    StrategyAction,
    get_default_strategy,
)
from .utils import extract_sharded_base
from .validation import (
    StrictHandling,
    determine_global_metadata,
    parse_strict_flag,
    validate_integrity_and_strict_load,
    validate_sharded_objects_handling,
    verify_checkpoint_and_load_strategy,
)

logger = logging.getLogger(__name__)


# flat state dict with sharded objects without any data
CkptShardedMetadata = Dict[str, Union[ShardedTensor, ShardedObject]]


def load(
    sharded_state_dict: ShardedStateDict,
    checkpoint_dir: str,
    sharded_strategy: Union[LoadShardedStrategy, Tuple[str, int], None] = None,
    common_strategy: Union[LoadCommonStrategy, Tuple[str, int], None] = None,
    validate_access_integrity: bool = True,
    strict: Union[str, StrictHandling] = StrictHandling.ASSUME_OK_UNEXPECTED,
) -> Union[StateDict, Tuple[StateDict, Set[str], Set[str]]]:
    """Loading entrypoint.

    In the steps below, the following verbs refer to corresponding objects:
    - load = load from checkpoint
    - extract = extract from sharded_state_dict
    - add = add to the final state dict
    Steps:
    1. Load common state dict and form the base of the result state dict
    2. Apply factories to sharded_state_dict
    3. Extract LocalNonPersistentObject and add
    4. (optional) Extract ShardedObjects, load and add
    5. Extract ShardedBase, load, apply factory merges and add

    Args:
        sharded_state_dict (ShardedStateDict): state dict of the existing model
            populated with ShardedTensors. Used as a mapping to determine which
            parts of global tensors stored in the checkpoint should be loaded.
        checkpoint_dir (str): directory with the checkpoint
        sharded_strategy (LoadShardedStrategy, Tuple[str, int], optional):
            configures loading behavior for sharded tensors
        common_strategy (LoadCommonStrategy, Tuple[str, int], optional):
            configures loading behavior for common data
        validate_access_integrity (bool default = True): checks if each tensor shard is accessed
            exactly once (as main replica) by some process
        strict (StrictHandling, str, optional): determines the behavior in case of a mismatch
            between the requested sharded state dict and the checkpoint. See `StrictHandling` docs
            for more details. Some values affect the return value of this function
            (missing and unexpected keys are returned).
            Defaults to `True` (StrictHandling.ASSUME_OK_UNEXPECTED) which doesn't
            incur any performance overhead. Other recommended values
            are: `False` (StrictHandling.LOG_UNEXPECTED) which logs only unexpected keys
            or `StrictHandling.RETURN_ALL` which returns all mismatch keys.

    Returns:
        StateDict or Tuple[StateDict, Set[str], Set[str]]: in most cases only
            the loaded state dict is returned. If `strict` flag was set to
    """
    sharded_strategy, common_strategy = verify_checkpoint_and_load_strategy(
        checkpoint_dir, sharded_strategy, common_strategy
    )

    checkpoint_dir = Path(checkpoint_dir)
    common_state_dict = common_strategy.load_common(checkpoint_dir)

    sharded_state_dict, nonpersistent_state_dict, sh_ten_factories = load_preprocess(
        sharded_state_dict
    )
    merge(common_state_dict, nonpersistent_state_dict)

    # At this point we are only dealing with ShardedBase objects
    sharded_state_dict, _ = extract_sharded_base(sharded_state_dict)

    # Validation
    ckpt_sharded_metadata = None
    local_metadata, global_metadata = None, None
    strict = parse_strict_flag(strict)
    if StrictHandling.requires_explicit_ckpt_mismatch_check(strict):
        ckpt_sharded_metadata = load_sharded_metadata(
            str(checkpoint_dir), sharded_strategy, common_strategy
        )
    if validate_access_integrity or StrictHandling.requires_global_app_metadata(strict):
        local_metadata, global_metadata = determine_global_metadata(sharded_state_dict)

    sharded_state_dict, missing_keys, unexpected_keys = validate_integrity_and_strict_load(
        sharded_state_dict,
        strict,
        validate_access_integrity,
        local_metadata,
        global_metadata,
        ckpt_sharded_metadata,
    )

    # ShardedBase loading
    if not sharded_strategy.can_handle_sharded_objects:
        validate_sharded_objects_handling(sharded_strategy, common_strategy)
        sharded_objects_state_dict, sharded_state_dict = extract_matching_values(
            sharded_state_dict, lambda v: isinstance(v, ShardedObject)
        )
        sharded_objects = common_strategy.load_sharded_objects(
            sharded_objects_state_dict, checkpoint_dir
        )
        merge(common_state_dict, sharded_objects)

    loaded_state_dict = sharded_strategy.load(sharded_state_dict, checkpoint_dir)

    merge(common_state_dict, loaded_state_dict)

    loaded_state_dict = apply_factory_merges(common_state_dict, sh_ten_factories)

    if StrictHandling.requires_returning_mismatch_keys(strict):
        return common_state_dict, missing_keys, unexpected_keys
    else:
        return common_state_dict


def load_common_state_dict(checkpoint_dir: Path) -> StateDict:
    """Load common (non-sharded) objects state dict from the checkpoint.

    Args:
        checkpoint_dir (Path): checkpoint directory

    Returns:
        StateDict: state dict with non-sharded objects from the checkpoint
    """
    sharded_strategy, common_strategy = verify_checkpoint_and_load_strategy(str(checkpoint_dir))
    return common_strategy.load_common(checkpoint_dir)


def load_tensors_metadata(
    checkpoint_dir: str, sharded_strategy: Union[LoadShardedStrategy, None] = None
) -> CkptShardedMetadata:
    """Load tensors metadata from the checkpoint.

    Returns a dictionary similar to a sharded state dict, but note that
    the dictionary keys are simply ShardedTensor keys (contrary to the
    actual sharded state dicts where keys correspond to state dict keys).

    Dict values are ShardedTensors without any sharding (so, the only useful
    information is tensors global shape and dtype).

    Concrete implementation depends on the loading strategy. If no strategy is
    given, a default for a given backend is used.

    Args:
        checkpoint_dir (str): checkpoint directory to load from
        sharded_strategy (LoadShardedStrategy, optional): sharded strategy to load metadata.
            Defaults to None - in this case a default load strategy for a given checkpoint type
            is used.

    Returns:
        CkptShardedMetadata: flat state dict without data describing ShardedTensors
            in the checkpoint
    """
    sharded_strategy, common_strategy = verify_checkpoint_and_load_strategy(
        checkpoint_dir, sharded_strategy
    )
    return sharded_strategy.load_tensors_metadata(Path(checkpoint_dir))


def load_sharded_metadata(
    checkpoint_dir: str,
    sharded_strategy: Union[LoadShardedStrategy, None] = None,
    common_strategy: Union[LoadCommonStrategy, None] = None,
) -> CkptShardedMetadata:
    """Load sharded metadata from the checkpoint.

    Similar to `load_tensors_metadata`, but includes also ShardedObjects.

    Returns a dictionary similar to a sharded state dict, but note that
    the dictionary keys are simply ShardedTensor keys (contrary to the
    actual sharded state dicts where keys correspond to state dict keys).

    Dict values are ShardedTensors without any sharding (so, the only useful
    information is tensors global shape and dtype).

    Concrete implementation depends on the loading strategy. If no strategy is
    given, a default for a given backend is used.

    Args:
        checkpoint_dir (str): checkpoint directory to load from
        sharded_strategy (LoadShardedStrategy, optional): sharded strategy to load metadata.
            Defaults to None - in this case a default load strategy for a given checkpoint type
            is used.
        common_strategy (LoadCommonStrategy, optional): common strategy to load metadata.
            Defaults to None - in this case a default load strategy for a given checkpoint type is
            used. This strategy won't be used unless `sharded_strategy` can't handle ShardedObjects

    Returns:
        CkptShardedMetadata: flat state dict without data describing ShardedTensors
            and ShardedObjects in the checkpoint
    """
    sharded_strategy, common_strategy = verify_checkpoint_and_load_strategy(
        checkpoint_dir, sharded_strategy, common_strategy
    )
    sharded_metadata = sharded_strategy.load_sharded_metadata(Path(checkpoint_dir))
    if not sharded_strategy.can_handle_sharded_objects:
        validate_sharded_objects_handling(sharded_strategy, common_strategy)
        common_metadata = common_strategy.load_sharded_metadata(Path(checkpoint_dir))
        sharded_metadata = merge(sharded_metadata, common_metadata)
    return sharded_metadata


def load_plain_tensors(checkpoint_dir: str) -> StateDict:
    """Load checkpoint tensors without any sharding and plain structure.

    NOTE: common state dict is NOT included.

    Args:
        checkpoint_dir (str): checkpoint directory to load the tensors from.

    Returns:
        StateDict: checkpoint state dict containing only torch.Tensors.
    """
    sharded_state_dict = load_tensors_metadata(checkpoint_dir)
    # Don't validate integrity because shards will be overlapped
    # if world_size > 1 (all processes load whole tensors)
    return load(sharded_state_dict, checkpoint_dir, validate_access_integrity=False)


#
# def load_plain_tensors_and_objects(checkpoint_dir: str) -> StateDict:
#     """Load checkpoint tensors and objects without any sharding and plain structure.
#
#     NOTE: state dict structure might be different than the one used for checkpoint saving.
#     NOTE: common state dict is NOT included.
#
#     Args:
#         checkpoint_dir (str): checkpoint directory to load the state dict from.
#
#     Returns:
#         StateDict: complete checkpoint state dict without any sharding.
#     """
#     sharded_state_dict = load_tensors_metadata(checkpoint_dir)
#     # Don't validate integrity because shards will be overlapped
#     # if world_size > 1 (all processes load whole tensors)
#     return load(sharded_state_dict, checkpoint_dir, validate_access_integrity=False)


def remove_sharded_tensors(checkpoint_dir: str, key_prefix: str):
    """determine the appropriate sharding strategy and delegate removal to the sharded strategy"""
    sharded_strategy, common_strategy = verify_checkpoint_and_load_strategy(checkpoint_dir)
    sharded_strategy.remove_sharded_tensors(checkpoint_dir, key_prefix)


def save(
    sharded_state_dict: ShardedStateDict,
    checkpoint_dir: str,
    sharded_strategy: Union[SaveShardedStrategy, Tuple[str, int], None] = None,
    common_strategy: Union[SaveCommonStrategy, Tuple[str, int], None] = None,
    validate_access_integrity: bool = True,
    async_sharded_save: bool = False,
    preprocess_common_before_consistancy_check: Callable[[CommonStateDict], StateDict] = None,
) -> Optional[AsyncRequest]:
    """Saving entrypoint.

    Extracts ShardedTensors from the given state dict. Rank 0 saves the
    "regular" part of the checkpoint to common torch file.
    The ShardedTensors are saved according to a strategy specified by the
    config.

    Steps:
    1. Apply factories
    2. Extract and discard LocalNonPersistentObject
    3. Extract all ShardedBase object
    4. Save all other objects to common.pt
    5. (optional) Extract and save ShardedObjects
    6. Save all ShardedBase objects
    7. Write metadata.json file with backend and version metadata.

    Step (6) can be performed asynchronously (see `async_sharded_save`), in this
    case the actual save is embodied in the returned async request and can be
    scheduled by the external caller. For async request, step (7) is added as
    one of the finalization functions, so that metadata.json is written only
    if the checkpoint is complete.

    Args:
        sharded_state_dict (ShardedStateDict): state dict of the populated with
            ShardedTensors. Used as a mapping to determine how local tensors
            should be saved as global tensors in the checkpoint.
        checkpoint_dir (str): directory to save the checkpoint to
        sharded_strategy (SaveShardedStrategy, Tuple[str, int], optional):
            configures sharded tensors saving behavior and backend
        common_strategy (SaveCommonStrategy, Tuple[str, int], optional):
            configures common data saving behavior and backend
        validate_access_integrity (bool default = True): checks if each tensor shard is accessed
            exactly once (as main replica) by some process.
            It also makes sure the common state dict is consistant across all ranks
        async_sharded_save (bool, optional): if True, for the sharded state dict part
            an async save implementation will be called, with the AsyncRequest
            being returned to the caller. Note that it is the caller responsibility to
            actually schedule the async save. Defaults to False.
        preprocess_common_before_consistancy_check (Callable[[CommonStateDict], StateDict], None):
            A callable function that will preprocess the common state dict (i.e can be used  to
            remove keys that we expect to be different in the state dict). The function must not
            modify the original state dict

    Returns:
        AsyncRequest (optional): if `async_sharded_save` is True, returns
            async request that should be scheduled by the caller of this function.
            None otherwise.
    """
    checkpoint_dir = Path(checkpoint_dir)

    if torch.distributed.get_rank() == 0:
        if not checkpoint_dir.exists():
            raise CheckpointingException(
                f'Checkpoint destination directory does not exist: {checkpoint_dir}'
            )

        if next(checkpoint_dir.iterdir(), None) is not None:
            # Don't throw exception here since this could cause a cascade of failures
            # without human intervention in cases where multiple jobs are queued up.
            if torch.distributed.get_rank() == 0:
                logger.warning("Overwriting old incomplete / corrupted checkpoint...")

    if common_strategy is not None:
        raise NotImplementedError('The only supported common strategy is torch')

    if sharded_strategy is None:
        sharded_strategy = get_default_save_sharded_strategy()
    if not isinstance(sharded_strategy, SaveShardedStrategy):
        assert isinstance(sharded_strategy, tuple), type(sharded_strategy)
        sharded_strategy = get_default_strategy(StrategyAction.SAVE_SHARDED, *sharded_strategy)

    if common_strategy is None:
        common_strategy = get_default_save_common_strategy()
    if not isinstance(common_strategy, SaveCommonStrategy):
        assert isinstance(common_strategy, tuple), type(common_strategy)
        common_strategy = get_default_strategy(StrategyAction.SAVE_COMMON, *common_strategy)

    sharded_state_dict, state_dict = save_preprocess(
        sharded_state_dict, validate_access_integrity, preprocess_common_before_consistancy_check
    )

    common_strategy.save_common(state_dict, checkpoint_dir)

    if not sharded_strategy.can_handle_sharded_objects:
        validate_sharded_objects_handling(sharded_strategy, common_strategy)
        sharded_objects_state_dict, sharded_state_dict = extract_matching_values(
            sharded_state_dict, lambda v: isinstance(v, ShardedObject)
        )
        common_strategy.save_sharded_objects(sharded_objects_state_dict, checkpoint_dir)

    def metadata_finalize_fn():
        if torch.distributed.get_rank() == 0:
            save_config(
                CheckpointingConfig(sharded_strategy.backend, sharded_strategy.version),
                checkpoint_dir,
            )
        torch.distributed.barrier()

    if not async_sharded_save:
        sharded_strategy.save(sharded_state_dict, checkpoint_dir)
        metadata_finalize_fn()
        return

    if not isinstance(sharded_strategy, AsyncSaveShardedStrategy):
        raise CheckpointingException(
            f'Cannot apply async_save to non-async strategy {sharded_strategy}'
        )
    async_request = sharded_strategy.async_save(sharded_state_dict, checkpoint_dir)
    async_request.finalize_fns.append(metadata_finalize_fn)
    return async_request


def get_default_save_sharded_strategy(
    backend: str = 'torch_dist', version: int = 1
) -> SaveShardedStrategy:
    """Get default save sharded strategy."""
    return get_default_strategy(StrategyAction.SAVE_SHARDED, backend, version)


def get_default_save_common_strategy(
    backend: str = 'torch', version: int = 1
) -> SaveCommonStrategy:
    """Get default save common strategy."""
    return get_default_strategy(StrategyAction.SAVE_COMMON, backend, version)


def get_default_load_sharded_strategy(checkpoint_dir: str) -> LoadShardedStrategy:
    """Get default load sharded strategy."""
    return verify_checkpoint_and_load_strategy(checkpoint_dir)[0]
