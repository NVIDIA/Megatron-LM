# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException, maybe_load_config
from megatron.core.dist_checkpointing.dict_utils import (
    diff,
    extract_matching_values,
    map_reduce,
    nested_values,
)
from megatron.core.dist_checkpointing.mapping import (
    CommonStateDict,
    ShardedBase,
    ShardedObject,
    ShardedStateDict,
    is_main_replica,
)
from megatron.core.dist_checkpointing.strategies.base import (
    LoadCommonStrategy,
    LoadShardedStrategy,
    SaveCommonStrategy,
    SaveShardedStrategy,
    StrategyAction,
    get_default_strategy,
)

if TYPE_CHECKING:
    from megatron.core.dist_checkpointing.serialization import CkptShardedMetadata

logger = logging.getLogger(__name__)
# pylint: disable=line-too-long
# list of local saved/loaded ShardedBase objects
_LocalMetadata = List[Union[ShardedTensor, ShardedObject]]
# list of lists of global saved/loaded ShardedBase objects (each element corresponds to global rank)
_GlobalMetadata = List[_LocalMetadata]


class StrictHandling(Enum):
    """Determines handling of load mismatch (non-empty "unexpected" or "missing" keys).

    Different flags carry different implications on performance and behaviour and
    are divided into two groups:
    - *_UNEXPECTED
    - *_ALL
    The first group ignores missing keys (present in the checkpoint but missing
    in the sharded state dict) which is created in order to avoid inter-rank
    metadata exchange. Note that the metadata exchange will happen anyway
    with `load(..., validate_access_integrity=True)` flag in which case using the
    `*_ALL` option is recommended as it provides a more thorough check with no
    performance penalty wrt. `*_UNEXPECTED` group.

    All options except for the first one (`ASSUME_OK_UNEXPECTED`) require
    extra disk access before the load in order to remove unexpected keys
    from the sharded state dict requested to load.
    """

    # Relies on the underlying strategy to raise error on unexpected keys
    ASSUME_OK_UNEXPECTED = 'assume_ok_unexpected'
    # Logs (with WARNING level) "unexpected" keys. Missing keys are ignored.
    # This is treated as a reasonable default for a "non-strict" load
    LOG_UNEXPECTED = 'log_unexpected'
    # Logs (with WARNING level) all mismatched keys.
    LOG_ALL = 'log_all'
    # Raise error on unexpected keys before load attempt.
    # Gives cleaner error message than `ASSUME_OK_UNEXPECTED` but requires
    # extra disk access.
    RAISE_UNEXPECTED = 'raise_unexpected'
    # Raise error on any mismatch. Similar to `RAISE_UNEXPECTED` but requires
    # metadata exchange.
    RAISE_ALL = 'raise_all'
    # "Unexpected" mismatches are not reported, but returned by the `load`
    # function along with the loaded state dict. Missing keys are ignored.
    RETURN_UNEXPECTED = 'return_unexpected'
    # All mismatches are returned along with the loaded state dict.
    RETURN_ALL = 'return_all'
    # Simply ignores mismatches (not recommended)
    IGNORE_ALL = 'ignore_all'

    @staticmethod
    def requires_explicit_ckpt_mismatch_check(val: 'StrictHandling') -> bool:
        """Whether a given strict flag involves mismatch check against the checkpoint."""
        return val != StrictHandling.ASSUME_OK_UNEXPECTED

    @staticmethod
    def requires_global_app_metadata(val: 'StrictHandling') -> bool:
        """Whether a given strict option requires global metadata for validation."""
        return val in (
            StrictHandling.IGNORE_ALL,
            StrictHandling.RAISE_ALL,
            StrictHandling.RETURN_ALL,
            StrictHandling.LOG_ALL,
        )

    @staticmethod
    def requires_returning_mismatch_keys(val: 'StrictHandling') -> bool:
        """Whether a given strict option results in extra return value from the `load` function."""
        return val in (StrictHandling.RETURN_UNEXPECTED, StrictHandling.RETURN_ALL)


def parse_strict_flag(strict: Union[str, StrictHandling]) -> StrictHandling:
    """Parse user passed strict flag from a string to StrictHandling instance.

    Args:
        strict (str, StrictHandling): strict flag to parse. If already an instance
            of StrictHandling, this function is a noop.

    Returns:
        StrictHandling: enum instance
    """
    if isinstance(strict, StrictHandling):
        return strict
    try:
        return StrictHandling(strict)
    except (ValueError, TypeError) as e:
        raise ValueError(f'Invalid strict flag: {e}') from e


def validate_integrity_and_strict_load(
    sharded_state_dict: ShardedStateDict,
    strict: StrictHandling,
    validate_access_integrity: bool,
    local_metadata: Optional[_LocalMetadata] = None,
    global_metadata: Optional[_GlobalMetadata] = None,
    ckpt_sharded_metadata: Optional['CkptShardedMetadata'] = None,
) -> Tuple[ShardedStateDict, Set[str], Set[str]]:
    """Validates sharding integrity and potential mismatches with the checkpoint.

    `validate_access_integrity` controls sharding integrity check (orthogonal
    to strictness checking) which verifies `sharded_state_dict` runtime completeness
    (in isolation from the actual checkpoint).

    `strict` flag controls handling of mismatches between the requested
    sharded state dict to load and the actual checkpoint. See `StrictHandling`
    docs for details regarding flag behavior and performance implications
    (disk interactions or inter-rank communication).

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict to verify.
        strict (StrictHandling): flag determining how to handle sharded keys mismatch.
        validate_access_integrity (bool): whether to perform sharding validation.
        local_metadata (_LocalMetadata, optional): local sharded state dict metadata.
            Defaults to None, in which case it's determined based on `sharded_state_dict`.
        global_metadata (_GlobalMetadata, optional): global sharded state dict metadata
            (exchanged between ranks). Defaults to None, in which case "missing"
            keys are not determined.
        ckpt_sharded_metadata (CkptShardedMetadata, optional): sharded metadata
            from the checkpoint. Defaults to None, which only makes sense
            for the `StrictHandling.ASSUME_OK_UNEXPECTED` strict value.

    Returns:
        Tuple[ShardedStateDict, Set[str], Set[str]]: tuple of: sharded state dict
            without unexpected keys, missing and unexpected keys. Missing keys are equal
            on all ranks, unexpected keys might differ across ranks. Additionally,
            missing keys might be erroneously empty (depending on `strict` value).
    """
    missing_keys, unexpected_keys = [], []
    if StrictHandling.requires_explicit_ckpt_mismatch_check(strict):
        if ckpt_sharded_metadata is None:
            raise CheckpointingException(
                'Cannot verify checkpoint mismatch with ckpt_sharded_metadata=None.'
            )
        if local_metadata is None:
            local_metadata = [
                sh_base.without_data() for sh_base in nested_values(sharded_state_dict)
            ]
        # We don't want to check for missing keys even if we could
        _skip_missing_keys = strict in (
            StrictHandling.ASSUME_OK_UNEXPECTED,
            StrictHandling.LOG_UNEXPECTED,
            StrictHandling.RAISE_UNEXPECTED,
            StrictHandling.RETURN_UNEXPECTED,
        )
        missing_keys, unexpected_keys = _determine_missing_and_unexpected_keys(
            ckpt_sharded_metadata, local_metadata, None if _skip_missing_keys else global_metadata
        )

        sharded_state_dict = adjust_non_strict_load(sharded_state_dict, unexpected_keys)

        if strict == StrictHandling.IGNORE_ALL:
            missing_keys, unexpected_keys = [], []
        elif strict in (StrictHandling.RAISE_UNEXPECTED, StrictHandling.RAISE_ALL):
            maybe_report_missing_and_unexpected_keys(missing_keys, unexpected_keys, True)
        elif strict in (StrictHandling.LOG_UNEXPECTED, StrictHandling.LOG_ALL):
            maybe_report_missing_and_unexpected_keys(missing_keys, unexpected_keys, False)

    if validate_access_integrity:
        if global_metadata is None:
            raise CheckpointingException(
                'Cannot check sharding intergrity without global_metadata (None).'
            )
        validate_sharding_integrity(global_metadata)

    return sharded_state_dict, missing_keys, unexpected_keys


def verify_checkpoint_and_load_strategy(
    checkpoint_dir: str,
    sharded_strategy: Union[LoadShardedStrategy, Tuple[str, int], None] = None,
    common_strategy: Union[LoadCommonStrategy, Tuple[str, int], None] = None,
) -> Tuple[LoadShardedStrategy, LoadCommonStrategy]:
    """Verifies if checkpoint metadata exists and matches given strategies.

    If no strategies are passed, they are determined based on the checkpoint metadata.

    Args:
        checkpoint_dir (str): checkpoint directory
        sharded_strategy (LoadShardedStrategy, Tuple[str, int], optional): sharded load strategy to be verified
            if compatible with the checkpoint content. If None, the default sharded load strategy
            for the checkpoint backend will be returned.
        common_strategy (LoadCommonStrategy, Tuple[str, int], optional): common load strategy to be verified
            if compatible with the checkpoint content. If None, the default common load strategy
            for the checkpoint backend will be returned.
    """
    if not Path(checkpoint_dir).exists():
        raise CheckpointingException(f'Checkpoint directory {checkpoint_dir} does not exist')

    saved_config = maybe_load_config(checkpoint_dir)
    if saved_config is None:
        raise CheckpointingException(f'{checkpoint_dir} is not a distributed checkpoint')

    if sharded_strategy is None:
        sharded_strategy = get_default_strategy(
            StrategyAction.LOAD_SHARDED,
            saved_config.sharded_backend,
            saved_config.sharded_backend_version,
        )
    elif isinstance(sharded_strategy, tuple):
        sharded_strategy = get_default_strategy(StrategyAction.LOAD_SHARDED, *sharded_strategy)

    if common_strategy is None:
        common_strategy = get_default_strategy(
            StrategyAction.LOAD_COMMON,
            saved_config.common_backend,
            saved_config.common_backend_version,
        )
    elif isinstance(common_strategy, tuple):
        sharded_strategy = get_default_strategy(StrategyAction.LOAD_COMMON, *common_strategy)

    sharded_strategy.check_backend_compatibility(saved_config.sharded_backend)
    sharded_strategy.check_version_compatibility(saved_config.sharded_backend_version)
    common_strategy.check_backend_compatibility(saved_config.common_backend)
    common_strategy.check_version_compatibility(saved_config.common_backend_version)
    return sharded_strategy, common_strategy


def adjust_non_strict_load(
    sharded_state_dict: ShardedStateDict, sharded_keys_to_remove: Set[str]
) -> ShardedStateDict:
    """Adjusts sharded state dict removing keys not existing in the checkpoint.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict to modify
        sharded_keys_to_remove (Set[str]): keys to remove from the state dict

    Returns:
        ShardedStateDict: state dict without ShardedBase objects with specified keys
    """

    def is_unexpected_key(x: ShardedBase):
        assert isinstance(x, ShardedBase), f'Unexpected type {type(x)}'
        return x.key in sharded_keys_to_remove

    _, sharded_state_dict = extract_matching_values(sharded_state_dict, is_unexpected_key)
    return sharded_state_dict


def _determine_missing_and_unexpected_keys(
    ckpt_sharded_metadata: 'CkptShardedMetadata',
    local_metadata: _LocalMetadata,
    global_metadata: Optional[_GlobalMetadata] = None,
) -> Tuple[Set[str], Set[str]]:
    """Determines load mismatches based on metadata.

    There is an asymmetry between "unexpected" and "missing" keys.
    Unexpected keys can be determined based only on local metadata.
    Missing keys must be based on global metadata, since other ranks might access
    different keys than the current rank.
    In consequence, the return value of this function is different on each rank:
    "missing_keys" are equal, but "unexpected_keys" might differ across ranks.

    Args:
        ckpt_sharded_metadata (CkptShardedMetadata): sharded state dict (without data)
            constructed based on the checkpoint content
        local_metadata (_LocalMetadata): list of local ShardedBase objects
            requested to be loaded by this rank
        global_metadata (_GlobalMetadata, optional): list of global ShardedBase objects
            requested to be loaded by all ranks. Defaults to None, in which case
            returned "missing" keys are empty.

    Returns:
        Tuple[Set[str], Set[str]]: missing and unexpected keys. Missing keys are equal
            on all ranks, unexpected keys might differ across ranks. If passed
            `global_metadata` is empty, returned missing keys are empty as well.

    """
    local_accessed_keys = set(sh_base.key for sh_base in local_metadata)
    ckpt_keys = set(sh_base.key for sh_base in ckpt_sharded_metadata.values())
    unexpected_keys = local_accessed_keys - ckpt_keys
    if global_metadata is not None:
        global_accessed_keys = set(
            sh_base.key for rank_metadata in global_metadata for sh_base in rank_metadata
        )
        missing_keys = ckpt_keys - global_accessed_keys
    else:
        missing_keys = set()

    if missing_keys:
        logger.debug(f'Dist ckpt load missing keys: {missing_keys}')
    if unexpected_keys:
        logger.debug(f'Dist ckpt load unexpected keys: {unexpected_keys}')

    return missing_keys, unexpected_keys


def maybe_report_missing_and_unexpected_keys(
    missing_keys: Set[str], unexpected_keys: Set[str], raise_error: bool = True
) -> None:
    """Raises or logs an error in case missing or unexpected keys are non-empty.

    Args:
        missing_keys (Set[str]): missing keys in the state dict
        unexpected_keys (Set[str]): unexpected keys in the state dict
        raise_error: If True, raises error on mismatch. Otherwise, logs mismatch
            with WARNING level.

    Returns:
        None

    Raises:
        CheckpointingException: if `raise_error` is True and at least one of
        `missing_keys` or `unexpected_keys` are non-empty.
    """
    if not missing_keys and not unexpected_keys:
        return
    missing_title_msg = (
        f'Some keys found in the checkpoint are missing in the provided sharded state dict. '
    )
    missing_body_msg = f'Missing keys (for all ranks): {missing_keys}. '
    unexpected_title_msg = f'Unexpected keys (not found in the checkpoint) encountered in the provided sharded state dict. '
    unexpected_body_msg = f'Unexpected keys (for this rank): {unexpected_keys}. '
    error_msg = ''
    if missing_keys:
        error_msg += missing_title_msg
    if unexpected_keys:
        error_msg += unexpected_title_msg

    error_msg += '\n'
    if missing_keys:
        error_msg += missing_body_msg
    if unexpected_keys:
        error_msg += unexpected_body_msg

    if raise_error:
        raise CheckpointingException(error_msg)
    else:
        logger.warning(error_msg)


def _validate_common_state_dict(common_state_dict: CommonStateDict):
    """Validate consistancy across ranks for the common state dict

    We save the common state dict only on rank 0. We validate to make sure that the common dict is consistant across ranks before saving.

    Args:
        common_state_dict: The common state dict present in all ransk
    """
    other_rank_state_dicts = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(other_rank_state_dicts, common_state_dict)
    common_state_dict_diff = {}
    if torch.distributed.get_rank() == 0:
        main_rank_state_dict = common_state_dict
        for rank, rank_state_dict in enumerate(other_rank_state_dicts[1:], 1):
            only_left, only_right, mismatch = diff(main_rank_state_dict, rank_state_dict)
            if only_left or only_right or mismatch:
                common_state_dict_diff[rank] = (only_left, only_right, mismatch)

        if len(common_state_dict_diff) != 0:
            logger.warning(
                f'There is difference in the common state dict in different ranks. The differences are {common_state_dict_diff}'
            )


def validate_sharding_integrity(
    global_metadata: _GlobalMetadata, common_state_dict: CommonStateDict = None
) -> None:
    """Validate if the ShardedTensors and ShardedObjects from multiple processes define correct sharding.

    Local ShardedTensors and ShardedObject metadata is exchanged with `torch.distributed.all_gather_object`
    and then process with global rank 0 checks if main replicas of the shards:
    - cover the whole global tensors
    - don't overlap

    Args:
        global_metadata (_GlobalMetadata): ShardedTensor and ShardedObject objects from all ranks.
        common_state_dict (CommonStateDict): The common state dict stored by rank 0

    Returns:
        None

    Raises:
        CheckpointingException for invalid access pattern
    """

    if common_state_dict:
        _validate_common_state_dict(common_state_dict)

    if torch.distributed.get_rank() != 0:
        return

    key_shardings = defaultdict(list)
    for rank, rank_shardings in enumerate(global_metadata):
        for sharding in rank_shardings:
            key_shardings[sharding.key].append((rank, sharding))
    for key, shardings in key_shardings.items():
        if isinstance(shardings[0][1], ShardedObject):
            _validate_objects_for_key(shardings)
        else:
            _validate_sharding_for_key(shardings)


def _validate_sharding_for_key(rank_sharding: List[Tuple[int, ShardedTensor]]):
    some_rank_shard = rank_sharding[0][1]
    global_shape = some_rank_shard.global_shape
    local_shape = some_rank_shard.local_shape
    dtype = some_rank_shard.dtype
    has_flattened_range = some_rank_shard.flattened_range is not None
    for rank, sharding in rank_sharding:
        assert sharding.dtype == dtype, (sharding.dtype, dtype, some_rank_shard)
        assert sharding.global_shape == global_shape, (
            sharding.global_shape,
            global_shape,
            some_rank_shard,
        )
        assert sharding.local_shape == local_shape, (
            sharding.local_shape,
            local_shape,
            some_rank_shard,
        )
        assert (sharding.flattened_range is not None) == has_flattened_range, (
            (sharding.flattened_range is not None),
            has_flattened_range,
            some_rank_shard,
        )

    shard_access_cnt = _compute_shards_access(rank_sharding)
    if has_flattened_range:
        map_reduce(
            rank_sharding,
            lambda x: x[1].global_offset,
            lambda x: x[1],
            _validate_sharding_for_key_flattened,
        )
    else:
        if not torch.all(shard_access_cnt == 1):
            logger.error(f'Invalid access pattern for {rank_sharding[0][1]}: {shard_access_cnt}')
            raise CheckpointingException(f'Invalid access pattern for {rank_sharding[0][1]}')


def _compute_shards_access(rank_sharding):
    shard_access_cnt = torch.zeros(
        rank_sharding[0][1].axis_fragmentations, dtype=torch.int, device='cpu'
    )
    for rank, sharding in rank_sharding:
        if is_main_replica(sharding.replica_id):
            shard_access_cnt[sharding.local_chunk_offset_in_global()] += 1
    return shard_access_cnt


def _validate_sharding_for_key_flattened(tensors_by_shard):
    all_slices = []
    local_shape = tensors_by_shard[0].local_shape
    for sharding in tensors_by_shard:
        assert sharding.local_shape == local_shape
        sharding: ShardedTensor
        if not is_main_replica(sharding.replica_id):
            continue

        all_slices.append((sharding.flattened_range.start, sharding.flattened_range.stop))

    starts, stops = map(np.asarray, zip(*sorted(all_slices)))
    if (
        starts[0] != 0
        or stops[-1] != np.product(local_shape)
        or not np.all(starts[1:] == stops[:-1])
    ):
        logger.error(
            f'Flattened ranges dont cover the whole shard {tensors_by_shard[0]}. Ranges: {(starts, stops)}'
        )
        raise CheckpointingException(
            f'Flattened ranges dont cover the whole shard {tensors_by_shard[0]}. Ranges: {(starts, stops)}'
        )


def _validate_objects_for_key(sharded_objects: List[ShardedObject]):
    """Ensure uniqueness of saved objects."""
    unique_keys = [
        sh_obj.unique_key for _, sh_obj in sharded_objects if is_main_replica(sh_obj.replica_id)
    ]
    if len(unique_keys) != len(set(unique_keys)):
        duplicates = {k: cnt for k, cnt in Counter(unique_keys).items() if cnt > 1}
        logger.error(f'Duplicate ShardedObject keys and counts: {duplicates}')
        raise CheckpointingException(f'Duplicate ShardedObject keys: {list(duplicates.keys())}')
    expected_shard_num = np.prod(sharded_objects[0][1].global_shape)
    if len(unique_keys) != expected_shard_num:
        err_msg = f'Invalid access pattern: {expected_shard_num - len(unique_keys)} ShardedObject are missing.'
        logger.error(f'{err_msg} Existing shards: {unique_keys}')
        raise CheckpointingException(err_msg)


def determine_global_metadata(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[_LocalMetadata, _GlobalMetadata]:
    """Exchanges local metadata with `all_gather_object` to determine global metadata.

    Args:
        sharded_state_dict (ShardedStateDict): local sharded state dict

    Returns:
        Tuple[_LocalMetadata, _GlobalMetadata]: local and global ShardedBase objects with stripped data
    """
    local_metadata = [ten.without_data() for ten in nested_values(sharded_state_dict)]
    global_metadata = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(global_metadata, local_metadata)
    return local_metadata, global_metadata


def validate_sharded_objects_handling(
    sharded_strategy: Union[SaveShardedStrategy, LoadShardedStrategy],
    common_strategy: Union[SaveCommonStrategy, LoadCommonStrategy],
) -> None:
    """Checks if either of the passed strategies can handle sharded objects.

    Args:
        sharded_strategy (Union[SaveShardedStrategy, LoadShardedStrategy]): sharded strategy used for saving/loading
        common_strategy (Union[SaveCommonStrategy, LoadCommonStrategy]): common strategy used for saving/loading

    Returns:
        None

    Raises:
        CheckpointingException: if both strategies can't handle ShardedObjects
    """
    if (
        not sharded_strategy.can_handle_sharded_objects
        and not common_strategy.can_handle_sharded_objects
    ):
        raise CheckpointingException(
            f'Either sharded strategy or common strategy must implement ShardedObjects handling.'
            f' Both {sharded_strategy} and {common_strategy} specify can_handle_sharded_objects=False'
        )
