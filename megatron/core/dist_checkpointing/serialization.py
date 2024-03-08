# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Entrypoints for saving and loading the distributed checkpoints.

Functions `load` and `save` are equivalents of `torch.load` and `torch.save`
but expect torch.Tensors to be wrapped with classes from the `mapping module`.
Additionally, `load` expects the sharded state dict argument as a guidance for loading the sharded tensors.
"""

import logging
import os
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from .core import CheckpointingConfig, maybe_load_config, save_config
from .dict_utils import (
    dict_list_map_inplace,
    diff,
    extract_matching_values,
    map_reduce,
    merge,
    nested_values,
)
from .mapping import (
    CheckpointingException,
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
    StateDict,
    apply_factories,
    apply_factory_merges,
    is_main_replica,
)
from .strategies.base import (
    LoadCommonStrategy,
    LoadShardedStrategy,
    SaveCommonStrategy,
    SaveShardedStrategy,
    StrategyAction,
    get_default_strategy,
)
from .utils import extract_sharded_tensors, extract_sharded_tensors_or_nonpersistent

COMMON_STATE_FNAME = 'common.pt'

logger = logging.getLogger(__name__)


def load(
    sharded_state_dict: ShardedStateDict,
    checkpoint_dir: str,
    sharded_strategy: Union[LoadShardedStrategy, Tuple[str, int], None] = None,
    common_strategy: Union[LoadCommonStrategy, Tuple[str, int], None] = None,
    validate_access_integrity: bool = True,
) -> StateDict:
    """Loading entrypoint.

    Arguments:
        sharded_state_dict (ShardedStateDict): state dict of the existing model
            populated with ShardedTensors. Used as a mapping to determine which
            parts of global tensors stored in the checkpoint should be loaded.
        checkpoint_dir (str): directory with the checkpoint
        sharded_strategy (LoadShardedStrategy, Tuple[str, int], optional): configures loading behavior for sharded tensors
        common_strategy (LoadCommonStrategy, Tuple[str, int], optional): configures loading behavior for common data
        validate_access_integrity (bool default = True): checks if each tensor shard is accessed
            exactly once (as main replica) by some process
    """
    if common_strategy is not None:
        raise NotImplementedError('The only supported common strategy is torch')

    sharded_strategy = _verify_checkpoint_and_load_strategy(checkpoint_dir, sharded_strategy)

    checkpoint_dir = Path(checkpoint_dir)
    common_state_dict = load_common_state_dict(checkpoint_dir)
    if not sharded_state_dict:
        return common_state_dict

    sharded_objects, sharded_state_dict = load_sharded_objects(sharded_state_dict, checkpoint_dir)
    merge(common_state_dict, sharded_objects)

    sh_ten_factories, _ = extract_matching_values(
        sharded_state_dict,
        lambda x: isinstance(x, ShardedTensorFactory),
        return_lists_as_dicts=True,
    )
    apply_factories(sharded_state_dict)
    sharded_state_dict, _ = extract_sharded_tensors_or_nonpersistent(sharded_state_dict)
    sharded_state_dict, nonpersistent_state_dict = extract_sharded_tensors(sharded_state_dict)
    dict_list_map_inplace(lambda o: o.unwrap(), nonpersistent_state_dict)
    merge(common_state_dict, nonpersistent_state_dict)

    if validate_access_integrity:
        validate_sharding_integrity(nested_values(sharded_state_dict))

    loaded_state_dict = sharded_strategy.load(sharded_state_dict, checkpoint_dir)

    loaded_state_dict = apply_factory_merges(loaded_state_dict, sh_ten_factories)

    merge(common_state_dict, loaded_state_dict)
    return common_state_dict


def _verify_checkpoint_and_load_strategy(
    checkpoint_dir: str, sharded_strategy: Union[LoadShardedStrategy, Tuple[str, int], None] = None,
) -> LoadShardedStrategy:
    """ Verifies if checkpoint metadata exists and matches given strategy.

    Args:
        checkpoint_dir (str): checkpoint directory
        sharded_strategy (LoadShardedStrategy, Tuple[str, int], optional): load strategy to be verified
            if compatible with the checkpoint content. If None, the default load strategy
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

    # TODO: implement consistency checks here
    return sharded_strategy


# TODO: implement it as common torch strategy
def load_common_state_dict(checkpoint_dir: Path) -> StateDict:
    """ Load common (non-sharded) objects state dict from the checkpoint.

    Args:
        checkpoint_dir (Path): checkpoint directory

    Returns:
        StateDict: state dict with non-sharded objects from the checkpoint
    """
    load_path = Path(checkpoint_dir) / COMMON_STATE_FNAME
    try:
        return torch.load(load_path, map_location='cpu')
    except FileNotFoundError as e:
        err_msg = f'Common file {load_path} does not exist'
        ckpt_files = [f.name for f in checkpoint_dir.iterdir()]
        logger.debug(f'{err_msg}. Checkpoint directory content: {ckpt_files}')
        raise CheckpointingException(err_msg) from e


def load_sharded_objects(sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
    """ Replaces all ShardedObject from a given state dict with values loaded from the checkpoint.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict defining what objects should be loaded.
        checkpoint_dir (Path): checkpoint directory

    Returns:
        None: state dict is modified in place
    """
    sharded_objects, sharded_state_dict = extract_matching_values(
        sharded_state_dict, lambda v: isinstance(v, ShardedObject)
    )

    def load_sharded_object(sh_obj: ShardedObject):
        sh_obj.data = None
        load_path = (checkpoint_dir / sh_obj.unique_key).with_suffix('.pt')
        try:
            loaded_obj = torch.load(load_path)
        except FileNotFoundError as e:
            err_msg = f'Object shard {load_path} not found'
            obj_subdir = checkpoint_dir / sh_obj.key
            if obj_subdir.exists():
                obj_files = [f.name for f in obj_subdir.iterdir()]
                logger.debug(f'{err_msg}. Object {sh_obj.key} directory content: {obj_files}')
            else:
                ckpt_files = [f.name for f in checkpoint_dir.iterdir()]
                logger.debug(
                    f'{err_msg}. Object {sh_obj.key} directory does not exist. Checkpoint directory content: {ckpt_files}'
                )
            raise CheckpointingException(err_msg) from e
        return loaded_obj

    return dict_list_map_inplace(load_sharded_object, sharded_objects), sharded_state_dict


def load_tensors_metadata(
    checkpoint_dir: str, sharded_strategy: Union[LoadShardedStrategy, None] = None
) -> ShardedStateDict:
    """Load tensors metadata from the checkpoint.

    Returns a dictionary similar to a sharded state dict, but note that
    the dictionary keys are simply ShardedTensor keys (contrary to the
    actual sharded state dicts where keys correspond to state dict keys).

    Dict values are ShardedTensors without any sharding (so, the only useful
    information is tensors global shape and dtype).

    Concrete implementation depends on the loading strategy. If no strategy is
    given, a default for a given backend is used.
    """
    sharded_strategy = _verify_checkpoint_and_load_strategy(checkpoint_dir, sharded_strategy)
    return sharded_strategy.load_tensors_metadata(Path(checkpoint_dir))


def load_plain_tensors(checkpoint_dir: str):
    """Load checkpoint tensors without any sharding.

    NOTE: common state dict is NOT included."""
    sharded_state_dict = load_tensors_metadata(checkpoint_dir)
    # Don't validate integrity because shards will be overlapped
    # if world_size > 1 (all processes load whole tensors)
    return load(sharded_state_dict, checkpoint_dir, validate_access_integrity=False)


def save(
    sharded_state_dict: ShardedStateDict,
    checkpoint_dir: str,
    sharded_strategy: Union[SaveShardedStrategy, Tuple[str, int], None] = None,
    common_strategy: Union[SaveCommonStrategy, Tuple[str, int], None] = None,
    validate_access_integrity: bool = True,
):
    """Saving entrypoint.

    Extracts ShardedTensors from the given state dict. Rank 0 saves the
    "regular" part of the checkpoint to common torch file.
    The ShardedTensors are saved according to a strategy specified by the
    config.

    Arguments:
        sharded_state_dict (ShardedStateDict): state dict of the populated with
            ShardedTensors. Used as a mapping to determine how local tensors
            should be saved as global tensors in the checkpoint.
        checkpoint_dir (str): directory to save the checkpoint to
        sharded_strategy (SaveShardedStrategy, Tuple[str, int], optional): configures sharded tensors saving behavior and backend
        common_strategy (SaveCommonStrategy, Tuple[str, int], optional): configures common data saving behavior and backend
        validate_access_integrity (bool default = True): checks if each tensor shard is accessed
            exactly once (as main replica) by some process
    """
    checkpoint_dir = Path(checkpoint_dir)

    if torch.distributed.get_rank() == 0:
        if not checkpoint_dir.exists():
            raise CheckpointingException(
                f'Checkpoint destination directory does not exist: {checkpoint_dir}'
            )

        if next(checkpoint_dir.iterdir(), None) is not None:
            raise CheckpointingException(
                f'Checkpoint destination directory ({checkpoint_dir}) is not empty'
            )

    if common_strategy is not None:
        raise NotImplementedError('The only supported common strategy is torch')

    if sharded_strategy is None:
        sharded_strategy = ('zarr', 1)
    if not isinstance(sharded_strategy, SaveShardedStrategy):
        assert isinstance(sharded_strategy, tuple), type(sharded_strategy)
        sharded_strategy = get_default_strategy(StrategyAction.SAVE_SHARDED, *sharded_strategy)

    apply_factories(sharded_state_dict)
    sharded_state_dict, state_dict = extract_sharded_tensors_or_nonpersistent(sharded_state_dict)
    sharded_state_dict, _ = extract_sharded_tensors(sharded_state_dict)
    sharded_tensors = list(nested_values(sharded_state_dict))
    if validate_access_integrity:
        validate_sharding_integrity(sharded_tensors)

    _save_common_dict(state_dict, checkpoint_dir, True)

    sharded_strategy.save(sharded_tensors, checkpoint_dir)
    save_config(
        CheckpointingConfig(sharded_strategy.backend, sharded_strategy.version), checkpoint_dir
    )


# TODO: implement it as common torch strategy
def _save_common_dict(
    state_dict: StateDict, checkpoint_dir: Path, validate_consistency: bool = False
):
    common_state_dict = _extract_and_save_sharded_objects(
        state_dict, checkpoint_dir, validate_consistency
    )
    if torch.distributed.get_rank() == 0:
        torch.save(common_state_dict, checkpoint_dir / COMMON_STATE_FNAME)
    if validate_consistency:
        # TODO: implement checking consistency with rank 0 common dict on other ranks
        pass
        # torch.distributed.barrier()
        # if not torch.distributed.get_rank() == 0:
        #     rank_0_state_dict = torch.load(checkpoint_dir / COMMON_STATE_FNAME)
        #     print(diff(common_state_dict, rank_0_state_dict))


def _extract_and_save_sharded_objects(
    state_dict: StateDict, checkpoint_dir: Path, validate_consistency: bool = False
):
    sharded_objects, state_dict = extract_matching_values(
        state_dict, lambda v: isinstance(v, ShardedObject)
    )
    sharded_objects = list(nested_values(sharded_objects))
    if validate_consistency:
        validate_objects_sharding_integrity(sharded_objects)
    for sh_obj in sharded_objects:
        if is_main_replica(sh_obj.replica_id):
            save_path = (checkpoint_dir / sh_obj.unique_key).with_suffix('.pt')
            os.makedirs(save_path.parent, exist_ok=True)
            torch.save(sh_obj.data, save_path)
    return state_dict


def validate_sharding_integrity(sharded_tensors: Iterable[ShardedTensor]):
    """ Validate if the ShardedTensors from multiple processes define correct sharding of a global tensor.

    Local ShardedTensors metadata is exchanged with `torch.distributed.all_gather_object`
    and then process with global rank 0 checks if main replicas of the shards:
    - cover the whole global tensors
    - don't overlap

    Args:
        sharded_tensors (Iterable[ShardedTensor]): sharded tensors local to this process

    Returns:
        None

    Raises:
        CheckpointingException for invalid access pattern
    """
    sharding = [ten.without_data() for ten in sharded_tensors]
    all_sharding = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(all_sharding, sharding)
    if torch.distributed.get_rank() != 0:
        return

    key_shardings = defaultdict(list)
    for rank, rank_shardings in enumerate(all_sharding):
        for sharding in rank_shardings:
            key_shardings[sharding.key].append((rank, sharding))
    for key, shardings in key_shardings.items():
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
    def chunk_offset(sharding):
        assert len(sharding.global_offset) == len(sharding.local_shape) + sharding.prepend_axis_num
        return tuple(
            chain(
                (off for off in sharding.global_offset[: sharding.prepend_axis_num]),
                (
                    off // sh
                    for off, sh in zip(
                        sharding.global_offset[sharding.prepend_axis_num :], sharding.local_shape
                    )
                ),
            )
        )

    shard_access_cnt = torch.zeros(
        rank_sharding[0][1].axis_fragmentations, dtype=torch.int, device='cpu'
    )
    for rank, sharding in rank_sharding:
        if is_main_replica(sharding.replica_id):
            shard_access_cnt[chunk_offset(sharding)] += 1
        # TODO: consider validating different replicas too
    return shard_access_cnt


def _validate_sharding_for_key_flattened(tensors_by_shard):
    all_slices = []
    local_shape = tensors_by_shard[0].local_shape
    for sharding in tensors_by_shard:
        assert sharding.local_shape == local_shape
        sharding: ShardedTensor
        if not is_main_replica(sharding.replica_id):
            # TODO: this checks only saving (and loading replica_id=0) consistency
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
            f'Flattened ranges dont cover the whole shard {tensors_by_shard[0]}'
        )


def validate_objects_sharding_integrity(sharded_objects: List[ShardedObject]):
    """ Ensure uniqueness of saved objects. """
    local_sh_objs = [sh_obj.without_data() for sh_obj in sharded_objects]
    all_sh_objs = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(all_sh_objs, local_sh_objs)
    if torch.distributed.get_rank() != 0:
        return
    unique_keys = [
        sh_obj.unique_key
        for sh_obj in chain.from_iterable(all_sh_objs)
        if is_main_replica(sh_obj.replica_id)
    ]
    if len(unique_keys) != len(set(unique_keys)):
        duplicates = {k: cnt for k, cnt in Counter(unique_keys).items() if cnt > 1}
        logger.error(f'Duplicate ShardedObject keys and counts: {duplicates}')
        raise CheckpointingException(f'Duplicate ShardedObject keys: {list(duplicates.keys())}')
