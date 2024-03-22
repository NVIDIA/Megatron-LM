# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Strategies base interfaces. """

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional

from ..mapping import CheckpointingException, ShardedStateDict, ShardedTensor, StateDict


class StrategyAction(Enum):
    LOAD_COMMON = 'load_common'
    LOAD_SHARDED = 'load_sharded'
    SAVE_COMMON = 'save_common'
    SAVE_SHARDED = 'save_sharded'


default_strategies = defaultdict(dict)


def get_default_strategy(action: StrategyAction, backend: str, version: int):
    """ Retrieves a default strategy for a given action, backend and version. """
    try:
        if backend == 'zarr':
            error_hint = ' Please install `zarr` and `tensorstore<=0.1.45` packages'
            from .tensorstore import _import_trigger
            from .zarr import _import_trigger
        elif backend == 'torch_dist':
            error_hint = ' Please use PyTorch version >=2.1'
            from .torch import _import_trigger
    except ImportError as e:
        raise CheckpointingException(
            f'Cannot import a default strategy for: {(action.value, backend, version)}. Error: {e}. Hint: {error_hint}'
        ) from e
    try:
        return default_strategies[action.value][(backend, version)]
    except KeyError as e:
        raise CheckpointingException(
            f'Cannot find a default strategy for: {(action.value, backend, version)}'
        ) from e


class LoadStrategyBase(ABC):
    """ Base class for a load strategy. Requires implementing checks for compatibility with a given checkpoint version. """

    @abstractmethod
    def check_backend_compatibility(self, loaded_version):
        raise NotImplementedError

    @abstractmethod
    def check_version_compatibility(self, loaded_version):
        raise NotImplementedError

    @property
    def can_handle_sharded_objects(self):
        """ Returns whether or not this strategy can handle loading ShardedObjects. """
        return False


class SaveStrategyBase(ABC):
    """ Base class for a save strategy. Requires defining a backend type and version of the saved format. """

    def __init__(self, backend: str, version: int):
        self.backend = backend
        self.version = version

    @property
    def can_handle_sharded_objects(self):
        """ Returns whether or not this strategy can handle saving ShardedObjects. """
        return False


class LoadCommonStrategy(LoadStrategyBase):
    """ Load strategy for common (non-sharded) objects """

    @abstractmethod
    def load(self, checkpoint_dir: str):
        raise NotImplementedError


class LoadShardedStrategy(LoadStrategyBase):
    """ Load strategy for sharded tensors """

    @abstractmethod
    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: str):
        raise NotImplementedError

    @abstractmethod
    def load_tensors_metadata(self, checkpoint_dir: str):
        """Load tensors metadata from the checkpoint.

        Returns a dictionary similar to a sharded state dict, but note that
        the dictionary keys are simply ShardedTensor keys (contrary to the
        actual sharded state dicts where keys correspond to state dict keys).

        Dict values are ShardedTensors without any sharding (so, the only useful
        information is tensors global shape and dtype).
        """
        raise NotImplementedError(
            f'{self.__class__.__name__} doesnt allow loading only sharded metadata'
        )


class SaveCommonStrategy(SaveStrategyBase):
    """ Save strategy for common (non-sharded) objects """

    @abstractmethod
    def save(self, common_state_dict: StateDict, checkpoint_dir: str):
        raise NotImplementedError


class SaveShardedStrategy(SaveStrategyBase):
    """ Save strategy for sharded tensors """

    @abstractmethod
    def save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: str):
        raise NotImplementedError
