# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Strategies base interfaces. """

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from pathlib import Path
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
        return default_strategies[action.value][(backend, version)]
    except KeyError as e:
        hint = ''
        if backend == 'zarr':
            try:
                import tensorstore
                import zarr
            except ImportError:
                hint = ' Please install `zarr` and `tensorstore<=0.1.45` packages'
        raise CheckpointingException(
            f'Cannot find a default strategy for: {(action.value, backend, version)}.{hint}'
        ) from e


class LoadStrategyBase(ABC):
    """ Base class for a load strategy. Requires implementing checks for compatibility with a given checkpoint version. """

    @abstractmethod
    def check_backend_compatibility(self, loaded_version):
        raise NotImplementedError

    @abstractmethod
    def check_version_compatibility(self, loaded_version):
        raise NotImplementedError


class SaveStrategyBase(ABC):
    """ Base class for a save strategy. Requires defining a backend type and version of the saved format. """

    def __init__(self, backend: str, version: int):
        self.backend = backend
        self.version = version


class LoadCommonStrategy(LoadStrategyBase):
    """ Load strategy for common (non-sharded) objects """

    @abstractmethod
    def load(self, checkpoint_dir: Path):
        raise NotImplementedError


class LoadShardedStrategy(LoadStrategyBase):
    """ Load strategy for sharded tensors """

    @abstractmethod
    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        raise NotImplementedError

    @abstractmethod
    def load_tensors_metadata(self, checkpoint_dir: Path):
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
    def save(self, common_state_dict: StateDict, checkpoint_dir: Path):
        raise NotImplementedError


class SaveShardedStrategy(SaveStrategyBase):
    """ Save strategy for sharded tensors """

    @abstractmethod
    def save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        raise NotImplementedError
