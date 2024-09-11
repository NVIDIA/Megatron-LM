# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Strategies base interfaces. """

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, DefaultDict, Union

from ..mapping import CheckpointingException, ShardedStateDict, StateDict
from .async_utils import AsyncCallsQueue, AsyncRequest


class StrategyAction(Enum):
    """Specifies save vs load and sharded vs common action."""

    LOAD_COMMON = 'load_common'
    LOAD_SHARDED = 'load_sharded'
    SAVE_COMMON = 'save_common'
    SAVE_SHARDED = 'save_sharded'


default_strategies: DefaultDict[str, dict[tuple, Any]] = defaultdict(dict)

async_calls = AsyncCallsQueue()


def get_default_strategy(action: StrategyAction, backend: str, version: int):
    """Retrieves a default strategy for a given action, backend and version."""
    try:
        if backend == 'zarr':
            error_hint = ' Please install `zarr` and `tensorstore<=0.1.45` packages'
            from .tensorstore import register_default_tensorstore_strategies

            register_default_tensorstore_strategies()
            from .zarr import register_default_zarr_strategies

            register_default_zarr_strategies()
        elif backend == 'torch_dist':
            error_hint = ' Please use PyTorch version >=2.1'
            from .torch import register_default_torch_strategies

            register_default_torch_strategies()
    except ImportError as e:
        raise CheckpointingException(
            f'Cannot import a default strategy for: {(action.value, backend, version)}. '
            f'Error: {e}. Hint: {error_hint}'
        ) from e
    try:
        return default_strategies[action.value][(backend, version)]
    except KeyError as e:
        raise CheckpointingException(
            f'Cannot find a default strategy for: {(action.value, backend, version)}'
        ) from e


def register_default_strategy(
    action: StrategyAction,
    backend: str,
    version: int,
    strategy: Union['SaveStrategyBase', 'LoadStrategyBase'],
):
    """Adds a given strategy to the registry of default strategies.

    Args:
        action (StrategyAction): specifies save/load and sharded/common
        backend (str): backend that the strategy becomes a default for
        version (int): version that the strategy becomes a default for
        strategy (SaveStrategyBase, LoadStrategyBase): strategy to register
    """
    default_strategies[action.value][(backend, version)] = strategy


class LoadStrategyBase(ABC):
    """Base class for a load strategy. Requires implementing checks for compatibility with a
    given checkpoint version."""

    @abstractmethod
    def check_backend_compatibility(self, loaded_backend):
        """Verifies if this strategy is compatible with `loaded_backend`."""
        raise NotImplementedError

    @abstractmethod
    def check_version_compatibility(self, loaded_version):
        """Verifies if this strategy is compatible with `loaded_version`."""
        raise NotImplementedError

    @property
    def can_handle_sharded_objects(self):
        """Returns whether or not this strategy can handle loading ShardedObjects."""
        return False


class SaveStrategyBase(ABC):
    """Base class for a save strategy. Requires defining a backend type and
    version of the saved format."""

    def __init__(self, backend: str, version: int):
        self.backend = backend
        self.version = version

    @property
    def can_handle_sharded_objects(self):
        """Returns whether or not this strategy can handle saving ShardedObjects."""
        return False

    def __str__(self):
        return f'{self.__class__.__name__}({self.backend}, {self.version})'


class LoadCommonStrategy(LoadStrategyBase):
    """Load strategy for common (non-sharded) objects"""

    @abstractmethod
    def load_common(self, checkpoint_dir: Path):
        """Load common part of the checkpoint."""
        raise NotImplementedError

    @abstractmethod
    def load_sharded_objects(
        self, sharded_objects_state_dict: ShardedStateDict, checkpoint_dir: Path
    ):
        """Load sharded objects from the checkpoint."""
        raise NotImplementedError

    def load_sharded_metadata(self, checkpoint_dir: Path) -> ShardedStateDict:
        """Load just the metadata from the checkpoint."""
        if not self.can_handle_sharded_objects:
            return {}
        raise NotImplementedError


class LoadShardedStrategy(LoadStrategyBase):
    """Load strategy for sharded tensors"""

    @abstractmethod
    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        """Load the sharded part of the checkpoint."""
        raise NotImplementedError

    @abstractmethod
    def load_tensors_metadata(self, checkpoint_dir: Path):
        """Load tensors metadata from the checkpoint for ShardedTensors.

        Returns a dictionary similar to a sharded state dict, but note that
        the dictionary keys are simply ShardedTensor keys (contrary to the
        actual sharded state dicts where keys correspond to state dict keys).

        Dict values are ShardedTensors without any data and sharding (so, the
        only useful information is tensors global shape and dtype).
        """
        raise NotImplementedError(
            f'Loading only tensors metadata not implemented for {self.__class__.__name__}'
        )

    def load_sharded_metadata(self, checkpoint_dir: Path):
        """Load sharded metadata from the checkpoint for ShardedTensors and ShardedObjects.

        Returns a dictionary similar to a sharded state dict, but note that
        the dictionary keys are simply sharded keys (contrary to the
        actual sharded state dicts where keys correspond to state dict keys).

        Dict values are ShardedTensors or ShardedObjects without any data and sharding.
        """
        if not self.can_handle_sharded_objects:
            return self.load_tensors_metadata(checkpoint_dir)
        raise NotImplementedError(
            f'Loading only sharded metadata not implemented for {self.__class__.__name__}'
        )


class SaveCommonStrategy(SaveStrategyBase):
    """Save strategy for common (non-sharded) objects"""

    @abstractmethod
    def save_common(self, common_state_dict: StateDict, checkpoint_dir: Path):
        """Save common part of the state dict."""
        raise NotImplementedError

    def save_sharded_objects(
        self, sharded_objects_state_dict: ShardedStateDict, checkpoint_dir: Path
    ):
        """Save sharded objects from the state dict."""
        raise NotImplementedError


class SaveShardedStrategy(SaveStrategyBase):
    """Save strategy for sharded tensors"""

    @abstractmethod
    def save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        """Save the sharded part of the state dict."""
        raise NotImplementedError


class AsyncSaveShardedStrategy(SaveShardedStrategy):
    """Save strategy suitable for async save."""

    @abstractmethod
    def async_save(
        self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path
    ) -> AsyncRequest:
        """Perform preparation and return an AsyncRequest to the external caller.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict to save
            checkpoint_dir (Path): checkpoint target directory

        Returns:
            AsyncRequest: represents the async save function and finalization function.
                It is the caller responsibility to actually schedule the async save.
        """
        raise NotImplementedError

    def save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        """Each async strategy can be trivially used as a sync strategy."""
        async_request = self.async_save(sharded_state_dict, checkpoint_dir)
        # multiprocessing routines  may cause issue when called on parent process
        # We keep this verbose call for now
        global async_calls
        async_calls.schedule_async_request(async_request)
        async_calls.maybe_finalize_async_calls(blocking=True)
