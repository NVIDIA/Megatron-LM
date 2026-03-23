# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Strategies base interfaces. """

import logging
from enum import Enum

from .torch import TorchDistLoadShardedStrategy, TorchDistSaveShardedStrategy

logger = logging.getLogger(__name__)


class LoadShardedStrategy:
    """Base class for load strategies to be removed in v0.15"""

    pass


class SaveShardedStrategy:
    """Base class for save strategies to be removed in v0.15"""

    def __init__(self, backend: str, version: int):
        pass


class StrategyAction(Enum):
    """Specifies save vs load and sharded vs common action.
    To be removed in v0.15"""

    LOAD_COMMON = 'load_common'
    LOAD_SHARDED = 'load_sharded'
    SAVE_COMMON = 'save_common'
    SAVE_SHARDED = 'save_sharded'


def get_default_strategy(action: StrategyAction, backend: str, version: int):
    """Retrieves a default strategy for a given action, backend and version."""

    logger.warning(
        'megatron.core.dist_checkpointing.strategies.base.get_default_strategy'
        ' is deprecated and will be removed in Megatron-Core v0.14. Please use'
        ' TorchDistLoadShardedStrategy() and TorchDistSaveShardedStrategy()'
        ' to get the default load and save sharded strategies.'
    )
    if backend != 'torch_dist':
        logger.warning(f'{backend} is not supported')
    if action == StrategyAction.LOAD_SHARDED:
        return TorchDistLoadShardedStrategy()
    else:
        assert action == StrategyAction.SAVE_SHARDED, f'{action} is not supported'
        return TorchDistSaveShardedStrategy()
