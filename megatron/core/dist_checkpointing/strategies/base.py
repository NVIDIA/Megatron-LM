# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict

from ..mapping import ShardedStateDict, ShardedTensor, CheckpointingException, \
    StateDict


class StrategyAction(Enum):
    LOAD_COMMON = 'load_common'
    LOAD_SHARDED = 'load_sharded'
    SAVE_COMMON = 'save_common'
    SAVE_SHARDED = 'save_sharded'


default_strategies = defaultdict(dict)


def get_default_strategy(action: StrategyAction, backend: str, version: int):
    try:
        return default_strategies[action.value][(backend, version)]
    except KeyError as e:
        raise CheckpointingException(f'Cannot find default strategy for: {(action, backend, version)}') from e



class LoadStrategyBase(ABC):
    @abstractmethod
    def check_backend_compatibility(self, loaded_version):
        raise NotImplementedError

    @abstractmethod
    def check_version_compatibility(self, loaded_version):
        raise NotImplementedError


class SaveStrategyBase(ABC):
    def __init__(self, backend: str, version: int):
        self.backend = backend
        self.version = version


class LoadCommonStrategy(LoadStrategyBase):
    @abstractmethod
    def load(self, checkpoint_dir: Path):
        raise NotImplementedError


class LoadShardedStrategy(LoadStrategyBase):
    @abstractmethod
    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        raise NotImplementedError



class SaveCommonStrategy(SaveStrategyBase):
    @abstractmethod
    def save(self, common_state_dict: StateDict, checkpoint_dir: Path):
        raise NotImplementedError


class SaveShardedStrategy(SaveStrategyBase):
    @abstractmethod
    def save(self, sharded_tensors: List[ShardedTensor], checkpoint_dir: Path):
        raise NotImplementedError
