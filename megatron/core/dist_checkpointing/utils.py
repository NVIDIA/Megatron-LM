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

from typing import Tuple

from .mapping import StateDict, ShardedStateDict, ShardedTensor, \
    LocalNonpersitentObject
from .dict_utils import extract_matching_values, dict_list_map_inplace


def extract_sharded_tensors(sharded_state_dict: ShardedStateDict) -> Tuple[ShardedStateDict, StateDict]:
    return extract_matching_values(sharded_state_dict, lambda v: isinstance(v, ShardedTensor))


def extract_sharded_tensors_or_nonpersistent(sharded_state_dict: ShardedStateDict) -> Tuple[ShardedStateDict, StateDict]:
    return extract_matching_values(sharded_state_dict, lambda v: isinstance(v, (ShardedTensor, LocalNonpersitentObject)))


def add_prefix_for_sharding(sharded_state_dict: ShardedStateDict, prefix: str):
    def add_prefix(t):
        if isinstance(t, ShardedTensor):
            t.key = f'{prefix}.{t.key}'
        return t
    dict_list_map_inplace(add_prefix, sharded_state_dict)
