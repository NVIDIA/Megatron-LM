# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from ..uneven_dtensor import (
    gather_and_compute_chunk_metadata,
    get_state_dict,
    make_uneven_dtensor,
    preprocess_state_dict_for_uneven_dtensor,
    redistribute_uneven_dtensor_to_replicated,
    split_dtensor,
    uneven_dtensor_to_full_tensor,
)
from .allocator import Bucket, TemporaryBucketAllocator
from .dp_buffer import BufferIndex, DataParallelBuffer
from .fully_shard import FSDPModule, fully_shard
from .param_group import ParameterGroup

__all__ = [
    "FSDPModule",
    "fully_shard",
    "ParameterGroup",
    "BufferIndex",
    "DataParallelBuffer",
    "Bucket",
    "TemporaryBucketAllocator",
    "make_uneven_dtensor",
    "get_state_dict",
    "preprocess_state_dict_for_uneven_dtensor",
    "gather_and_compute_chunk_metadata",
    "split_dtensor",
    "uneven_dtensor_to_full_tensor",
    "redistribute_uneven_dtensor_to_replicated",
]
