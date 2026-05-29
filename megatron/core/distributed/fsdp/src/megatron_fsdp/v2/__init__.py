# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# FIXME: The following sharding strategies are not yet supported in FSDP v2:
#   - no_shard
#   - optim
#   - optim_grads
# Currently only optim_grads_params is fully implemented and tested.
# We will add support for these strategies in a follow-up change.
# When using ParameterGroup directly or via fully_shard(), passing an unsupported
# sharding_strategy will raise NotImplementedError. See README.md for details.

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
from .mixed_precision import (
    FullyShardFP8Policy,
    FullyShardMixedPrecisionPolicy,
    FullyShardNVFP4Policy,
)
from .param_group import ParameterGroup

__all__ = [
    "FSDPModule",
    "fully_shard",
    "FullyShardFP8Policy",
    "FullyShardMixedPrecisionPolicy",
    "FullyShardNVFP4Policy",
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
