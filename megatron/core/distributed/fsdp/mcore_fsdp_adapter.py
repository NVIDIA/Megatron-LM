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

"""Public factory and re-exports for the MCore FSDP wrappers."""

from typing import List, Optional, Type

import torch

from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_adapter_v1 import FullyShardedDataParallelV1
from megatron.core.distributed.fsdp.mcore_adapter_v2 import FullyShardedDataParallelV2
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig

__all__ = ["FullyShardedDataParallel", "FullyShardedDataParallelV1", "FullyShardedDataParallelV2"]


def FullyShardedDataParallel(
    config: TransformerConfig,
    ddp_config: DistributedDataParallelConfig,
    module: torch.nn.Module,
    fsdp_unit_modules: Optional[List[Type[torch.nn.Module]]] = None,
    disable_bucketing: bool = False,
    device: Optional[torch.device] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> _BaseDataParallel:
    """Construct the configured Megatron-FSDP implementation.

    This is a factory function, not a wrapper type. Use the explicit V1 or V2
    implementation classes for type checks.
    """
    fsdp_class = (
        FullyShardedDataParallelV2
        if ddp_config.megatron_fsdp_version == 2
        else FullyShardedDataParallelV1
    )
    return fsdp_class(
        config, ddp_config, module, fsdp_unit_modules, disable_bucketing, device, pg_collection
    )
