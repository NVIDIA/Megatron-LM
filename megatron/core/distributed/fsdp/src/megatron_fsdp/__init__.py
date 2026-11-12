# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from importlib import import_module
from typing import Any

from .package_info import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __shortversion__,
    __version__,
)

__all__ = [
    "DistributedDataParallelConfig",
    "MegatronFSDP",
    "FSDPDistributedIndex",
    "MixedPrecisionPolicy",
    "fully_shard",
    "fully_shard_model",
    "fully_shard_optimizer",
    "__contact_emails__",
    "__contact_names__",
    "__description__",
    "__download_url__",
    "__homepage__",
    "__keywords__",
    "__license__",
    "__package_name__",
    "__repository_url__",
    "__shortversion__",
    "__version__",
]


# Configuration modules must be importable without loading the runtime, which
# imports MCore's DistributedDataParallelConfig when MCore is installed.
_RUNTIME_EXPORTS = {
    "DistributedDataParallelConfig": ".distributed_data_parallel_config",
    "MegatronFSDP": ".megatron_fsdp",
    "FSDPDistributedIndex": ".utils",
    "MixedPrecisionPolicy": ".mixed_precision",
    "fully_shard": ".fully_shard",
    "fully_shard_model": ".fully_shard",
    "fully_shard_optimizer": ".fully_shard",
}


def __getattr__(name: str) -> Any:
    """Load runtime exports only when requested."""
    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = _RUNTIME_EXPORTS[name]
    module = import_module(module_name, __name__)
    # Cache all exports from this module, including functions sharing its name.
    for export, source in _RUNTIME_EXPORTS.items():
        if source == module_name:
            globals()[export] = getattr(module, export)
    return globals()[name]
