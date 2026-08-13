# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Public context parallel sequence partition-mode APIs.

The implementation is split by responsibility; internal conversion and route-building
helpers remain in their respective submodules rather than being re-exported here.

Ownership summary:

- model builders choose the pipeline-stage input CP layout;
- blocks convert rank-local sequence tensors between layer preferences;
- model postprocess restores the public output boundary to the input layout;
- MTP validates its inner-layer layout preference but does not own outer conversion.
"""

from megatron.core.context_parallel_layout.conversion import (
    CpPartitionModeConverter,
    convert_module_input_tensors_cp_partition_mode,
)
from megatron.core.context_parallel_layout.routes import prebuild_thd_cp_partition_routes
from megatron.core.context_parallel_layout.types import CpPartitionMode, ThdCpRoute

__all__ = [
    "CpPartitionMode",
    "CpPartitionModeConverter",
    "ThdCpRoute",
    "convert_module_input_tensors_cp_partition_mode",
    "prebuild_thd_cp_partition_routes",
]
