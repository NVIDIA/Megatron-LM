# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Public context parallel sequence partition-mode APIs.

The implementation is split by responsibility; internal conversion and route-building
helpers remain in their respective submodules rather than being re-exported here.

Ownership summary:

- model builders choose the pipeline-stage input CP layout;
- Attention and recurrent modules convert to their required layout and restore their input layout;
- hybrid blocks may coalesce compatible transitions with main's cross-layer layout manager;
- model postprocess and MTP preserve the public model-boundary layout.
"""

from megatron.core.context_parallel_layout.conversion import (
    CpPartitionModeConverter,
    convert_module_input_tensors_cp_partition_mode,
)
from megatron.core.context_parallel_layout.routes import prebuild_thd_cp_partition_routes
from megatron.core.context_parallel_layout.types import CpPartitionMode, ThdCpRoute
from megatron.core.context_parallel_layout.utils import finalize_packed_seq_params

__all__ = [
    "CpPartitionMode",
    "CpPartitionModeConverter",
    "ThdCpRoute",
    "convert_module_input_tensors_cp_partition_mode",
    "finalize_packed_seq_params",
    "prebuild_thd_cp_partition_routes",
]
