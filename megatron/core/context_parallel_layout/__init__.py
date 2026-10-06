# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility exports for the context-parallel layout API.

Implementation and new imports live in :mod:`megatron.core.context_parallel`.
"""

from megatron.core.context_parallel import (
    CpPartitionMode,
    CpPartitionModeConverter,
    ThdCpRoute,
    convert_module_input_tensors_cp_partition_mode,
    finalize_packed_seq_params,
    prebuild_thd_cp_partition_routes,
)

__all__ = [
    "CpPartitionMode",
    "CpPartitionModeConverter",
    "ThdCpRoute",
    "convert_module_input_tensors_cp_partition_mode",
    "finalize_packed_seq_params",
    "prebuild_thd_cp_partition_routes",
]
