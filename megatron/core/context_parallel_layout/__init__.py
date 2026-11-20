# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility import path; algorithms live only in megatron.core.context_parallel.

Packed callers must use the metadata returned by finalize_packed_seq_params and the
canonical thd_plan. The legacy equal-length ThdCpRoute representation is not duplicated.
"""

from megatron.core.context_parallel.layout import CpPartitionMode, convert_cp_partition_mode
from megatron.core.context_parallel.module import (
    CpPartitionModeConverter,
    finalize_packed_seq_params,
    prepare_module_cp_input,
)

__all__ = [
    "CpPartitionMode",
    "CpPartitionModeConverter",
    "convert_cp_partition_mode",
    "finalize_packed_seq_params",
    "prepare_module_cp_input",
]
