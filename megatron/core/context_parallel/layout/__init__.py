# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Context-parallel layout package.

Fixed-layout conversion primitives, THD routes, packed-sequence metadata, the
per-layer ``ContextParallelLayoutManager`` and the module-boundary
``cp_partition_mode`` converter. This package consolidates the CP-layout
machinery that previously lived in the flat
``megatron.core.context_parallel.{layout,conversion,routes,metadata,...}``
modules and in the legacy ``megatron.core.context_parallel_layout`` package.
"""

from .conversion import CpPartitionModeConverter, convert_module_input_tensors_cp_partition_mode
from .manager import (
    ContextParallelLayoutManager,
    ContextParallelLayoutState,
    CPLayout,
    THDCPLayoutPlan,
    build_thd_cp_layout_plan,
    contiguous_to_zigzag,
    convert_cp_layout,
    zigzag_to_contiguous,
)
from .metadata import finalize_packed_seq_params
from .routes import prebuild_thd_cp_partition_routes
from .types import CpPartitionMode, ThdCpRoute

# Private helpers that main's ``context_parallel.utils`` and the CP-layout unit
# tests import from the (formerly flat) layout module. Re-export them so
# ``megatron.core.context_parallel.layout`` keeps exposing the same names the old
# ``layout.py`` module did.
from .manager import (  # noqa: F401
    _build_group_rank_by_logical_rank,
    _build_layout_redistribution_plan,
    _build_thd_cp_layout_plan_from_rank_order_indices,
    _build_thd_zigzag_metadata,
    _get_layout_parallel_context,
    _local_segment_ids,
    _segments_per_rank,
)

__all__ = [
    "CPLayout",
    "CpPartitionMode",
    "CpPartitionModeConverter",
    "ThdCpRoute",
    "convert_module_input_tensors_cp_partition_mode",
    "finalize_packed_seq_params",
    "prebuild_thd_cp_partition_routes",
    "ContextParallelLayoutManager",
    "ContextParallelLayoutState",
    "THDCPLayoutPlan",
    "build_thd_cp_layout_plan",
    "contiguous_to_zigzag",
    "convert_cp_layout",
    "zigzag_to_contiguous",
]
