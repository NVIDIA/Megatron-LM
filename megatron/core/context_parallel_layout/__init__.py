# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Context parallel sequence partition-mode helpers.

This package preserves the historical ``megatron.core.context_parallel_layout``
import surface while splitting the implementation by responsibility.

Ownership summary:

- model builders choose the pipeline-stage input CP layout;
- blocks convert rank-local sequence tensors between layer requirements;
- model postprocess restores the public output boundary to the input layout;
- MTP validates its own layout requirement but does not own outer conversion.
"""

from typing import Literal

CpPartitionMode = Literal["zigzag", "contiguous"]

from megatron.core.context_parallel_layout.conversion import (
    CpPartitionModeConverter,
    contiguous_to_zigzag_chunks,
    convert_cp_partition_mode,
    zigzag_to_contiguous_chunks,
)
from megatron.core.context_parallel_layout.metadata import (
    get_packed_seq_params_cp_partition_cu_seqlens,
    is_cp_rank_local_rotary_pos_emb,
    replace_packed_seq_params_cp_partition_mode,
)
from megatron.core.context_parallel_layout.policy import (
    get_context_parallel_layout_chunk_indices,
    get_required_cp_partition_mode_for_layer,
    get_stage_entry_partition_mode,
)
from megatron.core.context_parallel_layout.routes import (
    build_thd_cp_partition_route,
    decode_thd_cp_partition_route,
    get_thd_context_parallel_rank_indices,
    get_thd_cp_partition_route,
    prebuild_thd_cp_partition_routes,
)

__all__ = [
    "CpPartitionMode",
    "CpPartitionModeConverter",
    "build_thd_cp_partition_route",
    "contiguous_to_zigzag_chunks",
    "convert_cp_partition_mode",
    "decode_thd_cp_partition_route",
    "get_context_parallel_layout_chunk_indices",
    "get_packed_seq_params_cp_partition_cu_seqlens",
    "get_required_cp_partition_mode_for_layer",
    "get_stage_entry_partition_mode",
    "get_thd_cp_partition_route",
    "get_thd_context_parallel_rank_indices",
    "is_cp_rank_local_rotary_pos_emb",
    "prebuild_thd_cp_partition_routes",
    "replace_packed_seq_params_cp_partition_mode",
    "zigzag_to_contiguous_chunks",
]
