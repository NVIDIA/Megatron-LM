# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron Lite-owned parallel runtime exports."""

from __future__ import annotations

from megatron.lite.primitive.parallel.cp import (
    contiguous_to_zigzag_chunks,
    split_packed_for_cp,
    zigzag_position_ids_for_cp,
    zigzag_reconstruct_from_cp_parts,
    zigzag_slice_for_cp,
    zigzag_split_for_cp,
    zigzag_to_contiguous_chunks,
)
from megatron.lite.primitive.parallel.pipeline import forward_backward_pipelining
from megatron.lite.primitive.parallel.pp import (
    PipelineChunkLayout,
    build_pipeline_chunk_layout,
)
from megatron.lite.primitive.parallel.sp import (
    gather_for_non_sp_head,
    gather_from_sequence_parallel,
    scatter_to_sequence_parallel,
)
from megatron.lite.primitive.parallel.state import ParallelState, init_parallel
from megatron.lite.primitive.parallel.thd import (
    PackedSeqParams,
    PackedTHDBatch,
    has_packed_thd_params,
    pack_nested_thd,
    parallel_state_from_model,
    prepare_packed_thd_for_context_parallel,
    prepare_packed_thd_kwargs_for_context_parallel,
    reconstruct_packed_from_cp_parts,
    roll_packed_thd_left,
    split_packed_to_cp_local,
    unpack_packed_thd_to_nested,
)

_LAZY_LINEAR_EXPORTS = {
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VanillaColumnParallelLinear",
    "VocabParallelEmbedding",
    "VocabParallelOutput",
    "pad_vocab_for_tp",
}


def __getattr__(name: str):
    if name in _LAZY_LINEAR_EXPORTS:
        from megatron.lite.primitive.parallel import linear as _linear

        return getattr(_linear, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ColumnParallelLinear",
    "contiguous_to_zigzag_chunks",
    "PackedSeqParams",
    "PackedTHDBatch",
    "PipelineChunkLayout",
    "ParallelState",
    "RowParallelLinear",
    "VanillaColumnParallelLinear",
    "VocabParallelEmbedding",
    "VocabParallelOutput",
    "build_pipeline_chunk_layout",
    "forward_backward_pipelining",
    "gather_for_non_sp_head",
    "gather_from_sequence_parallel",
    "has_packed_thd_params",
    "init_parallel",
    "pad_vocab_for_tp",
    "pack_nested_thd",
    "parallel_state_from_model",
    "prepare_packed_thd_for_context_parallel",
    "prepare_packed_thd_kwargs_for_context_parallel",
    "reconstruct_packed_from_cp_parts",
    "roll_packed_thd_left",
    "scatter_to_sequence_parallel",
    "split_packed_to_cp_local",
    "split_packed_for_cp",
    "unpack_packed_thd_to_nested",
    "zigzag_position_ids_for_cp",
    "zigzag_reconstruct_from_cp_parts",
    "zigzag_slice_for_cp",
    "zigzag_split_for_cp",
    "zigzag_to_contiguous_chunks",
]
