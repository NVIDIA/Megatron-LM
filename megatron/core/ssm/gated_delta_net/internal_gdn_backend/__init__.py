# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Internal chunked gated delta rule backend."""

from .chunk import (
    chunk_gated_delta_rule,
    chunk_kda,
    prepare_cp_context_metadata,
    prepare_validated_chunk_metadata,
)

__all__ = [
    "chunk_gated_delta_rule",
    "chunk_kda",
    "prepare_cp_context_metadata",
    "prepare_validated_chunk_metadata",
]
