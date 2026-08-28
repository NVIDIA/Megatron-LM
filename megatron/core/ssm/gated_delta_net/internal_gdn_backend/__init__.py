# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Internal chunked gated delta rule backend."""

from megatron.core.ssm.gated_delta_net.internal_gdn_backend.chunk import (
    chunk_gated_delta_rule,
    prepare_validated_chunk_metadata,
)

__all__ = ["chunk_gated_delta_rule", "prepare_validated_chunk_metadata"]
