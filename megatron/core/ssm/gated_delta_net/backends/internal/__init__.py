# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Internal chunked gated delta rule backend."""

from megatron.core.ssm.gated_delta_net.backends.internal.chunk import chunk_gated_delta_rule

__all__ = ["chunk_gated_delta_rule"]
