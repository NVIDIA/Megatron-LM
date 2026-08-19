# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility import for the Transformer Engine backend.

The implementation moved to :mod:`megatron.core.ops.providers.transformer_engine`.
"""

from megatron.core.ops.providers.transformer_engine import TESpecProvider

__all__ = ["TESpecProvider"]
