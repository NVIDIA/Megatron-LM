# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility import for the transformer-layer schedule builder."""

from megatron.core.models.common.transformer_layer_callables import (
    build_transformer_layer_callables,
)

__all__ = ["build_transformer_layer_callables"]
