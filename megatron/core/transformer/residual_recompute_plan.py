# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compatibility exports for the shared residual-recompute helpers."""

# pylint: disable=unused-import

from megatron.core.transformer.residual_recompute import (
    build_recompute_block_end_plan as build_recompute_block_end_plan,
)
from megatron.core.transformer.residual_recompute import (
    build_recompute_layer_managers as build_recompute_layer_managers,
)
from megatron.core.transformer.residual_recompute import (
    finalize_recompute_block as finalize_recompute_block,
)
