# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit lifecycle for the optional Dist-MoE backend."""

from .runtime import close_dist_moe_runtime, initialize_dist_moe_runtime

__all__ = ["close_dist_moe_runtime", "initialize_dist_moe_runtime"]
