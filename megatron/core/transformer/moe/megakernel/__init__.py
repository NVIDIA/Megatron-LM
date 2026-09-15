# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Backend-neutral MoE megakernel integration interfaces."""

from .factory import (
    build_megakernel_backend,
    megakernel_shared_expert_init_context,
    prepare_megakernel_shared_expert_config,
)

__all__ = [
    "build_megakernel_backend",
    "megakernel_shared_expert_init_context",
    "prepare_megakernel_shared_expert_config",
]
