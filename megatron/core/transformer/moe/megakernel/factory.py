# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Construction entry point for MoE megakernel backends."""

from __future__ import annotations

from contextlib import nullcontext
from copy import copy
from typing import TYPE_CHECKING

from torch import nn
from torch.distributed import ProcessGroup

from megatron.core.transformer.moe.megakernel.backend import MegakernelBackend

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


def build_megakernel_backend(
    *,
    config: TransformerConfig,
    ep_group: ProcessGroup,
    routed_experts: nn.Module,
    shared_experts: nn.Module,
    num_local_experts: int,
) -> MegakernelBackend:
    """Build the configured megakernel backend.

    Backend imports are deliberately lazy so ordinary MCore MoE configurations
    do not acquire optional megakernel dependencies.
    """
    backend = config.moe_megakernel_backend
    if backend in ("mok", "dist_moe"):
        if backend == "mok":
            from megatron.core.transformer.moe.megakernel.mok.backend import MoKMegakernel

            backend_cls = MoKMegakernel
        else:
            from megatron.core.transformer.moe.megakernel.dist_moe.backend import DistMoEMegakernel

            backend_cls = DistMoEMegakernel
        return backend_cls(
            config=config,
            ep_group=ep_group,
            routed_experts=routed_experts,
            shared_experts=shared_experts,
            num_local_experts=num_local_experts,
        )
    raise ValueError(f"Unsupported MoE megakernel backend: {backend!r}")


def prepare_megakernel_shared_expert_config(config: TransformerConfig) -> TransformerConfig:
    """Return the config used to construct backend-compatible shared experts."""
    backend = config.moe_megakernel_backend
    if backend is None or (
        backend == "dist_moe"
        and not (config.moe_megakernel_backend_config or {}).get("shared_expert_bf16", False)
    ):
        return config
    if backend == "dist_moe":
        shared_config = copy(config)
        shared_config.fp8 = None
        shared_config.fp8_param = False
        return shared_config
    if backend in ("mok", "dist_moe"):
        from megatron.core.transformer.moe.megakernel.mok.weights import (
            prepare_shared_expert_config,
        )

        return prepare_shared_expert_config(config)
    raise ValueError(f"Unsupported MoE megakernel backend: {backend!r}")


def megakernel_shared_expert_init_context(config: TransformerConfig):
    """Return the backend-specific context used to construct shared experts."""
    return megakernel_shared_expert_context(config, is_init=True)


def megakernel_shared_expert_context(config: TransformerConfig, *, is_init=False):
    """Apply the requested shared precision during construction or computation."""
    backend = config.moe_megakernel_backend
    if backend is None or (
        backend == "dist_moe"
        and not (config.moe_megakernel_backend_config or {}).get("shared_expert_bf16", False)
    ):
        return nullcontext()
    if backend in ("mok", "dist_moe"):
        from megatron.core.fp8_utils import get_fp8_disabled_context

        # MOK consumes native BF16 shared weights. Disable any model-wide TE
        # fp8_model_init context while the authoritative MCore module is built.
        return get_fp8_disabled_context(config, is_init=is_init)
    raise ValueError(f"Unsupported MoE megakernel backend: {backend!r}")
