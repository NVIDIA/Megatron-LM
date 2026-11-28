# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Construction entry point for MoE megakernel backends."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

from torch import nn
from torch.distributed import ProcessGroup

from megatron.core.transformer.moe.megakernel.backend import MegakernelBackend

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


def split_megakernel_backend_config(config: TransformerConfig) -> tuple[str, dict]:
    """Separate MCore's shared policy from a copy of backend-specific options."""
    options = dict(config.moe_megakernel_backend_config or {})
    mode = options.pop("shared_expert_mode", "auto")
    if mode not in ("auto", "fused", "native"):
        raise ValueError("shared_expert_mode must be auto, fused, or native")
    return mode, options


def megakernel_fuses_shared_expert(config: TransformerConfig) -> bool:
    """Resolve shared execution from structural compatibility, independently of activation."""
    if config.moe_megakernel_backend is None:
        return False
    if config.moe_megakernel_backend != "mok":
        raise ValueError(f"Unsupported MoE megakernel backend: {config.moe_megakernel_backend!r}")
    mode, _ = split_megakernel_backend_config(config)
    compatible = (
        config.moe_shared_expert_intermediate_size is not None
        and config.tensor_model_parallel_size == 1
        and config.moe_latent_size is None
        and config.moe_shared_expert_intermediate_size == config.moe_ffn_hidden_size
        and not config.use_grouped_gemm_for_shared_expert
        and config.moe_shared_expert_glu_interleave_size is None
    )
    if mode == "fused" and not compatible:
        raise ValueError(
            "Fused MOK shared execution requires a dense, non-interleaved shared MLP "
            "with TP=1, the routed intermediate size, and no latent projection"
        )
    return mode != "native" and compatible


def build_megakernel_backend(
    *,
    config: TransformerConfig,
    ep_group: ProcessGroup,
    routed_experts: nn.Module,
    shared_experts: nn.Module | None,
    num_local_experts: int,
) -> MegakernelBackend:
    """Build the configured megakernel backend.

    Backend imports are deliberately lazy so ordinary MCore MoE configurations
    do not acquire optional megakernel dependencies.
    """
    backend = config.moe_megakernel_backend
    if backend == "mok":
        from megatron.core.transformer.moe.megakernel.mok.backend import MoKMegakernel

        return MoKMegakernel(
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
    if not megakernel_fuses_shared_expert(config):
        return config
    if backend == "mok":
        from megatron.core.transformer.moe.megakernel.mok.weights import (
            prepare_shared_expert_config,
        )

        return prepare_shared_expert_config(config)
    raise ValueError(f"Unsupported MoE megakernel backend: {backend!r}")


def megakernel_shared_expert_init_context(config: TransformerConfig):
    """Return the backend-specific context used to construct shared experts."""
    backend = config.moe_megakernel_backend
    if not megakernel_fuses_shared_expert(config):
        return nullcontext()
    if backend == "mok":
        from megatron.core.fp8_utils import get_fp8_disabled_context

        # MOK consumes native BF16 shared weights. Disable any model-wide TE
        # fp8_model_init context while the authoritative MCore module is built.
        return get_fp8_disabled_context(config, is_init=True)
    raise ValueError(f"Unsupported MoE megakernel backend: {backend!r}")
