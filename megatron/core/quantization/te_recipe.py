# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Format-neutral Transformer Engine quantization recipe and context helpers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, ContextManager

from megatron.core.enums import Fp8Recipe
from megatron.core.quantization.custom_recipe import get_cached_custom_recipe
from megatron.core.quantization.utils import is_custom_recipe_selected, is_quantization_enabled
from megatron.core.utils import is_te_min_version

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import transformer_engine.pytorch as te

    HAVE_TE = True
except ImportError:
    te = None
    HAVE_TE = False


def get_te_quantization_recipe(config: TransformerConfig) -> Any:
    """Build the selected global Transformer Engine quantization recipe.

    Args:
        config: Transformer configuration containing the global recipe selection.

    Returns:
        The selected Transformer Engine recipe, or ``None`` when quantization is disabled.
    """
    if config.custom_recipe is not None:
        return get_cached_custom_recipe(
            config,
            config.custom_recipe,
            fp8_dpa=config.fp8_dot_product_attention,
            fp8_mha=config.fp8_multi_head_attention,
        )
    if config.fp8:
        from megatron.core.fp8_utils import get_fp8_recipe

        return get_fp8_recipe(config)
    if config.fp4:
        from megatron.core.fp4_utils import get_fp4_recipe

        return get_fp4_recipe(config)
    return None


def get_quantization_alignment(config: TransformerConfig) -> int:
    """Return the required tensor alignment without probing an opaque factory."""
    if is_custom_recipe_selected(config):
        # Canonical and legacy custom spellings share the recipe's declared alignment;
        # the legacy FP8 enum would otherwise fall through to the 16-byte FP8 default.
        recipe = get_te_quantization_recipe(config)
        return getattr(recipe, "quantization_alignment", 128)
    if config.fp8:
        from megatron.core.fp8_utils import get_fp8_align_size

        return get_fp8_align_size(config.fp8_recipe)
    if config.fp4:
        from megatron.core.fp4_utils import get_fp4_align_size

        return get_fp4_align_size(config.fp4_recipe)
    return 0


def _te_validates_save_original_input() -> bool:
    """Feature-detect Transformer Engine's per-quantizer validation of ``save_original_input``."""
    if not HAVE_TE:
        return False
    try:
        from transformer_engine.pytorch.module._common import (  # noqa: F401
            can_reconstruct_wgrad_input_from_original,
        )
    except ImportError:
        return False
    return True


def supports_save_original_input(config: TransformerConfig) -> bool:
    """Return whether TE linears may keep their original input instead of a quantized copy.

    Megatron enables ``save_original_input`` on linears whose input is already retained by an
    upstream operation, such as the attention output projection consuming the saved
    dot-product-attention output. Built-in recipes keep their existing gates: delayed scaling
    cannot requantize deterministically, and older TE lacks the option. For custom recipes,
    Transformer Engine validates the resolved input quantizer at runtime and downgrades
    delayed-scaling, unsafe Hybrid, and stochastic-rounding NVFP4 quantizers with a warning,
    so the optimization is enabled only on TE builds that ship that validation.
    """
    if is_custom_recipe_selected(config):
        return _te_validates_save_original_input()
    if config.fp8:
        return config.fp8_recipe != Fp8Recipe.delayed and is_te_min_version("2.6.0dev0")
    if config.fp4:
        return is_te_min_version("2.7.0.dev0")
    return False


def get_quantization_context(
    config: TransformerConfig, layer_no: int = -1, is_init: bool = False
) -> ContextManager[Any]:
    """Return the selected global Transformer Engine quantization context.

    Args:
        config: Transformer configuration containing the global recipe selection.
        layer_no: Zero-based global layer index, or -1 for a context covering all layers.
        is_init: Whether the context is used during parameter initialization.

    Returns:
        A Transformer Engine context for the selected mode, or a no-op context.
    """
    if config.custom_recipe is None:
        if config.fp8:
            from megatron.core.fp8_utils import get_fp8_context

            return get_fp8_context(config, layer_no, is_init)
        if config.fp4:
            from megatron.core.fp4_utils import get_fp4_context

            return get_fp4_context(config, layer_no, is_init)
        return nullcontext()

    if not HAVE_TE:
        raise RuntimeError("--custom-recipe requires Transformer Engine.")
    if is_init:
        # Canonical custom parameter storage is intentionally not supported yet.
        # Materialize now so invalid import paths fail during model construction
        # and the same recipe object is reused by every forward context.
        get_te_quantization_recipe(config)
        return nullcontext()

    from megatron.core import parallel_state
    from megatron.core.fp8_utils import is_first_last_bf16_layer

    if is_first_last_bf16_layer(config, layer_no):
        return nullcontext()

    amax_group = None
    if parallel_state.model_parallel_is_initialized():
        # Compatibility point while quantization contexts still use MPU global groups.
        amax_group = parallel_state.get_amax_reduction_group(
            with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red
        )
    return te.autocast(
        enabled=True, recipe=get_te_quantization_recipe(config), amax_reduction_group=amax_group
    )


def get_quantization_disabled_context(
    config: TransformerConfig, is_init: bool = False
) -> ContextManager[Any]:
    """Return a context that disables the active Transformer Engine quantization mode."""
    if is_init:
        if not (config.fp8_param or config.fp4_param):
            return nullcontext()
        if config.custom_recipe is not None:
            return te.quantized_model_init(enabled=False)
        return te.fp8_model_init(enabled=False)
    if not is_quantization_enabled(config):
        return nullcontext()
    if config.custom_recipe is not None:
        return te.autocast(enabled=False)
    return te.fp8_autocast(enabled=False)
