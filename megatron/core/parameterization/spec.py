# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

SCALING_RECIPE_NONE = 'none'
SCALING_RECIPE_MUP = 'mup'
SCALING_RECIPE_DEPTH_MUP = 'depth_mup'
SCALING_RECIPE_VALUES = (SCALING_RECIPE_NONE, SCALING_RECIPE_MUP, SCALING_RECIPE_DEPTH_MUP)


@dataclass(frozen=True)
class ScalingUserConfig:
    recipe: Optional[Literal['none', 'mup', 'depth_mup']] = None
    base_hidden_size: Optional[int] = None
    base_num_layers: Optional[int] = None
    base_head_dim: Optional[float] = None
    residual_branch_depth_power: Optional[float] = None
    hidden_lr_depth_power: Optional[float] = None
    block_out_proj_init_depth_power: Optional[float] = None
    use_mup_alias: bool = False
    mup_width_mult: Optional[float] = None
    mup_width_mult_explicit: bool = False
    mup_base_hidden_size: Optional[int] = None
    mup_embedding_mult: float = 1.0
    mup_output_mult: float = 1.0
    mup_base_head_dim: Optional[float] = None
    mup_attn_scale_power: float = 1.0


@dataclass(frozen=True)
class ScalingContext:
    """Internal scaling context for standard, width-MuP, and depth-MuP."""

    recipe: Literal['none', 'mup', 'depth_mup']
    width_mult: float = 1.0
    depth_mult: float = 1.0
    embedding_mult: float = 1.0
    output_mult: float = 1.0
    base_hidden_size: Optional[int] = None
    base_num_layers: Optional[int] = None
    base_head_dim: Optional[float] = None
    current_head_dim: Optional[int] = None
    attention_scale_power: float = 1.0
    residual_branch_depth_power: float = 0.0
    hidden_lr_depth_power: float = 0.0
    block_out_proj_init_depth_power: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.recipe != SCALING_RECIPE_NONE

    @property
    def uses_width_mup(self) -> bool:
        return self.recipe in (SCALING_RECIPE_MUP, SCALING_RECIPE_DEPTH_MUP)

    @property
    def use_mup(self) -> bool:
        return self.recipe == SCALING_RECIPE_MUP

    @property
    def is_depth_mup(self) -> bool:
        return self.recipe == SCALING_RECIPE_DEPTH_MUP


def _resolve_aliased_value(
    explicit_value: Optional[float | int],
    legacy_value: Optional[float | int],
    *,
    explicit_name: str,
    legacy_name: str,
) -> Optional[float | int]:
    if explicit_value is None:
        return legacy_value
    if legacy_value is None:
        return explicit_value
    if explicit_value != legacy_value:
        raise ValueError(
            f"{explicit_name} ({explicit_value}) conflicts with {legacy_name} ({legacy_value}). "
            f"Specify only one or set them to the same value."
        )
    return explicit_value


def _non_default_scaling_fields(user_config: ScalingUserConfig) -> list[str]:
    candidates: dict[str, object] = {
        'scaling_base_hidden_size': user_config.base_hidden_size,
        'scaling_base_num_layers': user_config.base_num_layers,
        'scaling_base_head_dim': user_config.base_head_dim,
        'scaling_residual_branch_depth_power': user_config.residual_branch_depth_power,
        'scaling_hidden_lr_depth_power': user_config.hidden_lr_depth_power,
        'scaling_block_out_proj_init_depth_power': user_config.block_out_proj_init_depth_power,
        'mup_base_hidden_size': user_config.mup_base_hidden_size,
        'mup_base_head_dim': user_config.mup_base_head_dim,
    }
    if user_config.mup_embedding_mult != 1.0:
        candidates['mup_embedding_mult'] = user_config.mup_embedding_mult
    if user_config.mup_output_mult != 1.0:
        candidates['mup_output_mult'] = user_config.mup_output_mult
    if user_config.mup_attn_scale_power != 1.0:
        candidates['mup_attn_scale_power'] = user_config.mup_attn_scale_power
    if user_config.mup_width_mult_explicit:
        candidates['mup_width_mult'] = user_config.mup_width_mult
    return [name for name, value in candidates.items() if value is not None]


def _infer_current_head_dim(config: Any) -> Optional[int]:
    kv_channels = getattr(config, 'kv_channels', None)
    if kv_channels is not None:
        return kv_channels

    hidden_size = getattr(config, 'hidden_size', None)
    num_attention_heads = getattr(config, 'num_attention_heads', None)
    if hidden_size is None or num_attention_heads is None:
        return None
    if num_attention_heads is None or num_attention_heads <= 0:
        raise AttributeError(
            "Cannot resolve current head dimension without kv_channels or a positive "
            "num_attention_heads value."
        )
    return hidden_size // num_attention_heads


def build_scaling_user_config(config: Any) -> ScalingUserConfig:
    raw_mup_width_mult = getattr(config, 'mup_width_mult', None)
    marker_present = hasattr(config, '_mup_width_mult_explicit')
    mup_width_mult_explicit = bool(getattr(config, '_mup_width_mult_explicit', False))
    if (
        not marker_present
        and not mup_width_mult_explicit
        and raw_mup_width_mult not in (None, 1.0)
    ):
        # Direct TransformerConfig construction has no argparse provenance marker.
        mup_width_mult_explicit = True

    return ScalingUserConfig(
        recipe=getattr(config, 'scaling_recipe', None),
        base_hidden_size=getattr(config, 'scaling_base_hidden_size', None),
        base_num_layers=getattr(config, 'scaling_base_num_layers', None),
        base_head_dim=getattr(config, 'scaling_base_head_dim', None),
        residual_branch_depth_power=getattr(config, 'scaling_residual_branch_depth_power', None),
        hidden_lr_depth_power=getattr(config, 'scaling_hidden_lr_depth_power', None),
        block_out_proj_init_depth_power=getattr(
            config, 'scaling_block_out_proj_init_depth_power', None
        ),
        use_mup_alias=bool(getattr(config, 'use_mup', False)),
        mup_width_mult=raw_mup_width_mult if mup_width_mult_explicit else None,
        mup_width_mult_explicit=mup_width_mult_explicit,
        mup_base_hidden_size=getattr(config, 'mup_base_hidden_size', None),
        mup_embedding_mult=getattr(config, 'mup_embedding_mult', 1.0),
        mup_output_mult=getattr(config, 'mup_output_mult', 1.0),
        mup_base_head_dim=getattr(config, 'mup_base_head_dim', None),
        mup_attn_scale_power=getattr(config, 'mup_attn_scale_power', 1.0),
    )

def build_scaling_context(config: Any) -> ScalingContext:
    user_config = build_scaling_user_config(config)
    recipe = user_config.recipe
    if recipe is None:
        recipe = SCALING_RECIPE_MUP if user_config.use_mup_alias else SCALING_RECIPE_NONE
    elif user_config.use_mup_alias and recipe != SCALING_RECIPE_MUP:
        raise ValueError(
            f"--scaling-recipe {recipe} conflicts with --use-mup. "
            "Use either the canonical MuP recipe or the legacy MuP alias, not both."
        )
    if recipe not in SCALING_RECIPE_VALUES:
        raise ValueError(f"Unsupported scaling recipe: {recipe}")

    base_hidden_size = _resolve_aliased_value(
        user_config.base_hidden_size,
        user_config.mup_base_hidden_size,
        explicit_name='--scaling-base-hidden-size',
        legacy_name='--mup-base-hidden-size',
    )
    base_head_dim = _resolve_aliased_value(
        user_config.base_head_dim,
        user_config.mup_base_head_dim,
        explicit_name='--scaling-base-head-dim',
        legacy_name='--mup-base-head-dim',
    )

    if recipe == SCALING_RECIPE_NONE:
        non_default_fields = _non_default_scaling_fields(user_config)
        if non_default_fields:
            raise ValueError(
                "Scaling overrides require a non-'none' scaling recipe (for example `mup` "
                "or `depth_mup`). Non-default fields: " + ", ".join(non_default_fields)
            )
        return ScalingContext(
            recipe=SCALING_RECIPE_NONE,
            current_head_dim=_infer_current_head_dim(config),
        )

    if base_hidden_size is None:
        base_hidden_size = config.hidden_size
    if base_hidden_size <= 0:
        raise AssertionError('--scaling-base-hidden-size must be positive.')

    current_num_layers = getattr(config, 'num_layers', None)
    if current_num_layers is None:
        if recipe == SCALING_RECIPE_DEPTH_MUP:
            raise AttributeError("Cannot resolve depth_mup without num_layers.")
        current_num_layers = 1

    base_num_layers = user_config.base_num_layers
    if base_num_layers is None:
        base_num_layers = current_num_layers
    if base_num_layers <= 0:
        raise AssertionError('--scaling-base-num-layers must be positive.')
    if base_head_dim is not None and base_head_dim <= 0:
        raise AssertionError('--scaling-base-head-dim must be positive.')

    width_mult = config.hidden_size / base_hidden_size
    if (
        user_config.mup_width_mult_explicit
        and user_config.mup_width_mult is not None
        and not math.isclose(
            user_config.mup_width_mult, width_mult, rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise ValueError(
            "--mup-width-mult is deprecated as an input and must match the derived "
            f"hidden_size / scaling_base_hidden_size value ({width_mult}). "
            f"Got --mup-width-mult={user_config.mup_width_mult}."
        )

    output_mult = user_config.mup_output_mult
    if output_mult == 1.0 and width_mult != 1.0:
        output_mult = 1.0 / width_mult

    residual_branch_depth_power = user_config.residual_branch_depth_power
    if residual_branch_depth_power is None:
        residual_branch_depth_power = -1.0 if recipe == SCALING_RECIPE_DEPTH_MUP else 0.0

    hidden_lr_depth_power = user_config.hidden_lr_depth_power
    if hidden_lr_depth_power is None:
        hidden_lr_depth_power = 0.0

    block_out_proj_init_depth_power = user_config.block_out_proj_init_depth_power
    if block_out_proj_init_depth_power is None:
        block_out_proj_init_depth_power = 0.5 if recipe == SCALING_RECIPE_DEPTH_MUP else 0.0

    return ScalingContext(
        recipe=recipe,
        width_mult=width_mult,
        depth_mult=current_num_layers / base_num_layers,
        embedding_mult=user_config.mup_embedding_mult,
        output_mult=output_mult,
        base_hidden_size=base_hidden_size,
        base_num_layers=base_num_layers,
        base_head_dim=base_head_dim,
        current_head_dim=_infer_current_head_dim(config),
        attention_scale_power=user_config.mup_attn_scale_power,
        residual_branch_depth_power=float(residual_branch_depth_power),
        hidden_lr_depth_power=float(hidden_lr_depth_power),
        block_out_proj_init_depth_power=float(block_out_proj_init_depth_power),
    )


def sync_legacy_mup_fields(config: Any, context: ScalingContext) -> None:
    config.scaling_recipe = context.recipe
    config.use_mup = context.recipe == SCALING_RECIPE_MUP
    config.mup_width_mult = context.width_mult
    config._mup_width_mult_explicit = False
    config.mup_embedding_mult = context.embedding_mult
    config.mup_output_mult = context.output_mult
    config.mup_attn_scale_power = context.attention_scale_power

    if context.recipe == SCALING_RECIPE_NONE:
        config.scaling_base_hidden_size = None
        config.scaling_base_num_layers = None
        config.scaling_base_head_dim = None
        config.scaling_residual_branch_depth_power = None
        config.scaling_hidden_lr_depth_power = None
        config.scaling_block_out_proj_init_depth_power = None
        config.mup_base_hidden_size = None
        config.mup_base_head_dim = None
        return

    config.scaling_base_hidden_size = context.base_hidden_size
    config.scaling_base_num_layers = context.base_num_layers
    config.scaling_base_head_dim = context.base_head_dim
    config.scaling_residual_branch_depth_power = context.residual_branch_depth_power
    config.scaling_hidden_lr_depth_power = context.hidden_lr_depth_power
    config.scaling_block_out_proj_init_depth_power = context.block_out_proj_init_depth_power
    config.mup_base_hidden_size = context.base_hidden_size
    config.mup_base_head_dim = context.base_head_dim
