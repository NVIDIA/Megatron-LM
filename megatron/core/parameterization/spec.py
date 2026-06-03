# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

SCALING_RECIPE_NONE = 'none'
SCALING_RECIPE_MUP = 'mup'
SCALING_RECIPE_VALUES = (SCALING_RECIPE_NONE, SCALING_RECIPE_MUP)


@dataclass(frozen=True)
class ScalingUserConfig:
    recipe: Optional[Literal['none', 'mup']] = None
    base_hidden_size: Optional[int] = None
    base_head_dim: Optional[float] = None
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
    """Internal scaling context for standard and width-MuP parameterization."""

    recipe: Literal['none', 'mup']
    width_mult: float = 1.0
    embedding_mult: float = 1.0
    output_mult: float = 1.0
    base_hidden_size: Optional[int] = None
    base_head_dim: Optional[float] = None
    attention_scale_power: float = 1.0

    @property
    def enabled(self) -> bool:
        return self.recipe != SCALING_RECIPE_NONE

    @property
    def uses_width_mup(self) -> bool:
        return self.recipe == SCALING_RECIPE_MUP

    @property
    def use_mup(self) -> bool:
        return self.uses_width_mup


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
        'scaling_base_head_dim': user_config.base_head_dim,
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
        base_head_dim=getattr(config, 'scaling_base_head_dim', None),
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
                "Scaling overrides require a non-'none' scaling recipe. Non-default fields: "
                + ", ".join(non_default_fields)
            )
        return ScalingContext(recipe=SCALING_RECIPE_NONE)

    if base_hidden_size is None:
        base_hidden_size = config.hidden_size
    if base_hidden_size <= 0:
        raise AssertionError('--scaling-base-hidden-size must be positive.')
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

    return ScalingContext(
        recipe=SCALING_RECIPE_MUP,
        width_mult=width_mult,
        embedding_mult=user_config.mup_embedding_mult,
        output_mult=output_mult,
        base_hidden_size=base_hidden_size,
        base_head_dim=base_head_dim,
        attention_scale_power=user_config.mup_attn_scale_power,
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
        config.scaling_base_head_dim = None
        config.mup_base_hidden_size = None
        config.mup_base_head_dim = None
        return

    config.scaling_base_hidden_size = context.base_hidden_size
    config.scaling_base_head_dim = context.base_head_dim
    config.mup_base_hidden_size = context.base_hidden_size
    config.mup_base_head_dim = context.base_head_dim
