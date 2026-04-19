# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

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
    mup_base_hidden_size: Optional[int] = None
    mup_embedding_mult: float = 1.0
    mup_output_mult: float = 1.0
    mup_base_head_dim: Optional[float] = None
    mup_attn_scale_power: float = 1.0


@dataclass(frozen=True)
class ScalingReferences:
    current_hidden_size: int
    current_num_layers: int
    current_head_dim: int
    base_hidden_size: Optional[int] = None
    base_num_layers: Optional[int] = None
    base_head_dim: Optional[float] = None


@dataclass(frozen=True)
class CanonicalScalingSpec:
    recipe: Literal['none', 'mup', 'depth_mup']
    references: ScalingReferences
    embedding_mult: float = 1.0
    output_mult: float = 1.0
    attention_scale_power: float = 1.0
    residual_branch_depth_power: float = 0.0
    hidden_lr_depth_power: float = 0.0
    block_out_proj_init_depth_power: float = 0.0


@dataclass(frozen=True)
class ResolvedScalingContext:
    recipe: Literal['none', 'mup', 'depth_mup']
    references: ScalingReferences
    width_mult: float = 1.0
    depth_mult: float = 1.0
    embedding_mult: float = 1.0
    output_mult: float = 1.0
    attention_scale_power: float = 1.0
    residual_branch_depth_power: float = 0.0
    hidden_lr_depth_power: float = 0.0
    block_out_proj_init_depth_power: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.recipe != SCALING_RECIPE_NONE


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


def _non_default_scaling_fields(
    user_config: ScalingUserConfig, *, include_legacy_mup_fields: bool
) -> list[str]:
    candidates = {
        'scaling_base_hidden_size': user_config.base_hidden_size,
        'scaling_base_num_layers': user_config.base_num_layers,
        'scaling_base_head_dim': user_config.base_head_dim,
        'scaling_residual_branch_depth_power': user_config.residual_branch_depth_power,
        'scaling_hidden_lr_depth_power': user_config.hidden_lr_depth_power,
        'scaling_block_out_proj_init_depth_power': user_config.block_out_proj_init_depth_power,
    }
    if include_legacy_mup_fields:
        candidates['mup_base_hidden_size'] = user_config.mup_base_hidden_size
        candidates['mup_base_head_dim'] = user_config.mup_base_head_dim
        if user_config.mup_embedding_mult != 1.0:
            candidates['mup_embedding_mult'] = user_config.mup_embedding_mult
        if user_config.mup_output_mult != 1.0:
            candidates['mup_output_mult'] = user_config.mup_output_mult
        if user_config.mup_attn_scale_power != 1.0:
            candidates['mup_attn_scale_power'] = user_config.mup_attn_scale_power
    return [name for name, value in candidates.items() if value is not None]


def _infer_current_head_dim(config: Any) -> int:
    kv_channels = getattr(config, 'kv_channels', None)
    if kv_channels is not None:
        return kv_channels

    hidden_size = getattr(config, 'hidden_size')
    num_attention_heads = getattr(config, 'num_attention_heads')
    if num_attention_heads is None or num_attention_heads <= 0:
        raise AttributeError(
            "Cannot resolve current head dimension without kv_channels or a positive "
            "num_attention_heads value."
        )
    return hidden_size // num_attention_heads


def build_scaling_user_config(config: Any) -> ScalingUserConfig:
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
        mup_base_hidden_size=getattr(config, 'mup_base_hidden_size', None),
        mup_embedding_mult=getattr(config, 'mup_embedding_mult', 1.0),
        mup_output_mult=getattr(config, 'mup_output_mult', 1.0),
        mup_base_head_dim=getattr(config, 'mup_base_head_dim', None),
        mup_attn_scale_power=getattr(config, 'mup_attn_scale_power', 1.0),
    )


def canonicalize_scaling_user_config(user_config: ScalingUserConfig, config: Any) -> CanonicalScalingSpec:
    recipe = user_config.recipe
    if recipe is None:
        recipe = SCALING_RECIPE_MUP if user_config.use_mup_alias else SCALING_RECIPE_NONE
    elif user_config.use_mup_alias:
        if recipe != SCALING_RECIPE_MUP:
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
        non_default_fields = _non_default_scaling_fields(user_config, include_legacy_mup_fields=True)
        if non_default_fields:
            raise ValueError(
                "Scaling overrides require a non-'none' scaling recipe (for example `mup` or "
                "`depth_mup`). Non-default fields: "
                + ", ".join(non_default_fields)
            )
        references = ScalingReferences(
            current_hidden_size=config.hidden_size,
            current_num_layers=config.num_layers,
            current_head_dim=_infer_current_head_dim(config),
        )
        return CanonicalScalingSpec(recipe=SCALING_RECIPE_NONE, references=references)

    if base_hidden_size is None:
        base_hidden_size = config.hidden_size
    if base_hidden_size <= 0:
        raise AssertionError('--scaling-base-hidden-size must be positive.')

    base_num_layers = user_config.base_num_layers
    if base_num_layers is None:
        base_num_layers = config.num_layers
    if base_num_layers <= 0:
        raise AssertionError('--scaling-base-num-layers must be positive.')
    if base_head_dim is not None and base_head_dim <= 0:
        raise AssertionError('--scaling-base-head-dim must be positive.')

    references = ScalingReferences(
        current_hidden_size=config.hidden_size,
        current_num_layers=config.num_layers,
        current_head_dim=_infer_current_head_dim(config),
        base_hidden_size=base_hidden_size,
        base_num_layers=base_num_layers,
        base_head_dim=base_head_dim,
    )
    return CanonicalScalingSpec(
        recipe=recipe,
        references=references,
        embedding_mult=user_config.mup_embedding_mult,
        output_mult=user_config.mup_output_mult,
        attention_scale_power=user_config.mup_attn_scale_power,
        residual_branch_depth_power=float(user_config.residual_branch_depth_power or 0.0),
        hidden_lr_depth_power=float(user_config.hidden_lr_depth_power or 0.0),
        block_out_proj_init_depth_power=float(
            user_config.block_out_proj_init_depth_power or 0.0
        ),
    )


def resolve_scaling_context(canonical_spec: CanonicalScalingSpec) -> ResolvedScalingContext:
    if canonical_spec.recipe == SCALING_RECIPE_NONE:
        return ResolvedScalingContext(recipe=SCALING_RECIPE_NONE, references=canonical_spec.references)

    refs = canonical_spec.references
    assert refs.base_hidden_size is not None
    assert refs.base_num_layers is not None
    width_mult = refs.current_hidden_size / refs.base_hidden_size
    depth_mult = refs.current_num_layers / refs.base_num_layers
    output_mult = canonical_spec.output_mult
    if output_mult == 1.0 and width_mult != 1.0:
        output_mult = 1.0 / width_mult

    return ResolvedScalingContext(
        recipe=canonical_spec.recipe,
        references=refs,
        width_mult=width_mult,
        depth_mult=depth_mult,
        embedding_mult=canonical_spec.embedding_mult,
        output_mult=output_mult,
        attention_scale_power=canonical_spec.attention_scale_power,
        residual_branch_depth_power=canonical_spec.residual_branch_depth_power,
        hidden_lr_depth_power=canonical_spec.hidden_lr_depth_power,
        block_out_proj_init_depth_power=canonical_spec.block_out_proj_init_depth_power,
    )


def build_resolved_scaling_context(config: Any) -> ResolvedScalingContext:
    user_config = build_scaling_user_config(config)
    canonical_spec = canonicalize_scaling_user_config(user_config, config)
    return resolve_scaling_context(canonical_spec)


def sync_legacy_mup_fields(config: Any, context: ResolvedScalingContext) -> None:
    config.scaling_recipe = context.recipe
    config.use_mup = context.recipe == SCALING_RECIPE_MUP
    if context.recipe != SCALING_RECIPE_MUP:
        return

    config.scaling_base_hidden_size = context.references.base_hidden_size
    config.scaling_base_num_layers = context.references.base_num_layers
    config.scaling_base_head_dim = context.references.base_head_dim
    config.mup_base_hidden_size = context.references.base_hidden_size
    config.mup_base_head_dim = context.references.base_head_dim
    config.mup_width_mult = context.width_mult
    config.mup_output_mult = context.output_mult
