# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from .spec import (
    SCALING_RECIPE_MUP,
    ResolvedScalingContext,
    ScalingReferences,
    build_resolved_scaling_context,
)


@dataclass(frozen=True)
class ResolvedTrainingPolicy:
    context: ResolvedScalingContext
    optimizer_type: str = 'adam'

    def __post_init__(self) -> None:
        if self.context.is_depth_mup and not self.is_adam_optimizer:
            raise ValueError(
                "scaling_recipe='depth_mup' currently supports Adam/AdamW only. "
                "SGD depth-mup requires explicit hidden-weight, hidden-bias, norm/vector, "
                "and input/output-bias rules and is intentionally out of scope for v1."
            )

    @property
    def enabled(self) -> bool:
        return self.context.enabled

    @property
    def optimizer_type_lower(self) -> str:
        return self.optimizer_type.lower()

    @property
    def uses_width_mup(self) -> bool:
        return self.context.uses_width_mup

    @property
    def is_sgd_optimizer(self) -> bool:
        return self.optimizer_type_lower == 'sgd'

    @property
    def is_adam_optimizer(self) -> bool:
        return self.optimizer_type_lower in ('adam', 'adamw')

    @property
    def is_muon_optimizer(self) -> bool:
        return 'muon' in self.optimizer_type_lower

    @property
    def hidden_lr_width_power(self) -> float:
        if not self.enabled or not self.uses_width_mup:
            return 0.0
        return 0.0 if self.is_sgd_optimizer else -1.0

    @property
    def hidden_lr_multiplier(self) -> float:
        if not self.enabled:
            return 1.0
        return (self.context.width_mult**self.hidden_lr_width_power) * (
            self.context.depth_mult**self.context.hidden_lr_depth_power
        )

    @property
    def vector_like_lr_multiplier(self) -> float:
        if not (self.enabled and self.uses_width_mup and self.is_sgd_optimizer):
            return 1.0
        return self.context.width_mult

    @property
    def hidden_eps_depth_power(self) -> float:
        if not (self.enabled and self.uses_width_mup and self.is_adam_optimizer):
            return 0.0
        return -1.0 if self.context.is_depth_mup else 0.0

    @property
    def hidden_eps_multiplier(self) -> float:
        if not (self.enabled and self.uses_width_mup and self.is_adam_optimizer):
            return 1.0
        return (1.0 / self.context.width_mult) * (
            self.context.depth_mult**self.hidden_eps_depth_power
        )


def build_resolved_training_policy(config, optimizer_type: str = 'adam') -> ResolvedTrainingPolicy:
    return ResolvedTrainingPolicy(
        context=build_resolved_scaling_context(config), optimizer_type=optimizer_type
    )


def build_legacy_mup_training_policy(
    *, mup_width_mult: float, optimizer_type: str = 'adam'
) -> ResolvedTrainingPolicy:
    return ResolvedTrainingPolicy(
        context=ResolvedScalingContext(
            recipe=SCALING_RECIPE_MUP,
            references=ScalingReferences(
                current_hidden_size=1,
                current_num_layers=1,
                current_head_dim=1,
                base_hidden_size=1,
                base_num_layers=1,
                base_head_dim=1.0,
            ),
            width_mult=mup_width_mult,
            depth_mult=1.0,
        ),
        optimizer_type=optimizer_type,
    )
