# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from .spec import ResolvedScalingContext


@dataclass(frozen=True)
class ResolvedTrainingPolicy:
    """Optimizer-side policy for existing Megatron MuP hyperparameter multipliers."""

    context: ResolvedScalingContext
    optimizer_type: str = 'adam'

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
        return 'adam' in self.optimizer_type_lower

    @property
    def is_muon_optimizer(self) -> bool:
        return 'muon' in self.optimizer_type_lower

    @property
    def hidden_lr_multiplier(self) -> float:
        if not self.enabled or not self.uses_width_mup:
            return 1.0
        if self.is_sgd_optimizer:
            return 1.0
        return 1.0 / self.context.width_mult

    @property
    def hidden_vector_lr_multiplier(self) -> float:
        if not (self.enabled and self.uses_width_mup and self.is_sgd_optimizer):
            return 1.0
        return self.context.width_mult

    @property
    def hidden_eps_multiplier(self) -> float:
        if not (self.enabled and self.uses_width_mup and self.is_adam_optimizer):
            return 1.0
        return 1.0 / self.context.width_mult

    @property
    def vector_like_lr_multiplier(self) -> float:
        return self.hidden_vector_lr_multiplier


def build_legacy_mup_training_policy(
    *, mup_width_mult: float, optimizer_type: str = 'adam'
) -> ResolvedTrainingPolicy:
    return ResolvedTrainingPolicy(
        context=ResolvedScalingContext(use_mup=True, width_mult=mup_width_mult),
        optimizer_type=optimizer_type,
    )
