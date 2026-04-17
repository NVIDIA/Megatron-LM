# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from .spec import ResolvedScalingContext, build_resolved_scaling_context


@dataclass(frozen=True)
class ResolvedTrainingPolicy:
    context: ResolvedScalingContext

    @property
    def enabled(self) -> bool:
        return self.context.enabled


def build_resolved_training_policy(config) -> ResolvedTrainingPolicy:
    return ResolvedTrainingPolicy(build_resolved_scaling_context(config))
