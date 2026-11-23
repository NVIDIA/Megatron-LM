# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Application hooks for constructing optimizer parameter-group overrides."""

from dataclasses import dataclass

from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.optimizer_config import ParamKey
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.transformer.module import MegatronModule
from megatron.training.config.training_config import SchedulerConfig


@dataclass(frozen=True)
class OptimizerConfigOverrideContext:
    """Runtime inputs available after model construction and before optimizer construction."""

    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig
    model: list[MegatronModule]


@dataclass
class OptimizerConfigOverrideProvider:
    """Subclass to supply recipe-specific optimizer groups without patching training code.

    Dataclass subclasses can store recipe settings and round-trip through the training
    container's dictionary/YAML serialization. Custom import targets must be allowlisted
    when reading YAML, as for other application configuration classes.
    """

    def build_config_overrides(
        self, context: OptimizerConfigOverrideContext
    ) -> dict[ParamKey, ParamGroupOverride] | None:
        """Return overrides, or None to use the standard optimizer overrides.

        An empty dictionary intentionally disables standard overrides. To extend them,
        start with ``get_standard_config_overrides(context.optimizer_config)`` and merge
        the recipe's entries. Training merges its automatic muP entries afterwards,
        retaining their existing precedence for identical ParamKeys.
        """
        return None
