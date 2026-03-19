# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Data classes for MIMO rank role management in multi-module pipeline parallelism."""

from dataclasses import dataclass, field
from typing import Dict, List

# Fixed key for the language module in module_to_grid_map and RankRole.
# MIMO always has exactly one language model, so this is not configurable.
LANGUAGE_MODULE_KEY = "language"


@dataclass
class ModuleStageInfo:
    """Information about a rank's stage position within a module's pipeline.

    Args:
        is_first_stage: True if this rank is the first PP stage for this module.
        is_last_stage: True if this rank is the last PP stage for this module.
    """

    is_first_stage: bool
    is_last_stage: bool


@dataclass
class RankRole:
    """Describes what modules this rank participates in for multi-module PP.

    This class captures the role of a specific rank in a multi-module pipeline
    parallel setup, tracking which modules the rank participates in and their
    stage positions. The language module is always identified by LANGUAGE_MODULE_KEY.

    Args:
        modules: Dict mapping module names to their stage info for modules
            this rank participates in.
        colocated: If True, all modules run on all ranks (no multi-module PP).
            The forward path uses the colocated codepath which supports
            PartitionAdapter and PackedSeqParams.
    """

    modules: Dict[str, ModuleStageInfo] = field(default_factory=dict)
    colocated: bool = False

    @classmethod
    def all_modules(cls, module_names: List[str]) -> 'RankRole':
        """Create a role for the colocated case: every module, first+last stage."""
        return cls(
            modules={
                name: ModuleStageInfo(is_first_stage=True, is_last_stage=True)
                for name in module_names
            },
            colocated=True,
        )

    @property
    def has_modality_modules(self) -> bool:
        """Return True if this rank participates in any modality (non-language) module."""
        return any(name != LANGUAGE_MODULE_KEY for name in self.modules)

    @property
    def has_language_module(self) -> bool:
        """Return True if this rank participates in the language module."""
        return LANGUAGE_MODULE_KEY in self.modules

    @property
    def modality_module_names(self) -> List[str]:
        """Return names of modality modules (non-language) this rank participates in."""
        return [name for name in self.modules if name != LANGUAGE_MODULE_KEY]

    def is_first_stage(self, module_name: str) -> bool:
        """Check if this rank is the first stage for a given module."""
        if module_name not in self.modules:
            return False
        return self.modules[module_name].is_first_stage

    def is_last_stage(self, module_name: str) -> bool:
        """Check if this rank is the last stage for a given module."""
        if module_name not in self.modules:
            return False
        return self.modules[module_name].is_last_stage
