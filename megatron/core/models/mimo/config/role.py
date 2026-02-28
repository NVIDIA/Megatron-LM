# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Data classes for MIMO rank role management in multi-module pipeline parallelism."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
    stage positions.

    Args:
        modules: Dict mapping module names to their stage info for modules
            this rank participates in.
        language_module_name: Name of the language module, used to distinguish
            encoders from the language model.
    """

    modules: Dict[str, ModuleStageInfo] = field(default_factory=dict)
    language_module_name: Optional[str] = None

    @property
    def has_modality_modules(self) -> bool:
        """Return True if this rank participates in any modality (non-language) module."""
        return any(name != self.language_module_name for name in self.modules)

    @property
    def has_language_module(self) -> bool:
        """Return True if this rank participates in the language module."""
        return self.language_module_name is not None and self.language_module_name in self.modules

    @property
    def modality_module_names(self) -> List[str]:
        """Return names of modality modules (non-language) this rank participates in."""
        return [name for name in self.modules if name != self.language_module_name]

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
