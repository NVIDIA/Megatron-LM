# Copyright (c) 2025, 2026, NVIDIA CORPORATION. All rights reserved.

"""Data classes for MIMO rank role management in multi-module pipeline parallelism."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid

logger = logging.getLogger(__name__)

# Fixed key for the language module in module_to_grid_map and RankRole.
# MIMO always has exactly one language model, so this is not configurable.
MIMO_LANGUAGE_MODULE_KEY = "language"


class ModuleLayout(Enum):
    """Pipeline mode for MIMO multi-module parallelism.

    Determines how modules are distributed across ranks and which
    forward path is used.

    COLOCATED: All modules share the same ranks. Covers both legacy
        (no grid map, global parallel_state) and heterogeneous TP/DP
        (grid map with overlapping ranks). Uses _forward_all_modules.

    NON_COLOCATED: module_to_grid_map is set with non-overlapping rank
        ranges. Each rank runs EITHER encoder(s) OR the language model.
        Uses role-based dispatch with separate forward paths.
    """

    COLOCATED = "colocated"
    NON_COLOCATED = "non_colocated"


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
    stage positions. The language module is always identified by MIMO_LANGUAGE_MODULE_KEY.

    Args:
        modules: Dict mapping module names to their stage info for modules
            this rank participates in.
        mode: Pipeline mode determining forward path dispatch.
    """

    modules: Dict[str, ModuleStageInfo] = field(default_factory=dict)
    mode: ModuleLayout = ModuleLayout.COLOCATED

    @classmethod
    def build(
        cls,
        modality_module_names: List[str],
        module_to_grid_map: Optional[Dict[str, 'HyperCommGrid']] = None,
    ) -> 'RankRole':
        """Construct a RankRole from modality names and an optional grid map.

        This is the unified entry point — callers pass the same first
        argument (``modality_module_names``) in both colocated and
        non-colocated cases and let ``build`` dispatch based on what the
        grid map says:

            * ``module_to_grid_map is None`` → COLOCATED with no grid
              (legacy global parallel_state). Every module lives on
              every rank.
            * All grids span the same ranks (same ``rank_offset`` +
              ``size``) → COLOCATED across the shared rank range.
              Per-bridge ``ColocatedBridgeCommunicator`` instances
              handle heterogeneous TP/DP on the MimoModel side.
            * Grids differ → NON_COLOCATED; delegate to
              ``from_grid_map`` to pick up PP-stage info for each
              module this rank participates in.
        """
        if module_to_grid_map is None or cls._all_grids_colocated(module_to_grid_map):
            return cls.colocated(modality_module_names)
        return cls.from_grid_map(module_to_grid_map, modality_module_names)

    @staticmethod
    def _all_grids_colocated(module_to_grid_map: Dict[str, 'HyperCommGrid']) -> bool:
        grids = list(module_to_grid_map.values())
        first = grids[0]
        return all(
            g.rank_offset == first.rank_offset and g.size == first.size for g in grids[1:]
        )

    @classmethod
    def colocated(cls, modality_module_names: List[str]) -> 'RankRole':
        """Create a role for colocated layout: every module on every rank, PP=1.

        Args:
            modality_module_names: Modality module names (e.g. ``["images",
                "audio"]``). The language model key is appended automatically
                so the arg shape matches ``from_grid_map``.

        Prefer ``RankRole.build`` unless you specifically need the colocated
        factory without the grid-map dispatch.
        """
        all_module_names = list(modality_module_names) + [MIMO_LANGUAGE_MODULE_KEY]
        return cls(
            modules={
                name: ModuleStageInfo(is_first_stage=True, is_last_stage=True)
                for name in all_module_names
            },
            mode=ModuleLayout.COLOCATED,
        )

    @classmethod
    def from_grid_map(
        cls, module_to_grid_map: Dict[str, HyperCommGrid], modality_module_names: List[str]
    ) -> 'RankRole':
        """Create a role from a module-to-grid mapping for non-colocated PP.

        Determines which modules the current rank participates in and its
        pipeline stage position within each module.

        Args:
            module_to_grid_map: Dict mapping module names to HyperCommGrid objects.
                Must contain keys matching modality_module_names + MIMO_LANGUAGE_MODULE_KEY.
            modality_module_names: List of modality module names (e.g., ["images", "audio"]).

        Returns:
            RankRole for the current rank.

        Raises:
            ValueError: If grid map keys don't match expected module names.
            RuntimeError: If current rank is not in any module grid.

        Prefer ``RankRole.build`` unless you specifically need the non-colocated
        factory without the grid-map dispatch.
        """
        # Validate keys
        expected_keys = set(modality_module_names) | {MIMO_LANGUAGE_MODULE_KEY}
        grid_keys = set(module_to_grid_map.keys())
        if grid_keys != expected_keys:
            raise ValueError(
                f"module_to_grid_map keys must match modality module names + "
                f"'{MIMO_LANGUAGE_MODULE_KEY}'. Missing: {expected_keys - grid_keys}, "
                f"Extra: {grid_keys - expected_keys}"
            )

        current_rank = dist.get_rank()
        modules = {}

        for module_name, grid in module_to_grid_map.items():
            if not (grid.rank_offset <= current_rank < grid.rank_offset + grid.size):
                continue

            if "pp" not in grid.dim_names:
                modules[module_name] = ModuleStageInfo(is_first_stage=True, is_last_stage=True)
                continue

            pp_group = grid.get_pg("pp")
            pp_rank = pp_group.rank()
            pp_size = pp_group.size()
            is_first = pp_rank == 0
            is_last = pp_rank == pp_size - 1
            logger.info(
                f"[RankRole.from_grid_map] Rank {current_rank}: module={module_name}, "
                f"pp_rank={pp_rank}/{pp_size}, is_first_stage={is_first}, is_last_stage={is_last}"
            )
            modules[module_name] = ModuleStageInfo(is_first_stage=is_first, is_last_stage=is_last)

        if not modules:
            raise RuntimeError(
                f"Rank {current_rank} is not in any module grid. "
                f"Check module_to_grid_map configuration."
            )

        return cls(modules=modules, mode=ModuleLayout.NON_COLOCATED)

    @property
    def has_modality_modules(self) -> bool:
        """Return True if this rank participates in any modality (non-language) module."""
        return any(name != MIMO_LANGUAGE_MODULE_KEY for name in self.modules)

    @property
    def has_language_module(self) -> bool:
        """Return True if this rank participates in the language module."""
        return MIMO_LANGUAGE_MODULE_KEY in self.modules

    @property
    def modality_module_names(self) -> List[str]:
        """Return names of modality modules (non-language) this rank participates in."""
        return [name for name in self.modules if name != MIMO_LANGUAGE_MODULE_KEY]

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
