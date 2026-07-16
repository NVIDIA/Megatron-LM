# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model builder for the heterogeneous MIMO training example."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional

import torch

from examples.mimo.model_providers import resolve_provider
from examples.mimo.training.grad_sync import configure_grad_sync
from examples.mimo.training.runtime import (
    configure_composite_no_sync,
    configure_module_rng,
    configure_namespaced_module_rng,
    wrap_active_modules_with_ddp,
)
from examples.mimo.training.topology import HeteroTopology
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.module import Float16Module
from megatron.training.global_vars import get_args
from megatron.training.models.base import ModelBuilder, ModelConfig, compose_hooks

_LANGUAGE_SEED_OFFSET = 20_000
# Add per-encoder offsets before wiring more than one encoder grid.
_ENCODER_SEED_OFFSET = 10_000


@dataclass(kw_only=True)
class MimoBuildConfig(ModelConfig):
    """Runtime-only topology used by :class:`MimoModelBuilder`.

    ``_topology`` is underscore-prefixed so ``ModelConfig`` skips it during serialization;
    only the ``builder`` ClassVar is written into the checkpoint. The builder reads parsed
    args from the global :func:`get_args`.
    """

    builder: ClassVar[str] = "examples.mimo.training.builder.MimoModelBuilder"
    _topology: Optional[HeteroTopology] = field(default=None)


def _resolve_role(topology: HeteroTopology):
    """Resolve this rank's single active module (non-colocated: one grid per rank).

    Returns ``(module_name, is_language, pg_collection)`` for the module this rank
    participates in; raises if the rank is in zero or multiple module grids.
    """
    active = [name for name, grid in topology.grids.items() if grid.is_current_rank_in_grid()]
    if len(active) != 1:
        raise ValueError(
            "Non-colocated MIMO requires exactly one active language or encoder role per rank; "
            f"this rank is in {active}"
        )
    name = active[0]
    return name, name == MIMO_LANGUAGE_MODULE_KEY, topology.module_pgs[name]


def _active_module_names(topology: HeteroTopology) -> list[str]:
    active = [name for name, grid in topology.grids.items() if grid.is_current_rank_in_grid()]
    if topology.is_colocated:
        if set(active) != set(topology.grids):
            raise ValueError("colocated MIMO requires every module on every rank")
    elif len(active) != 1:
        raise ValueError(f"non-colocated MIMO expected one active module, got {active}")
    return active


class MimoModelBuilder(ModelBuilder[MimoModel, MimoBuildConfig]):
    """Build and prepare this rank's active heterogeneous MIMO module."""

    def __init__(self, model_config: MimoBuildConfig):
        super().__init__(model_config)
        if model_config._topology is None:
            raise ValueError("MimoBuildConfig requires a topology")
        self._topology = model_config._topology
        self._rng_namespaces: dict[str, str] = {}
        self._rng_construction_seeds: dict[str, int] = {}

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> MimoModel:
        """Build the bare rank-local MIMO model; the shared lifecycle places it later."""
        del pg_collection, pre_process, post_process, vp_stage
        topology = self._topology
        args = get_args()
        provider = resolve_provider(args)
        active_names = _active_module_names(topology)

        provider_token_ids = provider.special_token_ids(args)
        modality_submodules_spec = {}
        special_token_ids = {}
        for name, grid in topology.grids.items():
            if name == MIMO_LANGUAGE_MODULE_KEY:
                continue
            if name not in provider.encoder_specs or name not in provider_token_ids:
                raise ValueError(f"provider defines no encoder spec/token for module {name!r}")
            pg = topology.module_pgs[name] if name in active_names else None
            modality_submodules_spec[name] = provider.encoder_specs[name](args, pg, grid)
            special_token_ids[name] = provider_token_ids[name]

        mimo_config = MimoModelConfig(
            language_model_spec=provider.language_spec(
                args,
                (
                    topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
                    if MIMO_LANGUAGE_MODULE_KEY in active_names
                    else None
                ),
                topology.grids[MIMO_LANGUAGE_MODULE_KEY],
            ),
            modality_submodules_spec=modality_submodules_spec,
            special_token_ids=special_token_ids,
            module_to_grid_map=topology.grids,
        )
        return MimoModel(
            mimo_config,
            cp_group=(
                topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY].cp
                if MIMO_LANGUAGE_MODULE_KEY in active_names
                else None
            ),
            tp_group=(
                topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY].tp
                if MIMO_LANGUAGE_MODULE_KEY in active_names
                else None
            ),
            rng_namespaces=self._rng_namespaces,
            rng_construction_seeds=self._rng_construction_seeds,
        )

    def build_distributed_models(
        self,
        pg_collection: ProcessGroupCollection,
        ddp_config: DistributedDataParallelConfig | None = None,
        overlap_param_gather_with_optimizer_step: bool = False,
        use_megatron_fsdp: bool = False,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = False,
        mixed_precision_wrapper: (
            Callable[[Any, MegatronModule], MegatronModule] | None
        ) = Float16Module,
        model_type: ModelType = ModelType.encoder_or_decoder,
    ) -> list[MimoModel]:
        """Seed, build, prepare, and configure the active rank-local MIMO model."""
        if wrap_with_ddp and ddp_config is None:
            raise ValueError("ddp_config is required when wrap_with_ddp is True")

        topology = self._topology
        args = get_args()
        self._rng_namespaces = {}
        self._rng_construction_seeds = {}
        try:
            is_colocated = topology.is_colocated
            if is_colocated:
                active_names = _active_module_names(topology)
                encoder_index = 0
                for name in active_names:
                    if name == MIMO_LANGUAGE_MODULE_KEY:
                        offset = _LANGUAGE_SEED_OFFSET
                    else:
                        offset = _ENCODER_SEED_OFFSET + encoder_index * 1_000
                        encoder_index += 1
                    self._rng_namespaces[name] = f"mimo.{name}"
                    self._rng_construction_seeds[name] = configure_namespaced_module_rng(
                        args,
                        topology.module_pgs[name],
                        offset,
                        self._rng_namespaces[name],
                        data_parallel_random_init,
                    )
                module_pg = topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
                rng_state_key_prefix = ""
            else:
                active_name, is_language, module_pg = _resolve_role(topology)
                role_seed_offset = _LANGUAGE_SEED_OFFSET if is_language else _ENCODER_SEED_OFFSET
                rng_state_key_prefix = "language." if is_language else "encoder."
                configure_module_rng(args, module_pg, role_seed_offset, data_parallel_random_init)

            built_with_meta_device = getattr(args, "init_model_with_meta_device", False)
            if built_with_meta_device:
                with torch.device("meta"):
                    mimo_model = self.build_model(pg_collection)
            else:
                mimo_model = self.build_model(pg_collection)
        finally:
            self._rng_namespaces = {}
            self._rng_construction_seeds = {}

        raw_mimo_model = mimo_model
        if is_colocated:
            # Communicator groups belong to the bare model even if an outer hook replaces it.
            topology.register_model(raw_mimo_model)
        mimo_model.model_type = model_type
        model_list = compose_hooks(self._model_config.pre_wrap_hooks)([mimo_model])
        if len(model_list) != 1:
            raise ValueError(
                f"MIMO pre-wrap hooks must return exactly one outer model; got {len(model_list)}"
            )
        mimo_model = model_list[0]

        wrap_active_modules_with_ddp(args, mimo_model, topology, data_parallel_random_init)
        if is_colocated:
            configure_composite_no_sync(mimo_model)
        configure_grad_sync(args, mimo_model, topology)
        mimo_model.pg_collection = module_pg
        mimo_model.rng_state_key_prefix = rng_state_key_prefix

        model_list = compose_hooks(self._model_config.post_wrap_hooks)([mimo_model])
        if len(model_list) != 1:
            raise ValueError(
                f"MIMO post-wrap hooks must return exactly one outer model; got {len(model_list)}"
            )
        return model_list
