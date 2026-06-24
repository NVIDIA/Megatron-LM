# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-rank heterogeneous MIMO runtime assembly for the non-colocated nemotron VLM path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import torch

from examples.mimo.model_providers.nemotron_moe_vlm import (
    language_model_spec,
    vision_submodules_spec,
)
from examples.mimo.training.args import build_module_grid_specs
from examples.mimo.training.data import select_data_iterator
from examples.mimo.training.grad_sync import configure_grad_sync
from examples.mimo.training.runtime import wrap_active_modules_with_ddp
from examples.mimo.training.topology import HeteroTopology, create_topology
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.initialize import _set_random_seed

# RNG role offsets: distinct offsets keep the process-global CUDA RNG tracker
# independent across module roles (language +20_000, vision encoder +10_000).
_LANGUAGE_SEED_OFFSET = 20_000
_ENCODER_SEED_OFFSET = 10_000


@dataclass
class MimoRuntime:
    """Per-rank state the stock ``train()`` needs to drive a hetero MIMO run.

    Attributes:
        model: Single-element list holding this rank's DDP-wrapped :class:`MimoModel`.
        topology: The :class:`HeteroTopology` carrying grids and the schedule PGC.
        communicator: The :class:`MultiModulePipelineCommunicator` for cross-module P2P.
        data_iterator: This rank's role-aware data iterator, or ``None`` if it consumes no data.
        pg_collection: This rank's per-module PGC (language on LLM ranks, else encoder).
    """

    model: list
    topology: HeteroTopology
    communicator: MultiModulePipelineCommunicator
    data_iterator: Optional[object]
    pg_collection: ProcessGroupCollection


def _encoder_module_name(topology: HeteroTopology) -> Optional[str]:
    """Return the single modality (encoder) grid name, or ``None`` for an LLM-only run."""
    names = [name for name in topology.grids if name != MIMO_LANGUAGE_MODULE_KEY]
    return names[0] if names else None


def _module_dependency_map(topology: HeteroTopology) -> dict:
    """Build the encoder->language dependency map the communicator's topology arg needs.

    The encoder feeds the language module, the language module feeds nothing;
    LLM-only runs have no edges.
    """
    encoder_name = _encoder_module_name(topology)
    if encoder_name is None:
        return {MIMO_LANGUAGE_MODULE_KEY: []}
    return {encoder_name: [MIMO_LANGUAGE_MODULE_KEY], MIMO_LANGUAGE_MODULE_KEY: []}


def _seed_module_rng(
    args: argparse.Namespace, pg_collection: ProcessGroupCollection, role_seed_offset: int
) -> None:
    """Seed host + per-module CUDA RNG for one module role via the stock _set_random_seed.

    A per-role offset keeps disjoint modules' RNG independent; the role's parallel
    groups supply the pp/dp/tp/ep/etp ranks since these ranks have no initialized mpu.
    """
    _set_random_seed(
        args.seed + role_seed_offset,
        args.data_parallel_random_init,
        args.te_rng_tracker,
        args.inference_rng_tracker,
        use_cudagraphable_rng=args.cuda_graph_impl != "none",
        pp_group=pg_collection.pp,
        dp_group=pg_collection.dp_cp,
        tp_group=pg_collection.tp,
        ep_group=pg_collection.ep,
        etp_group=pg_collection.expt_tp,
    )


def build_mimo_runtime(args: argparse.Namespace) -> MimoRuntime:
    """Assemble the per-rank hetero MIMO runtime and return a :class:`MimoRuntime`.

    Builds the grid specs and topology, seeds per-role RNG, constructs the role
    ``MimoModel`` without a top-level DDP wrap (the MIMO optimizer needs
    per-submodule DDP), DDP-wraps the active submodules, attaches the per-module
    PGC, installs the grad-finalization hook, and builds the p2p communicator and
    data iterator.
    """
    world_size = torch.distributed.get_world_size()
    specs = build_module_grid_specs(args, world_size)
    topology = create_topology(specs)

    # --- 2. Resolve this rank's role from the per-module grids. ------------
    # Non-colocated: each rank is in exactly one module grid. A grid membership
    # check (is_current_rank_in_grid) is the source of truth; module_pgs only
    # carries the collections for modules this rank participates in.
    encoder_name = _encoder_module_name(topology)
    rank_in_language = topology.grids[MIMO_LANGUAGE_MODULE_KEY].is_current_rank_in_grid()
    rank_in_encoder = (
        encoder_name is not None
        and topology.grids[encoder_name].is_current_rank_in_grid()
    )

    language_pg = topology.module_pgs.get(MIMO_LANGUAGE_MODULE_KEY)
    encoder_pg = topology.module_pgs.get(encoder_name) if encoder_name is not None else None

    # --- 3. Seed RNG for the single active module role on this rank. -------
    # The CUDA RNG tracker is process-global; the non-colocated layout puts
    # exactly one module on each rank, so each rank seeds RNG for one role via
    # the stock _set_random_seed, threading that role's parallel groups.
    if rank_in_language:
        assert language_pg is not None
        _seed_module_rng(args, language_pg, _LANGUAGE_SEED_OFFSET)
    elif rank_in_encoder:
        assert encoder_pg is not None
        _seed_module_rng(args, encoder_pg, _ENCODER_SEED_OFFSET)

    # --- 4. Build the role MimoModel (no top-level DDP). -------------------
    modality_submodules_spec = {}
    special_token_ids = {}
    if encoder_name is not None:
        modality_submodules_spec[encoder_name] = vision_submodules_spec(
            args,
            encoder_pg if rank_in_encoder else None,
            topology.grids[encoder_name],
        )
        special_token_ids[encoder_name] = args.image_token_id

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec(
            args,
            language_pg if rank_in_language else None,
            topology.grids[MIMO_LANGUAGE_MODULE_KEY],
        ),
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        module_to_grid_map=topology.grids,
    )

    mimo_model = MimoModel(
        mimo_config,
        cp_group=language_pg.cp if rank_in_language else None,
        tp_group=language_pg.tp if rank_in_language else None,
    )
    mimo_model.to(torch.device("cuda"))
    if not getattr(args, "fp32", False):
        mimo_model.to(torch.bfloat16)

    # --- 5. DDP-wrap the active submodules (per-submodule DDP). ------------
    wrap_active_modules_with_ddp(args, mimo_model, topology)

    # --- 6. Attach this rank's per-module PGC for the stock train() path. --
    # train()/training_log reads model.pg_collection for DP-keyed reductions and
    # logging. This is the PER-MODULE PGC (language on LLM ranks, encoder on
    # encoder ranks) -- NOT topology.schedule_pg_collection (the multi-module
    # collection the schedule consumes). They are kept distinct deliberately.
    rank_pg_collection = language_pg if rank_in_language else encoder_pg
    mimo_model.pg_collection = rank_pg_collection

    # --- 7. Install the dual (language + encoder) grad-finalization hook. --
    configure_grad_sync(args, mimo_model, topology)

    # --- 8. p2p communicator + role-aware data iterator. ------------------
    communicator = build_pipeline_communicator(mimo_model, topology)
    data_iterator = select_data_iterator(args, topology)

    return MimoRuntime(
        model=[mimo_model],
        topology=topology,
        communicator=communicator,
        data_iterator=data_iterator,
        pg_collection=rank_pg_collection,
    )


def build_pipeline_communicator(
    model: MimoModel, topology: HeteroTopology
) -> MultiModulePipelineCommunicator:
    """Build the MIMO cross-module P2P communicator the train schedule uses.

    The vision encoder emits a 2D ``[B*S, H]`` activation, so its
    ``module_output_ndim`` is 2; the language module keeps the default 3D.
    ``model.config`` is the canonical ``ModelParallelConfig`` (the MimoModel's
    language config), identical on every rank, used for shape/dtype bookkeeping
    on cross-module transfers.
    """
    encoder_name = _encoder_module_name(topology)
    module_output_ndim = {}
    if encoder_name is not None:
        module_output_ndim[encoder_name] = 2

    return MultiModulePipelineCommunicator(
        topology.grids,
        _module_dependency_map(topology),
        model.config,
        dim_mapping={"s": 0, "h": 2, "b": 1},
        module_output_ndim=module_output_ndim,
    )
