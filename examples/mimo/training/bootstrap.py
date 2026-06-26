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
from examples.mimo.model_providers.radio_encoder import RADIO_ENCODER_MODULE_NAME
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

# Per-role seed offsets keep the disjoint modules' RNG independent.
_LANGUAGE_SEED_OFFSET = 20_000
_ENCODER_SEED_OFFSET = 10_000


@dataclass
class MimoRuntime:
    """Per-rank state the stock ``train()`` needs to drive a hetero MIMO run."""

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


def _resolve_role(topology: HeteroTopology):
    """Resolve this rank's module role from the grids (non-colocated: one grid per rank)."""
    encoder_name = _encoder_module_name(topology)
    rank_in_language = topology.grids[MIMO_LANGUAGE_MODULE_KEY].is_current_rank_in_grid()
    rank_in_encoder = (
        encoder_name is not None and topology.grids[encoder_name].is_current_rank_in_grid()
    )
    language_pg = topology.module_pgs.get(MIMO_LANGUAGE_MODULE_KEY)
    encoder_pg = topology.module_pgs.get(encoder_name) if encoder_name is not None else None
    return encoder_name, rank_in_language, rank_in_encoder, language_pg, encoder_pg


def _seed_module_rng(
    args: argparse.Namespace, pg_collection: ProcessGroupCollection, role_seed_offset: int
) -> None:
    """Seed host + CUDA RNG for one module role from its parallel groups (no mpu here)."""
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


def mimo_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage=None,
    *,
    config=None,
    pg_collection=None,
    topology: HeteroTopology,
    args: argparse.Namespace,
) -> MimoModel:
    """Build this rank's bare ``MimoModel`` (no DDP wrap). get_model-shaped provider."""
    encoder_name, rank_in_language, rank_in_encoder, language_pg, _ = _resolve_role(topology)

    modality_submodules_spec = {}
    special_token_ids = {}
    if encoder_name is not None:
        encoder_pg = topology.module_pgs.get(encoder_name)
        modality_submodules_spec[encoder_name] = vision_submodules_spec(
            args, encoder_pg if rank_in_encoder else None, topology.grids[encoder_name]
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
    return mimo_model


def build_mimo_runtime(args: argparse.Namespace) -> MimoRuntime:
    """Assemble this rank's hetero MIMO runtime for the stock ``train()`` loop."""
    world_size = torch.distributed.get_world_size()
    specs = build_module_grid_specs(args, world_size, RADIO_ENCODER_MODULE_NAME)
    topology = create_topology(specs)

    _, rank_in_language, rank_in_encoder, language_pg, encoder_pg = _resolve_role(topology)

    # The model is built eagerly (before pretrain() seeds), so weight init needs the
    # per-role seed now; pretrain's later re-seed resets the tracker.
    if rank_in_language:
        assert language_pg is not None
        _seed_module_rng(args, language_pg, _LANGUAGE_SEED_OFFSET)
    elif rank_in_encoder:
        assert encoder_pg is not None
        _seed_module_rng(args, encoder_pg, _ENCODER_SEED_OFFSET)

    mimo_model = mimo_model_provider(topology=topology, args=args)
    wrap_active_modules_with_ddp(args, mimo_model, topology)

    # Per-module PGC (not the multi-module schedule PGC) for train()'s DP reductions.
    rank_pg_collection = language_pg if rank_in_language else encoder_pg
    mimo_model.pg_collection = rank_pg_collection

    configure_grad_sync(args, mimo_model, topology)
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
    """Build the cross-module P2P communicator (vision encoder emits 2D activations)."""
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
