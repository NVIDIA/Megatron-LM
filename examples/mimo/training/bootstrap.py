# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Runtime orchestrator for hetero MIMO training on the stock Megatron loop.

This is PR-E4 of the NMFW-516 hetero-MIMO upstreaming effort. It wires together
the merged glue (E1 model provider, E2 topology/grid args, E3 forward step) plus
the in-flight RNG/DDP (MM2), grad-sync (MM3), and data-selection modules into a
single :func:`build_mimo_runtime` entry the stock ``train()`` can consume.

Scope: the NON-COLOCATED nemotron VLM path only (encoder grid and language grid
occupy disjoint rank spans). The colocated three-phase schedule is out of scope
here.

``build_mimo_runtime`` is deliberately kept in this ``bootstrap.py`` module rather
than ``runtime.py`` (which MM2 / #5285 owns). At final integration the two may
fold together, but keeping them separate avoids clobbering MM2's file while both
PRs are in flight.

Boundary with the stock loop (handoff section 3): the entry PR (E5) bypasses
``setup_model_and_optimizer`` -- which would call ``get_model`` and re-wrap a
single top-level DDP -- and instead calls ``build_mimo_runtime`` to assemble the
per-submodule-DDP MimoModel, then hands the resulting model list straight to the
stock ``train()``. The optimizer is built separately (the prototype's
``get_mimo_optimizer`` needs the per-submodule DDP wrapping this function
performs, which is why no top-level DDP wrap is applied here).
"""

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
from examples.mimo.training.runtime import configure_module_rng, wrap_active_modules_with_ddp
from examples.mimo.training.topology import HeteroTopology, create_topology
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection

# RNG role offsets, matched to the prototype (hetero/runtime.py): the language
# module seeds from +20_000, the vision encoder from +10_000. Distinct offsets
# keep the (process-global) CUDA RNG tracker independent across module roles.
_LANGUAGE_SEED_OFFSET = 20_000
_ENCODER_SEED_OFFSET = 10_000


@dataclass
class MimoRuntime:
    """Everything the stock ``train()`` entry (E5) needs to drive a hetero MIMO run.

    Fields:
        model:
            The DDP-wrapped :class:`MimoModel` for this rank, returned as a
            single-element list so it drops straight into the stock ``train()``
            ``model`` argument (which always expects a list of model chunks).
            ``model[0].pg_collection`` carries this rank's per-module PGC (see
            below); ``model[0].config`` is the language ``TransformerConfig``
            already carrying the MIMO ``finalize_model_grads_func`` /
            ``grad_scale_func`` installed by :func:`configure_grad_sync`.
        topology:
            The :class:`HeteroTopology`. The schedule reads
            ``topology.schedule_pg_collection`` (the per-rank
            ``MultiModuleProcessGroupCollection``) and the p2p communicator is
            built from ``topology.grids``.
        communicator:
            The :class:`MultiModulePipelineCommunicator` the MIMO schedule uses
            for cross-module (encoder->language) P2P.
        data_iterator:
            This rank's role-aware data iterator, or ``None`` when the rank
            consumes no data (interior PP stages).
        pg_collection:
            This rank's per-module :class:`ProcessGroupCollection` (the language
            PGC on language-grid ranks, else the encoder PGC). Identical object
            to ``model.pg_collection``; surfaced here so the entry need not reach
            into the model. Distinct from ``topology.schedule_pg_collection``
            (the multi-module collection) -- see the module docstring and the
            ``pg_collection`` note in :func:`build_mimo_runtime`.
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

    Mirrors the prototype's ``HeteroTopology.module_dependency_map``: the encoder
    feeds the language module, the language module feeds nothing. LLM-only runs
    have no edges.
    """
    encoder_name = _encoder_module_name(topology)
    if encoder_name is None:
        return {MIMO_LANGUAGE_MODULE_KEY: []}
    return {encoder_name: [MIMO_LANGUAGE_MODULE_KEY], MIMO_LANGUAGE_MODULE_KEY: []}


def build_mimo_runtime(args: argparse.Namespace) -> MimoRuntime:
    """Assemble the per-rank hetero MIMO runtime for the stock training loop.

    Ports the prototype build sequence (hetero/runtime.py +
    hetero/loop.py:86-150), dropping the prototype's bespoke loop, logging,
    prefetch, timeline, and checkpoint machinery: those are owned by the stock
    ``train()`` (E5) or deferred to later PRs.

    Steps:
      1. Build the per-module grid specs (E2) and the topology (E2/E3).
      2. Determine which module(s) live on this rank from the topology grids.
      3. Seed RNG per active module role (encoder vs language get distinct
         offsets) via MM2's :func:`configure_module_rng`.
      4. Assemble the language and/or vision ``ModuleSpec`` (E1) and construct
         the role ``MimoModel`` -- WITHOUT a top-level DDP wrap, because the MIMO
         optimizer needs per-submodule DDP.
      5. DDP-wrap the active submodules (MM2 :func:`wrap_active_modules_with_ddp`).
      6. Attach this rank's per-module PGC as ``mimo_model.pg_collection`` for the
         stock ``train()`` / ``training_log`` path.
      7. Install the dual grad-finalization hook (MM3
         :func:`configure_grad_sync`).
      8. Build the p2p communicator and select this rank's data iterator.

    Returns a :class:`MimoRuntime` (see its docstring for the consumption
    contract).
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
    # exactly one module on each rank, so each rank configures RNG for one role.
    if rank_in_language:
        assert language_pg is not None
        configure_module_rng(args, language_pg, role_seed_offset=_LANGUAGE_SEED_OFFSET)
    elif rank_in_encoder:
        assert encoder_pg is not None
        configure_module_rng(args, encoder_pg, role_seed_offset=_ENCODER_SEED_OFFSET)

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

    Ported from the prototype (hetero/loop.py:250). The vision encoder emits a
    2D ``[B*S, H]`` activation, so its ``module_output_ndim`` is 2; the language
    module keeps the default 3D. ``model.config`` is the canonical
    ``ModelParallelConfig`` (the MimoModel's language config), identical on every
    rank, which the communicator uses for shape/dtype bookkeeping on cross-module
    transfers.
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
