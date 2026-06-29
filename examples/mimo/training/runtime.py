# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-rank runtime assembly (RNG seeding, freezing, DDP wrapping, model build) for hetero MIMO."""

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
from examples.mimo.training.topology import HeteroTopology, create_topology
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_pg_rank, get_pg_size
from megatron.training.initialize import _set_random_seed
from megatron.training.training import resolve_ddp_bucket_size, wrap_model_chunks_with_ddp
from megatron.training.utils import print_rank_0

# Per-role seed offsets keep the disjoint modules' RNG independent.
_LANGUAGE_SEED_OFFSET = 20_000
_ENCODER_SEED_OFFSET = 10_000


class _EncoderFloat16Module(Float16Module):
    """Float16Module that keeps encoder outputs in model precision for the bridge."""

    def forward(self, *inputs, fp32_output=False, **kwargs):  # noqa: D102
        return super().forward(*inputs, fp32_output=fp32_output, **kwargs)


def configure_module_rng(
    args: argparse.Namespace, pg_collection: ProcessGroupCollection, role_seed_offset: int
) -> None:
    """Seed the CUDA RNG tracker for one module role from its tp/pp coordinates plus the offset.

    The seed is shared across a module's DP/CP replicas but distinct across PP stages and roles,
    so disjoint modules (and stages) get independent RNG state. Caller invokes once per active
    module on this rank.
    """
    for _required in ("pp", "tp", "ep", "expt_tp"):
        assert (
            getattr(pg_collection, _required, None) is not None
        ), f"pg_collection passed to configure_module_rng must define {_required}"
    pp_rank = get_pg_rank(pg_collection.pp)
    tp_rank = get_pg_rank(pg_collection.tp)
    ep_rank = get_pg_rank(pg_collection.ep)
    expt_tp_rank = get_pg_rank(pg_collection.expt_tp)
    seed = args.seed + role_seed_offset + (100 * pp_rank)
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(
        seed, tp_rank=tp_rank, ep_rank=ep_rank, etp_rank=expt_tp_rank, force_reset_rng=True
    )


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


def _freeze_modality_submodule(submodule: torch.nn.Module, args: argparse.Namespace) -> None:
    """Freeze the encoder backbone (--freeze-vit) and/or projector (--freeze-projection)."""
    if getattr(args, "freeze_vit", False):
        submodule.encoders.requires_grad_(False)
    if getattr(args, "freeze_projection", False):
        submodule.input_projections.requires_grad_(False)
        submodule.output_projections.requires_grad_(False)


def _module_config(module: torch.nn.Module):
    """Return the module's own config, else the first descendant config (e.g. an encoder)."""
    config = getattr(module, "config", None)
    if config is not None:
        return config
    for child in module.modules():
        config = getattr(child, "config", None)
        if config is not None:
            return config
    raise ValueError("Cannot resolve a config for DDP wrapping from module")


def _maybe_float16_wrap(module: torch.nn.Module, config, is_encoder: bool) -> torch.nn.Module:
    """Wrap a submodule in Float16Module when fp16/bf16 is enabled; encoders keep bf16 outputs."""
    if not (getattr(config, "fp16", False) or getattr(config, "bf16", False)):
        return module
    cls = _EncoderFloat16Module if is_encoder else Float16Module
    return cls(config, module)


def wrap_active_modules_with_ddp(
    args: argparse.Namespace, mimo_model: MimoModel, topology: HeteroTopology
) -> None:
    """Freeze (per --freeze-* flags), Float16Module-wrap, and DDP-wrap each active module."""
    pad_buckets = getattr(args, "ddp_pad_buckets_for_high_nccl_busbw", False)
    grad_reduce_in_fp32 = getattr(args, "accumulate_allreduce_grads_in_fp32", True)

    ddp_stream = torch.cuda.Stream()
    ddp_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(ddp_stream):
        if mimo_model.language_model is not None:
            if getattr(args, "freeze_lm", False):
                mimo_model.language_model.requires_grad_(False)
            overlap = getattr(args, "overlap_grad_reduce", False)
            ddp_config = DistributedDataParallelConfig(
                overlap_grad_reduce=overlap,
                overlap_param_gather=getattr(args, "overlap_param_gather", False),
                num_buckets=getattr(args, "ddp_num_buckets", None),
                bucket_size=getattr(args, "ddp_bucket_size", None),
                pad_buckets_for_high_nccl_busbw=pad_buckets,
                use_distributed_optimizer=True,
                grad_reduce_in_fp32=grad_reduce_in_fp32,
            )
            # Resolve the absolute bucket size on the real config, as get_model does.
            ddp_config.bucket_size = resolve_ddp_bucket_size(
                ddp_config,
                topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY].dp_cp,
                overlap,
                sum(p.numel() for p in mimo_model.language_model.parameters()),
            )
            lm_config = _module_config(mimo_model.language_model)
            lm_module = _maybe_float16_wrap(mimo_model.language_model, lm_config, is_encoder=False)
            print_rank_0("wrapping language model in DDP")
            mimo_model.language_model = wrap_model_chunks_with_ddp(
                [lm_module],
                lm_config,
                ddp_config,
                DP=DistributedDataParallel,
                pg_collection=topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY],
            )[0]

        for name, submodule in mimo_model.modality_submodules.items():
            if submodule is None or name not in topology.module_pgs:
                continue
            _freeze_modality_submodule(submodule, args)
            ddp_config = DistributedDataParallelConfig(
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                num_buckets=getattr(args, "ddp_num_buckets", None),
                bucket_size=getattr(args, "ddp_bucket_size", None),
                pad_buckets_for_high_nccl_busbw=pad_buckets,
                use_distributed_optimizer=True,
                grad_reduce_in_fp32=grad_reduce_in_fp32,
            )
            # Encoders keep overlap off; resolve_ddp_bucket_size returns None there.
            ddp_config.bucket_size = resolve_ddp_bucket_size(
                ddp_config,
                topology.module_pgs[name].dp_cp,
                False,
                sum(p.numel() for p in submodule.parameters()),
            )
            enc_config = _module_config(submodule)
            enc_module = _maybe_float16_wrap(submodule, enc_config, is_encoder=True)
            print_rank_0(f"wrapping modality submodule {name!r} in DDP")
            mimo_model.modality_submodules[name] = wrap_model_chunks_with_ddp(
                [enc_module],
                enc_config,
                ddp_config,
                DP=DistributedDataParallel,
                pg_collection=topology.module_pgs[name],
            )[0]
    torch.cuda.current_stream().wait_stream(ddp_stream)


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
    """Build the encoder->language dependency map the communicator's topology arg needs."""
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
