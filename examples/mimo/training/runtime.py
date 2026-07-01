# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-rank runtime setup (RNG seeding, freezing, DDP wrapping) for hetero MIMO training."""

from __future__ import annotations

import argparse

import torch

from examples.mimo.training.topology import HeteroTopology
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module
from megatron.training.initialize import _set_random_seed
from megatron.training.models.dist_utils import (
    prepare_existing_model_chunks_for_distributed_training,
)
from megatron.training.utils import print_rank_0


class _EncoderFloat16Module(Float16Module):
    """Float16Module that keeps encoder outputs in model precision for the bridge."""

    def forward(self, *inputs, fp32_output=False, **kwargs):  # noqa: D102
        return super().forward(*inputs, fp32_output=fp32_output, **kwargs)


def configure_module_rng(
    args: argparse.Namespace,
    pg_collection: ProcessGroupCollection,
    role_seed_offset: int,
    data_parallel_random_init: bool = False,
) -> None:
    """Seed one active module role through the stock explicit-process-group path.

    The seed is shared across a module's DP/CP replicas but distinct across PP stages and roles,
    so disjoint modules (and stages) get independent RNG state. Caller invokes once per active
    module on this rank.
    """
    for _required in ("pp", "dp", "tp", "ep", "expt_tp"):
        assert (
            getattr(pg_collection, _required, None) is not None
        ), f"pg_collection passed to configure_module_rng must define {_required}"
    _set_random_seed(
        args.seed + role_seed_offset,
        data_parallel_random_init,
        pp_group=pg_collection.pp,
        dp_group=pg_collection.dp,
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


def _ddp_config_from_args(
    args: argparse.Namespace, enable_overlap: bool
) -> DistributedDataParallelConfig:
    """Build a DDP config from CLI args; when ``enable_overlap`` is False both overlaps are off."""
    return DistributedDataParallelConfig(
        overlap_grad_reduce=enable_overlap and getattr(args, "overlap_grad_reduce", False),
        overlap_param_gather=enable_overlap and getattr(args, "overlap_param_gather", False),
        num_buckets=getattr(args, "ddp_num_buckets", None),
        bucket_size=getattr(args, "ddp_bucket_size", None),
        pad_buckets_for_high_nccl_busbw=getattr(args, "ddp_pad_buckets_for_high_nccl_busbw", False),
        use_distributed_optimizer=True,
        grad_reduce_in_fp32=getattr(args, "accumulate_allreduce_grads_in_fp32", True),
    )


def wrap_active_modules_with_ddp(
    args: argparse.Namespace,
    mimo_model: MimoModel,
    topology: HeteroTopology,
    data_parallel_random_init: bool = False,
) -> None:
    """Freeze (per --freeze-* flags), Float16Module-wrap, and DDP-wrap each active module."""
    if mimo_model.language_model is not None:
        if getattr(args, "freeze_lm", False):
            mimo_model.language_model.requires_grad_(False)
        lm_config = _module_config(mimo_model.language_model)
        print_rank_0("wrapping language model in DDP")
        mimo_model.language_model = prepare_existing_model_chunks_for_distributed_training(
            [mimo_model.language_model],
            lm_config,
            topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY],
            built_with_meta_device=lm_config.init_model_with_meta_device,
            ddp_config=_ddp_config_from_args(args, enable_overlap=True),
            data_parallel_random_init=data_parallel_random_init,
            mixed_precision_wrapper=Float16Module,
        )[0]

    for name, submodule in mimo_model.modality_submodules.items():
        if submodule is None or name not in topology.module_pgs:
            continue
        _freeze_modality_submodule(submodule, args)
        enc_config = _module_config(submodule)
        print_rank_0(f"wrapping modality submodule {name!r} in DDP")
        mimo_model.modality_submodules[name] = (
            prepare_existing_model_chunks_for_distributed_training(
                [submodule],
                enc_config,
                topology.module_pgs[name],
                built_with_meta_device=enc_config.init_model_with_meta_device,
                ddp_config=_ddp_config_from_args(args, enable_overlap=False),
                data_parallel_random_init=data_parallel_random_init,
                mixed_precision_wrapper=_EncoderFloat16Module,
            )[0]
        )
