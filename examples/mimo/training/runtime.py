# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-rank runtime setup (RNG seeding, freezing, DDP wrapping) for hetero MIMO training."""

from __future__ import annotations

import argparse
from typing import Optional

import torch

from examples.mimo.training.topology import HeteroTopology
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_pg_rank, get_pg_size
from megatron.training.training import resolve_ddp_bucket_size, wrap_model_chunks_with_ddp
from megatron.training.utils import print_rank_0


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


def set_module_requires_grad(module: Optional[torch.nn.Module], requires_grad: bool) -> None:
    """Set requires_grad for every parameter in a module when the module exists."""
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad


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
                set_module_requires_grad(mimo_model.language_model, False)
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
            if getattr(args, "freeze_encoders", False):
                set_module_requires_grad(submodule, False)
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
