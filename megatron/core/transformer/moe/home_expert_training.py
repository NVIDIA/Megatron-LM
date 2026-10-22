# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training-loop integration for the explicit MoEScheduler.step boundary."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from megatron.core.transformer.moe.home_expert_state import optimizer_leaves
from megatron.core.transformer.moe.moe_scheduler import MoEScheduler

if TYPE_CHECKING:
    from megatron.core.optimizer.optimizer import MegatronOptimizer


def validate_home_expert_checkpoint_args(
    args: Namespace, checkpoint_args: Namespace | None = None
) -> None:
    """Reject formats or topology changes that lose physical expert ownership."""
    active = getattr(args, "moe_scheduler_home_update_interval", 0)
    saved = getattr(checkpoint_args, "moe_scheduler_home_update_interval", 0)
    if not active and not saved:
        return
    if args.ckpt_format not in ("torch_dist", "fsdp_dtensor"):
        raise ValueError("Home expert checkpoints require torch_dist or fsdp_dtensor format.")
    if checkpoint_args is not None:
        topology = (
            "num_experts",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "world_size",
        )
        for name in topology:
            if getattr(args, name, None) != getattr(checkpoint_args, name, None):
                raise ValueError(f"Home expert checkpoint requires unchanged {name}.")
        if active and not saved and not getattr(checkpoint_args, "moe_enable_scheduler", False):
            return  # Ordinary logical checkpoint: loader starts a new training run.
        for name in (
            "moe_enable_scheduler",
            "moe_scheduler_planner_type",
            "moe_scheduler_home_update_interval",
            "moe_scheduler_num_idle_experts",
        ):
            if getattr(args, name, None) != getattr(checkpoint_args, name, None):
                raise ValueError(
                    f"Home expert checkpoint requires unchanged {name}; export a logical "
                    "checkpoint before disabling the scheduler."
                )


def home_expert_schedulers(model: Sequence[torch.nn.Module]) -> list[MoEScheduler]:
    """Find schedulers with physical home ownership across model chunks."""
    return [
        module
        for chunk in model
        for module in chunk.modules()
        if isinstance(module, MoEScheduler) and module.home_expert_dispatch is not None
    ]


def initialize_home_expert_optimizers(
    model: Sequence[torch.nn.Module], optimizer: MegatronOptimizer | torch.optim.Optimizer
) -> None:
    """Resolve authoritative optimizer bindings during initialization, before training."""
    for scheduler in home_expert_schedulers(model):
        if optimizer is None:
            raise ValueError("Home migration requires an optimizer.")
        scheduler.bind_optimizer(optimizer)


def prepare_home_expert_checkpoint_load(
    args: Namespace, model: Sequence[torch.nn.Module], checkpoint_args: Namespace
) -> bool:
    """Select strict resume or identity-owned initialization before building load templates."""
    active = getattr(args, "moe_scheduler_home_update_interval", 0)
    if active and checkpoint_args is None:
        raise ValueError(
            "Home checkpoint loading requires source args to verify ownership and topology."
        )
    validate_home_expert_checkpoint_args(args, checkpoint_args)
    if getattr(args, "moe_scheduler_export_logical_checkpoint", False) and getattr(
        checkpoint_args, "no_save_optim", False
    ):
        raise ValueError("Logical checkpoint export requires saved optimizer masters and state.")
    fresh = bool(
        active
        and not getattr(checkpoint_args, "moe_scheduler_home_update_interval", 0)
        and not getattr(checkpoint_args, "moe_enable_scheduler", False)
    )
    if fresh:
        if getattr(args, "moe_scheduler_export_logical_checkpoint", False):
            raise ValueError("Logical export requires a home-migration checkpoint.")
        args.finetune = True
        args.consumed_train_samples = 0
        args.consumed_valid_samples = 0
        args.skipped_train_samples = 0
        for scheduler in home_expert_schedulers(model):
            scheduler.begin_logical_checkpoint_load()
    return fresh


def finish_home_expert_checkpoint_load(model: Sequence[torch.nn.Module]) -> None:
    """End logical initialization after model tensors have finished loading."""
    for scheduler in home_expert_schedulers(model):
        scheduler.finish_logical_checkpoint_load()


@torch.no_grad()
def canonicalize_home_experts_for_export(
    model: Sequence[torch.nn.Module], optimizer: MegatronOptimizer | torch.optim.Optimizer
) -> None:
    """Restore logical expert order, including optimizer state, for conversion-only export.

    This is a terminal conversion operation. The caller saves with the scheduler
    disabled and exits; runtime token dispatchers are not rebuilt for training.
    Deferred planner proposals are deliberately not executed: only committed
    home ownership determines where the loaded checkpoint's tensors reside.
    """
    schedulers = home_expert_schedulers(model)
    if not schedulers or optimizer is None:
        raise ValueError("Logical export requires initialized home schedulers and an optimizer.")
    torch.cuda.synchronize()
    for scheduler in schedulers:
        scheduler.canonicalize_home_for_export()

    # Finish every layer's migration before rebuilding any shared gather buffer.
    leaves = optimizer_leaves(optimizer)
    chunks = {id(chunk): chunk for leaf in leaves for chunk in getattr(leaf, "model_chunks", ())}
    for chunk in chunks.values():
        chunk.zero_grad_buffer()
    for leaf in leaves:
        if not isinstance(leaf, torch.optim.Optimizer):
            leaf.refresh_model_weights(force=True)
    for chunk in chunks.values():
        chunk.start_param_sync(force_sync=True)
    torch.cuda.synchronize()
    for chunk in model:
        for module in list(chunk.modules()):
            scheduler = getattr(module, "moe_scheduler", None)
            if scheduler in schedulers:
                module.moe_scheduler = None
            config = getattr(module, "config", None)
            if getattr(config, "moe_enable_scheduler", False):
                config.moe_enable_scheduler = False
                config.moe_scheduler_home_update_interval = 0
                config.moe_scheduler_num_idle_experts = 0


@torch.no_grad()
def restore_home_expert_checkpoint_weights(optimizer: MegatronOptimizer) -> None:
    """Rebuild both MXFP8 orientations from restored optimizer master weights.

    A dequantized rowwise checkpoint cannot reconstruct the independent
    columnwise quantization used by backward. Stage all chained optimizer
    shards before gathering a shared DDP chunk, just as after a home exchange.
    FSDP restores its compute weights from master-weight checkpoint tensors.
    """
    leaves = [
        leaf
        for leaf in optimizer_leaves(optimizer)
        if getattr(getattr(leaf, "config", None), "reuse_grad_buf_for_mxfp8_param_ag", False)
        and not getattr(getattr(leaf, "ddp_config", None), "use_megatron_fsdp", False)
        and not getattr(leaf, "is_stub_optimizer", False)
    ]
    chunks = {id(chunk): chunk for leaf in leaves for chunk in leaf.model_chunks}
    for chunk in chunks.values():
        chunk.zero_grad_buffer()
    for leaf in leaves:
        leaf.refresh_model_weights(force=True)
    for chunk in chunks.values():
        chunk.start_param_sync(force_sync=True)


def prepare_home_expert_step(
    model: Sequence[torch.nn.Module], optimizer: MegatronOptimizer
) -> list[MoEScheduler]:
    """Drain old work and defer parameter gathers if any layer has a pending plan.

    Returns the schedulers that need a successful-step notification. This is
    intentionally training-loop glue, not a combined expert dispatch component.
    """
    schedulers = home_expert_schedulers(model)
    if any(s.has_pending_home_exchange for s in schedulers):
        for scheduler in schedulers:
            scheduler.prepare_home_exchange()
        # First version uses a device fence at rare migration boundaries. It
        # includes FSDP/replica reduction streams and previous gather consumers.
        torch.cuda.synchronize()
        for leaf in optimizer_leaves(optimizer):
            leaf.defer_param_sync_for_state_update(True)
    return schedulers


@torch.no_grad()
def finish_home_expert_step(
    schedulers: Sequence[MoEScheduler], optimizer: MegatronOptimizer, update_successful: bool
) -> None:
    """Migrate updated state and commit mappings before restarting gathers."""
    leaves = optimizer_leaves(optimizer)
    deferred = any(leaf.param_sync_deferred_for_state_update for leaf in leaves)
    if update_successful:
        for scheduler in schedulers:
            scheduler.step()
        if deferred:
            # All layers migrate before copying/casting model weights. Existing
            # DDP/FSDP/MXFP8 code owns every destination wrapper and scale layout.
            for leaf in leaves:
                leaf.refresh_model_weights()
    for leaf in leaves:
        leaf.defer_param_sync_for_state_update(False)
    if deferred and update_successful:
        chunks = {}
        for leaf in leaves:
            ddp_config = getattr(leaf, "ddp_config", None)
            if ddp_config is not None and (
                ddp_config.use_megatron_fsdp or not ddp_config.overlap_param_gather
            ):
                for chunk in leaf.model_chunks:
                    chunks[id(chunk)] = chunk
        for chunk in chunks.values():
            chunk.start_param_sync()
        if getattr(
            getattr(optimizer, "config", None), "overlap_param_gather_with_optimizer_step", False
        ):
            first_chunk = leaves[0].model_chunks[0]
            if id(first_chunk) not in chunks:
                first_chunk.start_param_sync(force_dispatch=True)
