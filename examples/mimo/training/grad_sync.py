# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dual gradient finalization for MIMO training on the stock Megatron loop."""

from __future__ import annotations

import torch
import torch.distributed as dist

from examples.mimo.training.topology import HeteroTopology
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY, ModuleLayout
from megatron.core.models.mimo.model.base import MimoModel

# Sentinel set on a modality submodule by forward_step when this rank had image input.
_PARTICIPATED_ATTR = "_mimo_rank_processed_input"


def mark_modality_participation(mimo_model: MimoModel, batch) -> None:
    """Tag each modality submodule with whether this rank had image input this step."""
    images = batch.get("images") if isinstance(batch, dict) else None
    if isinstance(images, torch.Tensor):
        had_input = images.numel() > 0
    elif isinstance(images, (list, tuple)):
        had_input = len(images) > 0
    else:
        had_input = False
    for submodule in mimo_model.modality_submodules.values():
        if submodule is not None:
            setattr(submodule, _PARTICIPATED_ATTR, had_input)


def reset_modality_participation(mimo_model: MimoModel) -> None:
    """Clear per-step participation flags at the top of each train step."""
    for submodule in mimo_model.modality_submodules.values():
        if submodule is not None:
            setattr(submodule, _PARTICIPATED_ATTR, False)


def _vision_participation_count(submodule, vision_dp_group) -> float:
    """Number of vision-DP ranks that processed image input this step."""
    val = 1.0 if getattr(submodule, _PARTICIPATED_ATTR, False) else 0.0
    indicator = torch.tensor([val], dtype=torch.float32, device="cuda")
    dist.all_reduce(indicator, op=dist.ReduceOp.SUM, group=vision_dp_group)
    return float(indicator.item())


def _is_pg_member(pg) -> bool:
    """Whether the current rank belongs to ``pg`` (defensive; -1 for non-members)."""
    return pg is not None and dist.get_rank(group=pg) >= 0


def configure_grad_sync(args, mimo_model: MimoModel, topology: HeteroTopology) -> None:
    """Install a finalize_model_grads_func that finalizes the LLM and each modality
    submodule over its own per-module process-group collection.
    """
    module_pgs = topology.module_pgs
    language_pg = module_pgs.get(MIMO_LANGUAGE_MODULE_KEY)
    non_colocated = mimo_model.role.mode is ModuleLayout.NON_COLOCATED
    correct_vision_grad = bool(
        getattr(args, "correct_encoder_grad_for_partial_participation", False)
    )

    def finalize_grads_func(_model_list, num_tokens, force_all_reduce=False, **_kwargs):
        # calculate_per_token_loss=True => DDP gradient_scaling_factor 1.0 (pure SUM),
        # so the per-token mean is applied here by dividing every shard by N_global.
        assert num_tokens is not None, (
            "MIMO grad sync expects calculate_per_token_loss=True so the schedule "
            "forwards total_num_tokens; got None."
        )

        # Lift the schedule's per-rank token sum to N_global over the LLM DP/CP group.
        n_global = 0.0
        if language_pg is not None:
            llm_dp_pg = language_pg.dp_cp if language_pg.dp_cp is not None else language_pg.dp
            dist.all_reduce(num_tokens, group=llm_dp_pg, op=dist.ReduceOp.SUM)
            n_global = float(num_tokens.item())
        inv = 1.0 / n_global if n_global > 0 else 0.0

        if mimo_model.language_model is not None:
            finalize_model_grads(
                [mimo_model.language_model],
                num_tokens=None,
                pg_collection=language_pg,
                force_all_reduce=force_all_reduce,
            )
            if inv != 0.0:
                mimo_model.language_model.scale_gradients(inv)

        for name, submodule in mimo_model.modality_submodules.items():
            if submodule is None:
                continue
            vision_pg = module_pgs.get(name)
            finalize_model_grads(
                [submodule],
                num_tokens=None,
                pg_collection=vision_pg,
                force_all_reduce=force_all_reduce,
            )

            vision_scale = inv
            # Partial-participation correction only applies when vision ranks are
            # disjoint from the LLM ranks (non-colocated); colocated ranks always
            # share the LLM's per-microbatch participation.
            if (
                non_colocated
                and correct_vision_grad
                and vision_pg is not None
                and vision_pg.dp is not None
            ):
                vision_dp_group = vision_pg.dp
                if _is_pg_member(vision_dp_group):
                    vision_dp_size = dist.get_world_size(vision_dp_group)
                    if vision_dp_size > 1:
                        participation = _vision_participation_count(submodule, vision_dp_group)
                        if 0.0 < participation < vision_dp_size:
                            vision_scale *= vision_dp_size / participation

            if vision_scale != 0.0:
                submodule.scale_gradients(vision_scale)

        # End-of-iter barrier across vision-DP ranks guards a cross-grid collective
        # deadlock observed only in the non-colocated (separate-grid) layout.
        if non_colocated:
            for name in mimo_model.modality_submodules:
                vision_pg = module_pgs.get(name)
                vision_dp_group = (
                    vision_pg.dp if vision_pg is not None and vision_pg.dp is not None else None
                )
                if _is_pg_member(vision_dp_group):
                    dist.barrier(group=vision_dp_group)

    mimo_model.config.finalize_model_grads_func = finalize_grads_func
    # The schedule always calls grad_scale_func with a Tensor loss; the per-token
    # mean is applied in finalize_grads_func, so no extra scaling is needed here.
    mimo_model.config.grad_scale_func = lambda loss: loss
