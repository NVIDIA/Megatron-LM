# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dual gradient finalization for MIMO training on the stock Megatron loop."""

from __future__ import annotations

import torch
import torch.distributed as dist

from examples.mimo.training.topology import HeteroTopology
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.pipeline_parallel.utils import is_pp_last_stage

# Sentinel set per modality submodule when this rank had that modality's input this step.
_PARTICIPATED_ATTR = "_mimo_rank_processed_input"


def _has_modality_input(value) -> bool:
    """Whether this rank received this modality's input this step.

    The batch omits a modality's key when absent, so ``value`` is None (not present) or a
    non-empty nested dict (present); an empty tensor also counts as absent.
    """
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    return bool(value)


def mark_modality_participation(mimo_model: MimoModel, batch) -> None:
    """Tag each modality submodule with whether this rank had that modality's input this step.

    Reads ``batch["modality_inputs"]`` (keyed by modality name) so the flag is per modality
    rather than vision-specific.
    """
    modality_inputs = batch.get("modality_inputs", {}) if isinstance(batch, dict) else {}
    for name, submodule in mimo_model.modality_submodules.items():
        if submodule is not None:
            setattr(submodule, _PARTICIPATED_ATTR, _has_modality_input(modality_inputs.get(name)))


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


def _is_token_source_rank(language_pg) -> bool:
    """Whether this rank is the single LLM coordinate that owns the global token count.

    Sourcing the count from one coordinate (last PP stage, TP rank 0) avoids
    double-counting it across TP/PP replicas. In non-colocated, language_pg is the LLM
    collection seen on every rank, so encoder-grid ranks reach here with non-member
    (None) pp/tp groups — the _is_pg_member guards short-circuit so they are never the
    source.
    """
    if language_pg is None:
        return False
    pp = getattr(language_pg, "pp", None)
    tp = getattr(language_pg, "tp", None)
    return (
        _is_pg_member(pp)
        and _is_pg_member(tp)
        and is_pp_last_stage(pp)
        and dist.get_rank(group=tp) == 0
    )


def _global_token_count(num_tokens, language_pg) -> float:
    """Total non-padded tokens in the global batch, visible on every rank.

    Only the LLM token-source ranks contribute: they sum over the LLM DP/CP group;
    a world MAX then publishes that N_global to every rank, including the
    non-colocated encoder grid (where ``language_pg`` is None and the count is 0).
    """
    global_num_tokens = torch.zeros(1, dtype=torch.float32, device="cuda")
    if _is_token_source_rank(language_pg):
        token_count = num_tokens.to(dtype=torch.float32).sum().view(1)
        dist.all_reduce(token_count, group=language_pg.dp_cp, op=dist.ReduceOp.SUM)
        if dist.get_rank(group=language_pg.dp_cp) == 0:
            global_num_tokens.copy_(token_count)
    dist.all_reduce(global_num_tokens, op=dist.ReduceOp.MAX)
    return float(global_num_tokens.item())


def configure_grad_sync(args, mimo_model: MimoModel, topology: HeteroTopology) -> None:
    """Configure per-module gradient finalization: each module finalizes over its own groups.

    The encoder and LLM have decoupled parallelism (separate grids), so each reduces its
    gradients over its own process-group collection; both then divide by one shared
    per-token mean (N_global).

    MimoModel structure (each a separately DDP-wrapped module on its own grid)::

        MimoModel
        ├─ language_model          (LLM)      -> own process groups
        └─ modality_submodules[*]  (encoders) -> own process groups
    """
    module_pgs = topology.module_pgs
    language_pg = module_pgs.get(MIMO_LANGUAGE_MODULE_KEY)
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

        # N_global is the global token count, published to every rank (including the
        # non-colocated encoder grid) so both modules divide by the same per-token mean.
        n_global = _global_token_count(num_tokens, language_pg)
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
            if correct_vision_grad and vision_pg is not None and vision_pg.dp is not None:
                vision_dp_group = vision_pg.dp
                if _is_pg_member(vision_dp_group):
                    vision_dp_size = dist.get_world_size(vision_dp_group)
                    if vision_dp_size > 1:
                        participation = _vision_participation_count(submodule, vision_dp_group)
                        if 0.0 < participation < vision_dp_size:
                            vision_scale *= vision_dp_size / participation

            if vision_scale != 0.0:
                submodule.scale_gradients(vision_scale)

    mimo_model.config.finalize_model_grads_func = finalize_grads_func
    # The schedule always calls grad_scale_func with a Tensor loss; the per-token
    # mean is applied in finalize_grads_func, so no extra scaling is needed here.
    mimo_model.config.grad_scale_func = lambda loss: loss
