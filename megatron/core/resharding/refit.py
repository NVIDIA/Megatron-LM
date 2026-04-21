# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

"""
High-level refit/reshard orchestration:
- prepare_swap_model_weights: build and cache the reshard plan without any transfer.
- swap_model_weights: public API; accepts a backend name or CopyService and delegates.
- reshard_model_weights: transport-agnostic core; builds/caches plan and executes.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, Union

import torch

from megatron.core import parallel_state
from megatron.core.inference.quantization.utils import (
    _should_quantize_param,
    quantize_params_to_mxfp8,
)
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.utils import unwrap_model

from . import build_centralized_reshard_plan, execute_reshard_plan
from .copy_services.base import CopyService
from .copy_services.gloo_copy_service import GlooCopyService
from .copy_services.nccl_copy_service import NCCLCopyService
from .copy_services.nvshmem_copy_service import NVSHMEMCopyService
from .transforms import MXFP8ReshardTransform, ReshardTransform

# Supported refit backend names
RefitBackendName = Literal["nccl", "gloo", "nvshmem"]


@dataclass(frozen=True)
class _PlanCacheKey:
    """
    Cache key for reshard plans.
    """

    rank: int
    # Parallelism configuration: (TP, PP, EP, DP, expt_tp) or None for non-collocated ranks
    src_config: Optional[Tuple[int, int, int, int, int]]
    dst_config: Optional[Tuple[int, int, int, int, int]]
    num_experts: Optional[int]


def _get_config_tuple(core) -> Optional[Tuple[int, int, int, int, int]]:
    """Extract (TP, PP, EP, DP, expt_tp) sizes from a model core.

    Returns:
        Tuple of (TP, PP, EP, DP, expt_tp) sizes, or None if core is None.
        - TP: Tensor parallelism
        - PP: Pipeline parallelism
        - EP: Expert parallelism
        - DP: Data parallelism
        - expt_tp: Expert tensor parallelism
    """
    if core is None:
        return None
    pg = core.pg_collection
    return (
        len(torch.distributed.get_process_group_ranks(pg.tp)) if pg.tp else 1,
        len(torch.distributed.get_process_group_ranks(pg.pp)) if pg.pp else 1,
        len(torch.distributed.get_process_group_ranks(pg.ep)) if pg.ep else 1,
        len(torch.distributed.get_process_group_ranks(pg.dp)) if pg.dp else 1,
        (
            len(torch.distributed.get_process_group_ranks(pg.expt_tp))
            if hasattr(pg, 'expt_tp') and pg.expt_tp
            else 1
        ),
    )


def _build_plan_cache_key(
    src_core, tgt_core, num_experts: Optional[int], group=None
) -> _PlanCacheKey:
    """Build cache key for reshard plan.

    Args:
        src_core: Source model core (or None for non-collocated destination/idle ranks)
        tgt_core: Target model core (or None for non-collocated source/idle ranks)
        num_experts: Number of MoE experts (or None for non-MoE models)
        group: Optional process group for rank query

    Returns:
        Cache key that uniquely identifies this reshard configuration for this rank
    """
    # Use group.rank() to support cross-cluster ProcessGroups
    rank = group.rank() if group is not None else torch.distributed.get_rank()
    src_config = _get_config_tuple(src_core)
    dst_config = _get_config_tuple(tgt_core)
    return _PlanCacheKey(
        rank=rank, src_config=src_config, dst_config=dst_config, num_experts=num_experts
    )


# Module-level cache for refit services to avoid repeated allocations
_service_cache: dict[str, CopyService] = {}
_plan_cache: dict[_PlanCacheKey, Any] = {}


def get_or_create_service(backend: RefitBackendName, group=None) -> CopyService:
    """Get or create a cached CopyService instance for the given backend.

    This avoids expensive repeated allocations (especially for NVSHMEM buffers)
    when swap_model_weights is called multiple times with the same backend.

    Args:
        backend: Backend name ("nccl", "gloo", or "nvshmem").
        group: Optional process group for NCCL backend.
    """
    if backend in _service_cache:
        return _service_cache[backend]

    if backend == "nccl":
        service = NCCLCopyService(group=group)
    elif backend == "gloo":
        service = GlooCopyService(group=group)
    elif backend == "nvshmem":
        service = NVSHMEMCopyService(group=group)
    else:
        raise ValueError(f"Unknown backend '{backend}'")

    _service_cache[backend] = service
    return service


def clear_service_cache():
    """Clear the cached refit services.

    Call this if you need to invalidate the cache, for example when
    reinitializing distributed state.

    This properly finalizes services to free GPU buffers
    before clearing the cache.
    """
    global _service_cache

    # Finalize services to free resources for NVSHMEM backend
    # NCCL/Gloo services have no cleanup needed
    for backend_name, service in _service_cache.items():
        if hasattr(service, '_remote') and hasattr(service._remote, 'finalize'):
            service._remote.finalize()

    _service_cache.clear()


def clear_plan_cache():
    """
    Clear the cached refit plans.
    """
    global _plan_cache
    _plan_cache.clear()


def clear_all_caches():
    """
    Clear both service and plan caches.
    """
    clear_service_cache()
    clear_plan_cache()


def _unwrap_model_cores(src_model, target_model):
    """Extract (src_core, tgt_core, num_experts) from model arguments.

    Handles list-wrapped modules and None (non-collocated) models.
    Fills in missing DP groups from Megatron's parallel state on the source.

    Returns:
        (src_core, tgt_core, num_experts)
    """
    src_core = None
    tgt_core = None
    num_experts = None

    if src_model is not None:
        src_lm = src_model[0] if isinstance(src_model, (list, tuple)) else src_model
        num_experts = src_lm.config.num_moe_experts
        src_core = unwrap_model(src_lm)
        if not hasattr(src_core, "pg_collection") or src_core.pg_collection is None:
            raise RuntimeError("Source model missing pg_collection required for reshard")
        # Fill missing DP group on the source using Megatron's parallel state if not provided
        if getattr(src_core.pg_collection, "dp", None) is None:
            src_core.pg_collection.dp = parallel_state.get_data_parallel_group()

    if target_model is not None:
        tgt_lm = target_model[0] if isinstance(target_model, (list, tuple)) else target_model
        if num_experts is None:
            num_experts = tgt_lm.config.num_moe_experts
        tgt_core = unwrap_model(tgt_lm)
        if not hasattr(tgt_core, "pg_collection") or tgt_core.pg_collection is None:
            raise RuntimeError("Target model missing pg_collection required for reshard")

    return src_core, tgt_core, num_experts


def _build_or_get_plan(src_core, tgt_core, num_experts, group, src_rank_offset, dst_rank_offset):
    """Return the cached reshard plan, building it (collectively) if not yet cached.

    All participating ranks must call this simultaneously when the plan is not
    yet cached, because build_centralized_reshard_plan uses collective communication.
    """
    global _plan_cache
    cache_key = _build_plan_cache_key(src_core, tgt_core, num_experts, group=group)
    if cache_key not in _plan_cache:
        _plan_cache[cache_key] = build_centralized_reshard_plan(
            src_core,
            tgt_core,
            num_experts=num_experts,
            group=group,
            src_rank_offset=src_rank_offset,
            dst_rank_offset=dst_rank_offset,
        )
    return _plan_cache[cache_key]


def _needs_mxfp8_conversion(model) -> bool:
    """Check if a model uses FlashInfer MXFP8 inference and needs weight conversion."""
    if model is None:
        return False
    lm = model[0] if isinstance(model, (list, tuple)) else model
    config = lm.config
    return (
        getattr(config, 'transformer_impl', None) == 'inference_optimized'
        and getattr(config, 'fp8_recipe', None) == 'mxfp8'
    )


def _setup_mxfp8_transform_on_plan(plan, target_model) -> None:
    """Detect MXFP8 needs and attach a transform to the plan if required.

    If the *target_model* uses an inference-optimized layer spec with MXFP8,
    this function:
      1. Computes which params are eligible for MXFP8 conversion.
      2. Quantizes the target model's decoder weights to FlashInfer MXFP8Tensor
         (creating persistent buffers whose addresses are later captured by
         CUDA graphs).
      3. Builds an ``MXFP8ReshardTransform`` and attaches it to the plan as
         ``plan.transform``.

    If the model doesn't need MXFP8, ``plan.transform`` is set to None.
    Subsequent calls are no-ops if the plan already has a transform attribute.
    """
    if hasattr(plan, 'transform'):
        return  # Already set up

    if not _needs_mxfp8_conversion(target_model):
        plan.transform = None
        return

    lm = target_model[0] if isinstance(target_model, (list, tuple)) else target_model
    core = unwrap_model(lm)
    decoder = core.decoder if hasattr(core, 'decoder') else core

    # 1. Compute which parameters are eligible for MXFP8 conversion.
    #    Must be done while params are still visible as nn.Parameter (BF16).
    convertible: set[str] = set()
    for name, param in decoder.named_parameters():
        if _should_quantize_param(param):
            convertible.add(f"decoder.{name}")

    # 2. Quantize decoder weights → persistent MXFP8Tensor buffers.
    persistent_buffers = quantize_params_to_mxfp8(decoder)

    # 3. Build the transform and attach it to the plan.
    plan.transform = MXFP8ReshardTransform(
        convertible_params=convertible,
        persistent_buffers=persistent_buffers,
        buffer_key_prefix="decoder.",
    )


def prepare_swap_model_weights(
    src_model: LanguageModule,
    target_model: LanguageModule,
    group=None,
    src_rank_offset: int = 0,
    dst_rank_offset: int = 0,
):
    """Pre-build and cache the reshard plan and any format-conversion transforms.

    Call this during initialization while models are in their native (BF16) format,
    before any weight format conversion (e.g., MXFP8).  The plan is stored in the
    same module-level cache as swap_model_weights, so subsequent calls reuse it
    without needing to inspect named_parameters() again.

    If the *target_model* uses an inference-optimized layer spec with MXFP8
    (``config.transformer_impl == 'inference_optimized'`` and
    ``config.fp8_recipe == 'mxfp8'``), this function also:
      - computes which parameters are eligible for MXFP8 conversion,
      - quantizes the target decoder weights to persistent FlashInfer
        MXFP8Tensor buffers (whose addresses are later baked into CUDA graphs),
      - creates an ``MXFP8ReshardTransform`` that subsequent
        ``swap_model_weights`` calls use automatically.

    Callers do **not** need to know about MXFP8 — the transform is created and
    cached transparently.

    All participating ranks must call this simultaneously — the plan builder uses
    collective communication internally.

    Args:
        src_model: Source model, or None if this rank only receives weights.
        target_model: Target model, or None if this rank only sends weights.
        group: Optional process group for collective communication.
        src_rank_offset: Rank offset for source (training) workers.
        dst_rank_offset: Rank offset for destination (inference) workers.
    """
    src_core, tgt_core, num_experts = _unwrap_model_cores(src_model, target_model)
    plan = _build_or_get_plan(
        src_core, tgt_core, num_experts, group, src_rank_offset, dst_rank_offset
    )

    # Auto-detect and set up MXFP8 transform on the plan for the target model.
    # This must happen after the plan is built (while BF16 params are still visible)
    # and before any swap_model_weights call.
    _setup_mxfp8_transform_on_plan(plan, target_model)


def swap_model_weights(
    src_model: LanguageModule,
    target_model: LanguageModule,
    refit_method: Union[RefitBackendName, CopyService],
    group=None,
    src_rank_offset: int = 0,
    dst_rank_offset: int = 0,
    transform: Optional[ReshardTransform] = None,
):
    """
    Orchestrate weight swap/refit.

    If *transform* is not explicitly provided, the function automatically uses
    any ``MXFP8ReshardTransform`` that was created and cached by a prior
    ``prepare_swap_model_weights`` call for the same model pair.  This makes
    MXFP8 handling transparent to callers.

    Args:
        refit_method: a string backend name (one of the supported refit
            backends) or a CopyService instance.
        group: Optional process group for communication.
        src_rank_offset / dst_rank_offset: Offsets applied to local process
            group ranks so that metadata contains globally unique rank IDs
            across independent torch.distributed worlds.
        transform: Optional ReshardTransform for custom format conversion.
            If None, the cached transform (from prepare_swap_model_weights)
            is used automatically when the receiver needs MXFP8 conversion.
    """
    if isinstance(refit_method, str):
        service = get_or_create_service(refit_method, group=group)
    elif hasattr(refit_method, 'submit_send') and hasattr(refit_method, 'run'):
        service = refit_method
    else:
        raise TypeError(
            "refit_method must be a str backend name or a CopyService-compatible instance"
        )

    # Auto-resolve MXFP8 transform from the cached plan when no
    # explicit transform was provided.
    if transform is None:
        src_core, tgt_core, num_experts = _unwrap_model_cores(src_model, target_model)
        plan = _build_or_get_plan(
            src_core, tgt_core, num_experts, group, src_rank_offset, dst_rank_offset
        )
        transform = getattr(plan, 'transform', None)

    reshard_model_weights(
        src_model,
        target_model,
        service=service,
        group=group,
        src_rank_offset=src_rank_offset,
        dst_rank_offset=dst_rank_offset,
        transform=transform,
    )


def reshard_model_weights(
    src_model: LanguageModule,
    target_model: LanguageModule,
    service: CopyService,
    group=None,
    src_rank_offset: int = 0,
    dst_rank_offset: int = 0,
    transform: Optional[ReshardTransform] = None,
):
    """Reshard and copy model weights from ``src_model`` to ``target_model`` using ``service``.

    Supports None for src_model and/or target_model to enable non-collocated mode:
    - (src_model, target_model): Both models present (collocated mode)
    - (src_model, None): Source rank - only sends data (non-collocated)
    - (None, target_model): Destination rank - only receives data (non-collocated)
    - (None, None): Idle rank - participates in collectives but has no transfers (non-collocated)

    Args:
        group: Optional process group for collective communication.
        src_rank_offset / dst_rank_offset: Offsets for mapping local ranks to global ranks
            in independent torch.distributed worlds.
        transform: Optional ReshardTransform for custom format conversion.
    """
    src_core, tgt_core, num_experts = _unwrap_model_cores(src_model, target_model)
    plan = _build_or_get_plan(
        src_core, tgt_core, num_experts, group, src_rank_offset, dst_rank_offset
    )
    execute_reshard_plan(
        plan, src_core, tgt_core, service=service, group=group, transform=transform
    )
