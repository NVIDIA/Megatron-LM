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
from .utils import invalidate_refit_tensor_cache, named_persistent_buffers

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
    # Rank offsets distinguish non-collocated configurations that would otherwise
    # share the same (rank, sizes, num_experts) tuple but route to different
    # global ranks.
    src_rank_offset: int = 0
    dst_rank_offset: int = 0


def _get_config_tuple(core) -> Optional[Tuple[int, int, int, int, int]]:
    """Extract (TP, PP, EP, DP, expt_tp) sizes from a model core, memoized on the core.

    Process-group sizes don't change after init, so the result is cached on the
    core object itself to avoid repeated ``get_process_group_ranks`` calls on
    the hot path (each refit looks the key up 2-3x).
    """
    if core is None:
        return None
    cached = getattr(core, '_refit_config_tuple', None)
    if cached is not None:
        return cached
    pg = core.pg_collection
    expt_tp = getattr(pg, 'expt_tp', None)
    result = (
        pg.tp.size() if pg.tp else 1,
        pg.pp.size() if pg.pp else 1,
        pg.ep.size() if pg.ep else 1,
        pg.dp.size() if pg.dp else 1,
        expt_tp.size() if expt_tp else 1,
    )
    core._refit_config_tuple = result
    return result


def _build_plan_cache_key(
    src_core,
    tgt_core,
    num_experts: Optional[int],
    group=None,
    src_rank_offset: int = 0,
    dst_rank_offset: int = 0,
) -> _PlanCacheKey:
    """Build cache key for reshard plan."""
    # group.rank() supports cross-cluster ProcessGroups.
    rank = group.rank() if group is not None else torch.distributed.get_rank()
    return _PlanCacheKey(
        rank=rank,
        src_config=_get_config_tuple(src_core),
        dst_config=_get_config_tuple(tgt_core),
        num_experts=num_experts,
        src_rank_offset=src_rank_offset,
        dst_rank_offset=dst_rank_offset,
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
    reinitializing distributed state.  Services are ``close()``-d first so
    backends owning GPU buffers (NVSHMEM) release them cleanly.
    """
    global _service_cache
    for service in _service_cache.values():
        service.close()
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
    cache_key = _build_plan_cache_key(
        src_core,
        tgt_core,
        num_experts,
        group=group,
        src_rank_offset=src_rank_offset,
        dst_rank_offset=dst_rank_offset,
    )
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
      3. Builds an ``MXFP8ReshardTransform`` and attaches it to ``plan.transform``.

    Idempotent: skips re-setup if ``plan.transform`` is already populated.
    """
    if plan.transform is not None:
        return

    if not _needs_mxfp8_conversion(target_model):
        return

    lm = target_model[0] if isinstance(target_model, (list, tuple)) else target_model
    core = unwrap_model(lm)
    decoder = core.decoder if hasattr(core, 'decoder') else core

    # Eligible params must be computed while still visible as nn.Parameter (BF16).
    convertible: set[str] = set()
    for name, param in decoder.named_parameters():
        if _should_quantize_param(param):
            convertible.add(f"decoder.{name}")

    persistent_buffers = quantize_params_to_mxfp8(decoder)

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
    elif isinstance(refit_method, CopyService):
        service = refit_method
    else:
        raise TypeError("refit_method must be a str backend name or a CopyService instance")

    # Auto-resolve MXFP8 transform from the cached plan when no
    # explicit transform was provided.
    if transform is None:
        src_core, tgt_core, num_experts = _unwrap_model_cores(src_model, target_model)
        plan = _build_or_get_plan(
            src_core, tgt_core, num_experts, group, src_rank_offset, dst_rank_offset
        )
        transform = plan.transform

    reshard_model_weights(
        src_model,
        target_model,
        service=service,
        group=group,
        src_rank_offset=src_rank_offset,
        dst_rank_offset=dst_rank_offset,
        transform=transform,
    )


def _harmonize_buffer_dtypes(plan, src_core, tgt_core, group=None):
    """Bring destination persistent-buffer dtypes into agreement with source.

    Some buffers (notably the MoE router ``expert_bias``) are upcast to fp32
    inside the trainer on first forward by ``_maintain_float32_expert_bias``,
    while the freshly-built inference model still holds them in bf16 from the
    ``Float16Module`` wrap.  The reshard send/recv path is dtype-strict —
    sending fp32 bytes into a bf16 receive buffer corrupts the data — so dst's
    buffer must match src's dtype before the transfer.

    The canonical dtype map is collected once via ``all_gather_object`` and
    cached on the plan.  Subsequent refits reuse the cached map and only do
    the per-buffer dtype check / replacement (no collective).
    """
    if plan.buffer_dtypes is None:
        local_src_dtypes: dict[str, torch.dtype] = {}
        if src_core is not None:
            for full_name, _sub, _buf_name, buf in named_persistent_buffers(src_core):
                local_src_dtypes[full_name] = buf.dtype

        world_size = group.size() if group is not None else torch.distributed.get_world_size()
        gathered: list = [None] * world_size
        torch.distributed.all_gather_object(gathered, local_src_dtypes, group=group)

        canonical: dict[str, torch.dtype] = {}
        for d in gathered:
            if not d:
                continue
            for name, dtype in d.items():
                # Replicated buffers agree across ranks; first writer wins.
                canonical.setdefault(name, dtype)
        plan.buffer_dtypes = canonical

    if tgt_core is None:
        return

    canonical = plan.buffer_dtypes
    invalidated = False
    for full_name, sub, buf_name, dst_buf in named_persistent_buffers(tgt_core):
        expected = canonical.get(full_name)
        if expected is not None and dst_buf.dtype != expected:
            # Replace the tensor in-place on the parent module so subsequent
            # recvs write the right number of bytes and the in-model lookup
            # (``self.expert_bias``) sees the new storage.
            sub._buffers[buf_name] = dst_buf.to(expected)
            invalidated = True
    if invalidated:
        invalidate_refit_tensor_cache(tgt_core)


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
    _harmonize_buffer_dtypes(plan, src_core, tgt_core, group=group)
    execute_reshard_plan(
        plan, src_core, tgt_core, service=service, group=group, transform=transform
    )
