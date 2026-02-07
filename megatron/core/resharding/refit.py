# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

"""
High-level refit/reshard orchestration:
- swap_model_weights: public API; accepts a backend name or CopyService and delegates.
- reshard_model_weights: transport-agnostic core; builds/caches plan and executes.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, Union

import torch

from megatron.core import parallel_state
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.utils import unwrap_model

from . import build_centralized_reshard_plan, execute_reshard_plan
from .copy_services.base import CopyService
from .copy_services.gloo_copy_service import GlooCopyService
from .copy_services.nccl_copy_service import NCCLCopyService
from .copy_services.nvshmem_copy_service import NVSHMEMCopyService

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


def _build_plan_cache_key(src_core, tgt_core, num_experts: Optional[int]) -> _PlanCacheKey:
    """Build cache key for reshard plan.

    Args:
        src_core: Source model core (or None for non-collocated destination/idle ranks)
        tgt_core: Target model core (or None for non-collocated source/idle ranks)
        num_experts: Number of MoE experts (or None for non-MoE models)

    Returns:
        Cache key that uniquely identifies this reshard configuration for this rank
    """
    rank = torch.distributed.get_rank()
    src_config = _get_config_tuple(src_core)
    dst_config = _get_config_tuple(tgt_core)
    return _PlanCacheKey(
        rank=rank, src_config=src_config, dst_config=dst_config, num_experts=num_experts
    )


# Module-level cache for refit services to avoid repeated allocations
_service_cache: dict[str, CopyService] = {}
_plan_cache: dict[_PlanCacheKey, Any] = {}


def get_or_create_service(backend: RefitBackendName) -> CopyService:
    """Get or create a cached CopyService instance for the given backend.

    This avoids expensive repeated allocations (especially for NVSHMEM buffers)
    when swap_model_weights is called multiple times with the same backend.
    """
    if backend in _service_cache:
        return _service_cache[backend]

    if backend == "nccl":
        service = NCCLCopyService()
    elif backend == "gloo":
        service = GlooCopyService()
    elif backend == "nvshmem":
        service = NVSHMEMCopyService()
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


def swap_model_weights(
    src_model: LanguageModule,
    target_model: LanguageModule,
    refit_method: Union[RefitBackendName, CopyService],
):
    """
    Orchestrate weight swap/refit.
    - refit_method can be:
        * a string backend name (one of the supported refit backends), or
        * a CopyService instance.
    """
    if isinstance(refit_method, CopyService):
        service = refit_method
        reshard_model_weights(src_model, target_model, service=service)
    elif isinstance(refit_method, str):
        service = get_or_create_service(refit_method)
        reshard_model_weights(src_model, target_model, service=service)
    else:
        raise TypeError("refit_method must be a str backend name or a CopyService instance")


def reshard_model_weights(
    src_model: LanguageModule, target_model: LanguageModule, service: CopyService
):
    """Reshard and copy model weights from ``src_model`` to ``target_model`` using ``service``.

    Supports None for src_model and/or target_model to enable non-collocated mode:
    - (src_model, target_model): Both models present (collocated mode)
    - (src_model, None): Source rank - only sends data (non-collocated)
    - (None, target_model): Destination rank - only receives data (non-collocated)
    - (None, None): Idle rank - participates in collectives but has no transfers (non-collocated)

    In non-collocated mode, metadata includes local rank positions within parallel groups,
    allowing the planner to correctly map between different process group configurations
    without requiring dummy models on every rank.
    """
    global _plan_cache

    # Handle idle ranks (both models None) - they participate in collectives but have no work
    if src_model is None and target_model is None:
        cache_key = _build_plan_cache_key(src_core=None, tgt_core=None, num_experts=None)

        # Use cached plan if available, otherwise build (with collective participation)
        if cache_key not in _plan_cache:
            plan = build_centralized_reshard_plan(None, None, num_experts=None)
            _plan_cache[cache_key] = plan
        else:
            plan = _plan_cache[cache_key]
        execute_reshard_plan(plan, None, None, service=service)
        return

    # Handle None models - extract core modules only from non-None models
    src_core = None
    tgt_core = None
    num_experts = None

    if src_model is not None:
        # Handle list-wrapped modules
        src_lm = src_model[0] if isinstance(src_model, (list, tuple)) else src_model
        num_experts = src_lm.config.num_moe_experts
        # Unwrap to get owning modules (with parameters and pg_collection)
        src_core = unwrap_model(src_lm)
        # Ensure pg_collection exists
        if not hasattr(src_core, "pg_collection") or src_core.pg_collection is None:
            raise RuntimeError("Source model missing pg_collection required for reshard")
        # Fill missing DP group on the source using Megatron's parallel state if not provided
        if getattr(src_core.pg_collection, "dp", None) is None:
            src_core.pg_collection.dp = parallel_state.get_data_parallel_group()

    if target_model is not None:
        # Handle list-wrapped modules
        tgt_lm = target_model[0] if isinstance(target_model, (list, tuple)) else target_model
        if num_experts is None:
            num_experts = tgt_lm.config.num_moe_experts
        # Unwrap to get owning modules (with parameters and pg_collection)
        tgt_core = unwrap_model(tgt_lm)
        # Ensure pg_collection exists
        if not hasattr(tgt_core, "pg_collection") or tgt_core.pg_collection is None:
            raise RuntimeError("Target model missing pg_collection required for reshard")

    # Build or retrieve cached plan
    cache_key = _build_plan_cache_key(src_core, tgt_core, num_experts)

    if cache_key not in _plan_cache:
        # All ranks must participate in planning (collective operations)
        plan = build_centralized_reshard_plan(src_core, tgt_core, num_experts=num_experts)
        _plan_cache[cache_key] = plan
    else:
        plan = _plan_cache[cache_key]

    execute_reshard_plan(plan, src_core, tgt_core, service=service)
