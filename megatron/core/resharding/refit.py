# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

"""
High-level refit/reshard orchestration:
- swap_model_weights: public API; accepts a backend name or CopyService and delegates.
- reshard_model_weights: transport-agnostic core; builds/caches plan and executes.
"""

from typing import Any, Literal, Optional, Union

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

# Module-level cache for refit services to avoid repeated allocations
_service_cache: dict[str, CopyService] = {}


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
    """
    global _service_cache
    _service_cache.clear()


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
    """Reshard and copy model weights from ``src_model`` to ``target_model`` using ``service``."""
    # Handle list-wrapped modules used throughout training utils
    src_lm = src_model[0] if isinstance(src_model, (list, tuple)) else src_model
    tgt_lm = target_model[0] if isinstance(target_model, (list, tuple)) else target_model

    num_experts = src_lm.config.num_moe_experts

    # Unwrap to get owning modules (with parameters and pg_collection)
    src_core = unwrap_model(src_lm)
    tgt_core = unwrap_model(tgt_lm)

    # Ensure pg_collection exists
    if not hasattr(src_core, "pg_collection") or src_core.pg_collection is None:
        raise RuntimeError("Source model missing pg_collection required for NCCL reshard")
    if not hasattr(tgt_core, "pg_collection") or tgt_core.pg_collection is None:
        raise RuntimeError("Target model missing pg_collection required for NCCL reshard")

    # Fill missing DP group on the source using Megatron's parallel state if not provided
    if getattr(src_core.pg_collection, "dp", None) is None:
        src_core.pg_collection.dp = parallel_state.get_data_parallel_group()

    # caching plan for reuse
    cached_plan: Optional[Any] = getattr(tgt_core, "_cached_reshard_plan", None)
    if cached_plan is None:
        plan = build_centralized_reshard_plan(src_core, tgt_core, num_experts=num_experts)
        setattr(tgt_core, "_cached_reshard_plan", plan)
    else:
        plan = cached_plan

    execute_reshard_plan(plan, src_core, tgt_core, service=service)
