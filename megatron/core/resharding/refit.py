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

# Global plan cache: maps rank -> ReshardPlan
# This is simpler and more consistent than caching on model objects or mixing
# model attributes with globals. All ranks cache in the same way.
_plan_cache: dict[int, Any] = {}


def clear_plan_cache():
    """Clear the cached reshard plans.

    Call this if you need to invalidate the cache, for example:
    - When switching between different model configurations
    - When model parallelism settings change
    - To free memory after refit operations are complete
    """
    global _plan_cache
    _plan_cache.clear()


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
        if refit_method == "nccl":
            service = NCCLCopyService()
            reshard_model_weights(src_model, target_model, service=service)
        elif refit_method == "gloo":
            # Debug / fallback backend: run refit over CPU/Gloo instead of NCCL.
            service = GlooCopyService()
            reshard_model_weights(src_model, target_model, service=service)
        elif refit_method == "nvshmem":
            service = NVSHMEMCopyService()
            reshard_model_weights(src_model, target_model, service=service)
        else:
            raise ValueError(f"Unknown refit_method '{refit_method}'")
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
    import torch.distributed as dist

    # Handle idle ranks (both models None) - they participate in collectives but have no work
    if src_model is None and target_model is None:
        rank = dist.get_rank()

        # Use cached plan if available, otherwise build (with collective participation)
        if rank not in _plan_cache:
            plan = build_centralized_reshard_plan(None, None, num_experts=None)
            _plan_cache[rank] = plan
        else:
            plan = _plan_cache[rank]
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
    # Use rank-based cache - simpler and more consistent than model attributes
    rank = dist.get_rank()

    if rank not in _plan_cache:
        # All ranks must participate in planning (collective operations)
        plan = build_centralized_reshard_plan(src_core, tgt_core, num_experts=num_experts)
        _plan_cache[rank] = plan
    else:
        plan = _plan_cache[rank]

    execute_reshard_plan(plan, src_core, tgt_core, service=service)
