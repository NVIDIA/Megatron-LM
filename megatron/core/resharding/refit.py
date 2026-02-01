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

# Supported refit backend names
RefitBackendName = Literal["nccl", "gloo"]


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
        else:
            raise ValueError(f"Unknown refit_method '{refit_method}'")
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
