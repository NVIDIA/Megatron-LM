from megatron.core.models.common.language_module.language_module import LanguageModule
import torch
import torch.distributed as dist
from typing import Any
from megatron.core import parallel_state
from megatron.core.resharding import build_centralized_reshard_plan, execute_reshard_plan
from typing import Any, Optional



def _unwrap_module(module: LanguageModule) -> Any:
    return module.module.module if hasattr(module, 'module') and hasattr(module.module, 'module') else module.module if hasattr(module, 'module') else module

def swap_model_weights(src_model: LanguageModule, target_model: LanguageModule, refit_method: str):
    if  refit_method == "nccl":
        nccl_model_swap(src_model, target_model)
    else:
        raise ValueError(f"Invalid refit method: {refit_method}")

def nccl_model_swap(src_model: LanguageModule, target_model: LanguageModule):
    # Handle list-wrapped modules used throughout training utils
    src_lm = src_model[0] if isinstance(src_model, (list, tuple)) else src_model
    tgt_lm = target_model[0] if isinstance(target_model, (list, tuple)) else target_model

    num_experts = src_lm.config.num_moe_experts

    # Unwrap to get owning modules (with parameters and pg_collection)
    src_core = _unwrap_module(src_lm)
    tgt_core = _unwrap_module(tgt_lm)

    # Ensure pg_collection exists
    if not hasattr(src_core, "pg_collection") or src_core.pg_collection is None:
        raise RuntimeError("Source model missing pg_collection required for NCCL reshard")
    if not hasattr(tgt_core, "pg_collection") or tgt_core.pg_collection is None:
        raise RuntimeError("Target model missing pg_collection required for NCCL reshard")

    #TODO(Peter): We should figure out why this happens. Seems like a bug in Orthotope.
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
    execute_reshard_plan(plan, src_core, tgt_core)