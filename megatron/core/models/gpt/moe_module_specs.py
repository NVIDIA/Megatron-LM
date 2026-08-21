# Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.

from functools import partial
from typing import Optional

from megatron.core.models.backends import BackendSpecProvider, get_backend
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.moe.shared_experts import FusedSharedExpertMLP, SharedExpertMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import MlpBuilder
from megatron.core.typed_torch import not_none


def _build_shared_experts(
    *,
    config: TransformerConfig,
    pg_collection: ProcessGroupCollection | None,
    gate: bool,
    submodules: MLPSubmodules,
    name: str | None = None,
):
    """Build the shared expert implementation requested by the config."""
    shared_expert_cls = (
        FusedSharedExpertMLP
        if getattr(config, "use_grouped_gemm_for_shared_expert", False)
        else SharedExpertMLP
    )
    return shared_expert_cls(
        config=config, submodules=submodules, gate=gate, pg_collection=pg_collection, name=name
    )


def get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
) -> MlpBuilder:
    """Helper function to get module spec for MoE.

    Called by hybrid_layer_specs.py for standard (non-inference) MoE specs.
    The GPT layer specs call get_moe_module_spec_for_backend directly.

    Args:
        use_te: Whether to use Transformer Engine.
        num_experts: Number of experts.
        moe_grouped_gemm: Whether to use grouped GEMM.
        moe_use_legacy_grouped_gemm: Whether to use legacy grouped GEMM.
    """
    backend = get_backend("transformer_engine" if use_te else "local")
    return get_moe_module_spec_for_backend(
        backend=backend, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )


def get_moe_module_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    use_te_activation_func: bool = False,
) -> MlpBuilder:
    """Helper function to get module spec for MoE"""
    assert num_experts is not None

    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func()

    mlp = MLPSubmodules(
        linear_fc1=linear_fc1, linear_fc2=linear_fc2, activation_func=activation_func
    )

    experts = backend.grouped_mlp_modules(moe_grouped_gemm is not None and moe_grouped_gemm)
    # shared experts spec
    shared_experts = partial(_build_shared_experts, submodules=mlp)

    # The router is an operation the backend owns: the inference backend needs compact
    # [tokens, topk] index routing, and every other backend keeps the MoESubmodules default
    # (training TopKRouter, dense [tokens, num_experts] map).
    router = backend.moe_router()
    submodule_kwargs = {"router": router} if router is not None else {}

    # MoE module spec
    return partial(
        MoELayer,
        submodules=MoESubmodules(
            experts=experts, shared_experts=shared_experts, **submodule_kwargs
        ),
    )


def get_inference_optimized_moe_spec() -> MlpBuilder:
    """MoE module spec for inference-optimized transformer impl.

    Uses the inference backend to select inference-optimized modules:
    InferenceTopKRouter, InferenceGroupedMLP. MoELayer detects inference mode
    via config.transformer_impl and sets up the inference dispatcher internally.

    Called by hybrid_layer_specs.py and gpt_layer_specs.py.
    """
    backend = get_backend("inference_optimized")
    activation_func = backend.activation_func()
    router = not_none(backend.moe_router())

    experts = backend.grouped_mlp_modules(True)
    shared_experts = partial(
        _build_shared_experts,
        submodules=MLPSubmodules(
            linear_fc1=backend.column_parallel_linear(),
            linear_fc2=backend.row_parallel_linear(),
            activation_func=activation_func,
        ),
    )

    return partial(
        MoELayer,
        submodules=MoESubmodules(router=router, experts=experts, shared_experts=shared_experts),
    )
