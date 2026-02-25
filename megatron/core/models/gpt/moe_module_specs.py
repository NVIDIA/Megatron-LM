# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider, InferenceSpecProvider
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_layer_inference import InferenceMoELayer
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.moe.router import InferenceTopKRouter


def get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE.

    Called by mamba_layer_specs.py for standard (non-inference) MoE specs.
    The GPT layer specs call get_moe_module_spec_for_backend directly.

    Args:
        use_te: Whether to use Transformer Engine.
        num_experts: Number of experts.
        moe_grouped_gemm: Whether to use grouped GEMM.
        moe_use_legacy_grouped_gemm: Whether to use legacy grouped GEMM.
    """
    if use_te is not None and use_te:
        backend: BackendSpecProvider = TESpecProvider()
    else:
        backend = LocalSpecProvider()
    return get_moe_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )

def get_moe_module_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    use_te_activation_func: bool = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    assert num_experts is not None

    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func()

    mlp = MLPSubmodules(
        linear_fc1=linear_fc1, linear_fc2=linear_fc2, activation_func=activation_func
    )

    expert_module, expert_submodule = backend.grouped_mlp_modules(
        moe_grouped_gemm is not None and moe_grouped_gemm,
        moe_use_legacy_grouped_gemm is not None and moe_use_legacy_grouped_gemm,
    )
    if expert_submodule is not None:
        expert_submodule.activation_func = activation_func

    experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

    # shared experts spec
    shared_experts = ModuleSpec(module=SharedExpertMLP, submodules=mlp)

    # MoE module spec
    moe_module_spec = ModuleSpec(
        module=MoELayer,
        submodules=MoESubmodules(experts=experts, shared_experts=shared_experts),
        metainfo={"fuse_pre_mlp_layernorm": False},
    )
    return moe_module_spec



def get_inference_optimized_moe_spec() -> ModuleSpec:
    """MoE module spec for inference-optimized transformer impl.

    Uses InferenceSpecProvider to select inference-optimized modules:
    InferenceMoELayer, InferenceTopKRouter, InferenceGroupedMLP.

    Called by mamba_layer_specs.py and gpt_layer_specs.py.
    """
    backend = InferenceSpecProvider()
    activation_func = backend.activation_func()

    expert_module, expert_submodule = backend.grouped_mlp_modules(True, False)
    if expert_submodule is not None:
        expert_submodule.activation_func = activation_func

    experts = ModuleSpec(module=expert_module, submodules=expert_submodule)
    shared_experts = ModuleSpec(
        module=SharedExpertMLP,
        submodules=MLPSubmodules(
            linear_fc1=backend.column_parallel_linear(),
            linear_fc2=backend.row_parallel_linear(),
            activation_func=activation_func,
        ),
    )

    return ModuleSpec(
        module=InferenceMoELayer,
        submodules=MoESubmodules(
            router=InferenceTopKRouter,
            experts=experts,
            shared_experts=shared_experts,
        ),
        metainfo={"fuse_pre_mlp_layernorm": False},
    )
