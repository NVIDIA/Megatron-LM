# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider, InferenceSpecProvider
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_layer_inference import InferenceMoELayer
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec


def get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    inference_optimized: bool = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE
    
    Args:
        use_te: Whether to use Transformer Engine.
        num_experts: Number of experts.
        moe_grouped_gemm: Whether to use grouped GEMM.
        moe_use_legacy_grouped_gemm: Whether to use legacy grouped GEMM.
        inference_optimized: If True, use InferenceMoELayer for optimized inference.
    """
    # This function is called my mamba_layer_specs.py 
    # The GPT layer specs directly calls get_moe_module_spec_for_backend

    if use_te is not None and use_te:
        backend: BackendSpecProvider = TESpecProvider()
    elif inference_optimized:
        backend = InferenceSpecProvider()
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
    """Helper function to get module spec for MoE
    
    Args:
        backend: Backend spec provider (TE or Local).
        num_experts: Number of experts.
        moe_grouped_gemm: Whether to use grouped GEMM.
        moe_use_legacy_grouped_gemm: Whether to use legacy grouped GEMM.
        use_te_activation_func: Whether to use TE activation function.
        inference_optimized: If True, use InferenceMoELayer for optimized inference.
    """
    assert num_experts is not None
    inference_optimized: bool = isinstance(backend, InferenceSpecProvider)
    
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

    # Select MoE layer class based on inference_optimized flag
    if inference_optimized:
        moe_layer_class = InferenceMoELayer
    else:
        moe_layer_class = MoELayer

    # MoE module spec
    moe_module_spec = ModuleSpec(
        module=moe_layer_class,
        submodules=MoESubmodules(experts=experts, shared_experts=shared_experts),
        metainfo={"fuse_pre_mlp_layernorm": False},
    )
    return moe_module_spec
