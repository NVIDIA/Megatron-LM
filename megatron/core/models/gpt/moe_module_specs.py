# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec

try:
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
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
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    assert num_experts is not None

    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = backend.row_parallel_linear()

    mlp = MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)

    expert_module, expert_submodule = backend.grouped_mlp_modules(
        moe_grouped_gemm is not None and moe_grouped_gemm,
        moe_use_legacy_grouped_gemm is not None and moe_use_legacy_grouped_gemm,
    )

    experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

    # shared experts spec
    shared_experts = ModuleSpec(module=SharedExpertMLP, params={"gate": False}, submodules=mlp)

    # MoE module spec
    moe_module_spec = ModuleSpec(
        module=MoELayer, submodules=MoESubmodules(experts=experts, shared_experts=shared_experts)
    )
    return moe_module_spec
