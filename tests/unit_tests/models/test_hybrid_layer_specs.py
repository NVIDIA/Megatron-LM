# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for configurable hybrid-stack module specs."""

from megatron.core.models.hybrid import hybrid_layer_specs


def test_no_grouped_gemm_hybrid_stack_spec_has_static_moe_implementation():
    stack_spec = hybrid_layer_specs.hybrid_stack_spec_no_moe_grouped_gemm

    assert stack_spec.submodules.moe_layer.submodules.mlp is hybrid_layer_specs.moe_no_grouped_gemm
    assert (
        stack_spec.submodules.moe_layer.module
        is hybrid_layer_specs.hybrid_stack_spec.submodules.moe_layer.module
    )
    assert (
        stack_spec.submodules.mamba_layer
        is hybrid_layer_specs.hybrid_stack_spec.submodules.mamba_layer
    )
    assert (
        stack_spec.submodules.csa_layer is hybrid_layer_specs.hybrid_stack_spec.submodules.csa_layer
    )


def test_no_grouped_gemm_mamba_stack_spec_is_backward_compatible_alias():
    assert (
        hybrid_layer_specs.mamba_stack_spec_no_moe_grouped_gemm
        is hybrid_layer_specs.hybrid_stack_spec_no_moe_grouped_gemm
    )
