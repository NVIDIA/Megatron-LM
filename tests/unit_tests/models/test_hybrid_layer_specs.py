# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for configurable hybrid-stack module specs."""

from unittest import mock

import pytest

from megatron.core.models.hybrid import hybrid_layer_specs


@pytest.mark.parametrize("moe_grouped_gemm", [False, True])
def test_te_hybrid_stack_spec_configures_moe_grouped_gemm(moe_grouped_gemm):
    moe_spec = object()

    with mock.patch.object(
        hybrid_layer_specs, "get_moe_module_spec", return_value=moe_spec
    ) as get_moe_module_spec:
        stack_spec = hybrid_layer_specs.get_te_hybrid_stack_spec(moe_grouped_gemm)

    get_moe_module_spec.assert_called_once_with(
        use_te=True, num_experts=8, moe_grouped_gemm=moe_grouped_gemm
    )
    assert stack_spec.submodules.moe_layer.submodules.mlp is moe_spec
    assert (
        stack_spec.submodules.moe_layer.module
        is hybrid_layer_specs.hybrid_stack_spec.submodules.moe_layer.module
    )
    assert (
        stack_spec.submodules.mamba_layer
        is hybrid_layer_specs.hybrid_stack_spec.submodules.mamba_layer
    )


def test_te_mamba_stack_spec_is_backward_compatible_alias():
    assert hybrid_layer_specs.get_te_mamba_stack_spec is hybrid_layer_specs.get_te_hybrid_stack_spec
