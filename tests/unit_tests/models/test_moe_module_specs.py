# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests that the inference-optimized backend wires InferenceTopKRouter."""

import pytest

from megatron.core.models.backends import InferenceSpecProvider, LocalSpecProvider
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.core.transformer.moe.router import InferenceTopKRouter, TopKRouter


def _router_of(spec):
    """Return the router builder from a get_moe_module_spec_for_backend() result."""
    submodules = spec.keywords["submodules"]
    assert isinstance(submodules, MoESubmodules)
    return submodules.router


class TestMoeModuleSpecRouter:
    @pytest.mark.parametrize("moe_grouped_gemm", [True, False])
    def test_inference_backend_wires_inference_router(self, moe_grouped_gemm):
        """InferenceSpecProvider must select InferenceTopKRouter."""
        spec = get_moe_module_spec_for_backend(
            InferenceSpecProvider(), num_experts=8, moe_grouped_gemm=moe_grouped_gemm
        )
        assert _router_of(spec) is InferenceTopKRouter

    @pytest.mark.parametrize("moe_grouped_gemm", [True, False])
    def test_non_inference_backend_uses_default_router(self, moe_grouped_gemm):
        """Non-inference backends keep the MoESubmodules default (training router)."""
        spec = get_moe_module_spec_for_backend(
            LocalSpecProvider(), num_experts=8, moe_grouped_gemm=moe_grouped_gemm
        )
        # No router override -> dataclass default.
        assert _router_of(spec) is TopKRouter
