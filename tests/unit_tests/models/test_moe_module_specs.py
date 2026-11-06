# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests that the inference-optimized backend wires InferenceTopKRouter."""

import pytest

from megatron.core.models.backends import InferenceSpecProvider, LocalSpecProvider
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
    get_mlp_module_spec,
)
from megatron.core.models.gpt.moe_module_specs import (
    get_moe_module_spec,
    get_moe_module_spec_for_backend,
)
from megatron.core.models.multimodal.llava_spec import (
    decoder_model_with_transformer_engine_default_spec,
)
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.core.transformer.moe.router import InferenceTopKRouter, TopKRouter
from megatron.core.transformer.spec_utils import get_submodules


def _router_of(spec):
    """Return the router builder from a get_moe_module_spec_for_backend() result."""
    submodules = spec.keywords["submodules"]
    assert isinstance(submodules, MoESubmodules)
    return submodules.router


@pytest.mark.parametrize("builder", [get_moe_module_spec, get_mlp_module_spec])
@pytest.mark.parametrize("use_te", [False, True])
@pytest.mark.parametrize("overrides", [{}, {"moe_grouped_gemm": True}, {"moe_grouped_gemm": False}])
def test_expert_builder_defaults_and_legacy_override(builder, use_te, overrides):
    spec = builder(use_te=use_te, num_experts=2, **overrides)
    experts = get_submodules(spec).experts
    expected = TEGroupedMLP if use_te and overrides.get("moe_grouped_gemm", True) else SequentialMLP
    assert experts.func is expected


@pytest.mark.parametrize(
    "builder",
    [
        get_gpt_layer_with_transformer_engine_spec,
        get_gpt_layer_with_transformer_engine_submodules,
        decoder_model_with_transformer_engine_default_spec,
    ],
)
def test_layer_builders_default_to_grouped_experts(builder):
    spec = builder(num_experts=2)
    submodules = spec if hasattr(spec, "mlp") else get_submodules(spec)
    assert get_submodules(submodules.mlp).experts.func is TEGroupedMLP


def test_grouped_moe_default_keeps_dense_mlp():
    assert get_mlp_module_spec(use_te=True).func == MLP.as_mlp_submodule


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
