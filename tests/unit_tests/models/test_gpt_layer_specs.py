# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests of the GPT layer specs."""

import pytest

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_inference_submodules
from megatron.core.transformer.identity_op import IdentityOp


@pytest.mark.skipif(not HAVE_TE, reason="the inference-optimized spec requires Transformer Engine")
class TestInferenceLayerSpec:
    @pytest.mark.parametrize("multi_latent_attention", [False, True])
    def test_moe_layers_normalize_the_mlp_input(self, multi_latent_attention):
        """A dense MLP normalizes its input inside linear_fc1, but MoE experts do not, so a MoE
        layer needs its own pre-MLP norm, with MLA as with standard attention."""
        dense = get_gpt_layer_with_inference_submodules(
            multi_latent_attention=multi_latent_attention
        )
        moe = get_gpt_layer_with_inference_submodules(
            multi_latent_attention=multi_latent_attention, num_experts=8
        )
        assert dense.pre_mlp_layernorm is IdentityOp
        assert moe.pre_mlp_layernorm is not IdentityOp
        assert (
            moe.pre_mlp_layernorm
            == get_gpt_layer_with_inference_submodules(num_experts=8).pre_mlp_layernorm
        )
