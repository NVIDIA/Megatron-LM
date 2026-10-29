# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests of the GPT layer specs."""

import dataclasses

import pytest

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_inference_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.identity_op import IdentityOp


def _norm_slots(submodules):
    """Which norm slots of a layer and of its self-attention hold a norm."""
    attention = submodules.self_attention.submodules
    return {
        f"{prefix}{field.name}": getattr(owner, field.name) not in (None, IdentityOp)
        for prefix, owner in (("", submodules), ("self_attention.", attention))
        for field in dataclasses.fields(owner)
        if field.name.endswith("layernorm")
    }


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

    @pytest.mark.parametrize("qk_layernorm", [False, True])
    @pytest.mark.parametrize("num_experts", [None, 8])
    @pytest.mark.parametrize("multi_latent_attention", [False, True])
    def test_norm_slots_match_the_transformer_engine_spec(
        self, multi_latent_attention, num_experts, qk_layernorm
    ):
        """Checkpoints are trained with the Transformer Engine spec, so the inference spec must
        normalize in the same places: each norm slot holds a norm in both specs or in neither."""
        kwargs = dict(
            multi_latent_attention=multi_latent_attention,
            num_experts=num_experts,
            qk_layernorm=qk_layernorm,
        )
        assert _norm_slots(get_gpt_layer_with_inference_submodules(**kwargs)) == _norm_slots(
            get_gpt_layer_with_transformer_engine_submodules(**kwargs)
        )
