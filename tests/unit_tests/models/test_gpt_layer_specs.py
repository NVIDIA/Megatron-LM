# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
)

_TE = get_gpt_layer_with_transformer_engine_spec
_LOCAL = get_gpt_layer_local_spec
_HC = HyperConnectionTransformerLayer
_HC_MOD = HyperConnectionModule
_TL = TransformerLayer
_ID = IdentityOp


class TestGptLayerSpecsHyperConnection:
    """Test that enable_hyper_connection controls module types in layer specs."""

    @pytest.mark.parametrize(
        "factory,kwargs,expected_module,expected_hc",
        [
            (_TE, {}, _TL, _ID),
            (_TE, {"enable_hyper_connection": True}, _HC, _HC_MOD),
            (_TE, {"enable_hyper_connection": False}, _TL, _ID),
            (_TE, {"multi_latent_attention": True, "enable_hyper_connection": False}, _TL, _ID),
            (_TE, {"multi_latent_attention": True, "enable_hyper_connection": True}, _HC, _HC_MOD),
            (_LOCAL, {}, _TL, _ID),
            (_LOCAL, {"enable_hyper_connection": True}, _HC, _HC_MOD),
            (_LOCAL, {"enable_hyper_connection": False}, _TL, _ID),
            (_LOCAL, {"multi_latent_attention": True, "enable_hyper_connection": False}, _TL, _ID),
            (
                _LOCAL,
                {"multi_latent_attention": True, "enable_hyper_connection": True},
                _HC,
                _HC_MOD,
            ),
            (_LOCAL, {"normalization": "RMSNorm", "enable_hyper_connection": False}, _TL, _ID),
            (_LOCAL, {"normalization": "RMSNorm", "enable_hyper_connection": True}, _HC, _HC_MOD),
        ],
        ids=[
            "te_default",
            "te_enable",
            "te_disable",
            "te_mla_disable",
            "te_mla_enable",
            "local_default",
            "local_enable",
            "local_disable",
            "local_mla_disable",
            "local_mla_enable",
            "local_rmsnorm_disable",
            "local_rmsnorm_enable",
        ],
    )
    def test_hyper_connection_spec(self, factory, kwargs, expected_module, expected_hc):
        spec = factory(**kwargs)
        assert spec.module is expected_module
        assert spec.submodules.self_attention_hyper_connection is expected_hc
        assert spec.submodules.mlp_hyper_connection is expected_hc
