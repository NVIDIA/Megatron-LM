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


class TestGptLayerSpecsHyperConnection:
    """Test that enable_hyper_connection controls module types in layer specs."""

    def test_te_spec_default_no_hyper_connection(self):
        spec = get_gpt_layer_with_transformer_engine_spec()
        assert spec.module is TransformerLayer
        assert spec.submodules.self_attention_hyper_connection is IdentityOp
        assert spec.submodules.mlp_hyper_connection is IdentityOp

    def test_te_spec_enable_hyper_connection(self):
        spec = get_gpt_layer_with_transformer_engine_spec(enable_hyper_connection=True)
        assert spec.module is HyperConnectionTransformerLayer
        assert spec.submodules.self_attention_hyper_connection is HyperConnectionModule
        assert spec.submodules.mlp_hyper_connection is HyperConnectionModule

    def test_te_spec_disable_hyper_connection_explicit(self):
        spec = get_gpt_layer_with_transformer_engine_spec(enable_hyper_connection=False)
        assert spec.module is TransformerLayer
        assert spec.submodules.self_attention_hyper_connection is IdentityOp
        assert spec.submodules.mlp_hyper_connection is IdentityOp

    def test_te_spec_mla_no_hyper_connection(self):
        spec = get_gpt_layer_with_transformer_engine_spec(
            multi_latent_attention=True, enable_hyper_connection=False
        )
        assert spec.module is TransformerLayer
        assert spec.submodules.self_attention_hyper_connection is IdentityOp
        assert spec.submodules.mlp_hyper_connection is IdentityOp

    def test_te_spec_mla_enable_hyper_connection(self):
        spec = get_gpt_layer_with_transformer_engine_spec(
            multi_latent_attention=True, enable_hyper_connection=True
        )
        assert spec.module is HyperConnectionTransformerLayer
        assert spec.submodules.self_attention_hyper_connection is HyperConnectionModule
        assert spec.submodules.mlp_hyper_connection is HyperConnectionModule

    def test_local_spec_default_no_hyper_connection(self):
        spec = get_gpt_layer_local_spec()
        assert spec.module is TransformerLayer
        assert spec.submodules.self_attention_hyper_connection is IdentityOp
        assert spec.submodules.mlp_hyper_connection is IdentityOp

    def test_local_spec_enable_hyper_connection(self):
        spec = get_gpt_layer_local_spec(enable_hyper_connection=True)
        assert spec.module is HyperConnectionTransformerLayer
        assert spec.submodules.self_attention_hyper_connection is HyperConnectionModule
        assert spec.submodules.mlp_hyper_connection is HyperConnectionModule

    def test_local_spec_disable_hyper_connection_explicit(self):
        spec = get_gpt_layer_local_spec(enable_hyper_connection=False)
        assert spec.module is TransformerLayer
        assert spec.submodules.self_attention_hyper_connection is IdentityOp
        assert spec.submodules.mlp_hyper_connection is IdentityOp

    def test_local_spec_mla_no_hyper_connection(self):
        spec = get_gpt_layer_local_spec(
            multi_latent_attention=True, enable_hyper_connection=False
        )
        assert spec.module is TransformerLayer
        assert spec.submodules.self_attention_hyper_connection is IdentityOp
        assert spec.submodules.mlp_hyper_connection is IdentityOp

    def test_local_spec_mla_enable_hyper_connection(self):
        spec = get_gpt_layer_local_spec(
            multi_latent_attention=True, enable_hyper_connection=True
        )
        assert spec.module is HyperConnectionTransformerLayer
        assert spec.submodules.self_attention_hyper_connection is HyperConnectionModule
        assert spec.submodules.mlp_hyper_connection is HyperConnectionModule

    def test_local_spec_rmsnorm_no_hyper_connection(self):
        spec = get_gpt_layer_local_spec(normalization="RMSNorm", enable_hyper_connection=False)
        assert spec.module is TransformerLayer
        assert spec.submodules.self_attention_hyper_connection is IdentityOp
        assert spec.submodules.mlp_hyper_connection is IdentityOp

    def test_local_spec_rmsnorm_enable_hyper_connection(self):
        spec = get_gpt_layer_local_spec(normalization="RMSNorm", enable_hyper_connection=True)
        assert spec.module is HyperConnectionTransformerLayer
        assert spec.submodules.self_attention_hyper_connection is HyperConnectionModule
        assert spec.submodules.mlp_hyper_connection is HyperConnectionModule
