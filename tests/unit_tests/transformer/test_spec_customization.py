# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, fields

import pytest
import torch
import transformer_engine as te

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module, import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from tests.unit_tests.test_utilities import Utils


class TestSpecCustomization:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )

        # specify Transformer Layer spec with all identity ops
        self.transformer_layer_spec = TransformerLayerSubmodules()

        # specify attention spec using already imported class
        self.attention_spec = ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TELayerNormColumnParallelLinear,
                dot_product_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear
            ),
        )

        # specify layernorm spec with module path to test dynamic importing
        self.layernorm_spec = ModuleSpec(
            module=("megatron.core.transformer.custom_layers.transformer_engine", "TENorm"),
        )

        # specify bias dropout add with module path
        self.bda_spec = ModuleSpec(
            module=("megatron.core.fusions.fused_bias_dropout", "get_bias_dropout_add")
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_import_module(self):
        self_attention_cls = import_module(
            module_path=('megatron.core.transformer.attention', 'SelfAttention')
        )
        assert id(self_attention_cls) == id(SelfAttention)

        layernorm_cls = import_module(module_path=self.layernorm_spec.module)
        assert id(layernorm_cls) == id(TENorm)

    def test_build_module(self):
        # Check NoOp TransformerLayer
        random_input = 12
        noop_transformer_layer = [
            build_module(getattr(self.transformer_layer_spec, field.name))
            for field in fields(self.transformer_layer_spec)
        ]

        x = random_input
        for mod in noop_transformer_layer:
            # checking for `IdentityFuncOp` before `IdentityOp` because former
            # is derived from the latter and so the second if statement will
            # always be `True`.
            if isinstance(mod, IdentityFuncOp):
                x = mod()(x)
            elif isinstance(mod, IdentityOp):
                x = mod(x)

        assert x == random_input

        # Check SelfAttention
        self_attention = build_module(
            self.attention_spec, config=self.config, layer_number=1,
        )
        assert isinstance(self_attention, SelfAttention)
        assert self_attention.layer_number == 1
        assert self_attention.attn_mask_type == self.attention_spec.params['attn_mask_type']

        num_weights = sum([p.numel() for p in self_attention.parameters()])
        assert num_weights == 648

        # Check SelfAttention but with already initialized module
        # `self_attention`. In this test, `build_module` acts as a no op as it
        # simply returns the initialized module.
        # NOTE: (sudhakars) Uncomment this test once this feature gets added
        # back.
        # self_attention2 = build_module(
        #     self_attention, config=self.config, spec=self.attention_spec,
        # )
        # assert isinstance(self_attention2, SelfAttention)
        # assert self_attention2.layer_number == 1
        # assert self_attention2.attn_mask_type == self.attention_spec.params['attn_mask_type']

        # num_weights = sum([p.numel() for p in self_attention2.parameters()])
        # assert num_weights == 648

        # Check LayerNorm
        layernorm = build_module(
            self.layernorm_spec,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        assert isinstance(layernorm, te.pytorch.LayerNorm)

        # Check BiasDropoutAdd
        bda_op = build_module(self.bda_spec)
        assert id(bda_op) == id(get_bias_dropout_add)
