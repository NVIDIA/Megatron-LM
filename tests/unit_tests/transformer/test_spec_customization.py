# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import sys
from dataclasses import dataclass, fields

import pytest
import torch
import transformer_engine as te

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.parallel_state import get_context_parallel_group, get_tensor_model_parallel_group
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module, import_module
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import is_te_min_version
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
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
                q_layernorm=IdentityOp,
                k_layernorm=IdentityOp,
            ),
        )

        # specify layernorm spec with module path to test dynamic importing
        self.layernorm_spec = ModuleSpec(
            module=("megatron.core.extensions.transformer_engine", "TENorm")
        )

        # specify bias dropout add with module path
        self.bda_spec = ModuleSpec(
            module=("megatron.core.fusions.fused_bias_dropout", "get_bias_dropout_add")
        )

        # Create model process groups for test.
        self.model_comm_pgs = ModelCommProcessGroups(
            tp=get_tensor_model_parallel_group(), cp=get_context_parallel_group()
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
            if field.name != 'sharded_state_dict_keys_map'
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
        self_attention = build_module(self.attention_spec, config=self.config, layer_number=1)
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

    def test_sliding_window_attention(self):
        if not is_te_min_version("1.2.0"):
            print("SWA not tested because TE version is not >= 1.2.0", file=sys.stderr)
            return

        config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            window_size=[10, 0],
        )
        # Make sure DotProductAttention throws (swa unsupported).
        threw = False
        try:
            attn = DotProductAttention(
                config,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type='self',
                model_comm_pgs=self.model_comm_pgs,
            )
        except:
            threw = True
        finally:
            assert threw, 'Expected DotProductAttention to throw exception for SWA'

        # Test TEDotProductAttention
        attn = TEDotProductAttention(
            config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            model_comm_pgs=self.model_comm_pgs,
        )
        # Make sure window-size is what we expect.
        assert attn.window_size == config.window_size

        # Single integer window-size unsupported, make sure it throws
        threw = False
        try:
            config.window_size = 11
            attn = TEDotProductAttention(
                config,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type='self',
                model_comm_pgs=self.model_comm_pgs,
            )
        except:
            threw = True
        finally:
            assert threw, "Expected TEDotProductAttention to throw for integer window-size"

        # `None` makes this causal.
        config.window_size = None
        attn = TEDotProductAttention(
            config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            model_comm_pgs=self.model_comm_pgs,
        )
        # Make sure it's causal.
        assert attn.window_size == (-1, 0)

    def test_transformer_block_custom(self):
        """
        This test checks that the two ways of passing `layer_spec` to  a
        `TransformerBlock` result in an identical model:
        1. ModuleSpec(module=..., submodules=...)
        2. TransformerBlockSubmodules(layer_specs=...)
        """

        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        layer_local_spec = get_gpt_layer_local_spec()

        # The following way can be used to pass a different `TransformerLayer`
        # and internally the `TransformerBlock` would fan out the single
        # `ModuleSpec` layer spec provided to all the layers of the block.
        layer_spec1 = ModuleSpec(module=TransformerLayer, submodules=layer_local_spec.submodules)
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(0)
        parallel_transformer_block1 = TransformerBlock(transformer_config, layer_spec1)

        layer_spec2 = TransformerBlockSubmodules(
            layer_specs=[
                ModuleSpec(module=TransformerLayer, submodules=layer_local_spec.submodules)
            ]
            * transformer_config.num_layers,
            layer_norm=TENorm,
        )
        # make sure the model init conditions are identical
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(0)
        parallel_transformer_block2 = TransformerBlock(transformer_config, layer_spec2)

        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_block1.cuda()
        parallel_transformer_block2.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, transformer_config.hidden_size)
        )
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        out1 = parallel_transformer_block1(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        out2 = parallel_transformer_block2(
            hidden_states=hidden_states, attention_mask=attention_mask
        )

        assert torch.all(torch.eq(out1, out2))
        assert out1.shape[0] == sequence_length == out2.shape[0]
        assert out1.shape[1] == micro_batch_size == out2.shape[1]
        assert out1.shape[2] == transformer_config.hidden_size == out2.shape[2]

    def test_l2_qk_norm(self):
        """Test L2 normalization for QK vectors using local spec."""
        layer_spec = get_gpt_layer_local_spec(qk_l2_norm=True)

        # Build the self-attention module from the spec
        self_attention = build_module(
            layer_spec.submodules.self_attention, config=self.config, layer_number=1
        )

        assert isinstance(self_attention, SelfAttention)
        # Verify that q_layernorm and k_layernorm are L2Norm instances
        assert isinstance(self_attention.q_layernorm, L2Norm)
        assert isinstance(self_attention.k_layernorm, L2Norm)

        # Test forward pass
        sequence_length = 32
        micro_batch_size = 2
        self_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.config.hidden_size)
        ).cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        output, bias = self_attention(hidden_states=hidden_states, attention_mask=attention_mask)

        # Assert output shape is same as input shape
        assert output.shape == hidden_states.shape
