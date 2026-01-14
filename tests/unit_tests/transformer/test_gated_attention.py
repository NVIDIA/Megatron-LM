# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


class TestGatedAttention:
    """Test suite for query-dependent gated attention functionality."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_parameter_initialization(self):
        """Test 1: Verify gate parameters (weight + bias) are created with correct shape/dtype."""
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            gated_attention=True,
            gated_attention_init_value=1.0,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        # Test DotProductAttention
        attention = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

        # Verify parameters exist
        assert hasattr(attention, "attention_gate_weight")
        assert hasattr(attention, "attention_gate_bias")
        assert attention.attention_gate_weight is not None
        assert attention.attention_gate_bias is not None

        # Verify shapes
        expected_heads = config.num_attention_heads // config.tensor_model_parallel_size
        assert attention.attention_gate_weight.shape == (expected_heads, config.hidden_size)
        assert attention.attention_gate_bias.shape == (expected_heads,)

        # Verify dtype
        assert attention.attention_gate_weight.dtype == config.params_dtype
        assert attention.attention_gate_bias.dtype == config.params_dtype

        # Verify bias initial value
        assert torch.allclose(
            attention.attention_gate_bias,
            torch.tensor(1.0, dtype=config.params_dtype),
            rtol=1e-3,
        )

    def test_parameter_initialization_disabled(self):
        """Test 4: Verify no overhead when gated_attention=False."""
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            gated_attention=False,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        attention = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

        # Verify no gate parameters when disabled
        assert attention.attention_gate_weight is None
        assert attention.attention_gate_bias is None

    def test_query_dependent_gating(self):
        """Test that gates are input-dependent (not static)."""
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            gated_attention=True,
            gated_attention_init_value=0.0,
            bf16=False,
            params_dtype=torch.float32,
        )

        attention = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        ).cuda()

        # Create two different hidden_states inputs
        sequence_length = 16
        batch_size = 2
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads

        hidden_states_1 = torch.randn(
            sequence_length, batch_size, config.hidden_size, dtype=torch.float32, device="cuda"
        )
        hidden_states_2 = torch.randn(
            sequence_length, batch_size, config.hidden_size, dtype=torch.float32, device="cuda"
        )

        # Create identical Q, K, V for both
        query = torch.randn(
            sequence_length, batch_size, num_heads, head_dim, dtype=torch.float32, device="cuda"
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        attention_mask = torch.ones(
            (batch_size, 1, sequence_length, sequence_length), dtype=bool, device="cuda"
        )

        # Forward with different hidden_states should give different outputs
        output_1 = attention(query, key, value, attention_mask, hidden_states=hidden_states_1)
        output_2 = attention(query, key, value, attention_mask, hidden_states=hidden_states_2)

        # Outputs should be different because gates depend on hidden_states
        assert not torch.allclose(output_1, output_2, rtol=1e-3)

    def test_forward_shape_invariance(self):
        """Test 2: Verify output shape same with/without gating."""
        config_no_gate = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            gated_attention=False,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        config_with_gate = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            gated_attention=True,
            gated_attention_init_value=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        attn_no_gate = DotProductAttention(
            config=config_no_gate,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        ).cuda()

        attn_with_gate = DotProductAttention(
            config=config_with_gate,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        ).cuda()

        sequence_length = 32
        batch_size = 2
        hidden_size = config_no_gate.hidden_size
        num_heads = config_no_gate.num_attention_heads
        head_dim = hidden_size // num_heads

        hidden_states = torch.randn(
            sequence_length, batch_size, hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        query = torch.randn(
            sequence_length, batch_size, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        attention_mask = torch.ones(
            (batch_size, 1, sequence_length, sequence_length), dtype=bool, device="cuda"
        )

        output_no_gate = attn_no_gate(query, key, value, attention_mask)
        output_with_gate = attn_with_gate(query, key, value, attention_mask, hidden_states=hidden_states)

        assert output_no_gate.shape == output_with_gate.shape
        assert output_no_gate.shape == (sequence_length, batch_size, hidden_size)

    def test_gradient_flow(self):
        """Test 3: Verify gates receive gradients during backprop."""
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            gated_attention=True,
            gated_attention_init_value=0.0,
            bf16=False,
            params_dtype=torch.float32,
        )

        attention = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        ).cuda()

        sequence_length = 16
        batch_size = 2
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads

        hidden_states = torch.randn(
            sequence_length, batch_size, hidden_size,
            dtype=torch.float32, device="cuda", requires_grad=True
        )
        query = torch.randn(
            sequence_length, batch_size, num_heads, head_dim,
            dtype=torch.float32, device="cuda", requires_grad=True
        )
        key = torch.randn_like(query, requires_grad=True)
        value = torch.randn_like(query, requires_grad=True)
        attention_mask = torch.ones(
            (batch_size, 1, sequence_length, sequence_length), dtype=bool, device="cuda"
        )

        output = attention(query, key, value, attention_mask, hidden_states=hidden_states)
        loss = output.sum()
        loss.backward()

        # Verify gradients exist for both weight and bias
        assert attention.attention_gate_weight.grad is not None
        assert attention.attention_gate_bias.grad is not None
        assert attention.attention_gate_weight.grad.shape == attention.attention_gate_weight.shape
        assert attention.attention_gate_bias.grad.shape == attention.attention_gate_bias.shape
        # Verify gradients are non-zero
        assert torch.abs(attention.attention_gate_weight.grad).sum() > 0
        assert torch.abs(attention.attention_gate_bias.grad).sum() > 0

    def test_gated_attention_with_self_attention(self):
        """Integration test with SelfAttention module."""
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            gated_attention=True,
            gated_attention_init_value=0.5,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        self_attention = SelfAttention(
            config=config,
            submodules=get_gpt_layer_local_spec().submodules.self_attention.submodules,
            layer_number=1,
        ).cuda()

        # Verify gate parameters exist in core attention
        assert hasattr(self_attention.core_attention, "attention_gate_weight")
        assert hasattr(self_attention.core_attention, "attention_gate_bias")
        assert self_attention.core_attention.attention_gate_weight is not None
        assert self_attention.core_attention.attention_gate_bias is not None

        sequence_length = 32
        batch_size = 2
        hidden_states = torch.randn(
            sequence_length, batch_size, config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        attention_mask = torch.ones(
            (batch_size, 1, 1, sequence_length), dtype=bool, device="cuda"
        )

        output, bias = self_attention(hidden_states, attention_mask)

        assert output.shape == (sequence_length, batch_size, config.hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.skipif(not is_te_min_version("1.0.0"), reason="TE not available")
    def test_te_gated_attention(self):
        """Test 5: Gated attention with TransformerEngine backend."""
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            gated_attention=True,
            gated_attention_init_value=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        self_attention = SelfAttention(
            config=config,
            submodules=get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
            layer_number=1,
        ).cuda()

        # Verify gate parameters exist
        assert hasattr(self_attention.core_attention, "attention_gate_weight")
        assert hasattr(self_attention.core_attention, "attention_gate_bias")
        assert self_attention.core_attention.attention_gate_weight is not None
        assert self_attention.core_attention.attention_gate_bias is not None

        sequence_length = 32
        batch_size = 2
        hidden_states = torch.randn(
            sequence_length, batch_size, config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        attention_mask = torch.ones(
            (batch_size, 1, 1, sequence_length), dtype=bool, device="cuda"
        )

        output, bias = self_attention(hidden_states, attention_mask)

        assert output.shape == (sequence_length, batch_size, config.hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
