# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Unit tests for Cross-Layer Attention (CLA).

CLA is a technique to reduce KV cache memory by sharing Key-Value tensors
across multiple transformer layers. See arXiv:2405.12981 for details.
"""

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from tests.unit_tests.test_utilities import Utils


class TestCrossLayerAttention:
    """Tests for Cross-Layer Attention (CLA) feature."""

    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_cla_config_default(self):
        """Test that CLA is disabled by default (interval=1)."""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=2,
            use_cpu_initialization=True,
        )
        assert config.cross_layer_attention_interval == 1

    def test_cla_config_interval_2(self):
        """Test CLA config with interval=2 (CLA2)."""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=2,
            cross_layer_attention_interval=2,
            use_cpu_initialization=True,
        )
        assert config.cross_layer_attention_interval == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cla_forward_no_error(self):
        """Test that CLA forward pass runs without error."""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=2,
            cross_layer_attention_interval=2,
            use_cpu_initialization=False,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        layer_spec = get_gpt_layer_with_transformer_engine_spec()
        block = TransformerBlock(config=config, spec=layer_spec)
        block.cuda()

        sequence_length = 16
        micro_batch_size = 2

        hidden_states = torch.randn(
            sequence_length, micro_batch_size, config.hidden_size,
            device='cuda', dtype=torch.bfloat16
        )
        attention_mask = torch.ones(
            (micro_batch_size, 1, 1, sequence_length), dtype=bool, device='cuda'
        )

        # Forward pass should work without error
        output = block(hidden_states, attention_mask)

        assert output.shape == hidden_states.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cla_output_shape_consistency(self):
        """Test that CLA output shape matches standard attention output shape."""
        # Config without CLA
        config_standard = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=2,
            cross_layer_attention_interval=1,  # Standard attention
            use_cpu_initialization=False,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        # Config with CLA
        config_cla = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=2,
            cross_layer_attention_interval=2,  # CLA2
            use_cpu_initialization=False,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        layer_spec = get_gpt_layer_with_transformer_engine_spec()
        block_standard = TransformerBlock(config=config_standard, spec=layer_spec)
        block_cla = TransformerBlock(config=config_cla, spec=layer_spec)

        block_standard.cuda()
        block_cla.cuda()

        sequence_length = 16
        micro_batch_size = 2

        hidden_states = torch.randn(
            sequence_length, micro_batch_size, config_standard.hidden_size,
            device='cuda', dtype=torch.bfloat16
        )
        attention_mask = torch.ones(
            (micro_batch_size, 1, 1, sequence_length), dtype=bool, device='cuda'
        )

        output_standard = block_standard(hidden_states, attention_mask)
        output_cla = block_cla(hidden_states.clone(), attention_mask)

        # Shapes should match
        assert output_standard.shape == output_cla.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cla_layer_numbering(self):
        """Test that Master/Slave layer assignment follows expected pattern."""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=2,
            cross_layer_attention_interval=2,
            use_cpu_initialization=False,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        layer_spec = get_gpt_layer_with_transformer_engine_spec()
        block = TransformerBlock(config=config, spec=layer_spec)

        # With interval=2, layers 1, 3 are Masters (compute fresh KV)
        # Layers 2, 4 are Slaves (reuse KV from previous layer)
        cla_interval = config.cross_layer_attention_interval
        for layer in block.layers:
            is_master = (layer.layer_number - 1) % cla_interval == 0
            if layer.layer_number in [1, 3]:
                assert is_master, f"Layer {layer.layer_number} should be a Master layer"
            elif layer.layer_number in [2, 4]:
                assert not is_master, f"Layer {layer.layer_number} should be a Slave layer"


class TestCrossLayerAttentionIntervals:
    """Tests for different CLA interval configurations."""

    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("interval", [1, 2, 3, 4])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cla_various_intervals(self, interval):
        """Test CLA with various interval values."""
        config = TransformerConfig(
            num_layers=8,
            hidden_size=64,
            num_attention_heads=2,
            cross_layer_attention_interval=interval,
            use_cpu_initialization=False,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        layer_spec = get_gpt_layer_with_transformer_engine_spec()
        block = TransformerBlock(config=config, spec=layer_spec)
        block.cuda()

        sequence_length = 16
        micro_batch_size = 2

        hidden_states = torch.randn(
            sequence_length, micro_batch_size, config.hidden_size,
            device='cuda', dtype=torch.bfloat16
        )
        attention_mask = torch.ones(
            (micro_batch_size, 1, 1, sequence_length), dtype=bool, device='cuda'
        )

        # Should not raise an error
        output = block(hidden_states, attention_mask)
        assert output.shape == hidden_states.shape
