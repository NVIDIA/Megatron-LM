# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import inspect
import os

import pytest
import torch

import megatron.core.transformer.utils as transformer_utils
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    is_layer_window_attention,
    set_model_to_sequence_parallel,
)
from tests.unit_tests.test_utilities import Utils


class TestGPTModel:

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

        self.tensor_model_parallel_size = 2
        Utils.initialize_model_parallel(self.tensor_model_parallel_size, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=48,
            num_attention_heads=4,
            use_cpu_initialization=True,
            tensor_model_parallel_size=self.tensor_model_parallel_size,
            sequence_parallel=False,
        )
        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=8,
            position_embedding_type="rope",
            parallel_output=False,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_post_process_forward(self):
        _ = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        self.gpt_model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = self.gpt_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.gpt_model.vocab_size

        set_model_to_sequence_parallel(self.gpt_model, set_to=True)
        logits = self.gpt_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        # Test cache has been built
        assert transformer_utils._sequence_parallel_attr_cache is not None

        # Check the modules have been flipped
        for attribute, modules in transformer_utils._sequence_parallel_attr_cache[
            id(self.gpt_model)
        ].items():
            for module in modules:
                assert getattr(module, attribute) == True

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.gpt_model.vocab_size

        set_model_to_sequence_parallel(self.gpt_model, set_to=False)
        logits = self.gpt_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.gpt_model.vocab_size

        # Check the modules have been flipped
        for attribute, modules in transformer_utils._sequence_parallel_attr_cache[
            id(self.gpt_model)
        ].items():
            for module in modules:
                assert getattr(module, attribute) == False


class TestIsLayerWindowAttention:
    """Comprehensive tests for is_layer_window_attention function."""

    def test_no_window_size(self):
        """Test when window_size is None or empty."""
        config = TransformerConfig(
            num_layers=4, hidden_size=64, num_attention_heads=8, window_size=None
        )

        # Should return False for any layer when window_size is None
        for layer_number in [1, 2, 3, 4]:
            assert (
                is_layer_window_attention(
                    config.window_size, config.window_attn_skip_freq, layer_number
                )
                == False
            )

        # Test with empty list
        config.window_size = []
        for layer_number in [1, 2, 3, 4]:
            assert (
                is_layer_window_attention(
                    config.window_size, config.window_attn_skip_freq, layer_number
                )
                == False
            )

    def test_window_size_with_no_skip_freq(self):
        """Test when window_size is set but window_attn_skip_freq is None."""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=8,
            window_size=(10, 0),
            window_attn_skip_freq=None,
        )

        # Should return True for all layers when skip_freq is None
        for layer_number in [1, 2, 3, 4]:
            assert (
                is_layer_window_attention(
                    config.window_size, config.window_attn_skip_freq, layer_number
                )
                == True
            )

    def test_integer_skip_frequency(self):
        """Test window attention with integer skip frequency."""
        config = TransformerConfig(
            num_layers=8,
            hidden_size=64,
            num_attention_heads=8,
            window_size=(10, 0),
            window_attn_skip_freq=3,  # Skip every 3rd layer
        )

        # Layer numbers are 1-indexed
        # Layers 3, 6 should NOT use window attention (skip)
        # Other layers should use window attention
        expected_results = {
            1: True,  # 1 % 3 != 0
            2: True,  # 2 % 3 != 0
            3: False,  # 3 % 3 == 0
            4: True,  # 4 % 3 != 0
            5: True,  # 5 % 3 != 0
            6: False,  # 6 % 3 == 0
            7: True,  # 7 % 3 != 0
            8: True,  # 8 % 3 != 0
        }

        for layer_number, expected in expected_results.items():
            result = is_layer_window_attention(
                config.window_size, config.window_attn_skip_freq, layer_number
            )
            assert result == expected, f"Layer {layer_number}: expected {expected}, got {result}"

    def test_list_skip_frequency(self):
        """Test window attention with list-based skip frequency."""
        config = TransformerConfig(
            num_layers=6,
            hidden_size=64,
            num_attention_heads=8,
            window_size=(10, 0),
            window_attn_skip_freq=[True, False, True, False, True, False],
        )

        # List is 0-indexed, but layer_number is 1-indexed
        # So layer 1 uses index 0, layer 2 uses index 1, etc.
        expected_results = {
            1: True,  # index 0: True
            2: False,  # index 1: False
            3: True,  # index 2: True
            4: False,  # index 3: False
            5: True,  # index 4: True
            6: False,  # index 5: False
        }

        for layer_number, expected in expected_results.items():
            result = is_layer_window_attention(
                config.window_size, config.window_attn_skip_freq, layer_number
            )
            assert result == expected, f"Layer {layer_number}: expected {expected}, got {result}"

    def test_list_skip_frequency_boolean_conversion(self):
        """Test that list values are properly converted to boolean."""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=8,
            window_size=(10, 0),
            window_attn_skip_freq=[1, 0, 2, 0],  # Non-boolean values
        )

        # bool(1) = True, bool(0) = False, bool(2) = True
        expected_results = {
            1: True,  # bool(1) = True
            2: False,  # bool(0) = False
            3: True,  # bool(2) = True
            4: False,  # bool(0) = False
        }

        for layer_number, expected in expected_results.items():
            result = is_layer_window_attention(
                config.window_size, config.window_attn_skip_freq, layer_number
            )
            assert result == expected, f"Layer {layer_number}: expected {expected}, got {result}"

    def test_invalid_skip_frequency_type(self):
        """Test error handling for invalid window_attn_skip_freq types."""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=8,
            window_size=(10, 0),
            window_attn_skip_freq="invalid",  # String is invalid
        )

        with pytest.raises(ValueError) as exc_info:
            is_layer_window_attention(config.window_size, config.window_attn_skip_freq, 1)

        assert "Invalid `window_attn_skip_freq`" in str(exc_info.value)
        assert "str" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_edge_cases(self):
        """Test edge cases for is_layer_window_attention."""
        # Test with minimal configuration
        config = TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=8, window_size=(5, 0)
        )

        assert (
            is_layer_window_attention(config.window_size, config.window_attn_skip_freq, 1) == True
        )

        # Test with large layer numbers
        config = TransformerConfig(
            num_layers=100,
            hidden_size=64,
            num_attention_heads=8,
            window_size=(10, 0),
            window_attn_skip_freq=10,
        )

        # Layer 100 should not use window attention (100 % 10 == 0)
        assert (
            is_layer_window_attention(config.window_size, config.window_attn_skip_freq, 100)
            == False
        )
        # Layer 99 should use window attention (99 % 10 != 0)
        assert (
            is_layer_window_attention(config.window_size, config.window_attn_skip_freq, 99) == True
        )

    def test_window_size_different_formats(self):
        """Test with different window_size formats."""
        # Test with tuple
        config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=8, window_size=(10, 0)
        )

        assert (
            is_layer_window_attention(config.window_size, config.window_attn_skip_freq, 1) == True
        )

        # Test with single integer (if supported)
        try:
            config.window_size = 10
            result = is_layer_window_attention(config.window_size, config.window_attn_skip_freq, 1)
            # If this doesn't raise an error, it should return True
            assert result == True
        except (TypeError, ValueError):
            # Some implementations might not support single integer
            pass

    def test_layer_number_edge_cases(self):
        """Test with different layer number edge cases."""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=8,
            window_size=[10, 0],
            window_attn_skip_freq=2,
        )

        # Test layer 0 (though typically layers are 1-indexed)
        # 0 % 2 == 0, so should return False
        assert (
            is_layer_window_attention(config.window_size, config.window_attn_skip_freq, 0) == False
        )

        # Test negative layer numbers
        # -1 % 2 != 0 in Python, so should return True
        assert (
            is_layer_window_attention(config.window_size, config.window_attn_skip_freq, -1) == True
        )
        assert (
            is_layer_window_attention(config.window_size, config.window_attn_skip_freq, -2) == False
        )  # -2 % 2 == 0

    def test_comprehensive_scenario(self):
        """Test a comprehensive scenario with multiple configurations."""
        scenarios = [
            {
                "config": TransformerConfig(
                    num_layers=12,
                    hidden_size=64,
                    num_attention_heads=8,
                    window_size=[16, 0],
                    window_attn_skip_freq=4,
                ),
                "expected": {
                    1: True,
                    2: True,
                    3: True,
                    4: False,  # First group
                    5: True,
                    6: True,
                    7: True,
                    8: False,  # Second group
                    9: True,
                    10: True,
                    11: True,
                    12: False,  # Third group
                },
            },
            {
                "config": TransformerConfig(
                    num_layers=6,
                    hidden_size=64,
                    num_attention_heads=8,
                    window_size=[8, 8],
                    window_attn_skip_freq=[True, True, False, False, True, False],
                ),
                "expected": {1: True, 2: True, 3: False, 4: False, 5: True, 6: False},
            },
        ]

        for i, scenario in enumerate(scenarios):
            config = scenario["config"]
            expected = scenario["expected"]

            for layer_number, expected_result in expected.items():
                result = is_layer_window_attention(
                    config.window_size, config.window_attn_skip_freq, layer_number
                )
                assert (
                    result == expected_result
                ), f"Scenario {i+1}, Layer {layer_number}: expected {expected_result}, got {result}"


class TestIsLayerWindowAttentionIntegration:
    """Integration tests for is_layer_window_attention with other components."""

    def test_with_transformer_config_validation(self):
        """Test that function works with fully validated TransformerConfig objects."""
        config = TransformerConfig(
            num_layers=8,
            hidden_size=768,
            num_attention_heads=12,
            window_size=[128, 0],
            window_attn_skip_freq=2,
            use_cpu_initialization=True,
        )

        # Function should work with complete config
        for layer in range(1, 9):
            result = is_layer_window_attention(
                config.window_size, config.window_attn_skip_freq, layer
            )
            expected = layer % 2 != 0  # Skip every 2nd layer
            assert result == expected

    def test_performance_with_many_layers(self):
        """Test performance with a large number of layers."""
        config = TransformerConfig(
            num_layers=1000,
            hidden_size=64,
            num_attention_heads=8,
            window_size=[10, 0],
            window_attn_skip_freq=list(range(1000)),  # Large list
        )

        # This should complete without performance issues
        results = []
        for layer in range(1, 101):  # Test first 100 layers
            result = is_layer_window_attention(
                config.window_size, config.window_attn_skip_freq, layer
            )
            results.append(result)

        # Verify some results
        assert len(results) == 100
        assert isinstance(results[0], bool)
