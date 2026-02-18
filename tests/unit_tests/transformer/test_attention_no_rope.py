# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestParallelAttentionWithNoRope:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        # use BF16 and a large enough hidden size to enable FlashAttention
        self.transformer_config = TransformerConfig(
            num_layers=8,  # Using 8 layers to test patterns like [0,0,0,1,0,0,0,1]
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
            flash_decode=False,  # Ensure flash_decode is off to test RoPE skipping
        )
        self.parallel_attention = SelfAttention(
            self.transformer_config,
            get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_integer_no_rope_freq_pattern(self):
        """Test that integer no_rope value is correctly converted to pattern."""
        config = self.transformer_config
        config.no_rope_freq = 4  # Should convert to [0,0,0,1,0,0,0,1]
        config.__post_init__()

        # Verify the pattern conversion
        assert isinstance(config.no_rope_freq, list)
        assert len(config.no_rope_freq) == config.num_layers
        assert config.no_rope_freq == [0, 0, 0, 1, 0, 0, 0, 1]

    def test_custom_no_rope_pattern(self):
        """Test custom no_rope pattern."""
        config = self.transformer_config
        config.no_rope_freq = [0, 1, 0, 1, 0, 1, 0, 1]  # Custom pattern
        config.__post_init__()

        # Verify the pattern is preserved
        assert isinstance(config.no_rope_freq, list)
        assert len(config.no_rope_freq) == config.num_layers
        assert config.no_rope_freq == [0, 1, 0, 1, 0, 1, 0, 1]

    def test_gpu_forward_with_no_rope(self):
        """Test forward pass with no_rope pattern."""
        config = self.parallel_attention.config
        config.no_rope_freq = 4  # Use pattern [0,0,0,1,0,0,0,1]
        config.__post_init__()  # Ensure pattern is converted

        sequence_length = 32
        micro_batch_size = 1

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None

        # Create rotary position embeddings
        # Shape: [seq_len, 1, 1, kv_channels]
        rotary_pos_emb = torch.randn(
            sequence_length, 1, 1, self.parallel_attention.config.kv_channels
        ).cuda()

        # For self-attention, rotary_pos_emb needs to be a tuple of (q_pos_emb, k_pos_emb)
        rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)

        # Test with layer 3 which should skip RoPE
        self.parallel_attention.layer_number = 3
        # Run forward pass without RoPE
        output_without_rope, _ = self.parallel_attention(
            hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb
        )

        # Test with layer 0 which should NOT skip RoPE
        self.parallel_attention.layer_number = 0
        # Run forward pass with RoPE (but should be skipped for this layer)
        output_with_rope, bias = self.parallel_attention(
            hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb
        )

        # Verify RoPE was skipped for this layer
        # If RoPE was skipped, outputs should be the same
        assert not torch.allclose(
            output_without_rope, output_with_rope
        ), "Outputs are expected to be different."

        # Verify output shapes
        assert config.recompute_granularity is None
        assert output_with_rope.shape[0] == sequence_length
        assert output_with_rope.shape[1] == micro_batch_size
        assert output_with_rope.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    def test_invalid_no_rope_freq_pattern(self):
        """Test invalid no_rope patterns raise appropriate errors."""
        config = self.transformer_config

        # Test invalid integer pattern
        with pytest.raises(AssertionError):
            config.no_rope_freq = 3  # Not divisible by num_layers=8
            config.__post_init__()

        # Test invalid list pattern
        with pytest.raises(AssertionError):
            config.no_rope_freq = [0, 1, 0, 1]  # Wrong length
            config.__post_init__()

    def test_gpu_forward_no_rope_freq_not_specified(self):
        """Test forward pass with no_rope pattern not provided."""
        config = self.parallel_attention.config
        config.no_rope_freq = None
        config.__post_init__()  # Ensure pattern is converted

        sequence_length = 32
        micro_batch_size = 1

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None

        # Create rotary position embeddings
        # Shape: [seq_len, 1, 1, kv_channels]
        rotary_pos_emb = torch.randn(
            sequence_length, 1, 1, self.parallel_attention.config.kv_channels
        ).cuda()

        # For self-attention, rotary_pos_emb needs to be a tuple of (q_pos_emb, k_pos_emb)
        rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)
        # Run forward pass
        output, bias = self.parallel_attention(
            hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb
        )
        # Verify output shapes
        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    def test_checkpointed_gpu_forward(self):
        """Test checkpointed forward pass with no_rope pattern."""
        transformer_config = self.transformer_config
        transformer_config.recompute_granularity = 'selective'
        transformer_config.no_rope_freq = 4  # Use pattern [0,0,0,1,0,0,0,1]
        transformer_config.__post_init__()

        checkpointed_parallel_attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        config = checkpointed_parallel_attention.config

        sequence_length = 32
        micro_batch_size = 1

        checkpointed_parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, checkpointed_parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None
        rotary_pos_emb = torch.ones(
            sequence_length, 1, 1, checkpointed_parallel_attention.config.kv_channels
        ).cuda()

        output, bias = checkpointed_parallel_attention(
            hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb
        )

        assert config.recompute_granularity == 'selective'
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    def test_flash_decode_with_no_rope_freq(self):
        """Test that flash_decode cannot be used with no_rope."""
        config = self.transformer_config
        config.flash_decode = True
        config.no_rope_freq = 4  # Use pattern [0,0,0,1,0,0,0,1]

        # Verify that setting both flash_decode and no_rope raises an assertion error
        with pytest.raises(AssertionError, match="flash_decode cannot be used with no_rope"):
            config.__post_init__()
