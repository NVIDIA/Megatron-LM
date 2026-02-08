# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for Maximal Update Parameterization (MuP) implementation.

These tests verify that MuP is correctly implemented in Megatron-LM:
1. Config validation and width_mult computation
2. Initialization scaling
3. Attention scaling
4. LR override computation
"""

import math

import pytest
import torch

from megatron.core.optimizer import get_mup_config_overrides
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer.multi_token_prediction import process_mtp_loss
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal, mup_scaled_init_method_normal


class TestMuPConfigValidation:
    """Tests for MuP config validation and width_mult computation."""

    def test_mup_defaults_base_hidden_size(self):
        """use_mup without base_hidden_size defaults to hidden_size (width_mult=1.0)."""
        config = TransformerConfig(
            hidden_size=512,
            num_layers=4,
            num_attention_heads=8,
            use_mup=True,
            # mup_base_hidden_size not set - should default to hidden_size
        )
        assert config.mup_base_hidden_size == 512
        assert config.mup_width_mult == 1.0

    def test_mup_width_mult_calculation(self):
        """width_mult = hidden_size / base_hidden_size."""
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=4,
            num_attention_heads=16,
            use_mup=True,
            mup_base_hidden_size=256,
        )
        assert config.mup_width_mult == 4.0

    def test_mup_width_mult_fractional(self):
        """width_mult can be fractional (smaller than base)."""
        config = TransformerConfig(
            hidden_size=128,
            num_layers=4,
            num_attention_heads=2,
            use_mup=True,
            mup_base_hidden_size=256,
        )
        assert config.mup_width_mult == 0.5

    def test_mup_backward_compatible(self):
        """Default config unchanged when MuP disabled."""
        config = TransformerConfig(hidden_size=512, num_layers=4, num_attention_heads=8)
        assert config.use_mup is False
        assert config.mup_width_mult == 1.0
        assert config.mup_base_hidden_size is None

    def test_mup_base_hidden_size_must_be_positive(self):
        """mup_base_hidden_size must be positive."""
        with pytest.raises(AssertionError) as exc_info:
            TransformerConfig(
                hidden_size=512,
                num_layers=4,
                num_attention_heads=8,
                use_mup=True,
                mup_base_hidden_size=0,
            )
        assert "positive" in str(exc_info.value).lower()


class TestMuPInitMethods:
    """Tests for MuP initialization methods."""

    def test_mup_init_variance_scaling(self):
        """Init std scales as 1/sqrt(width_mult)."""
        base_sigma = 0.02
        width_mult = 4.0

        init_fn = init_method_normal(base_sigma / math.sqrt(width_mult))
        tensor = torch.empty(1000, 1000)
        init_fn(tensor)

        expected_std = base_sigma / math.sqrt(width_mult)  # 0.02/sqrt(4) = 0.01
        actual_std = tensor.std().item()

        # Allow 5% tolerance for statistical variance
        assert (
            abs(actual_std - expected_std) < 0.002
        ), f"Expected std ~{expected_std:.4f}, got {actual_std:.4f}"

    def test_mup_scaled_init_variance(self):
        """MuP scaled init combines depth and width scaling."""
        sigma = 0.02
        num_layers = 8
        width_mult = 4.0

        init_fn = mup_scaled_init_method_normal(sigma, num_layers, width_mult)
        tensor = torch.empty(1000, 1000)
        init_fn(tensor)

        # std = sigma / (sqrt(2*num_layers) * sqrt(width_mult))
        expected_std = sigma / (math.sqrt(2 * num_layers) * math.sqrt(width_mult))
        actual_std = tensor.std().item()

        # Allow 10% tolerance
        assert (
            abs(actual_std - expected_std) < expected_std * 0.1
        ), f"Expected std ~{expected_std:.6f}, got {actual_std:.6f}"


class TestMuPAttentionScaling:
    """Tests for MuP attention scaling."""

    def test_mup_attention_scale_power_1(self):
        """MuP uses 1/d instead of 1/sqrt(d) when power=1.0."""
        config = TransformerConfig(
            hidden_size=512,
            num_layers=4,
            num_attention_heads=8,
            use_mup=True,
            mup_base_hidden_size=128,
            mup_attn_scale_power=1.0,  # MuP default
        )
        kv_channels = config.kv_channels  # 512 / 8 = 64

        expected_scale = 1.0 / kv_channels  # 1/64 = 0.015625
        assert config.softmax_scale == expected_scale

    def test_standard_attention_scale_power_05(self):
        """Standard uses 1/sqrt(d) when power=0.5."""
        config = TransformerConfig(
            hidden_size=512,
            num_layers=4,
            num_attention_heads=8,
            use_mup=True,
            mup_base_hidden_size=128,
            mup_attn_scale_power=0.5,  # Standard default
        )
        kv_channels = config.kv_channels  # 64

        expected_scale = 1.0 / math.sqrt(kv_channels)  # 1/8 = 0.125
        assert abs(config.softmax_scale - expected_scale) < 1e-6

    def test_attention_scale_not_set_when_disabled(self):
        """softmax_scale should not be auto-set when use_mup=False."""
        config = TransformerConfig(
            hidden_size=512, num_layers=4, num_attention_heads=8, use_mup=False  # MuP disabled
        )
        # softmax_scale defaults to None when MuP is disabled
        # (actual scaling is done in the attention layer)
        assert config.softmax_scale is None


class TestMuPLRScaling:
    """Tests for MuP learning rate scaling."""

    def test_mup_lr_override_computation(self):
        """Hidden LR scales as 1/width_mult."""
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5)
        width_mult = 4.0

        overrides = get_mup_config_overrides(optimizer_config, width_mult)

        # Should have one override for hidden layers
        assert len(overrides) == 1

        # Get the override values
        for param_key, override in overrides.items():
            expected_max_lr = 1e-3 / width_mult  # 2.5e-4
            expected_min_lr = 1e-5 / width_mult  # 2.5e-6

            assert abs(override['max_lr'] - expected_max_lr) < 1e-10
            assert abs(override['min_lr'] - expected_min_lr) < 1e-10

    def test_mup_lr_no_scaling_at_unity(self):
        """No LR scaling when width_mult=1.0."""
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5)
        width_mult = 1.0

        overrides = get_mup_config_overrides(optimizer_config, width_mult)

        # Should return empty dict when no scaling needed
        assert len(overrides) == 0

    def test_mup_lr_override_has_correct_predicate(self):
        """LR override predicate correctly identifies hidden vs embedding params.

        Per MuP paper and Microsoft's mup library:
        - Embedding layer: base LR (fan_in is vocab, finite dimension)
        - Hidden layers: scaled LR (fan_in is hidden, infinite dimension)
        - Output layer: scaled LR (fan_in is hidden, infinite dimension)
        """
        optimizer_config = OptimizerConfig(lr=1e-3)
        width_mult = 4.0

        overrides = get_mup_config_overrides(optimizer_config, width_mult)

        # Check the predicate is set and works correctly
        for param_key, override in overrides.items():
            assert param_key.with_name_predicate is not None
            predicate_fn = param_key.with_name_predicate.fn

            # Create mock parameters
            hidden_param = torch.nn.Parameter(torch.zeros(10, 10))

            # Hidden param with hidden-layer name should match (scaled LR)
            assert predicate_fn(hidden_param, 'decoder.layer.0.weight') is True

            # Output layer should match (scaled LR) - fan_in is hidden dimension
            assert predicate_fn(hidden_param, 'output_layer.weight') is True

            # Embedding layer should NOT match (base LR) when attribute is set.
            embedding_param = torch.nn.Parameter(torch.zeros(10, 10))
            embedding_param.is_embedding_parameter = True
            assert predicate_fn(embedding_param, 'decoder.layer.0.weight') is False

            # Backward-compatible fallback for older modules without the attribute.
            assert predicate_fn(hidden_param, 'embedding.word_embeddings.weight') is False


class TestMuPConfigIntegration:
    """Integration tests for MuP config with init methods."""

    def test_mup_output_layer_init(self):
        """Output layer init should also scale with MuP."""
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=8,
            num_attention_heads=16,
            use_mup=True,
            mup_base_hidden_size=256,
        )

        tensor = torch.empty(1000, 1000)
        config.output_layer_init_method(tensor)

        # Expected: sigma / (sqrt(2*L) * sqrt(m)) = 0.02 / (sqrt(16) * sqrt(4))
        expected_std = config.init_method_std / (
            math.sqrt(2 * config.num_layers) * math.sqrt(config.mup_width_mult)
        )
        actual_std = tensor.std().item()

        assert abs(actual_std - expected_std) < expected_std * 0.15


class TestMuPOptimizerTypeHandling:
    """Tests for MuP optimizer-specific LR scaling behavior."""

    def test_sgd_no_lr_scaling(self):
        """SGD optimizer should NOT scale LR with width (case insensitive)."""
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5)
        width_mult = 4.0

        # Test various case variants
        for sgd_variant in ['sgd', 'SGD', 'Sgd']:
            overrides = get_mup_config_overrides(
                optimizer_config, width_mult, optimizer_type=sgd_variant
            )
            # SGD should return empty overrides - no LR scaling
            assert len(overrides) == 0, f"SGD variant '{sgd_variant}' should not scale LR"

    def test_adam_scales_lr_by_default(self):
        """Adam optimizer should scale LR; default optimizer_type is adam."""
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5)
        width_mult = 4.0

        # Explicit adam
        overrides_explicit = get_mup_config_overrides(
            optimizer_config, width_mult, optimizer_type='adam'
        )
        assert len(overrides_explicit) == 1

        # Default (should behave like adam for backward compat)
        overrides_default = get_mup_config_overrides(optimizer_config, width_mult)
        assert len(overrides_default) == 1

        for param_key, override in overrides_explicit.items():
            expected_max_lr = 1e-3 / width_mult  # 2.5e-4
            assert abs(override['max_lr'] - expected_max_lr) < 1e-10


class TestMuPMTPLossScaling:
    """Tests for MuP scaling integration with MTP loss processing."""

    def test_process_mtp_loss_applies_scale_hook(self):
        config = TransformerConfig(
            hidden_size=8, num_layers=2, num_attention_heads=2, mtp_num_layers=1
        )
        hidden_states = torch.ones(2, 1, 4)
        labels = torch.ones(1, 4, dtype=torch.long)
        loss_mask = torch.ones_like(labels, dtype=torch.float32)
        observed_logits_mean = {'value': None}

        def output_layer(hidden, weight=None, runtime_gather_output=None):
            return hidden.clone(), None

        def scale_logits_fn(logits):
            return logits * 3.0

        def compute_language_model_loss(mtp_labels, mtp_logits):
            observed_logits_mean['value'] = mtp_logits.mean().item()
            return torch.ones_like(mtp_labels, dtype=mtp_logits.dtype)

        process_mtp_loss(
            hidden_states=hidden_states,
            labels=labels,
            loss_mask=loss_mask,
            output_layer=output_layer,
            output_weight=None,
            runtime_gather_output=None,
            is_training=False,
            compute_language_model_loss=compute_language_model_loss,
            config=config,
            cp_group=None,
            packed_seq_params=None,
            scale_logits_fn=scale_logits_fn,
        )

        assert observed_logits_mean['value'] == pytest.approx(3.0)
