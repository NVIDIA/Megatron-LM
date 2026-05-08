# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Unit tests for the GPT <-> Hybrid checkpoint conversion tool.

These tests validate:
- Hybrid layer pattern parsing
- Layer index mapping (GPT <-> Hybrid)
- State dict key renaming (final_layernorm <-> final_norm)
- Shared parameter copying (embeddings, output_layer)
- SSM parameter initialization shapes and dtypes
- Round-trip conversion: GPT -> Hybrid -> GPT preserves attention and MLP weights
- TP split dimension lookup
"""

import argparse
import math
import os
import sys
import tempfile
from collections import OrderedDict

import pytest
import torch

# Add the tools/checkpoint directory to the path so we can import the module
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tools', 'checkpoint')
)

from gpt_hybrid_conversion import (
    build_layer_index_mapping,
    convert_gpt_to_hybrid,
    convert_hybrid_to_gpt,
    get_layer_num_from_key,
    initialize_ssm_layer_params,
    is_attention_param,
    is_mlp_param,
    is_ssm_param,
    parse_hybrid_layer_pattern,
    replace_layer_num,
    validate_pattern_gpt_compatible,
    validate_source_args_gpt_compatible,
)

# ---------------------------------------------------------------------------
# Pattern parsing tests
# ---------------------------------------------------------------------------


class TestPatternParsing:
    def test_simple_pattern(self):
        result = parse_hybrid_layer_pattern("M*-M*-")
        assert result == ['M', '*', '-', 'M', '*', '-']

    def test_all_mamba(self):
        result = parse_hybrid_layer_pattern("MMMM")
        assert result == ['M', 'M', 'M', 'M']

    def test_all_attention(self):
        result = parse_hybrid_layer_pattern("****")
        assert result == ['*', '*', '*', '*']

    def test_with_mtp_separator(self):
        # Should strip MTP patterns (only main pattern)
        result = parse_hybrid_layer_pattern("M*-M*-/MM/MM")
        assert result == ['M', '*', '-', 'M', '*', '-']

    def test_with_pipe_separator(self):
        # Should strip pipeline stage separators
        result = parse_hybrid_layer_pattern("M*-|M*-")
        assert result == ['M', '*', '-', 'M', '*', '-']

    def test_with_both_separators(self):
        result = parse_hybrid_layer_pattern("M*-|M*-/MM/MM")
        assert result == ['M', '*', '-', 'M', '*', '-']

    def test_mixed_layers(self):
        result = parse_hybrid_layer_pattern("M*-EG")
        assert result == ['M', '*', '-', 'E', 'G']

    def test_invalid_symbol(self):
        with pytest.raises(ValueError, match="Invalid layer symbol"):
            parse_hybrid_layer_pattern("M*X")


# ---------------------------------------------------------------------------
# Layer index mapping tests
# ---------------------------------------------------------------------------


class TestLayerIndexMapping:
    def test_gpt_to_hybrid_basic(self):
        # Pattern: M*-M*- (2 attn at pos 1,4; 2 MLP at pos 2,5)
        layer_types = ['M', '*', '-', 'M', '*', '-']
        attn_map, mlp_map, ssm_indices = build_layer_index_mapping(layer_types, 'gpt-to-hybrid')
        # 2 GPT layers -> attn at [1,4], MLP at [2,5]
        assert attn_map == {0: 1, 1: 4}
        assert mlp_map == {0: 2, 1: 5}
        assert ssm_indices == [0, 3]

    def test_hybrid_to_gpt_basic(self):
        layer_types = ['M', '*', '-', 'M', '*', '-']
        attn_map, mlp_map, ssm_indices = build_layer_index_mapping(layer_types, 'hybrid-to-gpt')
        # attn at mamba layer 1 -> GPT layer 0, attn at 4 -> GPT layer 1
        assert attn_map == {1: 0, 4: 1}
        assert mlp_map == {2: 0, 5: 1}
        assert ssm_indices == [0, 3]

    def test_alternating_pattern(self):
        layer_types = ['*', '-', '*', '-', '*', '-']
        attn_map, mlp_map, ssm_indices = build_layer_index_mapping(layer_types, 'gpt-to-hybrid')
        assert attn_map == {0: 0, 1: 2, 2: 4}
        assert mlp_map == {0: 1, 1: 3, 2: 5}
        assert ssm_indices == []

    def test_mismatched_attn_mlp_count(self):
        # 2 attn but 1 MLP -> should raise
        layer_types = ['*', '*', '-', 'M']
        with pytest.raises(ValueError, match="must equal"):
            build_layer_index_mapping(layer_types, 'gpt-to-hybrid')

    def test_unknown_direction(self):
        with pytest.raises(ValueError, match="Unknown direction"):
            build_layer_index_mapping(['*', '-'], 'invalid')


# ---------------------------------------------------------------------------
# Key helper tests
# ---------------------------------------------------------------------------


class TestKeyHelpers:
    def test_get_layer_num(self):
        assert get_layer_num_from_key('decoder.layers.5.mlp.linear_fc1.weight') == 5
        assert get_layer_num_from_key('decoder.layers.0.self_attention.linear_qkv.weight') == 0
        assert get_layer_num_from_key('decoder.layers.99.mixer.A_log') == 99
        assert get_layer_num_from_key('embedding.word_embeddings.weight') is None

    def test_replace_layer_num(self):
        key = 'decoder.layers.3.mlp.linear_fc1.weight'
        assert replace_layer_num(key, 3, 7) == 'decoder.layers.7.mlp.linear_fc1.weight'

    def test_is_attention_param(self):
        assert is_attention_param('decoder.layers.0.self_attention.linear_qkv.weight')
        assert is_attention_param('decoder.layers.0.input_layernorm.weight')
        assert not is_attention_param('decoder.layers.0.mlp.linear_fc1.weight')
        assert not is_attention_param('decoder.layers.0.mixer.A_log')

    def test_is_mlp_param(self):
        assert is_mlp_param('decoder.layers.0.mlp.linear_fc1.weight')
        assert is_mlp_param('decoder.layers.0.pre_mlp_layernorm.weight')
        assert not is_mlp_param('decoder.layers.0.self_attention.linear_qkv.weight')

    def test_is_ssm_param(self):
        assert is_ssm_param('decoder.layers.0.mixer.A_log')
        assert is_ssm_param('decoder.layers.0.mixer.in_proj.weight')
        assert is_ssm_param('decoder.layers.0.mixer.conv1d.weight')
        assert is_ssm_param('decoder.layers.0.mixer.D')
        assert is_ssm_param('decoder.layers.0.mixer.dt_bias')
        assert is_ssm_param('decoder.layers.0.mixer.norm.weight')
        assert is_ssm_param('decoder.layers.0.mixer.out_proj.weight')
        assert not is_ssm_param('decoder.layers.0.mlp.linear_fc1.weight')
        assert not is_ssm_param('decoder.layers.0.self_attention.linear_qkv.weight')


# ---------------------------------------------------------------------------
# SSM initialization tests
# ---------------------------------------------------------------------------


class TestSSMInitialization:
    def test_shapes(self):
        d_model = 256
        d_inner = 512  # 2 * d_model
        d_state = 64
        n_groups = 4
        head_dim = 32
        n_heads = d_inner // head_dim  # 16
        d_conv = 4
        conv_dim = d_inner + 2 * n_groups * d_state

        params = initialize_ssm_layer_params(
            layer_idx=0,
            d_model=d_model,
            mamba_d_inner=d_inner,
            mamba_d_state=d_state,
            mamba2_n_groups=n_groups,
            mamba2_n_heads=n_heads,
            mamba_head_dim=head_dim,
            d_conv=d_conv,
            dtype=torch.float32,
        )

        prefix = 'decoder.layers.0.mixer.'

        # in_proj: [2*d_inner + 2*n_groups*d_state + n_heads, d_model]
        in_proj_out = 2 * d_inner + 2 * n_groups * d_state + n_heads
        assert params[prefix + 'in_proj.weight'].shape == (in_proj_out, d_model)

        # in_proj layer norm weight
        assert params[prefix + 'in_proj.layer_norm_weight'].shape == (d_model,)

        # conv1d: [conv_dim, 1, d_conv]
        assert params[prefix + 'conv1d.weight'].shape == (conv_dim, 1, d_conv)
        assert params[prefix + 'conv1d.bias'].shape == (conv_dim,)

        # A_log: [n_heads]
        assert params[prefix + 'A_log'].shape == (n_heads,)
        assert params[prefix + 'A_log'].dtype == torch.float32

        # D: [n_heads]
        assert params[prefix + 'D'].shape == (n_heads,)
        assert params[prefix + 'D'].dtype == torch.float32

        # dt_bias: [n_heads]
        assert params[prefix + 'dt_bias'].shape == (n_heads,)

        # norm: [d_inner]
        assert params[prefix + 'norm.weight'].shape == (d_inner,)

        # out_proj: [d_model, d_inner]
        assert params[prefix + 'out_proj.weight'].shape == (d_model, d_inner)

    def test_A_log_values(self):
        params = initialize_ssm_layer_params(
            layer_idx=0,
            d_model=64,
            mamba_d_inner=128,
            mamba_d_state=16,
            mamba2_n_groups=2,
            mamba2_n_heads=4,
            mamba_head_dim=32,
        )
        A_log = params['decoder.layers.0.mixer.A_log']
        # A was uniform in (1, 16), so A_log should be in (log(1), log(16)) = (0, 2.77)
        assert (A_log >= 0).all()
        assert (A_log <= math.log(16) + 0.01).all()

    def test_D_values(self):
        params = initialize_ssm_layer_params(
            layer_idx=0,
            d_model=64,
            mamba_d_inner=128,
            mamba_d_state=16,
            mamba2_n_groups=2,
            mamba2_n_heads=4,
            mamba_head_dim=32,
        )
        D = params['decoder.layers.0.mixer.D']
        assert torch.allclose(D, torch.ones_like(D))

    def test_conv1d_bias_zeros(self):
        params = initialize_ssm_layer_params(
            layer_idx=0,
            d_model=64,
            mamba_d_inner=128,
            mamba_d_state=16,
            mamba2_n_groups=2,
            mamba2_n_heads=4,
            mamba_head_dim=32,
        )
        bias = params['decoder.layers.0.mixer.conv1d.bias']
        assert torch.allclose(bias, torch.zeros_like(bias))

    def test_norm_weight_ones(self):
        params = initialize_ssm_layer_params(
            layer_idx=0,
            d_model=64,
            mamba_d_inner=128,
            mamba_d_state=16,
            mamba2_n_groups=2,
            mamba2_n_heads=4,
            mamba_head_dim=32,
        )
        norm = params['decoder.layers.0.mixer.norm.weight']
        assert torch.allclose(norm, torch.ones_like(norm))

    def test_layer_norm_weight_ones(self):
        params = initialize_ssm_layer_params(
            layer_idx=0,
            d_model=64,
            mamba_d_inner=128,
            mamba_d_state=16,
            mamba2_n_groups=2,
            mamba2_n_heads=4,
            mamba_head_dim=32,
        )
        ln = params['decoder.layers.0.mixer.in_proj.layer_norm_weight']
        assert torch.allclose(ln, torch.ones_like(ln))

    def test_different_layer_idx(self):
        params = initialize_ssm_layer_params(
            layer_idx=7,
            d_model=64,
            mamba_d_inner=128,
            mamba_d_state=16,
            mamba2_n_groups=2,
            mamba2_n_heads=4,
            mamba_head_dim=32,
        )
        assert 'decoder.layers.7.mixer.A_log' in params
        assert 'decoder.layers.0.mixer.A_log' not in params


# ---------------------------------------------------------------------------
# Synthetic GPT checkpoint builder
# ---------------------------------------------------------------------------


def make_synthetic_gpt_checkpoint(num_layers, d_model, dtype=torch.float32):
    """Create a minimal synthetic GPT state dict for testing."""
    state_dict = OrderedDict()

    # Embeddings
    state_dict['embedding.word_embeddings.weight'] = torch.randn(1000, d_model, dtype=dtype)

    # Transformer layers
    for i in range(num_layers):
        prefix = f'decoder.layers.{i}.'
        # Attention
        state_dict[prefix + 'input_layernorm.weight'] = torch.randn(d_model, dtype=dtype)
        state_dict[prefix + 'self_attention.linear_qkv.weight'] = torch.randn(
            3 * d_model, d_model, dtype=dtype
        )
        state_dict[prefix + 'self_attention.linear_proj.weight'] = torch.randn(
            d_model, d_model, dtype=dtype
        )
        # MLP
        state_dict[prefix + 'pre_mlp_layernorm.weight'] = torch.randn(d_model, dtype=dtype)
        state_dict[prefix + 'mlp.linear_fc1.weight'] = torch.randn(
            4 * d_model, d_model, dtype=dtype
        )
        state_dict[prefix + 'mlp.linear_fc2.weight'] = torch.randn(
            d_model, 4 * d_model, dtype=dtype
        )

    # Final norm
    state_dict['decoder.final_layernorm.weight'] = torch.randn(d_model, dtype=dtype)

    # Output layer
    state_dict['output_layer.weight'] = torch.randn(1000, d_model, dtype=dtype)

    return state_dict


# ---------------------------------------------------------------------------
# Full conversion tests
# ---------------------------------------------------------------------------


class TestGPTToHybridConversion:
    def setup_method(self):
        self.d_model = 64
        self.num_gpt_layers = 2
        self.pattern = "M*-M*-"  # 6 total: 2 SSM, 2 attn, 2 MLP
        self.gpt_state = make_synthetic_gpt_checkpoint(self.num_gpt_layers, self.d_model)
        self.args = argparse.Namespace(
            d_model=self.d_model,
            mamba_d_inner=self.d_model * 2,
            mamba_d_state=16,
            mamba2_n_groups=2,
            mamba2_n_heads=(self.d_model * 2) // 32,
            mamba2_head_dim=32,
            mamba_version=2,
            d_conv=4,
            init_method_std=0.02,
        )

    def test_shared_params_preserved(self):
        layer_types = parse_hybrid_layer_pattern(self.pattern)
        result = convert_gpt_to_hybrid(self.gpt_state, layer_types, self.args)

        # Embeddings should be identical
        assert torch.equal(
            result['embedding.word_embeddings.weight'],
            self.gpt_state['embedding.word_embeddings.weight'],
        )
        # Output layer
        assert torch.equal(result['output_layer.weight'], self.gpt_state['output_layer.weight'])

    def test_final_norm_renamed(self):
        layer_types = parse_hybrid_layer_pattern(self.pattern)
        result = convert_gpt_to_hybrid(self.gpt_state, layer_types, self.args)

        assert 'decoder.final_norm.weight' in result
        assert 'decoder.final_layernorm.weight' not in result
        assert torch.equal(
            result['decoder.final_norm.weight'], self.gpt_state['decoder.final_layernorm.weight']
        )

    def test_attention_params_mapped(self):
        layer_types = parse_hybrid_layer_pattern(self.pattern)
        result = convert_gpt_to_hybrid(self.gpt_state, layer_types, self.args)

        # GPT layer 0 attn -> Mamba layer 1 (first '*' in M*-M*-)
        assert torch.equal(
            result['decoder.layers.1.self_attention.linear_qkv.weight'],
            self.gpt_state['decoder.layers.0.self_attention.linear_qkv.weight'],
        )
        # GPT layer 1 attn -> Mamba layer 4 (second '*')
        assert torch.equal(
            result['decoder.layers.4.self_attention.linear_qkv.weight'],
            self.gpt_state['decoder.layers.1.self_attention.linear_qkv.weight'],
        )

    def test_mlp_params_mapped(self):
        layer_types = parse_hybrid_layer_pattern(self.pattern)
        result = convert_gpt_to_hybrid(self.gpt_state, layer_types, self.args)

        # GPT layer 0 MLP -> Mamba layer 2 (first '-')
        assert torch.equal(
            result['decoder.layers.2.mlp.linear_fc1.weight'],
            self.gpt_state['decoder.layers.0.mlp.linear_fc1.weight'],
        )
        # GPT layer 1 MLP -> Mamba layer 5 (second '-')
        assert torch.equal(
            result['decoder.layers.5.mlp.linear_fc2.weight'],
            self.gpt_state['decoder.layers.1.mlp.linear_fc2.weight'],
        )

    def test_ssm_layers_initialized(self):
        layer_types = parse_hybrid_layer_pattern(self.pattern)
        result = convert_gpt_to_hybrid(self.gpt_state, layer_types, self.args)

        # SSM layers at index 0 and 3
        for idx in [0, 3]:
            prefix = f'decoder.layers.{idx}.mixer.'
            assert prefix + 'A_log' in result
            assert prefix + 'D' in result
            assert prefix + 'dt_bias' in result
            assert prefix + 'conv1d.weight' in result
            assert prefix + 'conv1d.bias' in result
            assert prefix + 'in_proj.weight' in result
            assert prefix + 'norm.weight' in result
            assert prefix + 'out_proj.weight' in result

    def test_layer_count_mismatch_raises(self):
        # Pattern with 3 attn but only 2 GPT layers
        layer_types = parse_hybrid_layer_pattern("M*-*-*-")
        with pytest.raises(ValueError, match="layers"):
            convert_gpt_to_hybrid(self.gpt_state, layer_types, self.args)


class TestHybridToGPTConversion:
    def setup_method(self):
        self.d_model = 64
        self.pattern = "M*-M*-"
        self.args = argparse.Namespace(
            d_model=self.d_model,
            mamba_d_inner=self.d_model * 2,
            mamba_d_state=16,
            mamba2_n_groups=2,
            mamba2_n_heads=(self.d_model * 2) // 32,
            mamba2_head_dim=32,
            mamba_version=2,
            d_conv=4,
            init_method_std=0.02,
        )

    def _make_mamba_state(self):
        """Build a synthetic Mamba state dict matching pattern M*-M*-."""
        state_dict = OrderedDict()
        state_dict['embedding.word_embeddings.weight'] = torch.randn(1000, self.d_model)
        state_dict['output_layer.weight'] = torch.randn(1000, self.d_model)
        state_dict['decoder.final_norm.weight'] = torch.randn(self.d_model)

        layer_types = parse_hybrid_layer_pattern(self.pattern)
        d_inner = self.d_model * 2
        n_heads = self.args.mamba2_n_heads
        n_groups = self.args.mamba2_n_groups
        d_state = self.args.mamba_d_state

        for i, lt in enumerate(layer_types):
            prefix = f'decoder.layers.{i}.'
            if lt == 'M':
                # SSM params
                ssm = initialize_ssm_layer_params(
                    i, self.d_model, d_inner, d_state, n_groups, n_heads, self.args.mamba2_head_dim
                )
                state_dict.update(ssm)
            elif lt == '*':
                state_dict[prefix + 'input_layernorm.weight'] = torch.randn(self.d_model)
                state_dict[prefix + 'self_attention.linear_qkv.weight'] = torch.randn(
                    3 * self.d_model, self.d_model
                )
                state_dict[prefix + 'self_attention.linear_proj.weight'] = torch.randn(
                    self.d_model, self.d_model
                )
            elif lt == '-':
                state_dict[prefix + 'pre_mlp_layernorm.weight'] = torch.randn(self.d_model)
                state_dict[prefix + 'mlp.linear_fc1.weight'] = torch.randn(
                    4 * self.d_model, self.d_model
                )
                state_dict[prefix + 'mlp.linear_fc2.weight'] = torch.randn(
                    self.d_model, 4 * self.d_model
                )

        return state_dict

    def test_final_norm_renamed_back(self):
        mamba_state = self._make_mamba_state()
        layer_types = parse_hybrid_layer_pattern(self.pattern)
        result = convert_hybrid_to_gpt(mamba_state, layer_types, self.args)

        assert 'decoder.final_layernorm.weight' in result
        assert 'decoder.final_norm.weight' not in result

    def test_ssm_params_discarded(self):
        mamba_state = self._make_mamba_state()
        layer_types = parse_hybrid_layer_pattern(self.pattern)
        result = convert_hybrid_to_gpt(mamba_state, layer_types, self.args)

        # No SSM keys should remain
        for key in result:
            assert 'mixer.' not in key, f"SSM key not discarded: {key}"

    def test_attention_params_mapped(self):
        mamba_state = self._make_mamba_state()
        layer_types = parse_hybrid_layer_pattern(self.pattern)
        result = convert_hybrid_to_gpt(mamba_state, layer_types, self.args)

        # Mamba layer 1 (first *) -> GPT layer 0
        assert torch.equal(
            result['decoder.layers.0.self_attention.linear_qkv.weight'],
            mamba_state['decoder.layers.1.self_attention.linear_qkv.weight'],
        )
        # Mamba layer 4 (second *) -> GPT layer 1
        assert torch.equal(
            result['decoder.layers.1.self_attention.linear_qkv.weight'],
            mamba_state['decoder.layers.4.self_attention.linear_qkv.weight'],
        )

    def test_mlp_params_mapped(self):
        mamba_state = self._make_mamba_state()
        layer_types = parse_hybrid_layer_pattern(self.pattern)
        result = convert_hybrid_to_gpt(mamba_state, layer_types, self.args)

        # Mamba layer 2 (first -) -> GPT layer 0
        assert torch.equal(
            result['decoder.layers.0.mlp.linear_fc1.weight'],
            mamba_state['decoder.layers.2.mlp.linear_fc1.weight'],
        )

    def test_gpt_layer_count(self):
        mamba_state = self._make_mamba_state()
        layer_types = parse_hybrid_layer_pattern(self.pattern)
        result = convert_hybrid_to_gpt(mamba_state, layer_types, self.args)

        # Should have 2 GPT layers (layers 0 and 1)
        layer_nums = set()
        for key in result:
            lnum = get_layer_num_from_key(key)
            if lnum is not None:
                layer_nums.add(lnum)
        assert layer_nums == {0, 1}


# ---------------------------------------------------------------------------
# Round-trip test: GPT -> Hybrid -> GPT; using Mamba as the example below
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_gpt_hybrid_gpt_preserves_weights(self):
        """Converting GPT -> Hybrid -> GPT should preserve all attention & MLP weights."""
        d_model = 64
        num_layers = 2
        pattern = "M*-M*-"

        args = argparse.Namespace(
            d_model=d_model,
            mamba_d_inner=d_model * 2,
            mamba_d_state=16,
            mamba2_n_groups=2,
            mamba2_n_heads=(d_model * 2) // 32,
            mamba2_head_dim=32,
            mamba_version=2,
            d_conv=4,
            init_method_std=0.02,
        )

        original_gpt = make_synthetic_gpt_checkpoint(num_layers, d_model)
        layer_types = parse_hybrid_layer_pattern(pattern)

        # GPT -> Hybrid
        mamba_state = convert_gpt_to_hybrid(original_gpt, layer_types, args)

        # Hybrid -> GPT
        recovered_gpt = convert_hybrid_to_gpt(mamba_state, layer_types, args)

        # Check all original GPT keys are preserved
        for key in original_gpt:
            # final_layernorm is renamed in the round trip
            if 'final_layernorm' in key:
                continue
            assert key in recovered_gpt, f"Missing key after round-trip: {key}"
            assert torch.equal(
                original_gpt[key], recovered_gpt[key]
            ), f"Weight mismatch after round-trip for {key}"

        # Check final_layernorm was properly renamed back
        assert torch.equal(
            original_gpt['decoder.final_layernorm.weight'],
            recovered_gpt['decoder.final_layernorm.weight'],
        )

    def test_round_trip_different_pattern(self):
        """Test with a pattern that has more SSM layers."""
        d_model = 64
        num_layers = 3
        pattern = "M*-M*-M*-"

        args = argparse.Namespace(
            d_model=d_model,
            mamba_d_inner=d_model * 2,
            mamba_d_state=16,
            mamba2_n_groups=2,
            mamba2_n_heads=(d_model * 2) // 32,
            mamba2_head_dim=32,
            mamba_version=2,
            d_conv=4,
            init_method_std=0.02,
        )

        original_gpt = make_synthetic_gpt_checkpoint(num_layers, d_model)
        layer_types = parse_hybrid_layer_pattern(pattern)

        mamba_state = convert_gpt_to_hybrid(original_gpt, layer_types, args)
        recovered_gpt = convert_hybrid_to_gpt(mamba_state, layer_types, args)

        for key in original_gpt:
            if 'final_layernorm' in key:
                continue
            assert key in recovered_gpt, f"Missing key: {key}"
            assert torch.equal(original_gpt[key], recovered_gpt[key]), f"Mismatch for {key}"


# ---------------------------------------------------------------------------
# GPT compatibility whitelist tests
# ---------------------------------------------------------------------------


class TestPatternWhitelist:
    """validate_pattern_gpt_compatible rejects hybrid patterns GPTModel can't express."""

    def test_accepts_mamba_attn_mlp(self):
        # Standard hybrid with equal attn/MLP counts.
        layer_types = parse_hybrid_layer_pattern("M*-M*-M*-")
        validate_pattern_gpt_compatible(layer_types, 'gpt-to-hybrid')

    def test_accepts_pure_transformer_pattern(self):
        layer_types = parse_hybrid_layer_pattern("*-*-*-")
        validate_pattern_gpt_compatible(layer_types, 'hybrid-to-gpt')

    def test_accepts_pure_ssm_pattern(self):
        # Pure-SSM models have no attention/MLP, so trivially GPT-compatible
        # in the pattern sense (the GPT side would be empty).
        layer_types = parse_hybrid_layer_pattern("MMMM")
        validate_pattern_gpt_compatible(layer_types, 'gpt-to-hybrid')

    def test_accepts_moe_pattern(self):
        # MoE layers ('E') round-trip through the converter as long as every
        # MLP-bearing position is the same kind.
        layer_types = parse_hybrid_layer_pattern("M*EM*EM*E")
        validate_pattern_gpt_compatible(layer_types, 'gpt-to-hybrid')

    def test_accepts_pure_attn_moe_pattern(self):
        # No SSM, alternating attn/MoE — i.e. a Mixtral-like GPT.
        layer_types = parse_hybrid_layer_pattern("*E*E*E")
        validate_pattern_gpt_compatible(layer_types, 'hybrid-to-gpt')

    def test_rejects_mixed_dense_and_moe(self):
        # GPT layers must be uniform: '-' (dense) and 'E' (MoE) cannot both
        # appear in the same pattern.
        layer_types = parse_hybrid_layer_pattern("M*-M*E")
        with pytest.raises(ValueError, match="uniform"):
            validate_pattern_gpt_compatible(layer_types, 'gpt-to-hybrid')

    def test_rejects_gdn_symbol(self):
        layer_types = parse_hybrid_layer_pattern("G*-*-")
        with pytest.raises(ValueError, match="not GPT-compatible"):
            validate_pattern_gpt_compatible(layer_types, 'gpt-to-hybrid')

    def test_rejects_unequal_attn_mlp(self):
        layer_types = parse_hybrid_layer_pattern("M**-")  # 2 attn, 1 MLP
        with pytest.raises(ValueError, match="pair every attention"):
            validate_pattern_gpt_compatible(layer_types, 'gpt-to-hybrid')

    def test_unequal_attn_moe_also_rejected(self):
        # Same uniformity check, but with MoE — 2 attn, 1 MoE.
        layer_types = parse_hybrid_layer_pattern("M**E")
        with pytest.raises(ValueError, match="pair every attention"):
            validate_pattern_gpt_compatible(layer_types, 'gpt-to-hybrid')

    def test_error_lists_offending_symbols(self):
        # 'G' is still rejected; the error message should mention it.
        layer_types = parse_hybrid_layer_pattern("M*-G")
        with pytest.raises(ValueError) as exc:
            validate_pattern_gpt_compatible(layer_types, 'hybrid-to-gpt')
        assert 'G' in str(exc.value)


class TestSourceArgsWhitelist:
    """validate_source_args_gpt_compatible rejects source checkpoints with
    non-GPT-expressible features."""

    def _ok_args(self, **overrides):
        """Build a minimal args namespace that mimics a plain GPT/hybrid
        training run. Any GPT-incompatible flags default to their
        "off" value."""
        base = dict(
            num_moe_experts=None,
            moe_shared_expert_intermediate_size=None,
            moe_layer_freq=1,
            experimental_attention_variant=None,
            linear_attention_freq=None,
            heterogeneous_block_specs=False,
            heterogeneous_layers_config_path=None,
            heterogeneous_layers_config_encoded_json=None,
            multi_latent_attention=False,
            mtp_num_layers=None,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_accepts_plain_gpt_args(self):
        validate_source_args_gpt_compatible(self._ok_args(), 'gpt-to-hybrid')

    def test_none_args_is_noop(self):
        # Dist checkpoints sometimes have no cached args blob.
        validate_source_args_gpt_compatible(None, 'gpt-to-hybrid')

    def test_accepts_missing_optional_fields(self):
        # Older checkpoints may not have every field; the validator should
        # silently skip fields it doesn't find.
        minimal = argparse.Namespace(num_moe_experts=None)
        validate_source_args_gpt_compatible(minimal, 'hybrid-to-gpt')

    def test_accepts_moe_args(self):
        # MoE keys live under decoder.layers.<i>.mlp.* and round-trip as-is.
        validate_source_args_gpt_compatible(self._ok_args(num_moe_experts=8), 'gpt-to-hybrid')

    def test_accepts_shared_expert_args(self):
        # Shared experts also live under mlp.shared_experts.* and round-trip.
        validate_source_args_gpt_compatible(
            self._ok_args(num_moe_experts=8, moe_shared_expert_intermediate_size=4096),
            'gpt-to-hybrid',
        )

    def test_rejects_moe_layer_freq_list(self):
        # Heterogeneous interleaving (some dense, some MoE) breaks GPT uniformity.
        with pytest.raises(ValueError, match="interleaved"):
            validate_source_args_gpt_compatible(
                self._ok_args(moe_layer_freq=[1, 0, 1, 0]), 'gpt-to-hybrid'
            )

    def test_accepts_moe_layer_freq_1(self):
        validate_source_args_gpt_compatible(self._ok_args(moe_layer_freq=1), 'gpt-to-hybrid')

    def test_accepts_moe_layer_freq_all_ones_list(self):
        # An all-1s list is uniform (every layer is the same kind) and accepted.
        validate_source_args_gpt_compatible(
            self._ok_args(moe_layer_freq=[1, 1, 1, 1]), 'gpt-to-hybrid'
        )

    def test_rejects_experimental_attention(self):
        with pytest.raises(ValueError, match="experimental attention"):
            validate_source_args_gpt_compatible(
                self._ok_args(experimental_attention_variant='gated_delta_net'), 'gpt-to-hybrid'
            )

    def test_rejects_linear_attention(self):
        with pytest.raises(ValueError, match="linear attention"):
            validate_source_args_gpt_compatible(
                self._ok_args(linear_attention_freq=4), 'gpt-to-hybrid'
            )

    def test_rejects_heterogeneous_block_specs(self):
        with pytest.raises(ValueError, match="heterogeneous"):
            validate_source_args_gpt_compatible(
                self._ok_args(heterogeneous_block_specs=True), 'hybrid-to-gpt'
            )

    def test_rejects_heterogeneous_config_path(self):
        with pytest.raises(ValueError, match="heterogeneous"):
            validate_source_args_gpt_compatible(
                self._ok_args(heterogeneous_layers_config_path='/tmp/x.json'), 'gpt-to-hybrid'
            )

    def test_rejects_mla(self):
        with pytest.raises(ValueError, match="Multi-Latent"):
            validate_source_args_gpt_compatible(
                self._ok_args(multi_latent_attention=True), 'gpt-to-hybrid'
            )

    def test_rejects_mtp(self):
        with pytest.raises(ValueError, match="Multi-Token Prediction"):
            validate_source_args_gpt_compatible(self._ok_args(mtp_num_layers=2), 'gpt-to-hybrid')

    def test_reports_multiple_reasons(self):
        # Both heterogeneous moe_layer_freq and MLA set — both should be reported.
        with pytest.raises(ValueError) as exc:
            validate_source_args_gpt_compatible(
                self._ok_args(moe_layer_freq=[1, 0], multi_latent_attention=True), 'gpt-to-hybrid'
            )
        msg = str(exc.value)
        assert 'interleaved' in msg
        assert 'Multi-Latent' in msg
