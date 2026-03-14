# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for softmax_scale config propagation.

Verifies that TransformerConfig.softmax_scale is properly considered
in both Attention.flash_decode_and_prefill and MLASelfAttention.__init__.
"""

import math
from unittest.mock import patch

import pytest
import torch

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.common.embeddings import _yarn_get_mscale
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _get_mla_submodules():
    submodules = get_gpt_layer_with_transformer_engine_submodules(
        multi_latent_attention=True
    ).self_attention.submodules
    assert isinstance(submodules, MLASelfAttentionSubmodules)
    return submodules


class TestAttentionSoftmaxScale:
    """Tests that flash_decode_and_prefill respects config.softmax_scale."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
        if not torch.cuda.is_available():
            pytest.skip("GPU required")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def _make_attention(self, softmax_scale=None):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            softmax_scale=softmax_scale,
        )
        submodules = get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules
        attn = SelfAttention(config, submodules, layer_number=1)
        attn.eval()
        attn.cuda()
        return attn

    def _call_and_capture_softmax_scale(self, attn):
        """Call flash_decode_and_prefill with HAVE_FA3=False, capture softmax_scale from FA2."""
        head_dim = attn.config.kv_channels
        num_heads = attn.config.num_attention_heads
        # q shape: [total_tokens, num_heads, 1, head_dim] — squeezed to [total, heads, head_dim]
        q = torch.randn(2, num_heads, 1, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(2, num_heads, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(2, num_heads, head_dim, device='cuda', dtype=torch.float16)
        cu_seqlens_q = torch.tensor([0, 1, 2], device='cuda', dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, 1, 2], device='cuda', dtype=torch.int32)
        seqlens_k = torch.tensor([1, 1], device='cuda', dtype=torch.int32)
        block_table = torch.zeros(2, 1, device='cuda', dtype=torch.int32)

        captured = {}

        def fake_fa_varlen(*args, **kwargs):
            captured['softmax_scale'] = kwargs.get('softmax_scale')
            return torch.randn(2, num_heads, head_dim, device='cuda', dtype=torch.float16)

        with (
            patch('megatron.core.transformer.attention.HAVE_FA3', False),
            patch('megatron.core.transformer.attention.flash_attn_varlen_func', fake_fa_varlen),
        ):
            attn.batch_invariant_mode = False
            attn.flash_decode_and_prefill(
                q,
                k,
                v,
                max_seqlen_q=1,
                max_seqlen_k=1,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqlens_k=seqlens_k,
                block_table=block_table,
                is_decode_only=False,
            )

        return captured['softmax_scale']

    def test_config_softmax_scale_used(self):
        """When config.softmax_scale is set, flash_decode_and_prefill should use it."""
        attn = self._make_attention(softmax_scale=0.042)
        assert not hasattr(attn, 'softmax_scale')
        scale = self._call_and_capture_softmax_scale(attn)
        assert scale == 0.042

    def test_default_softmax_scale(self):
        """When config.softmax_scale is None and no instance attr, use 1/sqrt(head_dim)."""
        attn = self._make_attention(softmax_scale=None)
        assert not hasattr(attn, 'softmax_scale')
        scale = self._call_and_capture_softmax_scale(attn)
        head_dim = attn.config.kv_channels
        assert scale == pytest.approx(head_dim**-0.5)

    def test_instance_attr_takes_precedence(self):
        """When self.softmax_scale is set, it takes precedence over config.softmax_scale."""
        attn = self._make_attention(softmax_scale=0.042)
        attn.softmax_scale = 0.123
        scale = self._call_and_capture_softmax_scale(attn)
        assert scale == 0.123


class TestMLASoftmaxScale:
    """Tests that MLASelfAttention.__init__ respects config.softmax_scale."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
        if not torch.cuda.is_available():
            pytest.skip("GPU required")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def _make_mla(self, softmax_scale=None, rotary_scaling_factor=40.0, mscale_all_dim=0.0):
        config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rope_type='yarn',
            rotary_base=10000,
            original_max_position_embeddings=32,
            softmax_scale=softmax_scale,
            rotary_scaling_factor=rotary_scaling_factor,
            mscale_all_dim=mscale_all_dim,
        )
        attention = MLASelfAttention(
            config, _get_mla_submodules(), layer_number=1, attn_mask_type=AttnMaskType.causal
        )
        return config, attention

    def test_config_softmax_scale_applied(self):
        """When config.softmax_scale is set, MLA uses mscale^2 * config.softmax_scale."""
        config, attention = self._make_mla(softmax_scale=0.042)
        mscale = _yarn_get_mscale(config.rotary_scaling_factor, config.mscale_all_dim)
        expected = mscale * mscale * 0.042
        assert attention.softmax_scale == pytest.approx(expected)

    def test_default_softmax_scale(self):
        """When config.softmax_scale is None, MLA uses mscale^2 / sqrt(q_head_dim)."""
        config, attention = self._make_mla(softmax_scale=None)
        q_head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim
        mscale = _yarn_get_mscale(config.rotary_scaling_factor, config.mscale_all_dim)
        expected = mscale * mscale * (1 / math.sqrt(q_head_dim))
        assert attention.softmax_scale == pytest.approx(expected)

    def test_config_softmax_scale_with_yarn_scaling(self):
        """With rotary_scaling_factor > 1, mscale != 1, and config.softmax_scale is applied (not None)."""
        config, attention = self._make_mla(softmax_scale=0.042, rotary_scaling_factor=4.0, mscale_all_dim=1.0)
        mscale = _yarn_get_mscale(config.rotary_scaling_factor, config.mscale_all_dim)
        assert mscale != 1.0, "mscale should be nontrivial with scaling_factor > 1 and mscale_all_dim != 0"
        expected = mscale * mscale * 0.042
        assert attention.softmax_scale == pytest.approx(expected)

    def test_config_softmax_scale_with_trivial_scaling(self):
        """With rotary_scaling_factor and mscale = 0."""
        config, attention = self._make_mla(softmax_scale=0.042, rotary_scaling_factor=4.0)
        mscale = _yarn_get_mscale(config.rotary_scaling_factor, config.mscale_all_dim)
        assert mscale == 1.0, "mscale should be trivial with mscale_all_dim == 0"
        expected = mscale * mscale * 0.042
        assert attention.softmax_scale == pytest.approx(expected)

    def test_softmax_scale_propagated_to_core_attention(self):
        """The computed softmax_scale should be passed to core_attention."""
        _, attention = self._make_mla(softmax_scale=0.042)
        if isinstance(attention.core_attention, TEDotProductAttention):
            assert attention.core_attention.flash_attention.softmax_scale == attention.softmax_scale
            assert attention.core_attention.fused_attention.softmax_scale == attention.softmax_scale
            assert attention.core_attention.unfused_attention.softmax_scale == attention.softmax_scale
        else:
            assert attention.core_attention.softmax_scale == attention.softmax_scale
