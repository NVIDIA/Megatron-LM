# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Tests for softmax_scale config propagation.

Verifies that TransformerConfig.softmax_scale is properly considered in every code path that
consumes a softmax scale:
- MLASelfAttention.__init__ (mscale^2 * config.softmax_scale),
- Attention.get_softmax_scale resolution order,
- Attention.flash_decode (FA2 kvcache),
- Attention.flash_decode_and_prefill for all kernel backends (prefill: FA2/FA3/FA4 varlen, decode:
  FA2/FA3 kvcache, FA4 varlen, FlashMLA).

FlashAttention kernels are mocked so the tests do not require any specific FA generation to be
installed and stay small-scale.
"""

import contextlib
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

_ATTN_MODULE = 'megatron.core.transformer.attention'


def _get_mla_submodules():
    submodules = get_gpt_layer_with_transformer_engine_submodules(
        multi_latent_attention=True
    ).self_attention.submodules
    assert isinstance(submodules, MLASelfAttentionSubmodules)
    return submodules


def _make_self_attention(softmax_scale=None):
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


def _make_mla(softmax_scale=None, rotary_scaling_factor=40.0, mscale_all_dim=0.0):
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
    attention.eval()
    return config, attention


def _prefill_inputs(num_heads, head_dim):
    """Inputs for flash_decode_and_prefill with two 1-token prefills."""
    device, dtype = 'cuda', torch.float16
    return dict(
        # q: [total_tokens, 1, num_heads, head_dim] — squeezed to [total, heads, head_dim]
        q=torch.randn(2, 1, num_heads, head_dim, device=device, dtype=dtype),
        k=torch.randn(2, num_heads, head_dim, device=device, dtype=dtype),
        v=torch.randn(2, num_heads, head_dim, device=device, dtype=dtype),
        cu_seqlens_q=torch.tensor([0, 1, 2], device=device, dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 1, 2], device=device, dtype=torch.int32),
        seqlens_k=torch.tensor([1, 1], device=device, dtype=torch.int32),
        block_table=torch.zeros(2, 1, device=device, dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_k=1,
    )


def _decode_inputs(num_heads, head_dim, num_requests=2, cache_len=8):
    """Inputs for flash_decode_and_prefill with `num_requests` 1-token decodes."""
    device, dtype = 'cuda', torch.float16
    return dict(
        # q: [num_requests * tokens_per_request, 1, num_heads, head_dim]
        q=torch.randn(num_requests, 1, num_heads, head_dim, device=device, dtype=dtype),
        k=torch.randn(num_requests, cache_len, num_heads, head_dim, device=device, dtype=dtype),
        v=torch.randn(num_requests, cache_len, num_heads, head_dim, device=device, dtype=dtype),
        cu_seqlens_q=torch.tensor([0, 1, 2], device=device, dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, cache_len, 2 * cache_len], device=device, dtype=torch.int32),
        seqlens_k=torch.full((num_requests,), cache_len, device=device, dtype=torch.int32),
        block_table=torch.zeros(num_requests, 1, device=device, dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_k=cache_len,
    )


def _capture_factory(captured, output, with_lse=False):
    """Return a fake flash-attention kernel that records `softmax_scale`.

    Returns `output` (plus a `None` lse element if `with_lse`), matching the `(out, lse)` tuple
    expected by the FA4/FlashMLA call sites.
    """

    def fake_kernel(*args, **kwargs):
        captured['softmax_scale'] = kwargs.get('softmax_scale')
        return (output, None) if with_lse else output

    return fake_kernel


class _GpuTestFixtureBase:
    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
        if not torch.cuda.is_available():
            pytest.skip("GPU required")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()


class TestAttentionSoftmaxScale(_GpuTestFixtureBase):
    """Tests that the FA2 prefill path of flash_decode_and_prefill respects softmax_scale."""

    def _call_and_capture_softmax_scale(self, attn):
        """Call flash_decode_and_prefill in prefill mode, capture softmax_scale from FA2."""
        head_dim = attn.config.kv_channels
        num_heads = attn.config.num_attention_heads
        inputs = _prefill_inputs(num_heads, head_dim)
        captured = {}

        fake_fa_varlen = _capture_factory(
            captured, torch.randn(2, num_heads, head_dim, device='cuda', dtype=torch.float16)
        )

        with (
            patch(f'{_ATTN_MODULE}.HAVE_FA4', False),
            patch(f'{_ATTN_MODULE}.HAVE_FA3', False),
            patch(f'{_ATTN_MODULE}.flash_attn_varlen_func', fake_fa_varlen),
        ):
            attn.batch_invariant_mode = False
            attn.flash_decode_and_prefill(**inputs, is_decode_only=False)

        return captured['softmax_scale']

    def test_config_softmax_scale_used(self):
        """When config.softmax_scale is set, flash_decode_and_prefill should use it."""
        attn = _make_self_attention(softmax_scale=0.042)
        assert not hasattr(attn, 'softmax_scale')
        scale = self._call_and_capture_softmax_scale(attn)
        assert scale == 0.042

    def test_default_softmax_scale(self):
        """When config.softmax_scale is None and no instance attr, use 1/sqrt(head_dim)."""
        attn = _make_self_attention(softmax_scale=None)
        assert not hasattr(attn, 'softmax_scale')
        scale = self._call_and_capture_softmax_scale(attn)
        head_dim = attn.config.kv_channels
        assert scale == pytest.approx(head_dim**-0.5)

    def test_instance_attr_takes_precedence(self):
        """When self.softmax_scale is set, it takes precedence over config.softmax_scale."""
        attn = _make_self_attention(softmax_scale=0.042)
        attn.softmax_scale = 0.123
        scale = self._call_and_capture_softmax_scale(attn)
        assert scale == 0.123


class TestPrefillKernelsSoftmaxScale(_GpuTestFixtureBase):
    """Tests that the FA3/FA4 prefill paths of flash_decode_and_prefill respect config.softmax_scale
    (the kernels are mocked, no FA3/FA4 install needed).
    """

    def test_prefill_fa4_uses_config_softmax_scale(self):
        attn = _make_self_attention(softmax_scale=0.042)
        attn.flash_attention_version = 4  # Auto selection depends on the device.
        head_dim = attn.config.kv_channels
        num_heads = attn.config.num_attention_heads
        captured = {}

        fake_fa4 = _capture_factory(
            captured,
            torch.randn(2, num_heads, head_dim, device='cuda', dtype=torch.float16),
            with_lse=True,
        )

        with (
            patch(f'{_ATTN_MODULE}.HAVE_FA4', True),
            patch(f'{_ATTN_MODULE}.flash_attn4_varlen_func', fake_fa4),
        ):
            attn.flash_decode_and_prefill(
                **_prefill_inputs(num_heads, head_dim), is_decode_only=False
            )

        assert captured['softmax_scale'] == 0.042

    def test_prefill_fa3_uses_config_softmax_scale(self):
        attn = _make_self_attention(softmax_scale=0.042)
        attn.flash_attention_version = 3  # Auto selection depends on the device.
        head_dim = attn.config.kv_channels
        num_heads = attn.config.num_attention_heads
        captured = {}

        # The FA3 wrapper filters kwargs by the kernel function's signature,
        # so the fake has to declare `softmax_scale` as a named parameter.
        def fake_fa3_forward(softmax_scale=None, **kwargs):
            captured['softmax_scale'] = softmax_scale
            return torch.randn(2, num_heads, head_dim, device='cuda', dtype=torch.float16)

        with (
            patch(f'{_ATTN_MODULE}.HAVE_FA4', False),
            patch(f'{_ATTN_MODULE}.HAVE_FA3', True),
            patch(f'{_ATTN_MODULE}._flash_attn_forward', fake_fa3_forward, create=True),
        ):
            attn.flash_decode_and_prefill(
                **_prefill_inputs(num_heads, head_dim), is_decode_only=False
            )

        assert captured['softmax_scale'] == 0.042


class TestDecodeKernelsSoftmaxScale(_GpuTestFixtureBase):
    """Tests that the decode paths of flash_decode_and_prefill respect softmax_scale."""

    def _run_decode(self, attn, patches, with_lse=False):
        head_dim = attn.config.kv_channels
        num_heads = attn.config.num_attention_heads
        captured = {}
        output = torch.randn(2, 1, num_heads, head_dim, device='cuda', dtype=torch.float16)
        fake_kernel = _capture_factory(captured, output, with_lse=with_lse)
        with contextlib.ExitStack() as stack:
            for p in patches(fake_kernel):
                stack.enter_context(p)
            attn.flash_decode_and_prefill(
                **_decode_inputs(num_heads, head_dim), is_decode_only=True
            )
        return captured['softmax_scale']

    def test_decode_fa2_uses_config_softmax_scale(self):
        attn = _make_self_attention(softmax_scale=0.042)

        def patches(fake_kernel):
            return (
                patch(f'{_ATTN_MODULE}.HAVE_FA4', False),
                patch(f'{_ATTN_MODULE}.HAVE_FA3', False),
                patch(f'{_ATTN_MODULE}.flash_attn_with_kvcache', fake_kernel),
            )

        assert self._run_decode(attn, patches) == 0.042

    def test_decode_fa2_default_softmax_scale(self):
        """When config.softmax_scale is None, decode uses 1/sqrt(head_dim)."""
        attn = _make_self_attention(softmax_scale=None)

        def patches(fake_kernel):
            return (
                patch(f'{_ATTN_MODULE}.HAVE_FA4', False),
                patch(f'{_ATTN_MODULE}.HAVE_FA3', False),
                patch(f'{_ATTN_MODULE}.flash_attn_with_kvcache', fake_kernel),
            )

        head_dim = attn.config.kv_channels
        assert self._run_decode(attn, patches) == pytest.approx(head_dim**-0.5)

    def test_decode_fa3_uses_config_softmax_scale(self):
        attn = _make_self_attention(softmax_scale=0.042)
        attn.flash_attention_version = 3  # Auto selection depends on the device.

        def patches(fake_kernel):
            return (
                patch(f'{_ATTN_MODULE}.HAVE_FA4', False),
                patch(f'{_ATTN_MODULE}.HAVE_FA3', True),
                patch(f'{_ATTN_MODULE}.flash_attn3_with_kvcache', fake_kernel, create=True),
            )

        assert self._run_decode(attn, patches) == 0.042

    def test_decode_fa4_uses_config_softmax_scale(self):
        attn = _make_self_attention(softmax_scale=0.042)
        attn.flash_attention_version = 4  # Auto selection depends on the device.

        def patches(fake_kernel):
            return (
                patch(f'{_ATTN_MODULE}.HAVE_FA4', True),
                patch(f'{_ATTN_MODULE}.flash_attn4_varlen_func', fake_kernel),
            )

        assert self._run_decode(attn, patches, with_lse=True) == 0.042


class TestMLADecodeSoftmaxScale(_GpuTestFixtureBase):
    """Tests that the FlashMLA decode path uses the MLA softmax_scale."""

    def _run_mla_decode(self, config, attention):
        head_dim = config.kv_lora_rank
        num_heads = config.num_attention_heads
        captured = {}

        def fake_get_mla_metadata(*args, **kwargs):
            return torch.zeros(1, device='cuda', dtype=torch.int32), 1

        def fake_flash_mla(*args, **kwargs):
            captured['softmax_scale'] = kwargs.get('softmax_scale')
            return (
                torch.randn(2, 1, num_heads, head_dim, device='cuda', dtype=torch.float16),
                None,
            )

        with (
            patch(f'{_ATTN_MODULE}.get_mla_metadata', fake_get_mla_metadata),
            patch(f'{_ATTN_MODULE}.flash_mla_with_kvcache', fake_flash_mla),
        ):
            attention.flash_decode_and_prefill(
                **_decode_inputs(num_heads, head_dim), is_decode_only=True
            )
        return captured['softmax_scale']

    def test_decode_flash_mla_uses_mla_softmax_scale(self):
        """FlashMLA decode receives mscale^2 * config.softmax_scale (via self.softmax_scale)."""
        config, attention = _make_mla(softmax_scale=0.042)
        scale = self._run_mla_decode(config, attention)
        mscale = _yarn_get_mscale(config.rotary_scaling_factor, config.mscale_all_dim)
        assert scale == pytest.approx(mscale * mscale * 0.042)
        assert scale == pytest.approx(attention.softmax_scale)

    def test_decode_flash_mla_asserts_on_missing_softmax_scale(self):
        """The decode guard asserts when an MLA layer's softmax_scale is None."""
        config, attention = _make_mla(softmax_scale=0.042)
        attention.softmax_scale = None
        with pytest.raises(AssertionError):
            self._run_mla_decode(config, attention)


class TestFlashDecodeSoftmaxScale(_GpuTestFixtureBase):
    """Tests that Attention.flash_decode (FA2 kvcache) respects softmax_scale."""

    def _call_and_capture_softmax_scale(self, attn):
        head_dim = attn.config.kv_channels
        num_heads = attn.config.num_attention_heads
        captured = {}

        fake_kvcache = _capture_factory(
            captured, torch.randn(2, 1, num_heads, head_dim, device='cuda', dtype=torch.float16)
        )

        with patch(f'{_ATTN_MODULE}.flash_attn_with_kvcache', fake_kvcache):
            attn.flash_decode(
                sequence_len_offset=torch.tensor([4, 4], device='cuda', dtype=torch.int32),
                query_layer=torch.randn(
                    1, 2, num_heads, head_dim, device='cuda', dtype=torch.float16
                ),
                key_layer=torch.randn(
                    1, 2, num_heads, head_dim, device='cuda', dtype=torch.float16
                ),
                value_layer=torch.randn(
                    1, 2, num_heads, head_dim, device='cuda', dtype=torch.float16
                ),
                inference_key_memory=torch.randn(
                    8, 2, num_heads, head_dim, device='cuda', dtype=torch.float16
                ),
                inference_value_memory=torch.randn(
                    8, 2, num_heads, head_dim, device='cuda', dtype=torch.float16
                ),
                rotary_cos=None,
                rotary_sin=None,
            )

        return captured['softmax_scale']

    def test_config_softmax_scale_used(self):
        attn = _make_self_attention(softmax_scale=0.042)
        assert self._call_and_capture_softmax_scale(attn) == 0.042

    def test_default_softmax_scale(self):
        attn = _make_self_attention(softmax_scale=None)
        head_dim = attn.config.kv_channels
        assert self._call_and_capture_softmax_scale(attn) == pytest.approx(head_dim**-0.5)


class TestMLASoftmaxScale(_GpuTestFixtureBase):
    """Tests that MLASelfAttention.__init__ respects config.softmax_scale."""

    def test_config_softmax_scale_applied(self):
        """When config.softmax_scale is set, MLA uses mscale^2 * config.softmax_scale."""
        config, attention = _make_mla(softmax_scale=0.042)
        mscale = _yarn_get_mscale(config.rotary_scaling_factor, config.mscale_all_dim)
        expected = mscale * mscale * 0.042
        assert attention.softmax_scale == pytest.approx(expected)

    def test_default_softmax_scale(self):
        """When config.softmax_scale is None, MLA uses mscale^2 / sqrt(q_head_dim)."""
        config, attention = _make_mla(softmax_scale=None)
        q_head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim
        mscale = _yarn_get_mscale(config.rotary_scaling_factor, config.mscale_all_dim)
        expected = mscale * mscale * (1 / math.sqrt(q_head_dim))
        assert attention.softmax_scale == pytest.approx(expected)

    def test_config_softmax_scale_with_yarn_scaling(self):
        """With rotary_scaling_factor > 1, mscale != 1, and config.softmax_scale is applied (not
        None).
        """
        config, attention = _make_mla(
            softmax_scale=0.042, rotary_scaling_factor=4.0, mscale_all_dim=1.0
        )
        mscale = _yarn_get_mscale(config.rotary_scaling_factor, config.mscale_all_dim)
        assert (
            mscale != 1.0
        ), "mscale should be nontrivial with scaling_factor > 1 and mscale_all_dim != 0"
        expected = mscale * mscale * 0.042
        assert attention.softmax_scale == pytest.approx(expected)

    def test_config_softmax_scale_with_trivial_scaling(self):
        """With rotary_scaling_factor and mscale = 0."""
        config, attention = _make_mla(softmax_scale=0.042, rotary_scaling_factor=4.0)
        mscale = _yarn_get_mscale(config.rotary_scaling_factor, config.mscale_all_dim)
        assert mscale == 1.0, "mscale should be trivial with mscale_all_dim == 0"
        expected = mscale * mscale * 0.042
        assert attention.softmax_scale == pytest.approx(expected)

    def test_softmax_scale_propagated_to_core_attention(self):
        """The computed softmax_scale should be passed to core_attention."""
        _, attention = _make_mla(softmax_scale=0.042)
        if isinstance(attention.core_attention, TEDotProductAttention):
            assert attention.core_attention.flash_attention.softmax_scale == attention.softmax_scale
            assert attention.core_attention.fused_attention.softmax_scale == attention.softmax_scale
            assert (
                attention.core_attention.unfused_attention.softmax_scale == attention.softmax_scale
            )
        else:
            assert attention.core_attention.softmax_scale == attention.softmax_scale
