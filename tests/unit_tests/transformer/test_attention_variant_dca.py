# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dca import (
    HAVE_FLASH_ATTN,
    DCASubmodules,
    DualChunkAttention,
    _get_yarn_mscale,
    _merge_chunk_attention_outputs,
)
from tests.unit_tests.test_utilities import Utils


class TestMergeChunkAttentionOutputs:
    """Test the LSE-based attention output merging function."""

    def test_single_output_passthrough(self):
        """A single output should be returned unchanged."""
        output = torch.randn(2, 4, 8, 16)
        lse = torch.randn(2, 4, 8, 1)
        result = _merge_chunk_attention_outputs([output], [lse])
        assert torch.equal(result, output)

    def test_two_outputs_sum_to_one(self):
        """With identical LSE, outputs should be averaged."""
        batch, heads, q_len, dim = 2, 4, 8, 16
        out1 = torch.ones(batch, heads, q_len, dim)
        out2 = torch.zeros(batch, heads, q_len, dim)
        lse = torch.zeros(batch, heads, q_len, 1)

        result = _merge_chunk_attention_outputs([out1, out2], [lse, lse])
        expected = 0.5 * out1 + 0.5 * out2
        assert torch.allclose(result, expected, atol=1e-6)

    def test_dominant_lse_selects_output(self):
        """Output with much larger LSE should dominate the merged result."""
        batch, heads, q_len, dim = 1, 1, 4, 8
        out1 = torch.ones(batch, heads, q_len, dim)
        out2 = torch.zeros(batch, heads, q_len, dim)
        lse1 = torch.full((batch, heads, q_len, 1), 100.0)
        lse2 = torch.full((batch, heads, q_len, 1), -100.0)

        result = _merge_chunk_attention_outputs([out1, out2], [lse1, lse2])
        assert torch.allclose(result, out1, atol=1e-5)


class TestDualChunkAttentionInit:
    """Test DualChunkAttention initialization."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def _make_config(self, **overrides):
        defaults = dict(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_query_groups=4,
            use_cpu_initialization=True,
            dca_chunk_size=32,
            dca_local_size=8,
            experimental_attention_variant="dca",
            apply_rope_fusion=False,
        )
        defaults.update(overrides)
        return TransformerConfig(**defaults)

    def test_basic_construction(self):
        config = self._make_config()
        dca = DualChunkAttention(
            config=config,
            submodules=DCASubmodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        assert dca.chunk_size == 32
        assert dca.local_size == 8
        assert dca.chunk_len == 24

    def test_config_validation_chunk_size(self):
        with pytest.raises(AssertionError, match="dca_chunk_size"):
            self._make_config(dca_chunk_size=8, dca_local_size=8)

    def test_config_validation_rope_fusion(self):
        with pytest.raises(AssertionError, match="RoPE fusion"):
            self._make_config(apply_rope_fusion=True)


class TestDualChunkAttentionForward:
    """Test DualChunkAttention forward pass."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def _make_config(self, chunk_size=32, local_size=8, num_heads=4, num_kv_heads=4, hidden=64):
        return TransformerConfig(
            num_layers=2,
            hidden_size=hidden,
            num_attention_heads=num_heads,
            num_query_groups=num_kv_heads,
            use_cpu_initialization=True,
            dca_chunk_size=chunk_size,
            dca_local_size=local_size,
            experimental_attention_variant="dca",
            apply_rope_fusion=False,
        )

    def _make_dca(self, config):
        return DualChunkAttention(
            config=config,
            submodules=DCASubmodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

    def _make_rotary_pos_emb(self, seq_len, head_dim, device='cpu'):
        """Create mock rotary position embeddings."""
        base = 10000.0
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        emb = emb.unsqueeze(1).unsqueeze(1).to(device)
        return (emb, emb)

    def test_short_sequence_output_shape(self):
        """Short sequences (< chunk_len) should produce correct output shape."""
        config = self._make_config(chunk_size=32, local_size=8)
        dca = self._make_dca(config)
        head_dim = config.hidden_size // config.num_attention_heads

        seq_len, batch = 16, 2
        query = torch.randn(seq_len, batch, config.num_attention_heads, head_dim)
        key = torch.randn(seq_len, batch, config.num_query_groups, head_dim)
        value = torch.randn(seq_len, batch, config.num_query_groups, head_dim)
        rotary = self._make_rotary_pos_emb(seq_len, head_dim)

        output = dca(query, key, value, attention_mask=None, rotary_pos_emb=rotary)
        assert output.shape == (seq_len, batch, config.hidden_size)

    def test_long_sequence_output_shape(self):
        """Long sequences (> chunk_len) should produce correct output shape."""
        config = self._make_config(chunk_size=16, local_size=4)
        dca = self._make_dca(config)
        head_dim = config.hidden_size // config.num_attention_heads

        seq_len, batch = 48, 2
        query = torch.randn(seq_len, batch, config.num_attention_heads, head_dim)
        key = torch.randn(seq_len, batch, config.num_query_groups, head_dim)
        value = torch.randn(seq_len, batch, config.num_query_groups, head_dim)
        rotary = self._make_rotary_pos_emb(seq_len, head_dim)

        output = dca(query, key, value, attention_mask=None, rotary_pos_emb=rotary)
        assert output.shape == (seq_len, batch, config.hidden_size)

    def test_short_sequence_equivalence_to_standard(self):
        """For short sequences, DCA should match standard causal attention."""
        config = self._make_config(
            chunk_size=64, local_size=8, num_heads=2, num_kv_heads=2, hidden=32
        )
        dca = self._make_dca(config)
        head_dim = config.hidden_size // config.num_attention_heads

        seq_len, batch = 8, 1
        torch.manual_seed(42)
        query = torch.randn(seq_len, batch, config.num_attention_heads, head_dim)
        key = torch.randn(seq_len, batch, config.num_query_groups, head_dim)
        value = torch.randn(seq_len, batch, config.num_query_groups, head_dim)
        rotary = self._make_rotary_pos_emb(seq_len, head_dim)

        dca_output = dca(query, key, value, attention_mask=None, rotary_pos_emb=rotary)

        from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

        q_pos_emb, k_pos_emb = rotary
        q_with_rope = apply_rotary_pos_emb(query, q_pos_emb, config=config)
        k_with_rope = apply_rotary_pos_emb(key, k_pos_emb, config=config)

        q = q_with_rope.permute(1, 2, 0, 3)
        k = k_with_rope.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
        scale = head_dim**-0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        ref_output = torch.matmul(attn, v)
        ref_output = ref_output.permute(2, 0, 1, 3).contiguous()
        ref_output = ref_output.reshape(seq_len, batch, config.hidden_size)

        assert torch.allclose(
            dca_output, ref_output, atol=1e-5
        ), f"Max diff: {(dca_output - ref_output).abs().max().item()}"

    def test_gqa_support(self):
        """DCA should work with Grouped Query Attention (num_kv_heads < num_heads)."""
        config = self._make_config(
            chunk_size=16, local_size=4, num_heads=8, num_kv_heads=2, hidden=64
        )
        dca = self._make_dca(config)
        head_dim = config.hidden_size // config.num_attention_heads

        seq_len, batch = 32, 2
        query = torch.randn(seq_len, batch, config.num_attention_heads, head_dim)
        key = torch.randn(seq_len, batch, config.num_query_groups, head_dim)
        value = torch.randn(seq_len, batch, config.num_query_groups, head_dim)
        rotary = self._make_rotary_pos_emb(seq_len, head_dim)

        output = dca(query, key, value, attention_mask=None, rotary_pos_emb=rotary)
        assert output.shape == (seq_len, batch, config.hidden_size)

    def test_backward_gradient_flow(self):
        """Gradients should flow correctly through DCA."""
        config = self._make_config(
            chunk_size=16, local_size=4, num_heads=2, num_kv_heads=2, hidden=32
        )
        dca = self._make_dca(config)
        head_dim = config.hidden_size // config.num_attention_heads

        seq_len, batch = 32, 1
        query = torch.randn(
            seq_len, batch, config.num_attention_heads, head_dim, requires_grad=True
        )
        key = torch.randn(seq_len, batch, config.num_query_groups, head_dim, requires_grad=True)
        value = torch.randn(seq_len, batch, config.num_query_groups, head_dim, requires_grad=True)
        rotary = self._make_rotary_pos_emb(seq_len, head_dim)

        output = dca(query, key, value, attention_mask=None, rotary_pos_emb=rotary)
        loss = output.sum()
        loss.backward()

        assert query.grad is not None, "Query gradient is None"
        assert key.grad is not None, "Key gradient is None"
        assert value.grad is not None, "Value gradient is None"
        assert not torch.all(query.grad == 0), "Query gradient is all zeros"

    def test_multiple_chunks(self):
        """Test with a sequence that spans 3+ chunks to exercise inter-chunk attention."""
        config = self._make_config(
            chunk_size=12, local_size=4, num_heads=2, num_kv_heads=2, hidden=32
        )
        dca = self._make_dca(config)
        head_dim = config.hidden_size // config.num_attention_heads
        chunk_len = config.dca_chunk_size - config.dca_local_size

        seq_len = chunk_len * 4
        batch = 1
        query = torch.randn(seq_len, batch, config.num_attention_heads, head_dim)
        key = torch.randn(seq_len, batch, config.num_query_groups, head_dim)
        value = torch.randn(seq_len, batch, config.num_query_groups, head_dim)
        rotary = self._make_rotary_pos_emb(seq_len, head_dim)

        output = dca(query, key, value, attention_mask=None, rotary_pos_emb=rotary)
        assert output.shape == (seq_len, batch, config.hidden_size)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_non_divisible_sequence_length(self):
        """Test with sequence length not evenly divisible by chunk_len."""
        config = self._make_config(
            chunk_size=16, local_size=4, num_heads=2, num_kv_heads=2, hidden=32
        )
        dca = self._make_dca(config)
        head_dim = config.hidden_size // config.num_attention_heads
        chunk_len = config.dca_chunk_size - config.dca_local_size

        seq_len = chunk_len * 2 + 5
        batch = 1
        query = torch.randn(seq_len, batch, config.num_attention_heads, head_dim)
        key = torch.randn(seq_len, batch, config.num_query_groups, head_dim)
        value = torch.randn(seq_len, batch, config.num_query_groups, head_dim)
        rotary = self._make_rotary_pos_emb(seq_len, head_dim)

        output = dca(query, key, value, attention_mask=None, rotary_pos_emb=rotary)
        assert output.shape == (seq_len, batch, config.hidden_size)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        """Test forward pass on GPU."""
        config = self._make_config(
            chunk_size=16, local_size=4, num_heads=2, num_kv_heads=2, hidden=32
        )
        dca = self._make_dca(config).cuda()
        head_dim = config.hidden_size // config.num_attention_heads

        seq_len, batch = 32, 2
        query = torch.randn(seq_len, batch, config.num_attention_heads, head_dim).cuda()
        key = torch.randn(seq_len, batch, config.num_query_groups, head_dim).cuda()
        value = torch.randn(seq_len, batch, config.num_query_groups, head_dim).cuda()
        rotary = self._make_rotary_pos_emb(seq_len, head_dim, device='cuda')

        output = dca(query, key, value, attention_mask=None, rotary_pos_emb=rotary)
        assert output.shape == (seq_len, batch, config.hidden_size)
        assert output.device.type == 'cuda'

    def test_causal_property(self):
        """Verify that DCA preserves causality: future tokens don't affect past outputs."""
        config = self._make_config(
            chunk_size=16, local_size=4, num_heads=2, num_kv_heads=2, hidden=32
        )
        dca = self._make_dca(config)
        head_dim = config.hidden_size // config.num_attention_heads

        seq_len, batch = 20, 1
        torch.manual_seed(123)
        query = torch.randn(seq_len, batch, config.num_attention_heads, head_dim)
        key = torch.randn(seq_len, batch, config.num_query_groups, head_dim)
        value = torch.randn(seq_len, batch, config.num_query_groups, head_dim)
        rotary = self._make_rotary_pos_emb(seq_len, head_dim)

        output_full = dca(query, key, value, attention_mask=None, rotary_pos_emb=rotary)

        prefix_len = 10
        rotary_prefix = self._make_rotary_pos_emb(prefix_len, head_dim)
        output_prefix = dca(
            query[:prefix_len],
            key[:prefix_len],
            value[:prefix_len],
            attention_mask=None,
            rotary_pos_emb=rotary_prefix,
        )

        assert torch.allclose(output_full[:prefix_len], output_prefix, atol=1e-5), (
            f"Causality violated. Max diff: "
            f"{(output_full[:prefix_len] - output_prefix).abs().max().item()}"
        )

    def test_yarn_mscale_is_applied(self):
        """Verify that YARN mscale factor is retrieved from config."""
        config_no_yarn = self._make_config()
        mscale = _get_yarn_mscale(config_no_yarn)
        assert mscale == 1.0, f"Default mscale should be 1.0, got {mscale}"

    def test_flash_attn_availability_flag(self):
        """HAVE_FLASH_ATTN should be a boolean."""
        assert isinstance(HAVE_FLASH_ATTN, bool)

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_FLASH_ATTN,
        reason="CUDA and FlashAttention required",
    )
    def test_flash_attn_vs_unfused_equivalence(self):
        """FlashAttention and unfused paths should produce similar results."""
        config = self._make_config(
            chunk_size=16, local_size=4, num_heads=2, num_kv_heads=2, hidden=32
        )
        dca = self._make_dca(config).cuda()
        head_dim = config.hidden_size // config.num_attention_heads

        seq_len, batch = 32, 1
        torch.manual_seed(42)
        query = torch.randn(seq_len, batch, config.num_attention_heads, head_dim).cuda()
        key = torch.randn(seq_len, batch, config.num_query_groups, head_dim).cuda()
        value = torch.randn(seq_len, batch, config.num_query_groups, head_dim).cuda()
        rotary = self._make_rotary_pos_emb(seq_len, head_dim, device='cuda')

        fa_output = dca._flash_attention_with_lse(query[:8], key[:8], value[:8], causal=True)
        unfused_kv = key[:8].repeat_interleave(1, dim=2)
        unfused_output = dca._unfused_attention_with_lse(
            query[:8], unfused_kv, value[:8], causal=True
        )

        assert torch.allclose(
            fa_output[0], unfused_output[0], atol=1e-3
        ), f"FA vs unfused max diff: {(fa_output[0] - unfused_output[0]).abs().max().item()}"
