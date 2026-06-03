# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import gc
import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import megatron.core.parallel_state as parallel_state
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa_cp_utils import (
    exchange_left_boundary_tensor,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_native_parity import (
    _DSV4_VARIANTS,
)
from tests.unit_tests.test_utilities import Utils

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_transform

    HAVE_HADAMARD = True
except ImportError:
    HAVE_HADAMARD = False
    _hadamard_transform = None

_SEED = 42


def _mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return x * scale


@pytest.fixture(autouse=True)
def patch_hadamard_if_needed():
    """Patch hadamard_transform in dsa/csa modules if the library is not installed."""
    if not HAVE_HADAMARD:
        with (
            patch(
                'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
                _mock_hadamard_transform,
            ),
            patch(
                'megatron.core.transformer.experimental_attention_variant.csa.rotate_activation',
                lambda x: x * (x.size(-1) ** -0.5),
            ),
        ):
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# Config / spec helpers
# ---------------------------------------------------------------------------


def _make_config(
    num_layers=4,
    hidden_size=256,
    num_attention_heads=16,
    v_head_dim=64,
    qk_pos_emb_head_dim=32,
    q_lora_rank=64,
    o_groups=8,
    o_lora_rank=64,
    csa_compress_ratios=None,
    csa_window_size=8,
    tensor_model_parallel_size=1,
    sequence_parallel=False,
    dsa_indexer_n_heads=8,
    dsa_indexer_head_dim=64,
    dsa_indexer_topk=8,
    dsa_indexer_loss_coeff=0.0,
    **extra_config_kwargs,
):
    """Create an MLATransformerConfig for DSv4 hybrid attention tests."""
    if csa_compress_ratios is None:
        csa_compress_ratios = [0, 4, 128, 4]
    return MLATransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        tensor_model_parallel_size=tensor_model_parallel_size,
        sequence_parallel=sequence_parallel,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=v_head_dim - qk_pos_emb_head_dim,
        qk_head_dim=v_head_dim - qk_pos_emb_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        v_head_dim=v_head_dim,
        o_groups=o_groups,
        o_lora_rank=o_lora_rank,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        multi_latent_attention=True,
        experimental_attention_variant='dsv4_hybrid',
        csa_compress_ratios=csa_compress_ratios,
        csa_window_size=csa_window_size,
        dsa_indexer_n_heads=dsa_indexer_n_heads,
        dsa_indexer_head_dim=dsa_indexer_head_dim,
        dsa_indexer_topk=dsa_indexer_topk,
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
        **extra_config_kwargs,
    )


def _make_attention_spec(config):
    """Build the full DSv4HybridSelfAttention ModuleSpec using the canonical spec builder."""
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_dsv4_hybrid_module_spec_for_backend,
    )

    return get_dsv4_hybrid_module_spec_for_backend(config=config, backend=TESpecProvider())


def _build_attention(config, layer_number, pg_collection):
    """Instantiate a DSv4HybridSelfAttention from config."""
    from megatron.core.transformer.spec_utils import build_module

    spec = _make_attention_spec(config)
    return build_module(spec, config=config, layer_number=layer_number, pg_collection=pg_collection)


# ===========================================================================
# Constructor tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionConstructor:
    """Test construction of DSv4HybridSelfAttention across TP sizes."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def test_basic_construction(self):
        """Verify the layer builds and has the expected sub-modules."""
        from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
            DSv4HybridSelfAttention,
        )

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        config = _make_config()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=1, pg_collection=pg)

        assert isinstance(attn, DSv4HybridSelfAttention)
        assert hasattr(attn, 'linear_q_down_proj')
        assert hasattr(attn, 'linear_q_up_proj')
        assert hasattr(attn, 'linear_kv_proj')
        assert hasattr(attn, 'linear_proj')
        assert hasattr(attn, 'linear_o_group_proj')
        assert hasattr(attn, 'core_attention')
        assert hasattr(attn, 'q_layernorm')
        assert hasattr(attn, 'kv_layernorm')

    def test_q_head_dim_equals_v_head_dim(self):
        """q_head_dim must equal v_head_dim for DSv4 hybrid."""
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        config = _make_config()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=1, pg_collection=pg)

        assert attn.q_head_dim == config.v_head_dim

    @pytest.mark.parametrize("layer_number", [1, 2, 3, 4])
    def test_rope_base_varies_with_compress_ratio(self, layer_number):
        """Layers with compress_ratio > 1 should use csa_compress_rotary_base."""
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        ratios = [0, 4, 128, 4]
        config = _make_config(csa_compress_ratios=ratios)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=layer_number, pg_collection=pg)

        ratio = ratios[layer_number - 1]
        if ratio > 1:
            expected_base = config.csa_compress_rotary_base
        else:
            expected_base = config.rotary_base

        # inv_freq is derived from rotary_base; verify the correct base was used
        dim = config.qk_pos_emb_head_dim
        recomputed_inv_freq = 1.0 / (
            expected_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        assert torch.allclose(
            attn.rotary_pos_emb.inv_freq.cpu(), recomputed_inv_freq, rtol=1e-5, atol=1e-5
        )


# ===========================================================================
# Forward / backward tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionForwardBackward:
    """Test forward and backward passes of DSv4HybridSelfAttention."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.config = _make_config(dsa_indexer_loss_coeff=1.0)
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("layer_number", [1, 2, 3, 4])
    def test_forward_output_shape(self, layer_number):
        """Forward should produce [sq, b, hidden_size] output."""
        seq_len = 256
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()

        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        output, bias = attn(hidden_states=hidden, attention_mask=None)

        assert output.shape == (seq_len, batch_size, self.config.hidden_size)
        assert output.dtype == torch.bfloat16
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("layer_number", [1, 2])
    def test_backward_gradient_flow(self, layer_number):
        """Backward should produce gradients for all trainable parameters."""
        seq_len = 256
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        attn.train()

        hidden = (
            torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )

        output, bias = attn(hidden_states=hidden, attention_mask=None)
        loss = output.sum()
        loss.backward()

        assert hidden.grad is not None, "No gradient on hidden_states"
        for name, param in attn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"

    def test_eval_mode(self):
        """Forward should work in eval mode."""
        seq_len = 128
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=1, pg_collection=self.pg).cuda()
        attn.eval()

        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        with torch.no_grad():
            output, bias = attn(hidden_states=hidden, attention_mask=None)

        assert output.shape == (seq_len, batch_size, self.config.hidden_size)
        assert not torch.isnan(output).any()

    def test_different_seq_lengths(self):
        """Forward should handle various sequence lengths."""
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=2, pg_collection=self.pg).cuda()

        for seq_len in [64, 128, 256]:
            hidden = torch.randn(
                seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
            ).cuda()
            output, bias = attn(hidden_states=hidden, attention_mask=None)
            assert output.shape == (seq_len, batch_size, self.config.hidden_size)


# ===========================================================================
# get_query_key_value_tensors tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridQKV:
    """Test get_query_key_value_tensors internals."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.config = _make_config()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    def test_qkv_shapes(self):
        """Query, key, value should have correct shapes."""
        seq_len = 64
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=1, pg_collection=self.pg).cuda()
        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        q, k, v, q_compressed, kv_compressed = attn.get_query_key_value_tensors(hidden)

        n_heads = self.config.num_attention_heads
        v_dim = self.config.v_head_dim

        assert q.shape == (seq_len, batch_size, n_heads, v_dim)
        # key and value are single-head (MQA-style) with an extra head dim
        assert k.shape[-1] == v_dim
        assert v.shape[-1] == v_dim

    def test_key_equals_value(self):
        """In the wkv path, key and value should be the same tensor."""
        seq_len = 64
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=1, pg_collection=self.pg).cuda()
        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        q, k, v, _, _ = attn.get_query_key_value_tensors(hidden)
        assert torch.equal(k, v), "key and value should be identical in wkv path"


# ===========================================================================
# Grouped output projection tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridGroupedOutput:
    """Test that grouped output projection (wo_a) parameters are created."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def test_o_group_proj_shape(self):
        """linear_o_group_proj should have the correct shape."""
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        o_groups = 8
        o_lora_rank = 64
        config = _make_config(o_groups=o_groups, o_lora_rank=o_lora_rank)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=1, pg_collection=pg)

        expected_out = o_groups * o_lora_rank
        expected_in = (config.v_head_dim * config.num_attention_heads) // o_groups
        assert attn.linear_o_group_proj.shape == (expected_out, expected_in)
        assert attn.linear_o_group_proj.requires_grad


# ===========================================================================
# DSv4 Hybrid Attention + Hash MoE integration tests
# ===========================================================================


def _make_dsv4_hash_moe_config():
    """Create a compact DSv4 config that combines CSA/HCA with hash MoE."""
    return _make_config(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=8,
        v_head_dim=32,
        qk_pos_emb_head_dim=16,
        q_lora_rank=32,
        o_groups=4,
        o_lora_rank=32,
        csa_compress_ratios=[4, 128],
        csa_window_size=16,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=8,
        ffn_hidden_size=256,
        num_moe_experts=4,
        moe_ffn_hidden_size=256,
        moe_layer_freq=1,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_router_score_function="sqrtsoftplus",
        moe_n_hash_layers=1,
        actual_vocab_size=128,
        activation_func=F.silu,
        gated_linear_unit=True,
        activation_func_clamp_value=10.0,
        bias_activation_fusion=False,
        moe_grouped_gemm=False,
    )


def _build_dsv4_moe_layer(config, layer_number, pg_collection):
    """Instantiate a TransformerLayer from the DSv4 experimental attention spec."""
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_layer_with_experimental_attention_variant_spec,
    )
    from megatron.core.transformer.spec_utils import build_module

    layer_specs = get_transformer_layer_with_experimental_attention_variant_spec(
        config=config, backend=TESpecProvider()
    )
    return build_module(
        layer_specs[layer_number - 1],
        config=config,
        layer_number=layer_number,
        pg_collection=pg_collection,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridHashMoEIntegration:
    """Integration coverage for DSv4 hybrid attention with hash MoE and clamped SwiGLU."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.config = _make_dsv4_hash_moe_config()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    def test_csa_hash_moe_layer_forward_backward(self):
        """Layer 1 should combine DSv4 CSA, hash routing, and clamped SwiGLU."""
        from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
            DSv4HybridSelfAttention,
        )
        from megatron.core.transformer.moe.moe_layer import MoELayer

        seq_len = 256
        batch_size = 1

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        layer = _build_dsv4_moe_layer(self.config, layer_number=1, pg_collection=self.pg).cuda()
        layer.train()

        assert isinstance(layer.self_attention, DSv4HybridSelfAttention)
        assert layer.self_attention.core_attention.compress_ratio == 4
        assert isinstance(layer.mlp, MoELayer)
        assert layer.mlp.router.is_hash_layer is True
        assert layer.mlp.router.tid2eid is not None
        assert layer.config.activation_func_clamp_value == 10.0
        assert layer.config.activation_func is F.silu
        assert layer.config.gated_linear_unit is True

        hidden = torch.randn(
            seq_len,
            batch_size,
            self.config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        input_ids = torch.randint(
            0, self.config.actual_vocab_size, (batch_size, seq_len), device="cuda"
        )

        output, context = layer(hidden_states=hidden, attention_mask=None, input_ids=input_ids)
        loss = output.float().square().mean()
        loss.backward()

        assert context is None
        assert output.shape == hidden.shape
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()
        assert hidden.grad is not None
        assert torch.isfinite(hidden.grad).all()
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in layer.self_attention.parameters()
            if p.requires_grad
        )
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in layer.mlp.parameters()
            if p.requires_grad
        )

    def test_hash_moe_layer_requires_input_ids_but_hca_layer_does_not(self):
        """Hash routing is limited to leading layers while later HCA MoE layers remain runnable."""
        seq_len = 256
        batch_size = 1

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        hash_layer = _build_dsv4_moe_layer(
            self.config, layer_number=1, pg_collection=self.pg
        ).cuda()
        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        with pytest.raises(AssertionError, match="input_ids is required for hash-based routing"):
            hash_layer(hidden_states=hidden, attention_mask=None)

        hca_layer = _build_dsv4_moe_layer(self.config, layer_number=2, pg_collection=self.pg).cuda()
        assert hca_layer.self_attention.core_attention.compress_ratio == 128
        assert hca_layer.mlp.router.is_hash_layer is False

        output, context = hca_layer(hidden_states=hidden, attention_mask=None)

        assert context is None
        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()


# ===========================================================================
# apply_rope_fusion tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridRopeFusion:
    """Test that apply_rope_fusion=True works for both yarn and non-yarn layers.

    DSv4 Hybrid uses YarnRotaryEmbedding for layers with compress_ratio > 1
    and standard RotaryEmbedding for layers with compress_ratio <= 1. The
    fused RoPE path must obtain cos/sin from both embedding classes via
    get_cached_cos_sin.

    compress_ratios=[0, 4, 128, 4]: layer 1 has ratio 0 (standard
    RotaryEmbedding), layers 2-4 have ratio > 1 (YarnRotaryEmbedding).
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    def test_rope_fusion_forward_backward_parity(self):
        """Fused RoPE forward/backward succeeds and matches the unfused path."""
        seq_len = 128
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        fused_config = _make_config(apply_rope_fusion=True)
        attn_fused = _build_attention(fused_config, layer_number=4, pg_collection=self.pg).cuda()
        attn_fused.train()

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        unfused_config = _make_config(apply_rope_fusion=False)
        attn_unfused = _build_attention(
            unfused_config, layer_number=4, pg_collection=self.pg
        ).cuda()
        attn_unfused.train()

        hidden = torch.randn(
            seq_len, batch_size, fused_config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        out_fused, _ = attn_fused(hidden_states=hidden, attention_mask=None)
        out_unfused, _ = attn_unfused(hidden_states=hidden, attention_mask=None)

        assert out_fused.shape == (seq_len, batch_size, fused_config.hidden_size)
        assert torch.isfinite(out_fused).all()
        # Production code forces ``mscale=1.0`` (DSv4 contract) in both
        # fused and unfused paths, so the only residual is bf16 noise from
        # the fused Triton kernel's different accumulation order vs the
        # PyTorch eager ops. The residual concentrates at output positions
        # whose values are near zero (sign flips on tiny magnitudes drive
        # the worst-case max-abs-diff).
        torch.testing.assert_close(out_fused, out_unfused, atol=3e-2, rtol=3e-2)

        hidden_fused = hidden.detach().clone().requires_grad_(True)
        hidden_unfused = hidden.detach().clone().requires_grad_(True)

        attn_fused(hidden_states=hidden_fused, attention_mask=None)[0].sum().backward()
        attn_unfused(hidden_states=hidden_unfused, attention_mask=None)[0].sum().backward()

        assert hidden_fused.grad is not None
        for name, param in attn_fused.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"


# ===========================================================================
# THD packed-sequence end-to-end
# ===========================================================================
#
# Closes the highest-level integration gap: even though the CSA THD path
# is independently tested in test_attention_variant_csa.py
# (TestCompressedSparseAttentionThd), the DSv4HybridSelfAttention module
# adds its own THD-aware glue around CSA — packed_seq_params propagation
# through get_query_key_value_tensors, the output reshape at line ~290
# (``core_attn_out.reshape(total, 1, -1)`` to recover the 3-D contract),
# and the inverse-RoPE call with cu_seqlens. These tests verify the full
# DSv4Hybrid forward/backward works end-to-end for each ``compress_ratio``.


from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402


def _make_thd_packed_seq_params(seg_lens, padded_seg_lens=None, device='cuda'):
    """Build ``PackedSeqParams(qkv_format='thd', ...)`` for self-attention
    (``cu_seqlens_q == cu_seqlens_kv``) from a list of per-segment lengths.
    """
    if padded_seg_lens is None:
        padded_seg_lens = seg_lens
    assert len(seg_lens) == len(padded_seg_lens)
    assert all(actual <= padded for actual, padded in zip(seg_lens, padded_seg_lens))

    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(seg_lens, dtype=torch.int64).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_padded = torch.tensor(
        [0] + list(torch.tensor(padded_seg_lens, dtype=torch.int64).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    max_len = int(max(padded_seg_lens)) if padded_seg_lens else 0
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_len,
        max_seqlen_kv=max_len,
        qkv_format='thd',
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionThd:
    """End-to-end THD forward/backward of :class:`DSv4HybridSelfAttention`
    across all configured ``compress_ratio`` values (0/4/128).

    Each test runs a multi-segment THD batch through the full
    ``attn(hidden_states, packed_seq_params=...)`` pipeline and verifies:

      * Output shape ``(total_tokens, 1, hidden_size)`` — the layer
        re-adds the dummy ``b=1`` axis at line ~293 of
        ``deepseek_v4_hybrid_attention.py``.
      * No NaN.
      * (Backward test) grads flow on ``hidden_states`` and every
        learnable parameter.

    This is the highest-level integration test for THD; lower-level
    parity vs SBHD is established by ``TestCompressedSparseAttentionThd``
    in ``test_attention_variant_csa.py``.
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        # ``csa_compress_ratios=[0, 4, 128, 4]`` → layer_number ∈ {1,2,3,4}
        # cover all three CSA ratios (0 = window-only; 4 = full
        # indexer+compressed; 128 = compressor-only, no indexer).
        cls.config = _make_config(dsa_indexer_loss_coeff=1.0)
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2, 3, 4],
        ids=[
            "ratio_0_window_only",  # layer 1 → ratio=0
            "ratio_4_with_indexer",  # layer 2 → ratio=4
            "ratio_128_compressor_only",  # layer 3 → ratio=128
            "ratio_4_with_indexer_alt",  # layer 4 → ratio=4
        ],
    )
    def test_thd_forward_output_shape(self, layer_number):
        """THD forward through DSv4HybridSelfAttention produces
        ``(total_tokens, 1, hidden_size)`` output, no NaN, for every
        configured ``compress_ratio``.
        """
        seg_lens = [128, 96, 64]  # multi-segment, varied lengths
        total = sum(seg_lens)

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        attn.eval()

        # THD hidden_states shape is ``(total_tokens, 1, hidden_size)``
        # per the DSv4 hybrid contract.
        hidden = torch.randn(total, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda')
        packed = _make_thd_packed_seq_params(seg_lens)

        with torch.no_grad():
            output, _bias = attn(
                hidden_states=hidden, attention_mask=None, packed_seq_params=packed
            )

        assert output.shape == (total, 1, self.config.hidden_size), (
            f"layer {layer_number}: shape {tuple(output.shape)} != "
            f"expected {(total, 1, self.config.hidden_size)}"
        )
        assert output.dtype == torch.bfloat16
        assert not torch.isnan(output).any(), f"layer {layer_number}: NaN in THD forward output"

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2],  # ratio=0 (window-only) and ratio=4 (full indexer pipeline)
        ids=["ratio_0_window_only", "ratio_4_with_indexer"],
    )
    def test_thd_backward_gradient_flow(self, layer_number):
        """THD backward produces grads on ``hidden_states`` and every
        learnable parameter (covers the full indexer-loss path in Path
        B THD when ``layer_number=2`` triggers ratio=4).
        """
        seg_lens = [128, 96]
        total = sum(seg_lens)

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        attn.train()

        hidden = torch.randn(
            total, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda'
        ).requires_grad_(True)
        packed = _make_thd_packed_seq_params(seg_lens)

        output, _bias = attn(hidden_states=hidden, attention_mask=None, packed_seq_params=packed)
        output.sum().backward()

        assert hidden.grad is not None, "no grad on hidden_states"
        assert not torch.isnan(hidden.grad).any()
        for name, param in attn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"no grad on {name}"
                assert not torch.isnan(param.grad).any(), f"NaN grad on {name}"

    def test_thd_single_segment_matches_sbhd_b1(self):
        """B=1 single-segment THD output matches the SBHD-b=1 output on
        identical hidden states (sanity check that the DSv4Hybrid
        THD-vs-SBHD glue doesn't silently change the math).

        Uses ``layer_number=1`` (ratio=0, window-only) for determinism —
        no indexer top-K tie-breaking nondeterminism.
        """
        layer_number = 1  # ratio=0 → window-only path, no cuDNN topk
        sq = 128

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        attn.eval()

        hidden = torch.randn(sq, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda')

        with torch.no_grad():
            out_sbhd, _ = attn(hidden_states=hidden, attention_mask=None, packed_seq_params=None)
            packed = _make_thd_packed_seq_params([sq])
            out_thd, _ = attn(hidden_states=hidden, attention_mask=None, packed_seq_params=packed)

        assert out_sbhd.shape == out_thd.shape
        # Generous tol: the full DSv4Hybrid forward chains many bf16 ops
        # (QKV down/up proj, RoPE, attn, output proj). We're testing for
        # no plumbing bug, not bit-exactness.
        assert torch.allclose(out_sbhd.float(), out_thd.float(), atol=5e-2, rtol=5e-2), (
            f"B=1 SBHD/THD parity failed: max abs diff = "
            f"{(out_sbhd.float() - out_thd.float()).abs().max().item():.4e}"
        )


# ===========================================================================
# THD CP parity tests
# ===========================================================================


_DSV4_CP_PARITY_EPS = 1e-3
_DSV4_CP_TEST_VARIANT = "flash"
# Padded total is 4096, divisible by CP2/CP4. Only the final segment has tail
# padding, so the intermediate sequence boundaries match the unpadded layout.
_DSV4_CP_RAGGED_SEG_LENS = (1, 127, 1000, 23, 129, 900, 55, 257, 800, 95, 509, 148)
_DSV4_CP_RAGGED_PADDED_SEG_LENS = (
    1,
    127,
    1000,
    23,
    129,
    900,
    55,
    257,
    800,
    95,
    509,
    200,
)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.flatten().double().unsqueeze(0), b.flatten().double().unsqueeze(0)
    ).item()


def _tensor_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    denom = (a * a + b * b).sum()
    return (2.0 * (a * b).sum() / denom).item() if denom else 1.0


def _assert_cp_tensor_match(actual: torch.Tensor, expected: torch.Tensor, label: str):
    assert actual.shape == expected.shape, (
        f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    assert torch.isfinite(actual).all(), f"{label}: actual has non-finite values"
    assert torch.isfinite(expected).all(), f"{label}: expected has non-finite values"

    diff = (actual - expected).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    cosine_sim = _cosine_sim(actual, expected)
    tensor_sim = _tensor_sim(actual, expected)
    assert cosine_sim > 1 - _DSV4_CP_PARITY_EPS, (
        f"{label}: cosine_sim={cosine_sim:.10f}, "
        f"max_abs={max_abs:.6e}, eps={_DSV4_CP_PARITY_EPS}"
    )
    assert tensor_sim > 1 - _DSV4_CP_PARITY_EPS, (
        f"{label}: tensor_sim={tensor_sim:.10f}, "
        f"max_abs={max_abs:.6e}, eps={_DSV4_CP_PARITY_EPS}"
    )


def _make_dsv4_cp_config(
    *,
    context_parallel_size,
    dsa_indexer_loss_coeff=0.0,
    dsa_indexer_use_sparse_loss=True,
    apply_dsa_kernel_fusion=True,
    apply_rope_fusion=False,
):
    shape = _DSV4_VARIANTS[_DSV4_CP_TEST_VARIANT]
    return _make_config(
        hidden_size=shape["hidden_size"],
        num_attention_heads=shape["num_attention_heads"],
        v_head_dim=shape["v_head_dim"],
        qk_pos_emb_head_dim=shape["qk_pos_emb_head_dim"],
        q_lora_rank=shape["q_lora_rank"],
        o_groups=shape["o_groups"],
        o_lora_rank=shape["o_lora_rank"],
        csa_compress_ratios=[0, 4, 128, 4],
        csa_window_size=128,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=shape["dsa_indexer_topk"],
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
        dsa_indexer_use_sparse_loss=dsa_indexer_use_sparse_loss,
        context_parallel_size=context_parallel_size,
        csa_dense_mode=False,
        csa_compress_rotary_base=shape["csa_compress_rotary_base"],
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        qk_layernorm=True,
        layernorm_zero_centered_gamma=False,
        expert_model_parallel_size=1,
        apply_dsa_kernel_fusion=apply_dsa_kernel_fusion,
        apply_rope_fusion=apply_rope_fusion,
    )


def _cp_debug_trace(message: str) -> None:
    if not os.environ.get("DSV4_CP_DEBUG_TRACE"):
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else "?"
    print(f"[dsv4-cp-test rank={rank}] {message}", flush=True)


class _SingleRankCPGroup:

    def rank(self):
        return 0

    def size(self):
        return 1


def _copy_module_parameters(src, dst):
    src_params = dict(src.named_parameters())
    for name, param in dst.named_parameters():
        assert name in src_params
        param.data.copy_(src_params[name].data)
    return src_params


def _make_contiguous_cp_partition_indices(padded_total_tokens, cp_size, device='cuda'):
    assert padded_total_tokens % cp_size == 0
    local_tokens = padded_total_tokens // cp_size
    return tuple(
        torch.arange(
            rank * local_tokens,
            (rank + 1) * local_tokens,
            device=device,
            dtype=torch.long,
        )
        for rank in range(cp_size)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionTHDCP:
    """CP-sliced THD attention should match full-sequence THD reference output and gradients."""

    @pytest.fixture(scope='class', autouse=True, params=(2, 4), ids=lambda cp: f"cp{cp}")
    def setup_method(self, request):
        cp_size = request.param
        if Utils.world_size < cp_size:
            pytest.skip(f"THD CP path test requires at least {cp_size} distributed ranks")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.cp_size = cp_size
        cls.cp_rank = parallel_state.get_context_parallel_rank()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()
        # Tradeoff: reuse the current model-parallel groups and only disable CP for the
        # full-reference path to avoid extra group initialization. This is valid for
        # these CP-only tests; rebuild the reference groups if future tests add other
        # parallel dimensions.
        cls.ref_pg = ProcessGroupCollection.use_mpu_process_groups()
        cls.ref_pg.cp = _SingleRankCPGroup()

        yield
        Utils.destroy_model_parallel()

    def test_left_boundary_exchange_forward_backward(self):
        """CP boundary exchange receives the previous rank's tail window.

        In backward, the current rank's tail tokens receive gradient when the
        next rank used those tokens as its left boundary.
        """
        d_window = 2
        local_len = 4
        width = 3
        # The expected tensors below assume a positive tail window fully inside
        # the local tensor, with non-empty feature rows.
        assert d_window > 0
        assert local_len >= d_window
        assert width > 0
        local_numel = local_len * width
        local_start = self.cp_rank * local_numel
        values = torch.arange(
            local_start,
            local_start + local_numel,
            device='cuda',
            dtype=torch.float32,
        ).reshape(local_len, width)
        local = values.detach().clone().requires_grad_(True)

        boundary = exchange_left_boundary_tensor(local, d_window, self.pg.cp)
        if self.cp_rank == 0:
            # Rank 0 has no previous CP rank, so the fixed left boundary is zero-filled.
            expected_boundary = torch.zeros_like(boundary)
        else:
            # Nonzero ranks receive the previous rank's final d_window rows as their boundary.
            left_rank_start = (self.cp_rank - 1) * local_numel
            expected_boundary = torch.arange(
                left_rank_start + (local_len - d_window) * width,
                left_rank_start + local_numel,
                device='cuda',
                dtype=torch.float32,
            ).reshape(d_window, width)
        assert torch.equal(boundary, expected_boundary)

        boundary.sum().backward()
        # Local tokens receive no boundary-exchange grad unless the next rank reads them.
        expected_grad = torch.zeros_like(local)
        if self.cp_rank + 1 < self.cp_size:
            # The next rank uses this rank's tail as its left boundary, so boundary.sum()
            # contributes unit gradient to exactly those tail rows.
            expected_grad[-d_window:] = 1
        assert torch.equal(local.grad, expected_grad)

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2, 3],
        ids=["ratio_0_window_only", "ratio_4_indexer", "ratio_128_compressor"],
    )
    def test_thd_cp_matches_full_reference_forward_backward(self, layer_number):
        """CP path matches the full-sequence THD reference on ragged
        packed inputs using the DSv4 layer configuration.

        Verifies local CP output, hidden grad, and reduced parameter grads
        against the sliced full-reference tensors.
        """
        seg_lens = _DSV4_CP_RAGGED_SEG_LENS
        padded_seg_lens = _DSV4_CP_RAGGED_PADDED_SEG_LENS
        _cp_debug_trace(
            f"test start layer={layer_number} seg_lens={seg_lens} "
            f"padded_seg_lens={padded_seg_lens}"
        )
        padded_tokens = sum(padded_seg_lens)
        packed = _make_thd_packed_seq_params(seg_lens, padded_seg_lens)
        partition_indices = _make_contiguous_cp_partition_indices(padded_tokens, self.cp_size)
        local_idx = partition_indices[self.cp_rank]

        torch.manual_seed(_SEED + layer_number)
        model_parallel_cuda_manual_seed(_SEED + layer_number)
        config_cp = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
        )
        config_ref = _make_dsv4_cp_config(
            context_parallel_size=1,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
        )
        cp_attn = _build_attention(
            config_cp, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        ref_attn = _build_attention(
            config_ref, layer_number=layer_number, pg_collection=self.ref_pg
        ).cuda()
        _copy_module_parameters(cp_attn, ref_attn)

        full_hidden = torch.randn(
            padded_tokens, 1, config_cp.hidden_size, dtype=torch.bfloat16, device='cuda'
        )
        local_hidden = full_hidden.index_select(0, local_idx).detach().clone().requires_grad_(True)
        ref_hidden = full_hidden.detach().clone().requires_grad_(True)

        _cp_debug_trace("cp forward start")
        local_out, _ = cp_attn(
            hidden_states=local_hidden, attention_mask=None, packed_seq_params=packed
        )
        _cp_debug_trace("cp forward done")
        _cp_debug_trace("reference forward start")
        ref_out, _ = ref_attn(
            hidden_states=ref_hidden, attention_mask=None, packed_seq_params=packed
        )
        _cp_debug_trace("reference forward done")
        _assert_cp_tensor_match(
            local_out.detach(),
            ref_out.detach().index_select(0, local_idx),
            f"layer={layer_number}:output",
        )

        grad = torch.randn_like(ref_out)
        _cp_debug_trace("cp backward start")
        local_out.backward(grad.index_select(0, local_idx))
        _cp_debug_trace("cp backward done")
        _cp_debug_trace("reference backward start")
        ref_out.backward(grad)
        _cp_debug_trace("reference backward done")
        _assert_cp_tensor_match(
            local_hidden.grad.detach(),
            ref_hidden.grad.index_select(0, local_idx),
            f"layer={layer_number}:hidden_grad",
        )

        ref_params = dict(ref_attn.named_parameters())
        for name, param in cp_attn.named_parameters():
            ref_grad = ref_params[name].grad
            assert param.grad is not None, f"Missing CP grad for {name}"
            assert ref_grad is not None, f"Missing reference grad for {name}"
            grad_sum = param.grad.detach().clone()
            _cp_debug_trace(f"param grad all_reduce start {name}")
            dist.all_reduce(grad_sum, group=self.pg.cp)
            _cp_debug_trace(f"param grad all_reduce done {name}")
            _assert_cp_tensor_match(grad_sum, ref_grad, f"layer={layer_number}:param_grad:{name}")

        del cp_attn, ref_attn, full_hidden, local_hidden, ref_hidden, local_out, ref_out, grad
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2, 3],
        ids=["ratio_0_window_only", "ratio_4_indexer", "ratio_128_compressor"],
    )
    def test_thd_cp_fused_rope_matches_unfused_local_position_path(self, layer_number):
        """Fused RoPE and unfused RoPE match on the same CP-local THD shard.

        The unfused path passes explicit CP-aware positions into RoPE. This
        verifies local output, hidden grad, and parameter grads.
        """
        seg_lens = _DSV4_CP_RAGGED_SEG_LENS
        padded_seg_lens = _DSV4_CP_RAGGED_PADDED_SEG_LENS
        padded_tokens = sum(padded_seg_lens)
        packed = _make_thd_packed_seq_params(seg_lens, padded_seg_lens)
        partition_indices = _make_contiguous_cp_partition_indices(padded_tokens, self.cp_size)
        local_idx = partition_indices[self.cp_rank]

        torch.manual_seed(_SEED + 300 + layer_number)
        model_parallel_cuda_manual_seed(_SEED + 300 + layer_number)
        fused_config = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_dsa_kernel_fusion=True,
            apply_rope_fusion=True,
        )
        explicit_position_config = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_dsa_kernel_fusion=True,
            apply_rope_fusion=False,
        )
        fused_attn = _build_attention(
            fused_config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        explicit_position_attn = _build_attention(
            explicit_position_config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        _copy_module_parameters(fused_attn, explicit_position_attn)

        full_hidden = torch.randn(
            padded_tokens, 1, fused_config.hidden_size, dtype=torch.bfloat16, device='cuda'
        )
        fused_hidden = full_hidden.index_select(0, local_idx).detach().clone().requires_grad_(True)
        explicit_position_hidden = (
            full_hidden.index_select(0, local_idx).detach().clone().requires_grad_(True)
        )

        fused_out, _ = fused_attn(
            hidden_states=fused_hidden, attention_mask=None, packed_seq_params=packed
        )
        explicit_position_out, _ = explicit_position_attn(
            hidden_states=explicit_position_hidden, attention_mask=None, packed_seq_params=packed
        )
        _assert_cp_tensor_match(
            fused_out.detach(),
            explicit_position_out.detach(),
            "apply_rope_fusion:output",
        )

        grad = torch.randn_like(fused_out)
        fused_out.backward(grad)
        explicit_position_out.backward(grad)
        _assert_cp_tensor_match(
            fused_hidden.grad.detach(),
            explicit_position_hidden.grad.detach(),
            "apply_rope_fusion:hidden_grad",
        )

        explicit_position_params = dict(explicit_position_attn.named_parameters())
        for name, param in fused_attn.named_parameters():
            explicit_position_grad = explicit_position_params[name].grad
            assert param.grad is not None, f"Missing fused grad for {name}"
            assert explicit_position_grad is not None, f"Missing explicit-position grad for {name}"
            _assert_cp_tensor_match(
                param.grad.detach(),
                explicit_position_grad.detach(),
                f"apply_rope_fusion:param_grad:{name}",
            )

        del fused_attn, explicit_position_attn, full_hidden, fused_hidden
        del explicit_position_hidden, fused_out, explicit_position_out, grad
        gc.collect()
        torch.cuda.empty_cache()
