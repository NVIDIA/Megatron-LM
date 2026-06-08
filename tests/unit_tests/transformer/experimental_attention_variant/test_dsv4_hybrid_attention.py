# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import gc
import os
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import megatron.core.parallel_state as parallel_state
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    SINGLE_RANK_CP_GROUP,
    unfused_precomputed_indexer_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import indexer_topk
from megatron.core.transformer.experimental_attention_variant.csa_cp_utils import (
    DSV4_CP_PARTITION_CONTIGUOUS,
    DSV4_CP_PARTITION_PACKED_STREAM_TWO_CHUNK,
    all_gather_fixed_cp_tensor,
    build_chunked_compressor_prep_compact_fused,
    build_chunked_cp_flat_idxs_for_indexer_loss_fused,
    build_chunked_rank_major_compressed_metadata_fused,
    build_global_compressed_cu_seqlens_fused,
    chunked_cp_partition,
    compute_chunked_cp_indexer_topk_logical_fused,
    exchange_left_boundary_tensor,
    repack_rank_major_compressed_to_seq_major_fused,
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

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2, 3],
        ids=["ratio_0_window_only", "ratio_4_with_indexer", "ratio_128_compressor_only"],
    )
    def test_thd_cuda_graph_matches_eager_forward_backward(self, layer_number):
        """CUDA graph replay matches eager THD forward/backward without CP.

        This covers the no-CP THD path that computes compressed sequence
        metadata from ``PackedSeqParams`` before capture. A failure here usually
        means graph capture found a GPU-to-CPU sync or the graph replay math no
        longer matches eager execution.
        """
        seg_lens = _DSV4_CP_RAGGED_SEG_LENS
        padded_seg_lens = _DSV4_CP_RAGGED_PADDED_SEG_LENS
        padded_tokens = sum(padded_seg_lens)
        packed = _make_thd_packed_seq_params(seg_lens, padded_seg_lens)

        torch.manual_seed(_SEED + 500 + layer_number)
        model_parallel_cuda_manual_seed(_SEED + 500 + layer_number)
        config = _make_dsv4_cp_config(
            context_parallel_size=1,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_rope_fusion=True,
        )
        graph_attn = _build_attention(config, layer_number=layer_number, pg_collection=self.pg).cuda()
        eager_attn = _build_attention(config, layer_number=layer_number, pg_collection=self.pg).cuda()
        graph_attn.train()
        eager_attn.train()
        _copy_module_parameters(graph_attn, eager_attn)

        test_hidden = torch.randn(
            padded_tokens, 1, config.hidden_size, dtype=torch.bfloat16, device='cuda'
        )
        test_grad = torch.randn_like(test_hidden)
        static_hidden = test_hidden.detach().clone().requires_grad_(True)
        eager_hidden = test_hidden.detach().clone().requires_grad_(True)
        static_grad = test_grad.detach().clone()

        graph, graph_output = _capture_dsv4_attention_forward_backward(
            graph_attn, static_hidden, static_grad, packed
        )
        with torch.no_grad():
            static_hidden.copy_(test_hidden)
            static_grad.copy_(test_grad)
            if static_hidden.grad is not None:
                static_hidden.grad.zero_()
            for param in graph_attn.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        graph.replay()
        graph_out = graph_output.detach().clone()
        graph_hidden_grad = static_hidden.grad.detach().clone()
        graph_param_grads = {
            name: param.grad.detach().clone()
            for name, param in graph_attn.named_parameters()
            if param.grad is not None
        }

        eager_out, eager_hidden_grad, eager_param_grads = _run_dsv4_attention_forward_backward(
            eager_attn, eager_hidden, test_grad, packed
        )
        _assert_cp_graph_match(graph_out, eager_out, f"layer={layer_number}:output")
        _assert_cp_graph_match(
            graph_hidden_grad, eager_hidden_grad, f"layer={layer_number}:hidden_grad"
        )
        for name, graph_grad in graph_param_grads.items():
            assert name in eager_param_grads, f"Missing eager grad for {name}"
            _assert_cp_graph_match(
                graph_grad, eager_param_grads[name], f"layer={layer_number}:param_grad:{name}"
            )

        del graph_attn, eager_attn, test_hidden, test_grad, static_hidden
        del eager_hidden, static_grad, graph, graph_output, graph_out, graph_hidden_grad
        gc.collect()
        torch.cuda.empty_cache()

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
_DSV4_CP_GRAPH_FUSED_SIM_EPS = 1e-6
_DSV4_CP_GRAPH_FUSED_RTOL = 1e-6
# Fused BF16 backward uses atomic accumulation; graph/eager order can differ by
# one BF16-sized absolute step while preserving strict vector similarity.
_DSV4_CP_GRAPH_FUSED_BF16_ATOL = 1.0
_DSV4_CP_GRAPH_FUSED_FP32_ATOL = 1e-2
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
    actual_norm = actual.double().norm().item()
    expected_norm = expected.double().norm().item()
    assert cosine_sim > 1 - _DSV4_CP_PARITY_EPS, (
        f"{label}: cosine_sim={cosine_sim:.10f}, "
        f"tensor_sim={tensor_sim:.10f}, max_abs={max_abs:.6e}, "
        f"actual_norm={actual_norm:.6e}, expected_norm={expected_norm:.6e}, "
        f"eps={_DSV4_CP_PARITY_EPS}"
    )
    assert tensor_sim > 1 - _DSV4_CP_PARITY_EPS, (
        f"{label}: tensor_sim={tensor_sim:.10f}, "
        f"cosine_sim={cosine_sim:.10f}, max_abs={max_abs:.6e}, "
        f"actual_norm={actual_norm:.6e}, expected_norm={expected_norm:.6e}, "
        f"eps={_DSV4_CP_PARITY_EPS}"
    )


def _assert_cp_graph_match(actual: torch.Tensor, expected: torch.Tensor, label: str):
    assert actual.shape == expected.shape, (
        f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    if torch.equal(actual, expected):
        return
    _assert_cp_tensor_match(actual, expected, label)


def _assert_cp_graph_bitwise_match(actual: torch.Tensor, expected: torch.Tensor, label: str):
    assert actual.shape == expected.shape, (
        f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    if torch.equal(actual, expected):
        return
    diff = (actual - expected).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    cosine_sim = _cosine_sim(actual, expected)
    tensor_sim = _tensor_sim(actual, expected)
    raise AssertionError(
        f"{label}: graph/eager must be bitwise equal; max_abs={max_abs:.6e}, "
        f"cosine_sim={cosine_sim:.10f}, tensor_sim={tensor_sim:.10f}"
    )


def _assert_cp_graph_fused_backward_match(
    actual: torch.Tensor, expected: torch.Tensor, label: str
):
    assert actual.shape == expected.shape, (
        f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    assert actual.dtype == expected.dtype, f"{label}: dtype {actual.dtype} != {expected.dtype}"
    assert torch.isfinite(actual).all(), f"{label}: actual has non-finite values"
    assert torch.isfinite(expected).all(), f"{label}: expected has non-finite values"

    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    cosine_sim = _cosine_sim(actual, expected)
    tensor_sim = _tensor_sim(actual, expected)
    assert cosine_sim > 1 - _DSV4_CP_GRAPH_FUSED_SIM_EPS, (
        f"{label}: cosine_sim={cosine_sim:.10f}, "
        f"max_abs={max_abs:.6e}, eps={_DSV4_CP_GRAPH_FUSED_SIM_EPS}"
    )
    assert tensor_sim > 1 - _DSV4_CP_GRAPH_FUSED_SIM_EPS, (
        f"{label}: tensor_sim={tensor_sim:.10f}, "
        f"max_abs={max_abs:.6e}, eps={_DSV4_CP_GRAPH_FUSED_SIM_EPS}"
    )

    if actual.dtype == torch.bfloat16:
        atol = _DSV4_CP_GRAPH_FUSED_BF16_ATOL
    elif actual.dtype == torch.float32:
        atol = _DSV4_CP_GRAPH_FUSED_FP32_ATOL
    else:
        raise AssertionError(f"{label}: unsupported dtype for fused graph close check: {actual.dtype}")
    torch.testing.assert_close(
        actual,
        expected,
        rtol=_DSV4_CP_GRAPH_FUSED_RTOL,
        atol=atol,
        msg=(
            f"{label}: fused graph/eager backward mismatch; "
            f"rtol={_DSV4_CP_GRAPH_FUSED_RTOL}, atol={atol}, "
            f"cosine_sim={cosine_sim:.10f}, tensor_sim={tensor_sim:.10f}, "
            f"max_abs={max_abs:.6e}"
        ),
    )


@contextmanager
def _deterministic_torch_algorithms():
    old_enabled = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old_enabled, warn_only=old_warn_only)


def _make_dsv4_cp_config(
    *,
    context_parallel_size,
    dsa_indexer_loss_coeff=0.0,
    dsa_indexer_use_sparse_loss=True,
    apply_dsa_kernel_fusion=True,
    apply_rope_fusion=False,
    dsv4_cp_partition_mode=DSV4_CP_PARTITION_CONTIGUOUS,
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
        dsv4_cp_partition_mode=dsv4_cp_partition_mode,
    )


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


def _make_chunked_cp_partition_indices(padded_total_tokens, cp_size, device='cuda'):
    assert padded_total_tokens % (2 * cp_size) == 0
    return tuple(
        torch.cat(
            [
                torch.arange(start, end, device=device, dtype=torch.long)
                for start, end in chunked_cp_partition(padded_total_tokens, cp_size, rank)
            ],
            dim=0,
        )
        for rank in range(cp_size)
    )


def _run_dsv4_attention_forward_backward(
    attn, hidden, grad, packed_seq_params, *, collect_result=True
):
    hidden.grad = None
    attn.zero_grad(set_to_none=True)
    output, _ = attn(hidden_states=hidden, attention_mask=None, packed_seq_params=packed_seq_params)
    output.backward(grad)
    if not collect_result:
        return None
    param_grads = {
        name: param.grad.detach().clone()
        for name, param in attn.named_parameters()
        if param.grad is not None
    }
    return output.detach().clone(), hidden.grad.detach().clone(), param_grads


def _capture_dsv4_attention_forward_backward(attn, static_hidden, static_grad, packed_seq_params):
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            _run_dsv4_attention_forward_backward(
                attn, static_hidden, static_grad, packed_seq_params, collect_result=False
            )
    torch.cuda.current_stream().wait_stream(warmup_stream)

    static_hidden.grad = None
    attn.zero_grad(set_to_none=True)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        graph_output, _ = attn(
            hidden_states=static_hidden,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
        )
        graph_output.backward(static_grad)
    return graph, graph_output


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

    @pytest.mark.parametrize(
        "partition_mode, uses_two_chunk",
        [
            (DSV4_CP_PARTITION_CONTIGUOUS, False),
            (DSV4_CP_PARTITION_PACKED_STREAM_TWO_CHUNK, True),
        ],
        ids=["contiguous", "packed_stream_two_chunk"],
    )
    def test_thd_cp_partition_mode_selects_expected_row_order(
        self, partition_mode, uses_two_chunk
    ):
        """DSv4 CP partition mode controls the attention-layer row order.

        Expected: ``contiguous`` keeps the original one-range CP partition,
        while ``packed_stream_two_chunk`` makes this rank own chunk ``rank`` and
        chunk ``2*cp_size-1-rank``. A failure here means the config knob could
        drift from the layout helpers even if lower-level utility tests pass.
        """
        config = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsv4_cp_partition_mode=partition_mode,
        )
        attn = _build_attention(config, layer_number=1, pg_collection=self.pg).cuda()

        assert attn._use_packed_stream_two_chunk_cp_partition() == uses_two_chunk
        if uses_two_chunk:
            l_local = 16
            assert attn._packed_stream_two_chunk_cp_ranges(l_local) == chunked_cp_partition(
                l_local * self.cp_size, self.cp_size, self.cp_rank
            )

        del attn
        gc.collect()
        torch.cuda.empty_cache()

    def test_thd_cp_partition_mode_rejects_unknown_value(self):
        """Unknown DSv4 CP partition modes fail before a forward pass.

        Expected: the attention layer rejects stale values such as the old
        chunked bool spelling instead of silently falling back to contiguous
        mode. A failure here could hide a misconfigured benchmark or test.
        """
        config = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsv4_cp_partition_mode="chunked",
        )
        attn = _build_attention(config, layer_number=1, pg_collection=self.pg).cuda()

        with pytest.raises(RuntimeError, match="Unsupported DSv4 CP partition mode"):
            attn._use_packed_stream_two_chunk_cp_partition()

        del attn
        gc.collect()
        torch.cuda.empty_cache()

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
            apply_rope_fusion=True,
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

        local_out, _ = cp_attn(
            hidden_states=local_hidden, attention_mask=None, packed_seq_params=packed
        )
        ref_out, _ = ref_attn(
            hidden_states=ref_hidden, attention_mask=None, packed_seq_params=packed
        )
        _assert_cp_tensor_match(
            local_out.detach(),
            ref_out.detach().index_select(0, local_idx),
            f"layer={layer_number}:output",
        )

        grad = torch.randn_like(ref_out)
        local_out.backward(grad.index_select(0, local_idx))
        ref_out.backward(grad)
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
            dist.all_reduce(grad_sum, group=self.pg.cp)
            _assert_cp_tensor_match(grad_sum, ref_grad, f"layer={layer_number}:param_grad:{name}")

        del cp_attn, ref_attn, full_hidden, local_hidden, ref_hidden, local_out, ref_out, grad
        gc.collect()
        torch.cuda.empty_cache()

    def test_thd_chunked_cp_indexer_inputs_match_full_reference(self):
        """Chunked CP indexer Q/weights and compressed K match full THD reference."""
        seg_lens = _DSV4_CP_RAGGED_SEG_LENS
        padded_seg_lens = _DSV4_CP_RAGGED_PADDED_SEG_LENS
        padded_tokens = sum(padded_seg_lens)
        packed = _make_thd_packed_seq_params(seg_lens, padded_seg_lens)
        partition_indices = _make_chunked_cp_partition_indices(padded_tokens, self.cp_size)
        local_idx = partition_indices[self.cp_rank]

        torch.manual_seed(_SEED + 400)
        model_parallel_cuda_manual_seed(_SEED + 400)
        config_cp = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_rope_fusion=True,
            dsv4_cp_partition_mode=DSV4_CP_PARTITION_PACKED_STREAM_TWO_CHUNK,
        )
        config_ref = _make_dsv4_cp_config(
            context_parallel_size=1,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
        )
        cp_attn = _build_attention(config_cp, layer_number=2, pg_collection=self.pg).cuda()
        ref_attn = _build_attention(config_ref, layer_number=2, pg_collection=self.ref_pg).cuda()
        _copy_module_parameters(cp_attn, ref_attn)

        full_hidden = torch.randn(
            padded_tokens, 1, config_cp.hidden_size, dtype=torch.bfloat16, device='cuda'
        )
        local_hidden = full_hidden.index_select(0, local_idx).detach().clone()
        ref_hidden = full_hidden.detach().clone()

        query_local, _, _, qr_local, _ = cp_attn.get_query_key_value_tensors(
            local_hidden, packed_seq_params=packed
        )
        query_ref, _, _, qr_ref, _ = ref_attn.get_query_key_value_tensors(
            ref_hidden, packed_seq_params=packed
        )

        core_cp = cp_attn.core_attention
        core_ref = ref_attn.core_attention
        cu_seqlens = packed.cu_seqlens_q_padded
        chunk_ranges = chunked_cp_partition(padded_tokens, self.cp_size, self.cp_rank)
        chunk_len = chunk_ranges[0][1] - chunk_ranges[0][0]
        d_window = cp_attn._dsv4_cp_boundary_window()
        d_comp = 8
        ratio = 4

        q_local, weights_local = core_cp.indexer._forward_thd_query_weights_cp(
            local_hidden.detach(),
            qr_local.detach(),
            cu_seqlens,
            self.cp_rank,
            self.cp_size,
            local_hidden.shape[0],
            int(packed.max_seqlen_q),
            chunk_ranges=chunk_ranges,
        )
        q_ref, k_ref, weights_ref, cu_ref = core_ref.indexer.forward_before_topk(
            ref_hidden.detach(), qr_ref.detach(), packed
        )
        _assert_cp_tensor_match(
            q_local.detach(),
            q_ref.squeeze(1).detach().index_select(0, local_idx),
            "chunked-indexer:q",
        )
        _assert_cp_tensor_match(
            weights_local.detach(),
            weights_ref.squeeze(1).detach().index_select(0, local_idx),
            "chunked-indexer:weights",
        )

        boundary_hidden = cp_attn._exchange_cp_boundary_hidden(local_hidden, d_window)
        (
            hidden_compact,
            cu_compact,
            _seq_ids_local,
            comp_ids_local,
            _valid_local,
            c_cap,
            c_cap_per_chunk,
        ) = build_chunked_compressor_prep_compact_fused(
            local_hidden,
            boundary_hidden,
            cu_seqlens,
            chunk_ranges,
            ratio,
            d_comp,
            d_window,
        )
        k_local, _ = core_cp.indexer.compressor._forward_thd(
            hidden_compact.detach(),
            cu_compact,
            max_seqlen_q=int(packed.max_seqlen_q),
            cp_group=SINGLE_RANK_CP_GROUP,
            rope_positions=comp_ids_local,
            fixed_total_comp=c_cap,
            pre_grouped_compact_input=True,
        )
        k_rank_major = all_gather_fixed_cp_tensor(k_local.squeeze(1), self.pg.cp)
        seq_ids, comp_ids, valid = build_chunked_rank_major_compressed_metadata_fused(
            cu_seqlens,
            self.cp_size,
            chunk_len,
            ratio,
            d_comp,
            c_cap_per_chunk,
        )
        cu_compressed = build_global_compressed_cu_seqlens_fused(cu_seqlens, ratio)
        k_seq_major, _rank_by_seq_major = repack_rank_major_compressed_to_seq_major_fused(
            k_rank_major,
            seq_ids,
            comp_ids,
            valid,
            cu_compressed,
            output_capacity=(padded_tokens // ratio),
        )
        actual_comp = int(cu_ref[-1].item())
        _assert_cp_tensor_match(
            k_seq_major[:actual_comp].detach(),
            k_ref.squeeze(1).detach()[:actual_comp],
            "chunked-indexer:k_seq_major",
        )

        topk_width = core_cp.indexer.index_topk
        max_seqlen_compressed_idx = int(packed.max_seqlen_q) // ratio
        cp_topk = compute_chunked_cp_indexer_topk_logical_fused(
            q_local,
            weights_local,
            k_seq_major,
            cu_seqlens,
            cu_compressed,
            chunk_ranges,
            ratio,
            topk_width,
            core_cp.indexer.softmax_scale,
            max_seqlen_q=int(packed.max_seqlen_q),
            max_seqlen_kv=max_seqlen_compressed_idx,
        )
        ref_topk, _ = indexer_topk(
            q_ref.squeeze(1),
            k_ref.squeeze(1),
            weights_ref.squeeze(1),
            topk=min(topk_width, max_seqlen_compressed_idx),
            ratio=ratio,
            indexer_softmax_scale=core_ref.indexer.softmax_scale,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_ref,
            max_seqlen_q=int(packed.max_seqlen_q),
            max_seqlen_kv=max_seqlen_compressed_idx,
            fixed_topk_width=topk_width,
            compute_topk_length=False,
        )
        ref_topk_local = ref_topk.index_select(0, local_idx)
        assert torch.equal(cp_topk.cpu(), ref_topk_local.cpu()), (
            "chunked-indexer:topk logical mismatch: "
            f"num_diff={(cp_topk != ref_topk_local).sum().item()}"
        )
        _loss_topk, cp_rank_major_topk = build_chunked_cp_flat_idxs_for_indexer_loss_fused(
            cu_seqlens,
            cu_compressed,
            chunk_ranges,
            tuple(0 for _ in chunk_ranges),
            d_window,
            core_cp.window_size,
            ratio,
            cp_topk,
            _rank_by_seq_major,
        )
        row_seq = torch.searchsorted(cu_seqlens, local_idx.to(cu_seqlens.dtype), right=True) - 1
        row_seq = row_seq.clamp(min=0, max=cu_seqlens.numel() - 2)
        seq_comp_start = cu_compressed.index_select(0, row_seq).unsqueeze(1)
        safe_topk = ref_topk_local.clamp(min=0).to(torch.long)
        safe_seq_major = (seq_comp_start.to(torch.long) + safe_topk).reshape(-1)
        expected_rank_major = _rank_by_seq_major.index_select(0, safe_seq_major).reshape_as(
            ref_topk_local
        )
        expected_rank_major = torch.where(
            ref_topk_local >= 0,
            expected_rank_major,
            torch.full_like(expected_rank_major, -1),
        )
        assert torch.equal(cp_rank_major_topk.cpu(), expected_rank_major.cpu()), (
            "chunked-indexer:rank-major topk mismatch: "
            f"num_diff={(cp_rank_major_topk != expected_rank_major).sum().item()}"
        )

        attn_compressed_local, _ = core_cp.compressor._forward_thd(
            hidden_compact.detach(),
            cu_compact,
            max_seqlen_q=int(packed.max_seqlen_q),
            cp_group=SINGLE_RANK_CP_GROUP,
            rope_positions=comp_ids_local,
            fixed_total_comp=c_cap,
            pre_grouped_compact_input=True,
        )
        attn_compressed_rank_major = all_gather_fixed_cp_tensor(
            attn_compressed_local.squeeze(1), self.pg.cp
        )
        attn_compressed_ref, _ = core_ref.compressor(
            ref_hidden.detach(), packed_seq_params=packed
        )

        dummy_topk_local = torch.full(
            (local_hidden.shape[0], 1), -1, dtype=torch.int32, device='cuda'
        )
        dummy_topk_ref = torch.full((padded_tokens, 1), -1, dtype=torch.int32, device='cuda')
        dummy_kv_local = query_local.new_zeros((1, query_local.shape[-1]))
        dummy_kv_ref = query_ref.new_zeros((1, query_ref.shape[-1]))

        k_cp_for_loss = k_rank_major.detach().clone().requires_grad_(True)
        _out_cp, cp_loss = unfused_precomputed_indexer_sparse_attn(
            query_local.detach(),
            dummy_kv_local,
            core_cp.attn_sink.float().detach(),
            dummy_topk_local,
            q_local.detach(),
            k_cp_for_loss,
            weights_local.detach(),
            cp_rank_major_topk.int(),
            attn_compressed_rank_major.detach(),
            core_cp.softmax_scale,
            core_cp.indexer.softmax_scale,
            1.0,
            False,
            padded_tokens,
        )

        global_rows = torch.arange(padded_tokens, dtype=cu_seqlens.dtype, device='cuda')
        ref_row_seq = torch.searchsorted(cu_seqlens, global_rows, right=True) - 1
        ref_row_seq = ref_row_seq.clamp(min=0, max=cu_seqlens.numel() - 2)
        ref_seq_major_topk = cu_ref.index_select(0, ref_row_seq).unsqueeze(1).to(
            torch.long
        ) + ref_topk.clamp(min=0).to(torch.long)
        ref_seq_major_topk = torch.where(
            ref_topk >= 0,
            ref_seq_major_topk,
            torch.full_like(ref_seq_major_topk, -1),
        )
        k_ref_for_loss = k_ref.squeeze(1).detach().clone().requires_grad_(True)
        _out_ref, ref_loss = unfused_precomputed_indexer_sparse_attn(
            query_ref.detach(),
            dummy_kv_ref,
            core_ref.attn_sink.float().detach(),
            dummy_topk_ref,
            q_ref.squeeze(1).detach(),
            k_ref_for_loss,
            weights_ref.squeeze(1).detach(),
            ref_seq_major_topk.int(),
            attn_compressed_ref.squeeze(1).detach(),
            core_ref.softmax_scale,
            core_ref.indexer.softmax_scale,
            1.0,
            False,
            padded_tokens,
        )
        cp_loss_total = cp_loss.detach().clone()
        dist.all_reduce(cp_loss_total, group=self.pg.cp)
        _assert_cp_tensor_match(
            cp_loss_total.reshape(1),
            ref_loss.detach().reshape(1),
            "chunked-indexer:loss",
        )

        cp_loss.backward()
        dk_cp_local = k_cp_for_loss.grad.detach().clone()
        dk_cp = dk_cp_local.clone()
        dist.all_reduce(dk_cp, group=self.pg.cp)
        ref_loss.backward()
        dk_ref_seq_major = k_ref_for_loss.grad.detach().clone()
        expected_dk_rank_major = torch.zeros_like(dk_cp)
        expected_dk_rank_major.index_copy_(
            0,
            _rank_by_seq_major[:actual_comp].to(torch.long),
            dk_ref_seq_major[:actual_comp],
        )
        _assert_cp_tensor_match(
            dk_cp,
            expected_dk_rank_major,
            "chunked-indexer:dK_rank_major",
        )

        core_cp.indexer.compressor.zero_grad(set_to_none=True)
        core_ref.indexer.compressor.zero_grad(set_to_none=True)
        k_rank_major.backward(dk_cp_local)
        k_ref.backward(dk_ref_seq_major.unsqueeze(1))
        cp_ape_grad = core_cp.indexer.compressor.ape.grad.detach().clone()
        dist.all_reduce(cp_ape_grad, group=self.pg.cp)
        _assert_cp_tensor_match(
            cp_ape_grad,
            core_ref.indexer.compressor.ape.grad.detach(),
            "chunked-indexer:compressor_ape_grad_from_dK",
        )

        del cp_attn, ref_attn, full_hidden, local_hidden, ref_hidden
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2, 3],
        ids=["ratio_0_window_only", "ratio_4_indexer", "ratio_128_compressor"],
    )
    def test_thd_chunked_cp_matches_full_reference_forward_backward(self, layer_number):
        """Two-chunk THD CP path matches full-sequence THD reference.

        Verifies the packed-stream two-chunk row order: rank r owns
        chunk r followed by chunk 2*cp_size-1-r. Sparse attention still runs as
        one call; only the layout metadata, boundary windows, RoPE positions,
        and indexer top-k inputs are chunk-aware.
        """
        seg_lens = _DSV4_CP_RAGGED_SEG_LENS
        padded_seg_lens = _DSV4_CP_RAGGED_PADDED_SEG_LENS
        padded_tokens = sum(padded_seg_lens)
        packed = _make_thd_packed_seq_params(seg_lens, padded_seg_lens)
        partition_indices = _make_chunked_cp_partition_indices(padded_tokens, self.cp_size)
        local_idx = partition_indices[self.cp_rank]

        torch.manual_seed(_SEED + 300 + layer_number)
        model_parallel_cuda_manual_seed(_SEED + 300 + layer_number)
        config_cp = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_rope_fusion=True,
            dsv4_cp_partition_mode=DSV4_CP_PARTITION_PACKED_STREAM_TWO_CHUNK,
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

        local_out, _ = cp_attn(
            hidden_states=local_hidden, attention_mask=None, packed_seq_params=packed
        )
        ref_out, _ = ref_attn(
            hidden_states=ref_hidden, attention_mask=None, packed_seq_params=packed
        )
        _assert_cp_tensor_match(
            local_out.detach(),
            ref_out.detach().index_select(0, local_idx),
            f"chunked layer={layer_number}:output",
        )

        grad = torch.randn_like(ref_out)
        local_out.backward(grad.index_select(0, local_idx))
        ref_out.backward(grad)
        _assert_cp_tensor_match(
            local_hidden.grad.detach(),
            ref_hidden.grad.index_select(0, local_idx),
            f"chunked layer={layer_number}:hidden_grad",
        )

        ref_params = dict(ref_attn.named_parameters())
        for name, param in cp_attn.named_parameters():
            ref_grad = ref_params[name].grad
            assert param.grad is not None, f"Missing CP grad for {name}"
            assert ref_grad is not None, f"Missing reference grad for {name}"
            grad_sum = param.grad.detach().clone()
            dist.all_reduce(grad_sum, group=self.pg.cp)
            _assert_cp_tensor_match(
                grad_sum, ref_grad, f"chunked layer={layer_number}:param_grad:{name}"
            )

        del cp_attn, ref_attn, full_hidden, local_hidden, ref_hidden, local_out, ref_out, grad
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2, 3],
        ids=["ratio_0_window_only", "ratio_4_indexer", "ratio_128_compressor"],
    )
    @pytest.mark.parametrize("fused", [True, False])
    @pytest.mark.parametrize(
        "partition_mode",
        [DSV4_CP_PARTITION_CONTIGUOUS, DSV4_CP_PARTITION_PACKED_STREAM_TWO_CHUNK],
        ids=["contiguous", "packed_stream_two_chunk"],
    )
    def test_thd_cp_cuda_graph_matches_eager_forward_backward(
        self, layer_number, fused, partition_mode
    ):
        """CUDA graph replay matches eager THD CP forward/backward.

        Captures the DSv4 attention layer's CP-local forward and backward
        graph, replays it with fresh static-buffer contents, and compares the
        local output, hidden grad, and parameter grads against an eager module
        with identical weights. The fused sparse-attn/indexer forward kernels
        are deterministic, so fused output must match bitwise. Fused backward
        uses atomic adds inside the sparse-attn/indexer kernels, so accumulation
        order can vary and backward uses strict similarity plus elementwise
        ``assert_close`` gates. The unfused sparse-attn/indexer path is
        deterministic and must match bitwise for both forward and backward.
        """
        context = nullcontext() if fused else _deterministic_torch_algorithms()
        mode = "fused" if fused else "unfused"
        with context:
            seg_lens = _DSV4_CP_RAGGED_SEG_LENS
            padded_seg_lens = _DSV4_CP_RAGGED_PADDED_SEG_LENS
            padded_tokens = sum(padded_seg_lens)
            packed = _make_thd_packed_seq_params(seg_lens, padded_seg_lens)
            if partition_mode == DSV4_CP_PARTITION_PACKED_STREAM_TWO_CHUNK:
                partition_indices = _make_chunked_cp_partition_indices(padded_tokens, self.cp_size)
            else:
                partition_indices = _make_contiguous_cp_partition_indices(
                    padded_tokens, self.cp_size
                )
            local_idx = partition_indices[self.cp_rank]

            torch.manual_seed(_SEED + 700 + layer_number)
            model_parallel_cuda_manual_seed(_SEED + 700 + layer_number)
            config = _make_dsv4_cp_config(
                context_parallel_size=self.cp_size,
                dsa_indexer_loss_coeff=1.0,
                dsa_indexer_use_sparse_loss=True,
                apply_dsa_kernel_fusion=fused,
                apply_rope_fusion=True,
                dsv4_cp_partition_mode=partition_mode,
            )
            graph_attn = _build_attention(
                config, layer_number=layer_number, pg_collection=self.pg
            ).cuda()
            eager_attn = _build_attention(
                config, layer_number=layer_number, pg_collection=self.pg
            ).cuda()
            graph_attn.train()
            eager_attn.train()
            _copy_module_parameters(graph_attn, eager_attn)

            full_hidden = torch.randn(
                padded_tokens, 1, config.hidden_size, dtype=torch.bfloat16, device='cuda'
            )
            test_hidden = full_hidden.index_select(0, local_idx).detach().clone()
            test_grad = torch.randn_like(test_hidden)
            static_hidden = test_hidden.detach().clone().requires_grad_(True)
            eager_hidden = test_hidden.detach().clone().requires_grad_(True)
            static_grad = test_grad.detach().clone()

            graph, graph_output = _capture_dsv4_attention_forward_backward(
                graph_attn, static_hidden, static_grad, packed
            )
            with torch.no_grad():
                static_hidden.copy_(test_hidden)
                static_grad.copy_(test_grad)
                if static_hidden.grad is not None:
                    static_hidden.grad.zero_()
                for param in graph_attn.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
            graph.replay()
            graph_out = graph_output.detach().clone()
            graph_hidden_grad = static_hidden.grad.detach().clone()
            graph_param_grads = {
                name: param.grad.detach().clone()
                for name, param in graph_attn.named_parameters()
                if param.grad is not None
            }

            eager_out, eager_hidden_grad, eager_param_grads = _run_dsv4_attention_forward_backward(
                eager_attn, eager_hidden, test_grad, packed
            )
            assert graph_param_grads.keys() == eager_param_grads.keys()
            # Fused forward kernels are deterministic, so graph replay must
            # match eager output bitwise. Fused backward uses atomic adds in the
            # sparse-attn/indexer kernels, so its accumulation order can differ.
            _assert_cp_graph_bitwise_match(
                graph_out, eager_out, f"layer={layer_number}:{mode}:{partition_mode}:output"
            )
            if fused:
                bwd_match_fn = _assert_cp_graph_fused_backward_match
            else:
                bwd_match_fn = _assert_cp_graph_bitwise_match
            bwd_match_fn(
                graph_hidden_grad,
                eager_hidden_grad,
                f"layer={layer_number}:{mode}:{partition_mode}:hidden_grad",
            )
            for name, graph_grad in graph_param_grads.items():
                bwd_match_fn(
                    graph_grad,
                    eager_param_grads[name],
                    f"layer={layer_number}:{mode}:{partition_mode}:param_grad:{name}",
                )

            del graph_attn, eager_attn, full_hidden, test_hidden, test_grad, static_hidden
            del eager_hidden, static_grad, graph, graph_output, graph_out, graph_hidden_grad
            gc.collect()
            torch.cuda.empty_cache()

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2, 3],
        ids=["ratio_0_window_only", "ratio_4_indexer", "ratio_128_compressor"],
    )
    def test_thd_cp_unfused_rope_is_rejected(self, layer_number):
        """THD CP rejects the unfused RoPE path for every DSv4 ratio.

        Production THD CP requires fused RoPE because CP-local rows can start in
        the middle of a packed sequence; the unfused path is not implemented for
        that position reconstruction.
        """
        seg_lens = _DSV4_CP_RAGGED_SEG_LENS
        padded_seg_lens = _DSV4_CP_RAGGED_PADDED_SEG_LENS
        padded_tokens = sum(padded_seg_lens)
        packed = _make_thd_packed_seq_params(seg_lens, padded_seg_lens)
        partition_indices = _make_contiguous_cp_partition_indices(padded_tokens, self.cp_size)
        local_idx = partition_indices[self.cp_rank]

        torch.manual_seed(_SEED + 300 + layer_number)
        model_parallel_cuda_manual_seed(_SEED + 300 + layer_number)
        config = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_dsa_kernel_fusion=True,
            apply_rope_fusion=False,
        )
        attn = _build_attention(config, layer_number=layer_number, pg_collection=self.pg).cuda()

        full_hidden = torch.randn(
            padded_tokens, 1, config.hidden_size, dtype=torch.bfloat16, device='cuda'
        )
        local_hidden = full_hidden.index_select(0, local_idx).detach().clone().requires_grad_(True)

        with pytest.raises(
            RuntimeError,
            match="DSv4 THD CP requires apply_rope_fusion=True",
        ):
            attn(hidden_states=local_hidden, attention_mask=None, packed_seq_params=packed)

        del attn, full_hidden, local_hidden
        gc.collect()
        torch.cuda.empty_cache()
