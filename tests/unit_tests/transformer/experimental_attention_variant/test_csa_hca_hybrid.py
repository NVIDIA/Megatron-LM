# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the CSA/HCA hybrid attention modules."""

from unittest.mock import patch

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_allocation import (
    Symbols,
    parse_hybrid_pattern,
    validate_segment_layers,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    CompressedSparseAttentionSubmodules,
    Compressor,
    CompressorSubmodules,
    CSAIndexer,
    CSAIndexerSubmodules,
    _get_compress_topk_idxs_cached,
    _get_window_topk_idxs_cached,
    get_compress_topk_idxs,
    get_window_topk_idxs,
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

# ---------------------------------------------------------------------------
# Hadamard (rotate_activation) shim for environments without the kernel.
# ---------------------------------------------------------------------------

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_transform  # noqa: F401

    HAVE_HADAMARD = True
except ImportError:
    HAVE_HADAMARD = False


def _mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return x * scale


@pytest.fixture(autouse=True)
def patch_hadamard_if_needed():
    if not HAVE_HADAMARD:
        with patch(
            'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
            _mock_hadamard_transform,
        ):
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# Helper-only tests (do not require model-parallel initialization).
# ---------------------------------------------------------------------------


@pytest.mark.internal
class TestLayerSymbols:
    def test_csa_hca_symbols_present(self):
        assert Symbols.CSA_ATTENTION == "C"
        assert Symbols.HCA_ATTENTION == "H"
        assert Symbols.CSA_ATTENTION in Symbols.VALID_LAYERS
        assert Symbols.HCA_ATTENTION in Symbols.VALID_LAYERS

    def test_pattern_with_csa_hca(self):
        layers = validate_segment_layers("MCMHM-")
        assert layers == ['M', 'C', 'M', 'H', 'M', '-']

    def test_csa_hca_not_combinable_with_standard_attention(self):
        with pytest.raises(ValueError):
            validate_segment_layers("M*MC")
        with pytest.raises(ValueError):
            validate_segment_layers("M*MH")

    def test_parse_pattern_with_csa_hca(self):
        parsed = parse_hybrid_pattern("MCMHMD")
        assert parsed.main_pattern == "MCMHMD"
        assert parsed.mtp_pattern is None


@pytest.mark.internal
class TestIndexHelpers:
    def test_window_indices_shape_and_validity(self):
        idxs = _get_window_topk_idxs_cached(window_size=4, seqlen=8, device_str="cpu")
        assert idxs.shape == (8, 4)
        # First row only has position 0 visible (window starts before seq)
        assert idxs[0, 0].item() == 0
        assert idxs[0, 1].item() == -1
        # Last row sees positions 4..7
        assert idxs[7].tolist() == [4, 5, 6, 7]

    def test_window_indices_batch_expand(self):
        idxs = get_window_topk_idxs(
            window_size=2, batch_size=3, seqlen=4, device=torch.device("cpu")
        )
        assert idxs.shape == (3, 4, 2)
        assert torch.equal(idxs[0], idxs[1])

    def test_compress_indices_causal(self):
        # ratio=4, seq=8 -> 2 compressed positions
        idxs = _get_compress_topk_idxs_cached(ratio=4, seqlen=8, offset=8, device_str="cpu")
        assert idxs.shape == (8, 2)
        # Positions 0..3 cannot see compressed entry 0 yet (1//4 == 0)
        assert (idxs[0:4] == -1).all().item()
        # Positions 4..7 can see compressed entry 0 (offset=8 -> index 8)
        assert (idxs[4:8, 0] == 8).all().item()
        # No position can see compressed entry 1 because 8//4 == 2 only at i=8
        assert (idxs[:, 1] == -1).all().item()

    def test_compress_indices_batch_expand(self):
        idxs = get_compress_topk_idxs(
            ratio=4, batch_size=2, seqlen=8, offset=0, device=torch.device("cpu")
        )
        assert idxs.shape == (2, 8, 2)


@pytest.mark.internal
class TestUnfusedAttnCpuShape:
    """Sanity-check the unfused sparse attn kernel on CPU (no CUDA).

    Verifies output shape and that an all-invalid row produces zero output when
    ``use_attn_sink=False``.
    """

    def test_output_shape(self):
        sq, b, np_, hn = 4, 2, 3, 16
        topk = 5
        query = torch.randn(sq, b, np_, hn)
        kv_full = torch.randn(7, b, hn)
        attn_sink = torch.zeros(np_)
        topk_idxs = torch.full((b, sq, topk), -1, dtype=torch.int32)
        # Make some valid positions
        topk_idxs[:, :, 0] = 0
        topk_idxs[:, 1:, 1] = 1

        out = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, topk_idxs, softmax_scale=hn**-0.5, use_attn_sink=False
        )
        assert out.shape == (sq, b, np_ * hn)

    def test_all_invalid_row_no_sink_yields_zero(self):
        sq, b, np_, hn = 2, 1, 1, 4
        query = torch.randn(sq, b, np_, hn)
        kv_full = torch.randn(2, b, hn)
        attn_sink = torch.zeros(np_)
        topk_idxs = torch.full((b, sq, 3), -1, dtype=torch.int32)
        # Row 0 all invalid; row 1 has one valid position
        topk_idxs[:, 1, 0] = 0

        out = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, topk_idxs, softmax_scale=hn**-0.5, use_attn_sink=False
        )
        # Row 0 should be exactly zero
        assert torch.allclose(out[0], torch.zeros_like(out[0]))


# ---------------------------------------------------------------------------
# CUDA-required tests
# ---------------------------------------------------------------------------


def _make_mla_config(num_layers: int = 2, csa_compress_ratios=None):
    return MLATransformerConfig(
        num_layers=num_layers,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=64,
        ffn_hidden_size=128,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        rope_type="rope",
        rotary_base=10000,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        # CSA / HCA defaults
        csa_window_size=8,
        csa_compress_ratios=csa_compress_ratios,
        csa_dense_mode=False,
        csa_attention_sink=True,
        # Indexer
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=4,
        dsa_indexer_loss_coeff=0.0,
        # Output projection
        o_groups=2,
        o_lora_rank=16,
        params_dtype=torch.bfloat16,
        bf16=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.internal
class TestCompressorAndIndexer:
    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def _build_pg(self):
        return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

    def _build_rotary(self, config):
        from megatron.core.models.common.embeddings import RotaryEmbedding

        pg = self._build_pg()
        return RotaryEmbedding(
            config.qk_pos_emb_head_dim,
            rotary_percent=1.0,
            rotary_base=config.rotary_base,
            cp_group=pg.cp,
        )

    def _build_compressor_spec(self):
        from megatron.core.extensions.transformer_engine import TELinear, TENorm

        return ModuleSpec(
            module=Compressor,
            submodules=CompressorSubmodules(
                linear_wkv=TELinear, linear_wgate=TELinear, norm=TENorm
            ),
        )

    def test_compressor_csa_overlap_ratio_4(self):
        from megatron.core.transformer.spec_utils import build_module

        config = _make_mla_config(num_layers=1)
        rotary = self._build_rotary(config)

        compressor = (
            build_module(
                self._build_compressor_spec(),
                config=config,
                compress_ratio=4,
                head_dim=config.v_head_dim,
                rotate=False,
                rotary_pos_emb=rotary,
                pg_collection=self._build_pg(),
            )
            .cuda()
            .to(torch.bfloat16)
        )

        sq, b = 16, 2
        x = torch.randn(sq, b, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        out = compressor(x)
        # Ratio 4 -> sq/4 compressed entries
        assert out.shape == (sq // 4, b, config.v_head_dim)

    def test_compressor_hca_no_overlap_ratio_8(self):
        from megatron.core.transformer.spec_utils import build_module

        config = _make_mla_config(num_layers=1)
        rotary = self._build_rotary(config)

        compressor = (
            build_module(
                self._build_compressor_spec(),
                config=config,
                compress_ratio=8,
                head_dim=config.v_head_dim,
                rotate=False,
                rotary_pos_emb=rotary,
                pg_collection=self._build_pg(),
            )
            .cuda()
            .to(torch.bfloat16)
        )

        sq, b = 16, 2
        x = torch.randn(sq, b, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        out = compressor(x)
        assert out.shape == (sq // 8, b, config.v_head_dim)

    def test_compressor_returns_none_when_seq_too_short(self):
        from megatron.core.transformer.spec_utils import build_module

        config = _make_mla_config(num_layers=1)
        rotary = self._build_rotary(config)

        compressor = (
            build_module(
                self._build_compressor_spec(),
                config=config,
                compress_ratio=8,
                head_dim=config.v_head_dim,
                rotate=False,
                rotary_pos_emb=rotary,
                pg_collection=self._build_pg(),
            )
            .cuda()
            .to(torch.bfloat16)
        )

        x = torch.randn(4, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        assert compressor(x) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.internal
class TestCompressedSparseAttention:
    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def _build_attention(self, compress_ratio: int):
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.models.common.embeddings import RotaryEmbedding
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.transformer.spec_utils import build_module

        config = _make_mla_config(num_layers=1)
        pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        rotary = RotaryEmbedding(
            config.qk_pos_emb_head_dim,
            rotary_percent=1.0,
            rotary_base=config.rotary_base,
            cp_group=pg.cp,
        )

        compressor_spec = ModuleSpec(
            module=Compressor,
            submodules=CompressorSubmodules(
                linear_wkv=TELinear, linear_wgate=TELinear, norm=TENorm
            ),
        )
        indexer_spec = ModuleSpec(
            module=CSAIndexer,
            submodules=CSAIndexerSubmodules(
                linear_wq_b=TELinear, linear_weights_proj=TELinear, compressor=compressor_spec
            ),
        )

        attention = (
            build_module(
                ModuleSpec(
                    module=CompressedSparseAttention,
                    submodules=CompressedSparseAttentionSubmodules(
                        compressor=compressor_spec, indexer=indexer_spec
                    ),
                ),
                config=config,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
                softmax_scale=None,
                k_channels=config.v_head_dim,
                v_channels=config.v_head_dim,
                cp_comm_type="p2p",
                pg_collection=pg,
                rotary_pos_emb=rotary,
                compress_ratio=compress_ratio,
            )
            .cuda()
            .to(torch.bfloat16)
        )

        return config, attention

    def _make_inputs(self, config, sq=16, b=2):
        np_ = config.num_attention_heads
        hn = config.v_head_dim
        query = torch.randn(sq, b, np_, hn, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(sq, b, 1, hn, device="cuda", dtype=torch.bfloat16)
        value = key.clone()
        x = torch.randn(sq, b, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        qr = torch.randn(sq, b, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
        return query, key, value, x, qr

    def test_csa_forward_shape(self):
        config, attn = self._build_attention(compress_ratio=4)
        query, key, value, x, qr = self._make_inputs(config)

        out = attn(query, key, value, attention_mask=None, x=x, qr=qr)
        sq, b, np_, hn = query.shape
        assert out.shape == (sq, b, np_ * hn)
        assert attn.indexer is not None  # CSA path
        assert attn.compressor is not None

    def test_hca_forward_shape(self):
        config, attn = self._build_attention(compress_ratio=8)
        query, key, value, x, qr = self._make_inputs(config)

        out = attn(query, key, value, attention_mask=None, x=x, qr=qr)
        sq, b, np_, hn = query.shape
        assert out.shape == (sq, b, np_ * hn)
        # HCA -> no indexer, dense over compressed positions
        assert attn.indexer is None
        assert attn.compressor is not None

    def test_window_only_no_compression(self):
        config, attn = self._build_attention(compress_ratio=0)
        query, key, value, x, qr = self._make_inputs(config)

        out = attn(query, key, value, attention_mask=None, x=x, qr=qr)
        sq, b, np_, hn = query.shape
        assert out.shape == (sq, b, np_ * hn)
        assert attn.indexer is None
        assert attn.compressor is None

    def test_csa_backward_runs(self):
        config, attn = self._build_attention(compress_ratio=4)
        query, key, value, x, qr = self._make_inputs(config)
        # Make x and qr require grad so that the indexer-loss path is exercised.
        x = x.detach().requires_grad_(True)
        qr = qr.detach().requires_grad_(True)
        query = query.detach().requires_grad_(True)
        key = key.detach().requires_grad_(True)

        attn.train()
        out = attn(query, key, value, attention_mask=None, x=x, qr=qr)
        out.sum().backward()
        assert query.grad is not None
        assert key.grad is not None
