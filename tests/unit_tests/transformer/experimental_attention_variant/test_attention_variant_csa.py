# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    CompressedSparseAttentionSubmodules,
    Compressor,
    CompressorSubmodules,
    CSAIndexer,
    CSAIndexerSubmodules,
    _apply_rope,
    build_cu_seqlens_kv_full,
    cat_per_segment,
    get_compress_topk_idxs,
    get_compress_topk_idxs_thd,
    get_window_topk_idxs,
    get_window_topk_idxs_thd,
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_transform

    HAVE_HADAMARD = True
except ImportError:
    HAVE_HADAMARD = False
    _hadamard_transform = None


def mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Mock implementation of hadamard_transform for testing without the library installed."""
    return x * scale


@pytest.fixture(autouse=True)
def patch_hadamard_if_needed():
    """Automatically patch hadamard_transform in both dsa and csa modules if not installed."""
    if not HAVE_HADAMARD:
        with (
            patch(
                'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
                mock_hadamard_transform,
            ),
            patch(
                'megatron.core.transformer.experimental_attention_variant.csa.rotate_activation',
                lambda x: x * (x.size(-1) ** -0.5),
            ),
        ):
            yield
    else:
        yield


# ===========================================================================
# Helper function tests
# ===========================================================================


class TestGetWindowTopkIdxs:
    """Test get_window_topk_idxs helper."""

    def test_basic_shape(self):
        batch_size, seqlen, window_size = 2, 16, 4
        idxs = get_window_topk_idxs(window_size, batch_size, seqlen, torch.device("cpu"))
        assert idxs.shape == (batch_size, seqlen, window_size)

    def test_causal_no_future(self):
        """Indices should never exceed the query position."""
        seqlen, window_size = 32, 8
        idxs = get_window_topk_idxs(window_size, 1, seqlen, torch.device("cpu"))
        for i in range(seqlen):
            valid = idxs[0, i][idxs[0, i] >= 0]
            assert torch.all(valid <= i), f"Position {i} has future indices"

    def test_invalid_marked_minus_one(self):
        """Early positions that cannot fill the window should use -1."""
        seqlen, window_size = 8, 4
        idxs = get_window_topk_idxs(window_size, 1, seqlen, torch.device("cpu"))
        assert idxs[0, 0, 0] == -1 or idxs[0, 0, 0] == 0
        for pos in range(window_size, seqlen):
            assert torch.all(idxs[0, pos] >= 0), f"Position {pos} has invalid -1"

    def test_window_larger_than_seqlen(self):
        """Window larger than sequence length should still work."""
        seqlen, window_size = 4, 16
        idxs = get_window_topk_idxs(window_size, 1, seqlen, torch.device("cpu"))
        assert idxs.shape == (1, seqlen, window_size)


class TestGetCompressTopkIdxs:
    """Test get_compress_topk_idxs helper."""

    def test_basic_shape(self):
        ratio, batch_size, seqlen, offset = 4, 2, 32, 32
        idxs = get_compress_topk_idxs(ratio, batch_size, seqlen, offset, torch.device("cpu"))
        n_compressed = seqlen // ratio
        assert idxs.shape == (batch_size, seqlen, n_compressed)

    def test_offset_applied(self):
        """Valid indices should be >= offset."""
        ratio, seqlen, offset = 4, 32, 100
        idxs = get_compress_topk_idxs(ratio, 1, seqlen, offset, torch.device("cpu"))
        valid = idxs[idxs >= 0]
        if valid.numel() > 0:
            assert torch.all(valid >= offset), "Valid indices should be offset"

    def test_causal_no_future(self):
        """Compressed indices should respect causality."""
        ratio, seqlen, offset = 4, 32, 32
        idxs = get_compress_topk_idxs(ratio, 1, seqlen, offset, torch.device("cpu"))
        for i in range(seqlen):
            n_valid = (i + 1) // ratio
            valid = idxs[0, i][idxs[0, i] >= 0]
            assert valid.numel() <= n_valid, f"Position {i} has too many valid compressed indices"

    def test_ratio_128(self):
        """Test with large compression ratio."""
        ratio, seqlen, offset = 128, 256, 256
        idxs = get_compress_topk_idxs(ratio, 1, seqlen, offset, torch.device("cpu"))
        assert idxs.shape == (1, seqlen, seqlen // ratio)


# ===========================================================================
# unfused_compressed_sparse_attn tests
# ===========================================================================


class TestUnfusedCompressedSparseAttn:
    """Test the unfused compressed sparse attention kernel."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_output_shape(self):
        """Test output shape of unfused compressed sparse attention."""
        sq, b, np_, hn = 16, 2, 4, 64
        n_kv = sq + sq // 4
        topk = 8

        query = torch.randn(sq, b, np_, hn, dtype=torch.bfloat16).cuda()
        kv_full = torch.randn(n_kv, b, hn, dtype=torch.bfloat16).cuda()
        attn_sink = torch.zeros(np_, dtype=torch.float32).cuda()
        topk_indices = torch.randint(0, n_kv, (b, sq, topk), dtype=torch.int32).cuda()
        softmax_scale = hn**-0.5

        output = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, topk_indices, softmax_scale
        )

        assert output.shape == (sq, b, np_ * hn)
        assert output.dtype == query.dtype

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_invalid_indices_masked(self):
        """Test that -1 indices are properly masked."""
        sq, b, np_, hn = 8, 1, 2, 32
        n_kv = sq
        topk = 4

        query = torch.randn(sq, b, np_, hn, dtype=torch.bfloat16).cuda()
        kv_full = torch.randn(n_kv, b, hn, dtype=torch.bfloat16).cuda()
        attn_sink = torch.zeros(np_, dtype=torch.float32).cuda()

        topk_indices = torch.full((b, sq, topk), -1, dtype=torch.int32).cuda()
        topk_indices[:, :, 0] = 0
        softmax_scale = hn**-0.5

        output = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, topk_indices, softmax_scale
        )
        assert not torch.isnan(output).any(), "Output should not contain NaN"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gradient_flow(self):
        """Test that gradients flow through sparse attention."""
        sq, b, np_, hn = 8, 1, 2, 32
        n_kv = sq
        topk = 4

        query = torch.randn(sq, b, np_, hn, dtype=torch.float32).cuda().requires_grad_(True)
        kv_full = torch.randn(n_kv, b, hn, dtype=torch.float32).cuda().requires_grad_(True)
        attn_sink = torch.nn.Parameter(torch.zeros(np_, dtype=torch.float32).cuda())

        topk_indices = torch.randint(0, n_kv, (b, sq, topk), dtype=torch.int32).cuda()
        softmax_scale = hn**-0.5

        output = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, topk_indices, softmax_scale
        )
        loss = output.sum()
        loss.backward()

        assert query.grad is not None
        assert kv_full.grad is not None
        assert attn_sink.grad is not None


# ===========================================================================
# Compressor tests
# ===========================================================================


def _make_mla_config(
    num_layers=4,
    hidden_size=256,
    num_attention_heads=16,
    v_head_dim=64,
    qk_pos_emb_head_dim=32,
    csa_compress_ratios=None,
    csa_window_size=8,
    csa_dense_mode=False,
    tensor_model_parallel_size=1,
    sequence_parallel=False,
    dsa_indexer_n_heads=8,
    dsa_indexer_head_dim=64,
    dsa_indexer_topk=8,
    dsa_indexer_loss_coeff=0.0,
    dsa_indexer_use_sparse_loss=False,
    rope_type='rope',
):
    """Helper to create MLATransformerConfig for CSA tests."""
    if csa_compress_ratios is None:
        csa_compress_ratios = [0] * num_layers
    return MLATransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=tensor_model_parallel_size,
        sequence_parallel=sequence_parallel,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=v_head_dim - qk_pos_emb_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        v_head_dim=v_head_dim,
        rope_type=rope_type,
        rotary_base=10000,
        rotary_percent=1.0,
        multi_latent_attention=True,
        experimental_attention_variant='dsv4_hybrid',
        csa_compress_ratios=csa_compress_ratios,
        csa_window_size=csa_window_size,
        csa_dense_mode=csa_dense_mode,
        dsa_indexer_n_heads=dsa_indexer_n_heads,
        dsa_indexer_head_dim=dsa_indexer_head_dim,
        dsa_indexer_topk=dsa_indexer_topk,
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
        dsa_indexer_use_sparse_loss=dsa_indexer_use_sparse_loss,
    )


def _make_compressor_submodules():
    """Create Compressor submodules spec."""
    from megatron.core.extensions.transformer_engine import TELinear, TENorm
    from megatron.core.transformer.spec_utils import ModuleSpec

    return CompressorSubmodules(
        linear_wkv=ModuleSpec(module=TELinear),
        linear_wgate=ModuleSpec(module=TELinear),
        norm=ModuleSpec(module=TENorm),
    )


def _make_csa_indexer_submodules():
    """Create CSAIndexer submodules spec."""
    from megatron.core.extensions.transformer_engine import TELinear, TENorm
    from megatron.core.transformer.spec_utils import ModuleSpec

    return CSAIndexerSubmodules(
        linear_wq_b=ModuleSpec(module=TELinear),
        linear_weights_proj=ModuleSpec(module=TELinear),
        compressor=ModuleSpec(module=Compressor, submodules=_make_compressor_submodules()),
    )


def _make_csa_submodules():
    """Create CompressedSparseAttention submodules spec."""
    from megatron.core.transformer.spec_utils import ModuleSpec

    return CompressedSparseAttentionSubmodules(
        compressor=ModuleSpec(module=Compressor, submodules=_make_compressor_submodules()),
        indexer=ModuleSpec(module=CSAIndexer, submodules=_make_csa_indexer_submodules()),
    )


# ===========================================================================
# Compressor tests
# ===========================================================================


@pytest.mark.parametrize("compress_ratio", [4, 128])
class TestCompressor:
    """Test Compressor module."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _make_mla_config(csa_compress_ratios=[4, 128, 4, 128])
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compressor_output_shape(self, compress_ratio):
        """Test that compressor produces correct output shape."""
        seq_len = 256
        batch_size = 2
        head_dim = self.config.v_head_dim

        compressor = Compressor(
            config=self.config,
            submodules=_make_compressor_submodules(),
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            rotate=False,
            rotary_pos_emb=self.rotary_pos_emb,
            pg_collection=self.pg_collection,
        ).cuda()

        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        output = compressor(x)

        expected_len = seq_len // compress_ratio
        assert output is not None
        assert output.shape == (expected_len, batch_size, head_dim)
        assert output.dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compressor_too_short_input(self, compress_ratio):
        """Test that compressor returns None when input is shorter than compress_ratio."""
        short_len = compress_ratio - 1
        batch_size = 2
        head_dim = self.config.v_head_dim

        compressor = Compressor(
            config=self.config,
            submodules=_make_compressor_submodules(),
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            rotate=False,
            rotary_pos_emb=self.rotary_pos_emb,
            pg_collection=self.pg_collection,
        ).cuda()

        x = torch.randn(short_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        output = compressor(x)
        assert output is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compressor_gradient_flow(self, compress_ratio):
        """Test that gradients flow through the compressor."""
        seq_len = 256
        batch_size = 2
        head_dim = self.config.v_head_dim

        compressor = Compressor(
            config=self.config,
            submodules=_make_compressor_submodules(),
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            rotate=False,
            rotary_pos_emb=self.rotary_pos_emb,
            pg_collection=self.pg_collection,
        ).cuda()

        x = (
            torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        output = compressor(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in compressor.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"


# ===========================================================================
# CSAIndexer tests
# ===========================================================================


@pytest.mark.parametrize("seqlen", [32, 128])
class TestCSAIndexer:
    """Test CSAIndexer module basic functionality."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.compress_ratio = 4
        cls.config = _make_mla_config(csa_compress_ratios=[4, 4, 4, 4], dsa_indexer_topk=8)
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        cls.indexer = CSAIndexer(
            config=cls.config,
            submodules=_make_csa_indexer_submodules(),
            compress_ratio=cls.compress_ratio,
            rotary_pos_emb=cls.rotary_pos_emb,
            pg_collection=cls.pg_collection,
        )

        yield
        Utils.destroy_model_parallel()

    def test_csa_indexer_constructor(self, seqlen):
        """Test CSAIndexer initialization."""
        assert isinstance(self.indexer, CSAIndexer)
        assert self.indexer.compress_ratio == self.compress_ratio
        assert self.indexer.index_n_heads == self.config.dsa_indexer_n_heads
        assert self.indexer.index_head_dim == self.config.dsa_indexer_head_dim
        assert self.indexer.index_topk == self.config.dsa_indexer_topk

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_csa_indexer_forward(self, seqlen):
        """Test CSAIndexer forward pass."""
        batch_size = 2
        self.indexer.cuda()

        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        index_scores, topk_indices = self.indexer(x, qr)
        n_compressed = seqlen // self.compress_ratio
        effective_topk = min(self.config.dsa_indexer_topk, n_compressed)

        assert index_scores.shape == (batch_size, seqlen, n_compressed)
        assert topk_indices.shape == (batch_size, seqlen, effective_topk)
        assert index_scores.dtype == torch.float32
        assert topk_indices.dtype == torch.long

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_csa_indexer_forward_before_topk(self, seqlen):
        """Test CSAIndexer forward_before_topk."""
        batch_size = 2
        self.indexer.cuda()

        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        q, k, weights = self.indexer.forward_before_topk(x, qr)

        assert q.shape == (
            seqlen,
            batch_size,
            self.config.dsa_indexer_n_heads,
            self.config.dsa_indexer_head_dim,
        )
        n_compressed = seqlen // self.compress_ratio
        assert k.shape == (n_compressed, batch_size, self.config.dsa_indexer_head_dim)
        assert weights.shape == (seqlen, batch_size, self.config.dsa_indexer_n_heads)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_csa_indexer_with_mask(self, seqlen):
        """Test CSAIndexer with causal mask."""
        batch_size = 2
        self.indexer.cuda()

        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        n_compressed = seqlen // self.compress_ratio
        causal_mask = torch.arange(n_compressed, device=x.device).unsqueeze(0).expand(seqlen, -1)
        positions = torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1)
        causal_mask = (
            torch.where(causal_mask >= positions // self.compress_ratio, float("-inf"), 0.0)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        index_scores, topk_indices = self.indexer(x, qr, mask=causal_mask)

        effective_topk = min(self.config.dsa_indexer_topk, n_compressed)
        assert index_scores.shape == (batch_size, seqlen, n_compressed)
        assert topk_indices.shape == (batch_size, seqlen, effective_topk)


# ===========================================================================
# CompressedSparseAttention tests
# ===========================================================================


class TestCompressedSparseAttentionRatio1:
    """Test CompressedSparseAttention with compress_ratio=1 (window-only)."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _make_mla_config(csa_compress_ratios=[0, 0, 0, 0], csa_window_size=8)
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        cls.csa = CompressedSparseAttention(
            config=cls.config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=cls.pg_collection,
            rotary_pos_emb=rotary_pos_emb,
            compress_ratio=0,
        )

        yield
        Utils.destroy_model_parallel()

    def test_ratio1_no_compressor(self):
        """With ratio=1, compressor and indexer should not be built."""
        assert self.csa.compressor is None
        assert self.csa.indexer is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ratio1_forward(self):
        """Test forward pass with window-only attention."""
        seq_len = 32
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        self.csa.cuda()

        query = torch.randn(seq_len, batch_size, np_, hn, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, 1, hn, dtype=torch.bfloat16).cuda()
        value = key.clone()
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = self.csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)

        assert output.shape == (seq_len, batch_size, np_ * hn)
        assert output.dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ratio1_backward(self):
        """Test backward pass with window-only attention."""
        seq_len = 32
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        self.csa.train()
        self.csa.cuda()

        query = (
            torch.randn(seq_len, batch_size, np_, hn, dtype=torch.float32)
            .cuda()
            .requires_grad_(True)
        )
        key = (
            torch.randn(seq_len, batch_size, 1, hn, dtype=torch.float32).cuda().requires_grad_(True)
        )
        value = key.clone().detach().requires_grad_(True)
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = self.csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)
        loss = output.sum()
        loss.backward()

        assert query.grad is not None
        assert key.grad is not None


@pytest.mark.parametrize("compress_ratio", [4, 128])
class TestCompressedSparseAttentionCompressed:
    """Test CompressedSparseAttention with compress_ratio > 1."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _make_mla_config(
            csa_compress_ratios=[4, 128, 4, 128],
            csa_window_size=8,
            dsa_indexer_topk=8,
            dsa_indexer_loss_coeff=1.0,
        )
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        yield
        Utils.destroy_model_parallel()

    def _get_layer_number(self, compress_ratio):
        """Return a layer_number (1-indexed) whose compress_ratio matches."""
        for i, r in enumerate(self.config.csa_compress_ratios):
            if r == compress_ratio:
                return i + 1
        raise ValueError(f"No layer with compress_ratio={compress_ratio}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_constructor(self, compress_ratio):
        """Test that compressor/indexer are conditionally built."""
        layer_number = self._get_layer_number(compress_ratio)
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=compress_ratio,
        ).cuda()

        assert csa.compressor is not None
        if compress_ratio == 4:
            assert csa.indexer is not None
        elif compress_ratio == 128:
            assert csa.indexer is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward(self, compress_ratio):
        """Test forward pass with compressed attention."""
        seq_len = 256
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        layer_number = self._get_layer_number(compress_ratio)
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=compress_ratio,
        ).cuda()

        query = torch.randn(seq_len, batch_size, np_, hn, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, 1, hn, dtype=torch.bfloat16).cuda()
        value = key.clone()
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)

        assert output.shape == (seq_len, batch_size, np_ * hn)
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward(self, compress_ratio):
        """Test backward pass with compressed attention."""
        seq_len = 256
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        layer_number = self._get_layer_number(compress_ratio)
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=compress_ratio,
        ).cuda()
        csa.train()

        query = (
            torch.randn(seq_len, batch_size, np_, hn, dtype=torch.float32)
            .cuda()
            .requires_grad_(True)
        )
        key = (
            torch.randn(seq_len, batch_size, 1, hn, dtype=torch.float32).cuda().requires_grad_(True)
        )
        value = key.clone().detach().requires_grad_(True)
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)
        loss = output.sum()
        loss.backward()

        assert query.grad is not None
        assert key.grad is not None

        for name, param in csa.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_eval_mode(self, compress_ratio):
        """Test forward pass in eval mode."""
        seq_len = 256
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        layer_number = self._get_layer_number(compress_ratio)
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=compress_ratio,
        ).cuda()
        csa.eval()

        query = torch.randn(seq_len, batch_size, np_, hn, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, 1, hn, dtype=torch.bfloat16).cuda()
        value = key.clone()
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        with torch.no_grad():
            output = csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)

        assert output.shape == (seq_len, batch_size, np_ * hn)
        assert not torch.isnan(output).any()


# ===========================================================================
# csa_dense_mode tests
# ===========================================================================


class TestCompressedSparseAttentionDenseMode:
    """Test that csa_dense_mode=True disables the indexer for ratio=4 layers."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _make_mla_config(
            csa_compress_ratios=[4, 128, 4, 128], csa_window_size=8, csa_dense_mode=True
        )
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dense_mode_disables_indexer_for_ratio4(self):
        """With csa_dense_mode=True, ratio=4 layers should NOT build an indexer."""
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=4,
        ).cuda()

        assert csa.compress_ratio == 4
        assert csa.compressor is not None, "Compressor should still be built"
        assert csa.indexer is None, "Indexer should be disabled in dense mode"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dense_mode_forward_ratio4(self):
        """Forward pass should work for ratio=4 in dense mode (uses all compressed positions)."""
        seq_len = 256
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=4,
        ).cuda()

        query = torch.randn(seq_len, batch_size, np_, hn, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, 1, hn, dtype=torch.bfloat16).cuda()
        value = key.clone()
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)

        assert output.shape == (seq_len, batch_size, np_ * hn)
        assert not torch.isnan(output).any()


# ===========================================================================
# _apply_rope tests
# ===========================================================================


class TestApplyRope:
    """Test ``_apply_rope`` — the layout-aware RoPE wrapper used by
    Compressor / CSAIndexer / DSv4HybridAttention.

    Behaviours covered:

    * 3-D ``[seq, batch, head_dim]`` and 4-D ``[seq, batch, heads, head_dim]``
      inputs both work (3-D gets a temporary head-dim unsqueeze).
    * Only the trailing ``pos_dim`` components are rotated; the leading
      ``nope_dim`` slice is bit-exact unchanged.
    * Both ``RotaryEmbedding`` (returns ``Tensor``) and
      ``YarnRotaryEmbedding`` (returns ``(emb, mscale)`` tuple) — DSv4
      hybrid silently swaps the class based on ``compress_ratio``.
    * Both unfused and fused (``config.apply_rope_fusion=True``) paths
      produce the same output (within bf16 precision).
    * For ``ratio > 1`` the rotary table is built at
      ``rotary_seq_len * ratio`` and strided by ``ratio``.
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(0)
        model_parallel_cuda_manual_seed(0)
        cls = request.cls
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        # head_dim 32 = nope 24 + pos 8
        cls.config = _make_mla_config(v_head_dim=32, qk_pos_emb_head_dim=8)
        yield
        Utils.destroy_model_parallel()

    def _make_rotary(self, kind: str):
        from megatron.core.models.common.embeddings import RotaryEmbedding, YarnRotaryEmbedding

        pos_dim = self.config.qk_pos_emb_head_dim
        if kind == 'rope':
            return RotaryEmbedding(
                pos_dim, rotary_percent=1.0, rotary_base=10000, cp_group=self.pg_collection.cp
            )
        if kind == 'yarn':
            return YarnRotaryEmbedding(
                pos_dim,
                rotary_base=40000,
                scaling_factor=40,
                original_max_position_embeddings=4096,
                beta_fast=32,
                beta_slow=1,
                mscale=1.0,
                mscale_all_dim=0.0,
                cp_group=self.pg_collection.cp,
            )
        raise ValueError(kind)

    def _config_with(self, *, apply_rope_fusion: bool):
        # Reuse the class-level config; only flip the fusion flag.
        cfg = self.config
        cfg.apply_rope_fusion = apply_rope_fusion
        return cfg

    _ROTARY_FUSION_COMBOS = [
        pytest.param('rope', False, id='rope-unfused'),
        pytest.param('rope', True, id='rope-fused'),
        pytest.param('yarn', False, id='yarn-unfused'),
        pytest.param('yarn', True, id='yarn-fused'),
    ]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(("rotary_kind", "apply_rope_fusion"), _ROTARY_FUSION_COMBOS)
    @pytest.mark.parametrize("input_ndim", [3, 4], ids=['3d', '4d'])
    @pytest.mark.parametrize("ratio", [1, 4], ids=['ratio_1', 'ratio_4'])
    def test_apply_rope(self, rotary_kind, apply_rope_fusion, input_ndim, ratio):
        """Output shape == input shape; no NaN; nope-dim slice is
        bit-exact unchanged. Sweeps the valid combinations of rotary
        class × apply_rope_fusion × input rank × ratio. Yarn's
        tuple-return is covered by the ``'yarn-*'`` combos.
        """
        rotary = self._make_rotary(rotary_kind).cuda()
        nope = self.config.v_head_dim - self.config.qk_pos_emb_head_dim
        pos = self.config.qk_pos_emb_head_dim
        head_dim = nope + pos
        seq, batch, heads = 8, 2, 4
        cfg = self._config_with(apply_rope_fusion=apply_rope_fusion)

        shape = (seq, batch, head_dim) if input_ndim == 3 else (seq, batch, heads, head_dim)
        x = torch.randn(*shape, dtype=torch.bfloat16, device='cuda')
        # ``fused_mla_rope_inplace`` mutates the input — give it a copy so
        # the nope-dim equality check below still has the original.
        out = _apply_rope(
            x.clone(),
            nope,
            pos,
            rotary,
            cfg,
            rotary_seq_len=seq,
            ratio=ratio,
            cp_group=self.pg_collection.cp,
        )

        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert not torch.isnan(out).any()
        # The leading nope_dim slice is the identity portion of RoPE.
        assert torch.equal(
            out[..., :nope], x[..., :nope]
        ), "RoPE must not touch the first nope_dim components"
        # Trailing pos_dim should rotate at non-zero positions.
        pe_changed = (out[..., nope:] != x[..., nope:]).any(dim=-1).flatten()
        assert pe_changed[
            1:
        ].any(), "RoPE should rotate the trailing pos_dim components for seq > 0"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("rotary_kind", ['rope', 'yarn'])
    def test_3d_input_matches_4d_with_single_head(self, rotary_kind):
        """For a single-head input, the 3-D ``(s, b, d)`` and 4-D
        ``(s, b, 1, d)`` invocations must produce numerically identical
        output (3-D path just inserts a temporary head dim).
        """
        rotary = self._make_rotary(rotary_kind).cuda()
        nope = self.config.v_head_dim - self.config.qk_pos_emb_head_dim
        pos = self.config.qk_pos_emb_head_dim
        head_dim = nope + pos
        seq, batch = 8, 2
        cfg = self._config_with(apply_rope_fusion=False)

        x_3d = torch.randn(seq, batch, head_dim, dtype=torch.bfloat16, device='cuda')
        x_4d = x_3d.unsqueeze(-2)

        out_3d = _apply_rope(
            x_3d,
            nope,
            pos,
            rotary,
            cfg,
            rotary_seq_len=seq,
            ratio=1,
            cp_group=self.pg_collection.cp,
        )
        out_4d = _apply_rope(
            x_4d,
            nope,
            pos,
            rotary,
            cfg,
            rotary_seq_len=seq,
            ratio=1,
            cp_group=self.pg_collection.cp,
        )

        assert out_3d.shape == x_3d.shape
        assert out_4d.shape == x_4d.shape
        assert torch.equal(out_3d, out_4d.squeeze(-2))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("rotary_kind", ['rope', 'yarn'])
    def test_ratio_strides_rotary_table(self, rotary_kind):
        """For ``ratio > 1``, the rotary table is built at
        ``rotary_seq_len * ratio`` and strided by ``ratio``. The result
        with ``ratio=k`` must equal an ``apply_rope`` call on the same
        positions of a length-``rotary_seq_len * k`` table.
        """
        rotary = self._make_rotary(rotary_kind).cuda()
        nope = self.config.v_head_dim - self.config.qk_pos_emb_head_dim
        pos = self.config.qk_pos_emb_head_dim
        head_dim = nope + pos
        seq, batch, heads, ratio = 4, 1, 2, 4
        cfg = self._config_with(apply_rope_fusion=False)

        x_comp = torch.randn(seq, batch, heads, head_dim, dtype=torch.bfloat16, device='cuda')
        out_comp = _apply_rope(
            x_comp.clone(),
            nope,
            pos,
            rotary,
            cfg,
            rotary_seq_len=seq,
            ratio=ratio,
            cp_group=self.pg_collection.cp,
        )

        x_full = torch.zeros(
            seq * ratio, batch, heads, head_dim, dtype=torch.bfloat16, device='cuda'
        )
        x_full[::ratio][:seq] = x_comp
        out_full = _apply_rope(
            x_full,
            nope,
            pos,
            rotary,
            cfg,
            rotary_seq_len=seq * ratio,
            ratio=1,
            cp_group=self.pg_collection.cp,
        )
        out_ref = out_full[::ratio][:seq]

        assert torch.allclose(out_comp, out_ref, rtol=1e-3, atol=1e-3), (
            f"ratio={ratio} stride mismatch: "
            f"max abs diff = {(out_comp - out_ref).abs().max().item():.3e}"
        )


# ===========================================================================
# THD packed-sequence helpers
# ===========================================================================


def _cu_seqlens(seg_lens, device='cpu'):
    """``(B+1,)`` int32 cu_seqlens from a list of per-segment lengths."""
    return torch.tensor(
        [0] + list(torch.tensor(seg_lens, dtype=torch.int64).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )


class TestCsaThdIndexHelpers:
    """CSA THD index helpers — pure-Python, no GPU. Mirrors the
    organisation of ``TestThdPureHelpers`` in ``test_dsa_kernels.py``:
    one mega-class with section comments per helper, since each helper
    only needs 2–3 tests and they share no fixtures.

    Helpers covered:

    * ``get_window_topk_idxs_thd``     — per-segment sliding window.
    * ``get_compress_topk_idxs_thd``   — per-segment all-compressed
                                         indices shifted to full-KV space.
    * ``build_cu_seqlens_kv_full``     — per-segment lens of the
                                         ``[kv, compressed_kv]`` concat.
    * ``cat_per_segment``              — per-segment concat into the
                                         THD-packed full-KV layout.
    """

    # ---- get_window_topk_idxs_thd --------------------------------------

    def test_window_shape_dtype_and_local_indices(self):
        """Window indices are LOCAL within each segment — they reset to 0
        at each segment boundary (not global flat KV ids).
        """
        cu = _cu_seqlens([4, 3])  # = [0, 4, 7]
        out = get_window_topk_idxs_thd(window_size=3, cu_seqlens_q=cu, device='cpu')
        assert out.shape == (7, 3)
        assert out.dtype == torch.int32
        expected = torch.tensor(
            [[0, -1, -1], [0, 1, -1], [0, 1, 2], [1, 2, 3], [0, -1, -1], [0, 1, -1], [0, 1, 2]],
            dtype=torch.int32,
        )
        assert torch.equal(out, expected)

    def test_window_causality_no_future(self):
        """No window index should exceed the query's position-in-segment."""
        cu = _cu_seqlens([5, 6, 3])
        out = get_window_topk_idxs_thd(window_size=4, cu_seqlens_q=cu, device='cpu')
        seq_lens = (cu[1:] - cu[:-1]).tolist()
        offsets = cu[:-1].tolist()
        for b, (offset, slen) in enumerate(zip(offsets, seq_lens)):
            for s in range(slen):
                row = out[offset + s]
                valid = row[row >= 0]
                assert (valid <= s).all(), f"seg {b}, pos {s}: window index exceeds position"

    # ---- get_compress_topk_idxs_thd ------------------------------------

    @pytest.mark.parametrize(
        "q_segs, kv_segs, comp_segs, expected_shape, expected_ranges",
        [
            ([8, 4], [5, 3], [2, 1], (12, 2), {(0, 8): (5, 7), (8, 12): (3, 4)}),
            ([3, 2], [3, 2], [0, 0], (5, 0), {}),
        ],
        ids=["multi_seg_offsets", "no_compressed_empty"],
    )
    def test_compress_shape_and_offset(
        self, q_segs, kv_segs, comp_segs, expected_shape, expected_ranges
    ):
        """Valid indices live in the correct per-segment range, or output is
        empty when all segments are shorter than ratio.
        """
        ratio = 4
        out = get_compress_topk_idxs_thd(
            ratio, _cu_seqlens(q_segs), _cu_seqlens(kv_segs), _cu_seqlens(comp_segs), device='cpu'
        )
        assert out.shape == expected_shape
        for (start, end), (lo, hi) in expected_ranges.items():
            valid = out[start:end][out[start:end] >= 0]
            assert (valid >= lo).all() and (valid < hi).all()

    def test_compress_causal_n_valid_per_pos(self):
        """Per-row valid count == ``min(seqlen_compressed[b], (pos+1)//ratio)``."""
        ratio = 4
        out = get_compress_topk_idxs_thd(
            ratio, _cu_seqlens([8]), _cu_seqlens([5]), _cu_seqlens([2]), device='cpu'
        )
        for pos in range(8):
            n_valid_expected = min(2, (pos + 1) // ratio)
            n_valid_actual = int((out[pos] >= 0).sum())
            assert n_valid_actual == n_valid_expected, f"pos {pos}: count mismatch"

    # ---- build_cu_seqlens_kv_full --------------------------------------

    def test_build_cu_seqlens_kv_full_basic(self):
        cu_kv = _cu_seqlens([4, 3, 5])
        cu_comp = _cu_seqlens([1, 0, 2])
        out = build_cu_seqlens_kv_full(cu_kv, cu_comp)
        # full lens = [4+1, 3+0, 5+2] = [5, 3, 7]; cumsum = [0, 5, 8, 15].
        assert out.tolist() == [0, 5, 8, 15]
        assert out.dtype == cu_kv.dtype

    def test_build_cu_seqlens_kv_full_empty_compressed(self):
        """When compressed is all zeros, full == kv."""
        cu_kv = _cu_seqlens([3, 4])
        cu_comp = _cu_seqlens([0, 0])
        out = build_cu_seqlens_kv_full(cu_kv, cu_comp)
        assert torch.equal(out, cu_kv)

    # ---- cat_per_segment ------------------------------------------------

    def test_cat_per_segment_basic_concat(self):
        kv_lens = [3, 2]
        comp_lens = [1, 2]
        d = 2
        cu_kv = _cu_seqlens(kv_lens)
        cu_comp = _cu_seqlens(comp_lens)
        cu_full = build_cu_seqlens_kv_full(cu_kv, cu_comp)

        # Distinct values so we can verify each row's source.
        kv = torch.arange(sum(kv_lens) * d, dtype=torch.float32).reshape(-1, d)
        comp = (torch.arange(sum(comp_lens) * d, dtype=torch.float32) + 100).reshape(-1, d)

        out = cat_per_segment(kv, comp, cu_kv, cu_comp, cu_full)
        assert out.shape == (sum(kv_lens) + sum(comp_lens), d)
        # Segment 0: kv rows 0..2, then comp row 0.
        assert torch.equal(out[0:3], kv[0:3])
        assert torch.equal(out[3:4], comp[0:1])
        # Segment 1: kv rows 3..4, then comp rows 1..2.
        assert torch.equal(out[4:6], kv[3:5])
        assert torch.equal(out[6:8], comp[1:3])

    def test_cat_per_segment_none_compressed_returns_kv(self):
        """``compressed_kv_thd is None`` short-circuits to ``kv_thd``."""
        cu_kv = _cu_seqlens([3, 2])
        cu_comp = _cu_seqlens([0, 0])
        cu_full = build_cu_seqlens_kv_full(cu_kv, cu_comp)
        kv = torch.randn(5, 2)
        out = cat_per_segment(kv, None, cu_kv, cu_comp, cu_full)
        assert out is kv


# ===========================================================================
# unfused_compressed_sparse_attn THD branch
# ===========================================================================


class TestUnfusedCompressedSparseAttnThd:
    """``unfused_compressed_sparse_attn`` dispatches on ``query.ndim``:
    3-D selects the THD branch (flat layout, global topk ids).
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_output_shape(self):
        """THD inputs (3-D query, 2-D kv) produce 2-D ``(total_q, np * hn)``."""
        total_q, np_, hn = 12, 4, 64
        total_kv = 24
        topk = 4

        query = torch.randn(total_q, np_, hn, dtype=torch.bfloat16).cuda()
        kv_full = torch.randn(total_kv, hn, dtype=torch.bfloat16).cuda()
        attn_sink = torch.zeros(np_, dtype=torch.float32).cuda()
        topk_indices = torch.randint(0, total_kv, (total_q, topk), dtype=torch.int32).cuda()

        out = unfused_compressed_sparse_attn(query, kv_full, attn_sink, topk_indices, hn**-0.5)
        assert out.shape == (total_q, np_ * hn)
        assert out.dtype == query.dtype

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_invalid_indices_masked(self):
        """``-1`` indices in the THD topk should contribute 0 (no NaN)."""
        total_q, np_, hn = 6, 2, 32
        total_kv = 8
        topk = 4

        query = torch.randn(total_q, np_, hn, dtype=torch.bfloat16).cuda()
        kv_full = torch.randn(total_kv, hn, dtype=torch.bfloat16).cuda()
        attn_sink = torch.zeros(np_, dtype=torch.float32).cuda()

        topk_indices = torch.full((total_q, topk), -1, dtype=torch.int32).cuda()
        topk_indices[:, 0] = 0  # one valid position per row

        out = unfused_compressed_sparse_attn(query, kv_full, attn_sink, topk_indices, hn**-0.5)
        assert not torch.isnan(out).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_matches_sbhd_b1_equivalent(self):
        """THD (3-D query) on a single-batch problem produces the same
        per-token output as SBHD (4-D query) with ``b=1`` on the same
        data — both should hit the shared core inlined into the function.
        """
        sq, np_, hn = 8, 2, 32
        n_kv = 16
        topk = 4
        sm = hn**-0.5

        torch.manual_seed(0)
        # SBHD layout (b=1) and THD-equivalent (squeezed).
        query_sbhd = torch.randn(sq, 1, np_, hn, dtype=torch.bfloat16).cuda()
        kv_sbhd = torch.randn(n_kv, 1, hn, dtype=torch.bfloat16).cuda()
        attn_sink = torch.zeros(np_, dtype=torch.float32).cuda()

        # SBHD topk: per-batch LOCAL ids in [0, n_kv).
        topk_local = torch.randint(0, n_kv, (1, sq, topk), dtype=torch.int32).cuda()

        # THD topk: flat-global ids; for b=1 these match the local ids.
        topk_global = topk_local.squeeze(0)

        out_sbhd = unfused_compressed_sparse_attn(
            query_sbhd, kv_sbhd, attn_sink, topk_local, sm
        )  # (sq, 1, np * hn)
        out_thd = unfused_compressed_sparse_attn(
            query_sbhd.squeeze(1), kv_sbhd.squeeze(1), attn_sink, topk_global, sm
        )  # (sq, np * hn)

        # Same math, just different output layout.
        assert torch.allclose(out_sbhd.squeeze(1), out_thd, atol=1e-3, rtol=1e-3)


# ===========================================================================
# THD: Compressor / CSAIndexer / CompressedSparseAttention integration
# ===========================================================================
#
# These integration tests exercise the THD branches of the full
# Compressor / CSAIndexer / CompressedSparseAttention modules — the
# layer above the kernel-level THD tests in test_dsa_kernels.py and the
# autograd-Function tests in test_attention_variant_dsa.py.
#
# Strategy: most tests use a B=1 single-segment THD input and compare
# against the same data run through the SBHD path with b=1. For B=1
# the two layouts go through equivalent math (sparse-attention kernels
# are layout-agnostic; THD just adds slicing/concat glue), so any
# divergence beyond float-precision tolerance signals a plumbing bug.


def _make_packed_seq_params_thd(seg_lens, device='cuda'):
    """Build a ``PackedSeqParams(qkv_format='thd', ...)`` from a list of
    per-segment seq lengths. Self-attention contract: ``cu_seqlens_q ==
    cu_seqlens_kv``; ``*_padded`` mirrors the unpadded (no padding tested).
    """
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(seg_lens, dtype=torch.int64).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    max_len = int(max(seg_lens)) if seg_lens else 0
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=max_len,
        max_seqlen_kv=max_len,
        qkv_format='thd',
    )


@pytest.mark.parametrize("compress_ratio", [4, 128])
class TestCompressorThd:
    """``Compressor`` THD-packed forward path
    (``Compressor.forward(x, packed_seq_params=...)`` → ``_forward_thd``).

    Covers:
      * Per-segment compressed-length contract:
        ``cu_seqlens_compressed[b+1] - cu_seqlens_compressed[b]
          == seqlen[b] // ratio``.
      * Shape + dtype of the packed compressed-KV tensor.
      * All-segments-too-short fast path (returns ``(None, cu_seqlens_compressed)``).
      * B=1 single-segment THD matches SBHD-b=1 (numerical parity — same
        per-segment math, just different layout glue).
      * Gradient flow through the THD compression path.
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _make_mla_config(csa_compress_ratios=[4, 128, 4, 128])
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        yield
        Utils.destroy_model_parallel()

    def _make_compressor(self, compress_ratio):
        return Compressor(
            config=self.config,
            submodules=_make_compressor_submodules(),
            compress_ratio=compress_ratio,
            head_dim=self.config.v_head_dim,
            rotate=False,
            rotary_pos_emb=self.rotary_pos_emb,
            pg_collection=self.pg_collection,
        ).cuda()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_output_shape_and_cu_seqlens(self, compress_ratio):
        """Multi-segment THD: each segment's compressed length is
        ``seqlen[b] // ratio``; totals match the concat'd output.
        """
        # Pick three segment lengths that each compress non-trivially.
        seg_lens = [compress_ratio * 5, compress_ratio * 3, compress_ratio * 7]
        total = sum(seg_lens)
        packed = _make_packed_seq_params_thd(seg_lens)
        compressor = self._make_compressor(compress_ratio)

        x = torch.randn(total, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda')
        out, cu_seqlens_compressed = compressor(x, packed_seq_params=packed)

        # Per-segment compressed lengths.
        expected_per_seg = [s // compress_ratio for s in seg_lens]
        expected_total = sum(expected_per_seg)

        assert out is not None
        assert out.shape == (expected_total, 1, self.config.v_head_dim), (
            f"compressed_thd shape {tuple(out.shape)} != expected "
            f"{(expected_total, 1, self.config.v_head_dim)}"
        )
        assert out.dtype == torch.bfloat16
        # cu_seqlens_compressed[b+1] - cu_seqlens_compressed[b] == seqlen[b] // ratio.
        diffs = (cu_seqlens_compressed[1:] - cu_seqlens_compressed[:-1]).cpu().tolist()
        assert (
            diffs == expected_per_seg
        ), f"cu_seqlens_compressed segment lengths {diffs} != {expected_per_seg}"
        assert not torch.isnan(out).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_all_segments_too_short(self, compress_ratio):
        """All segments shorter than ``ratio`` → returns
        ``(None, cu_seqlens_compressed_all_zeros)``.
        """
        seg_lens = [compress_ratio - 1, compress_ratio - 1]
        total = sum(seg_lens)
        packed = _make_packed_seq_params_thd(seg_lens)
        compressor = self._make_compressor(compress_ratio)

        x = torch.randn(total, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda')
        out, cu_seqlens_compressed = compressor(x, packed_seq_params=packed)

        assert out is None
        # All per-segment compressed lengths are zero.
        diffs = (cu_seqlens_compressed[1:] - cu_seqlens_compressed[:-1]).cpu().tolist()
        assert all(
            d == 0 for d in diffs
        ), f"all-short batch should have cu_seqlens_compressed all zero, got {diffs}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_b1_matches_sbhd_b1(self, compress_ratio):
        """B=1 single-segment THD output matches SBHD-b=1 on identical
        input (compressor weights shared between the two calls). For the
        same hidden states the per-segment math is identical, so the
        outputs must agree within bf16-precision tolerance.
        """
        seq_len = compress_ratio * 8
        compressor = self._make_compressor(compress_ratio)

        torch.manual_seed(42)
        x_thd = torch.randn(
            seq_len, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda'
        )
        # SBHD-b=1 input is the same data, no reshape needed (already (sq, 1, h)).
        x_sbhd = x_thd

        # SBHD path: pass packed_seq_params=None → _forward_sbhd.
        out_sbhd = compressor(x_sbhd, packed_seq_params=None)

        # THD path: pass packed_seq_params with single segment.
        packed = _make_packed_seq_params_thd([seq_len])
        out_thd, cu_comp = compressor(x_thd, packed_seq_params=packed)

        assert out_sbhd is not None and out_thd is not None
        assert (
            out_sbhd.shape == out_thd.shape
        ), f"shape mismatch: sbhd={tuple(out_sbhd.shape)}, thd={tuple(out_thd.shape)}"
        # cu_seqlens_compressed = [0, n_compressed].
        assert cu_comp[-1].item() == seq_len // compress_ratio

        # Numerical parity. bf16 + small per-segment-loop ordering differences
        # mean we need a wider tol than fp32 would warrant.
        assert torch.allclose(out_sbhd.float(), out_thd.float(), atol=5e-2, rtol=5e-2), (
            f"B=1 SBHD/THD parity failed: max abs diff = "
            f"{(out_sbhd.float() - out_thd.float()).abs().max().item():.4e}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_gradient_flow(self, compress_ratio):
        """Backward through Compressor THD populates grad on ``x`` and
        every learnable parameter in the Compressor.
        """
        seg_lens = [compress_ratio * 4, compress_ratio * 6]
        total = sum(seg_lens)
        packed = _make_packed_seq_params_thd(seg_lens)
        compressor = self._make_compressor(compress_ratio)

        x = torch.randn(
            total, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda'
        ).requires_grad_(True)
        out, _ = compressor(x, packed_seq_params=packed)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None and not torch.isnan(x.grad).any()
        for name, p in compressor.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Compressor param {name} has no grad"


class TestCSAIndexerThd:
    """``CSAIndexer`` THD-packed paths:
      * ``forward_before_topk(packed_seq_params=...)`` — 4-tuple return
        with ``cu_seqlens_compressed_idx``.
      * ``forward(packed_seq_params=...)`` — THD dispatch through
        :func:`fused_qk_topk_naive_thd`. New in the THD-completion turn.

    Multi-segment shape contract + B=1 SBHD-b=1 parity.
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.compress_ratio = 4
        cls.config = _make_mla_config(csa_compress_ratios=[4, 4, 4, 4], dsa_indexer_topk=8)
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        cls.indexer = CSAIndexer(
            config=cls.config,
            submodules=_make_csa_indexer_submodules(),
            compress_ratio=cls.compress_ratio,
            rotary_pos_emb=cls.rotary_pos_emb,
            pg_collection=cls.pg_collection,
        ).cuda()

        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_forward_before_topk_returns_4_tuple(self):
        """THD ``forward_before_topk`` returns
        ``(q, k, weights, cu_seqlens_compressed_idx)`` with THD shapes
        (dummy ``b=1`` dim retained for layout consistency with the SBHD
        4-D / 3-D contract that downstream THD callers ``.squeeze(1)``).
        """
        ratio = self.compress_ratio
        seg_lens = [ratio * 6, ratio * 4]
        total = sum(seg_lens)
        expected_total_comp = sum(s // ratio for s in seg_lens)
        packed = _make_packed_seq_params_thd(seg_lens)

        x = torch.randn(total, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda')
        qr = torch.randn(total, 1, self.config.q_lora_rank, dtype=torch.bfloat16, device='cuda')

        result = self.indexer.forward_before_topk(x, qr, packed)
        assert len(result) == 4, "THD forward_before_topk should return a 4-tuple"
        q, k, weights, cu_seqlens_compressed_idx = result

        assert q.shape == (
            total,
            1,
            self.config.dsa_indexer_n_heads,
            self.config.dsa_indexer_head_dim,
        )
        assert weights.shape == (total, 1, self.config.dsa_indexer_n_heads)
        assert k.shape == (expected_total_comp, 1, self.config.dsa_indexer_head_dim)
        # cu_seqlens_compressed_idx mirrors the compressor's cu_seqlens.
        diffs = (cu_seqlens_compressed_idx[1:] - cu_seqlens_compressed_idx[:-1]).cpu().tolist()
        assert diffs == [s // ratio for s in seg_lens]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_forward_shape_and_dtype(self):
        """THD ``forward`` returns ``(None, (total_q, topk) int64)``."""
        ratio = self.compress_ratio
        seg_lens = [ratio * 5, ratio * 3]
        total = sum(seg_lens)
        packed = _make_packed_seq_params_thd(seg_lens)

        x = torch.randn(total, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda')
        qr = torch.randn(total, 1, self.config.q_lora_rank, dtype=torch.bfloat16, device='cuda')

        index_scores, topk = self.indexer(x, qr, packed_seq_params=packed)
        # THD return contract: per-segment scores aren't surfaced
        # (heterogeneous shapes); only consumers in csa.py
        # force_unfused inference use this path and discard scores.
        assert index_scores is None
        assert topk.shape == (total, self.config.dsa_indexer_topk)
        assert topk.dtype == torch.int64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_per_segment_kv_scope(self):
        """Top-K LOCAL ids stay in ``[0, seqlen_compressed[b])`` per-segment
        (NOT flat-global ids into the concat'd indexer-K).
        """
        ratio = self.compress_ratio
        seg_lens = [ratio * 8, ratio * 4]
        total = sum(seg_lens)
        n_comp_per_seg = [s // ratio for s in seg_lens]
        packed = _make_packed_seq_params_thd(seg_lens)

        x = torch.randn(total, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda')
        qr = torch.randn(total, 1, self.config.q_lora_rank, dtype=torch.bfloat16, device='cuda')

        _, topk = self.indexer(x, qr, packed_seq_params=packed)

        # Segment 0 rows: valid ids must be in [0, n_comp_per_seg[0]).
        seg0 = topk[: seg_lens[0]]
        seg0_valid = seg0[seg0 >= 0]
        if seg0_valid.numel() > 0:
            assert (seg0_valid < n_comp_per_seg[0]).all(), (
                f"segment 0 ids out of range: max={seg0_valid.max().item()}, "
                f"expected < {n_comp_per_seg[0]}"
            )
        # Segment 1 rows: valid ids must be in [0, n_comp_per_seg[1]).
        seg1 = topk[seg_lens[0] :]
        seg1_valid = seg1[seg1 >= 0]
        if seg1_valid.numel() > 0:
            assert (seg1_valid < n_comp_per_seg[1]).all(), (
                f"segment 1 ids out of range: max={seg1_valid.max().item()}, "
                f"expected < {n_comp_per_seg[1]}"
            )


class TestCompressedSparseAttentionThd:
    """End-to-end ``CompressedSparseAttention(packed_seq_params=...)``
    integration tests covering all THD-supported Path × fused/force_unfused
    combinations. Each test verifies no NaN + expected output shape; the
    deep numerical correctness is established at lower layers by the
    real-kernel parity tests (``TestRealKernelFusedIndexerSparseAttn*``,
    ``TestFusedDSAIndexerLossThd``, ``TestFusedQkTopkNaiveThd``).

    THD output shape is ``(total_q, 1, np * v_head_dim)`` — the dummy
    ``b=1`` axis is re-added inside ``_forward_thd`` so downstream
    callers can keep the SBHD ``(seq, batch, hidden)`` 3-D contract.
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _make_mla_config(
            csa_compress_ratios=[4, 128, 4, 128],
            csa_window_size=8,
            dsa_indexer_topk=8,
            dsa_indexer_loss_coeff=1.0,
        )
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        yield
        Utils.destroy_model_parallel()

    def _get_layer_number(self, compress_ratio):
        """Return the (1-indexed) layer number whose
        ``csa_compress_ratios`` entry matches ``compress_ratio``."""
        for i, r in enumerate(self.config.csa_compress_ratios):
            if r == compress_ratio:
                return i + 1
        raise ValueError(f"No layer with compress_ratio={compress_ratio}")

    def _build_csa(self, compress_ratio, *, force_unfused_dsa=False):
        # ``force_unfused_dsa`` is a config-level attribute consumed by
        # ``CompressedSparseAttention.__init__`` via ``getattr(config,
        # 'force_unfused_dsa', False)``; set it on the config object
        # before constructing the module.
        self.config.force_unfused_dsa = force_unfused_dsa
        return CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=self._get_layer_number(compress_ratio),
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=compress_ratio,
        ).cuda()

    def _make_thd_inputs(self, seg_lens):
        """Build a ``(query, key, value, x, qr, packed_seq_params)``
        tuple for a multi-segment THD batch of given segment lengths.
        """
        total = sum(seg_lens)
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim
        query = torch.randn(total, np_, hn, dtype=torch.bfloat16, device='cuda')
        key = torch.randn(total, 1, 1, hn, dtype=torch.bfloat16, device='cuda')
        value = key.clone()
        x = torch.randn(total, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda')
        qr = torch.randn(total, 1, self.config.q_lora_rank, dtype=torch.bfloat16, device='cuda')
        packed = _make_packed_seq_params_thd(seg_lens)
        return query, key, value, x, qr, packed

    # ---- Path A (compress_ratio=128: indexer disabled, all-compressed) ----

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_path_a_forward(self):
        """Path A (THD): compress_ratio=128 → indexer=None → attend to
        ALL compressed positions per segment via ``get_compress_topk_idxs_thd``.
        """
        compress_ratio = 128
        csa = self._build_csa(compress_ratio)
        # Make segment lengths long enough that each compresses ≥1 position.
        seg_lens = [compress_ratio * 2 + 50, compress_ratio + 30]
        total = sum(seg_lens)
        query, key, value, x, qr, packed = self._make_thd_inputs(seg_lens)

        csa.eval()
        with torch.no_grad():
            output = csa(
                query=query,
                key=key,
                value=value,
                attention_mask=None,
                x=x,
                qr=qr,
                packed_seq_params=packed,
            )
        np_ = self.config.num_attention_heads
        assert output.shape == (total, 1, np_ * self.config.v_head_dim)
        assert not torch.isnan(output).any()

    # ---- Path B (compress_ratio=4, training): fused × sparse/dense × force_unfused ----

    @pytest.mark.parametrize(
        "sparse_loss, force_unfused_dsa",
        [
            (False, False),  # fused, dense loss (cuDNN dense kernels)
            (True, False),  # fused, sparse loss (cuDNN sparse kernels)
            (True, True),  # force_unfused (PyTorch ref) — uses
            # config.dsa_indexer_use_sparse_loss directly
        ],
        ids=['fused_dense', 'fused_sparse', 'force_unfused'],
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_path_b_training_forward_backward(self, sparse_loss, force_unfused_dsa):
        """Path B (THD training): all three supported combos exercise
        the indexer + KL-loss path with grad flow through Q/K/x/qr.
        """
        # Set the sparse-loss config flag (read inside _forward_thd).
        self.config.dsa_indexer_use_sparse_loss = sparse_loss

        compress_ratio = 4
        csa = self._build_csa(compress_ratio, force_unfused_dsa=force_unfused_dsa)
        # Multi-segment with enough length for indexer top-K to be exercised.
        seg_lens = [compress_ratio * 16, compress_ratio * 8]
        total = sum(seg_lens)
        query, key, value, x, qr, packed = self._make_thd_inputs(seg_lens)

        # Require grad on differentiable inputs (mirrors the SBHD backward test).
        query.requires_grad_(True)
        key.requires_grad_(True)
        x.requires_grad_(True)
        qr.requires_grad_(True)

        csa.train()
        output = csa(
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            x=x,
            qr=qr,
            packed_seq_params=packed,
        )
        np_ = self.config.num_attention_heads
        assert output.shape == (total, 1, np_ * self.config.v_head_dim)
        assert not torch.isnan(output).any()

        # Backward: indexer loss is attached via DSAIndexerLossAutoScaler so
        # ``output.sum().backward()`` triggers grads through both the attn
        # output path AND the indexer-loss path.
        output.sum().backward()
        # Differentiable leaves should have grads.
        assert query.grad is not None and not torch.isnan(query.grad).any()
        assert key.grad is not None and not torch.isnan(key.grad).any()
        # CSA params (compressor + indexer + attn_sink) should be reached.
        seen_any_param_grad = False
        for name, p in csa.named_parameters():
            if p.requires_grad and p.grad is not None:
                seen_any_param_grad = True
                assert not torch.isnan(p.grad).any(), f"param {name} grad has NaN"
        assert seen_any_param_grad, "no CSA param received a gradient"

    # ---- Path C (compress_ratio=4, inference): fused × force_unfused ----

    @pytest.mark.parametrize("force_unfused_dsa", [False, True], ids=['fused', 'force_unfused'])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_path_c_inference_forward(self, force_unfused_dsa):
        """Path C (THD inference): indexer top-K + sparse attn, no loss.
        Both the cuDNN fused path and the PyTorch-ref force_unfused path
        produce a well-formed output.
        """
        compress_ratio = 4
        csa = self._build_csa(compress_ratio, force_unfused_dsa=force_unfused_dsa)
        seg_lens = [compress_ratio * 16, compress_ratio * 12]
        total = sum(seg_lens)
        query, key, value, x, qr, packed = self._make_thd_inputs(seg_lens)

        csa.eval()
        with torch.no_grad():
            output = csa(
                query=query,
                key=key,
                value=value,
                attention_mask=None,
                x=x,
                qr=qr,
                packed_seq_params=packed,
            )
        np_ = self.config.num_attention_heads
        assert output.shape == (total, 1, np_ * self.config.v_head_dim)
        assert not torch.isnan(output).any()

    # ---- B=1 SBHD/THD parity (one happy-path sanity check) ----

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_b1_sbhd_thd_parity_inference_path_c(self):
        """B=1 single-segment THD inference output matches SBHD-b=1 on
        the same data (force_unfused path → fully deterministic, no
        cuDNN/FlashMLA topk-tie nondeterminism).

        Wider tol than the kernel-level tests because the full CSA
        forward chains many bf16 ops together; we just verify "no
        plumbing bug" rather than tight numerical equality.
        """
        compress_ratio = 4
        # force_unfused → uses the PyTorch indexer reference (no cuDNN
        # radix-topK tie-breaking nondeterminism).
        csa = self._build_csa(compress_ratio, force_unfused_dsa=True)
        sq = compress_ratio * 16
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        torch.manual_seed(7)
        query = torch.randn(sq, 1, np_, hn, dtype=torch.bfloat16, device='cuda')
        key = torch.randn(sq, 1, 1, hn, dtype=torch.bfloat16, device='cuda')
        value = key.clone()
        x = torch.randn(sq, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda')
        qr = torch.randn(sq, 1, self.config.q_lora_rank, dtype=torch.bfloat16, device='cuda')

        csa.eval()
        with torch.no_grad():
            # SBHD path: packed_seq_params=None.
            out_sbhd = csa(
                query=query,
                key=key,
                value=value,
                attention_mask=None,
                x=x,
                qr=qr,
                packed_seq_params=None,
            )
            # THD path: single-segment packed_seq_params. Query is 3-D
            # ``(total_q, np, hn)`` per TE THD convention, so drop the
            # SBHD b=1 head dimension for the THD call.
            packed = _make_packed_seq_params_thd([sq])
            out_thd = csa(
                query=query.squeeze(1),
                key=key,
                value=value,
                attention_mask=None,
                x=x,
                qr=qr,
                packed_seq_params=packed,
            )

        # SBHD output: (sq, 1, np*hn). THD output: (sq, 1, np*hn). Same shape.
        assert out_sbhd.shape == out_thd.shape
        assert torch.allclose(out_sbhd.float(), out_thd.float(), atol=5e-2, rtol=5e-2), (
            f"SBHD/THD B=1 parity failed: max abs diff = "
            f"{(out_sbhd.float() - out_thd.float()).abs().max().item():.4e}"
        )


# ===========================================================================
# _apply_rope direct THD tests (4 corners: ratio={1, >1} × fused={False, True})
# ===========================================================================
#
# Direct tests of ``_apply_rope`` THD branches. Previously these were
# only exercised indirectly via ``Compressor._forward_thd`` (ratio>1)
# and ``CSAIndexer.forward_before_topk`` (ratio=1). Direct tests give
# clearer failure attribution and pin down the contract for each of the
# 4 supported (ratio, apply_rope_fusion) combinations.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestApplyRopeThd:
    """Direct tests of :func:`_apply_rope` THD branches.

    The class is parametrized over ``rope_type`` (``"rope"`` /
    ``"yarn"``), so every test runs with both ``RotaryEmbedding``
    and ``YarnRotaryEmbedding``.

    For each rope type the function has four distinct THD paths:
      * (ratio=1, fused=False): forward ``cu_seqlens`` to the rotary
        module's packed mode + ``apply_rotary_pos_emb``.
      * (ratio=1, fused=True): forward ``cu_seqlens`` to the fused MLA
        RoPE kernel.
      * (ratio>1, fused=False): build a per-segment-strided rotary
        table by slicing a global ``max_seg * ratio`` table with stride
        ``ratio`` per segment, concat into a packed table aligned with
        ``cu_seqlens``, then ``apply_rotary_pos_emb``.
      * (ratio>1, fused=True): same per-segment-strided slice + concat
        construction but applied to cos/sin tables instead of the
        rotary embedding tensor, fed to the fused kernel.

    For each corner we verify:
      * Output shape == input shape (RoPE is in-place w.r.t. shape).
      * No NaN in the output (per-segment).
      * B=1 single-segment THD output matches the equivalent SBHD-b=1
        call on the same input (numerical parity within bf16 tol).
    """

    @pytest.fixture(scope='class', autouse=True, params=["rope", "yarn"], ids=["rope", "yarn"])
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        rope_type = request.param
        cls = request.cls
        cls.rope_type = rope_type
        cls.config = _make_mla_config(rope_type=rope_type)
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        if rope_type == "yarn":
            from megatron.core.models.common.embeddings import YarnRotaryEmbedding

            cls.rotary_pos_emb = YarnRotaryEmbedding(
                cls.config.qk_pos_emb_head_dim,
                rotary_base=cls.config.rotary_base,
                scaling_factor=cls.config.rotary_scaling_factor,
                original_max_position_embeddings=cls.config.original_max_position_embeddings,
                beta_fast=cls.config.beta_fast,
                beta_slow=cls.config.beta_slow,
                mscale=cls.config.mscale,
                mscale_all_dim=cls.config.mscale_all_dim,
                cp_group=cls.pg_collection.cp,
            )
        else:
            from megatron.core.models.common.embeddings import RotaryEmbedding

            cls.rotary_pos_emb = RotaryEmbedding(
                cls.config.qk_pos_emb_head_dim,
                rotary_percent=cls.config.rotary_percent,
                rotary_base=cls.config.rotary_base,
                cp_group=cls.pg_collection.cp,
            )

        cls.pos_dim = cls.config.qk_pos_emb_head_dim
        cls.nope_dim = cls.config.v_head_dim - cls.pos_dim
        cls.head_dim = cls.config.v_head_dim

        yield
        Utils.destroy_model_parallel()

    def _make_input_thd(self, total_q):
        # 3-D input ``(seq, batch=1, head_dim)`` — the shape that
        # ``Compressor._forward_thd`` and ``CSAIndexer.forward_before_topk``
        # feed in (with the dummy ``b=1`` axis preserved). ``_apply_rope``
        # also accepts 4-D (with explicit head dim); both branches go
        # through the same code path after a temporary head-dim insert.
        return torch.randn(total_q, 1, self.head_dim, dtype=torch.bfloat16, device='cuda')

    @pytest.mark.parametrize("ratio", [1, 4], ids=["ratio_1", "ratio_4"])
    @pytest.mark.parametrize("apply_rope_fusion", [False, True], ids=["unfused", "fused"])
    def test_thd_shape_and_no_nan(self, ratio, apply_rope_fusion):
        """All 4 corners produce same-shape, NaN-free output for a
        multi-segment THD batch.
        """
        prev_fusion = self.config.apply_rope_fusion
        self.config.apply_rope_fusion = apply_rope_fusion
        try:
            seg_lens = [16, 24, 8]
            total = sum(seg_lens)
            x = self._make_input_thd(total)
            cu_seqlens = _cu_seqlens(seg_lens, device='cuda')

            out = _apply_rope(
                x,
                self.nope_dim,
                self.pos_dim,
                self.rotary_pos_emb,
                self.config,
                rotary_seq_len=0,  # unused when cu_seqlens supplied
                ratio=ratio,
                cp_group=self.pg_collection.cp,
                cu_seqlens=cu_seqlens,
            )

            tag = f"(rope={self.rope_type}, ratio={ratio}, fused={apply_rope_fusion})"
            assert (
                out.shape == x.shape
            ), f"{tag}: shape {tuple(out.shape)} != input {tuple(x.shape)}"
            offset = 0
            for i, seg_len in enumerate(seg_lens):
                assert not torch.isnan(
                    out[offset : offset + seg_len]
                ).any(), f"{tag}: segment {i} produced NaN"
                offset += seg_len
        finally:
            self.config.apply_rope_fusion = prev_fusion

    @pytest.mark.parametrize("ratio", [1, 4], ids=["ratio_1", "ratio_4"])
    @pytest.mark.parametrize("apply_rope_fusion", [False, True], ids=["unfused", "fused"])
    def test_thd_b1_matches_sbhd_b1(self, ratio, apply_rope_fusion):
        """B=1 single-segment THD matches SBHD-b=1 on the same input
        for all 4 corners. The two paths build their rotary tables
        independently but for a single segment with ``cu_seqlens = [0,
        sq]`` they should produce numerically identical output.
        """
        prev_fusion = self.config.apply_rope_fusion
        self.config.apply_rope_fusion = apply_rope_fusion
        try:
            sq = 16
            x = self._make_input_thd(sq)
            cu_seqlens = _cu_seqlens([sq], device='cuda')

            # SBHD: cu_seqlens=None. For ratio>1 the SBHD branch slices
            # a length ``sq*ratio`` table with stride ratio. ``x`` must
            # have a sequence-first layout (which it does: (sq, 1, head_dim)).
            out_sbhd = _apply_rope(
                x.clone(),
                self.nope_dim,
                self.pos_dim,
                self.rotary_pos_emb,
                self.config,
                rotary_seq_len=sq,
                ratio=ratio,
                cp_group=self.pg_collection.cp,
                cu_seqlens=None,
            )
            out_thd = _apply_rope(
                x.clone(),
                self.nope_dim,
                self.pos_dim,
                self.rotary_pos_emb,
                self.config,
                rotary_seq_len=0,  # unused for THD
                ratio=ratio,
                cp_group=self.pg_collection.cp,
                cu_seqlens=cu_seqlens,
            )

            tag = f"(rope={self.rope_type}, ratio={ratio}, fused={apply_rope_fusion})"
            assert out_sbhd.shape == out_thd.shape, f"{tag} shape mismatch"
            assert torch.allclose(out_sbhd.float(), out_thd.float(), atol=1e-2, rtol=1e-2), (
                f"{tag} SBHD/THD B=1 parity failed: max abs diff = "
                f"{(out_sbhd.float() - out_thd.float()).abs().max().item():.4e}"
            )
        finally:
            self.config.apply_rope_fusion = prev_fusion
