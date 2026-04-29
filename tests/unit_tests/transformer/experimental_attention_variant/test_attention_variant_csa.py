# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

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
    get_compress_topk_idxs,
    get_window_topk_idxs,
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
        rope_type='rope',
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
        cls.config = _make_mla_config(csa_compress_ratios=[4, 128, 4, 128],)
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
            self.config.dsa_indexer_head_dim
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
        ).cuda()

        query = torch.randn(seq_len, batch_size, np_, hn, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, 1, hn, dtype=torch.bfloat16).cuda()
        value = key.clone()
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)

        assert output.shape == (seq_len, batch_size, np_ * hn)
        assert not torch.isnan(output).any()
