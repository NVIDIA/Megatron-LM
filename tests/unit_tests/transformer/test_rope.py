# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import (
    _set_capture_end,
    _set_capture_start,
    capture_constants,
    release_capture_constants,
)
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from transformer_engine.pytorch.attention.rope import apply_fused_qkv_rotary_pos_emb

    HAVE_FUSED_QKV_ROPE = True
except ImportError:
    HAVE_FUSED_QKV_ROPE = False

from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("rope_type", (RotaryEmbedding, YarnRotaryEmbedding))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_graph_rope_tables_survive_more_than_lru_capacity(rope_type, monkeypatch):
    kwargs = {"original_max_position_embeddings": 32} if rope_type is YarnRotaryEmbedding else {}
    rope = rope_type(8, rotary_percent=1.0, use_cpu_initialization=False, **kwargs)
    vars(rope)["_cuda_graph_rotary_embeddings"] = {}
    rope._cuda_graph_rotary_cache_active = True
    graph = None
    _set_capture_start()
    try:
        first = rope(64)
        first_table = first[0] if isinstance(first, tuple) else first
        first_ptr = first_table.data_ptr()
        expected = first_table.detach().clone()
        static_input = torch.ones_like(first_table)
        graph = torch.cuda.CUDAGraph()
        with capture_constants(graph), torch.cuda.graph(graph):
            captured = rope(64)
            table = captured[0] if isinstance(captured, tuple) else captured
            output = static_input * table
        del first, first_table, captured, table

        for length in range(65, 105):
            rope(length)
        assert len(vars(rope)["_cuda_graph_rotary_embeddings"]) == 41
        rope.clear_cache()
        torch.cuda.empty_cache()
        graph.replay()
        torch.testing.assert_close(output, expected)

        with capture_constants(graph), monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
            cached = rope(64)
            assert (cached[0] if isinstance(cached, tuple) else cached).data_ptr() == first_ptr
            with pytest.raises(RuntimeError, match="not prepared before CUDA graph"):
                rope(105)
    finally:
        _set_capture_end()
        if graph is not None:
            graph.reset()
            release_capture_constants(graph)
        vars(rope).pop("_cuda_graph_rotary_embeddings", None)
        vars(rope).pop("_cuda_graph_rotary_cache_active", None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_retained_te_rotary_cache_allows_local_capture_miss():
    rope = RotaryEmbedding(8, rotary_percent=1.0)
    retained = rope(64)
    vars(rope)["_cuda_graph_rotary_embeddings"] = {"retained": retained}
    expected = rope(65).clone()
    rope.clear_cache()
    graph = torch.cuda.CUDAGraph()
    _set_capture_start()
    try:
        with capture_constants(graph), torch.cuda.graph(graph):
            output = rope(65).clone()
        graph.replay()
        torch.testing.assert_close(output, expected)
        assert vars(rope)["_cuda_graph_rotary_embeddings"]["retained"] is retained
    finally:
        _set_capture_end()
        graph.reset()
        release_capture_constants(graph)


class TestMultimodalRotaryEmbedding:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.kv_channels = 128
        self.rotary_percent = 1.0
        self.rope_gpu_init = MultimodalRotaryEmbedding(self.kv_channels, self.rotary_percent)

    def teardown_method(self, method):
        del self.rope_gpu_init
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_constructor(self):
        assert isinstance(self.rope_gpu_init, MultimodalRotaryEmbedding)
        assert self.rope_gpu_init.inv_freq.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        output = self.rope_gpu_init(torch.Tensor(3, 1, 64), mrope_section=[16, 24, 24])
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("num_positions", [4096, 16384, 21845, 21846, 32768])
    def test_angles_exact_when_tf32_allowed(self, num_positions):
        rope = MultimodalRotaryEmbedding(kv_channels=256, rotary_percent=0.25, rotary_base=10000000)
        positions = torch.arange(90000 - num_positions, 90000, device='cuda', dtype=torch.float32)
        position_ids = positions[None, None, :].expand(3, 1, -1)
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            outputs = {}
            for tf32 in (True, False):
                torch.backends.cuda.matmul.allow_tf32 = tf32
                outputs[tf32] = rope(position_ids, mrope_section=[11, 11, 10])
        finally:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        angles = positions.double()[:, None] * rope.inv_freq.double()[None, :]
        expected = torch.cat((angles, angles), dim=-1)
        torch.testing.assert_close(outputs[True][:, 0, 0].double(), expected, rtol=0, atol=1e-2)
        assert torch.equal(outputs[True], outputs[False])


class _FakeCPGroup:
    """Minimal CP process-group stand-in for packed mRoPE gate checks."""

    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def _cpu_multimodal_rope(kv_channels: int = 64) -> MultimodalRotaryEmbedding:
    """Build MultimodalRotaryEmbedding on CPU without touching CUDA init paths."""
    emb = MultimodalRotaryEmbedding.__new__(MultimodalRotaryEmbedding)
    torch.nn.Module.__init__(emb)
    dim = kv_channels
    emb.rotary_interleaved = False
    emb.interleaved_mrope = False
    emb.seq_len_interpolation_factor = None
    emb.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    emb.cp_group = None
    return emb


class TestMultimodalRotaryEmbeddingPackedSeqCP:
    """Packed THD mRoPE must keep full-sequence freqs under CP (issue #7679)."""

    def test_packed_seq_cp2_matches_cp1_freqs(self):
        """Same position IDs: packed_seq=True with CP=2 equals CP=1 (no early slice)."""
        kv_channels = 64
        seq_len = 8
        mrope_section = [8, 12, 12]
        # Distinct T/H/W positions so a wrong CP slice would change values, not only shape.
        position_ids = (
            torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len).repeat(3, 1, 1)
        )
        position_ids[1] = position_ids[1] + 10
        position_ids[2] = position_ids[2] + 20

        rope = _cpu_multimodal_rope(kv_channels)
        out_cp1 = rope(position_ids, mrope_section, packed_seq=True, cp_group=_FakeCPGroup(0, 1))
        out_cp2 = rope(position_ids, mrope_section, packed_seq=True, cp_group=_FakeCPGroup(0, 2))

        assert out_cp1.shape[0] == seq_len
        assert out_cp2.shape[0] == seq_len
        assert torch.equal(out_cp1, out_cp2)

    def test_non_packed_cp_still_slices(self):
        """Non-packed CP>1 must keep the intentional early slice (do not regress)."""
        kv_channels = 64
        seq_len = 8
        mrope_section = [8, 12, 12]
        position_ids = torch.zeros(3, 1, seq_len, dtype=torch.float32)
        rope = _cpu_multimodal_rope(kv_channels)
        sliced = torch.zeros(seq_len // 2, 1, 1, kv_channels)

        with patch(
            'megatron.core.models.common.embeddings.rotary_pos_embedding.get_pos_emb_on_this_cp_rank',
            return_value=sliced,
        ) as mock_slice:
            out = rope(position_ids, mrope_section, packed_seq=False, cp_group=_FakeCPGroup(0, 2))

        mock_slice.assert_called_once()
        assert out.shape[0] == seq_len // 2

    def test_packed_seq_skips_cp_slice_helper(self):
        """packed_seq=True must not call get_pos_emb_on_this_cp_rank."""
        kv_channels = 64
        seq_len = 8
        mrope_section = [8, 12, 12]
        position_ids = torch.zeros(3, 1, seq_len, dtype=torch.float32)
        rope = _cpu_multimodal_rope(kv_channels)

        with patch(
            'megatron.core.models.common.embeddings.rotary_pos_embedding.get_pos_emb_on_this_cp_rank'
        ) as mock_slice:
            out = rope(position_ids, mrope_section, packed_seq=True, cp_group=_FakeCPGroup(1, 2))

        mock_slice.assert_not_called()
        assert out.shape[0] == seq_len


class TestRotaryEmbedding:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.kv_channels = 8
        self.rotary_percent = 1.0
        self.rope_cpu_init = RotaryEmbedding(
            self.kv_channels, self.rotary_percent, use_cpu_initialization=True
        )
        self.rope_gpu_init = RotaryEmbedding(
            self.kv_channels, self.rotary_percent, use_cpu_initialization=False
        )

    def teardown_method(self, method):
        del self.rope_gpu_init
        del self.rope_cpu_init
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_constructor(self):
        assert isinstance(self.rope_cpu_init, RotaryEmbedding)
        assert self.rope_cpu_init.inv_freq.device.type == 'cpu'
        assert isinstance(self.rope_gpu_init, RotaryEmbedding)
        assert self.rope_gpu_init.inv_freq.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        output = self.rope_gpu_init(64)
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_forward(self):
        output = self.rope_cpu_init(64)
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_get_cos_sin(self):
        """The fused-RoPE cache path lazily moves CPU-initialized frequencies to CUDA."""
        assert self.rope_cpu_init.inv_freq.device.type == 'cpu'

        cos, sin = self.rope_cpu_init.get_cos_sin(64)
        expected_cos, expected_sin = self.rope_gpu_init.get_cos_sin(64)

        assert self.rope_cpu_init.inv_freq.device.type == 'cuda'
        assert cos.device.type == sin.device.type == 'cuda'
        assert torch.allclose(cos, expected_cos, atol=1e-5)
        assert torch.allclose(sin, expected_sin, atol=1e-5)


class TestQKVRotaryEmbedding:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.seq_len = 64
        self.num_heads = 1
        self.kv_channels = 128
        self.rotary_percent = 1.0
        self.rope_gpu_init = RotaryEmbedding(
            self.kv_channels, self.rotary_percent, use_cpu_initialization=False
        )
        self.transformer_config = TransformerConfig(
            num_attention_heads=self.num_heads, num_layers=1, apply_rope_fusion=True
        )

    def teardown_method(self, method):
        del self.rope_gpu_init
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_constructor(self):
        assert isinstance(self.rope_gpu_init, RotaryEmbedding)
        assert self.rope_gpu_init.inv_freq.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not HAVE_FUSED_QKV_ROPE, reason="Fused QKV RoPE not available.")
    def test_gpu_forward(self):
        pos_embed = self.rope_gpu_init(self.seq_len)
        assert pos_embed.shape[0] == self.seq_len
        assert pos_embed.shape[1] == 1
        assert pos_embed.shape[2] == 1
        assert pos_embed.shape[3] == self.kv_channels
        assert pos_embed.dtype == torch.float32
        assert pos_embed.device.type == 'cuda'

        qkv_split_arg_list = [self.kv_channels * 4, self.kv_channels, self.kv_channels]
        # Create input tensors
        qkv = torch.randn(self.seq_len, 1, self.num_heads, self.kv_channels * 6, device="cuda")
        query_in, key_in, value_in = torch.split(qkv, qkv_split_arg_list, dim=3)

        query_in = query_in.reshape(query_in.shape[0], query_in.shape[1], -1, self.kv_channels)
        q_out_ref = apply_rotary_pos_emb(query_in, pos_embed, self.transformer_config)
        k_out_ref = apply_rotary_pos_emb(key_in, pos_embed, self.transformer_config)
        q_out, k_out, _ = apply_fused_qkv_rotary_pos_emb(
            qkv, pos_embed, pos_embed, qkv_split_arg_list
        )

        assert (
            q_out_ref.numel() == q_out.numel()
        ), f"Output sizes do not match for Q: {q_out.shape} != {q_out_ref.shape}"
        assert (
            k_out_ref.numel() == k_out.numel()
        ), f"Output sizes do not match for K: {k_out.shape} != {k_out_ref.shape}"
        assert torch.allclose(q_out_ref, q_out), f"Outputs do not match for Q"
        assert torch.allclose(k_out_ref, k_out), f"Outputs do not match for K"
