# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

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
