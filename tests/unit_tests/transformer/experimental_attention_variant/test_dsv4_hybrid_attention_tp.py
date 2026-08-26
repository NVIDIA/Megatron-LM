# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    _SEED,
    _build_attention,
    _make_config,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionTP:
    """Shape and one-layer backward gate for the experimental TP path."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        request.cls.config = _make_config(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=16,
            v_head_dim=64,
            qk_pos_emb_head_dim=32,
            q_lora_rank=64,
            o_groups=8,
            o_lora_rank=64,
            csa_compress_ratios=[0],
            csa_window_size=8,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=32,
            dsa_indexer_topk=8,
        )
        request.cls.pg = ProcessGroupCollection.use_mpu_process_groups()
        yield
        Utils.destroy_model_parallel()

    def test_tp_local_parameter_shapes(self):
        attn = _build_attention(self.config, layer_number=1, pg_collection=self.pg).cuda()

        assert attn.linear_q_up_proj.weight.shape == (
            self.config.num_attention_heads * self.config.v_head_dim // 2,
            self.config.q_lora_rank,
        )
        assert attn.linear_kv_proj.weight.shape == (
            self.config.v_head_dim,
            self.config.hidden_size,
        )
        assert attn.linear_o_group_proj.shape == (
            self.config.o_groups * self.config.o_lora_rank // 2,
            self.config.num_attention_heads * self.config.v_head_dim // self.config.o_groups,
        )
        assert attn.core_attention.attn_sink.shape == (
            self.config.num_attention_heads // 2,
        )

    def test_tp_window_attention_forward_backward(self):
        attn = _build_attention(self.config, layer_number=1, pg_collection=self.pg).cuda().train()
        local_seq = 32
        hidden = torch.randn(
            local_seq,
            1,
            self.config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        output, bias = attn(hidden_states=hidden, attention_mask=None)
        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()
        output.float().square().mean().backward()
        assert hidden.grad is not None
        assert attn.linear_kv_proj.weight.grad is not None
        assert attn.linear_q_up_proj.weight.grad is not None
        assert attn.linear_o_group_proj.grad is not None
