# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
from torch import Tensor

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils
from megatron.core.context_parallel import TEContextParallelHandler
from megatron.core import parallel_state

from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.process_groups_config import ProcessGroupCollection


class TestContextParallelHandler:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1, context_parallel_size=8)
        model_parallel_cuda_manual_seed(123)
        # use BF16 and a large enough hidden size to enable FlashAttention for thd format.
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=2048,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
            context_parallel_size=8,
        )

        pgs = ProcessGroupCollection()
        pgs.tp = parallel_state.get_tensor_model_parallel_group()
        pgs.cp = parallel_state.get_context_parallel_group()

        self.parallel_attention = SelfAttention(
            self.transformer_config,
            get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pgs,
        )

        self.rope = RotaryEmbedding(
            kv_channels=self.transformer_config.kv_channels, rotary_percent=1.0
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skip
    def test_dispatch_combine_thd(self):
        config = self.parallel_attention.config
        self.parallel_attention.cuda()

        cu_seqlens = torch.IntTensor([0, 18, 44, 52, 96, 128]).cuda()
        sequence_length = 128
        micro_batch_size = 1

        # [sequence length, batch size, hidden size]
        hidden_states = torch.arange(0, sequence_length)[:, None, None].expand(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        handler = TEContextParallelHandler(
            qkv_format="thd",
            attn_backend="te_fused_attention",
            cp_comm_type="p2p",
            cp_group=parallel_state.get_context_parallel_group(),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
        )

        rotary_pos_emb = self.rope(handler.max_seqlen_q, cp_handler=handler)

        local_states = handler.dispatch(seq_dim=0, tensor=hidden_states)
        total_states = handler.combine(seq_dim=0, tensor=local_states)

        assert torch.equal(total_states, hidden_states)

        hidden_states_rotated_ref = apply_rotary_pos_emb(
            hidden_states,
            rotary_pos_emb,
            config=config,
            cu_seqlens=cu_seqlens,
            cp_group=parallel_state.get_tensor_model_parallel_group(),
        )

        local_states_rotated = handler.apply_rotary_pos_emb(
            local_states, rotary_pos_emb, config=config
        )

        total_states_rotated = handler.combine(seq_dim=0, tensor=local_states_rotated)
        torch.testing.assert_close(
            total_states_rotated, hidden_states_rotated_ref, atol=1e-10, rtol=1e-4
        )

    @pytest.mark.skip
    def test_dispatch_combine_sbhd(self):
        config = self.parallel_attention.config
        self.parallel_attention.cuda()

        sequence_length = 128
        micro_batch_size = 1

        # [batch size, sequence length, hidden size]
        hidden_states = torch.arange(0, sequence_length)[:, None, None, None].expand(
            (sequence_length, micro_batch_size, config.num_attention_heads, config.kv_channels)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        handler = TEContextParallelHandler(
            qkv_format="sbhd",
            attn_backend="te_fused_attention",
            cp_comm_type="p2p",
            cp_group=parallel_state.get_context_parallel_group(),
            max_seqlen_q=sequence_length,
            max_seqlen_kv=sequence_length,
        )

        rotary_pos_emb = self.rope(handler.max_seqlen_q, cp_handler=handler)
        rotary_pos_emb_ref = self.rope(handler.max_seqlen_q)

        local_states = handler.dispatch(seq_dim=0, tensor=hidden_states)
        total_states = handler.combine(seq_dim=0, tensor=local_states)

        assert torch.equal(total_states, hidden_states)

        hidden_states_rotated_ref = apply_rotary_pos_emb(
            hidden_states,
            rotary_pos_emb_ref,
            config=config,
            cp_group=parallel_state.get_tensor_model_parallel_group(),
        )

        local_states_rotated = handler.apply_rotary_pos_emb(
            local_states, rotary_pos_emb, config=config
        )

        total_states_rotated = handler.combine(seq_dim=0, tensor=local_states_rotated)
        torch.testing.assert_close(
            total_states_rotated, hidden_states_rotated_ref, atol=1e-10, rtol=1e-4
        )

    @pytest.mark.skip
    def test_parallel_attention_sbhd(self):
        config = self.parallel_attention.config
        self.parallel_attention.cuda()

        sequence_length = 8192 * 8
        micro_batch_size = 1

        # [batch size, sequence length, hidden size]
        hidden_states = torch.arange(0, sequence_length)[:, None, None].expand(
            (sequence_length, micro_batch_size, config.num_attention_heads * config.kv_channels)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        handler = TEContextParallelHandler(
            qkv_format="sbhd",
            attn_backend="te_fused_attention",
            cp_comm_type="p2p",
            cp_group=parallel_state.get_context_parallel_group(),
            max_seqlen_q=sequence_length,
            max_seqlen_kv=sequence_length,
        )

        local_hidden_states = handler.dispatch(seq_dim=0, tensor=hidden_states)
        rotary_pos_emb = self.rope(handler.max_seqlen_q, cp_handler=handler)

        torch.cuda.profiler.start()
        for i in range(100):
            with torch.cuda.nvtx.range(f"iter_{i}"):
                out = self.parallel_attention(
                    local_hidden_states,
                    attention_mask=None,
                    rotary_pos_emb=rotary_pos_emb,
                    cp_handler=handler,
                )
        torch.cuda.profiler.stop()
