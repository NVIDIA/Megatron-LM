# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# Note: this test requires TE >= 0.13 as well as Flash Attention to run
# FIXME this unit test doesn't work in the current test container. to be fixed soon
"""
def make_test_packed_seq_params(sequence_length):
    cu_seqlens = torch.IntTensor([0, 6, 19, 22, sequence_length]).cuda()
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen, _ = seqlens.max(dim=0, keepdim=True)
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
    )
    return packed_seq_params


class TestParallelAttentionWithPackedSequence:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1,1)
        model_parallel_cuda_manual_seed(123)
        # use BF16 and a large enough hidden size to enable FlashAttention for thd format.
        self.transformer_config = TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True,
                                                    bf16=True, params_dtype=torch.bfloat16,
                                                    pipeline_dtype=torch.bfloat16, autocast_dtype=torch.bfloat16)
        self.parallel_attention = SelfAttention(self.transformer_config,
                                                get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
                                                layer_number=1,
                                                attn_mask_type=AttnMaskType.causal)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self):

        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 1

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size))
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None

        packed_seq_params = make_test_packed_seq_params(sequence_length)
        output, bias = self.parallel_attention(hidden_states, attention_mask, packed_seq_params=packed_seq_params)

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    def test_fused_rope_gpu_forward(self):
        self.parallel_attention.config.apply_rope_fusion = True
        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 1

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size))
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None
        rotary_pos_emb = torch.ones(sequence_length, 1, 1, self.parallel_attention.config.kv_channels).cuda()
        
        packed_seq_params = make_test_packed_seq_params(sequence_length)
        output, bias = self.parallel_attention(hidden_states, attention_mask, packed_seq_params=packed_seq_params)

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size
        self.parallel_attention.config.apply_rope_fusion = False

    def test_checkpointed_gpu_forward(self):
        transformer_config = self.transformer_config
        transformer_config.recompute_granularity='selective'
        checkpointed_parallel_attention = SelfAttention(transformer_config,
                                                        get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
                                                        layer_number=1,
                                                        attn_mask_type=AttnMaskType.causal)
        config = checkpointed_parallel_attention.config

        sequence_length = 32
        micro_batch_size = 1

        checkpointed_parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, checkpointed_parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None

        packed_seq_params = make_test_packed_seq_params(sequence_length)
        output, bias = checkpointed_parallel_attention(hidden_states, attention_mask, packed_seq_params=packed_seq_params)

        assert config.recompute_granularity == 'selective'
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size
"""
