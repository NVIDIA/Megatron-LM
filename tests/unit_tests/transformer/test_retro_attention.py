# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

# import pytest
import torch
import types

from megatron.core.models.retro import RetroConfig, get_retro_decoder_block_spec
from megatron.core.models.retro.decoder_attention import (
    RetroDecoderCrossAttention,
    RetroDecoderBiasDropoutAdd,
)
from megatron.core.models.retro.encoder_attention import (
    RetroEncoderCrossAttention,
    RetroEncoderBiasDropoutAdd,
    RetroEncoderLayerNorm,
)
# from megatron.core.transformer.attention import SelfAttention
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import build_module
# from megatron.core.transformer.transformer_config import TransformerConfig
# from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
# from megatron.core.models.retro.decoder_attention import (
#     RetroDecoderBiasDropoutAdd,
#     RetroDecoderCrossAttention,
# )
from tests.unit_tests.test_utilities import Utils


class TestRetroAttention:

    def setup_method(self, method):

        # Setup.
        Utils.initialize_model_parallel(1,1)
        model_parallel_cuda_manual_seed(123)

        # Retro config.
        config = RetroConfig(
            num_layers=12,
            hidden_size=16,
            num_attention_heads=4,
            use_cpu_initialization=True,
            # >>>
            retro_num_neighbors=2,
            retro_preprocess=types.SimpleNamespace(
                # retro_gpt_chunk_length=64,
                # retro_gpt_retrieved_length=128,
                retro_gpt_chunk_length=4,
                retro_gpt_retrieved_length=8,
            ),
            # <<<
        )

        # Retro decoder layer.
        # >>>
        decoder_block_spec = get_retro_decoder_block_spec(
            config, use_transformer_engine=False) # True
        # <<<
        decoder_block = build_module(decoder_block_spec, config=config)
        decoder_layers = [ layer for layer in decoder_block.layers if isinstance(layer.cross_attention, RetroDecoderCrossAttention) ]
        decoder_layer = decoder_layers[0]

        # Retro encoder layer.
        encoder_block = decoder_layer.cross_attention.encoder
        encoder_layers = [ layer for layer in encoder_block.layers if isinstance(layer.cross_attention, RetroEncoderCrossAttention) ]
        encoder_layer = encoder_layers[0]

        self.decoder_attn = decoder_layer.cross_attention
        self.decoder_bda = decoder_layer.cross_attn_bda
        self.encoder_attn = encoder_layer.cross_attention
        self.encoder_bda = encoder_layer.cross_attn_bda
        self.encoder_norm = encoder_layer.pre_mlp_layernorm


    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):

        assert isinstance(self.decoder_attn, RetroDecoderCrossAttention)
        assert isinstance(self.decoder_bda, RetroDecoderBiasDropoutAdd)
        assert isinstance(self.encoder_attn, RetroEncoderCrossAttention)
        assert isinstance(self.encoder_bda, RetroEncoderBiasDropoutAdd)
        assert isinstance(self.encoder_norm, RetroEncoderLayerNorm)

        assert self.decoder_attn.attn.layer_number == 6
        assert self.encoder_attn.attn.layer_number == 1

        get_nparams = lambda m : sum(p.numel() for p in m.parameters())
        assert get_nparams(self.decoder_attn) == 8768
        assert get_nparams(self.decoder_bda) == 0
        assert get_nparams(self.encoder_attn) == 1088
        assert get_nparams(self.encoder_bda) == 0
        assert get_nparams(self.encoder_norm) == 32

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self):

        config = self.decoder_attn.config
        seq_length = 32
        micro_batch_size = 2
        n_chunks_per_sample = seq_length // config.retro_preprocess.retro_gpt_chunk_length

        self.decoder_attn.cuda()
        self.decoder_bda.cuda()
        self.encoder_attn.cuda()
        self.encoder_bda.cuda()
        self.encoder_norm.cuda()

        # Init tensors.
        hidden_states = torch.ones((
            seq_length,
            micro_batch_size,
            config.hidden_size,
        )).cuda()
        attention_mask = None
        decoder_context = torch.ones((
            config.retro_preprocess.retro_gpt_retrieved_length,
            config.retro_num_neighbors * micro_batch_size * n_chunks_per_sample,
            config.hidden_size,
        )).cuda()
        encoder_context = torch.ones((
            config.retro_preprocess.retro_gpt_chunk_length,
            micro_batch_size * n_chunks_per_sample,
            config.hidden_size,
        )).cuda()

        # Forward decoder.
        decoder_attn_output = self.decoder_attn(
            hidden_states,
            attention_mask,
            decoder_context,
        )
        with torch.enable_grad():
            decoder_bda_output = self.decoder_bda(True, True)(
                decoder_attn_output,
                hidden_states,
                config.hidden_dropout,
            )

        # Forward encoder.
        encoder_attn_output_tuples = self.encoder_attn(
            decoder_context,
            None,
            encoder_context,
        )
        with torch.enable_grad():
            encoder_bda_output = self.encoder_bda(True, True)(
                encoder_attn_output_tuples,
                decoder_context,
                config.retro_encoder_hidden_dropout,
            )
        encoder_norm_output = self.encoder_norm(encoder_bda_output)

        # Verify decoder.
        assert set(decoder_attn_output.keys()) == set([ "ns", "bs", "d", "l", "pad", "attention_output", "attention_bias", "context"])
        assert decoder_attn_output["ns"] == seq_length
        assert decoder_attn_output["bs"] == micro_batch_size
        assert decoder_attn_output["d"] == config.hidden_size
        assert decoder_attn_output["l"] == n_chunks_per_sample
        assert decoder_attn_output["pad"] == 3
        assert tuple(decoder_attn_output["attention_output"].shape) == (
            config.retro_preprocess.retro_gpt_chunk_length,
            micro_batch_size * n_chunks_per_sample,
            config.hidden_size,
        )
        assert tuple(decoder_attn_output["attention_bias"].shape) == (
            config.hidden_size,
        )
        assert decoder_attn_output["context"].shape == (
            config.retro_preprocess.retro_gpt_retrieved_length * config.retro_num_neighbors,
            micro_batch_size * n_chunks_per_sample,
            config.hidden_size,
        )
        assert decoder_bda_output.shape == hidden_states.shape

        # Verify encoder.
        assert len(encoder_attn_output_tuples) == config.retro_num_neighbors
        for output, bias, residual in encoder_attn_output_tuples:
            assert tuple(output.shape) == (
                config.retro_preprocess.retro_gpt_retrieved_length,
                micro_batch_size * n_chunks_per_sample,
                config.hidden_size,
            )
            assert tuple(bias.shape) == (config.hidden_size,)
            assert tuple(residual.shape) == (
                config.retro_preprocess.retro_gpt_retrieved_length,
                micro_batch_size * n_chunks_per_sample,
                config.hidden_size,
            )
        assert encoder_bda_output.shape == (
            config.retro_preprocess.retro_gpt_retrieved_length,
            config.retro_num_neighbors * micro_batch_size * n_chunks_per_sample,
            config.hidden_size,
        )
        assert encoder_norm_output.shape == (
            config.retro_preprocess.retro_gpt_retrieved_length,
            config.retro_num_neighbors * micro_batch_size * n_chunks_per_sample,
            config.hidden_size,
        )

    def test_checkpointed_gpu_forward(self):
        raise Exception("hi.")
        transformer_config = self.transformer_config
        transformer_config.recompute_granularity='selective'
        checkpointed_parallel_attention = SelfAttention(transformer_config,
                                                        get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules)
        config = checkpointed_parallel_attention.config

        seq_length = 32
        micro_batch_size = 2

        checkpointed_parallel_attention.cuda()

        # [seq length, batch size, hidden size]
        hidden_states = torch.ones(
            (seq_length, micro_batch_size, checkpointed_parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, seq_length, seq_length), dtype=bool).cuda()

        output, bias = checkpointed_parallel_attention(hidden_states, attention_mask)

        assert config.recompute_granularity == 'selective'
        assert output.shape[0] == seq_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size
