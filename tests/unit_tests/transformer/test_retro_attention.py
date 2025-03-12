# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
import types

import pytest
import torch

from megatron.core.models.retro import RetroConfig, get_retro_decoder_block_spec
from megatron.core.models.retro.decoder_attention import (
    RetroDecoderBiasDropoutAdd,
    RetroDecoderCrossAttention,
)
from megatron.core.models.retro.encoder_attention import (
    RetroEncoderBiasDropoutAdd,
    RetroEncoderCrossAttention,
    RetroEncoderLayerNorm,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from tests.unit_tests.test_utilities import Utils


class TestRetroAttention:

    @classmethod
    def get_config(cls):
        return RetroConfig(
            num_layers=12,
            hidden_size=16,
            num_attention_heads=4,
            use_cpu_initialization=True,
            retro_num_neighbors=2,
            retro_chunk_length=4,
            retro_retrieved_length=8,
            retro_split_preprocessing="98,2,0",
        )

    @classmethod
    def get_modules(cls, config, use_transformer_engine, use_gpu):

        # Retro decoder layer.
        decoder_block_spec = get_retro_decoder_block_spec(
            config, use_transformer_engine=use_transformer_engine
        )
        decoder_block = TransformerBlock(config=config, spec=decoder_block_spec)
        decoder_layers = [
            layer
            for layer in decoder_block.layers
            if isinstance(layer.cross_attention, RetroDecoderCrossAttention)
        ]
        decoder_layer = decoder_layers[0]

        # Retro encoder layer.
        encoder_block = decoder_layer.cross_attention.encoder
        encoder_layers = [
            layer
            for layer in encoder_block.layers
            if isinstance(layer.cross_attention, RetroEncoderCrossAttention)
        ]
        encoder_layer = encoder_layers[0]

        # Modules.
        modules = types.SimpleNamespace(
            decoder_attn=decoder_layer.cross_attention,
            decoder_bda=decoder_layer.cross_attn_bda,
            encoder_attn=encoder_layer.cross_attention,
            encoder_bda=encoder_layer.cross_attn_bda,
            encoder_norm=encoder_layer.pre_mlp_layernorm,
        )

        # GPU.
        if use_gpu:
            [m.cuda() for m in vars(modules).values()]

        return modules

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        os.environ['NVTE_FLASH_ATTN'] = "0"
        os.environ['NVTE_FUSED_ATTN'] = "0"

        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):

        config = self.get_config()
        modules = self.get_modules(config, use_transformer_engine=True, use_gpu=False)

        assert isinstance(modules.decoder_attn, RetroDecoderCrossAttention)
        assert isinstance(modules.decoder_bda, RetroDecoderBiasDropoutAdd)
        assert isinstance(modules.encoder_attn, RetroEncoderCrossAttention)
        assert isinstance(modules.encoder_bda, RetroEncoderBiasDropoutAdd)
        assert isinstance(modules.encoder_norm, RetroEncoderLayerNorm)

        assert modules.decoder_attn.attn.layer_number == 6
        assert modules.encoder_attn.attn.layer_number == 1

        get_nparams = lambda m: sum(p.numel() for p in m.parameters())
        assert get_nparams(modules.decoder_attn) == 8768
        assert get_nparams(modules.decoder_bda) == 0
        assert get_nparams(modules.encoder_attn) == 1088
        assert get_nparams(modules.encoder_bda) == 0
        assert get_nparams(modules.encoder_norm) == 32

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def run_gpu_forward(self, recompute_granularity, use_transformer_engine):

        config = self.get_config()
        config.recompute_granularity = recompute_granularity
        modules = self.get_modules(config, use_transformer_engine, use_gpu=True)

        seq_length = 32
        micro_batch_size = 2
        n_chunks_per_sample = seq_length // config.retro_chunk_length

        # Init tensors.
        hidden_states = torch.ones((seq_length, micro_batch_size, config.hidden_size)).cuda()
        attention_mask = None
        decoder_context = torch.ones(
            (
                config.retro_retrieved_length,
                config.retro_num_neighbors * micro_batch_size * n_chunks_per_sample,
                config.hidden_size,
            )
        ).cuda()
        encoder_context = torch.ones(
            (config.retro_chunk_length, micro_batch_size * n_chunks_per_sample, config.hidden_size)
        ).cuda()

        # Forward decoder.
        decoder_attn_output = modules.decoder_attn(hidden_states, attention_mask, decoder_context)
        with torch.enable_grad():
            decoder_bda_output = modules.decoder_bda(True, True)(
                decoder_attn_output, hidden_states, config.hidden_dropout
            )

        # Forward encoder.
        encoder_attn_output_tuples = modules.encoder_attn(decoder_context, None, encoder_context)
        with torch.enable_grad():
            encoder_bda_output = modules.encoder_bda(True, True)(
                encoder_attn_output_tuples, decoder_context, config.retro_encoder_hidden_dropout
            )
        encoder_norm_output = modules.encoder_norm(encoder_bda_output)

        # Verify decoder.
        assert set(decoder_attn_output.keys()) == set(
            ["ns", "bs", "d", "l", "pad", "attention_output", "attention_bias", "context"]
        )
        assert decoder_attn_output["ns"] == seq_length
        assert decoder_attn_output["bs"] == micro_batch_size
        assert decoder_attn_output["d"] == config.hidden_size
        assert decoder_attn_output["l"] == n_chunks_per_sample
        assert decoder_attn_output["pad"] == 3
        assert tuple(decoder_attn_output["attention_output"].shape) == (
            config.retro_chunk_length,
            micro_batch_size * n_chunks_per_sample,
            config.hidden_size,
        )
        assert tuple(decoder_attn_output["attention_bias"].shape) == (config.hidden_size,)
        assert decoder_attn_output["context"].shape == (
            config.retro_retrieved_length * config.retro_num_neighbors,
            micro_batch_size * n_chunks_per_sample,
            config.hidden_size,
        )
        assert decoder_bda_output.shape == hidden_states.shape

        # Verify encoder.
        assert len(encoder_attn_output_tuples) == config.retro_num_neighbors
        for output, bias, residual in encoder_attn_output_tuples:
            assert tuple(output.shape) == (
                config.retro_retrieved_length,
                micro_batch_size * n_chunks_per_sample,
                config.hidden_size,
            )
            assert tuple(bias.shape) == (config.hidden_size,)
            assert tuple(residual.shape) == (
                config.retro_retrieved_length,
                micro_batch_size * n_chunks_per_sample,
                config.hidden_size,
            )
        assert encoder_bda_output.shape == (
            config.retro_retrieved_length,
            config.retro_num_neighbors * micro_batch_size * n_chunks_per_sample,
            config.hidden_size,
        )
        assert encoder_norm_output.shape == (
            config.retro_retrieved_length,
            config.retro_num_neighbors * micro_batch_size * n_chunks_per_sample,
            config.hidden_size,
        )

    @pytest.mark.flaky_in_dev
    def test_gpu_forward(self):
        for recompute_granularity in (None, 'selective'):
            for use_transformer_engine in (True, False):
                self.run_gpu_forward(recompute_granularity, use_transformer_engine)
