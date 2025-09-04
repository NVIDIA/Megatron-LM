# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
from copy import deepcopy

import pytest
import torch
from packaging.version import Version as PkgVersion
from pytest_mock import mocker

import megatron.core.parallel_state as ps
from megatron.core.datasets.t5_dataset import T5MaskedWordPieceDataset
from megatron.core.models.T5.t5_model import T5Model
from megatron.core.models.T5.t5_spec import (
    get_t5_decoder_with_local_block_spec,
    get_t5_decoder_with_transformer_engine_block_spec,
    get_t5_encoder_with_local_block_spec,
    get_t5_encoder_with_transformer_engine_block_spec,
)
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestT5Model:

    def setup_method(self, method):
        tp = 4
        pp = 1
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp
        )
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            kv_channels=64,
            ffn_hidden_size=3072,
            use_cpu_initialization=True,
            pipeline_dtype=torch.bfloat16,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
        )
        rank = ps.get_pipeline_model_parallel_rank()
        world_size = ps.get_pipeline_model_parallel_world_size()
        en_block_spec = get_t5_encoder_with_transformer_engine_block_spec(12)
        de_block_spec = get_t5_decoder_with_transformer_engine_block_spec(12)

        pre_process = True
        post_process = True
        add_encoder = True
        add_decoder = True

        self.t5_model = T5Model(
            encoder_config=transformer_config,
            config=transformer_config,
            transformer_encoder_layer_spec=en_block_spec,
            transformer_decoder_layer_spec=de_block_spec,
            vocab_size=29184,
            max_sequence_length=4,
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=add_encoder,
            add_decoder=add_decoder,
            model_comm_pgs=ModelCommProcessGroups.use_mpu_process_groups(
                required_pgs=['tp', 'cp', 'pp']
            ),
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.t5_model, T5Model)
        assert Utils.world_size == 8

        assert self.t5_model.max_sequence_length == 4
        assert self.t5_model.add_decoder
        assert self.t5_model.decoder.num_layers_per_pipeline_rank == 12
        assert self.t5_model.decoder.num_layers_per_pipeline_rank == 12
        assert self.t5_model.pre_process
        assert self.t5_model.post_process

    def test_set_input_tensor(self):
        config: TransformerConfig = self.t5_model.config
        sequence_length = self.t5_model.max_sequence_length
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        self.t5_model.set_input_tensor(input_tensor)

        if self.t5_model.add_encoder:
            assert self.t5_model.encoder.input_tensor.shape[0] == sequence_length
            assert self.t5_model.encoder.input_tensor.shape[1] == micro_batch_size
            assert self.t5_model.encoder.input_tensor.shape[2] == config.hidden_size
        else:
            assert self.t5_model.encoder is None
            assert self.t5_model.encoder_hidden_state.shape[0] == sequence_length
            assert self.t5_model.encoder_hidden_state.shape[1] == micro_batch_size
            assert self.t5_model.encoder_hidden_state.shape[2] == config.hidden_size

    @pytest.mark.flaky_in_dev
    def test_post_process_forward(self):
        pass

    def test_forward_output_encoder_hidden_only(self):
        pass

    def test_forward_with_encoder_hidden_states(self):
        pass

    def test_no_post_process_forward(self):
        pass

    def test_no_preprocess_forward(self):
        pass

    def test_state_dict_for_save_checkpoint(self):
        pass

    def test_load_state_dict(self):
        pass


class TestT5ModelAttentionDimensions:

    def teardown_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)

    def setup_method(self, method):
        self.bs = 4
        self.seq_len = 512
        self.seq_len_dec = 128
        self.encoder_tokens = torch.ones([self.bs, self.seq_len])
        self.decoder_tokens = torch.ones([self.bs, self.seq_len_dec])
        self.encoder_mask = torch.ones([self.bs, self.seq_len]) < 0.5
        self.decoder_mask = torch.ones([self.bs, self.seq_len_dec]) < 0.5

    @pytest.mark.internal
    def test_local_spec(self):
        encoder_mask, decoder_mask, encoder_decoder_mask = (
            T5MaskedWordPieceDataset.config_attention_mask(
                self.encoder_tokens,
                self.decoder_tokens,
                self.encoder_mask,
                self.decoder_mask,
                use_local=True,
            )
        )

        assert list(encoder_mask.shape) == [self.bs, 1, self.seq_len, self.seq_len]
        assert list(decoder_mask.shape) == [self.bs, 1, self.seq_len_dec, self.seq_len_dec]
        assert list(encoder_decoder_mask.shape) == [self.bs, 1, self.seq_len_dec, self.seq_len]

    @pytest.mark.internal
    def test_transformer_engine_version_1_10(self):
        encoder_mask, decoder_mask, encoder_decoder_mask = (
            T5MaskedWordPieceDataset.config_attention_mask(
                self.encoder_tokens,
                self.decoder_tokens,
                self.encoder_mask,
                self.decoder_mask,
                use_local=False,
                test_te_version="1.10",
            )
        )

        assert list(encoder_mask.shape) == [self.bs, 1, 1, self.seq_len]
        assert decoder_mask is None
        assert list(encoder_decoder_mask[0].shape) == [self.bs, 1, 1, self.seq_len_dec]
        assert list(encoder_decoder_mask[1].shape) == [self.bs, 1, 1, self.seq_len]

    @pytest.mark.internal
    def test_transformer_engine_version_1_7_to_1_10_flashfused_attn(self):
        os.environ['NVTE_FLASH_ATTN'] = '1'
        os.environ['NVTE_FUSED_ATTN'] = '1'

        encoder_mask, decoder_mask, encoder_decoder_mask = (
            T5MaskedWordPieceDataset.config_attention_mask(
                self.encoder_tokens,
                self.decoder_tokens,
                self.encoder_mask,
                self.decoder_mask,
                use_local=False,
                test_te_version="1.8",
            )
        )

        assert list(encoder_mask.shape) == [self.bs, 1, 1, self.seq_len]
        assert decoder_mask is None
        assert list(encoder_decoder_mask[0].shape) == [self.bs, 1, 1, self.seq_len_dec]
        assert list(encoder_decoder_mask[1].shape) == [self.bs, 1, 1, self.seq_len]

    @pytest.mark.internal
    def test_transformer_engine_version_1_7_to_1_10_unfused_attention(self):
        os.environ['NVTE_FLASH_ATTN'] = '0'
        os.environ['NVTE_FUSED_ATTN'] = '0'

        encoder_mask, decoder_mask, encoder_decoder_mask = (
            T5MaskedWordPieceDataset.config_attention_mask(
                self.encoder_tokens,
                self.decoder_tokens,
                self.encoder_mask,
                self.decoder_mask,
                use_local=False,
                test_te_version="1.8",
            )
        )

        assert list(encoder_mask.shape) == [self.bs, 1, self.seq_len, self.seq_len]
        assert decoder_mask is None
        assert list(encoder_decoder_mask.shape) == [self.bs, 1, self.seq_len_dec, self.seq_len]

    @pytest.mark.internal
    def test_transformer_engine_version_less_than_1_7(self):
        os.environ['NVTE_FLASH_ATTN'] = '1'
        with pytest.raises(Exception) as exc_info:
            encoder_mask, decoder_mask, encoder_decoder_mask = (
                T5MaskedWordPieceDataset.config_attention_mask(
                    self.encoder_tokens,
                    self.decoder_tokens,
                    self.encoder_mask,
                    self.decoder_mask,
                    use_local=False,
                    test_te_version="1.5",
                )
            )

        assert str(exc_info.value) == (
            "Flash and fused attention is not supported with transformer "
            "engine version < 1.7. Set NVTE_FLASH_ATTN=0 and NVTE_FUSED_ATTN=0"
            "or upgrade transformer engine >= 1.7"
        )
