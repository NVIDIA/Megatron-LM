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
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars


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
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=['tp', 'cp', 'pp', 'embd']
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

    @pytest.mark.flaky
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


def test_constructor_process_groups(mocker):
    """T5 and LanguageModule must agree on the caller's process-group collection."""
    clear_nvte_env_vars()
    mocker.patch("torch.distributed.is_initialized", return_value=True)
    pg_collection = ProcessGroupCollection(
        tp=mocker.sentinel.tp,
        cp=mocker.sentinel.cp,
        pp=mocker.sentinel.pp,
        embd=mocker.sentinel.embd,
    )
    global_groups = mocker.patch.object(
        ProcessGroupCollection, "use_mpu_process_groups", return_value=pg_collection
    )
    global_groups.side_effect = AssertionError("Explicit groups must not read the global grid")

    config = TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=4)
    # An empty pipeline stage exercises both constructors without constructing GPU layers.
    model = T5Model(
        config=config,
        encoder_config=config,
        transformer_encoder_layer_spec=None,
        transformer_decoder_layer_spec=None,
        vocab_size=32,
        max_sequence_length=4,
        pre_process=False,
        post_process=False,
        add_encoder=False,
        add_decoder=False,
        pg_collection=pg_collection,
    )

    assert model.pg_collection is pg_collection
    assert model.tp_group is pg_collection.tp
    assert model.cp_group is pg_collection.cp
    assert model.pp_group is pg_collection.pp
    assert model.embd_group is pg_collection.embd
    global_groups.assert_not_called()


@pytest.mark.parametrize(
    'pp_size, pre_process, post_process, tied, embedding_state, valid',
    [
        (1, False, False, False, 'omitted', False),
        (2, False, True, True, 'omitted', False),
        (1, True, True, True, 'none', True),
        (2, False, False, True, 'none', True),
        (2, True, False, False, 'none', True),
        (2, False, True, False, 'none', True),
        (2, True, False, True, 'none', False),
        (2, True, True, True, 'none', False),
        (2, False, True, True, 'none', False),
        (2, True, False, True, 'nonmember', False),
        (2, False, False, True, 'nonmember', False),
    ],
)
def test_constructor_embedding_group_contract(
    mocker, pp_size, pre_process, post_process, tied, embedding_state, valid
):
    """An explicit collection cannot silently disable tied pipeline embedding synchronization."""
    clear_nvte_env_vars()
    mocker.patch('torch.distributed.is_initialized', return_value=True)
    mocker.patch('megatron.core.models.T5.t5_model.LanguageModelEmbedding')
    mocker.patch('megatron.core.models.T5.t5_model.T5LMHead')
    mocker.patch.object(T5Model, 'setup_embeddings_and_output_layer')
    global_groups = mocker.patch.object(
        ProcessGroupCollection,
        'use_mpu_process_groups',
        side_effect=AssertionError('explicit collection must not resolve global groups'),
    )
    pg_collection = ProcessGroupCollection(
        tp=mocker.sentinel.tp, cp=mocker.sentinel.cp, pp=mocker.sentinel.pp
    )
    if embedding_state != 'omitted':
        pg_collection.embd = (
            None if embedding_state == 'none' else torch.distributed.GroupMember.NON_GROUP_MEMBER
        )
    config = TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.float32,
    )
    kwargs = dict(
        config=config,
        encoder_config=config,
        transformer_encoder_layer_spec=None,
        transformer_decoder_layer_spec=None,
        vocab_size=32,
        max_sequence_length=4,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=False,
        add_decoder=False,
        share_embeddings_and_output_weights=tied,
        pg_collection=pg_collection,
    )
    if valid:
        model = T5Model(**kwargs)
        assert model.embd_group is pg_collection.embd
    else:
        with pytest.raises(AssertionError, match='embd'):
            T5Model(**kwargs)
    global_groups.assert_not_called()


def test_pipeline_tied_embeddings_use_supplied_group(mocker):
    """Real PP endpoints synchronize nonzero embedding weights through the supplied group."""
    clear_nvte_env_vars()
    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip('requires a world size divisible by two')
    Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(123)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'cp', 'pp', 'embd', 'gtp_remat']
        )
        mocker.patch.object(
            ProcessGroupCollection,
            'use_mpu_process_groups',
            side_effect=AssertionError('explicit collection must not resolve global groups'),
        )
        config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            use_cpu_initialization=True,
        )
        model = T5Model(
            config=config,
            encoder_config=config,
            transformer_encoder_layer_spec=None,
            transformer_decoder_layer_spec=None,
            vocab_size=32,
            max_sequence_length=4,
            pre_process=pg_collection.pp.rank() == 0,
            post_process=pg_collection.pp.rank() == 1,
            add_encoder=False,
            add_decoder=False,
            share_embeddings_and_output_weights=True,
            pg_collection=pg_collection,
        )
        weight = model.shared_embedding_or_output_weight().detach()
        assert torch.count_nonzero(weight).item() > 0
        endpoint_weights = [torch.empty_like(weight) for _ in range(pg_collection.embd.size())]
        torch.distributed.all_gather(endpoint_weights, weight, group=pg_collection.embd)
        for endpoint_weight in endpoint_weights:
            torch.testing.assert_close(endpoint_weight, weight, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()


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
