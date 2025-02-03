# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
from importlib.metadata import version

import pytest
import torch
from packaging.version import Version as PkgVersion
from pytest_mock import mocker

from megatron.core.models.bert.bert_layer_specs import (
    bert_layer_local_spec,
    bert_layer_with_transformer_engine_spec,
)
from megatron.core.models.bert.bert_model import BertModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestBertModel:

    def setup_method(self, method):
        tp = 1
        pp = 1
        Utils.initialize_model_parallel(tp, pp)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            perform_initialization=True,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            pipeline_dtype=torch.bfloat16,
            attention_backend=AttnBackend.unfused,
        )
        self.bert_model = BertModel(
            config=transformer_config,
            num_tokentypes=0,
            transformer_layer_spec=bert_layer_with_transformer_engine_spec,
            vocab_size=100,
            max_sequence_length=4,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.bert_model, BertModel)

        assert self.bert_model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in self.bert_model.parameters()])
        assert num_weights == 6702

    @pytest.mark.internal
    def test_set_input_tensor(self):
        config: TransformerConfig = self.bert_model.config
        sequence_length = self.bert_model.max_sequence_length
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        self.bert_model.set_input_tensor(input_tensor)

        assert self.bert_model.encoder.input_tensor.shape[0] == sequence_length
        assert self.bert_model.encoder.input_tensor.shape[1] == micro_batch_size
        assert self.bert_model.encoder.input_tensor.shape[2] == config.hidden_size

    @pytest.mark.internal
    def test_post_process_forward(self):
        config: TransformerConfig = self.bert_model.config
        sequence_length = self.bert_model.max_sequence_length
        micro_batch_size = 2

        self.bert_model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones((micro_batch_size, sequence_length), dtype=bool).cuda()

        logits = self.bert_model.forward(input_ids=input_ids, attention_mask=attention_mask)

        assert logits[0].shape[0] == micro_batch_size
        assert logits[0].shape[1] == sequence_length
        assert logits[0].shape[2] == self.bert_model.vocab_size


class TestBertModelAttentionDimensions:

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            pipeline_dtype=torch.bfloat16,
            attention_backend=AttnBackend.auto,
        )
        # This should convert arbitray mask to padding mask
        self.bert_model = BertModel(
            config=self.transformer_config,
            num_tokentypes=0,
            transformer_layer_spec=bert_layer_with_transformer_engine_spec,
            vocab_size=100,
            max_sequence_length=4,
        )

    @pytest.mark.internal
    def test_local_spec(self, mocker):
        self.bert_model.config.attention_backend = AttnBackend.local
        self.bert_model.transformer_layer_spec = bert_layer_local_spec
        attn_mask_dimensions = self.bert_model._sanity_check_attention_and_get_attn_mask_dimension()
        assert (
            attn_mask_dimensions == "b1ss"
        ), f"Expected b1ss for attn_mask_dimensions but got {attn_mask_dimensions}"

    @pytest.mark.internal
    def test_local_spec_exception(self, mocker):
        self.bert_model.config.attention_backend = AttnBackend.flash
        self.bert_model.transformer_layer_spec = bert_layer_local_spec
        with pytest.raises(Exception) as exc_info:
            self.bert_model._sanity_check_attention_and_get_attn_mask_dimension()
        assert (
            str(exc_info.value)
            == 'Expected AttnBackend to be local or auto while using mcore self attention, but found AttnBackend.flash. Set --attn-backend to local or dont use MCore SelfAttention submodule in layer specs'
        )

    @pytest.mark.internal
    def test_transformer_engine_version_1_10(self, mocker):
        bert_layer_with_transformer_engine_spec.submodules.self_attention.params[
            'attn_mask_type'
        ] == AttnMaskType.arbitrary

        mocker.patch("megatron.core.utils.get_te_version", return_value=PkgVersion("1.10"))
        self.bert_model.transformer_layer_spec = bert_layer_with_transformer_engine_spec
        attn_mask_dimensions = self.bert_model._sanity_check_attention_and_get_attn_mask_dimension()
        attn_mask_type = self.bert_model.transformer_layer_spec.submodules.self_attention.params[
            'attn_mask_type'
        ]
        assert (
            attn_mask_type == AttnMaskType.padding
        ), f"Exepcted attn mask type to be padding, but got {attn_mask_type}"
        assert (
            attn_mask_dimensions == "b11s"
        ), f"Expected b11s for attn_mask_dimensions but got {attn_mask_dimensions}"

    @pytest.mark.internal
    def test_transformer_engine_version_1_7_to_1_10_flash_attn(self, mocker):
        self.bert_model.config.attention_backend = AttnBackend.flash
        mocker.patch("megatron.core.utils.get_te_version", return_value=PkgVersion("1.8"))
        self.bert_model.transformer_layer_spec = bert_layer_with_transformer_engine_spec
        attn_mask_dimensions = self.bert_model._sanity_check_attention_and_get_attn_mask_dimension()
        assert (
            attn_mask_dimensions == "b11s"
        ), f"Expected b11s for attn_mask_dimensions but got {attn_mask_dimensions}"

    @pytest.mark.internal
    @pytest.mark.flaky_in_dev
    def test_transformer_engine_version_1_7_to_1_10_rng_error(self, mocker):
        bert_layer_with_transformer_engine_spec.submodules.self_attention.params[
            'attn_mask_type'
        ] == AttnMaskType.padding
        mocker.patch("megatron.core.utils.get_te_version", return_value=PkgVersion("1.8"))
        with pytest.raises(Exception) as exc_info:
            self.bert_model = BertModel(
                config=self.transformer_config,
                num_tokentypes=0,
                transformer_layer_spec=bert_layer_with_transformer_engine_spec,
                vocab_size=100,
                max_sequence_length=4,
            )
        assert str(exc_info.value) == (
            "Linear.__init__() got an unexpected keyword argument 'rng_tracker_name' when "
            "instantiating TERowParallelLinear when instantiating SelfAttention when "
            "instantiating TransformerLayer"
        )

    @pytest.mark.internal
    def test_transformer_engine_version_1_7_to_1_10_unfused_attention(self, mocker):
        self.bert_model.config.attention_backend = AttnBackend.unfused
        bert_layer_with_transformer_engine_spec.submodules.self_attention.params[
            'attn_mask_type'
        ] == AttnMaskType.padding
        mocker.patch("megatron.core.utils.get_te_version", return_value=PkgVersion("1.8"))
        self.bert_model.transformer_layer_spec = bert_layer_with_transformer_engine_spec
        attn_mask_dimensions = self.bert_model._sanity_check_attention_and_get_attn_mask_dimension()
        attn_mask_type = self.bert_model.transformer_layer_spec.submodules.self_attention.params[
            'attn_mask_type'
        ]
        assert (
            attn_mask_type == AttnMaskType.arbitrary
        ), f"Exepcted attn mask type to be arbitrary, but got {attn_mask_type}"
        assert (
            attn_mask_dimensions == "b1ss"
        ), f"Expected b1ss for attn_mask_dimensions but got {attn_mask_dimensions}"

    @pytest.mark.internal
    def test_transformer_engine_version_less_than_1_7(self, mocker):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        self.bert_model.config.attention_backend = AttnBackend.flash
        with pytest.raises(Exception) as exc_info:
            mocker.patch("megatron.core.utils.get_te_version", return_value=PkgVersion("1.5"))
            self.bert_model = BertModel(
                config=self.transformer_config,
                num_tokentypes=0,
                transformer_layer_spec=bert_layer_with_transformer_engine_spec,
                vocab_size=100,
                max_sequence_length=4,
            )

        assert str(exc_info.value) == (
            "Flash and fused attention is not supported with transformer engine version "
            "< 1.7. Set --attention-backend to unfused or leave it to be default (auto) or upgrade transformer engine >= 1.7"
        )
