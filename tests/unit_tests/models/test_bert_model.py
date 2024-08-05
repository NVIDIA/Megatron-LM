# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch
import os
from pkg_resources import packaging
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.bert.bert_model import BertModel
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.models.bert.bert_layer_specs import bert_layer_with_transformer_engine_spec
from pytest_mock import mocker 

class TestBertModel:

    def setup_method(self, method):
        os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0' #Bert does not support flash attention
        tp = 1
        pp = 1
        Utils.initialize_model_parallel(tp, pp)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4,
            use_cpu_initialization=True, perform_initialization=True,
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp, pipeline_dtype=torch.bfloat16
        )
        self.bert_model = BertModel(
            config=transformer_config, num_tokentypes=0,
            transformer_layer_spec=bert_layer_with_transformer_engine_spec, vocab_size=100, max_sequence_length=4
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.bert_model, BertModel)

        assert self.bert_model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in self.bert_model.parameters()])
        assert num_weights == 6702

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


class TestBertModelAssertions:

    def test_te_assertions_te_less_than_1_7(self, mocker):
        os.environ.pop('NVTE_ALLOW_NONDETERMINISTIC_ALGO',None)
        os.environ.pop('NVTE_FLASH_ATTN',None)
        os.environ.pop('NVTE_FUSED_ATTN',None)
        tp = 1
        pp = 1
        Utils.initialize_model_parallel(tp, pp)        
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4,
            use_cpu_initialization=True, perform_initialization=True,
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp, pipeline_dtype=torch.bfloat16
        )

        with pytest.raises(Exception) as exc_info:
            mocker.patch("megatron.core.models.bert.bert_model.get_te_version", return_value = packaging.version.Version("1.4"))
            self.bert_model = BertModel(
                config=transformer_config, num_tokentypes=0,
                transformer_layer_spec=bert_layer_with_transformer_engine_spec, vocab_size=100, max_sequence_length=4
            )
        assert str(exc_info.value) == "Flash and fused attention is not supported with transformer engine version < 1.7. Set NVTE_FLASH_ATTN=0 and NVTE_FUSED_ATTN=0 or upgrade transformer engine >= 1.7 or set NVTE_ALLOW_NONDETERMINISTIC_ALGO=0"

    def test_te_assertions_te_equal_to_1_7_exception(self, mocker):
        os.environ.pop('NVTE_ALLOW_NONDETERMINISTIC_ALGO',None)
        os.environ['NVTE_FLASH_ATTN'] = '0'
        os.environ['NVTE_FUSED_ATTN'] = '0'
        tp = 1
        pp = 1
        Utils.initialize_model_parallel(tp, pp)        
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4,
            use_cpu_initialization=True, perform_initialization=True,
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp, pipeline_dtype=torch.bfloat16
        )

        with pytest.raises(Exception) as exc_info:
            mocker.patch("megatron.core.models.bert.bert_model.get_te_version", return_value = packaging.version.Version("1.7"))
            self.bert_model = BertModel(
                config=transformer_config, num_tokentypes=0,
                transformer_layer_spec=bert_layer_with_transformer_engine_spec, vocab_size=100, max_sequence_length=4
            )
        assert str(exc_info.value) == "Set env variable NVTE_FLASH_ATTN to 1 or NVTE_FUSED_ATTN to 1 to use a more optimized attention kernal. Currently using unfused attention path. If you want to proceed with this path set AttnMaskType in module spec to be arbitrary"

    def test_te_assertions_te_equal_to_1_7_no_exception(self, mocker):
        os.environ.pop('NVTE_ALLOW_NONDETERMINISTIC_ALGO',None)
        os.environ.pop('NVTE_FLASH_ATTN',None)
        os.environ.pop('NVTE_FUSED_ATTN',None)
        tp = 1
        pp = 1
        Utils.initialize_model_parallel(tp, pp)        
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4,
            use_cpu_initialization=True, perform_initialization=True,
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp, pipeline_dtype=torch.bfloat16
        )

        mocker.patch("megatron.core.models.bert.bert_model.get_te_version", return_value = packaging.version.Version("1.7"))
        self.bert_model = BertModel(
            config=transformer_config, num_tokentypes=0,
            transformer_layer_spec=bert_layer_with_transformer_engine_spec, vocab_size=100, max_sequence_length=4
        )
        Utils.destroy_model_parallel()