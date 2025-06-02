# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import inspect
import os

import pytest
import torch

from megatron.core.device_utils import get_current_device
import megatron.core.transformer.utils as transformer_utils
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec, get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_device_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import set_model_to_sequence_parallel
from tests.unit_tests.test_utilities import Utils

try:
    import transformer_engine  # pylint: disable=unused-import
    HAVE_TE =True
except ImportError:
    HAVE_TE = False

class TestGPTModel:

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

        self.tensor_model_parallel_size = 2
        Utils.initialize_model_parallel(self.tensor_model_parallel_size, 1)
        model_parallel_device_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=48,
            num_attention_heads=4,
            use_cpu_initialization=True,
            tensor_model_parallel_size=self.tensor_model_parallel_size,
            sequence_parallel=False,
        )
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec() if HAVE_TE else get_gpt_layer_local_spec()
        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=100,
            max_sequence_length=8,
            position_embedding_type="rope",
            parallel_output=False,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_post_process_forward(self):
        _ = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        self.gpt_model.to(get_current_device())

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(get_current_device())
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(get_current_device())
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).to(get_current_device())

        logits = self.gpt_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.gpt_model.vocab_size

        set_model_to_sequence_parallel(self.gpt_model, set_to=True)
        logits = self.gpt_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        # Test cache has been built
        assert transformer_utils._sequence_parallel_attr_cache is not None

        # Check the modules have been flipped
        for attribute, modules in transformer_utils._sequence_parallel_attr_cache[
            id(self.gpt_model)
        ].items():
            for module in modules:
                assert getattr(module, attribute) == True

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.gpt_model.vocab_size

        set_model_to_sequence_parallel(self.gpt_model, set_to=False)
        logits = self.gpt_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.gpt_model.vocab_size

        # Check the modules have been flipped
        for attribute, modules in transformer_utils._sequence_parallel_attr_cache[
            id(self.gpt_model)
        ].items():
            for module in modules:
                assert getattr(module, attribute) == False
