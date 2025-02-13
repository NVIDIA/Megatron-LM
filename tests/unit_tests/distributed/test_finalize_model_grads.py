# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import inspect
import os

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed.finalize_model_grads import _allreduce_layernorm_grads
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestAllReduceLNGrads:

    def init_model(self):
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            tensor_model_parallel_size=self.tp_size,
            qk_layernorm=True,
        )

        self.model = GPTModel(
            config=self.transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
            vocab_size=100,
            max_sequence_length=4,
        )

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("freeze_model,tp_size", [(True, 2), (False, 2)])
    def test_allreduce_layernorm_grads(self, freeze_model, tp_size):

        self.tp_size = tp_size
        Utils.initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        model_parallel_cuda_manual_seed(123)

        self.init_model()
        self.model.cuda()

        for param in self.model.parameters():
            if freeze_model:
                param.requires_grad = False
            else:
                param.grad = torch.ones_like(param)

        _allreduce_layernorm_grads([self.model], self.transformer_config)
