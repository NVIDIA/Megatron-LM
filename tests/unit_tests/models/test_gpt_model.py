# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import inspect
import os

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_mlp_module_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


class TestGPTModel:

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.gpt_model, GPTModel)

        assert self.gpt_model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in self.gpt_model.parameters()])
        assert num_weights == 6240

    @pytest.mark.internal
    def test_set_input_tensor(self):
        config: TransformerConfig = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        self.gpt_model.set_input_tensor(input_tensor)

        assert self.gpt_model.decoder.input_tensor.shape[0] == sequence_length
        assert self.gpt_model.decoder.input_tensor.shape[1] == micro_batch_size
        assert self.gpt_model.decoder.input_tensor.shape[2] == config.hidden_size

    @pytest.mark.internal
    def test_post_process_forward(self):
        _ = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        self.gpt_model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = self.gpt_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.gpt_model.vocab_size


def test_get_mlp_module_spec_interface():
    # Get the function signature
    sig = inspect.signature(get_mlp_module_spec)

    # Define the expected signature
    expected_params = {
        "use_te": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "num_experts": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "moe_grouped_gemm": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "fp8": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "moe_use_legacy_grouped_gemm": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "use_te_op_fuser": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }

    expected_defaults = {
        "use_te": True,
        "num_experts": None,
        "moe_grouped_gemm": False,
        "fp8": None,
        "moe_use_legacy_grouped_gemm": False,
        "use_te_op_fuser": False,
    }

    # Check expected parameters are in function signature
    for param_name, param_kind in expected_params.items():
        assert param_name in sig.parameters, f"Unexpected parameter: {param_name}"
        assert (
            param_kind is sig.parameters[param_name].kind
        ), f"Wrong kind for parameter: {param_name}"

    # Check default values
    sig_defaults = {
        k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty
    }
    for k, v in expected_defaults.items():
        assert (
            k in sig_defaults and v == sig_defaults[k]
        ), f"Default value of {sig_defaults[k]} does not match the expected value of {v} for parameter {k}."


@pytest.mark.skipif(
    not is_te_min_version("1.13.0"), reason="TEFusedMLP is only supported with TE 1.13+."
)
class TestGPTWithFusedOps:
    """GPT model with Transformer Engine operation-based API"""

    def setup_method(self, method) -> None:
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(use_te_op_fuser=True),
            vocab_size=100,
            max_sequence_length=4,
        )

    def teardown_method(self, method) -> None:
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_forward(self) -> None:
        _ = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        self.gpt_model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = self.gpt_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.gpt_model.vocab_size
