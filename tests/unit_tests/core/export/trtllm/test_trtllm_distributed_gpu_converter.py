import torch
from pytest_mock import mocker

from megatron.core.export.data_type import DataType
from megatron.core.export.trtllm.model_to_trllm_mapping.default_conversion_dict import (
    DEFAULT_CONVERSION_DICT,
)

# pylint: disable=line-too-long
from megatron.core.export.trtllm.trtllm_weights_converter.distributed_trtllm_model_weights_converter import (
    DistributedTRTLLMModelWeightsConverter,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_SEQUENCE_LENGTH = 64
_VOCAB_SIZE = 256


class TestTRTLLMDistributedGPUConverter:
    """
    Test Distributed converter
    """

    def setup_method(self, method):
        """
        Setup method
        """
        Utils.initialize_model_parallel(2, 1)
        model_parallel_cuda_manual_seed(123)

        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=2,
            use_cpu_initialization=True,
            pipeline_dtype=torch.float32,
            add_qkv_bias=False,
            add_bias_linear=False,
        )
        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=_VOCAB_SIZE,
            max_sequence_length=_SEQUENCE_LENGTH,
        )

    def teardown_method(self, method):
        """
        teardown method
        """
        Utils.destroy_model_parallel()

    def test_get_model_weights_converter(self, mocker):
        """
        test model weights onverter
        """
        device = torch.device("cuda")
        self.gpt_model.to(device)

        transformer_config = self.gpt_model.config

        mocker.patch(
            "megatron.core.export.trtllm.trtllm_weights_converter.distributed_trtllm_model_weights_converter.str_dtype_to_torch",
            return_value=torch.float32,
        )

        dtype = DataType.bfloat16
        distributed_converter = DistributedTRTLLMModelWeightsConverter(
            transformer_config, dtype, activation="gelu"
        )

        model_state_dict = {}
        for key, val in self.gpt_model.state_dict().items():
            # val is non for _extra_state layers . We filter it out
            if val is not None:
                model_state_dict[key] = val

        distributed_converter.convert(
            model_state_dict=model_state_dict,
            trtllm_conversion_dict=DEFAULT_CONVERSION_DICT,
            tokenizer_vocab_size=_VOCAB_SIZE,
        )

        expected_result = {
            'transformer.vocab_embedding.weight': torch.Size([128, 64]),
            'transformer.position_embedding.weight': torch.Size([32, 64]),
            'lm_head.weight': torch.Size([128, 64]),
            'transformer.ln_f.weight': torch.Size([64]),
            'transformer.ln_f.bias': torch.Size([64]),
            'transformer.layers.0.input_layernorm.weight': torch.Size([64]),
            'transformer.layers.0.input_layernorm.bias': torch.Size([64]),
            'transformer.layers.0.attention.dense.weight': torch.Size([64, 32]),
            'transformer.layers.0.attention.qkv.weight': torch.Size([96, 64]),
            'transformer.layers.0.post_layernorm.weight': torch.Size([64]),
            'transformer.layers.0.post_layernorm.bias': torch.Size([64]),
            'transformer.layers.0.mlp.fc.weight': torch.Size([128, 64]),
            'transformer.layers.0.mlp.proj.weight': torch.Size([64, 128]),
            'transformer.layers.1.input_layernorm.weight': torch.Size([64]),
            'transformer.layers.1.input_layernorm.bias': torch.Size([64]),
            'transformer.layers.1.attention.dense.weight': torch.Size([64, 32]),
            'transformer.layers.1.attention.qkv.weight': torch.Size([96, 64]),
            'transformer.layers.1.post_layernorm.weight': torch.Size([64]),
            'transformer.layers.1.post_layernorm.bias': torch.Size([64]),
            'transformer.layers.1.mlp.fc.weight': torch.Size([128, 64]),
            'transformer.layers.1.mlp.proj.weight': torch.Size([64, 128]),
        }

        for key, value in distributed_converter.trtllm_model_weights.items():
            assert (
                expected_result[key] == value.shape
            ), f"Shape mismatch for {key}. Expected {expected_result[key]} but got {value.shape}"
