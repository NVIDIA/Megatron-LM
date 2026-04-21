import torch
from pytest_mock import mocker

from megatron.core.export.data_type import DataType
from megatron.core.export.export_config import ExportConfig
from megatron.core.export.trtllm.trtllm_layers import TRTLLMLayers
from megatron.core.export.trtllm.trtllm_weights_converter.single_device_trtllm_model_weights_converter import (
    SingleDeviceTRTLLMModelWeightsConverter,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class TestTRTLLMSingleDeviceConverter:
    def test_get_model_weights_converter(self, mocker):

        export_config = ExportConfig(inference_tp_size=2)

        vocab_size = 10
        hidden_dim = 4
        seq_len = 8
        num_layers = 2
        num_attn_heads = 2

        model_config = TransformerConfig(
            num_layers=num_layers,
            num_attention_heads=num_attn_heads,
            num_query_groups=0,
            hidden_size=hidden_dim,
            ffn_hidden_size=hidden_dim * 4,
        )

        dtype = DataType.bfloat16

        model_state_dict = {
            "decoder.position_embedding.weight": torch.randn(seq_len, hidden_dim),
            "decoder.word_embedding.weight": torch.randn(vocab_size, hidden_dim),
            "decoder.lm_head.weight": torch.randn(vocab_size, hidden_dim),
            "decoder.final_layernorm.weight": torch.randn(hidden_dim),
            "decoder.layers.input_layernorm.weight": torch.randn(num_layers, hidden_dim),
            "decoder.layers.attention.qkv.weight": torch.randn(
                num_layers, hidden_dim * 3, hidden_dim
            ),
            "decoder.layers.attention.qkv.bias": torch.randn(num_layers, hidden_dim * 3),
            "decoder.layers.attention.dense.weight": torch.randn(
                num_layers, hidden_dim, hidden_dim
            ),
            "deocder.layers.mlp.fc.weight": torch.randn(num_layers, 4 * hidden_dim, hidden_dim),
            "decoder.layers.mlp.fc.expert": torch.randn(num_layers, hidden_dim, hidden_dim * 4),
            "decoder.layers.mlp.proj.expert": torch.randn(num_layers, hidden_dim * 4, hidden_dim),
        }

        trtllm_conversion_dict = {
            "decoder.position_embedding.weight": TRTLLMLayers.position_embedding,
            "decoder.word_embedding.weight": TRTLLMLayers.vocab_embedding,
            "decoder.final_layernorm.weight": TRTLLMLayers.final_layernorm_weight,
            "decoder.lm_head.weight": TRTLLMLayers.lm_head,
            "decoder.layers.input_layernorm.weight": TRTLLMLayers.input_layernorm_weight,
            "decoder.layers.attention.qkv.weight": TRTLLMLayers.attention_qkv_weight,
            "decoder.layers.attention.qkv.bias": TRTLLMLayers.attention_qkv_bias,
            "decoder.layers.attention.dense.weight": TRTLLMLayers.attention_dense_weight,
            "deocder.layers.mlp.fc.weight": TRTLLMLayers.mlp_fc_weight,
            "decoder.layers.mlp.fc.expert": TRTLLMLayers.mlp_fc_weight_mixture_of_experts,
            "decoder.layers.mlp.proj.expert": TRTLLMLayers.mlp_projection_weight_mixture_of_experts,
        }

        mocker.patch(
            "megatron.core.export.trtllm.trtllm_weights_converter.single_device_trtllm_model_weights_converter.str_dtype_to_torch",
            return_value=torch.float32,
        )

        trtllm_model_weights_converter_cpu = SingleDeviceTRTLLMModelWeightsConverter(
            export_config, model_config, dtype, activation="swiglu"
        )

        mocker.patch(
            "megatron.core.export.trtllm.trtllm_weights_converter.single_device_trtllm_model_weights_converter.pad_vocab_size",
            return_value=10,
        )

        trtllm_model_weights_converter_cpu.convert(
            model_state_dict=model_state_dict,
            trtllm_conversion_dict=trtllm_conversion_dict,
            state_dict_split_by_layer_numbers=False,
        )

        expected_shapes = {
            'transformer.vocab_embedding.weight': (10, 4),
            'transformer.position_embedding.weight': (8, 4),
            'lm_head.weight': (10, 4),
            'transformer.ln_f.weight': (4,),
            'transformer.layers.0.input_layernorm.weight': (4,),
            'transformer.layers.1.input_layernorm.weight': (4,),
            'transformer.layers.0.attention.qkv.weight.0.bin': (6, 4),
            'transformer.layers.0.attention.qkv.weight.1.bin': (6, 4),
            'transformer.layers.1.attention.qkv.weight.0.bin': (6, 4),
            'transformer.layers.1.attention.qkv.weight.1.bin': (6, 4),
            'transformer.layers.0.attention.qkv.bias.0.bin': (6,),
            'transformer.layers.0.attention.qkv.bias.1.bin': (6,),
            'transformer.layers.1.attention.qkv.bias.0.bin': (6,),
            'transformer.layers.1.attention.qkv.bias.1.bin': (6,),
            'transformer.layers.0.attention.dense.weight.0.bin': (4, 2),
            'transformer.layers.0.attention.dense.weight.1.bin': (4, 2),
            'transformer.layers.1.attention.dense.weight.0.bin': (4, 2),
            'transformer.layers.1.attention.dense.weight.1.bin': (4, 2),
            'transformer.layers.0.mlp.gate.weight.0.bin': (4, 4),
            'transformer.layers.0.mlp.gate.weight.1.bin': (4, 4),
            'transformer.layers.0.mlp.fc.weight.0.bin': (16, 2),
            'transformer.layers.0.mlp.fc.weight.1.bin': (16, 2),
            'transformer.layers.1.mlp.gate.weight.0.bin': (4, 4),
            'transformer.layers.1.mlp.gate.weight.1.bin': (4, 4),
            'transformer.layers.1.mlp.fc.weight.0.bin': (16, 2),
            'transformer.layers.1.mlp.fc.weight.1.bin': (16, 2),
            'transformer.layers.0.mlp.proj.weight.0.bin': (4, 8),
            'transformer.layers.0.mlp.proj.weight.1.bin': (4, 8),
            'transformer.layers.1.mlp.proj.weight.0.bin': (4, 8),
            'transformer.layers.1.mlp.proj.weight.1.bin': (4, 8),
        }

        for key, value in trtllm_model_weights_converter_cpu.trtllm_model_weights.items():
            assert (
                expected_shapes[key] == value.shape
            ), f"Shape mismatch for {key}. Expected {expected_shapes[key]} but got {value.shape}"

        class SampleMapping:

            def __init__(self):
                self.tp_size = 2
                self.tp_rank = 1

            def pp_layers(self, num_layers):
                return [0, 1]

            def is_first_pp_rank(self):
                return True

            def is_last_pp_rank(self):
                return True

        trtllm_model_weights_per_gpu = (
            trtllm_model_weights_converter_cpu.get_local_model_weights_per_gpu(
                mapping=SampleMapping(), trtllm_model_config=None
            )
        )

        expected_result_per_gpu = {
            'transformer.layers.0.input_layernorm.weight': (4,),
            'transformer.layers.1.input_layernorm.weight': (4,),
            'transformer.layers.0.attention.qkv.weight': (6, 4),
            'transformer.layers.1.attention.qkv.weight': (6, 4),
            'transformer.layers.0.attention.qkv.bias': (6,),
            'transformer.layers.1.attention.qkv.bias': (6,),
            'transformer.layers.0.attention.dense.weight': (4, 2),
            'transformer.layers.1.attention.dense.weight': (4, 2),
            'transformer.layers.0.mlp.gate.weight': (4, 4),
            'transformer.layers.0.mlp.fc.weight': (16, 2),
            'transformer.layers.1.mlp.gate.weight': (4, 4),
            'transformer.layers.1.mlp.fc.weight': (16, 2),
            'transformer.layers.0.mlp.proj.weight': (4, 8),
            'transformer.layers.1.mlp.proj.weight': (4, 8),
            'transformer.vocab_embedding.weight': (10, 4),
            'transformer.position_embedding.weight': (8, 4),
            'lm_head.weight': (5, 4),
            'transformer.ln_f.weight': (4,),
        }

        for key, value in trtllm_model_weights_per_gpu.items():
            assert (
                expected_result_per_gpu[key] == value.shape
            ), f"Shape mismatch for {key}. Expected {expected_result_per_gpu[key]} but got {value.shape}"
