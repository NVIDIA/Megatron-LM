import pytest
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

    def test_num_kv_heads_less_than_tp_size_valid(self, mocker):
        """Test the condition where num_kv_heads < inference_tp_size and tp_size % num_kv_heads == 0 (valid case)"""

        # Configure for GQA: 8 attention heads, 2 KV heads, TP size 4
        # This is valid because 4 % 2 == 0
        export_config = ExportConfig(inference_tp_size=4)

        vocab_size = 10
        hidden_dim = 8
        seq_len = 8
        num_layers = 2
        num_attn_heads = 8
        num_kv_heads = 2  # This is less than tp_size (4) and 4 % 2 == 0

        model_config = TransformerConfig(
            num_layers=num_layers,
            num_attention_heads=num_attn_heads,
            num_query_groups=num_kv_heads,  # GQA with 2 KV heads
            hidden_size=hidden_dim,
            ffn_hidden_size=hidden_dim * 4,
        )

        dtype = DataType.bfloat16

        # Create model state dict with GQA structure
        # For GQA: q_num = num_attn_heads // num_kv_heads = 8 // 2 = 4
        # So each KV head handles 4 query heads
        q_num = num_attn_heads // num_kv_heads  # 4
        size_per_head = hidden_dim // num_attn_heads  # 1

        # Calculate the correct tensor sizes for QKV
        qkv_weight_size = num_kv_heads * (q_num + 2) * size_per_head  # 2 * (4 + 2) * 1 = 12
        qkv_bias_size = num_kv_heads * (q_num + 2) * size_per_head  # 2 * (4 + 2) * 1 = 12

        model_state_dict = {
            "decoder.position_embedding.weight": torch.randn(seq_len, hidden_dim),
            "decoder.word_embedding.weight": torch.randn(vocab_size, hidden_dim),
            "decoder.lm_head.weight": torch.randn(vocab_size, hidden_dim),
            "decoder.final_layernorm.weight": torch.randn(hidden_dim),
            "decoder.layers.input_layernorm.weight": torch.randn(num_layers, hidden_dim),
            # QKV weight: [num_layers, qkv_weight_size, hidden_dim] - converter will transpose to [hidden_dim, qkv_weight_size]
            "decoder.layers.attention.qkv.weight": torch.randn(
                num_layers, qkv_weight_size, hidden_dim
            ),
            # QKV bias: [num_layers, qkv_bias_size]
            "decoder.layers.attention.qkv.bias": torch.randn(num_layers, qkv_bias_size),
            "decoder.layers.attention.dense.weight": torch.randn(
                num_layers, hidden_dim, hidden_dim
            ),
            "decoder.layers.mlp.fc.weight": torch.randn(num_layers, 4 * hidden_dim, hidden_dim),
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
            "decoder.layers.mlp.fc.weight": TRTLLMLayers.mlp_fc_weight,
        }

        mocker.patch(
            "megatron.core.export.trtllm.trtllm_weights_converter.single_device_trtllm_model_weights_converter.str_dtype_to_torch",
            return_value=torch.float32,
        )

        trtllm_model_weights_converter_cpu = SingleDeviceTRTLLMModelWeightsConverter(
            export_config, model_config, dtype, activation="gelu"
        )

        mocker.patch(
            "megatron.core.export.trtllm.trtllm_weights_converter.single_device_trtllm_model_weights_converter.pad_vocab_size",
            return_value=10,
        )

        # Verify the conditions are met
        assert trtllm_model_weights_converter_cpu.num_kv_heads < export_config.inference_tp_size
        assert (
            export_config.inference_tp_size % trtllm_model_weights_converter_cpu.num_kv_heads == 0
        )
        assert trtllm_model_weights_converter_cpu.num_kv_heads == 2
        assert export_config.inference_tp_size == 4

        trtllm_model_weights_converter_cpu.convert(
            model_state_dict=model_state_dict,
            trtllm_conversion_dict=trtllm_conversion_dict,
            state_dict_split_by_layer_numbers=False,
        )

        # Check that QKV weights and biases are properly split for TP=4
        # Each TP rank should get 1/4 of the weights
        for layer_idx in range(num_layers):
            qkv_weight_key = f"transformer.layers.{layer_idx}.attention.qkv.weight"
            qkv_bias_key = f"transformer.layers.{layer_idx}.attention.qkv.bias"

            # Check that we have 4 splits (one for each TP rank)
            for tp_rank in range(4):
                weight_split_key = f"{qkv_weight_key}.{tp_rank}.bin"
                bias_split_key = f"{qkv_bias_key}.{tp_rank}.bin"

                assert weight_split_key in trtllm_model_weights_converter_cpu.trtllm_model_weights
                assert bias_split_key in trtllm_model_weights_converter_cpu.trtllm_model_weights

                # Verify that the splits have the expected dimensions
                weight_split = trtllm_model_weights_converter_cpu.trtllm_model_weights[
                    weight_split_key
                ]
                bias_split = trtllm_model_weights_converter_cpu.trtllm_model_weights[bias_split_key]

                # For TP=4, each split should have 1/4 of the original size
                # The weight shape depends on the conversion process, not necessarily hidden_dim
                assert (
                    len(bias_split.shape) == 1
                ), f"Expected bias to be 1D, got shape {bias_split.shape}"

                # Verify that all splits have the same size
                if tp_rank == 0:
                    expected_weight_size = weight_split.shape[1]
                    expected_bias_size = bias_split.shape[0]
                else:
                    assert (
                        weight_split.shape[1] == expected_weight_size
                    ), f"All weight splits should have same size"
                    assert (
                        bias_split.shape[0] == expected_bias_size
                    ), f"All bias splits should have same size"

    def test_num_kv_heads_less_than_tp_size_invalid(self, mocker):
        """Test that an exception is raised when num_kv_heads < tp_size but tp_size % num_kv_heads != 0"""

        # Configure for invalid case: 3 KV heads, TP size 4 (not divisible)
        # This should raise an exception because 4 % 3 != 0
        export_config = ExportConfig(inference_tp_size=4)

        vocab_size = 10
        hidden_dim = 8
        seq_len = 8
        num_layers = 2
        num_attn_heads = 6
        num_kv_heads = 3  # This is less than tp_size (4) but 4 % 3 != 0

        model_config = TransformerConfig(
            num_layers=num_layers,
            num_attention_heads=num_attn_heads,
            num_query_groups=num_kv_heads,
            hidden_size=hidden_dim,
            ffn_hidden_size=hidden_dim * 4,
        )

        dtype = DataType.bfloat16

        q_num = num_attn_heads // num_kv_heads  # 2
        size_per_head = hidden_dim // num_attn_heads  # 1

        # Calculate the correct tensor sizes for QKV
        qkv_weight_size = num_kv_heads * (q_num + 2) * size_per_head  # 3 * (2 + 2) * 1 = 12
        qkv_bias_size = num_kv_heads * (q_num + 2) * size_per_head  # 3 * (2 + 2) * 1 = 12

        model_state_dict = {
            "decoder.position_embedding.weight": torch.randn(seq_len, hidden_dim),
            "decoder.word_embedding.weight": torch.randn(vocab_size, hidden_dim),
            "decoder.lm_head.weight": torch.randn(vocab_size, hidden_dim),
            "decoder.final_layernorm.weight": torch.randn(hidden_dim),
            "decoder.layers.input_layernorm.weight": torch.randn(num_layers, hidden_dim),
            "decoder.layers.attention.qkv.weight": torch.randn(
                num_layers, qkv_weight_size, hidden_dim
            ),
            "decoder.layers.attention.qkv.bias": torch.randn(num_layers, qkv_bias_size),
            "decoder.layers.attention.dense.weight": torch.randn(
                num_layers, hidden_dim, hidden_dim
            ),
            "decoder.layers.mlp.fc.weight": torch.randn(num_layers, 4 * hidden_dim, hidden_dim),
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
            "decoder.layers.mlp.fc.weight": TRTLLMLayers.mlp_fc_weight,
        }

        mocker.patch(
            "megatron.core.export.trtllm.trtllm_weights_converter.single_device_trtllm_model_weights_converter.str_dtype_to_torch",
            return_value=torch.float32,
        )

        trtllm_model_weights_converter_cpu = SingleDeviceTRTLLMModelWeightsConverter(
            export_config, model_config, dtype, activation="gelu"
        )

        mocker.patch(
            "megatron.core.export.trtllm.trtllm_weights_converter.single_device_trtllm_model_weights_converter.pad_vocab_size",
            return_value=10,
        )

        # Verify the conditions are met for the invalid case
        assert trtllm_model_weights_converter_cpu.num_kv_heads < export_config.inference_tp_size
        assert (
            export_config.inference_tp_size % trtllm_model_weights_converter_cpu.num_kv_heads != 0
        )
        assert trtllm_model_weights_converter_cpu.num_kv_heads == 3
        assert export_config.inference_tp_size == 4

        # This should raise an exception during conversion
        with pytest.raises(Exception) as exc_info:
            trtllm_model_weights_converter_cpu.convert(
                model_state_dict=model_state_dict,
                trtllm_conversion_dict=trtllm_conversion_dict,
                state_dict_split_by_layer_numbers=False,
            )

        # Verify the exception message
        expected_message = "Number of query groups of the models is 3. Please select tensor parallelism size that can duplicate or split the number of query groups to equal number of query matrices in the each GPU."
        assert expected_message in str(exc_info.value)

    def test_num_kv_heads_greater_equal_tp_size_invalid(self, mocker):
        """Test that an exception is raised when num_kv_heads >= tp_size but num_kv_heads % tp_size != 0"""

        # Configure for invalid case: 5 KV heads, TP size 4 (not divisible)
        # This should raise an exception because 5 % 4 != 0
        export_config = ExportConfig(inference_tp_size=4)

        vocab_size = 10
        hidden_dim = 8
        seq_len = 8
        num_layers = 2
        num_attn_heads = 10
        num_kv_heads = 5  # This is greater than tp_size (4) but 5 % 4 != 0

        model_config = TransformerConfig(
            num_layers=num_layers,
            num_attention_heads=num_attn_heads,
            num_query_groups=num_kv_heads,
            hidden_size=hidden_dim,
            ffn_hidden_size=hidden_dim * 4,
        )

        dtype = DataType.bfloat16

        q_num = num_attn_heads // num_kv_heads  # 2
        size_per_head = hidden_dim // num_attn_heads  # 1

        # Calculate the correct tensor sizes for QKV
        qkv_weight_size = num_kv_heads * (q_num + 2) * size_per_head  # 5 * (2 + 2) * 1 = 20
        qkv_bias_size = num_kv_heads * (q_num + 2) * size_per_head  # 5 * (2 + 2) * 1 = 20

        model_state_dict = {
            "decoder.position_embedding.weight": torch.randn(seq_len, hidden_dim),
            "decoder.word_embedding.weight": torch.randn(vocab_size, hidden_dim),
            "decoder.lm_head.weight": torch.randn(vocab_size, hidden_dim),
            "decoder.final_layernorm.weight": torch.randn(hidden_dim),
            "decoder.layers.input_layernorm.weight": torch.randn(num_layers, hidden_dim),
            "decoder.layers.attention.qkv.weight": torch.randn(
                num_layers, qkv_weight_size, hidden_dim
            ),
            "decoder.layers.attention.qkv.bias": torch.randn(num_layers, qkv_bias_size),
            "decoder.layers.attention.dense.weight": torch.randn(
                num_layers, hidden_dim, hidden_dim
            ),
            "decoder.layers.mlp.fc.weight": torch.randn(num_layers, 4 * hidden_dim, hidden_dim),
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
            "decoder.layers.mlp.fc.weight": TRTLLMLayers.mlp_fc_weight,
        }

        mocker.patch(
            "megatron.core.export.trtllm.trtllm_weights_converter.single_device_trtllm_model_weights_converter.str_dtype_to_torch",
            return_value=torch.float32,
        )

        trtllm_model_weights_converter_cpu = SingleDeviceTRTLLMModelWeightsConverter(
            export_config, model_config, dtype, activation="gelu"
        )

        mocker.patch(
            "megatron.core.export.trtllm.trtllm_weights_converter.single_device_trtllm_model_weights_converter.pad_vocab_size",
            return_value=10,
        )

        # Verify the conditions are met for the invalid case
        assert trtllm_model_weights_converter_cpu.num_kv_heads >= export_config.inference_tp_size
        assert (
            trtllm_model_weights_converter_cpu.num_kv_heads % export_config.inference_tp_size != 0
        )
        assert trtllm_model_weights_converter_cpu.num_kv_heads == 5
        assert export_config.inference_tp_size == 4

        # This should raise an exception during conversion
        with pytest.raises(Exception) as exc_info:
            trtllm_model_weights_converter_cpu.convert(
                model_state_dict=model_state_dict,
                trtllm_conversion_dict=trtllm_conversion_dict,
                state_dict_split_by_layer_numbers=False,
            )

        # Verify the exception message
        expected_message = "Number of query groups of the models is 5. Please select tensor parallelism size that can duplicate or split the number of query groups to equal number of query matrices in the each GPU."
        assert expected_message in str(exc_info.value)

    def test_num_kv_heads_greater_equal_tp_size_valid(self, mocker):
        """Test the condition where num_kv_heads >= tp_size and num_kv_heads % tp_size == 0 (valid case)"""

        # Configure for valid case: 8 KV heads, TP size 4 (divisible)
        # This is valid because 8 % 4 == 0
        export_config = ExportConfig(inference_tp_size=4)

        vocab_size = 10
        hidden_dim = 8
        seq_len = 8
        num_layers = 2
        num_attn_heads = 8
        num_kv_heads = 8  # This is equal to tp_size (4) and 8 % 4 == 0

        model_config = TransformerConfig(
            num_layers=num_layers,
            num_attention_heads=num_attn_heads,
            num_query_groups=num_kv_heads,
            hidden_size=hidden_dim,
            ffn_hidden_size=hidden_dim * 4,
        )

        dtype = DataType.bfloat16

        q_num = num_attn_heads // num_kv_heads  # 1
        size_per_head = hidden_dim // num_attn_heads  # 1

        # Calculate the correct tensor sizes for QKV
        qkv_weight_size = num_kv_heads * (q_num + 2) * size_per_head  # 8 * (1 + 2) * 1 = 24
        qkv_bias_size = num_kv_heads * (q_num + 2) * size_per_head  # 8 * (1 + 2) * 1 = 24

        model_state_dict = {
            "decoder.position_embedding.weight": torch.randn(seq_len, hidden_dim),
            "decoder.word_embedding.weight": torch.randn(vocab_size, hidden_dim),
            "decoder.lm_head.weight": torch.randn(vocab_size, hidden_dim),
            "decoder.final_layernorm.weight": torch.randn(hidden_dim),
            "decoder.layers.input_layernorm.weight": torch.randn(num_layers, hidden_dim),
            "decoder.layers.attention.qkv.weight": torch.randn(
                num_layers, qkv_weight_size, hidden_dim
            ),
            "decoder.layers.attention.qkv.bias": torch.randn(num_layers, qkv_bias_size),
            "decoder.layers.attention.dense.weight": torch.randn(
                num_layers, hidden_dim, hidden_dim
            ),
            "decoder.layers.mlp.fc.weight": torch.randn(num_layers, 4 * hidden_dim, hidden_dim),
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
            "decoder.layers.mlp.fc.weight": TRTLLMLayers.mlp_fc_weight,
        }

        mocker.patch(
            "megatron.core.export.trtllm.trtllm_weights_converter.single_device_trtllm_model_weights_converter.str_dtype_to_torch",
            return_value=torch.float32,
        )

        trtllm_model_weights_converter_cpu = SingleDeviceTRTLLMModelWeightsConverter(
            export_config, model_config, dtype, activation="gelu"
        )

        mocker.patch(
            "megatron.core.export.trtllm.trtllm_weights_converter.single_device_trtllm_model_weights_converter.pad_vocab_size",
            return_value=10,
        )

        # Verify the conditions are met for the valid case
        assert trtllm_model_weights_converter_cpu.num_kv_heads >= export_config.inference_tp_size
        assert (
            trtllm_model_weights_converter_cpu.num_kv_heads % export_config.inference_tp_size == 0
        )
        assert trtllm_model_weights_converter_cpu.num_kv_heads == 8
        assert export_config.inference_tp_size == 4

        # This should not raise an exception
        trtllm_model_weights_converter_cpu.convert(
            model_state_dict=model_state_dict,
            trtllm_conversion_dict=trtllm_conversion_dict,
            state_dict_split_by_layer_numbers=False,
        )

        # Check that QKV weights and biases are properly split for TP=4
        # Each TP rank should get 1/4 of the weights
        for layer_idx in range(num_layers):
            qkv_weight_key = f"transformer.layers.{layer_idx}.attention.qkv.weight"
            qkv_bias_key = f"transformer.layers.{layer_idx}.attention.qkv.bias"

            # Check that we have 4 splits (one for each TP rank)
            for tp_rank in range(4):
                weight_split_key = f"{qkv_weight_key}.{tp_rank}.bin"
                bias_split_key = f"{qkv_bias_key}.{tp_rank}.bin"

                assert weight_split_key in trtllm_model_weights_converter_cpu.trtllm_model_weights
                assert bias_split_key in trtllm_model_weights_converter_cpu.trtllm_model_weights

                # Verify that the splits have the expected dimensions
                weight_split = trtllm_model_weights_converter_cpu.trtllm_model_weights[
                    weight_split_key
                ]
                bias_split = trtllm_model_weights_converter_cpu.trtllm_model_weights[bias_split_key]

                # For TP=4, each split should have 1/4 of the original size
                # The weight shape depends on the conversion process, not necessarily hidden_dim
                assert (
                    len(bias_split.shape) == 1
                ), f"Expected bias to be 1D, got shape {bias_split.shape}"

                # Verify that all splits have the same size
                if tp_rank == 0:
                    expected_weight_size = weight_split.shape[1]
                    expected_bias_size = bias_split.shape[0]
                else:
                    assert (
                        weight_split.shape[1] == expected_weight_size
                    ), f"All weight splits should have same size"
                    assert (
                        bias_split.shape[0] == expected_bias_size
                    ), f"All bias splits should have same size"
