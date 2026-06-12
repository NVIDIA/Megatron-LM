import pytest

from megatron.core.export.trtllm.trtllm_layers import TRTLLMLayers, get_layer_name_without_prefix


class TestTRTLLMLayers:

    def test_rename_input_layer_names_to_trtllm_layer_names_without_layer_numbers(self):

        conversion_dict = {
            "transformer.layers.attn.dense.bias": TRTLLMLayers.attention_dense_bias,
            "transformer.layers.mlp.fc1.weight": TRTLLMLayers.mlp_fc_weight,
        }
        sample_dict = {
            "transformer.layers.attn.dense.bias": 0,
            "transformer.layers.mlp.fc1.weight": 1,
        }

        converted_dict = TRTLLMLayers.rename_input_layer_names_to_trtllm_layer_names(
            model_state_dict=sample_dict,
            trtllm_conversion_dict=conversion_dict,
            state_dict_split_by_layer_numbers=False,
        )
        assert (
            converted_dict[TRTLLMLayers.attention_dense_bias.value] == 0
        ), "Something wrong with conversion dict"
        assert (
            converted_dict[TRTLLMLayers.mlp_fc_weight.value] == 1
        ), "Something wrong with conversion dict"

    def test_rename_input_layer_names_to_trtllm_layer_names_exception(self):

        with pytest.raises(AssertionError):
            conversion_dict = {
                "transformer.layers.attn.dense.bias": "randomValue",
                "transformer.layers.mlp.fc1.weight": TRTLLMLayers.mlp_fc_weight,
            }
            sample_dict = {
                "transformer.layers.attn.dense.bias": 0,
                "transformer.layers.mlp.fc1.weight": 1,
            }
            TRTLLMLayers.rename_input_layer_names_to_trtllm_layer_names(
                model_state_dict=sample_dict,
                trtllm_conversion_dict=conversion_dict,
                state_dict_split_by_layer_numbers=False,
            )

        with pytest.raises(Exception):
            sample_dict = {
                "transformer.layers.attn.dense.bias": 0,
                "transformer.layers.mlp.fc1.weight": 1,
            }
            del conversion_dict["attn.dense.bias"]
            TRTLLMLayers.rename_input_layer_names_to_trtllm_layer_names(
                model_state_dict=sample_dict,
                trtllm_conversion_dict=conversion_dict,
                state_dict_split_by_layer_numbers=False,
            )

        with pytest.raises(Exception):
            conversion_dict = {
                "transformer.layers.attn.dense.bias": TRTLLMLayers.attention_dense_bias,
                "transformer.layers.mlp.fc1.weight": TRTLLMLayers.mlp_fc_weight,
            }
            sample_dict = {
                "transformer.layers.attn.dense.bias": 0,
                "transformer.layers.mlp.fc1.weight": 1,
            }

            TRTLLMLayers.rename_input_layer_names_to_trtllm_layer_names(
                model_state_dict=sample_dict,
                trtllm_conversion_dict=conversion_dict,
                state_dict_split_by_layer_numbers=True,
            )

    def test_rename_input_layer_names_to_trtllm_layer_names_with_layer_numbers(self):

        conversion_dict = {
            "decoder.lm_head.weight": TRTLLMLayers.lm_head,
            "decoder.layers.attn.dense.bias": TRTLLMLayers.attention_dense_bias,
            "deocder.layers.mlp.fc1.weight": TRTLLMLayers.mlp_fc_weight,
        }
        sample_dict = {
            "decoder.lm_head.weight": 2,
            "decoder.layers.0.attn.dense.bias": 0,
            "deocder.layers.43.mlp.fc1.weight": 1,
        }

        converted_dict = TRTLLMLayers.rename_input_layer_names_to_trtllm_layer_names(
            model_state_dict=sample_dict,
            trtllm_conversion_dict=conversion_dict,
            state_dict_split_by_layer_numbers=False,
        )

        assert (
            converted_dict['transformer.layers.0.attention.dense.bias'] == 0
        ), "Something wrong with conversion of layer names"
        assert (
            converted_dict['transformer.layers.43.mlp.fc.weight'] == 1
        ), "Something wrong with conversion of layer names"
        assert (
            converted_dict['lm_head.weight'] == 2
        ), "Something wrong with conversion of layer names"

    def test_get_layer_name_without_prefix(self):
        layer_name_without_prefix = get_layer_name_without_prefix(
            TRTLLMLayers.attention_dense_weight
        )
        assert (
            layer_name_without_prefix == "attention.dense.weight"
        ), f"get_layer_name_without_prefix returned {layer_name_without_prefix}, expected attention.dense.weight"
