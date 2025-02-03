# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import re
from enum import Enum
from typing import Tuple


class TRTLLMLayers(Enum):
    """TRTLLM Layer names

    This Enum will be used to map input model layer names to TRTLLM Layer names
    """

    # ONE TIME LAYERS (NOT ASSOCIATED TO TRANSFORMER BLOCK)
    # Input layers
    position_embedding = 'transformer.position_embedding.weight'
    vocab_embedding = 'transformer.vocab_embedding.weight'
    lm_head = 'lm_head.weight'

    # Output layers
    final_layernorm_weight = 'transformer.ln_f.weight'
    final_layernorm_bias = 'transformer.ln_f.bias'

    # TRANSFORMER LAYERS
    # Attention block related layers
    input_layernorm_weight = 'transformer.layers.input_layernorm.weight'
    input_layernorm_bias = 'transformer.layers.input_layernorm.bias'
    attention_qkv_weight = 'transformer.layers.attention.qkv.weight'
    attention_qkv_bias = 'transformer.layers.attention.qkv.bias'
    attention_dense_weight = 'transformer.layers.attention.dense.weight'
    attention_dense_bias = 'transformer.layers.attention.dense.bias'

    # mlp layers
    mlp_fc_weight = 'transformer.layers.mlp.fc.weight'
    mlp_fc_bias = 'transformer.layers.mlp.fc.bias'
    post_layernorm_weight = 'transformer.layers.post_layernorm.weight'
    post_layernorm_bias = 'transformer.layers.post_layernorm.bias'
    mlp_projection_weight = 'transformer.layers.mlp.proj.weight'
    mlp_projection_bias = 'transformer.layers.mlp.proj.bias'

    # mixture of expert layers
    mlp_router_weight = 'transformer.layers.mlp.router.weight'
    mlp_fc_weight_mixture_of_experts = 'transformer.layers.mlp.fc.weight.expert'
    mlp_projection_weight_mixture_of_experts = 'transformer.layers.mlp.proj.weight.expert'

    @staticmethod
    def return_layer_name_and_number(layer_name: str) -> Tuple[str, int]:
        """Helper function to return layer name and number
        Given an input layer e.g decoder.layers.2.self_attention.linear_qkv.weight,
        this function returns decoder.layers.self_attention.linear_qkv.weight and layernumber 2.
        In case no layer number is present, it returns None for the layer number
        Args:
            layer_name (dict): The input layer name

        Returns:
            Tuple[str, int]: The layer name , layer number (layer number could be None)
        """
        # Use regular expression to find the number specifically after 'layers.'
        match = re.search(r'(?<=layers\.)\d+(?=\.)', layer_name)
        if match:
            # Extract the number and remove it from the layer name
            number = match.group(0)
            layer_name_without_number = re.sub(r'\.{}\.'.format(number), '.', layer_name)
            return layer_name_without_number, int(number)
        else:
            # Return the original name if no number is found
            return layer_name, None

    # pylint: disable=line-too-long
    @staticmethod
    def rename_input_layer_names_to_trtllm_layer_names(
        model_state_dict: dict,
        trtllm_conversion_dict: dict,
        state_dict_split_by_layer_numbers: bool = True,
    ) -> dict:
        """Helper function to rename model layer names to TRTLLM Layer names

        We go through each layer (keys) in the model state dict,
        and map it to the equivalent TRTLLMLayer name (megatron/core/export/trtllm/trtllm).
        If we have a layer number associated with layer, we extract it out,
        map the original layer name to equivalent trtllm layer name and add layer number back.
        CPU Conversion will pass in model state dict without layer numbers
        (i.e decoder.layers.mlp.linear_fc1.weight of shape [num_layers, hidden_dim, 4 * hidden_dim]) .
        GPU conversion will pass model state dict with each layer seperated
        (i.e decoder.layers.2.mlp.linear_fc1.weight of shape [hidden_dim, 4 * hidden_dim]).

        Args:
            model_state_dict (dict): The original model state dict
            trtllm_conversion_dict (dict): The conversion dictionary mapping input model layer names to trtllm layer names
            state_dict_split_by_layer_numbers (bool, optional): Are the model layers split by layer numbers in state dict. For example : mlp.fc1.weight can be represented like mlp.fc1.weight of shape [num_layers, hidden_dim, ffn_hidden_dim]} or it can be like mlp.fc1.layers.0.weight of shape [hidden_dim, ffn_hidden_dim], then mlp.fc1.layers.1.weight ... for all layers. If you use represenation 2 set this to True. Defaults to True

        Raises:
            ValueError: In case the keys dont match to trtllm keys or if all model layers are not mapped to equivalent trtllm keys

        Returns:
            dict: The model state dict with the key (i.e original model layer name) replaced by trtllm layer names
        """
        for original_model_layer_name in list(model_state_dict.keys()):
            if "_extra_state" in original_model_layer_name:
                del model_state_dict[original_model_layer_name]
                continue

            original_layer_name_without_number, layer_number = (
                TRTLLMLayers.return_layer_name_and_number(original_model_layer_name)
            )
            if 'layers' in original_layer_name_without_number and state_dict_split_by_layer_numbers:
                assert (
                    layer_number is not None
                ), f"Layer number is None for {original_model_layer_name} and state_dict_split_by_layer_numbers is set to True. Consider setting it False"

            if original_layer_name_without_number not in trtllm_conversion_dict:
                raise ValueError(
                    f'Unable to rename key {original_layer_name_without_number}. Provide an appropriate mapping in the trtllm_conversion_dict when you initialize TRTLLMHelper'
                )

            trtllm_layer = trtllm_conversion_dict[original_layer_name_without_number]
            assert isinstance(
                trtllm_layer, TRTLLMLayers
            ), f"{trtllm_layer} is not supported for conversion. Please use one of the TRTLLMLayerNames we provided in megatron/core/export/trtllm/trtllm_layer_names"

            value = model_state_dict.pop(original_model_layer_name)

            if layer_number is not None:
                trtllm_layer_name_with_number = re.sub(
                    r'(?<=layers\.)', f'{layer_number}.', trtllm_layer.value
                )
                model_state_dict[trtllm_layer_name_with_number] = value
            else:
                model_state_dict[trtllm_layer.value] = value

        return model_state_dict


# These layers are not associated within the transformer block.
# So they dont have a layer number (i.e independant of number of layers in the model)
NON_TRANSFORMER_LAYERS_NAMES = [
    TRTLLMLayers.vocab_embedding.value,
    TRTLLMLayers.position_embedding.value,
    TRTLLMLayers.lm_head.value,
    TRTLLMLayers.final_layernorm_weight.value,
    TRTLLMLayers.final_layernorm_bias.value,
]


def get_layer_name_without_prefix(layer: TRTLLMLayers) -> str:
    """Get TRTLayer name without prefix

    Given a layer e.g TRTLLMLayers.attention_qkv_weight it returns 'attention.qkv.weight'

    Args:
        layer (TRTLLMLayers): The TRTLLMLayer

    Returns:
        str: The TRTLLMLayers suffix (i.e Removing transformer.layers. fromt he layer name)
    """
    layer_name_without_prefix = layer.value.replace("transformer.layers.", "")
    return layer_name_without_prefix
