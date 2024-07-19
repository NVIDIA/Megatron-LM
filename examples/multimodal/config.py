# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import torch

from megatron.training.activations import quick_gelu, squared_relu


def get_language_model_config(config):
    if config.language_model_type == "2b":
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = True
        config.layernorm_zero_centered_gamma = True
        config.bias_dropout_fusion = False
        config.rotary_percent = 0.5
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
    elif config.language_model_type == "8b":
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.gated_linear_unit = False
        config.apply_query_key_layer_scaling = True
        config.layernorm_zero_centered_gamma = True
        config.bias_dropout_fusion = False
        config.rotary_percent = 0.5
        config.attention_dropout = 0.0
        config.apply_rope_fusion = False
        config.activation_func = squared_relu
        config.ffn_hidden_size = 16384
        config.masked_softmax_fusion = True
        config.attention_softmax_in_fp32 = True
        config.num_query_groups = 32
        config.kv_channels = 128
        config.rotary_interleaved = False
    elif config.language_model_type == "llama3_8b":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 14336
    elif config.language_model_type == "mistral_7b":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 14336

    return config


def get_vision_model_config(config, apply_query_key_layer_scaling=False):
    config.num_layers = 24
    config.num_attention_heads = 16
    config.add_bias_linear = True
    config.add_qkv_bias = True
    config.hidden_size = 1024
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.ffn_hidden_size = 4096
    config.gated_linear_unit = False
    config.activation_func = quick_gelu
    config.kv_channels = 64
    config.num_attention_heads = 16
    config.num_query_groups = 16
    config.layernorm_zero_centered_gamma = False
    config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.normalization = 'LayerNorm'

    return config


def get_vision_projection_config(config, hidden_size):
    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.add_bias_linear = False
    config.hidden_size = hidden_size
    if config.language_model_type == "2b":
        config.ffn_hidden_size = 5440
        config.activation_func = torch.nn.functional.gelu
    if config.language_model_type == "8b":
        config.ffn_hidden_size = 16384
        config.activation_func = squared_relu
    elif config.language_model_type == "llama3_8b":
        config.ffn_hidden_size = 14336
        config.activation_func = torch.nn.functional.silu
    elif config.language_model_type == "mistral_7b":
        config.ffn_hidden_size = 14336
        config.activation_func = torch.nn.functional.silu

    return config
