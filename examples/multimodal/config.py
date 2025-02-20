# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
from dataclasses import dataclass

import torch

from megatron.training.activations import fast_gelu, quick_gelu, squared_relu


def get_language_model_config(config):
    if config.language_model_type == "llama3_8b":
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
    elif config.language_model_type == "llama3.1_8b":
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
    elif config.language_model_type == "llama3.1_70B":
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
        config.ffn_hidden_size = 28672
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
    elif config.language_model_type == "yi-34b":
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
        config.ffn_hidden_size = 20480
    elif config.language_model_type == "qwen2.5_7B":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.add_qkv_bias = True
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 18944
    elif config.language_model_type == "qwen2.0_72B":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.add_qkv_bias = True
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 29568
    elif config.language_model_type == "llama3.2_1b":
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
        config.ffn_hidden_size = 8192
    elif config.language_model_type.startswith("huggingface"):
        # Loaded from HuggingFace config file.
        pass
    else:
        raise ValueError(f"unknown language model type {config.language_model_type}")

    return config


def get_vision_model_config(config, apply_query_key_layer_scaling):
    if config.vision_model_type == "clip":
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
        config.num_query_groups = 16
        config.layernorm_zero_centered_gamma = False
        config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.normalization = 'LayerNorm'
        config.apply_rope_fusion = False
    elif config.vision_model_type == "siglip":
        config.num_layers = 27
        config.num_attention_heads = 16
        config.add_bias_linear = True
        config.add_qkv_bias = True
        config.hidden_size = 1152
        config.hidden_dropout = 0.0
        config.attention_dropout = 0.0
        config.ffn_hidden_size = 4304
        config.gated_linear_unit = False
        config.activation_func = fast_gelu
        config.kv_channels = 72
        config.num_query_groups = 16
        config.layernorm_zero_centered_gamma = False
        config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.normalization = 'LayerNorm'
        config.apply_rope_fusion = False
        config.qk_layernorm = False
        config.layernorm_epsilon = 1e-6
    elif config.vision_model_type == "internvit":
        config.num_layers = 45
        config.num_attention_heads = ((24 // config.tensor_model_parallel_size) + 1) * config.tensor_model_parallel_size
        config.num_query_groups = config.num_attention_heads
        config.add_bias_linear = True
        config.add_qkv_bias = False
        config.hidden_size = 3200
        config.hidden_dropout = 0.0
        config.attention_dropout = 0.0
        config.ffn_hidden_size = 12800
        config.gated_linear_unit = False
        config.activation_func = torch.nn.functional.gelu
        config.layernorm_zero_centered_gamma = False
        config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.normalization = 'RMSNorm'
        config.layernorm_epsilon = 1e-6
        config.apply_rope_fusion = False
    elif config.vision_model_type == "radio":
        config.num_layers = 32
        config.num_attention_heads = 16
        config.add_bias_linear = True
        config.add_qkv_bias = True
        config.hidden_size = 1280
        config.ffn_hidden_size = 5120
        config.gated_linear_unit = False
        config.activation_func = fast_gelu
        config.kv_channels = 80
        config.num_query_groups = 16
        config.layernorm_zero_centered_gamma = False
        config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.normalization = 'LayerNorm'
        config.apply_rope_fusion = False
        config.qk_layernorm = False
        config.layernorm_epsilon = 1e-6
    elif config.vision_model_type.startswith("huggingface"):
        # Loaded from HuggingFace config file.
        pass
    else:
        raise ValueError(f"unknown vision model type {config.vision_model_type}")

    return config


def get_vision_projection_config(config, hidden_size):
    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.add_bias_linear = False
    config.hidden_size = hidden_size  # Used as the vision projection output size, i.e., the input to the language model.
    if config.language_model_type == "llama3_8b":
        config.ffn_hidden_size = 14336
        config.activation_func = torch.nn.functional.gelu
    elif config.language_model_type == "llama3.1_8b":
        config.ffn_hidden_size = 4096
        config.activation_func = torch.nn.functional.gelu
        config.layernorm_epsilon = 1e-5
        config.add_bias_linear = True
        config.normalization = "LayerNorm"
    elif config.language_model_type == "mistral_7b":
        config.ffn_hidden_size = 14336
        config.activation_func = torch.nn.functional.gelu
        config.normalization = None
    elif config.language_model_type == "yi-34b":
        config.ffn_hidden_size = 20480
        config.normalization = "LayerNorm"
        config.activation_func = torch.nn.functional.gelu
    elif config.language_model_type == "qwen2.5_7B":
        config.ffn_hidden_size = 3584
        config.activation_func = torch.nn.functional.gelu
    elif config.language_model_type == "qwen2.0_72B":
        config.ffn_hidden_size = 29568
        config.normalization = "LayerNorm"
        config.activation_func = torch.nn.functional.gelu
    elif config.language_model_type == "llama3.2_1b":
        config.ffn_hidden_size = 2048
        config.activation_func = torch.nn.functional.gelu
        config.normalization = "LayerNorm"
    elif config.language_model_type.startswith("huggingface"):
        config.activation_func = torch.nn.functional.gelu
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(config.huggingface_model_name_or_path)
        if "qwen" in hf_config.model_type:
            config.ffn_hidden_size = 1536
    else:
        raise ValueError(f"unknown language model type {config.language_model_type}")

    return config


@dataclass
class EvaluationConfig:
    """Evaluation related configuration."""
    task: str

    temperature: float = 1.0
    top_p: float = 0.0
    top_k: int = 0

    out_seq_length: int = 32

    output_path: str = ""

    input_image_path: str = ""
    gt_path: str = ""

    num_partitions: int = 1
    partition_id: int = 0
    num_samples_per_partition: int = 0
