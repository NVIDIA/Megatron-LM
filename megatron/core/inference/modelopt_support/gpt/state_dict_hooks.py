# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from logging import getLogger

import torch

logger = getLogger(__name__)


def mcore_gpt_load_legacy_state_dict_pre_hook(
    state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    """Register a pre-hook to fix the state_dict key difference.

    This prehook is used when trying to load the legacy Megatron-LM GPTModel into its
    megatron/core variant that uses native ParallelLinear and Transformer-Engine Norm.
    Only this particular spec supports post-training quantization and TensorRT-LLM
    config export through `nvidia-modelopt` package.

    Args:
        state_dict: state dictionary
        prefix: module name prefix
        local_metadata: local metatdata
        strict: whether is in strict mode
        missing_keys: missing state dict keys
        unexpected_keys: unexpected state dict keys
        error_msgs: error messages
    """
    if "modelopt_state" in state_dict:
        state_dict.pop("modelopt_state")

    if "language_model" in state_dict:
        language_model_state_dict = state_dict.pop("language_model")
        if "embedding" in language_model_state_dict:
            if "word_embeddings" in language_model_state_dict["embedding"]:
                for key, param in language_model_state_dict["embedding"]["word_embeddings"].items():
                    state_dict.update({"embedding.word_embeddings." + key: param})
            if "position_embeddings" in language_model_state_dict["embedding"]:
                for key, param in language_model_state_dict["embedding"][
                    "position_embeddings"
                ].items():
                    state_dict.update({"embedding.position_embeddings." + key: param})
        if "transformer" in language_model_state_dict:
            for key, param in language_model_state_dict["transformer"].items():
                state_dict.update({"decoder." + key: param})
        else:
            for key, param in language_model_state_dict["encoder"].items():
                state_dict.update({"decoder." + key: param})
        if "output_layer" in language_model_state_dict:
            for key, param in language_model_state_dict["output_layer"].items():
                state_dict.update({"output_layer." + key: param})

    if torch.distributed.get_rank() == 0:
        logger.info("ModelOptGPTModel {}".format(state_dict.keys()))

    module_name_rewrite_list = [
        ("input_norm", "input_layernorm"),
        (".attention.query_key_value", ".self_attention.linear_qkv"),
        (".attention.dense", ".self_attention.linear_proj"),
        ("self_attention.query_key_value", "self_attention.linear_qkv"),
        ("self_attention.dense", "self_attention.linear_proj"),
        ("post_attention_layernorm", "pre_mlp_layernorm"),
        ("post_attention_norm", "pre_mlp_layernorm"),
        ("dense_h_to_4h", "linear_fc1"),
        ("dense_4h_to_h", "linear_fc2"),
        ("final_norm", "final_layernorm"),
    ]

    key_rewrite_list = []

    for key, _ in state_dict.items():
        for old_name, new_name in module_name_rewrite_list:
            if old_name in key:
                key_rewrite_list += [(key, key.replace(old_name, new_name))]

    for old_key, new_key in key_rewrite_list:
        if torch.distributed.get_rank() == 0:
            logger.info("replace {} with {}".format(old_key, new_key))
        state_dict[new_key] = state_dict[old_key]
        state_dict.pop(old_key)


def mcore_gpt_load_te_state_dict_pre_hook(
    state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    """Register a pre-hook to fix the state_dict key difference of.

    This prehook is used when trying to load the megatron/core GPTModel that uses a
    fused Transformer-Engine ParallelLinear into the variant that uses native ParallelLinear
    and Transformer-Engine Norm (effectively to restore the fusion).
    Only this particular spec supports post-training quantization and TensorRT-LLM
    config export through `nvidia-modelopt` package.

    Args:
        state_dict: state dictionary
        prefix: module name prefix
        local_metadata: local metatdata
        strict: whether is in strict mode
        missing_keys: missing state dict keys
        unexpected_keys: unexpected state dict keys
        error_msgs: error messages
    """
    if "modelopt_state" in state_dict:
        state_dict.pop("modelopt_state")

    key_with_te_extra_state_to_pop = []

    for key, _ in state_dict.items():
        if "_extra_state" in key:
            key_with_te_extra_state_to_pop += [key]

    for key in key_with_te_extra_state_to_pop:
        state_dict.pop(key)

    module_name_rewrite_list = [
        ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
        ("self_attention.linear_qkv.layer_norm_bias", "input_layernorm.bias"),
        ("mlp.linear_fc1.layer_norm_weight", "pre_mlp_layernorm.weight"),
        ("mlp.linear_fc1.layer_norm_bias", "pre_mlp_layernorm.bias"),
    ]

    key_rewrite_list = []

    for key, _ in state_dict.items():
        for old_name, new_name in module_name_rewrite_list:
            if old_name in key:
                key_rewrite_list += [(key, key.replace(old_name, new_name))]

    for old_key, new_key in key_rewrite_list:
        if torch.distributed.get_rank() == 0:
            logger.info("replace {} with {}".format(old_key, new_key))
        state_dict[new_key] = state_dict[old_key]
        state_dict.pop(old_key)
