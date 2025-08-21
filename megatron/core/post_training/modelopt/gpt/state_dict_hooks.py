# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from logging import getLogger

import torch

logger = getLogger(__name__)


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

    for key in key_with_te_extra_state_to_pop:
        state_dict.pop(key)

    module_name_rewrite_list = [
        ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
        ("self_attention.linear_qkv.layer_norm_bias", "input_layernorm.bias"),
        ("self_attention.linear_q_up_proj.layer_norm_weight", "self_attention.q_layernorm.weight"),
        ("self_attention.linear_q_up_proj.layer_norm_bias", "self_attention.q_layernorm.bias"),
        (
            "self_attention.linear_kv_up_proj.layer_norm_weight",
            "self_attention.kv_layernorm.weight",
        ),
        ("self_attention.linear_kv_up_proj.layer_norm_bias", "self_attention.kv_layernorm.bias"),
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
