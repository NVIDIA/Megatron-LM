# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
from megatron.core.export.trtllm.trtllm_layers import TRTLLMLayers

def mamba_preprocess_weight(model_state_dict: dict):
    for k in list(model_state_dict.keys()):
        if 'mixer.in_proj.weight' in k:
            if k[-1] == 'z':
                prefix = k[:-2]
                z = model_state_dict.pop(k)
                x = model_state_dict.pop(f"{prefix}.x")
                B = model_state_dict.pop(f"{prefix}.B")
                C = model_state_dict.pop(f"{prefix}.C")
                dt = model_state_dict.pop(f"{prefix}.dt")
                model_state_dict[prefix] = torch.concatenate(
                    [z, x, B, C, dt], dim=0
                )
        elif 'conv1d' in k:
            if k[-1] == 'x':
                prefix = k[:-2]
                x = model_state_dict.pop(k)
                B = model_state_dict.pop(f"{prefix}.B")
                C = model_state_dict.pop(f"{prefix}.C")
                model_state_dict[prefix] = torch.concatenate(
                    [x, B, C], dim=0
                )


MAMBA_HYBRID_DICT = {
    # INPUT
    'model.embedding.word_embeddings.weight': TRTLLMLayers.vocab_embedding,
    # ATTENTION
    'model.decoder.layers.self_attention.linear_qkv.weight': TRTLLMLayers.attention_qkv_weight,
    'model.decoder.layers.self_attention.linear_qkv.layer_norm_weight': TRTLLMLayers.input_layernorm_weight,
    'model.decoder.layers.self_attention.linear_proj.weight': TRTLLMLayers.attention_dense_weight,
    # MLP
    'model.decoder.layers.mlp.linear_fc1.weight': TRTLLMLayers.mlp_fc_weight,
    'model.decoder.layers.mlp.linear_fc2.weight': TRTLLMLayers.mlp_projection_weight,
    'model.decoder.layers.mlp.linear_fc1.layer_norm_weight': TRTLLMLayers.input_layernorm_weight,
    # Mixer
    'model.decoder.layers.mixer.dt_bias': TRTLLMLayers.mixer_dt_bias,
    'model.decoder.layers.mixer.A_log': TRTLLMLayers.mixer_A_log,
    'model.decoder.layers.mixer.D': TRTLLMLayers.mixer_D,
    'model.decoder.layers.mixer.in_proj.layer_norm_weight': TRTLLMLayers.input_layernorm_weight,
    'model.decoder.layers.mixer.in_proj.weight': TRTLLMLayers.mixer_in_proj_weight,
    'model.decoder.layers.mixer.conv1d.weight':TRTLLMLayers.mixer_conv_weight,
    'model.decoder.layers.mixer.conv1d.bias': TRTLLMLayers.mixer_conv_bias,
    'model.decoder.layers.mixer.out_proj.weight': TRTLLMLayers.mixer_out_proj_weight,
    'model.decoder.layers.mixer.norm.weight': TRTLLMLayers.mixer_norm_weight,
    # FINAL LAYER NORM
    'model.decoder.final_norm.weight': TRTLLMLayers.final_layernorm_weight,
    # OUTPUT LAYER
    'model.output_layer.weight': TRTLLMLayers.lm_head,

    'preprocess_weight': mamba_preprocess_weight,
}
