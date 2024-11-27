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
    # MLP
    'decoder.layers.mlp.linear_fc1.layer_norm_weight': TRTLLMLayers.input_layernorm_weight,
    # Mixer
    'decoder.layers.mixer.dt_bias': TRTLLMLayers.mixer_dt_bias,
    'decoder.layers.mixer.A_log': TRTLLMLayers.mixer_A_log,
    'decoder.layers.mixer.D': TRTLLMLayers.mixer_D,
    'decoder.layers.mixer.in_proj.layer_norm_weight': TRTLLMLayers.input_layernorm_weight,
    'decoder.layers.mixer.in_proj.weight': TRTLLMLayers.mixer_in_proj_weight,
    'decoder.layers.mixer.conv1d.weight':TRTLLMLayers.mixer_conv_weight,
    'decoder.layers.mixer.conv1d.bias': TRTLLMLayers.mixer_conv_bias,
    'decoder.layers.mixer.out_proj.weight': TRTLLMLayers.mixer_out_proj_weight,
    'decoder.layers.mixer.norm.weight': TRTLLMLayers.mixer_norm_weight,
    # FINAL LAYER NORM
    'decoder.final_norm.weight': TRTLLMLayers.final_layernorm_weight,

    'preprocess_weight': mamba_preprocess_weight,
}
