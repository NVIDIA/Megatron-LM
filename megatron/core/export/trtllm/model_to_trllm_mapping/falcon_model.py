# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from megatron.core.export.trtllm.trtllm_layers import TRTLLMLayers

# pylint: disable=line-too-long
FALCON_DICT = {
    # INPUT
    'embedding.word_embeddings.weight': TRTLLMLayers.vocab_embedding,
    'embedding.position_embeddings.weight': TRTLLMLayers.position_embedding,
    # ATTENTION
    'decoder.layers.input_layernorm.weight': TRTLLMLayers.input_layernorm_weight,
    'decoder.layers.input_layernorm.bias': TRTLLMLayers.input_layernorm_bias,
    'decoder.layers.self_attention.linear_qkv.weight': TRTLLMLayers.attention_qkv_weight,
    'decoder.layers.self_attention.linear_qkv.layer_norm_weight': TRTLLMLayers.input_layernorm_weight,
    'decoder.layers.self_attention.linear_proj.weight': TRTLLMLayers.attention_dense_weight,
    # MLP
    'decoder.layers.pre_mlp_layernorm.weight': TRTLLMLayers.post_layernorm_weight,
    'decoder.layers.pre_mlp_layernorm.bias': TRTLLMLayers.post_layernorm_bias,
    'decoder.layers.mlp.linear_fc1.weight': TRTLLMLayers.mlp_fc_weight,
    'decoder.layers.mlp.linear_fc2.weight': TRTLLMLayers.mlp_projection_weight,
    # FINAL LAYER NORM
    'decoder.final_layernorm.weight': TRTLLMLayers.final_layernorm_weight,
    'decoder.final_layernorm.bias': TRTLLMLayers.final_layernorm_bias,
    # OUTPUT LAYER
    'output_layer.weight': TRTLLMLayers.lm_head,
}
